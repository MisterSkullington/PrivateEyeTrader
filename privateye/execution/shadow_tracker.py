"""
Shadow tracker — compares paper-simulation fill prices against live exchange prices.

For every FILL event (paper only), the tracker fetches the live mid-price from
the exchange and records the divergence (slippage).  When the divergence exceeds
`divergence_alert_threshold_pct` a SHADOW_DIVERGENCE event is published.

This lets you run the full paper simulation while quantifying how much your
simulated fills diverge from what you'd actually pay live — a key sanity check
before switching from paper to live mode.

Config keys (under robustness.shadow_trading):
  enabled                        (bool,  default False)
  divergence_alert_threshold_pct (float, default 2.0)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Any

from privateye.core.types import EventType, Fill, OrderSide
from privateye.utils.logging import get_logger

log = get_logger()


@dataclass
class ShadowFillRecord:
    """Snapshot of one paper fill vs. the simultaneous live mid-price."""
    symbol: str
    paper_fill_price: float      # price at which the simulated exchange filled
    live_mid_price: float        # (bid + ask) / 2  (or last if bid/ask absent)
    slippage_pct: float          # (paper_fill_price - live_mid) / live_mid
    fill_side: OrderSide
    quantity: float
    timestamp: datetime


@dataclass
class ShadowReport:
    """Aggregate slippage statistics across all recorded fills."""
    n_fills: int
    mean_slippage_pct: float
    max_slippage_pct: float       # max of |slippage_pct|
    std_slippage_pct: float
    pct_fills_above_threshold: float   # fraction where |slippage| > threshold
    records: list[ShadowFillRecord] = field(default_factory=list)


class ShadowTracker:
    """
    Wires into the FILL event stream and compares each simulated fill
    against the live exchange price at fill time.

    Usage::

        tracker = ShadowTracker(bus, adapter, cfg["robustness"]["shadow_trading"])
        bus.subscribe(EventType.FILL, tracker.on_fill)
    """

    def __init__(
        self,
        bus: Any,
        adapter: Any,                   # ExchangeAdapter (read-only, price queries only)
        config: dict[str, Any],
    ) -> None:
        self.enabled: bool = config.get("enabled", False)
        self._threshold: float = float(
            config.get("divergence_alert_threshold_pct", 2.0)
        ) / 100.0  # convert pct → fraction

        self._bus = bus
        self._adapter = adapter
        self._records: list[ShadowFillRecord] = []

    # ── Event handler ─────────────────────────────────────────────────────────

    async def on_fill(self, fill: Fill) -> None:
        """
        Called on every FILL event.
        Fetches the live mid-price and records the divergence from the paper fill.
        Publishes SHADOW_DIVERGENCE if the divergence exceeds the configured threshold.
        On adapter error: logs a warning and skips (no crash).
        """
        if not self.enabled:
            return

        try:
            ticker = await self._adapter.fetch_ticker(fill.symbol)
        except Exception as exc:
            log.warning(
                f"[ShadowTracker] fetch_ticker failed for {fill.symbol}: {exc} — skipping"
            )
            return

        live_mid = _extract_mid(ticker)
        if live_mid is None or live_mid <= 0:
            log.warning(
                f"[ShadowTracker] No valid price in ticker for {fill.symbol} — skipping"
            )
            return

        slippage_pct = (fill.price - live_mid) / live_mid

        record = ShadowFillRecord(
            symbol=fill.symbol,
            paper_fill_price=fill.price,
            live_mid_price=live_mid,
            slippage_pct=slippage_pct,
            fill_side=fill.side,
            quantity=fill.quantity,
            timestamp=datetime.now(timezone.utc),
        )
        self._records.append(record)

        log.debug(
            f"[ShadowTracker] {fill.symbol} "
            f"paper={fill.price:.4f} mid={live_mid:.4f} "
            f"slippage={slippage_pct:.4%}"
        )

        if abs(slippage_pct) > self._threshold:
            log.info(
                f"[ShadowTracker] Divergence alert: {fill.symbol} "
                f"slippage={slippage_pct:.4%} "
                f"(threshold={self._threshold:.4%})"
            )
            await self._bus.publish(EventType.SHADOW_DIVERGENCE, record)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def get_report(self) -> ShadowReport:
        """Compute aggregate slippage statistics from all recorded fills."""
        if not self._records:
            return ShadowReport(
                n_fills=0,
                mean_slippage_pct=0.0,
                max_slippage_pct=0.0,
                std_slippage_pct=0.0,
                pct_fills_above_threshold=0.0,
                records=[],
            )

        slippages = [r.slippage_pct for r in self._records]
        abs_slippages = [abs(s) for s in slippages]
        above = sum(1 for s in abs_slippages if s > self._threshold)

        return ShadowReport(
            n_fills=len(self._records),
            mean_slippage_pct=mean(slippages),
            max_slippage_pct=max(abs_slippages),
            std_slippage_pct=stdev(slippages) if len(slippages) > 1 else 0.0,
            pct_fills_above_threshold=above / len(self._records),
            records=list(self._records),
        )

    def reset(self) -> None:
        """Clear all recorded fills (e.g. at session boundaries)."""
        self._records.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_mid(ticker: dict) -> float | None:
    """
    Extract the mid-price from a CCXT ticker dict.
    Prefers (bid + ask) / 2; falls back to 'last'.
    Returns None if no usable price is found.
    """
    bid = ticker.get("bid")
    ask = ticker.get("ask")
    if bid is not None and ask is not None:
        try:
            return (float(bid) + float(ask)) / 2.0
        except (TypeError, ValueError):
            pass

    last = ticker.get("last")
    if last is not None:
        try:
            return float(last)
        except (TypeError, ValueError):
            pass

    return None
