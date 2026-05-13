"""
Shadow tracker — compares paper-simulation fill prices against live exchange prices.

For every FILL event (paper only), the tracker fetches the live mid-price from
the exchange and records the divergence (slippage).  When the divergence exceeds
`divergence_alert_threshold_pct` a SHADOW_DIVERGENCE event is published.

This lets you run the full paper simulation while quantifying how much your
simulated fills diverge from what you'd actually pay live — a key sanity check
before switching from paper to live mode.

Phase 3 — Shadow Mode 2.0 extension:
  Adds Reality Score tracking — a rolling measure of how well the slippage model
  predicts actual slippage.  When the Reality Score drops below the configured
  threshold, a CONSERVATIVE_MODE event is published and the tracker enters
  conservative mode (which the ExecutionEngine uses to halve position sizes).

Config keys (under robustness.shadow_trading):
  enabled                        (bool,  default False)
  divergence_alert_threshold_pct (float, default 2.0)
  reality_score_window           (int,   default 50)   -- Phase 3
  conservative_mode_threshold    (float, default 0.75) -- Phase 3
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Any

import numpy as np

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
        baseline_slippage: float = 0.0005,  # Phase 3: from cfg["backtesting"]["slippage_pct"]
    ) -> None:
        self.enabled: bool = config.get("enabled", False)
        self._threshold: float = float(
            config.get("divergence_alert_threshold_pct", 2.0)
        ) / 100.0  # convert pct → fraction

        self._bus = bus
        self._adapter = adapter
        self._records: list[ShadowFillRecord] = []

        # ── Phase 3: Shadow Mode 2.0 — Reality Score tracking ─────────────────
        self._reality_score_window: int = int(config.get("reality_score_window", 50))
        self._conservative_threshold: float = float(
            config.get("conservative_mode_threshold", 0.75)
        )
        self._baseline_slippage: float = max(baseline_slippage, 1e-9)  # guard /0
        self._reality_scores: deque[float] = deque(maxlen=self._reality_score_window)
        self._conservative_mode: bool = False

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

        # ── Phase 3: Reality Score update ─────────────────────────────────────
        realized = abs(record.slippage_pct)
        predicted = self._baseline_slippage
        score = max(
            0.0,
            min(1.0, 1.0 - abs(realized - predicted) / predicted),
        )
        self._reality_scores.append(score)

        if len(self._reality_scores) >= 10:
            rolling = self.reality_score
            if rolling < self._conservative_threshold and not self._conservative_mode:
                self._conservative_mode = True
                self._bus.publish_sync(
                    EventType.CONSERVATIVE_MODE,
                    {"reason": "reality_score_dropped", "score": rolling},
                )
                log.warning(
                    f"[ShadowTracker] Conservative mode ACTIVATED "
                    f"(reality_score={rolling:.3f} < {self._conservative_threshold:.3f})"
                )
            elif rolling >= self._conservative_threshold and self._conservative_mode:
                self._conservative_mode = False
                log.info(
                    f"[ShadowTracker] Conservative mode DEACTIVATED "
                    f"(reality_score={rolling:.3f} >= {self._conservative_threshold:.3f})"
                )

        log.debug(
            f"[ShadowTracker] {fill.symbol} "
            f"paper={fill.price:.4f} mid={live_mid:.4f} "
            f"slippage={slippage_pct:.4%} reality_score={self.reality_score:.3f}"
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
        self._reality_scores.clear()
        self._conservative_mode = False

    # ── Phase 3: Reality Score API ────────────────────────────────────────────

    @property
    def reality_score(self) -> float:
        """Rolling mean Reality Score across the last `reality_score_window` fills.

        1.0 = perfect model calibration; 0.0 = severe divergence.
        Returns 1.0 when no fills have been recorded yet (optimistic default).
        """
        return float(np.mean(self._reality_scores)) if self._reality_scores else 1.0

    @property
    def conservative_mode(self) -> bool:
        """True when rolling reality_score < conservative_threshold for ≥10 fills."""
        return self._conservative_mode

    def get_reality_stats(self) -> dict:
        """Synchronous stats dict for dashboard callbacks.

        Keys mirror ShadowReport fields plus Phase 3 additions.
        """
        base = self.get_report()
        return {
            "n_fills":                   base.n_fills,
            "mean_slippage_pct":         base.mean_slippage_pct,
            "max_slippage_pct":          base.max_slippage_pct,
            "pct_fills_above_threshold": base.pct_fills_above_threshold,
            "reality_score":             self.reality_score,
            "conservative_mode":         self._conservative_mode,
        }


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
