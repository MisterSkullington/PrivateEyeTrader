"""
AssetFilter — pre-signal scam/shitcoin and illiquidity screen.

Checks (in order):
  1. Banned symbols list (explicit blacklist)
  2. 24h rolling USD volume floor (liquidity check)
  3. Daily historical volatility ceiling (extreme-vol / low-liquidity proxy)

All checks are disabled when enabled=False (default). Pass bars=None to
evaluate_signal() and the AssetFilter gate is skipped transparently.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from privateye.utils.logging import get_logger

log = get_logger()


class AssetFilter:
    def __init__(
        self,
        enabled: bool = False,
        min_volume_usd_24h: float = 1_000_000,
        max_daily_vol_pct: float = 0.15,
        banned_symbols: list[str] | None = None,
    ) -> None:
        self.enabled = enabled
        self.min_volume_usd_24h = min_volume_usd_24h
        self.max_daily_vol_pct = max_daily_vol_pct
        self.banned_symbols: set[str] = {s.upper() for s in (banned_symbols or [])}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AssetFilter":
        return cls(
            enabled=config.get("enabled", False),
            min_volume_usd_24h=float(config.get("min_volume_usd_24h", 1_000_000)),
            max_daily_vol_pct=float(config.get("max_daily_vol_pct", 0.15)),
            banned_symbols=config.get("banned_symbols", []),
        )

    def is_tradeable(
        self,
        symbol: str,
        bars: pd.DataFrame,
        hv_period: int = 20,
    ) -> tuple[bool, str]:
        """
        Returns (tradeable: bool, reason: str).
        reason is empty string on pass.
        Always returns (True, "") when enabled=False.
        """
        if not self.enabled:
            return True, ""

        # 1. Banned symbols
        if symbol.upper() in self.banned_symbols:
            return False, f"Symbol {symbol} is in banned_symbols list"

        if bars is None or len(bars) == 0:
            return True, ""  # no data → can't screen, pass through

        # 2. 24h rolling USD volume (last 24 bars × price)
        if len(bars) >= 24:
            last_24 = bars.tail(24)
            vol_usd = float((last_24["volume"] * last_24["close"]).sum())
            if vol_usd < self.min_volume_usd_24h:
                return False, (
                    f"Insufficient liquidity: 24h vol ${vol_usd:,.0f} "
                    f"< ${self.min_volume_usd_24h:,.0f} floor"
                )

        # 3. Daily HV ceiling: annualised HV / sqrt(365)
        if len(bars) >= hv_period + 1:
            from privateye.indicators.library import historical_volatility
            hv_ann = historical_volatility(bars, hv_period).iloc[-1]
            if not math.isnan(hv_ann):
                daily_hv = hv_ann / math.sqrt(365)
                if daily_hv > self.max_daily_vol_pct:
                    return False, (
                        f"Excessive daily volatility: {daily_hv:.1%} "
                        f"> {self.max_daily_vol_pct:.1%} ceiling"
                    )

        return True, ""
