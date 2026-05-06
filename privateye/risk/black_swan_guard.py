"""
BlackSwanGuard — per-bar market anomaly detector.

Triggers (checked in order):
  1. FLASH_CRASH    — single-bar return ≤ flash_crash_pct (e.g. -8%)
  2. VOLUME_SPIKE   — bar volume ≥ multiplier × rolling mean
  3. FUNDING_EXTREME — |funding_rate| ≥ max_funding_rate_abs
  4. DEPEG_DETECTED — stablecoin close deviates from $1.00 ≥ stablecoin_depeg_pct

Returns the first triggered BlackSwanEvent or None.

Integration: BacktestEngine calls check() each bar. If an event is returned,
the engine flattens all positions via the existing FLAT-signal path and halts
the RiskManager for the remainder of the backtest.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import pandas as pd

from privateye.utils.logging import get_logger

log = get_logger()


class BlackSwanTrigger(str, Enum):
    FLASH_CRASH     = "flash_crash"
    VOLUME_SPIKE    = "volume_spike"
    FUNDING_EXTREME = "funding_extreme"
    DEPEG_DETECTED  = "depeg_detected"


@dataclass
class BlackSwanEvent:
    trigger: BlackSwanTrigger
    symbol: str
    severity: float       # 0.0–1.0 normalised intensity
    detail: str
    timestamp: datetime


class BlackSwanGuard:
    def __init__(
        self,
        enabled: bool = False,
        flash_crash_pct: float = -0.08,
        volume_spike_multiplier: float = 5.0,
        volume_lookback: int = 20,
        max_funding_rate_abs: float = 0.003,
        stablecoin_depeg_pct: float = 0.005,
        stablecoin_symbols: list[str] | None = None,
    ) -> None:
        self.enabled = enabled
        self.flash_crash_pct = flash_crash_pct
        self.volume_spike_multiplier = volume_spike_multiplier
        self.volume_lookback = volume_lookback
        self.max_funding_rate_abs = max_funding_rate_abs
        self.stablecoin_depeg_pct = stablecoin_depeg_pct
        self.stablecoin_symbols: set[str] = set(stablecoin_symbols or [])

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BlackSwanGuard":
        return cls(
            enabled=config.get("enabled", False),
            flash_crash_pct=float(config.get("flash_crash_pct", -0.08)),
            volume_spike_multiplier=float(config.get("volume_spike_multiplier", 5.0)),
            volume_lookback=int(config.get("volume_lookback", 20)),
            max_funding_rate_abs=float(config.get("max_funding_rate_abs", 0.003)),
            stablecoin_depeg_pct=float(config.get("stablecoin_depeg_pct", 0.005)),
            stablecoin_symbols=config.get("stablecoin_symbols", []),
        )

    def check(
        self,
        symbol: str,
        bars: pd.DataFrame,
        funding_rate: float = 0.0,
    ) -> BlackSwanEvent | None:
        """
        Returns a BlackSwanEvent if any trigger fires, else None.
        Returns None immediately when enabled=False or len(bars) < 2.
        """
        if not self.enabled or bars is None or len(bars) < 2:
            return None

        ts = datetime.now(timezone.utc)

        # 1. Flash crash
        close_now  = float(bars["close"].iloc[-1])
        close_prev = float(bars["close"].iloc[-2])
        if close_prev > 0:
            bar_return = (close_now - close_prev) / close_prev
            if bar_return <= self.flash_crash_pct:
                severity = min(1.0, abs(bar_return) / abs(self.flash_crash_pct))
                return BlackSwanEvent(
                    trigger=BlackSwanTrigger.FLASH_CRASH,
                    symbol=symbol,
                    severity=severity,
                    detail=f"Bar return {bar_return:.2%} ≤ threshold {self.flash_crash_pct:.2%}",
                    timestamp=ts,
                )

        # 2. Volume spike
        if len(bars) >= self.volume_lookback + 1:
            current_vol = float(bars["volume"].iloc[-1])
            rolling_mean = float(bars["volume"].iloc[-(self.volume_lookback + 1):-1].mean())
            if rolling_mean > 0 and current_vol >= rolling_mean * self.volume_spike_multiplier:
                ratio = current_vol / rolling_mean
                severity = min(1.0, ratio / self.volume_spike_multiplier)
                return BlackSwanEvent(
                    trigger=BlackSwanTrigger.VOLUME_SPIKE,
                    symbol=symbol,
                    severity=severity,
                    detail=(
                        f"Volume {ratio:.1f}× rolling mean "
                        f"(threshold {self.volume_spike_multiplier:.1f}×)"
                    ),
                    timestamp=ts,
                )

        # 3. Extreme funding rate
        if abs(funding_rate) >= self.max_funding_rate_abs:
            severity = min(1.0, abs(funding_rate) / self.max_funding_rate_abs)
            return BlackSwanEvent(
                trigger=BlackSwanTrigger.FUNDING_EXTREME,
                symbol=symbol,
                severity=severity,
                detail=(
                    f"Funding rate {funding_rate:.4f} ≥ threshold "
                    f"±{self.max_funding_rate_abs:.4f}"
                ),
                timestamp=ts,
            )

        # 4. Stablecoin depeg
        if symbol in self.stablecoin_symbols:
            deviation = abs(close_now - 1.0)
            if deviation >= self.stablecoin_depeg_pct:
                severity = min(1.0, deviation / self.stablecoin_depeg_pct)
                return BlackSwanEvent(
                    trigger=BlackSwanTrigger.DEPEG_DETECTED,
                    symbol=symbol,
                    severity=severity,
                    detail=(
                        f"Price ${close_now:.4f} deviates {deviation:.3%} from $1.00 "
                        f"(threshold {self.stablecoin_depeg_pct:.3%})"
                    ),
                    timestamp=ts,
                )

        return None
