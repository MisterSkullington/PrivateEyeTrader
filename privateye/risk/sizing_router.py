"""
SizingRouter — dispatches to the configured position-sizing algorithm.

Replaces the two inline sizing lines in RiskManager, making the algorithm
a config-time choice without any new math.

Available methods:
  fixed_risk         — existing default (ATR stop-distance based)
  kelly              — half-Kelly from realised trade history
  volatility_targeted — scales to a daily vol contribution target
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Any

import pandas as pd

from privateye.core.types import TradingSignal, TradeRecord
from privateye.risk.sizing import (
    cap_to_max_notional,
    fixed_risk_size,
    kelly_size,
    volatility_targeted_size,
)
from privateye.utils.logging import get_logger

log = get_logger()

_KELLY_MIN_TRADES = 10   # require at least this many trades before using Kelly


class SizingMethod(str, Enum):
    FIXED_RISK          = "fixed_risk"
    KELLY               = "kelly"
    VOLATILITY_TARGETED = "volatility_targeted"


def route_sizing(
    method: SizingMethod,
    equity: float,
    config: dict[str, Any],
    signal: TradingSignal,
    bars: pd.DataFrame | None = None,
    trade_history: list[TradeRecord] | None = None,
) -> float:
    """
    Compute raw quantity for the signal (before cap_to_max_notional).

    Falls back to fixed_risk when required data is absent (no bars for
    vol-targeted, insufficient history for Kelly).
    """
    if method == SizingMethod.KELLY:
        qty = _kelly(equity, config, signal, trade_history)
    elif method == SizingMethod.VOLATILITY_TARGETED:
        qty = _vol_targeted(equity, config, signal, bars)
    else:
        qty = _fixed_risk(equity, config, signal)

    if qty <= 0:
        # Any failure path → fall back to fixed_risk
        log.debug(f"[SizingRouter] {method.value} returned 0 → falling back to fixed_risk")
        qty = _fixed_risk(equity, config, signal)

    return qty


# ── Private helpers ───────────────────────────────────────────────────────────

def _fixed_risk(equity: float, config: dict, signal: TradingSignal) -> float:
    return fixed_risk_size(
        equity=equity,
        risk_pct=config.get("max_risk_per_trade_pct", 0.01),
        entry_price=signal.entry_price,
        stop_price=signal.stop_price,
    )


def _kelly(
    equity: float,
    config: dict,
    signal: TradingSignal,
    trade_history: list[TradeRecord] | None,
) -> float:
    if not trade_history or len(trade_history) < _KELLY_MIN_TRADES:
        return 0.0  # triggers fallback

    wins   = [t.pnl for t in trade_history if t.pnl > 0]
    losses = [abs(t.pnl) for t in trade_history if t.pnl <= 0]
    if not wins or not losses:
        return 0.0

    win_rate = len(wins) / len(trade_history)
    avg_win  = sum(wins) / len(wins)
    avg_loss = sum(losses) / len(losses)
    fraction = config.get("kelly", {}).get("fraction", 0.5)

    notional = kelly_size(equity, win_rate, avg_win, avg_loss, fraction)
    price = signal.entry_price
    return notional / price if price > 0 else 0.0


def _vol_targeted(
    equity: float,
    config: dict,
    signal: TradingSignal,
    bars: pd.DataFrame | None,
) -> float:
    if bars is None or len(bars) < 21:
        return 0.0  # triggers fallback

    from privateye.indicators.library import historical_volatility
    hv_series = historical_volatility(bars, 20)
    if hv_series.empty or math.isnan(hv_series.iloc[-1]):
        return 0.0

    hv_ann   = float(hv_series.iloc[-1])
    daily_hv = hv_ann / math.sqrt(365) if hv_ann > 0 else 0.0
    if daily_hv <= 0:
        return 0.0

    target_vol = config.get("volatility_targeted", {}).get("target_vol_pct", 0.01)
    return volatility_targeted_size(equity, target_vol, daily_hv, signal.entry_price)
