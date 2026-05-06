"""Position sizing algorithms: fixed-risk (ATR-based), Kelly criterion, volatility-targeted."""
from __future__ import annotations

import math

from privateye.utils.logging import get_logger

log = get_logger()


def fixed_risk_size(
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
) -> float:
    """
    Size a position so that hitting the stop loses exactly risk_pct × equity.

    risk_amount = equity × risk_pct
    stop_distance = |entry - stop|
    quantity = risk_amount / stop_distance
    """
    if entry_price <= 0 or stop_price <= 0:
        return 0.0
    stop_distance = abs(entry_price - stop_price)
    if stop_distance < 1e-9:
        log.warning("Stop distance near zero — skipping position sizing")
        return 0.0
    risk_amount = equity * risk_pct
    qty = risk_amount / stop_distance
    return qty


def kelly_size(
    equity: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.5,  # half-Kelly for safety
) -> float:
    """
    Kelly fraction position size (half-Kelly by default).
    Returns fraction of equity to allocate.
    """
    if avg_loss <= 0 or win_rate <= 0:
        return 0.0
    odds = avg_win / avg_loss
    kelly_f = (win_rate * odds - (1 - win_rate)) / odds
    kelly_f = max(0.0, kelly_f)  # no negative sizing
    return equity * kelly_f * fraction


def volatility_targeted_size(
    equity: float,
    target_vol_pct: float,  # e.g. 0.01 = 1% daily vol target
    asset_daily_vol: float,  # realised daily vol of the asset (e.g. 0.03 = 3%)
    price: float,
) -> float:
    """
    Scale position so it contributes target_vol_pct to portfolio daily vol.
    quantity = (equity × target_vol_pct) / (price × asset_daily_vol)
    """
    if asset_daily_vol <= 0 or price <= 0:
        return 0.0
    notional = equity * target_vol_pct / asset_daily_vol
    return notional / price


def cap_to_max_notional(
    quantity: float,
    price: float,
    equity: float,
    max_notional_pct: float,
) -> float:
    """Cap quantity so notional <= max_notional_pct × equity."""
    max_notional = equity * max_notional_pct
    max_qty = max_notional / price if price > 0 else 0.0
    return min(quantity, max_qty)
