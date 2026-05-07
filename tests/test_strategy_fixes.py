"""
Phase 13 test suite — DirectionalStrategy fixes.

Six tests covering:
  H-3: on_fill mirrors the simulator's Position (real stop_price), not 1% ATR fab
  H-8: on_bar_end has explicit guard (survives python -O), no AssertionError
  H-9: Trailing stop fires on bar.low (LONG) / bar.high (SHORT), not just close
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from privateye.core.types import (
    DataSnapshot, Direction, Fill, OrderSide, PortfolioState, Position,
)
from privateye.strategies.directional import DirectionalStrategy


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bars(n: int = 250, base: float = 40_000.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = base + np.cumsum(rng.normal(0, 200, n))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
        "open":   closes,
        "high":   closes * 1.01,
        "low":    closes * 0.99,
        "close":  closes,
        "volume": rng.uniform(500, 5000, n),
    })


def _strategy(**overrides) -> DirectionalStrategy:
    cfg = {
        "enabled": True,
        "timeframe": "1h",
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "ema_trend": 200, "rsi_period": 14,
        "rsi_overbought": 70, "rsi_oversold": 30,
        "atr_stop_multiplier": 2.0,
        "atr_target_multiplier": 4.0,
        "max_bars_in_trade": 48,
    }
    cfg.update(overrides)
    return DirectionalStrategy(cfg)


def _portfolio_with(symbol: str, position: Position, cash: float = 10_000.0,
                    equity: float = 10_000.0) -> PortfolioState:
    return PortfolioState(equity=equity, cash=cash, positions={symbol: position})


def _fill(side: OrderSide, qty: float = 0.1, price: float = 40_000.0,
          symbol: str = "BTC/USDT") -> Fill:
    return Fill(
        order_id="o1", symbol=symbol, side=side, quantity=qty,
        price=price, fee=qty * price * 0.001, strategy_id="directional",
    )


# ── H-3: real stop_price propagation ──────────────────────────────────────────

class TestOnFillUsesRealStop:

    def test_on_fill_mirrors_simulator_position_stop(self):
        """The simulator's Position carries the real ATR-derived stop. on_fill
        must mirror that, not fabricate fill_price * 0.99 from a 1% ATR proxy."""
        strat = _strategy()
        # Simulator position has a tight stop derived from real ATR
        real_stop = 39_500.0
        sim_pos = Position(
            symbol="BTC/USDT", side=Direction.LONG, quantity=0.1,
            entry_price=40_000.0, stop_price=real_stop, target_price=42_000.0,
            strategy_id="directional",
        )
        portfolio = _portfolio_with("BTC/USDT", sim_pos)
        strat.on_fill(_fill(OrderSide.BUY), portfolio)
        tracked = strat._get_position("BTC/USDT")
        assert tracked is not None
        assert tracked.stop_price == real_stop
        assert tracked.target_price == 42_000.0
        assert tracked.trailing_stop == real_stop

    def test_on_fill_short_mirrors_sim_position(self):
        """SHORT entry path: previously a no-op, now creates strategy-side tracker."""
        strat = _strategy()
        sim_pos = Position(
            symbol="BTC/USDT", side=Direction.SHORT, quantity=0.1,
            entry_price=40_000.0, stop_price=40_500.0, target_price=38_000.0,
            strategy_id="directional",
        )
        portfolio = _portfolio_with("BTC/USDT", sim_pos)
        strat.on_fill(_fill(OrderSide.SELL), portfolio)
        tracked = strat._get_position("BTC/USDT")
        assert tracked is not None
        assert tracked.side == Direction.SHORT
        assert tracked.stop_price == 40_500.0


# ── H-8: explicit guard ───────────────────────────────────────────────────────

class TestOnBarEndGuard:

    def test_on_bar_end_no_position_returns_empty(self):
        """No assert; explicit guard returns []."""
        strat = _strategy()
        bars = _bars(250)
        snap = DataSnapshot(symbol="BTC/USDT", timeframe="1h", bars=bars)
        portfolio = PortfolioState(equity=10_000.0, cash=10_000.0)
        # No position registered with the strategy
        result = strat.on_bar_end(snap, portfolio)
        assert result == []


# ── H-9: trailing stop uses low/high ──────────────────────────────────────────

class TestTrailingStopUsesLowHigh:

    def test_long_trailing_fires_on_bar_low_pierce(self):
        """Bar dips below trailing stop intra-bar even if close stays above."""
        strat = _strategy()
        bars = _bars(250)
        # Last bar: high above stop, close above stop, but LOW below stop
        bars.iloc[-1, bars.columns.get_loc("low")]   = 39_500.0
        bars.iloc[-1, bars.columns.get_loc("close")] = 40_100.0
        bars.iloc[-1, bars.columns.get_loc("high")]  = 40_200.0
        # Inject a long position with trailing_stop above the bar's low.
        # bars_held=1 simulates "this is the second on_bar_end call after entry";
        # on_bar_end increments to 2, so the post-2026-05-07 entry-bar guard
        # (skip exit checks when bars_held==1) does not block the stop.
        pos = Position(
            symbol="BTC/USDT", side=Direction.LONG, quantity=0.1,
            entry_price=40_000.0, stop_price=39_800.0, target_price=42_000.0,
            strategy_id="directional", trailing_stop=39_800.0, bars_held=1,
        )
        strat._open_position("BTC/USDT", pos)
        strat._extremes["BTC/USDT"] = 40_100.0

        snap = DataSnapshot(symbol="BTC/USDT", timeframe="1h", bars=bars)
        portfolio = _portfolio_with("BTC/USDT", pos)
        signals = strat.on_bar_end(snap, portfolio)
        # Trailing stop must fire: bar.low (39500) <= trailing_stop (39800)
        assert len(signals) == 1
        assert signals[0].direction == Direction.FLAT
        assert "trailing_stop" in signals[0].metadata.get("reason", "")

    def test_short_trailing_fires_on_bar_high_pierce(self):
        strat = _strategy()
        bars = _bars(250)
        bars.iloc[-1, bars.columns.get_loc("high")]  = 40_500.0
        bars.iloc[-1, bars.columns.get_loc("close")] = 39_900.0
        bars.iloc[-1, bars.columns.get_loc("low")]   = 39_800.0
        # bars_held=1: see test_long_trailing_fires_on_bar_low_pierce above.
        pos = Position(
            symbol="BTC/USDT", side=Direction.SHORT, quantity=0.1,
            entry_price=40_000.0, stop_price=40_200.0, target_price=38_000.0,
            strategy_id="directional", trailing_stop=40_200.0, bars_held=1,
        )
        strat._open_position("BTC/USDT", pos)
        strat._extremes["BTC/USDT"] = 39_900.0
        snap = DataSnapshot(symbol="BTC/USDT", timeframe="1h", bars=bars)
        portfolio = _portfolio_with("BTC/USDT", pos)
        signals = strat.on_bar_end(snap, portfolio)
        assert len(signals) == 1
        assert "trailing_stop" in signals[0].metadata.get("reason", "")

    def test_long_trailing_does_not_fire_when_low_above_stop(self):
        """Sanity: bars that don't pierce the stop do not exit."""
        strat = _strategy()
        bars = _bars(250)
        # Force a bar whose low stays above the trailing stop
        bars.iloc[-1, bars.columns.get_loc("low")]   = 39_900.0
        bars.iloc[-1, bars.columns.get_loc("close")] = 40_100.0
        bars.iloc[-1, bars.columns.get_loc("high")]  = 40_200.0
        pos = Position(
            symbol="BTC/USDT", side=Direction.LONG, quantity=0.1,
            entry_price=40_000.0, stop_price=39_500.0, target_price=80_000.0,
            strategy_id="directional", trailing_stop=39_500.0,
        )
        strat._open_position("BTC/USDT", pos)
        strat._extremes["BTC/USDT"] = 40_100.0
        snap = DataSnapshot(symbol="BTC/USDT", timeframe="1h", bars=bars)
        portfolio = _portfolio_with("BTC/USDT", pos)
        signals = strat.on_bar_end(snap, portfolio)
        # No exit signal — stop not pierced, target far away
        assert signals == []
