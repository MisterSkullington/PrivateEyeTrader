"""
Entry-bar exit-guard tests (hotfix 2026-05-07, Option B).

Discovered while running a full-history backtest after today's earlier hotfixes:
the strategy was producing thousands of trades with avg_bars_held=0 and 100%
loss rate. Diagnosis: ``on_bar_end`` runs on the same bar as entry (line 128
of ``backtesting/engine.py``). On that bar:

  - The position has just been opened with ``trailing_stop = close − 2*ATR``
  - ``bar_low`` is the entry-bar's actual low — which happened *before* the
    fill in real time (entry would only have been possible after MACD-cross
    confirmed at the close)
  - ``bar_low <= trailing_stop`` therefore fires for any bar whose intra-bar
    range exceeds 2*ATR — which is most volatile bars

The fix (Option B): still increment ``bars_held`` and update the trailing
stop on the entry bar (so the stop is correctly armed), but skip the exit
*check* when ``bars_held == 1``. The first bar that can produce an exit is
the bar AFTER entry (bars_held == 2 after the increment).

Mirror tests cover both DirectionalStrategy and MeanReversionStrategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from privateye.core.types import DataSnapshot, Direction, PortfolioState, Position
from privateye.strategies.directional import DirectionalStrategy
from privateye.strategies.mean_reversion import MeanReversionStrategy


def _bars(n: int = 250, base: float = 40_000.0, seed: int = 17) -> pd.DataFrame:
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


def _portfolio_with(symbol: str, position: Position) -> PortfolioState:
    return PortfolioState(equity=10_000.0, cash=5_000.0, positions={symbol: position})


# ── DirectionalStrategy ──────────────────────────────────────────────────────

class TestDirectionalEntryBarGuard:

    @pytest.fixture
    def strat(self):
        return DirectionalStrategy({
            "enabled": True, "timeframe": "1h",
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "ema_trend": 200, "rsi_period": 14,
            "rsi_overbought": 70, "rsi_oversold": 30,
            "atr_stop_multiplier": 2.0, "atr_target_multiplier": 4.0,
            "max_bars_in_trade": 48,
        })

    def _seed_long_with_low_below_stop(self, strat, symbol: str = "BTC/USDT"):
        """Plant a fresh LONG position (bars_held=0) and a bar whose low
        already pierces the trailing_stop. Pre-fix, on_bar_end would emit
        an instant 'trailing_stop' exit on the entry bar."""
        pos = Position(
            symbol=symbol, side=Direction.LONG, quantity=0.1,
            entry_price=40_000.0, stop_price=39_800.0,
            target_price=42_000.0,
            strategy_id="directional",
            trailing_stop=39_800.0,
            bars_held=0,  # explicit: this is the entry bar
        )
        strat._open_position(symbol, pos)
        strat._extremes[symbol] = 40_100.0
        return pos

    def test_no_exit_on_entry_bar_even_when_low_pierces_stop(self, strat):
        """The exact production failure: bar_low (39_500) <= trailing_stop
        (39_800), but bars_held was 0 going in. On the entry bar the strategy
        must NOT exit — that low pre-dates the fill in real time."""
        pos = self._seed_long_with_low_below_stop(strat)
        bars = _bars(250)
        bars.iloc[-1, bars.columns.get_loc("low")]   = 39_500.0
        bars.iloc[-1, bars.columns.get_loc("close")] = 40_100.0
        bars.iloc[-1, bars.columns.get_loc("high")]  = 40_200.0

        snap = DataSnapshot(symbol="BTC/USDT", timeframe="1h", bars=bars)
        signals = strat.on_bar_end(snap, _portfolio_with("BTC/USDT", pos))

        assert signals == [], "Entry bar must not produce an exit signal"
        # Side effects: bars_held incremented, trailing_stop still updated.
        assert pos.bars_held == 1
        # trailing_stop is max(39_800, close - 2*ATR); for our synthetic bars
        # it should remain a positive number well above the bar's low.
        assert pos.trailing_stop > 0

    def test_exit_fires_on_second_bar_when_low_still_pierces(self, strat):
        """Sanity: on the bar AFTER entry, the same condition does fire."""
        pos = self._seed_long_with_low_below_stop(strat)
        # Simulate that one full bar has elapsed since the fill
        pos.bars_held = 1
        bars = _bars(250)
        bars.iloc[-1, bars.columns.get_loc("low")]   = 39_500.0
        bars.iloc[-1, bars.columns.get_loc("close")] = 40_100.0
        bars.iloc[-1, bars.columns.get_loc("high")]  = 40_200.0

        snap = DataSnapshot(symbol="BTC/USDT", timeframe="1h", bars=bars)
        signals = strat.on_bar_end(snap, _portfolio_with("BTC/USDT", pos))
        assert len(signals) == 1
        assert "trailing_stop" in signals[0].metadata.get("reason", "")

    def test_short_no_exit_on_entry_bar_even_when_high_pierces_stop(self, strat):
        pos = Position(
            symbol="BTC/USDT", side=Direction.SHORT, quantity=0.1,
            entry_price=40_000.0, stop_price=40_200.0,
            target_price=38_000.0,
            strategy_id="directional",
            trailing_stop=40_200.0,
            bars_held=0,
        )
        strat._open_position("BTC/USDT", pos)
        strat._extremes["BTC/USDT"] = 39_900.0
        bars = _bars(250)
        bars.iloc[-1, bars.columns.get_loc("high")]  = 40_500.0   # pierces stop
        bars.iloc[-1, bars.columns.get_loc("close")] = 39_900.0
        bars.iloc[-1, bars.columns.get_loc("low")]   = 39_800.0

        snap = DataSnapshot(symbol="BTC/USDT", timeframe="1h", bars=bars)
        signals = strat.on_bar_end(snap, _portfolio_with("BTC/USDT", pos))

        assert signals == [], "SHORT entry bar must not produce an exit signal"
        assert pos.bars_held == 1

    def test_trailing_stop_still_armed_on_entry_bar(self, strat):
        """Option B preserves arming behavior — trailing_stop must still
        update on the entry bar so the next bar's check uses a current value."""
        pos = self._seed_long_with_low_below_stop(strat)
        original_stop = pos.trailing_stop
        bars = _bars(250)
        # Make sure the bar's close is above original stop so trailing_stop
        # could be ratcheted UP; in practice it max()es with new_stop.
        bars.iloc[-1, bars.columns.get_loc("close")] = 41_000.0
        bars.iloc[-1, bars.columns.get_loc("high")]  = 41_100.0
        bars.iloc[-1, bars.columns.get_loc("low")]   = 40_800.0

        snap = DataSnapshot(symbol="BTC/USDT", timeframe="1h", bars=bars)
        strat.on_bar_end(snap, _portfolio_with("BTC/USDT", pos))
        # bars_held should advance, trailing_stop should not regress
        assert pos.bars_held == 1
        assert pos.trailing_stop >= original_stop


# ── MeanReversionStrategy ────────────────────────────────────────────────────

class TestMeanReversionEntryBarGuard:

    @pytest.fixture
    def strat(self):
        return MeanReversionStrategy({
            "enabled": True, "timeframe": "1h",
            "bb_period": 20, "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_overbought": 70, "rsi_oversold": 30,
            "atr_stop_multiplier": 1.0, "max_bars_in_trade": 24,
        })

    def test_no_exit_on_entry_bar_even_when_close_pierces_stop(self, strat):
        """Symmetric MR test: the close that produced the entry signal would
        also satisfy ``close <= stop_price`` for any pullback-buy scenario.
        The entry bar must not be allowed to immediately exit."""
        pos = Position(
            symbol="BTC/USDT", side=Direction.LONG, quantity=0.1,
            entry_price=40_000.0,
            stop_price=40_500.0,    # contrived: stop above entry to force the trigger
            target_price=42_000.0,
            strategy_id="mean_reversion",
            bars_held=0,
        )
        strat._open_position("BTC/USDT", pos)
        bars = _bars(250)
        bars.iloc[-1, bars.columns.get_loc("close")] = 40_300.0   # < stop_price 40_500

        snap = DataSnapshot(symbol="BTC/USDT", timeframe="1h", bars=bars)
        signals = strat.on_bar_end(snap, _portfolio_with("BTC/USDT", pos))

        assert signals == [], "MR entry bar must not produce an exit signal"
        assert pos.bars_held == 1

    def test_exit_fires_on_second_bar(self, strat):
        pos = Position(
            symbol="BTC/USDT", side=Direction.LONG, quantity=0.1,
            entry_price=40_000.0, stop_price=40_500.0, target_price=42_000.0,
            strategy_id="mean_reversion",
            bars_held=1,   # second bar after entry
        )
        strat._open_position("BTC/USDT", pos)
        bars = _bars(250)
        bars.iloc[-1, bars.columns.get_loc("close")] = 40_300.0

        snap = DataSnapshot(symbol="BTC/USDT", timeframe="1h", bars=bars)
        signals = strat.on_bar_end(snap, _portfolio_with("BTC/USDT", pos))
        assert len(signals) == 1
        assert signals[0].direction == Direction.FLAT
