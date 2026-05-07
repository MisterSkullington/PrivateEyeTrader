"""
Timeframe-guard tests for ``on_bar_end`` (hotfix 2026-05-07).

Discovered in production paper-mode log: with five timeframes configured
(1m/5m/1h/4h/1d), ``on_bar_end`` of both ``DirectionalStrategy`` and
``MeanReversionStrategy`` ran on every MARKET_DATA event regardless of
timeframe. Effects:

  • ``bars_held`` increments 5× per poll → ``max_bars_in_trade`` timeouts hit
    in minutes instead of hours
  • ``trailing_stop`` recomputed with ATR from the wrong timeframe
  • Stop/target compared against stale closes (the 1d "close" is yesterday's
    daily close — frequently lower than today's BTC price, tripping the stop
    immediately after entry)

The fix mirrors the timeframe guard already present on ``on_data``:
``if snapshot.timeframe != self.timeframe: return []``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from privateye.core.types import DataSnapshot, Direction, PortfolioState, Position
from privateye.strategies.directional import DirectionalStrategy
from privateye.strategies.mean_reversion import MeanReversionStrategy


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _bars(n: int = 250, base: float = 40_000.0, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = base + np.cumsum(rng.normal(0, 200, n))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
        "open":   closes,
        "high":   closes * 1.005,
        "low":    closes * 0.995,
        "close":  closes,
        "volume": rng.uniform(500, 5000, n),
    })


def _portfolio_with(symbol: str, position: Position) -> PortfolioState:
    return PortfolioState(equity=10_000.0, cash=5_000.0,
                          positions={symbol: position})


# ── DirectionalStrategy ──────────────────────────────────────────────────────

class TestDirectionalOnBarEndTimeframe:

    @pytest.fixture
    def strat(self):
        return DirectionalStrategy({
            "enabled": True,
            "timeframe": "1h",
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "ema_trend": 200, "rsi_period": 14,
            "rsi_overbought": 70, "rsi_oversold": 30,
            "atr_stop_multiplier": 2.0,
            "atr_target_multiplier": 4.0,
            "max_bars_in_trade": 48,
        })

    def _seed_long_position(self, strat, symbol: str = "BTC/USDT"):
        """Plant a LONG position whose stop is well below current price so
        only a timeframe filter (not a stop-pierce) can suppress an exit."""
        pos = Position(
            symbol=symbol, side=Direction.LONG, quantity=0.1,
            entry_price=40_000.0, stop_price=39_000.0,
            target_price=42_000.0,
            strategy_id="directional",
            trailing_stop=39_000.0,
        )
        strat._open_position(symbol, pos)
        strat._extremes[symbol] = 40_000.0
        return pos

    def test_returns_empty_on_non_matching_timeframe(self, strat):
        """A 1m snapshot must not advance bars_held or trigger exits when the
        strategy's configured timeframe is 1h."""
        pos = self._seed_long_position(strat)
        snap = DataSnapshot(symbol="BTC/USDT", timeframe="1m", bars=_bars(250))
        portfolio = _portfolio_with("BTC/USDT", pos)

        signals = strat.on_bar_end(snap, portfolio)
        assert signals == []
        # bars_held must not be incremented by a non-target timeframe
        assert pos.bars_held == 0

    @pytest.mark.parametrize("foreign_tf", ["1m", "5m", "4h", "1d"])
    def test_each_non_matching_timeframe_is_skipped(self, strat, foreign_tf):
        pos = self._seed_long_position(strat)
        snap = DataSnapshot(symbol="BTC/USDT", timeframe=foreign_tf, bars=_bars(250))
        signals = strat.on_bar_end(snap, _portfolio_with("BTC/USDT", pos))
        assert signals == []
        assert pos.bars_held == 0

    def test_matching_timeframe_still_runs(self, strat):
        """Sanity: 1h snapshot still increments bars_held and does normal exit checks."""
        pos = self._seed_long_position(strat)
        snap = DataSnapshot(symbol="BTC/USDT", timeframe="1h", bars=_bars(250))
        strat.on_bar_end(snap, _portfolio_with("BTC/USDT", pos))
        assert pos.bars_held == 1   # incremented exactly once

    def test_five_timeframes_only_increment_once(self, strat):
        """Regression: simulating one full polling cycle (1m/5m/1h/4h/1d)
        must increment bars_held exactly 1× — not 5×."""
        pos = self._seed_long_position(strat)
        portfolio = _portfolio_with("BTC/USDT", pos)
        for tf in ("1m", "5m", "1h", "4h", "1d"):
            snap = DataSnapshot(symbol="BTC/USDT", timeframe=tf, bars=_bars(250))
            strat.on_bar_end(snap, portfolio)
        assert pos.bars_held == 1


# ── MeanReversionStrategy ────────────────────────────────────────────────────

class TestMeanReversionOnBarEndTimeframe:

    @pytest.fixture
    def strat(self):
        return MeanReversionStrategy({
            "enabled": True,
            "timeframe": "1h",
            "bb_period": 20, "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_overbought": 70, "rsi_oversold": 30,
            "atr_stop_multiplier": 1.0,
            "max_bars_in_trade": 24,
        })

    def _seed_long_position(self, strat, symbol: str = "BTC/USDT"):
        pos = Position(
            symbol=symbol, side=Direction.LONG, quantity=0.1,
            entry_price=40_000.0, stop_price=39_000.0,
            target_price=42_000.0,
            strategy_id="mean_reversion",
        )
        strat._open_position(symbol, pos)
        return pos

    @pytest.mark.parametrize("foreign_tf", ["1m", "5m", "4h", "1d"])
    def test_non_matching_timeframe_short_circuits(self, strat, foreign_tf):
        pos = self._seed_long_position(strat)
        snap = DataSnapshot(symbol="BTC/USDT", timeframe=foreign_tf, bars=_bars(250))
        signals = strat.on_bar_end(snap, _portfolio_with("BTC/USDT", pos))
        assert signals == []
        assert pos.bars_held == 0

    def test_matching_timeframe_runs(self, strat):
        pos = self._seed_long_position(strat)
        snap = DataSnapshot(symbol="BTC/USDT", timeframe="1h", bars=_bars(250))
        strat.on_bar_end(snap, _portfolio_with("BTC/USDT", pos))
        assert pos.bars_held == 1

    def test_stale_daily_close_does_not_trip_stop(self, strat):
        """The exact production failure: a 1d snapshot whose latest close is
        below the LONG stop must NOT trigger an exit when the strategy's
        timeframe is 1h. Pre-fix, it did."""
        pos = self._seed_long_position(strat)
        # Force the 1d snapshot's last close BELOW the position's stop (38_500 < 39_000).
        bars = _bars(250)
        bars.iloc[-1, bars.columns.get_loc("close")] = 38_500.0
        snap = DataSnapshot(symbol="BTC/USDT", timeframe="1d", bars=bars)
        signals = strat.on_bar_end(snap, _portfolio_with("BTC/USDT", pos))
        assert signals == [], (
            "1d snapshot must not be allowed to trip the 1h strategy's stop"
        )

    def test_five_timeframes_only_increment_once(self, strat):
        pos = self._seed_long_position(strat)
        portfolio = _portfolio_with("BTC/USDT", pos)
        for tf in ("1m", "5m", "1h", "4h", "1d"):
            snap = DataSnapshot(symbol="BTC/USDT", timeframe=tf, bars=_bars(250))
            strat.on_bar_end(snap, portfolio)
        assert pos.bars_held == 1
