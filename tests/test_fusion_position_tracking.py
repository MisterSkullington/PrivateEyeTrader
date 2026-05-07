"""
FusionStrategy position-tracking delegation tests (hotfix 2026-05-07 — phase 2).

Discovered in production paper-mode log: when FusionStrategy emits a LONG
signal (with stops computed by ``_build_base_signal`` using ATR), the BUY
fill arrived at FusionStrategy.on_fill which delegated to BOTH
``_directional.on_fill`` AND ``_mean_reversion.on_fill``. The MR delegation
was a bug:

  1. MR.on_fill opened a phantom internal position with placeholder values
     (stop = fill.price * 0.98, target = fill.price * 1.02), unrelated to
     the actual order's stops.
  2. MR.on_bar_end then evaluated those placeholder exit conditions on
     every poll, emitting spurious flat signals.
  3. Because MR.on_data short-circuits when ``_has_position`` is True, the
     phantom position also suppressed MR's contribution to the entry vote.

The fix removes both delegations:
  - ``FusionStrategy.on_fill`` only calls ``_directional.on_fill``
  - ``FusionStrategy.on_bar_end`` only calls ``_directional.on_bar_end``

Directional is the only sub-strategy whose ``on_fill`` mirrors the simulator's
authoritative Position via ``dataclasses.replace(sim_pos)``, so it carries the
real stop/target from the originating order — the only viable exit manager.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from privateye.core.types import (
    DataSnapshot, Direction, Fill, OrderSide, PortfolioState, Position,
)
from privateye.strategies.fusion_strategy import FusionStrategy


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _bars(n: int = 250, base: float = 40_000.0, seed: int = 13) -> pd.DataFrame:
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


def _portfolio_with_long(symbol: str = "BTC/USDT") -> PortfolioState:
    """Simulate the simulator's authoritative Position after a fill."""
    pos = Position(
        symbol=symbol, side=Direction.LONG, quantity=0.1,
        entry_price=40_000.0,
        stop_price=39_000.0,           # real ATR-derived stop from the order
        target_price=42_000.0,
        strategy_id="fusion",
    )
    return PortfolioState(equity=10_000.0, cash=5_000.0, positions={symbol: pos})


@pytest.fixture
def fusion():
    return FusionStrategy({
        "strategy_id":         "fusion",
        "timeframe":           "1h",
        "enabled":             True,
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "ema_trend": 200, "rsi_period": 14,
        "rsi_overbought": 70, "rsi_oversold": 30,
        "bb_period": 20, "bb_std": 2.0,
        "atr_stop_multiplier": 2.0, "max_bars_in_trade": 48,
        "gbm_gate": True, "gbm_gate_threshold": 0.45,
        "regime_gate": True, "sharpe_window_trades": 30,
    })


def _buy_fill(symbol: str = "BTC/USDT", qty: float = 0.1, price: float = 40_000.0) -> Fill:
    return Fill(
        order_id="o1", symbol=symbol, side=OrderSide.BUY,
        quantity=qty, price=price, fee=qty * price * 0.001,
        strategy_id="fusion",
    )


# ── Phantom-position regression ──────────────────────────────────────────────

class TestNoPhantomMRPosition:

    def test_buy_fill_does_not_create_mr_phantom_position(self, fusion):
        """The exact production bug: BUY fill must NOT create an MR internal
        position. Pre-fix, ``self._mean_reversion._positions`` would contain
        a placeholder Position with stop=fill*0.98 / target=fill*1.02."""
        portfolio = _portfolio_with_long()
        fusion.on_fill(_buy_fill(), portfolio)

        assert fusion._mean_reversion._positions == {}, (
            "MR must not track positions for FusionStrategy-emitted trades"
        )

    def test_directional_still_tracks_after_buy_fill(self, fusion):
        """Sanity: the legitimate delegation to directional still works.
        Directional.on_fill mirrors the simulator's Position so it carries
        the real stop/target from the order."""
        portfolio = _portfolio_with_long()
        fusion.on_fill(_buy_fill(), portfolio)

        assert "BTC/USDT" in fusion._directional._positions
        tracked = fusion._directional._positions["BTC/USDT"]
        assert tracked.stop_price == 39_000.0       # mirrored from sim_pos
        assert tracked.target_price == 42_000.0     # mirrored from sim_pos

    def test_on_bar_end_does_not_call_mr_exit_logic(self, fusion):
        """Even if MR somehow had a phantom position (e.g. left over from a
        previous bug), on_bar_end must not surface MR exit signals."""
        # Simulate the legacy phantom: manually plant an MR position
        phantom = Position(
            symbol="BTC/USDT", side=Direction.LONG, quantity=0.1,
            entry_price=40_000.0,
            stop_price=39_200.0,    # placeholder stop = fill * 0.98
            target_price=40_800.0,  # placeholder target = fill * 1.02
            strategy_id="fusion_mean_reversion",
        )
        fusion._mean_reversion._open_position("BTC/USDT", phantom)

        # Bars whose close (39_900) sits below the phantom MR stop (39_200)
        bars = _bars(250)
        bars.iloc[-1, bars.columns.get_loc("close")] = 39_100.0   # below phantom MR stop

        snap = DataSnapshot(symbol="BTC/USDT", timeframe="1h", bars=bars)
        portfolio = _portfolio_with_long()

        # Pre-fix this would emit an MR-driven flat signal. Post-fix, MR is
        # no longer consulted, so any exit must come from directional only.
        # Directional has no internal position → returns []. No exit fired.
        signals = fusion.on_bar_end(snap, portfolio)
        assert signals == [], (
            "MR exit logic must not be reachable from FusionStrategy.on_bar_end"
        )

    def test_sell_fill_does_not_touch_mr_state(self, fusion):
        """Closing fill must also not invoke MR (it was never opened)."""
        # Open via BUY first
        portfolio = _portfolio_with_long()
        fusion.on_fill(_buy_fill(), portfolio)

        # Now SELL fill closes the position in the simulator
        sell = Fill(
            order_id="o2", symbol="BTC/USDT", side=OrderSide.SELL,
            quantity=0.1, price=40_500.0, fee=4.05, strategy_id="fusion",
            realised_pnl=50.0,
        )
        # Simulate empty portfolio after sell
        empty_portfolio = PortfolioState(equity=10_050.0, cash=10_050.0, positions={})
        fusion.on_fill(sell, empty_portfolio)

        # MR was never opened, still empty
        assert fusion._mean_reversion._positions == {}
        # Directional tracker correctly cleared on SELL
        assert fusion._directional._positions == {}
        # P&L recorded into source PnL queues for Sharpe weighting
        for q in fusion._source_pnls.values():
            assert 50.0 in list(q)


# ── strategy_id round-trip on exit signals ──────────────────────────────────

class TestExitSignalStrategyIdRoundTrip:
    """Regression for the third bug discovered 2026-05-07: ``_directional``
    builds exit signals via ``_flat_signal`` using its own ``strategy_id``
    ("fusion_directional"). The main.py and backtest engine on_fill filter
    matches by parent strategy_id ("fusion"), so without the rewrite the
    exit fill is silently dropped — directional's tracker never closes,
    bars_held keeps growing, and exit signals fire on every subsequent bar."""

    def test_on_bar_end_rewrites_exit_signal_strategy_id_to_parent(self, fusion):
        # Plant a directional position with bars_held >= 2 so the entry-bar
        # guard doesn't skip the exit check, and force the trailing stop to
        # sit above the bar's low so the trailing_stop branch fires.
        pos = Position(
            symbol="BTC/USDT", side=Direction.LONG, quantity=0.1,
            entry_price=40_000.0, stop_price=39_800.0,
            target_price=42_000.0,
            strategy_id="fusion_directional",
            trailing_stop=39_800.0,
            bars_held=2,
        )
        fusion._directional._open_position("BTC/USDT", pos)
        fusion._directional._extremes["BTC/USDT"] = 40_100.0

        bars = _bars(250)
        bars.iloc[-1, bars.columns.get_loc("low")]   = 39_500.0
        bars.iloc[-1, bars.columns.get_loc("close")] = 40_100.0
        bars.iloc[-1, bars.columns.get_loc("high")]  = 40_200.0

        snap = DataSnapshot(symbol="BTC/USDT", timeframe="1h", bars=bars)
        portfolio = _portfolio_with_long()
        signals = fusion.on_bar_end(snap, portfolio)

        assert len(signals) == 1
        # Critical: the signal's strategy_id must be the parent ("fusion"),
        # not the sub-strategy's ("fusion_directional"). Without this rewrite
        # the engine's on_fill filter drops the resulting fill.
        assert signals[0].strategy_id == FusionStrategy.STRATEGY_ID
        assert signals[0].strategy_id == "fusion"

    def test_no_exits_returns_empty_list_not_error(self, fusion):
        """Sanity: the rewrite loop must not crash on empty results."""
        # No internal position planted
        bars = _bars(250)
        snap = DataSnapshot(symbol="BTC/USDT", timeframe="1h", bars=bars)
        portfolio = PortfolioState(equity=10_000.0, cash=10_000.0, positions={})
        result = fusion.on_bar_end(snap, portfolio)
        assert result == []


# ── Voting still works (MR still contributes via on_data) ────────────────────

class TestVotingStillWorks:
    """The fix only removes on_fill/on_bar_end delegation. on_data — which is
    where MR contributes votes during entry evaluation — is untouched."""

    def test_mr_can_still_contribute_votes(self, fusion, monkeypatch):
        """Verify FusionStrategy.on_data still reaches _mean_reversion.on_data."""
        called = {"mr": False}

        def fake_mr_on_data(snapshot):
            called["mr"] = True
            return []

        monkeypatch.setattr(fusion._mean_reversion, "on_data", fake_mr_on_data)
        # Disable the regime gate so MR is consulted regardless of regime —
        # otherwise the live regime detector's output (driven by the synthetic
        # bars) decides whether MR or directional is picked.
        fusion.regime_gate = False

        snap = DataSnapshot(symbol="BTC/USDT", timeframe="1h", bars=_bars(300))
        fusion.on_data(snap)
        assert called["mr"], "MR.on_data must still be called for entry voting"
