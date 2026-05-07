"""
BacktestEngine exit-fill on_fill propagation tests (hotfix 2026-05-07, fourth bug).

Discovered via direct trace during backtest verification: ``engine.run`` line 128
calls ``process_bar`` and dispatches resulting fills to ``strategy.on_fill``.
But the symmetric call at the end of an exit-signal loop (where the SELL is
submitted and a SECOND ``process_bar`` is invoked to fill it on the same bar)
DID NOT capture the resulting fills or call ``on_fill``. Consequence:

  - SELL fill closed the position in the simulator (TradeRecord recorded)
  - Strategy's internal position tracker never received the fill notification
  - ``_directional._has_position`` stayed True
  - Next bar's ``on_bar_end`` re-emitted a stale exit signal
  - Exit signal evaluated, ``_build_exit_order`` returned None (sim has no
    position), logged as "Signal rejected: exit"
  - Cycle repeated every bar; new BUYs opened the simulator position but
    couldn't open the strategy tracker because ``not _has_position`` was False
  - Strategy/simulator state diverged permanently

The fix: the exit ``process_bar`` call now captures the fills and dispatches
them through the same ``strategy.on_fill`` chain as entry fills.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from privateye.backtesting.engine import BacktestEngine
from privateye.backtesting.simulator import SimulatedExchange
from privateye.core.types import (
    DataSnapshot, Direction, Fill, OrderSide, PortfolioState, Position,
    TradingSignal,
)
from privateye.risk.manager import RiskManager
from privateye.strategies.base import AbstractStrategy


def _bars(n: int = 250, base: float = 40_000.0, seed: int = 11) -> pd.DataFrame:
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


# A minimal strategy that opens a position on the first bar past warmup, then
# emits a single FLAT signal on bar 5 — letting us observe the engine's
# exit-fill → on_fill plumbing in isolation.
class _ScriptedStrategy(AbstractStrategy):
    STRATEGY_ID = "scripted"

    def __init__(self):
        super().__init__({"strategy_id": "scripted", "enabled": True, "timeframe": "1h"})
        self.timeframe = "1h"
        self._counter = 0
        self.on_fill_invocations: list[OrderSide] = []   # for assertion

    def on_data(self, snapshot):
        self._counter += 1
        if self._counter == 1 and not self._has_position(snapshot.symbol):
            close = float(snapshot.bars["close"].iloc[-1])
            return [TradingSignal(
                symbol=snapshot.symbol, direction=Direction.LONG, confidence=0.9,
                entry_price=close, stop_price=close * 0.98, target_price=close * 1.04,
                strategy_id=self.STRATEGY_ID, timeframe="1h",
            )]
        return []

    def on_bar_end(self, snapshot, portfolio):
        if self._counter == 5 and self._has_position(snapshot.symbol):
            return [self._flat_signal(snapshot, "scripted_exit")]
        return []

    def on_fill(self, fill, portfolio):
        self.on_fill_invocations.append(fill.side)
        if fill.side == OrderSide.BUY and not self._has_position(fill.symbol):
            self._open_position(fill.symbol, Position(
                symbol=fill.symbol, side=Direction.LONG, quantity=fill.quantity,
                entry_price=fill.price, stop_price=fill.price * 0.98,
                target_price=fill.price * 1.04, strategy_id=self.STRATEGY_ID,
            ))
        elif fill.side == OrderSide.SELL and self._has_position(fill.symbol):
            self._close_position(fill.symbol)


# ── Engine plumbing ──────────────────────────────────────────────────────────

class TestEngineExitFillOnFillPropagation:

    def test_exit_fill_invokes_on_fill_so_strategy_tracker_clears(self):
        """The exact production failure: after the exit fires, the strategy's
        internal _positions dict must be empty — confirming on_fill was called
        for the SELL fill."""
        cfg = {
            "backtesting": {"initial_capital": 10_000.0},
            "data": {"bar_window": 500},
            "risk": {
                "max_risk_per_trade_pct": 0.01,
                "max_daily_drawdown_pct": 0.05,
                "max_position_notional_pct": 0.20,
                "atr_period": 14,
                "atr_stop_multiplier": 2.0,
                "max_bars_in_trade": 48,
                "min_confidence": 0.55,
                "sizing_method": "fixed_risk",
                "cash_buffer_pct": 0.01,
                "exchange_default_type": "spot",
                "advanced": {
                    "asset_filter": {"enabled": False},
                    "exposure_monitor": {"enabled": False},
                    "black_swan": {"enabled": False},
                },
            },
        }
        strat = _ScriptedStrategy()
        sim = SimulatedExchange(10_000.0, fee_taker=0.001, slippage_pct=0.0005)
        rm = RiskManager(cfg["risk"])
        bars = _bars(280)   # min_warmup = max(200, bar_window//2 = 250) = 250

        BacktestEngine([strat], rm, sim, cfg).run(bars, "BTC/USDT", "1h")

        # Both BUY (entry) and SELL (exit) on_fill must have fired.
        assert OrderSide.BUY in strat.on_fill_invocations, (
            "Entry on_fill did not fire (line-128 process_bar plumbing)"
        )
        assert OrderSide.SELL in strat.on_fill_invocations, (
            "Exit on_fill did not fire — engine's exit process_bar must "
            "capture fills and dispatch them through strategy.on_fill"
        )
        # And the strategy tracker must be empty (position closed).
        assert strat._positions == {}, (
            "Strategy tracker still has a position — on_fill SELL handler "
            "did not run, leaving a stale _positions entry"
        )

    def test_at_least_one_round_trip_recorded(self):
        """Sanity: TradeRecord exists and entry_time != exit_time."""
        cfg = {
            "backtesting": {"initial_capital": 10_000.0},
            "data": {"bar_window": 500},
            "risk": {
                "max_risk_per_trade_pct": 0.01,
                "max_daily_drawdown_pct": 0.05,
                "max_position_notional_pct": 0.20,
                "atr_period": 14, "atr_stop_multiplier": 2.0,
                "max_bars_in_trade": 48, "min_confidence": 0.55,
                "sizing_method": "fixed_risk", "cash_buffer_pct": 0.01,
                "exchange_default_type": "spot",
                "advanced": {
                    "asset_filter": {"enabled": False},
                    "exposure_monitor": {"enabled": False},
                    "black_swan": {"enabled": False},
                },
            },
        }
        strat = _ScriptedStrategy()
        sim = SimulatedExchange(10_000.0, fee_taker=0.001, slippage_pct=0.0005)
        rm = RiskManager(cfg["risk"])
        bars = _bars(280)
        BacktestEngine([strat], rm, sim, cfg).run(bars, "BTC/USDT", "1h")

        assert len(sim.trade_records) >= 1
        t = sim.trade_records[0]
        # Entry on bar 201 (first past warmup), exit on bar 205 (5th on_data call).
        # entry_time and exit_time must be DIFFERENT timestamps.
        assert t.entry_time != t.exit_time, (
            "Entry and exit on the same bar — engine plumbing or strategy guard"
            " is broken"
        )
