"""
Phase 5 advanced risk tests.

Covers:
  - AssetFilter: banned symbols, volume floor, daily HV ceiling, disabled passthrough
  - BlackSwanGuard: flash crash, volume spike, funding extreme, depeg, severity
  - BlackSwanGuard × BacktestEngine: flatten on trigger, halt on subsequent bars
  - ExposureMonitor: concentration, total notional, beta, disabled passthrough
  - SizingRouter: fixed_risk / kelly / vol_targeted dispatch + fallback
  - RiskManager: full gate ordering, backward-compatible None kwargs
"""
from __future__ import annotations

import datetime
import math
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from privateye.core.types import (
    Direction, Fill, Order, OrderSide, OrderStatus, OrderType,
    PortfolioState, Position, TradingSignal, TradeRecord,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

_NOW = datetime.datetime.now(datetime.timezone.utc)


def _bars(n: int = 100, close: float = 50_000.0, volume: float = 1_000.0,
          daily_vol_pct: float = 0.01) -> pd.DataFrame:
    """Synthetic OHLCV with controllable daily volatility."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, daily_vol_pct, n)
    closes = close * np.exp(np.cumsum(returns))
    highs = closes * 1.005
    lows = closes * 0.995
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volume,
    })


def _signal(entry: float = 50_000.0, stop: float = 49_000.0,
            target: float = 52_000.0, confidence: float = 0.70,
            symbol: str = "BTC/USDT") -> TradingSignal:
    return TradingSignal(
        symbol=symbol,
        direction=Direction.LONG,
        confidence=confidence,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        strategy_id="test",
        timeframe="1h",
        timestamp=_NOW,
    )


def _portfolio(equity: float = 10_000.0, cash: float | None = None,
               positions: dict | None = None) -> PortfolioState:
    return PortfolioState(
        equity=equity,
        cash=cash if cash is not None else equity,
        positions=positions or {},
        daily_pnl=0.0,
        daily_drawdown_pct=0.0,
        peak_equity=equity,
    )


def _trade(pnl: float) -> TradeRecord:
    return TradeRecord(
        symbol="BTC/USDT", side=Direction.LONG,
        entry_price=50_000.0, exit_price=50_000.0 * (1 + pnl / 500),
        quantity=0.01, entry_time=_NOW, exit_time=_NOW,
        pnl=pnl, pnl_pct=pnl / 500, fees=0.5,
        strategy_id="test", exit_reason="target", bars_held=5,
    )


# ── AssetFilter ───────────────────────────────────────────────────────────────

def test_asset_filter_disabled_passes_everything():
    from privateye.risk.asset_filter import AssetFilter
    af = AssetFilter(enabled=False, banned_symbols=["BTC/USDT"])
    ok, reason = af.is_tradeable("BTC/USDT", _bars())
    assert ok is True
    assert reason == ""


def test_banned_symbol_rejected():
    from privateye.risk.asset_filter import AssetFilter
    af = AssetFilter(enabled=True, banned_symbols=["LUNA/USDT"])
    ok, reason = af.is_tradeable("LUNA/USDT", _bars())
    assert ok is False
    assert "banned_symbols" in reason


def test_banned_symbol_case_insensitive():
    from privateye.risk.asset_filter import AssetFilter
    af = AssetFilter(enabled=True, banned_symbols=["LUNA/USDT"])
    ok, _ = af.is_tradeable("luna/usdt", _bars())
    assert ok is False


def test_insufficient_volume_rejected():
    from privateye.risk.asset_filter import AssetFilter
    # volume=1 per bar × close=50000 × 24 bars = $1.2M → below $2B floor
    af = AssetFilter(enabled=True, min_volume_usd_24h=2_000_000_000)
    ok, reason = af.is_tradeable("BTC/USDT", _bars(volume=1.0))
    assert ok is False
    assert "liquidity" in reason.lower()


def test_sufficient_volume_passes():
    from privateye.risk.asset_filter import AssetFilter
    # 1000 vol × 50000 price × 24 bars = $1.2B → above $1M floor
    af = AssetFilter(enabled=True, min_volume_usd_24h=1_000_000)
    ok, _ = af.is_tradeable("BTC/USDT", _bars(volume=1_000.0))
    assert ok is True


def test_high_volatility_rejected():
    from privateye.risk.asset_filter import AssetFilter
    # daily_vol_pct=0.50 → daily HV >> 15% ceiling
    af = AssetFilter(enabled=True, max_daily_vol_pct=0.15, min_volume_usd_24h=1)
    bars = _bars(n=100, daily_vol_pct=0.50)
    ok, reason = af.is_tradeable("SHIB/USDT", bars)
    assert ok is False
    assert "volatility" in reason.lower()


def test_normal_volatility_passes():
    from privateye.risk.asset_filter import AssetFilter
    # daily_vol_pct=0.005 → ~0.5% daily → well below 15% ceiling
    af = AssetFilter(enabled=True, max_daily_vol_pct=0.15, min_volume_usd_24h=1)
    bars = _bars(n=100, daily_vol_pct=0.005)
    ok, _ = af.is_tradeable("BTC/USDT", bars)
    assert ok is True


def test_none_bars_skips_filter_gate():
    """When bars=None is passed to evaluate_signal, AssetFilter gate is bypassed."""
    from privateye.risk.manager import RiskManager
    from privateye.risk.asset_filter import AssetFilter
    af = AssetFilter(enabled=True, banned_symbols=[], min_volume_usd_24h=9_999_999_999)
    rm = RiskManager({"min_confidence": 0.0, "max_risk_per_trade_pct": 0.01,
                      "max_daily_drawdown_pct": 0.5, "max_position_notional_pct": 1.0},
                     asset_filter=af)
    approved, _, _ = rm.evaluate_signal(_signal(), _portfolio(), bars=None)
    assert approved is True  # filter skipped because bars=None


# ── BlackSwanGuard ────────────────────────────────────────────────────────────

def test_guard_disabled_returns_none():
    from privateye.risk.black_swan_guard import BlackSwanGuard
    guard = BlackSwanGuard(enabled=False, flash_crash_pct=-0.01)
    bars = _bars(10)
    # Manufacture a -20% crash bar
    bars.iloc[-1, bars.columns.get_loc("close")] = float(bars["close"].iloc[-2]) * 0.79
    assert guard.check("BTC/USDT", bars) is None


def test_flash_crash_trigger():
    from privateye.risk.black_swan_guard import BlackSwanGuard, BlackSwanTrigger
    guard = BlackSwanGuard(enabled=True, flash_crash_pct=-0.08)
    bars = _bars(10, close=50_000.0, daily_vol_pct=0.001)
    # Set last bar to a -10% drop
    bars = bars.copy()
    bars.iloc[-1, bars.columns.get_loc("close")] = float(bars["close"].iloc[-2]) * 0.89
    event = guard.check("BTC/USDT", bars)
    assert event is not None
    assert event.trigger == BlackSwanTrigger.FLASH_CRASH


def test_flash_crash_not_triggered_small_drop():
    from privateye.risk.black_swan_guard import BlackSwanGuard
    guard = BlackSwanGuard(enabled=True, flash_crash_pct=-0.08)
    bars = _bars(10, daily_vol_pct=0.001)
    # -3% drop: below threshold
    bars = bars.copy()
    bars.iloc[-1, bars.columns.get_loc("close")] = float(bars["close"].iloc[-2]) * 0.97
    event = guard.check("BTC/USDT", bars)
    assert event is None


def test_volume_spike_trigger():
    from privateye.risk.black_swan_guard import BlackSwanGuard, BlackSwanTrigger
    guard = BlackSwanGuard(enabled=True, volume_spike_multiplier=5.0, volume_lookback=5,
                           flash_crash_pct=-0.99)  # disable flash crash
    bars = _bars(10, volume=100.0, daily_vol_pct=0.0001)
    bars = bars.copy()
    bars.iloc[-1, bars.columns.get_loc("volume")] = 700.0  # 7× rolling avg
    event = guard.check("BTC/USDT", bars)
    assert event is not None
    assert event.trigger == BlackSwanTrigger.VOLUME_SPIKE


def test_volume_spike_not_triggered_below_threshold():
    from privateye.risk.black_swan_guard import BlackSwanGuard
    guard = BlackSwanGuard(enabled=True, volume_spike_multiplier=5.0, volume_lookback=5,
                           flash_crash_pct=-0.99)
    bars = _bars(10, volume=100.0, daily_vol_pct=0.0001)
    bars = bars.copy()
    bars.iloc[-1, bars.columns.get_loc("volume")] = 400.0  # 4× < 5× threshold
    event = guard.check("BTC/USDT", bars)
    assert event is None


def test_funding_extreme_trigger():
    from privateye.risk.black_swan_guard import BlackSwanGuard, BlackSwanTrigger
    guard = BlackSwanGuard(enabled=True, max_funding_rate_abs=0.003, flash_crash_pct=-0.99,
                           volume_spike_multiplier=999.0)
    bars = _bars(10, daily_vol_pct=0.0001)
    event = guard.check("BTC/USDT", bars, funding_rate=0.005)
    assert event is not None
    assert event.trigger == BlackSwanTrigger.FUNDING_EXTREME


def test_depeg_trigger():
    from privateye.risk.black_swan_guard import BlackSwanGuard, BlackSwanTrigger
    guard = BlackSwanGuard(enabled=True, stablecoin_depeg_pct=0.005,
                           stablecoin_symbols=["USDC/USDT"],
                           flash_crash_pct=-0.99, volume_spike_multiplier=999.0,
                           max_funding_rate_abs=999.0)
    bars = _bars(10, close=0.993, daily_vol_pct=0.0001)  # $0.993 → 0.7% depeg
    event = guard.check("USDC/USDT", bars)
    assert event is not None
    assert event.trigger == BlackSwanTrigger.DEPEG_DETECTED


def test_severity_clamped_to_one():
    from privateye.risk.black_swan_guard import BlackSwanGuard, BlackSwanTrigger
    guard = BlackSwanGuard(enabled=True, flash_crash_pct=-0.08)
    bars = _bars(10, daily_vol_pct=0.0001)
    bars = bars.copy()
    # -50% crash: well beyond threshold
    bars.iloc[-1, bars.columns.get_loc("close")] = float(bars["close"].iloc[-2]) * 0.50
    event = guard.check("BTC/USDT", bars)
    assert event is not None
    assert event.severity <= 1.0


def test_insufficient_bars_no_crash():
    from privateye.risk.black_swan_guard import BlackSwanGuard
    guard = BlackSwanGuard(enabled=True, flash_crash_pct=-0.08)
    single_bar = _bars(1)
    event = guard.check("BTC/USDT", single_bar)
    assert event is None   # len < 2 → skip


# ── BlackSwanGuard × BacktestEngine ──────────────────────────────────────────

def _make_backtest_engine(guard=None):
    """Build a minimal BacktestEngine for integration tests."""
    from privateye.backtesting.engine import BacktestEngine
    from privateye.backtesting.simulator import SimulatedExchange
    from privateye.risk.manager import RiskManager
    from privateye.strategies.base import AbstractStrategy

    class _AlwaysBuy(AbstractStrategy):
        """Emits one LONG signal per bar."""
        STRATEGY_ID = "always_buy"
        def on_data(self, snapshot):
            close = float(snapshot.bars["close"].iloc[-1])
            return [TradingSignal(
                symbol=snapshot.symbol, direction=Direction.LONG,
                confidence=0.9, entry_price=close,
                stop_price=close * 0.90, target_price=close * 1.10,
                strategy_id=self.STRATEGY_ID, timeframe=snapshot.timeframe,
            )]
        def on_bar_end(self, snapshot, portfolio): return []
        def on_fill(self, fill, portfolio): pass

    exchange = SimulatedExchange(initial_capital=10_000.0)
    rm = RiskManager({"max_risk_per_trade_pct": 0.01, "min_confidence": 0.0,
                      "max_daily_drawdown_pct": 0.99, "max_position_notional_pct": 0.5})
    strategy = _AlwaysBuy({"strategy_id": "always_buy", "enabled": True, "timeframe": "1h"})
    cfg = {"data": {"bar_window": 200}, "backtesting": {"initial_capital": 10_000.0}}
    return BacktestEngine([strategy], rm, exchange, cfg, black_swan_guard=guard)


def test_black_swan_flattens_positions():
    from privateye.risk.black_swan_guard import BlackSwanGuard
    # Use a very tight flash_crash threshold so a normal bar triggers it
    guard = BlackSwanGuard(enabled=True, flash_crash_pct=-0.001)

    n = 250
    bars = _bars(n, daily_vol_pct=0.0001)
    # Make bar 240 have a huge drop
    bars = bars.copy()
    bars.iloc[240, bars.columns.get_loc("close")] = float(bars["close"].iloc[239]) * 0.50

    engine = _make_backtest_engine(guard)
    report = engine.run(bars, "BTC/USDT", "1h")
    # After the black-swan bar, positions should have been flattened
    assert len(engine.exchange.positions) == 0


def test_black_swan_halts_subsequent_entries():
    from privateye.risk.black_swan_guard import BlackSwanGuard
    guard = BlackSwanGuard(enabled=True, flash_crash_pct=-0.001)

    n = 250
    bars = _bars(n, daily_vol_pct=0.0001)
    bars = bars.copy()
    bars.iloc[240, bars.columns.get_loc("close")] = float(bars["close"].iloc[239]) * 0.50

    engine = _make_backtest_engine(guard)
    engine.run(bars, "BTC/USDT", "1h")
    # After the event, RiskManager should be halted
    assert engine.risk_manager.is_halted()


def test_black_swan_disabled_no_interrupt():
    from privateye.risk.black_swan_guard import BlackSwanGuard
    guard = BlackSwanGuard(enabled=False, flash_crash_pct=-0.001)

    n = 250
    bars = _bars(n, daily_vol_pct=0.0001)
    bars = bars.copy()
    bars.iloc[240, bars.columns.get_loc("close")] = float(bars["close"].iloc[239]) * 0.50

    engine = _make_backtest_engine(guard)
    engine.run(bars, "BTC/USDT", "1h")
    # Disabled guard: RiskManager should not be halted by the guard
    # (may be halted by drawdown — just check the guard didn't trigger)
    assert not engine.risk_manager._halt_reason.startswith("black_swan")


def test_no_crash_no_flatten():
    """Normal bars with guard enabled should not flatten positions."""
    from privateye.risk.black_swan_guard import BlackSwanGuard
    guard = BlackSwanGuard(enabled=True, flash_crash_pct=-0.20,
                           volume_spike_multiplier=50.0)
    n = 250
    bars = _bars(n, daily_vol_pct=0.001)
    engine = _make_backtest_engine(guard)
    report = engine.run(bars, "BTC/USDT", "1h")
    # No black-swan events → halt reason should not start with "black_swan"
    assert not engine.risk_manager._halt_reason.startswith("black_swan")


# ── ExposureMonitor ───────────────────────────────────────────────────────────

def test_exposure_monitor_disabled_approves_all():
    from privateye.risk.exposure_monitor import ExposureMonitor
    em = ExposureMonitor(enabled=False, max_concentration_pct=0.01)
    ok, _ = em.check_signal(_signal(), _portfolio(10_000), {}, proposed_notional=9_999)
    assert ok is True


def test_concentration_breach():
    from privateye.risk.exposure_monitor import ExposureMonitor
    em = ExposureMonitor(enabled=True, max_concentration_pct=0.40)
    # Proposed notional = 5000 on 10000 equity = 50% > 40% max
    ok, reason = em.check_signal(_signal(), _portfolio(10_000), {}, proposed_notional=5_000)
    assert ok is False
    assert "concentration" in reason.lower()


def test_concentration_passes_below_limit():
    from privateye.risk.exposure_monitor import ExposureMonitor
    em = ExposureMonitor(enabled=True, max_concentration_pct=0.40)
    # Proposed notional = 3500 / 10000 = 35% < 40% max
    ok, _ = em.check_signal(_signal(), _portfolio(10_000), {}, proposed_notional=3_500)
    assert ok is True


def test_total_notional_breach():
    from privateye.risk.exposure_monitor import ExposureMonitor
    # Existing position at 70% of equity + new 15% = 85% > 80% max
    pos = Position(symbol="BTC/USDT", side=Direction.LONG, quantity=0.14,
                   entry_price=50_000.0, stop_price=45_000.0, target_price=55_000.0,
                   strategy_id="test")
    portfolio = _portfolio(10_000, cash=3_000, positions={"BTC/USDT": pos})
    bars_by_sym = {"BTC/USDT": _bars(close=50_000.0)}
    em = ExposureMonitor(enabled=True, max_concentration_pct=1.0, max_total_notional_pct=0.80)
    # Existing BTC notional = 0.14 × 50000 = 7000 (70%). Adding 1500 (15%) → 85% > 80%
    ok, reason = em.check_signal(
        _signal(entry=50_000.0), portfolio, bars_by_sym, proposed_notional=1_500
    )
    assert ok is False
    assert "notional" in reason.lower()


def test_btc_beta_breach():
    from privateye.risk.exposure_monitor import ExposureMonitor
    # Portfolio already at 1.5 BTC-beta; adding ETH (beta 0.85) would push it over 2.0
    pos = Position(symbol="BTC/USDT", side=Direction.LONG, quantity=0.15,
                   entry_price=50_000.0, stop_price=45_000.0, target_price=55_000.0,
                   strategy_id="test")
    portfolio = _portfolio(10_000, cash=2_500, positions={"BTC/USDT": pos})
    bars_by_sym = {"BTC/USDT": _bars(close=50_000.0)}
    em = ExposureMonitor(
        enabled=True, max_concentration_pct=1.0, max_total_notional_pct=2.0,
        max_portfolio_beta=0.80,   # tight ceiling so BTC + ETH projected beta breaches it
        btc_beta_symbols={"ETH/USDT": 0.85},
    )
    eth_sig = _signal(symbol="ETH/USDT", entry=3_000.0, stop=2_700.0, target=3_300.0)
    ok, reason = em.check_signal(eth_sig, portfolio, bars_by_sym, proposed_notional=3_000)
    assert ok is False
    assert "beta" in reason.lower()


def test_single_asset_no_beta_symbols_passes():
    from privateye.risk.exposure_monitor import ExposureMonitor
    em = ExposureMonitor(enabled=True, max_concentration_pct=1.0,
                         max_total_notional_pct=1.0, max_portfolio_beta=2.0,
                         btc_beta_symbols={})  # no extra beta config
    ok, _ = em.check_signal(_signal(), _portfolio(10_000), {}, proposed_notional=3_000)
    assert ok is True


def test_compute_report_concentration():
    from privateye.risk.exposure_monitor import ExposureMonitor
    pos = Position(symbol="BTC/USDT", side=Direction.LONG, quantity=0.1,
                   entry_price=50_000.0, stop_price=45_000.0, target_price=55_000.0,
                   strategy_id="test")
    portfolio = _portfolio(10_000, cash=5_000, positions={"BTC/USDT": pos})
    # daily_vol_pct=0.0 → all closes exactly 50000; notional = 0.1×50000 = 5000 = 50%
    bars_by_sym = {"BTC/USDT": _bars(close=50_000.0, daily_vol_pct=0.0)}
    em = ExposureMonitor(enabled=True, max_concentration_pct=0.40)
    report = em.compute_report(portfolio, bars_by_sym)
    # 0.1 BTC × $50000 = $5000; concentration = 5000/10000 = 50%
    assert abs(report.concentration_by_symbol["BTC/USDT"] - 0.50) < 1e-6
    assert report.is_over_concentrated is True


def test_correlation_matrix_none_single_position():
    from privateye.risk.exposure_monitor import ExposureMonitor
    pos = Position(symbol="BTC/USDT", side=Direction.LONG, quantity=0.1,
                   entry_price=50_000.0, stop_price=45_000.0, target_price=55_000.0,
                   strategy_id="test")
    portfolio = _portfolio(10_000, positions={"BTC/USDT": pos})
    em = ExposureMonitor(enabled=True)
    report = em.compute_report(portfolio, {"BTC/USDT": _bars()})
    assert report.correlation_matrix is None


# ── SizingRouter ─────────────────────────────────────────────────────────────

def test_fixed_risk_matches_direct_call():
    from privateye.risk.sizing import fixed_risk_size
    from privateye.risk.sizing_router import SizingMethod, route_sizing
    sig = _signal(entry=50_000.0, stop=49_000.0)
    cfg = {"max_risk_per_trade_pct": 0.01}
    qty = route_sizing(SizingMethod.FIXED_RISK, 10_000.0, cfg, sig)
    expected = fixed_risk_size(10_000.0, 0.01, 50_000.0, 49_000.0)
    assert abs(qty - expected) < 1e-9


def test_kelly_uses_trade_history():
    from privateye.risk.sizing_router import SizingMethod, route_sizing
    sig = _signal(entry=50_000.0, stop=49_000.0)
    cfg = {"kelly": {"fraction": 0.5}}
    # 15 trades: 10 wins of +100, 5 losses of -50 → win_rate=0.667, avg_win=100, avg_loss=50
    trades = [_trade(100)] * 10 + [_trade(-50)] * 5
    qty = route_sizing(SizingMethod.KELLY, 10_000.0, cfg, sig, trade_history=trades)
    assert qty > 0


def test_kelly_falls_back_when_insufficient_history():
    from privateye.risk.sizing_router import SizingMethod, route_sizing
    sig = _signal()
    cfg = {"max_risk_per_trade_pct": 0.01, "kelly": {"fraction": 0.5}}
    # Only 5 trades (< 10 minimum) → fallback to fixed_risk
    trades = [_trade(50)] * 5
    qty = route_sizing(SizingMethod.KELLY, 10_000.0, cfg, sig, trade_history=trades)
    assert qty > 0   # fallback produces a non-zero result


def test_volatility_targeted_dispatch():
    from privateye.risk.sizing_router import SizingMethod, route_sizing
    sig = _signal(entry=50_000.0, stop=49_000.0)
    bars = _bars(n=100, daily_vol_pct=0.01)
    cfg = {"volatility_targeted": {"target_vol_pct": 0.01}, "max_risk_per_trade_pct": 0.01}
    qty = route_sizing(SizingMethod.VOLATILITY_TARGETED, 10_000.0, cfg, sig, bars=bars)
    assert qty > 0


def test_vol_targeted_fallback_no_bars():
    from privateye.risk.sizing_router import SizingMethod, route_sizing
    sig = _signal()
    cfg = {"max_risk_per_trade_pct": 0.01, "volatility_targeted": {"target_vol_pct": 0.01}}
    qty = route_sizing(SizingMethod.VOLATILITY_TARGETED, 10_000.0, cfg, sig, bars=None)
    assert qty > 0  # fallback to fixed_risk


# ── RiskManager full-gate integration ────────────────────────────────────────

def test_all_advanced_disabled_same_as_baseline():
    """Phase 5 components instantiated but enabled=False → identical to plain RiskManager."""
    from privateye.risk.manager import RiskManager
    from privateye.risk.asset_filter import AssetFilter
    from privateye.risk.exposure_monitor import ExposureMonitor
    cfg = {"max_risk_per_trade_pct": 0.01, "min_confidence": 0.5,
           "max_daily_drawdown_pct": 0.5, "max_position_notional_pct": 0.5}
    baseline = RiskManager(cfg)
    advanced = RiskManager(
        cfg,
        asset_filter=AssetFilter(enabled=False, banned_symbols=["BTC/USDT"]),
        exposure_monitor=ExposureMonitor(enabled=False, max_concentration_pct=0.001),
    )
    port = _portfolio()
    sig = _signal()
    r1 = baseline.evaluate_signal(sig, port)
    r2 = advanced.evaluate_signal(sig, port)
    assert r1[0] == r2[0]  # same approval outcome
    assert r1[1] == r2[1]  # same reason


def test_gate_ordering_asset_filter_before_exposure_monitor():
    """Banned symbol should be rejected by AssetFilter; ExposureMonitor never called."""
    from privateye.risk.manager import RiskManager
    from privateye.risk.asset_filter import AssetFilter
    from privateye.risk.exposure_monitor import ExposureMonitor

    mock_monitor = MagicMock(spec=ExposureMonitor)
    mock_monitor.enabled = True
    mock_monitor.check_signal.return_value = (True, "")

    cfg = {"max_risk_per_trade_pct": 0.01, "min_confidence": 0.0,
           "max_daily_drawdown_pct": 0.5, "max_position_notional_pct": 1.0}
    af = AssetFilter(enabled=True, banned_symbols=["BTC/USDT"])
    rm = RiskManager(cfg, asset_filter=af, exposure_monitor=mock_monitor)

    approved, reason, _ = rm.evaluate_signal(_signal(), _portfolio(), bars=_bars())
    assert approved is False
    assert "AssetFilter" in reason
    # ExposureMonitor should NOT have been called (asset was banned before sizing)
    mock_monitor.check_signal.assert_not_called()
