"""Phase 5 — Multi-Asset Portfolio, MTF Fusion & Portfolio Circuit Breakers.

30 tests covering:
  TestPortfolioBacktestEngine      (10) — PortfolioBacktestReport structure, per-symbol
    reports, equity curve length, aggregate metrics, single-symbol run, CLI registration
  TestPortfolioOptimizer            (7) — EqualWeightOptimizer, RiskParityOptimizer,
    build_portfolio_optimizer factory
  TestPortfolioCircuitBreaker       (6) — Gate 12 trigger, cross-symbol block, reset, HWM
  TestRiskManagerOptimizerIntegration (3) — optimizer cap vs config cap
  TestMTFConfirmation               (4) — disabled, agrees, disagrees, unavailable (fail-open)
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _minimal_cfg() -> dict:
    """Minimal config understood by PortfolioBacktestEngine and RiskManager."""
    return {
        "backtesting": {
            "initial_capital": 10000.0,
            "fee_maker": 0.001,
            "fee_taker": 0.001,
            "slippage_pct": 0.0005,
            "max_fill_pct_of_volume": 0.30,
        },
        "strategies": {
            "directional": {
                "enabled": True,
                "timeframe": "1h",
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "ema_trend": 200,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
            },
            "mean_reversion": {"enabled": False},
        },
        "risk": {
            "max_risk_per_trade_pct": 0.01,
            "max_daily_drawdown_pct": 0.05,
            "max_position_notional_pct": 0.20,
            "atr_period": 14,
            "atr_stop_multiplier": 2.0,
            "max_bars_in_trade": 48,
            "min_confidence": 0.55,
            "sizing_method": "fixed_risk",
            "portfolio_dd_halt_pct": 0.0,
            "advanced": {
                "asset_filter":     {"enabled": False},
                "exposure_monitor": {"enabled": False},
                "black_swan":       {"enabled": False},
            },
        },
        "ml":   {"enabled": False},
        "data": {"bar_window": 500},
    }


def _synthetic_bars(
    n: int = 120,
    seed: int = 42,
    start: str = "2024-01-01",
    freq: str = "1h",
    base_price: float = 30000.0,
):
    """Return a minimal OHLCV DataFrame with n rows."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    closes = base_price + np.cumsum(rng.normal(0, 200, n))
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=n, freq=freq, tz="UTC"),
        "open":   closes,
        "high":   closes + 100,
        "low":    closes - 100,
        "close":  closes,
        "volume": rng.uniform(500, 5000, n),
    })


def _make_portfolio(equity: float = 10000.0, cash: float = 10000.0):
    from privateye.core.types import PortfolioState
    return PortfolioState(equity=equity, cash=cash, peak_equity=equity)


def _make_signal(
    symbol: str = "BTC/USDT",
    direction=None,
    confidence: float = 0.7,
    entry: float = 30000.0,
    stop: float = 29000.0,
):
    from privateye.core.types import Direction, TradingSignal
    if direction is None:
        direction = Direction.LONG
    return TradingSignal(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        entry_price=entry,
        stop_price=stop,
        target_price=entry + (entry - stop) * 2.0,
        strategy_id="test",
        timeframe="1h",
    )


# ===========================================================================
# TestPortfolioBacktestEngine
# ===========================================================================

class TestPortfolioBacktestEngine:
    """PortfolioBacktestEngine — 10 tests."""

    def _run_small(self, n: int = 10) -> Any:
        """Run the engine with two synthetic symbols, n bars each (below warmup)."""
        from privateye.backtesting.portfolio_engine import PortfolioBacktestEngine
        engine = PortfolioBacktestEngine(_minimal_cfg())
        bars = {
            "BTC/USDT": _synthetic_bars(n, seed=1),
            "ETH/USDT": _synthetic_bars(n, seed=2, base_price=3000.0),
        }
        return engine.run(bars, "1h")

    # 1
    def test_returns_portfolio_backtest_report(self):
        """run() must return a PortfolioBacktestReport instance."""
        from privateye.backtesting.portfolio_engine import PortfolioBacktestReport
        report = self._run_small()
        assert isinstance(report, PortfolioBacktestReport)

    # 2
    def test_per_symbol_reports_keyed_by_symbol(self):
        """per_symbol_reports dict keys must match the input symbol set."""
        report = self._run_small()
        assert set(report.per_symbol_reports.keys()) == {"BTC/USDT", "ETH/USDT"}

    # 3
    def test_empty_bars_returns_empty_report(self):
        """Passing an empty bars_by_symbol dict must return a valid empty report."""
        from privateye.backtesting.portfolio_engine import (
            PortfolioBacktestEngine,
            PortfolioBacktestReport,
        )
        engine = PortfolioBacktestEngine(_minimal_cfg())
        report = engine.run({}, "1h")
        assert isinstance(report, PortfolioBacktestReport)
        assert report.symbols == []
        assert report.total_trades == 0
        assert report.aggregate_pnl == pytest.approx(0.0)

    # 4
    def test_total_trades_sums_across_symbols(self):
        """total_trades == sum of per-symbol total_trades."""
        report = self._run_small()
        expected = sum(r.total_trades for r in report.per_symbol_reports.values())
        assert report.total_trades == expected

    # 5
    def test_aggregate_equity_curve_length_equals_unique_timestamps(self):
        """One equity point is appended per unique timestamp, not per bar-symbol."""
        from privateye.backtesting.portfolio_engine import PortfolioBacktestEngine
        n = 15
        # Both symbols share identical timestamps → 15 unique timestamps
        bars = {
            "BTC/USDT": _synthetic_bars(n, seed=1),
            "ETH/USDT": _synthetic_bars(n, seed=2, base_price=3000.0),
        }
        engine = PortfolioBacktestEngine(_minimal_cfg())
        report = engine.run(bars, "1h")
        assert len(report.aggregate_equity_curve) == n

    # 6
    def test_aggregate_pnl_sums_per_symbol_pnl(self):
        """aggregate_pnl == sum of per-symbol total_pnl values."""
        report = self._run_small()
        expected = sum(r.total_pnl for r in report.per_symbol_reports.values())
        assert report.aggregate_pnl == pytest.approx(expected, abs=1e-6)

    # 7
    def test_symbols_list_matches_input(self):
        """report.symbols must contain all symbols from the input dict."""
        report = self._run_small()
        assert sorted(report.symbols) == sorted(["BTC/USDT", "ETH/USDT"])

    # 8
    def test_str_contains_portfolio_and_symbol_names(self):
        """__str__ output must mention 'PORTFOLIO' and each symbol."""
        report = self._run_small()
        s = str(report)
        assert "PORTFOLIO" in s
        assert "BTC/USDT" in s
        assert "ETH/USDT" in s

    # 9
    def test_single_symbol_run_produces_valid_report(self):
        """A single-symbol run must return a report with exactly one entry."""
        from privateye.backtesting.portfolio_engine import PortfolioBacktestEngine
        engine = PortfolioBacktestEngine(_minimal_cfg())
        bars = {"BTC/USDT": _synthetic_bars(10, seed=1)}
        report = engine.run(bars, "1h")
        assert len(report.symbols) == 1
        assert "BTC/USDT" in report.per_symbol_reports

    # 10
    def test_portfolio_backtest_in_allowed_modes_and_main(self):
        """'portfolio_backtest' must appear in config loader allowed modes and main.py."""
        from privateye.config.loader import _ALLOWED_MODES
        from privateye.main import PHASE5_COMPLETE
        assert "portfolio_backtest" in _ALLOWED_MODES
        assert isinstance(PHASE5_COMPLETE, bool)


# ===========================================================================
# TestPortfolioOptimizer
# ===========================================================================

class TestPortfolioOptimizer:
    """EqualWeightOptimizer, RiskParityOptimizer, and factory — 7 tests."""

    # 1
    def test_equal_weight_splits_evenly(self):
        """3-symbol run: each symbol gets exactly 1/3."""
        from privateye.risk.portfolio_optimizer import EqualWeightOptimizer
        opt = EqualWeightOptimizer()
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        bars = {s: _synthetic_bars(50, seed=i) for i, s in enumerate(symbols)}
        w = opt.compute_weights(symbols, bars)
        for s in symbols:
            assert w[s] == pytest.approx(1.0 / 3, abs=1e-9)

    # 2
    def test_equal_weights_sum_to_one(self):
        """Weights must sum to exactly 1.0 for any number of symbols."""
        from privateye.risk.portfolio_optimizer import EqualWeightOptimizer
        opt = EqualWeightOptimizer()
        symbols = ["A", "B", "C", "D", "E"]
        w = opt.compute_weights(symbols, {s: _synthetic_bars(30) for s in symbols})
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)

    # 3
    def test_risk_parity_high_vol_gets_smaller_weight(self):
        """Lower-volatility asset (ETH) must receive higher weight than BTC.

        The optimizer uses log-returns, so what matters is σ/price (percentage
        volatility).  Both assets start at the same base price (1000) so that
        absolute σ values translate directly to different log-return stds.
        BTC σ=200 → ~20% per-step return vol;  ETH σ=50 → ~5% return vol.
        ETH is 4× less volatile → ETH should receive a larger weight.
        """
        import pandas as pd
        from privateye.risk.portfolio_optimizer import RiskParityOptimizer

        rng = np.random.default_rng(42)
        n = 100
        # Both start at price 1000 so σ/price gives clearly different return vols.
        # BTC: σ=200  → ~20% per-step return vol  (high)
        # ETH: σ=50   →  ~5% per-step return vol  (low, 4× less volatile)
        btc_closes = 1000.0 + np.cumsum(rng.normal(0, 200, n))
        eth_closes = 1000.0 + np.cumsum(rng.normal(0,  50, n))
        dates = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")

        btc_bars = pd.DataFrame({"timestamp": dates, "close": btc_closes,
                                  "open": btc_closes, "high": btc_closes + 100,
                                  "low": btc_closes - 100, "volume": 1000.0})
        eth_bars = pd.DataFrame({"timestamp": dates, "close": eth_closes,
                                  "open": eth_closes, "high": eth_closes + 10,
                                  "low": eth_closes - 10, "volume": 1000.0})

        opt = RiskParityOptimizer(min_bars=20)
        symbols = ["BTC/USDT", "ETH/USDT"]
        w = opt.compute_weights(symbols, {"BTC/USDT": btc_bars, "ETH/USDT": eth_bars})
        assert w["ETH/USDT"] > w["BTC/USDT"]

    # 4
    def test_risk_parity_falls_back_to_equal_on_insufficient_bars(self):
        """When a symbol has fewer bars than min_bars, return equal weights."""
        from privateye.risk.portfolio_optimizer import RiskParityOptimizer
        opt = RiskParityOptimizer(min_bars=30)
        symbols = ["BTC/USDT", "ETH/USDT"]
        bars = {
            "BTC/USDT": _synthetic_bars(100),
            # Only 5 bars for ETH — well below min_bars=30
            "ETH/USDT": _synthetic_bars(5, base_price=3000.0),
        }
        w = opt.compute_weights(symbols, bars)
        assert w["BTC/USDT"] == pytest.approx(0.5, abs=1e-9)
        assert w["ETH/USDT"] == pytest.approx(0.5, abs=1e-9)

    # 5
    def test_build_portfolio_optimizer_returns_equal_weight(self):
        """Factory returns EqualWeightOptimizer for 'equal_weight'."""
        from privateye.risk.portfolio_optimizer import (
            EqualWeightOptimizer,
            build_portfolio_optimizer,
        )
        cfg = {"portfolio_backtest": {"optimizer": "equal_weight"}}
        opt = build_portfolio_optimizer(cfg)
        assert isinstance(opt, EqualWeightOptimizer)

    # 6
    def test_build_portfolio_optimizer_returns_risk_parity(self):
        """Factory returns RiskParityOptimizer for 'risk_parity'."""
        from privateye.risk.portfolio_optimizer import (
            RiskParityOptimizer,
            build_portfolio_optimizer,
        )
        cfg = {
            "portfolio_backtest": {
                "optimizer": "risk_parity",
                "min_bars_for_optimizer": 30,
                "use_correlation": False,
            }
        }
        opt = build_portfolio_optimizer(cfg)
        assert isinstance(opt, RiskParityOptimizer)

    # 7
    def test_single_symbol_returns_weight_of_one(self):
        """With a single symbol, RiskParityOptimizer must return weight = 1.0."""
        from privateye.risk.portfolio_optimizer import RiskParityOptimizer
        opt = RiskParityOptimizer(min_bars=20)
        w = opt.compute_weights(["BTC/USDT"], {"BTC/USDT": _synthetic_bars(60)})
        assert w["BTC/USDT"] == pytest.approx(1.0, abs=1e-9)


# ===========================================================================
# TestPortfolioCircuitBreaker
# ===========================================================================

class TestPortfolioCircuitBreaker:
    """RiskManager Gate 12 — portfolio-level circuit breaker — 6 tests."""

    def _make_rm(self, halt_pct: float = 0.10):
        from privateye.risk.manager import RiskManager
        return RiskManager({
            "max_risk_per_trade_pct": 0.01,
            "max_daily_drawdown_pct": 0.99,   # disable per-symbol DD gate
            "max_position_notional_pct": 0.20,
            "min_confidence": 0.0,            # accept any confidence
            "sizing_method": "fixed_risk",
            "atr_period": 14,
            "atr_stop_multiplier": 2.0,
            "max_bars_in_trade": 48,
            "portfolio_dd_halt_pct": halt_pct,
        })

    # 1
    def test_circuit_disabled_when_zero(self):
        """portfolio_dd_halt_pct=0.0 → Gate 12 branch skipped entirely."""
        rm = self._make_rm(halt_pct=0.0)
        # Equity at 50% of any plausible HWM — far beyond any threshold
        portfolio = _make_portfolio(equity=5000.0, cash=5000.0)
        signal = _make_signal()
        approved, reason, _ = rm.evaluate_signal(signal, portfolio)
        # Gate 12 must not be the rejection reason
        assert "Portfolio circuit breaker" not in reason

    # 2
    def test_circuit_fires_at_threshold(self):
        """Gate 12 fires when equity drop exactly reaches portfolio_dd_halt_pct."""
        rm = self._make_rm(halt_pct=0.10)
        rm._portfolio_daily_hwm = 10000.0          # manually set past HWM
        portfolio = _make_portfolio(equity=9000.0, cash=9000.0)   # 10% drop
        signal = _make_signal()
        approved, reason, order = rm.evaluate_signal(signal, portfolio)
        assert approved is False
        assert "Portfolio circuit breaker" in reason

    # 3
    def test_circuit_blocks_all_symbols_once_open(self):
        """After circuit opens on BTC drop, ETH signals are also blocked."""
        rm = self._make_rm(halt_pct=0.10)
        rm._portfolio_daily_hwm = 10000.0
        portfolio = _make_portfolio(equity=8500.0, cash=8500.0)   # 15% drop

        # First call opens the circuit
        btc_sig = _make_signal(symbol="BTC/USDT")
        rm.evaluate_signal(btc_sig, portfolio)
        assert rm._portfolio_circuit_open is True

        # Second call with a different symbol must also be blocked
        eth_sig = _make_signal(symbol="ETH/USDT", entry=3000.0, stop=2700.0)
        approved, reason, _ = rm.evaluate_signal(eth_sig, portfolio)
        assert approved is False
        assert "Portfolio circuit breaker" in reason

    # 4
    def test_reset_portfolio_daily_hwm_clears_circuit(self):
        """reset_portfolio_daily_hwm() must clear both HWM and circuit flag."""
        rm = self._make_rm(halt_pct=0.10)
        rm._portfolio_daily_hwm = 10000.0
        rm._portfolio_circuit_open = True          # force open

        rm.reset_portfolio_daily_hwm()

        assert rm._portfolio_circuit_open is False
        assert rm._portfolio_daily_hwm == float("-inf")

    # 5
    def test_hwm_updates_when_equity_rises_above_previous_hwm(self):
        """HWM must be updated to the new peak equity value."""
        rm = self._make_rm(halt_pct=0.10)
        rm._portfolio_daily_hwm = 9000.0           # old HWM
        portfolio = _make_portfolio(equity=11000.0, cash=11000.0)  # new high
        signal = _make_signal()
        rm.evaluate_signal(signal, portfolio)
        # Gate 12 must have updated HWM to 11000
        assert rm._portfolio_daily_hwm == pytest.approx(11000.0)

    # 6
    def test_portfolio_dd_halt_pct_defaults_to_zero(self):
        """RiskManager with no portfolio_dd_halt_pct in config defaults to 0.0."""
        from privateye.risk.manager import RiskManager
        rm = RiskManager({
            "max_risk_per_trade_pct": 0.01,
            "max_daily_drawdown_pct": 0.05,
        })
        assert rm._portfolio_dd_halt_pct == pytest.approx(0.0)


# ===========================================================================
# TestRiskManagerOptimizerIntegration
# ===========================================================================

class TestRiskManagerOptimizerIntegration:
    """RiskManager + PortfolioOptimizer cap integration — 3 tests."""

    def _make_rm(self, optimizer_weight: float, config_max: float = 0.20):
        """RiskManager with a mock optimizer returning `optimizer_weight` for BTC."""
        from privateye.risk.manager import RiskManager
        mock_opt = MagicMock()
        mock_opt.compute_weights.return_value = {
            "BTC/USDT": optimizer_weight,
            "ETH/USDT": 1.0 - optimizer_weight,
        }
        return RiskManager(
            {
                "max_risk_per_trade_pct": 0.01,
                "max_daily_drawdown_pct": 0.99,
                "max_position_notional_pct": config_max,
                "min_confidence": 0.0,
                "sizing_method": "fixed_risk",
                "atr_period": 14,
                "atr_stop_multiplier": 2.0,
                "max_bars_in_trade": 48,
                "portfolio_dd_halt_pct": 0.0,
            },
            portfolio_optimizer=mock_opt,
        )

    # 1
    def test_optimizer_weight_lower_than_config_wins(self):
        """Optimizer weight 0.10 < config 0.20 → order bounded by optimizer cap."""
        rm = self._make_rm(optimizer_weight=0.10, config_max=0.20)
        portfolio = _make_portfolio(equity=10000.0, cash=10000.0)
        signal = _make_signal(entry=30000.0, stop=29000.0)
        bars_by = {"BTC/USDT": _synthetic_bars(10), "ETH/USDT": _synthetic_bars(10, seed=2)}

        approved, reason, order = rm.evaluate_signal(
            signal, portfolio, bars_by_symbol=bars_by
        )
        assert approved, f"Expected approval; got: {reason}"
        assert order is not None
        # Max notional = min(0.10, 0.20) * 10000 = 1000
        max_notional = 0.10 * 10000.0
        assert order.quantity * signal.entry_price <= max_notional + 1e-4

    # 2
    def test_no_optimizer_uses_config_cap(self):
        """Without a portfolio_optimizer, RiskManager uses config max_position_notional_pct."""
        from privateye.risk.manager import RiskManager
        rm = RiskManager({
            "max_risk_per_trade_pct": 0.01,
            "max_daily_drawdown_pct": 0.99,
            "max_position_notional_pct": 0.15,
            "min_confidence": 0.0,
            "sizing_method": "fixed_risk",
            "portfolio_dd_halt_pct": 0.0,
        })
        assert rm._portfolio_optimizer is None
        assert rm.max_notional_pct == pytest.approx(0.15)

        portfolio = _make_portfolio(equity=10000.0, cash=10000.0)
        signal = _make_signal(entry=30000.0, stop=29000.0)
        approved, reason, order = rm.evaluate_signal(signal, portfolio)
        assert approved, f"Expected approval; got: {reason}"
        assert order is not None
        max_notional = 0.15 * 10000.0
        assert order.quantity * signal.entry_price <= max_notional + 1e-4

    # 3
    def test_config_wins_when_optimizer_weight_is_higher(self):
        """Optimizer weight 0.30 > config 0.20 → order bounded by config cap (0.20)."""
        rm = self._make_rm(optimizer_weight=0.30, config_max=0.20)
        portfolio = _make_portfolio(equity=10000.0, cash=10000.0)
        signal = _make_signal(entry=30000.0, stop=29000.0)
        bars_by = {"BTC/USDT": _synthetic_bars(10), "ETH/USDT": _synthetic_bars(10, seed=2)}

        approved, reason, order = rm.evaluate_signal(
            signal, portfolio, bars_by_symbol=bars_by
        )
        assert approved, f"Expected approval; got: {reason}"
        assert order is not None
        # effective_max_pct = min(0.20, 0.30) = 0.20 → config wins
        max_notional = 0.20 * 10000.0
        assert order.quantity * signal.entry_price <= max_notional + 1e-4


# ===========================================================================
# TestMTFConfirmation
# ===========================================================================

class TestMTFConfirmation:
    """FusionStrategy MTF gate tested directly via _mtf_suppresses — 4 tests."""

    def _make_fusion(self, mtf_enabled: bool = False) -> Any:
        from privateye.strategies.fusion_strategy import FusionStrategy
        cfg = {
            "enabled": True,
            "timeframe": "1h",
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "ema_trend": 200,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "gbm_gate": False,
            "regime_gate": False,
            "gbm_gate_threshold": 0.45,
            "sharpe_window_trades": 30,
            "atr_stop_multiplier": 2.0,
            "max_bars_in_trade": 48,
            "mtf_confirmation": {
                "enabled": mtf_enabled,
                "higher_timeframe": "4h",
            },
            "artifacts_dir": str(Path(tempfile.mkdtemp())),
        }
        return FusionStrategy(cfg)

    def _htf_bars(self, n: int = 60, uptrend: bool = True):
        """Higher-timeframe bars with a clear price trend relative to EMA(20)."""
        import pandas as pd
        rng = np.random.default_rng(99)
        # Strong monotonic trend so price[-1] is far from EMA[-1]
        if uptrend:
            closes = 30000.0 + np.arange(n) * 200.0 + rng.normal(0, 5, n)
        else:
            closes = 45000.0 - np.arange(n) * 200.0 + rng.normal(0, 5, n)
        return pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC"),
            "close":  closes,
            "open":   closes,
            "high":   closes + 50,
            "low":    closes - 50,
            "volume": rng.uniform(100, 500, n),
        })

    # 1
    def test_mtf_disabled_no_suppression(self):
        """With MTF gate disabled, _mtf_suppresses always returns False."""
        strat = self._make_fusion(mtf_enabled=False)
        # Load downtrend bars — even if gate were active, this would suppress
        strat.update_higher_tf_bars("BTC/USDT", self._htf_bars(uptrend=False))
        assert strat._mtf_suppresses("BTC/USDT", "long") is False

    # 2
    def test_mtf_enabled_agrees_no_suppression(self):
        """HTF EMA(20) direction matches lower-TF → signal is NOT suppressed."""
        strat = self._make_fusion(mtf_enabled=True)
        # Uptrend: price[-1] >> EMA(20)[-1] → htf_direction = "long"
        strat.update_higher_tf_bars("BTC/USDT", self._htf_bars(uptrend=True))
        # Lower-TF also says "long" → agree → not suppressed
        assert strat._mtf_suppresses("BTC/USDT", "long") is False

    # 3
    def test_mtf_enabled_disagrees_suppresses(self):
        """HTF EMA(20) direction contradicts lower-TF → signal IS suppressed."""
        strat = self._make_fusion(mtf_enabled=True)
        # Downtrend: price[-1] << EMA(20)[-1] → htf_direction = "flat"
        strat.update_higher_tf_bars("BTC/USDT", self._htf_bars(uptrend=False))
        # Lower-TF says "long" but HTF says "flat" → suppress
        assert strat._mtf_suppresses("BTC/USDT", "long") is True

    # 4
    def test_mtf_unavailable_fail_open(self):
        """No HTF bars set → _mtf_suppresses returns False (fail-open contract)."""
        strat = self._make_fusion(mtf_enabled=True)
        # Never called update_higher_tf_bars → _higher_tf_bars is empty
        assert strat._mtf_suppresses("BTC/USDT", "long") is False
