"""
Phase 10 — Strategy Parameter Optimizer
Tests for OptimizationConfig, CombinationResult, OptimizationReport,
GridSearchOptimizer, and the --mode optimize CLI integration.

Target: 349 total tests (319 existing + 30 new).
"""
from __future__ import annotations

import copy
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from privateye.backtesting.metrics import BacktestReport
from privateye.backtesting.optimizer import (
    CombinationResult,
    GridSearchOptimizer,
    OptimizationConfig,
    OptimizationReport,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_bars(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 30_000.0 + np.cumsum(rng.normal(0, 200, n))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
        "open":   closes,
        "high":   closes + 100.0,
        "low":    closes - 100.0,
        "close":  closes,
        "volume": rng.uniform(500, 5000, n),
    })


_MINI_CFG = {
    "backtesting": {
        "initial_capital": 10_000.0,
        "fee_maker": 0.001,
        "fee_taker": 0.001,
        "slippage_pct": 0.0005,
        "max_fill_pct_of_volume": 0.30,
        "data_dir": "data/historical",
    },
    "strategies": {
        "directional": {
            "enabled": True, "timeframe": "1h",
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "ema_trend": 200, "rsi_period": 14,
            "rsi_overbought": 70, "rsi_oversold": 30,
        },
        "mean_reversion": {
            "enabled": False, "timeframe": "1h",
            "bb_period": 20, "bb_std": 2.0, "rsi_period": 14,
            "rsi_overbought": 70, "rsi_oversold": 30,
        },
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
        "restricted_assets": [],
        "advanced": {
            "asset_filter": {"enabled": False},
            "exposure_monitor": {"enabled": False},
            "black_swan": {"enabled": False},
        },
    },
    "ml": {"enabled": False},
    "data": {"bar_window": 200},
}


def _make_report(
    sharpe: float = 1.0,
    total_pnl: float = 500.0,
    calmar: float = 0.8,
    total_trades: int = 10,
) -> BacktestReport:
    """Minimal BacktestReport for testing."""
    return BacktestReport(
        total_trades=total_trades,
        winning_trades=6,
        losing_trades=4,
        win_rate=0.6,
        total_pnl=total_pnl,
        total_pnl_pct=5.0,
        avg_win=100.0,
        avg_loss=-60.0,
        profit_factor=1.5,
        max_drawdown_pct=0.05,
        max_drawdown_abs=500.0,
        sharpe_ratio=sharpe,
        sortino_ratio=1.2,
        calmar_ratio=calmar,
        avg_bars_held=12.0,
        annualised_return_pct=0.20,
        initial_capital=10_000.0,
        final_equity=10_500.0,
        total_fees_paid=20.0,
        fee_drag_pct=0.002,
        slippage_cost_pct=0.001,
        limit_fill_rate=0.0,
        limit_orders_placed=0,
        limit_orders_filled=0,
        equity_curve=[10_000.0, 10_500.0],
        trades=[],
    )


def _make_opt_config(
    strategy_name: str = "directional",
    param_grid: dict | None = None,
    metric: str = "sharpe_ratio",
    min_trades: int = 5,
) -> OptimizationConfig:
    if param_grid is None:
        param_grid = {"macd_fast": [10, 12]}
    return OptimizationConfig(
        strategy_name=strategy_name,
        param_grid=param_grid,
        metric=metric,
        min_trades=min_trades,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Group 1: TestOptimizerCombinations  (6 tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizerCombinations:
    """Verify that GridSearchOptimizer.run() generates the correct cartesian product."""

    def _run_with_mock(self, param_grid: dict, mock_reports: list) -> OptimizationReport:
        """Run optimizer with mocked _run_combination."""
        opt_config = OptimizationConfig(
            strategy_name="directional",
            param_grid=param_grid,
            metric="sharpe_ratio",
            min_trades=1,
        )
        opt = GridSearchOptimizer(_MINI_CFG, opt_config)
        with patch.object(
            GridSearchOptimizer, "_run_combination",
            side_effect=mock_reports,
        ):
            return opt.run(_make_bars(100))

    def test_single_param_single_value_one_combination(self):
        report = self._run_with_mock({"macd_fast": [12]}, [_make_report()])
        assert report.n_combinations_total == 1

    def test_two_params_cartesian_product_count(self):
        reports = [_make_report() for _ in range(4)]
        report = self._run_with_mock({"a": [1, 2], "b": [3, 4]}, reports)
        assert report.n_combinations_total == 4

    def test_three_params_product_count(self):
        n = 2 * 3 * 2
        reports = [_make_report() for _ in range(n)]
        report = self._run_with_mock({"x": [1, 2], "y": [3, 4, 5], "z": [6, 7]}, reports)
        assert report.n_combinations_total == n

    def test_combination_dicts_contain_all_keys(self):
        reports = [_make_report() for _ in range(4)]
        report = self._run_with_mock({"a": [1, 2], "b": [3, 4]}, reports)
        for res in report.results:
            assert "a" in res.params
            assert "b" in res.params

    def test_empty_grid_produces_zero_combinations(self):
        opt_config = OptimizationConfig(
            strategy_name="directional",
            param_grid={},
            metric="sharpe_ratio",
            min_trades=1,
        )
        opt = GridSearchOptimizer(_MINI_CFG, opt_config)
        report = opt.run(_make_bars(100))
        assert report.n_combinations_total == 0
        assert report.results == []

    def test_run_combination_called_n_times(self):
        param_grid = {"macd_fast": [8, 10, 12]}
        reports = [_make_report(sharpe=float(i)) for i in range(3)]
        opt_config = OptimizationConfig(
            strategy_name="directional",
            param_grid=param_grid,
            metric="sharpe_ratio",
            min_trades=1,
        )
        opt = GridSearchOptimizer(_MINI_CFG, opt_config)
        with patch.object(
            GridSearchOptimizer, "_run_combination",
            side_effect=reports,
        ) as mock_rc:
            opt.run(_make_bars(100))
            assert mock_rc.call_count == 3


# ─────────────────────────────────────────────────────────────────────────────
# Group 2: TestOptimizationConfig  (5 tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizationConfig:
    def test_from_config_reads_strategy_name(self):
        cfg = {"strategy_name": "mean_reversion", "param_grid": {"mean_reversion": {}}}
        oc = OptimizationConfig.from_config(cfg)
        assert oc.strategy_name == "mean_reversion"

    def test_from_config_reads_metric(self):
        cfg = {"metric": "calmar_ratio", "param_grid": {}}
        oc = OptimizationConfig.from_config(cfg)
        assert oc.metric == "calmar_ratio"

    def test_from_config_min_trades_defaults_to_5(self):
        cfg = {"param_grid": {}}
        oc = OptimizationConfig.from_config(cfg)
        assert oc.min_trades == 5

    def test_from_config_reads_param_grid(self):
        cfg = {
            "strategy_name": "directional",
            "param_grid": {
                "directional": {"macd_fast": [8, 12], "rsi_period": [10, 14]},
            },
        }
        oc = OptimizationConfig.from_config(cfg)
        assert oc.param_grid == {"macd_fast": [8, 12], "rsi_period": [10, 14]}

    def test_from_config_reads_symbol_and_timeframe(self):
        cfg = {
            "symbol": "ETH/USDT",
            "timeframe": "4h",
            "param_grid": {},
        }
        oc = OptimizationConfig.from_config(cfg)
        assert oc.symbol == "ETH/USDT"
        assert oc.timeframe == "4h"


# ─────────────────────────────────────────────────────────────────────────────
# Group 3: TestOptimizationReport  (6 tests)
# ─────────────────────────────────────────────────────────────────────────────

def _make_opt_report(results: list[CombinationResult]) -> OptimizationReport:
    passed = [r for r in results if r.passed_min_trades]
    best_params = passed[0].params if passed else {}
    best_score  = passed[0].score  if passed else float("nan")
    return OptimizationReport(
        strategy_name="directional",
        metric="sharpe_ratio",
        min_trades=5,
        symbol="BTC/USDT",
        timeframe="1h",
        n_combinations_total=len(results),
        n_combinations_passed=len(passed),
        results=results,
        best_params=best_params,
        best_score=best_score,
    )


class TestOptimizationReport:
    def test_results_sorted_best_score_first(self):
        results = [
            CombinationResult({"p": 1}, _make_report(sharpe=0.5), 0.5, True),
            CombinationResult({"p": 2}, _make_report(sharpe=1.5), 1.5, True),
            CombinationResult({"p": 3}, _make_report(sharpe=1.0), 1.0, True),
        ]
        results.sort(key=lambda r: (not r.passed_min_trades, -r.score))
        report = _make_opt_report(results)
        assert report.results[0].score == 1.5
        assert report.results[1].score == 1.0
        assert report.results[2].score == 0.5

    def test_passed_results_before_failed_results(self):
        results = [
            CombinationResult({"p": 1}, _make_report(total_trades=0), 0.5, False),
            CombinationResult({"p": 2}, _make_report(sharpe=1.0), 1.0, True),
        ]
        results.sort(key=lambda r: (not r.passed_min_trades, -r.score))
        report = _make_opt_report(results)
        assert report.results[0].passed_min_trades is True
        assert report.results[1].passed_min_trades is False

    def test_best_params_equals_top_result_params(self):
        results = [
            CombinationResult({"p": 2}, _make_report(sharpe=1.5), 1.5, True),
            CombinationResult({"p": 1}, _make_report(sharpe=0.5), 0.5, True),
        ]
        results.sort(key=lambda r: (not r.passed_min_trades, -r.score))
        report = _make_opt_report(results)
        assert report.best_params == {"p": 2}

    def test_best_score_equals_top_result_score(self):
        results = [
            CombinationResult({"p": 2}, _make_report(sharpe=1.5), 1.5, True),
            CombinationResult({"p": 1}, _make_report(sharpe=0.5), 0.5, True),
        ]
        results.sort(key=lambda r: (not r.passed_min_trades, -r.score))
        report = _make_opt_report(results)
        assert report.best_score == 1.5

    def test_all_failed_min_trades_best_params_empty(self):
        results = [
            CombinationResult({"p": 1}, _make_report(total_trades=0), 0.5, False),
            CombinationResult({"p": 2}, _make_report(total_trades=2), 1.0, False),
        ]
        results.sort(key=lambda r: (not r.passed_min_trades, -r.score))
        report = _make_opt_report(results)
        assert report.best_params == {}
        import math
        assert math.isnan(report.best_score)

    def test_str_contains_expected_labels(self):
        results = [
            CombinationResult({"macd_fast": 10}, _make_report(sharpe=1.2), 1.2, True),
        ]
        report = _make_opt_report(results)
        text = str(report)
        assert "OPTIMIZATION REPORT" in text
        assert "directional" in text
        assert "sharpe_ratio" in text
        assert "BTC/USDT" in text


# ─────────────────────────────────────────────────────────────────────────────
# Group 4: TestGridSearchOptimizerRun  (9 tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestGridSearchOptimizerRun:
    """All tests mock _run_combination to avoid slow actual backtest runs."""

    def _make_opt(self, param_grid=None, metric="sharpe_ratio", min_trades=5):
        if param_grid is None:
            param_grid = {"macd_fast": [10, 12]}
        oc = OptimizationConfig(
            strategy_name="directional",
            param_grid=param_grid,
            metric=metric,
            min_trades=min_trades,
        )
        return GridSearchOptimizer(_MINI_CFG, oc)

    def test_run_returns_optimization_report(self):
        opt = self._make_opt()
        with patch.object(GridSearchOptimizer, "_run_combination",
                          side_effect=[_make_report(), _make_report()]):
            result = opt.run(_make_bars(100))
        assert isinstance(result, OptimizationReport)

    def test_run_calls_combination_for_each_product(self):
        param_grid = {"macd_fast": [8, 10], "rsi_period": [10, 14]}  # 4 combos
        opt = self._make_opt(param_grid=param_grid)
        reports = [_make_report(sharpe=float(i)) for i in range(4)]
        with patch.object(GridSearchOptimizer, "_run_combination",
                          side_effect=reports) as mock_rc:
            opt.run(_make_bars(100))
        assert mock_rc.call_count == 4

    def test_run_combination_receives_correct_params(self):
        param_grid = {"macd_fast": [10]}
        opt = self._make_opt(param_grid=param_grid)
        with patch.object(GridSearchOptimizer, "_run_combination",
                          side_effect=[_make_report()]) as mock_rc:
            opt.run(_make_bars(100))
        # First positional arg to _run_combination is params dict
        called_params = mock_rc.call_args[0][0]
        assert called_params == {"macd_fast": 10}

    def test_run_combination_receives_bars_dataframe(self):
        opt = self._make_opt(param_grid={"macd_fast": [10]})
        bars = _make_bars(100)
        with patch.object(GridSearchOptimizer, "_run_combination",
                          side_effect=[_make_report()]) as mock_rc:
            opt.run(bars)
        called_bars = mock_rc.call_args[0][1]
        assert isinstance(called_bars, pd.DataFrame)
        assert len(called_bars) == 100

    def test_n_combinations_total_set_correctly(self):
        param_grid = {"macd_fast": [8, 10, 12]}  # 3 combos
        opt = self._make_opt(param_grid=param_grid)
        reports = [_make_report() for _ in range(3)]
        with patch.object(GridSearchOptimizer, "_run_combination",
                          side_effect=reports):
            report = opt.run(_make_bars(100))
        assert report.n_combinations_total == 3

    def test_n_combinations_passed_counts_min_trades(self):
        param_grid = {"p": [1, 2, 3]}
        opt = self._make_opt(param_grid=param_grid, min_trades=5)
        # 2 reports with >=5 trades, 1 with 0 trades
        side_effects = [
            _make_report(total_trades=10),
            _make_report(total_trades=0),
            _make_report(total_trades=8),
        ]
        with patch.object(GridSearchOptimizer, "_run_combination",
                          side_effect=side_effects):
            report = opt.run(_make_bars(100))
        assert report.n_combinations_passed == 2

    def test_metric_sharpe_ratio_used_for_scoring(self):
        opt = self._make_opt(param_grid={"p": [1, 2]}, metric="sharpe_ratio")
        reports = [_make_report(sharpe=0.5), _make_report(sharpe=1.5)]
        with patch.object(GridSearchOptimizer, "_run_combination",
                          side_effect=reports):
            report = opt.run(_make_bars(100))
        scores = {r.params["p"]: r.score for r in report.results}
        assert scores[1] == pytest.approx(0.5)
        assert scores[2] == pytest.approx(1.5)

    def test_metric_total_pnl_used_for_scoring(self):
        opt = self._make_opt(param_grid={"p": [1, 2]}, metric="total_pnl")
        reports = [_make_report(total_pnl=200.0), _make_report(total_pnl=800.0)]
        with patch.object(GridSearchOptimizer, "_run_combination",
                          side_effect=reports):
            report = opt.run(_make_bars(100))
        scores = {r.params["p"]: r.score for r in report.results}
        assert scores[1] == pytest.approx(200.0)
        assert scores[2] == pytest.approx(800.0)

    def test_metric_calmar_ratio_used_for_scoring(self):
        opt = self._make_opt(param_grid={"p": [1, 2]}, metric="calmar_ratio")
        reports = [_make_report(calmar=0.3), _make_report(calmar=1.1)]
        with patch.object(GridSearchOptimizer, "_run_combination",
                          side_effect=reports):
            report = opt.run(_make_bars(100))
        scores = {r.params["p"]: r.score for r in report.results}
        assert scores[1] == pytest.approx(0.3)
        assert scores[2] == pytest.approx(1.1)


# ─────────────────────────────────────────────────────────────────────────────
# Group 5: TestCLIAndMutation  (4 tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIAndMutation:
    def test_run_combination_does_not_mutate_original_cfg(self):
        """Deep-copy must protect the master cfg from combination overrides."""
        cfg = copy.deepcopy(_MINI_CFG)
        original_macd_fast = cfg["strategies"]["directional"]["macd_fast"]

        opt_config = OptimizationConfig(
            strategy_name="directional",
            param_grid={"macd_fast": [999]},
            metric="sharpe_ratio",
            min_trades=1,
        )
        opt = GridSearchOptimizer(cfg, opt_config)
        with patch.object(GridSearchOptimizer, "_run_combination",
                          return_value=_make_report()) as mock_rc:
            opt.run(_make_bars(50))
            # The actual cfg passed into _run_combination must not be the original
            called_params = mock_rc.call_args[0][0]
            assert called_params["macd_fast"] == 999

        # Original cfg unchanged
        assert cfg["strategies"]["directional"]["macd_fast"] == original_macd_fast

    def test_run_combination_overrides_strategy_param_in_copy(self):
        """_run_combination must update the copied cfg's strategy block."""
        cfg = copy.deepcopy(_MINI_CFG)
        opt_config = OptimizationConfig(
            strategy_name="directional",
            param_grid={"macd_fast": [42]},
            metric="sharpe_ratio",
            min_trades=1,
        )
        opt = GridSearchOptimizer(cfg, opt_config)
        captured_cfg = {}

        def mock_build_strategies(c):
            # Capture the cfg that arrives at _build_strategies
            captured_cfg.update(c)
            return []

        with patch("privateye.backtesting.optimizer.GridSearchOptimizer._run_combination",
                   wraps=opt._run_combination):
            with patch("privateye.main._build_strategies",
                       side_effect=mock_build_strategies):
                with patch("privateye.main._build_advanced_risk",
                           return_value=(None, None, None)):
                    with patch("privateye.risk.manager.RiskManager"):
                        with patch("privateye.backtesting.simulator.SimulatedExchange"):
                            with patch("privateye.backtesting.engine.BacktestEngine") as MockEngine:
                                mock_engine_instance = MockEngine.return_value
                                mock_engine_instance.run.return_value = _make_report()
                                opt.run(_make_bars(50))

        if captured_cfg:
            assert captured_cfg["strategies"]["directional"]["macd_fast"] == 42

    def test_optimize_in_cli_choices(self):
        """'optimize' must be a valid --mode choice."""
        import argparse
        from privateye.main import cli
        import sys
        # Temporarily replace sys.argv and catch SystemExit from --help
        old_argv = sys.argv
        sys.argv = ["main", "--mode", "optimize", "--help"]
        try:
            cli()
        except SystemExit:
            pass  # --help always exits; we just want no "invalid choice" error
        finally:
            sys.argv = old_argv

    def test_run_optimize_calls_optimizer_run(self):
        """run_optimize must call GridSearchOptimizer.run() exactly once."""
        from privateye.main import run_optimize
        cfg = copy.deepcopy(_MINI_CFG)
        cfg["optimization"] = {
            "strategy_name": "directional",
            "metric": "sharpe_ratio",
            "min_trades": 1,
            "param_grid": {"directional": {"macd_fast": [10]}},
        }
        cfg["symbols"] = ["BTC/USDT"]
        cfg["primary_timeframe"] = "1h"

        mock_report = OptimizationReport(
            strategy_name="directional",
            metric="sharpe_ratio",
            min_trades=1,
            symbol="BTC/USDT",
            timeframe="1h",
            n_combinations_total=1,
            n_combinations_passed=1,
            results=[],
            best_params={"macd_fast": 10},
            best_score=1.5,
        )

        with patch("privateye.data.providers.csv_provider.CSVProvider") as MockCSV:
            mock_provider = MockCSV.return_value
            mock_provider.load.return_value = _make_bars(200)
            with patch("privateye.backtesting.optimizer.GridSearchOptimizer.run",
                       return_value=mock_report) as mock_run:
                result = run_optimize(cfg)

        mock_run.assert_called_once()
        assert result is mock_report
