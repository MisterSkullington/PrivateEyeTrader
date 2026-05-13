"""
Phase 0 — Foundation & Rigorous Baseline

30 tests across:
  TestDataQuality      (8)  — duplicate/gap/zero-vol/price-jump detection, clean pass,
                               checksum consistency, manifest keys
  TestParquetProvider  (6)  — save+load roundtrip, sorted output, missing file,
                               limit param, available(), column normalisation
  TestGoldenSuite      (8)  — seed_all numpy/torch, run top-level keys,
                               sub-dict fields (sharpe_ratio, mean_oos_sharpe,
                               cv_mean_score, ruin_probability), JSON written
  TestMLflowTracker    (4)  — disabled noop, graceful import failure,
                               log_metrics numeric-only, log_params when supplied
  TestOptunaOptimizer  (4)  — missing optuna ImportError, n_trials forwarded,
                               best_score = max across trials, maximize direction

Target: 535 total tests (505 existing + 30 new).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from privateye.data.quality import generate_manifest, validate_bars


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_bars(n: int = 60, freq: str = "1h", seed: int = 42) -> pd.DataFrame:
    """Return a clean, gap-free OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    closes = 30_000.0 + np.cumsum(rng.normal(0, 100, n))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC"),
            "open":   closes,
            "high":   closes + 50.0,
            "low":    closes - 50.0,
            "close":  closes,
            "volume": rng.uniform(200, 2_000, n),
        }
    )


_MINI_CFG: dict = {
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
        "max_position_notional_pct": 0.2,
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
    "data": {"bar_window": 500},
    "walk_forward": {
        "train_bars": 40,
        "test_bars": 15,
        "step_bars": 15,
        "min_train_bars": 30,
        "min_test_trades": 0,
    },
    "kfold": {
        "n_folds": 2,
        "purge_bars": 5,
        "embargo_bars": 2,
        "min_train_bars": 30,
        "min_test_trades": 0,
        "metric": "sharpe_ratio",
    },
    "phase0": {
        "golden_seed": 42,
        "golden_date_from": "2020-01-01",
        "golden_date_to": "2030-01-01",
        "golden_output": "artifacts/golden_results_v1.0.json",
        "mlflow": {"enabled": False},
        "optuna": {"enabled": False, "n_trials": 10},
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# TestDataQuality  (8 tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestDataQuality:
    """validate_bars() and generate_manifest() in privateye.data.quality."""

    def test_duplicate_timestamps_detected(self) -> None:
        """Duplicate timestamps set is_clean=False and increment n_duplicates."""
        df = _make_bars(10)
        df_dup = pd.concat([df, df.iloc[[2]]], ignore_index=True)
        report = validate_bars(df_dup, "BTC/USDT", "1h")
        assert report.n_duplicates >= 1
        assert not report.is_clean

    def test_gap_exceeding_max_bars_flagged(self) -> None:
        """A timestamp gap > max_gap_bars must increment n_gaps."""
        df = _make_bars(30)
        # Drop rows 5–14 → 10-bar gap in a 1h series
        df_gapped = pd.concat([df.iloc[:5], df.iloc[15:]], ignore_index=True)
        report = validate_bars(df_gapped, "BTC/USDT", "1h", max_gap_bars=3)
        assert report.n_gaps >= 1

    def test_small_gap_within_threshold_does_not_flag(self) -> None:
        """A gap of ≤ max_gap_bars missing bars must NOT increment n_gaps."""
        df = _make_bars(20)
        # Drop 2 rows → 2-bar gap, below the default threshold of 3
        df_gapped = pd.concat([df.iloc[:5], df.iloc[7:]], ignore_index=True)
        report = validate_bars(df_gapped, "BTC/USDT", "1h", max_gap_bars=3)
        assert report.n_gaps == 0
        assert report.is_clean  # no duplicates, no negative values

    def test_zero_volume_flagged(self) -> None:
        """Zero-volume bars must increment n_zero_volume."""
        df = _make_bars(10).copy()
        df.loc[4, "volume"] = 0.0
        report = validate_bars(df, "BTC/USDT", "1h")
        assert report.n_zero_volume >= 1

    def test_price_jump_above_threshold_flagged(self) -> None:
        """A single-bar close change > max_price_jump_pct must increment n_price_jumps."""
        df = _make_bars(10).copy()
        df.loc[6, "close"] = df.loc[5, "close"] * 1.30  # +30% jump
        report = validate_bars(df, "BTC/USDT", "1h", max_price_jump_pct=0.15)
        assert report.n_price_jumps >= 1

    def test_clean_data_passes_all_checks(self) -> None:
        """A perfectly-formed DataFrame must produce is_clean=True with zero issues."""
        df = _make_bars(50)
        report = validate_bars(df, "BTC/USDT", "1h")
        assert report.is_clean
        assert report.n_duplicates == 0
        assert report.n_negative == 0

    def test_checksum_is_consistent(self) -> None:
        """Same DataFrame → same checksum every time (SHA-256, 64-char hex)."""
        df = _make_bars(30)
        r1 = validate_bars(df, "BTC/USDT", "1h")
        r2 = validate_bars(df, "BTC/USDT", "1h")
        assert r1.checksum == r2.checksum
        assert len(r1.checksum) == 64

    def test_manifest_has_required_keys(self, tmp_path: Path) -> None:
        """generate_manifest produces entries with all required metadata keys."""
        df = _make_bars(20)
        (tmp_path / "BTC_USDT_1h.csv").write_text("")  # create file placeholder
        df.to_csv(tmp_path / "BTC_USDT_1h.csv", index=False)

        manifest = generate_manifest(str(tmp_path))
        assert len(manifest) > 0

        # Navigate to the first entry, however it's keyed
        first_symbol = next(iter(manifest))
        first_entry = next(iter(manifest[first_symbol].values()))
        for key in ("n_bars", "checksum", "date_from", "date_to", "is_clean"):
            assert key in first_entry, f"manifest entry missing required key: '{key}'"


# ─────────────────────────────────────────────────────────────────────────────
# TestParquetProvider  (6 tests)
# ─────────────────────────────────────────────────────────────────────────────

# Skip the entire class if pyarrow is not installed
pytestmark_parquet = pytest.importorskip("pyarrow", reason="pyarrow not installed")


class TestParquetProvider:
    """ParquetProvider in privateye.data.providers.parquet_provider."""

    def test_save_load_roundtrip_matches_original(self, tmp_path: Path) -> None:
        """save() then load() must produce close prices matching the original."""
        from privateye.data.providers.parquet_provider import ParquetProvider

        df = _make_bars(50)
        provider = ParquetProvider(tmp_path)
        provider.save(df, "BTC/USDT", "1h")
        loaded = provider.load("BTC/USDT", "1h")

        assert len(loaded) == len(df)
        np.testing.assert_allclose(loaded["close"].values, df["close"].values, rtol=1e-5)

    def test_load_returns_sorted_by_timestamp(self, tmp_path: Path) -> None:
        """Loaded DataFrame must be sorted ascending by timestamp."""
        from privateye.data.providers.parquet_provider import ParquetProvider

        df = _make_bars(20)
        df_shuffled = df.sample(frac=1, random_state=7).reset_index(drop=True)
        provider = ParquetProvider(tmp_path)
        provider.save(df_shuffled, "ETH/USDT", "1h")
        loaded = provider.load("ETH/USDT", "1h")

        assert loaded["timestamp"].is_monotonic_increasing

    def test_missing_file_returns_empty_df(self, tmp_path: Path) -> None:
        """Loading a non-existent symbol/timeframe must return an empty DataFrame."""
        from privateye.data.providers.parquet_provider import ParquetProvider

        df = ParquetProvider(tmp_path).load("SOL/USDT", "1h")
        assert df.empty

    def test_load_with_limit_respected(self, tmp_path: Path) -> None:
        """load(limit=N) must return exactly N rows (the most recent N)."""
        from privateye.data.providers.parquet_provider import ParquetProvider

        df = _make_bars(50)
        provider = ParquetProvider(tmp_path)
        provider.save(df, "BTC/USDT", "4h")
        loaded = provider.load("BTC/USDT", "4h", limit=10)

        assert len(loaded) == 10

    def test_available_detects_parquet_files(self, tmp_path: Path) -> None:
        """available() must list every saved (symbol, timeframe) pair."""
        from privateye.data.providers.parquet_provider import ParquetProvider

        provider = ParquetProvider(tmp_path)
        provider.save(_make_bars(10), "BTC/USDT", "1h")
        provider.save(_make_bars(10), "ETH/USDT", "4h")
        pairs = provider.available()

        assert ("BTC/USDT", "1h") in pairs
        assert ("ETH/USDT", "4h") in pairs

    def test_column_normalization_uppercase_to_lowercase(self, tmp_path: Path) -> None:
        """Columns saved in UPPERCASE must be returned as lowercase."""
        from privateye.data.providers.parquet_provider import ParquetProvider

        df = _make_bars(10)
        df_upper = df.copy()
        df_upper.columns = [c.upper() for c in df_upper.columns]
        provider = ParquetProvider(tmp_path)
        provider.save(df_upper, "BTC/USDT", "1d")
        loaded = provider.load("BTC/USDT", "1d")

        assert "close" in loaded.columns
        assert "CLOSE" not in loaded.columns


# ─────────────────────────────────────────────────────────────────────────────
# TestGoldenSuite  (8 tests)
# ─────────────────────────────────────────────────────────────────────────────

# Reusable mock return values — mirrors what the real sub-runners return
_MOCK_BT = {"sharpe_ratio": 1.0, "total_pnl": 100.0, "total_trades": 5, "win_rate": 0.6}
_MOCK_WF = {
    "n_folds": 2, "mean_oos_sharpe": 0.8, "std_oos_sharpe": 0.2,
    "stability_score": 1.0, "mean_efficiency_ratio": 0.9,
    "mean_oos_max_dd": 0.04, "mean_oos_win_rate": 0.5,
    "total_oos_trades": 8, "total_oos_pnl": 80.0,
}
_MOCK_KF = {
    "n_folds": 2, "cv_mean_score": 0.7, "cv_std_score": 0.1,
    "stability_score": 1.0, "mean_is_score": 0.8,
    "overfitting_gap": 0.1, "total_cv_trades": 6, "total_cv_pnl": 60.0,
}
_MOCK_MC = {
    "n_simulations": 10, "ruin_probability": 0.0,
    "p5_sharpe": 0.3, "p50_sharpe": 0.8, "p95_sharpe": 1.5,
}


def _enter_golden_patches(output_path: str):
    """Context manager stack that bypasses all heavy computation in run_golden_backtest."""
    bars = _make_bars(60)
    mock_report = MagicMock()
    mock_report.trades = []
    mock_engine_instance = MagicMock()
    mock_engine_instance.run.return_value = mock_report

    # patch() objects are used as context managers below
    return (
        patch("privateye.backtesting.golden_suite._load_data", return_value=bars),
        patch("privateye.backtesting.golden_suite._run_backtest", return_value=_MOCK_BT),
        patch("privateye.backtesting.golden_suite._run_walk_forward", return_value=_MOCK_WF),
        patch("privateye.backtesting.golden_suite._run_kfold", return_value=_MOCK_KF),
        patch("privateye.backtesting.golden_suite._run_monte_carlo", return_value=_MOCK_MC),
        # Patch BacktestEngine on its own module — affects dynamic `from … import` inside the fn
        patch(
            "privateye.backtesting.engine.BacktestEngine",
            return_value=mock_engine_instance,
        ),
    )


class TestGoldenSuite:
    """privateye.backtesting.golden_suite — seed_all and run_golden_backtest."""

    def test_seed_all_sets_numpy_seed(self) -> None:
        """seed_all(N) locks NumPy RNG — two calls with the same seed yield the same value."""
        from privateye.backtesting.golden_suite import seed_all

        seed_all(42)
        v1 = float(np.random.rand())
        seed_all(42)
        v2 = float(np.random.rand())
        assert v1 == pytest.approx(v2)

    def test_seed_all_calls_torch_manual_seed_when_available(self) -> None:
        """When torch is importable, seed_all must call torch.manual_seed with the seed."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        # Replacing torch in sys.modules makes `import torch` inside seed_all get the mock
        with patch.dict("sys.modules", {"torch": mock_torch}):
            from privateye.backtesting.golden_suite import seed_all
            seed_all(7)
        mock_torch.manual_seed.assert_called_with(7)

    # ── Helpers for the structure tests ──────────────────────────────────────

    def _run_with_mocks(self, tmp_path: Path) -> dict:
        from privateye.backtesting.golden_suite import run_golden_backtest

        cfg = copy.deepcopy(_MINI_CFG)
        cfg["phase0"]["golden_output"] = str(tmp_path / "golden.json")

        p = _enter_golden_patches(cfg["phase0"]["golden_output"])
        with p[0], p[1], p[2], p[3], p[4], p[5]:
            return run_golden_backtest(cfg)

    def test_run_returns_dict_with_required_top_level_keys(self, tmp_path: Path) -> None:
        """run_golden_backtest must return a dict containing all required top-level keys."""
        result = self._run_with_mocks(tmp_path)
        required = {
            "generated_at", "symbol", "timeframe",
            "date_from", "date_to", "seed", "n_bars",
            "backtest", "walk_forward", "kfold", "monte_carlo",
        }
        for key in required:
            assert key in result, f"missing top-level key: '{key}'"

    def test_backtest_subdict_has_sharpe_ratio(self, tmp_path: Path) -> None:
        """The 'backtest' sub-dict must contain 'sharpe_ratio'."""
        result = self._run_with_mocks(tmp_path)
        assert "sharpe_ratio" in result["backtest"]

    def test_walk_forward_subdict_has_mean_oos_sharpe(self, tmp_path: Path) -> None:
        """The 'walk_forward' sub-dict must contain 'mean_oos_sharpe'."""
        result = self._run_with_mocks(tmp_path)
        assert "mean_oos_sharpe" in result["walk_forward"]

    def test_kfold_subdict_has_cv_mean_score(self, tmp_path: Path) -> None:
        """The 'kfold' sub-dict must contain 'cv_mean_score'."""
        result = self._run_with_mocks(tmp_path)
        assert "cv_mean_score" in result["kfold"]

    def test_monte_carlo_subdict_has_ruin_probability(self, tmp_path: Path) -> None:
        """The 'monte_carlo' sub-dict must contain 'ruin_probability'."""
        result = self._run_with_mocks(tmp_path)
        assert "ruin_probability" in result["monte_carlo"]

    def test_output_json_file_written_to_disk(self, tmp_path: Path) -> None:
        """run_golden_backtest must write a valid JSON file at the configured output path."""
        from privateye.backtesting.golden_suite import run_golden_backtest

        output_path = tmp_path / "golden.json"
        cfg = copy.deepcopy(_MINI_CFG)
        cfg["phase0"]["golden_output"] = str(output_path)

        p = _enter_golden_patches(str(output_path))
        with p[0], p[1], p[2], p[3], p[4], p[5]:
            run_golden_backtest(cfg)

        assert output_path.exists(), "golden_results JSON was not written to disk"
        with open(output_path) as fh:
            data = json.load(fh)
        assert "backtest" in data


# ─────────────────────────────────────────────────────────────────────────────
# TestMLflowTracker  (4 tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestMLflowTracker:
    """privateye.tracking.mlflow_tracker.MLflowTracker."""

    def test_disabled_log_backtest_is_noop(self) -> None:
        """When enabled=False, log_backtest must be a silent no-op (no crash, no calls)."""
        from privateye.tracking.mlflow_tracker import MLflowTracker

        tracker = MLflowTracker({"enabled": False})
        assert not tracker.enabled
        # Must not raise even if report is a mock
        tracker.log_backtest(MagicMock(), run_name="test")

    def test_mlflow_not_installed_gracefully_disables(self) -> None:
        """When mlflow is absent (sys.modules['mlflow']=None), tracker must silently disable."""
        with patch.dict("sys.modules", {"mlflow": None}):
            from privateye.tracking.mlflow_tracker import MLflowTracker
            tracker = MLflowTracker({"enabled": True})
        assert not tracker.enabled

    def test_log_backtest_calls_log_metrics_with_numeric_values_only(self) -> None:
        """log_backtest must filter out non-numeric fields before calling log_metrics."""
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            from privateye.tracking.mlflow_tracker import MLflowTracker
            tracker = MLflowTracker(
                {"enabled": True, "tracking_uri": "sqlite:///x.db", "experiment_name": "x"}
            )

        # Provide a report whose to_dict() has mixed value types
        mock_report = MagicMock()
        mock_report.to_dict.return_value = {
            "sharpe_ratio": 1.5,
            "total_pnl": 200.0,
            "total_trades": 10,        # int — included (cast to float)
            "symbol": "BTC/USDT",      # str  — excluded
            "equity_curve": [1, 2, 3], # list — excluded
        }
        tracker.log_backtest(mock_report, run_name="run-1")

        assert mock_mlflow.log_metrics.called
        metrics = mock_mlflow.log_metrics.call_args[0][0]
        assert "symbol" not in metrics
        assert "equity_curve" not in metrics
        for v in metrics.values():
            assert isinstance(v, float), f"non-float value in log_metrics: {v!r}"

    def test_log_backtest_calls_log_params_when_params_supplied(self) -> None:
        """log_backtest must call log_params when a non-empty params dict is given."""
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            from privateye.tracking.mlflow_tracker import MLflowTracker
            tracker = MLflowTracker(
                {"enabled": True, "tracking_uri": "sqlite:///x.db", "experiment_name": "x"}
            )

        mock_report = MagicMock()
        mock_report.to_dict.return_value = {"sharpe_ratio": 1.0}
        tracker.log_backtest(mock_report, run_name="run-2", params={"seed": 42})

        assert mock_mlflow.log_params.called
        params_arg = mock_mlflow.log_params.call_args[0][0]
        assert params_arg.get("seed") == 42


# ─────────────────────────────────────────────────────────────────────────────
# TestOptunaOptimizer  (4 tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestOptunaOptimizer:
    """privateye.tracking.optuna_optimizer.OptunaOptimizer."""

    def _make_opt_config(self, grid: dict | None = None):
        from privateye.backtesting.optimizer import OptimizationConfig

        return OptimizationConfig(
            strategy_name="directional",
            param_grid=grid or {"macd_fast": [10, 12]},
            metric="sharpe_ratio",
            min_trades=0,
        )

    def test_optuna_not_installed_raises_importerror(self) -> None:
        """OptunaOptimizer.__init__ must raise ImportError with a helpful message."""
        with patch.dict("sys.modules", {"optuna": None}):
            from privateye.tracking.optuna_optimizer import OptunaOptimizer
            with pytest.raises(ImportError, match="optuna"):
                OptunaOptimizer(_MINI_CFG, self._make_opt_config())

    def test_run_forwards_n_trials_to_study_optimize(self) -> None:
        """run(bars, n_trials=N) must call study.optimize(..., n_trials=N, ...)."""
        mock_study = MagicMock()
        mock_optuna = MagicMock()
        mock_optuna.create_study.return_value = mock_study

        with patch.dict("sys.modules", {"optuna": mock_optuna}):
            from privateye.tracking.optuna_optimizer import OptunaOptimizer
            opt = OptunaOptimizer(_MINI_CFG, self._make_opt_config())

        mock_report = MagicMock()
        mock_report.sharpe_ratio = 0.5
        mock_report.total_trades = 3

        with patch.object(opt, "_run_combination", return_value=mock_report):
            opt.run(_make_bars(), n_trials=7)

        mock_study.optimize.assert_called_once()
        call_kw = mock_study.optimize.call_args[1]
        assert call_kw["n_trials"] == 7

    def test_best_score_equals_max_over_all_trials(self) -> None:
        """report.best_score must equal the maximum score returned by any trial."""
        scores = [0.3, 1.8, 0.7]  # trial 1 is best
        score_iter = iter(scores)

        def fake_optimize(objective, n_trials, **kwargs):
            for i in range(n_trials):
                trial = MagicMock()
                trial.number = i
                # suggest_categorical cycles through the grid values
                idx = i  # captured by value via default-arg trick
                trial.suggest_categorical.side_effect = (
                    lambda key, vals, _i=idx: vals[_i % len(vals)]
                )
                objective(trial)

        mock_study = MagicMock()
        mock_study.optimize.side_effect = fake_optimize
        mock_optuna = MagicMock()
        mock_optuna.create_study.return_value = mock_study

        with patch.dict("sys.modules", {"optuna": mock_optuna}):
            from privateye.tracking.optuna_optimizer import OptunaOptimizer
            opt = OptunaOptimizer(
                _MINI_CFG,
                self._make_opt_config(grid={"macd_fast": [8, 10, 12]}),
            )

        def _side(params, bars):
            rep = MagicMock()
            rep.sharpe_ratio = next(score_iter, 0.0)
            rep.total_trades = 5
            return rep

        with patch.object(opt, "_run_combination", side_effect=_side):
            report = opt.run(_make_bars(), n_trials=3)

        assert report.best_score == pytest.approx(max(scores))

    def test_study_created_with_maximize_direction(self) -> None:
        """Optuna study must be created with direction='maximize'."""
        mock_study = MagicMock()
        mock_optuna = MagicMock()
        mock_optuna.create_study.return_value = mock_study

        with patch.dict("sys.modules", {"optuna": mock_optuna}):
            from privateye.tracking.optuna_optimizer import OptunaOptimizer
            opt = OptunaOptimizer(_MINI_CFG, self._make_opt_config())

        mock_report = MagicMock()
        mock_report.sharpe_ratio = 1.0
        mock_report.total_trades = 2
        with patch.object(opt, "_run_combination", return_value=mock_report):
            opt.run(_make_bars(), n_trials=1)

        mock_optuna.create_study.assert_called_once_with(direction="maximize")
