"""
Phase 8 test suite — Walk-Forward Backtesting Engine.

30 tests across:
  TestWalkForwardConfig     (4)  — defaults, from_config, missing keys, step default
  TestWalkForwardEngine     (9)  — n_folds, windows, disjoint, state, symbol, empty, length
  TestWalkForwardReport     (6)  — mean Sharpe, stability, efficiency, total trades, str
  TestWalkForwardFoldResult (4)  — efficiency formula, zero IS, has_enough_trades
  TestCLIIntegration        (7)  — import, choices, calls engine, prints, empty bars, config, returns
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from privateye.backtesting.metrics import BacktestReport
from privateye.backtesting.walk_forward import (
    WalkForwardConfig,
    WalkForwardEngine,
    WalkForwardFoldResult,
    WalkForwardReport,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_bars(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate *n* synthetic OHLCV bars."""
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

# train=400, test=200, step=200 → 3 folds with n=1000
_SMALL_WF = WalkForwardConfig(
    train_bars=400, test_bars=200, step_bars=200,
    min_train_bars=200, min_test_trades=5,
)

_DUMMY = BacktestReport(
    sharpe_ratio=1.0, total_trades=10, win_rate=0.6,
    max_drawdown_pct=0.05, total_pnl=100.0, initial_capital=10_000.0,
)


# ── TestWalkForwardConfig ─────────────────────────────────────────────────────

class TestWalkForwardConfig:

    def test_default_values(self):
        cfg = WalkForwardConfig()
        assert cfg.train_bars      == 6048
        assert cfg.test_bars       == 1512
        assert cfg.step_bars       == 1512
        assert cfg.min_train_bars  == 500
        assert cfg.min_test_trades == 5

    def test_from_config_reads_all_keys(self):
        raw = {"train_bars": 100, "test_bars": 50, "step_bars": 25,
               "min_train_bars": 30, "min_test_trades": 3}
        cfg = WalkForwardConfig.from_config(raw)
        assert cfg.train_bars      == 100
        assert cfg.test_bars       == 50
        assert cfg.step_bars       == 25
        assert cfg.min_train_bars  == 30
        assert cfg.min_test_trades == 3

    def test_from_config_missing_keys_use_defaults(self):
        cfg     = WalkForwardConfig.from_config({})
        default = WalkForwardConfig()
        assert cfg.train_bars     == default.train_bars
        assert cfg.test_bars      == default.test_bars
        assert cfg.step_bars      == default.step_bars
        assert cfg.min_train_bars == default.min_train_bars

    def test_step_defaults_to_test_bars_when_missing(self):
        """step_bars must fall back to test_bars when absent from config."""
        cfg = WalkForwardConfig.from_config({"train_bars": 200, "test_bars": 300})
        assert cfg.step_bars == 300


# ── TestWalkForwardEngine ─────────────────────────────────────────────────────

class TestWalkForwardEngine:

    def test_run_returns_walk_forward_report(self):
        with patch.object(WalkForwardEngine, "_run_fold", return_value=_DUMMY):
            engine = WalkForwardEngine(_MINI_CFG, _SMALL_WF)
            report = engine.run(_make_bars(1000), "BTC/USDT", "1h")
        assert isinstance(report, WalkForwardReport)

    def test_n_folds_matches_window_arithmetic(self):
        """n=1000, train=400, test=200, step=200 → exactly 3 folds."""
        with patch.object(WalkForwardEngine, "_run_fold", return_value=_DUMMY):
            engine = WalkForwardEngine(_MINI_CFG, _SMALL_WF)
            report = engine.run(_make_bars(1000), "BTC/USDT", "1h")
        assert report.n_folds == 3

    def test_oos_windows_dont_overlap(self):
        """Consecutive OOS windows must not share any bar index."""
        with patch.object(WalkForwardEngine, "_run_fold", return_value=_DUMMY):
            engine = WalkForwardEngine(_MINI_CFG, _SMALL_WF)
            report = engine.run(_make_bars(1000), "BTC/USDT", "1h")
        for i in range(len(report.fold_results) - 1):
            curr = report.fold_results[i]
            nxt  = report.fold_results[i + 1]
            assert curr.oos_end <= nxt.oos_start, (
                f"Fold {i} OOS [{curr.oos_start},{curr.oos_end}) overlaps "
                f"fold {i+1} OOS [{nxt.oos_start},{nxt.oos_end})"
            )

    def test_is_oos_disjoint_within_fold(self):
        """IS and OOS windows within a fold must not share any bar index."""
        with patch.object(WalkForwardEngine, "_run_fold", return_value=_DUMMY):
            engine = WalkForwardEngine(_MINI_CFG, _SMALL_WF)
            report = engine.run(_make_bars(1000), "BTC/USDT", "1h")
        for fr in report.fold_results:
            is_range  = set(range(fr.is_start,  fr.is_end))
            oos_range = set(range(fr.oos_start, fr.oos_end))
            assert is_range.isdisjoint(oos_range), \
                f"Fold {fr.fold_num}: IS and OOS overlap"

    def test_run_fold_called_twice_per_fold(self):
        """_run_fold is called IS+OOS=2× per fold; bar lengths match config."""
        with patch.object(WalkForwardEngine, "_run_fold", return_value=_DUMMY) as mock_fold:
            engine = WalkForwardEngine(_MINI_CFG, _SMALL_WF)
            engine.run(_make_bars(1000), "BTC/USDT", "1h")
        assert mock_fold.call_count == 6  # 3 folds × 2
        for i, call in enumerate(mock_fold.call_args_list):
            bars_arg = call[0][0]
            expected = 400 if i % 2 == 0 else 200  # IS=train_bars, OOS=test_bars
            assert len(bars_arg) == expected, \
                f"Call {i}: expected {expected} bars, got {len(bars_arg)}"

    def test_report_has_correct_symbol_and_timeframe(self):
        with patch.object(WalkForwardEngine, "_run_fold", return_value=_DUMMY):
            engine = WalkForwardEngine(_MINI_CFG, _SMALL_WF)
            report = engine.run(_make_bars(1000), "ETH/USDT", "4h")
        assert report.symbol    == "ETH/USDT"
        assert report.timeframe == "4h"

    def test_insufficient_data_returns_empty_report(self):
        """When bars < train_bars + test_bars, n_folds=0 and _run_fold not called."""
        with patch.object(WalkForwardEngine, "_run_fold", return_value=_DUMMY) as mock_fold:
            engine = WalkForwardEngine(_MINI_CFG, _SMALL_WF)
            report = engine.run(_make_bars(100), "BTC/USDT", "1h")  # too small
        assert report.n_folds      == 0
        assert report.fold_results == []
        mock_fold.assert_not_called()

    def test_fold_results_length_matches_n_folds(self):
        with patch.object(WalkForwardEngine, "_run_fold", return_value=_DUMMY):
            engine = WalkForwardEngine(_MINI_CFG, _SMALL_WF)
            report = engine.run(_make_bars(1000), "BTC/USDT", "1h")
        assert len(report.fold_results) == report.n_folds

    def test_initial_capital_per_fold_from_config(self):
        """Each fold report must carry the configured initial_capital."""
        cap_report = BacktestReport(sharpe_ratio=1.0, total_trades=10,
                                    initial_capital=10_000.0)
        with patch.object(WalkForwardEngine, "_run_fold", return_value=cap_report):
            engine = WalkForwardEngine(_MINI_CFG, _SMALL_WF)
            report = engine.run(_make_bars(1000), "BTC/USDT", "1h")
        for fr in report.fold_results:
            assert fr.is_report.initial_capital  == 10_000.0
            assert fr.oos_report.initial_capital == 10_000.0


# ── TestWalkForwardReport ─────────────────────────────────────────────────────

class TestWalkForwardReport:

    def _run_with_oos_sharpes(self, oos_sharpes, oos_trades=None):
        """Run engine; IS Sharpe=2.0 for every fold, OOS Sharpe from list."""
        if oos_trades is None:
            oos_trades = [10] * len(oos_sharpes)
        side_effects = []
        for s, ot in zip(oos_sharpes, oos_trades):
            side_effects.append(BacktestReport(sharpe_ratio=2.0, total_trades=20))
            side_effects.append(BacktestReport(sharpe_ratio=s,   total_trades=ot))
        with patch.object(WalkForwardEngine, "_run_fold", side_effect=side_effects):
            engine = WalkForwardEngine(_MINI_CFG, _SMALL_WF)
            return engine.run(_make_bars(1000), "BTC/USDT", "1h")

    def test_mean_oos_sharpe_equals_arithmetic_mean(self):
        oos = [0.8, 0.5, 1.2]
        report = self._run_with_oos_sharpes(oos)
        assert abs(report.mean_oos_sharpe - sum(oos) / len(oos)) < 1e-9

    def test_stability_score_is_fraction_with_positive_sharpe(self):
        """2 of 3 OOS Sharpes positive → stability_score = 2/3."""
        report = self._run_with_oos_sharpes([0.8, -0.3, 0.5])
        assert abs(report.stability_score - 2 / 3) < 1e-9

    def test_stability_zero_when_all_folds_negative(self):
        report = self._run_with_oos_sharpes([-1.0, -0.5, -0.3])
        assert report.stability_score == 0.0

    def test_mean_efficiency_ratio_formula(self):
        """IS=2.0, OOS=1.0 ⇒ efficiency=0.5 per fold → mean=0.5."""
        side_effects = []
        for _ in range(3):
            side_effects.append(BacktestReport(sharpe_ratio=2.0, total_trades=20))
            side_effects.append(BacktestReport(sharpe_ratio=1.0, total_trades=8))
        with patch.object(WalkForwardEngine, "_run_fold", side_effect=side_effects):
            engine = WalkForwardEngine(_MINI_CFG, _SMALL_WF)
            report = engine.run(_make_bars(1000), "BTC/USDT", "1h")
        assert abs(report.mean_efficiency_ratio - 0.5) < 1e-9

    def test_total_oos_trades_sums_all_folds(self):
        oos_trades = [8, 6, 9]
        report = self._run_with_oos_sharpes([0.8, 0.5, 1.0], oos_trades=oos_trades)
        assert report.total_oos_trades == sum(oos_trades)

    def test_str_contains_expected_labels(self):
        report = WalkForwardReport(
            symbol="BTC/USDT", timeframe="1h", n_folds=0, config=WalkForwardConfig()
        )
        text = str(report)
        assert "WALK-FORWARD REPORT"     in text
        assert "OUT-OF-SAMPLE AGGREGATE" in text
        assert "BTC/USDT"                in text


# ── TestWalkForwardFoldResult ─────────────────────────────────────────────────

class TestWalkForwardFoldResult:

    def _run_fold_results(self, is_sharpes, oos_sharpes, oos_trades=None,
                          min_test_trades=5):
        if oos_trades is None:
            oos_trades = [10] * len(is_sharpes)
        side_effects = []
        for is_s, oos_s, ot in zip(is_sharpes, oos_sharpes, oos_trades):
            side_effects.append(BacktestReport(sharpe_ratio=is_s, total_trades=20))
            side_effects.append(BacktestReport(sharpe_ratio=oos_s, total_trades=ot))
        wf = WalkForwardConfig(train_bars=400, test_bars=200, step_bars=200,
                               min_train_bars=200, min_test_trades=min_test_trades)
        with patch.object(WalkForwardEngine, "_run_fold", side_effect=side_effects):
            engine = WalkForwardEngine(_MINI_CFG, wf)
            report = engine.run(_make_bars(1000), "BTC/USDT", "1h")
        return report.fold_results

    def test_efficiency_ratio_formula(self):
        """efficiency_ratio = oos_sharpe / is_sharpe."""
        folds = self._run_fold_results([2.0, 2.0, 2.0], [1.0, 1.0, 1.0])
        for fr in folds:
            assert abs(fr.efficiency_ratio - 0.5) < 1e-9

    def test_efficiency_ratio_zero_when_is_sharpe_zero(self):
        """No ZeroDivisionError: efficiency_ratio=0.0 when IS Sharpe=0."""
        folds = self._run_fold_results([0.0, 0.0, 0.0], [0.8, 0.8, 0.8])
        for fr in folds:
            assert fr.efficiency_ratio == 0.0

    def test_has_enough_trades_true_when_at_threshold(self):
        """has_enough_trades=True when OOS trades >= min_test_trades."""
        folds = self._run_fold_results(
            [1.5, 1.5, 1.5], [0.5, 0.5, 0.5],
            oos_trades=[5, 5, 5], min_test_trades=5,
        )
        for fr in folds:
            assert fr.has_enough_trades is True

    def test_has_enough_trades_false_when_below_threshold(self):
        """has_enough_trades=False when OOS trades < min_test_trades."""
        folds = self._run_fold_results(
            [1.5, 1.5, 1.5], [0.5, 0.5, 0.5],
            oos_trades=[3, 3, 3], min_test_trades=10,
        )
        for fr in folds:
            assert fr.has_enough_trades is False


# ── TestCLIIntegration ────────────────────────────────────────────────────────

class TestCLIIntegration:

    def test_walk_forward_in_cli_choices(self):
        """--mode walk_forward must be a valid argparse choice."""
        with patch("sys.argv", ["main", "--mode", "walk_forward"]):
            with patch("privateye.config.loader.load_config", return_value={
                "symbols": ["BTC/USDT"], "primary_timeframe": "1h",
                "walk_forward": {},
                "backtesting": {"data_dir": "data/historical",
                                "initial_capital": 10_000.0},
            }):
                with patch("privateye.main.run_walk_forward",
                           return_value=None) as mock_wf:
                    from privateye.main import cli
                    cli()
                    mock_wf.assert_called_once()

    def test_run_walk_forward_importable(self):
        from privateye.main import run_walk_forward  # noqa: F401
        assert callable(run_walk_forward)

    def test_run_walk_forward_calls_engine_run(self):
        """run_walk_forward must invoke WalkForwardEngine.run() for each symbol."""
        from privateye.main import run_walk_forward

        cfg = {**_MINI_CFG, "symbols": ["BTC/USDT"], "primary_timeframe": "1h",
               "walk_forward": {"train_bars": 400, "test_bars": 200, "step_bars": 200}}
        dummy = WalkForwardReport(symbol="BTC/USDT", timeframe="1h",
                                  n_folds=0, config=WalkForwardConfig())
        with patch("privateye.data.providers.csv_provider.CSVProvider") as mock_cls:
            mock_cls.return_value.load.return_value = _make_bars(500)
            with patch("privateye.backtesting.walk_forward.WalkForwardEngine.run",
                       return_value=dummy) as mock_run:
                run_walk_forward(cfg)
                mock_run.assert_called_once()

    def test_run_walk_forward_prints_report(self, capsys):
        """run_walk_forward must print the WalkForwardReport to stdout."""
        from privateye.main import run_walk_forward

        cfg = {**_MINI_CFG, "symbols": ["BTC/USDT"], "primary_timeframe": "1h",
               "walk_forward": {}}
        dummy = WalkForwardReport(symbol="BTC/USDT", timeframe="1h",
                                  n_folds=0, config=WalkForwardConfig())
        with patch("privateye.data.providers.csv_provider.CSVProvider") as mock_cls:
            mock_cls.return_value.load.return_value = _make_bars(100)
            with patch("privateye.backtesting.walk_forward.WalkForwardEngine.run",
                       return_value=dummy):
                run_walk_forward(cfg)
        assert "WALK-FORWARD" in capsys.readouterr().out

    def test_run_walk_forward_handles_empty_bars(self):
        """Empty bars (missing data file) must not crash; returns None."""
        from privateye.main import run_walk_forward

        cfg = {**_MINI_CFG, "symbols": ["BTC/USDT"], "primary_timeframe": "1h",
               "walk_forward": {}}
        with patch("privateye.data.providers.csv_provider.CSVProvider") as mock_cls:
            mock_cls.return_value.load.return_value = pd.DataFrame()
            result = run_walk_forward(cfg)
        assert result is None

    def test_from_config_called_with_cfg_walk_forward(self):
        """run_walk_forward must pass cfg['walk_forward'] to from_config."""
        from privateye.main import run_walk_forward

        wf_section = {"train_bars": 300, "test_bars": 150, "step_bars": 150}
        cfg = {**_MINI_CFG, "symbols": ["BTC/USDT"], "primary_timeframe": "1h",
               "walk_forward": wf_section}
        with patch.object(WalkForwardConfig, "from_config",
                          wraps=WalkForwardConfig.from_config) as spy_fc:
            with patch("privateye.data.providers.csv_provider.CSVProvider") as mock_cls:
                mock_cls.return_value.load.return_value = pd.DataFrame()
                run_walk_forward(cfg)
            spy_fc.assert_called_once_with(wf_section)

    def test_run_walk_forward_returns_walk_forward_report(self):
        """run_walk_forward must return WalkForwardReport when bars are found."""
        from privateye.main import run_walk_forward

        cfg = {**_MINI_CFG, "symbols": ["BTC/USDT"], "primary_timeframe": "1h",
               "walk_forward": {}}
        dummy = WalkForwardReport(symbol="BTC/USDT", timeframe="1h",
                                  n_folds=0, config=WalkForwardConfig())
        with patch("privateye.data.providers.csv_provider.CSVProvider") as mock_cls:
            mock_cls.return_value.load.return_value = _make_bars(100)
            with patch("privateye.backtesting.walk_forward.WalkForwardEngine.run",
                       return_value=dummy):
                result = run_walk_forward(cfg)
        assert isinstance(result, WalkForwardReport)
