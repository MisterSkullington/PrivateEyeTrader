"""
Phase 11 test suite — Purged K-Fold Cross-Validation Engine.

30 tests across:
  TestFoldHelpers        (4)  — _compute_fold_ranges, _get_train_indices
  TestPurgedKFoldConfig  (4)  — defaults, from_config, missing keys, partial override
  TestKFoldReport        (5)  — cv_mean, stability, stability_zero, overfitting_gap, __str__
  TestPurgedKFoldEngine (12)  — returns report, n_folds, twice per fold, total calls,
                                skip insufficient, no folds guard, cv_score metric,
                                has_enough_trades, cv bars length, symbol/timeframe,
                                metric=total_pnl, fold_results length
  TestCLIIntegration     (5)  — import, choices, calls engine, empty bars, returns report
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from privateye.backtesting.kfold import (
    KFoldReport,
    KFoldResult,
    PurgedKFoldConfig,
    PurgedKFoldEngine,
    _compute_fold_ranges,
    _get_train_indices,
)
from privateye.backtesting.metrics import BacktestReport


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


def _make_report(
    sharpe: float = 1.0,
    total_pnl: float = 500.0,
    calmar: float = 0.8,
    total_trades: int = 10,
    win_rate: float = 0.6,
) -> BacktestReport:
    return BacktestReport(
        total_trades=total_trades,
        win_rate=win_rate,
        total_pnl=total_pnl,
        max_drawdown_pct=0.05,
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        initial_capital=10_000.0,
    )


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

# n=1000, n_folds=5 → fold_size=200; small purge so no folds skipped
_SMALL_KF = PurgedKFoldConfig(
    n_folds=5, purge_bars=50, embargo_bars=10,
    min_train_bars=100, min_test_trades=5,
)

_DUMMY = _make_report()


# ── TestFoldHelpers ───────────────────────────────────────────────────────────

class TestFoldHelpers:

    def test_compute_fold_ranges_even_division(self):
        """n=10, n_folds=2 → [(0,5),(5,10)]; no remainder."""
        ranges = _compute_fold_ranges(10, 2)
        assert ranges == [(0, 5), (5, 10)]

    def test_compute_fold_ranges_last_absorbs_remainder(self):
        """n=11, n_folds=2 → [(0,5),(5,11)]; last fold absorbs extra bar."""
        ranges = _compute_fold_ranges(11, 2)
        assert ranges == [(0, 5), (5, 11)]

    def test_get_train_indices_excludes_purge_and_embargo(self):
        """
        n=100, test=[30,60), purge=5, embargo=5
        excluded=[25,65); train=[0,25)∪[65,100)
        """
        idx = _get_train_indices(100, 30, 60, purge_bars=5, embargo_bars=5)
        idx_set = set(idx.tolist())
        assert 24 in idx_set       # last bar before purge zone
        assert 25 not in idx_set   # first excluded bar
        assert 64 not in idx_set   # last excluded bar
        assert 65 in idx_set       # first bar after embargo
        assert len(idx) == 60      # 25 + 35

    def test_get_train_indices_clamped_to_bounds(self):
        """
        Purge before index 0 and embargo past index n are clamped.
        n=100, test=[0,20), purge=50, embargo=200 → excluded=[0,100), train=[]
        """
        idx = _get_train_indices(100, 0, 20, purge_bars=50, embargo_bars=200)
        assert len(idx) == 0  # entire dataset is excluded


# ── TestPurgedKFoldConfig ─────────────────────────────────────────────────────

class TestPurgedKFoldConfig:

    def test_default_values(self):
        cfg = PurgedKFoldConfig()
        assert cfg.n_folds         == 5
        assert cfg.purge_bars      == 100
        assert cfg.embargo_bars    == 10
        assert cfg.min_train_bars  == 500
        assert cfg.min_test_trades == 5

    def test_from_config_reads_all_keys(self):
        raw = {
            "n_folds": 3, "purge_bars": 50, "embargo_bars": 5,
            "min_train_bars": 200, "min_test_trades": 3,
        }
        cfg = PurgedKFoldConfig.from_config(raw)
        assert cfg.n_folds         == 3
        assert cfg.purge_bars      == 50
        assert cfg.embargo_bars    == 5
        assert cfg.min_train_bars  == 200
        assert cfg.min_test_trades == 3

    def test_from_config_missing_keys_use_defaults(self):
        cfg     = PurgedKFoldConfig.from_config({})
        default = PurgedKFoldConfig()
        assert cfg.n_folds         == default.n_folds
        assert cfg.purge_bars      == default.purge_bars
        assert cfg.embargo_bars    == default.embargo_bars
        assert cfg.min_train_bars  == default.min_train_bars
        assert cfg.min_test_trades == default.min_test_trades

    def test_from_config_partial_override(self):
        """Only provided keys are overridden; the rest keep defaults."""
        cfg = PurgedKFoldConfig.from_config({"n_folds": 10})
        assert cfg.n_folds    == 10
        assert cfg.purge_bars == 100     # default unchanged


# ── TestKFoldReport ───────────────────────────────────────────────────────────

class TestKFoldReport:
    """Test KFoldReport aggregate computation via PurgedKFoldEngine.run()."""

    def _run_with_mock(self, n_folds: int, is_sharpes, cv_sharpes,
                       cv_trades=None, cv_pnls=None, cv_win_rates=None):
        """
        Run engine with mocked _run_fold.
        side_effects interleaved: IS₁, CV₁, IS₂, CV₂, …
        """
        if cv_trades    is None: cv_trades    = [10] * n_folds
        if cv_pnls      is None: cv_pnls      = [100.0] * n_folds
        if cv_win_rates is None: cv_win_rates = [0.5] * n_folds

        kf_config = PurgedKFoldConfig(
            n_folds=n_folds, purge_bars=50, embargo_bars=10,
            min_train_bars=100, min_test_trades=5,
        )
        side_effects = []
        for is_s, cv_s, ct, cp, cw in zip(
            is_sharpes, cv_sharpes, cv_trades, cv_pnls, cv_win_rates
        ):
            side_effects.append(_make_report(sharpe=is_s, total_trades=20))
            side_effects.append(_make_report(sharpe=cv_s, total_trades=ct,
                                             total_pnl=cp, win_rate=cw))

        with patch.object(PurgedKFoldEngine, "_run_fold", side_effect=side_effects):
            engine = PurgedKFoldEngine(_MINI_CFG, kf_config, metric="sharpe_ratio")
            return engine.run(_make_bars(1000), "BTC/USDT", "1h")

    def test_cv_mean_score_equals_arithmetic_mean(self):
        cv = [0.8, 0.5, 1.2]
        report = self._run_with_mock(3, [2.0] * 3, cv)
        assert abs(report.cv_mean_score - sum(cv) / len(cv)) < 1e-9

    def test_stability_score_fraction_with_positive_cv(self):
        """2 of 3 positive CV scores → stability = 2/3."""
        report = self._run_with_mock(3, [2.0] * 3, [0.8, -0.3, 0.5])
        assert abs(report.stability_score - 2 / 3) < 1e-9

    def test_stability_zero_when_all_cv_negative(self):
        report = self._run_with_mock(3, [2.0] * 3, [-1.0, -0.5, -0.3])
        assert report.stability_score == 0.0

    def test_overfitting_gap_equals_is_minus_cv_mean(self):
        """IS=2.0, CV=0.8 → gap = 1.2."""
        report = self._run_with_mock(3, [2.0] * 3, [0.8] * 3)
        assert abs(report.overfitting_gap - 1.2) < 1e-9

    def test_str_contains_expected_labels(self):
        report = KFoldReport(
            symbol="BTC/USDT", timeframe="1h",
            n_folds=0, config=PurgedKFoldConfig(), metric="sharpe_ratio",
        )
        text = str(report)
        assert "PURGED K-FOLD" in text
        assert "CROSS-VALIDATION" in text
        assert "BTC/USDT" in text
        assert "CV Mean Score" in text
        assert "Overfitting Gap" in text


# ── TestPurgedKFoldEngine ─────────────────────────────────────────────────────

class TestPurgedKFoldEngine:

    def test_run_returns_kfold_report(self):
        with patch.object(PurgedKFoldEngine, "_run_fold", return_value=_DUMMY):
            engine = PurgedKFoldEngine(_MINI_CFG, _SMALL_KF)
            report = engine.run(_make_bars(1000), "BTC/USDT", "1h")
        assert isinstance(report, KFoldReport)

    def test_n_folds_matches_config(self):
        """n=1000, n_folds=5 → all 5 folds complete (purge=50, min_train=100)."""
        with patch.object(PurgedKFoldEngine, "_run_fold", return_value=_DUMMY):
            engine = PurgedKFoldEngine(_MINI_CFG, _SMALL_KF)
            report = engine.run(_make_bars(1000), "BTC/USDT", "1h")
        assert report.n_folds == 5

    def test_run_fold_called_twice_per_fold(self):
        """Each fold triggers IS call + CV call → 2 calls per fold."""
        with patch.object(PurgedKFoldEngine, "_run_fold", return_value=_DUMMY) as mock_fold:
            engine = PurgedKFoldEngine(_MINI_CFG, _SMALL_KF)
            engine.run(_make_bars(1000), "BTC/USDT", "1h")
        # 5 folds × 2 runs = 10 total calls
        assert mock_fold.call_count == 10

    def test_total_run_fold_calls_equals_twice_n_folds(self):
        """Parameterised check: n_folds=3 → 6 _run_fold calls."""
        kf3 = PurgedKFoldConfig(
            n_folds=3, purge_bars=50, embargo_bars=10,
            min_train_bars=100, min_test_trades=5,
        )
        with patch.object(PurgedKFoldEngine, "_run_fold", return_value=_DUMMY) as mock_fold:
            engine = PurgedKFoldEngine(_MINI_CFG, kf3)
            engine.run(_make_bars(1000), "BTC/USDT", "1h")
        assert mock_fold.call_count == 6

    def test_fold_with_insufficient_train_bars_is_skipped(self):
        """
        Large purge_bars leaves some folds with too few training bars.
        With purge=400, embargo=50, min_train_bars=450, n=1000, n_folds=5:
          - Fold 1: train=[250,1000)=750 ✓
          - Fold 2: train=[450,1000)=550 ✓
          - Fold 3: train=[0,0)+[650,1000)=350 < 450 → skip
          - Fold 4: train=[0,200)+[850,1000)=350 < 450 → skip
          - Fold 5: train=[0,400)+[]    =400 < 450 → skip
        Expected: n_folds=2, _run_fold called 4 times.
        """
        kf_config = PurgedKFoldConfig(
            n_folds=5, purge_bars=400, embargo_bars=50,
            min_train_bars=450, min_test_trades=5,
        )
        with patch.object(PurgedKFoldEngine, "_run_fold", return_value=_DUMMY) as mock_fold:
            engine = PurgedKFoldEngine(_MINI_CFG, kf_config)
            report = engine.run(_make_bars(1000), "BTC/USDT", "1h")
        assert report.n_folds == 2
        assert mock_fold.call_count == 4

    def test_no_folds_complete_returns_empty_report(self):
        """When ALL folds are skipped, return KFoldReport with n_folds=0."""
        kf_config = PurgedKFoldConfig(
            n_folds=5, purge_bars=400, embargo_bars=50,
            min_train_bars=900,  # impossibly large — all folds skipped
            min_test_trades=5,
        )
        with patch.object(PurgedKFoldEngine, "_run_fold", return_value=_DUMMY) as mock_fold:
            engine = PurgedKFoldEngine(_MINI_CFG, kf_config)
            report = engine.run(_make_bars(1000), "BTC/USDT", "1h")
        assert report.n_folds == 0
        assert report.fold_results == []
        mock_fold.assert_not_called()

    def test_cv_score_extracted_from_sharpe_ratio(self):
        """cv_score for each fold = cv_report.sharpe_ratio when metric=sharpe_ratio."""
        kf_config = PurgedKFoldConfig(
            n_folds=3, purge_bars=50, embargo_bars=10,
            min_train_bars=100, min_test_trades=5,
        )
        # IS Sharpe=2.0, CV Sharpe alternates 0.5, 1.0, 1.5
        cv_sharpes = [0.5, 1.0, 1.5]
        side_effects = []
        for cs in cv_sharpes:
            side_effects.append(_make_report(sharpe=2.0, total_trades=20))
            side_effects.append(_make_report(sharpe=cs,  total_trades=10))

        with patch.object(PurgedKFoldEngine, "_run_fold", side_effect=side_effects):
            engine = PurgedKFoldEngine(_MINI_CFG, kf_config, metric="sharpe_ratio")
            report = engine.run(_make_bars(1000), "BTC/USDT", "1h")

        extracted = [fr.cv_score for fr in report.fold_results]
        assert extracted == pytest.approx(cv_sharpes)

    def test_has_enough_trades_flagged_below_threshold(self):
        """Folds with CV trades < min_test_trades have has_enough_trades=False."""
        kf_config = PurgedKFoldConfig(
            n_folds=3, purge_bars=50, embargo_bars=10,
            min_train_bars=100, min_test_trades=10,
        )
        side_effects = []
        for _ in range(3):
            side_effects.append(_make_report(total_trades=20))  # IS
            side_effects.append(_make_report(total_trades=3))   # CV — 3 < 10

        with patch.object(PurgedKFoldEngine, "_run_fold", side_effect=side_effects):
            engine = PurgedKFoldEngine(_MINI_CFG, kf_config)
            report = engine.run(_make_bars(1000), "BTC/USDT", "1h")

        for fr in report.fold_results:
            assert fr.has_enough_trades is False

    def test_cv_bars_have_fold_size_length(self):
        """
        CV (test) bars passed to each odd-indexed _run_fold call must equal
        fold_size = n // n_folds = 1000 // 5 = 200 (last fold absorbs remainder).
        """
        with patch.object(PurgedKFoldEngine, "_run_fold",
                          return_value=_DUMMY) as mock_fold:
            engine = PurgedKFoldEngine(_MINI_CFG, _SMALL_KF)
            engine.run(_make_bars(1000), "BTC/USDT", "1h")

        calls = mock_fold.call_args_list
        for i, call in enumerate(calls):
            if i % 2 == 1:  # odd index → CV run
                bars_arg = call[0][0]
                assert len(bars_arg) == 200, \
                    f"CV call {i}: expected 200 bars, got {len(bars_arg)}"

    def test_report_has_correct_symbol_and_timeframe(self):
        with patch.object(PurgedKFoldEngine, "_run_fold", return_value=_DUMMY):
            engine = PurgedKFoldEngine(_MINI_CFG, _SMALL_KF)
            report = engine.run(_make_bars(1000), "ETH/USDT", "4h")
        assert report.symbol    == "ETH/USDT"
        assert report.timeframe == "4h"

    def test_metric_total_pnl_used_for_cv_score(self):
        """When metric='total_pnl', cv_score = cv_report.total_pnl."""
        kf_config = PurgedKFoldConfig(
            n_folds=2, purge_bars=50, embargo_bars=10,
            min_train_bars=100, min_test_trades=1,
        )
        side_effects = [
            _make_report(total_pnl=0.0, total_trades=20),    # IS fold 1
            _make_report(total_pnl=300.0, total_trades=5),   # CV fold 1
            _make_report(total_pnl=0.0, total_trades=20),    # IS fold 2
            _make_report(total_pnl=700.0, total_trades=5),   # CV fold 2
        ]
        with patch.object(PurgedKFoldEngine, "_run_fold", side_effect=side_effects):
            engine = PurgedKFoldEngine(_MINI_CFG, kf_config, metric="total_pnl")
            report = engine.run(_make_bars(1000), "BTC/USDT", "1h")

        cv_scores = [fr.cv_score for fr in report.fold_results]
        assert cv_scores[0] == pytest.approx(300.0)
        assert cv_scores[1] == pytest.approx(700.0)

    def test_fold_results_length_matches_n_folds(self):
        with patch.object(PurgedKFoldEngine, "_run_fold", return_value=_DUMMY):
            engine = PurgedKFoldEngine(_MINI_CFG, _SMALL_KF)
            report = engine.run(_make_bars(1000), "BTC/USDT", "1h")
        assert len(report.fold_results) == report.n_folds


# ── TestCLIIntegration ────────────────────────────────────────────────────────

class TestCLIIntegration:

    def test_run_kfold_importable(self):
        from privateye.main import run_kfold  # noqa: F401
        assert callable(run_kfold)

    def test_kfold_in_cli_choices(self):
        """--mode kfold must be a valid argparse choice."""
        with patch("sys.argv", ["main", "--mode", "kfold"]):
            with patch("privateye.config.loader.load_config", return_value={
                "symbols": ["BTC/USDT"], "primary_timeframe": "1h",
                "kfold": {},
                "backtesting": {"data_dir": "data/historical",
                                "initial_capital": 10_000.0},
            }):
                with patch("privateye.main.run_kfold", return_value=None) as mock_kf:
                    from privateye.main import cli
                    cli()
                    mock_kf.assert_called_once()

    def test_run_kfold_calls_engine_run(self):
        """run_kfold must invoke PurgedKFoldEngine.run() for each symbol."""
        from privateye.main import run_kfold

        cfg = {**_MINI_CFG, "symbols": ["BTC/USDT"], "primary_timeframe": "1h",
               "kfold": {"n_folds": 3, "purge_bars": 50, "embargo_bars": 10}}
        dummy = KFoldReport(
            symbol="BTC/USDT", timeframe="1h",
            n_folds=0, config=PurgedKFoldConfig(), metric="sharpe_ratio",
        )
        with patch("privateye.data.providers.csv_provider.CSVProvider") as mock_cls:
            mock_cls.return_value.load.return_value = _make_bars(500)
            with patch("privateye.backtesting.kfold.PurgedKFoldEngine.run",
                       return_value=dummy) as mock_run:
                run_kfold(cfg)
                mock_run.assert_called_once()

    def test_run_kfold_handles_empty_bars(self):
        """Empty bars (missing data) must not crash; returns None."""
        from privateye.main import run_kfold

        cfg = {**_MINI_CFG, "symbols": ["BTC/USDT"], "primary_timeframe": "1h",
               "kfold": {}}
        with patch("privateye.data.providers.csv_provider.CSVProvider") as mock_cls:
            mock_cls.return_value.load.return_value = pd.DataFrame()
            result = run_kfold(cfg)
        assert result is None

    def test_run_kfold_returns_kfold_report(self):
        """run_kfold returns a KFoldReport when bars are available."""
        from privateye.main import run_kfold

        cfg = {**_MINI_CFG, "symbols": ["BTC/USDT"], "primary_timeframe": "1h",
               "kfold": {}}
        dummy = KFoldReport(
            symbol="BTC/USDT", timeframe="1h",
            n_folds=0, config=PurgedKFoldConfig(), metric="sharpe_ratio",
        )
        with patch("privateye.data.providers.csv_provider.CSVProvider") as mock_cls:
            mock_cls.return_value.load.return_value = _make_bars(100)
            with patch("privateye.backtesting.kfold.PurgedKFoldEngine.run",
                       return_value=dummy):
                result = run_kfold(cfg)
        assert isinstance(result, KFoldReport)
