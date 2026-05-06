"""
Purged K-Fold Cross-Validation Engine.

Splits historical bars into K roughly-equal folds and evaluates the strategy
on each held-out fold, removing ``purge_bars`` at the training/test boundary
to prevent look-back indicator leakage and optionally enforcing an
``embargo_bars`` gap after each test fold.

Unlike walk-forward (always past → future), purged K-fold uses *all* data
for testing by rotating the held-out fold, giving a lower-variance estimate
of strategy performance over the full dataset.

Usage::

    from privateye.backtesting.kfold import PurgedKFoldConfig, PurgedKFoldEngine

    kf_config = PurgedKFoldConfig(n_folds=5, purge_bars=100, embargo_bars=10)
    engine    = PurgedKFoldEngine(cfg, kf_config, metric="sharpe_ratio")
    report    = engine.run(bars, symbol="BTC/USDT", timeframe="1h")
    print(report)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from privateye.backtesting.metrics import BacktestReport
from privateye.utils.logging import get_logger

log = get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Index helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_fold_ranges(n: int, n_folds: int) -> list[tuple[int, int]]:
    """
    Divide *n* bars into *n_folds* consecutive chunks.

    Returns a list of ``(start, end)`` pairs (end is exclusive) such that
    the last fold absorbs any remainder bars.
    """
    fold_size = n // n_folds
    ranges: list[tuple[int, int]] = []
    for i in range(n_folds):
        start = i * fold_size
        end   = start + fold_size if i < n_folds - 1 else n
        ranges.append((start, end))
    return ranges


def _get_train_indices(
    n:            int,
    test_start:   int,
    test_end:     int,
    purge_bars:   int,
    embargo_bars: int,
) -> np.ndarray:
    """
    Return training bar indices — everything *outside* the excluded zone.

    The excluded zone is
    ``[max(0, test_start - purge_bars), min(n, test_end + embargo_bars))``.

    * **purge_bars** — bars immediately *before* the test fold that are
      excluded from training to prevent look-back indicator leakage.
    * **embargo_bars** — bars immediately *after* the test fold that are
      excluded to prevent label leakage from fast-moving post-test signals.
    """
    excluded_start = max(0, test_start - purge_bars)
    excluded_end   = min(n, test_end   + embargo_bars)
    return np.concatenate([
        np.arange(0, excluded_start),
        np.arange(excluded_end, n),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PurgedKFoldConfig:
    n_folds:        int = 5     # number of CV folds
    purge_bars:     int = 100   # bars before test fold excluded from training
    embargo_bars:   int = 10    # bars after  test fold excluded from training
    min_train_bars: int = 500   # skip fold if training set is too small
    min_test_trades: int = 5    # flag folds with too few test trades

    @classmethod
    def from_config(cls, cfg: dict) -> "PurgedKFoldConfig":
        """Build from a ``kfold:`` settings dict (all keys optional)."""
        return cls(
            n_folds=int(cfg.get("n_folds", 5)),
            purge_bars=int(cfg.get("purge_bars", 100)),
            embargo_bars=int(cfg.get("embargo_bars", 10)),
            min_train_bars=int(cfg.get("min_train_bars", 500)),
            min_test_trades=int(cfg.get("min_test_trades", 5)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KFoldResult:
    fold_num:         int
    test_start:       int            # first bar index of test fold
    test_end:         int            # last bar index of test fold (exclusive)
    train_bars_used:  int            # training bars after purge/embargo exclusion
    is_report:        BacktestReport  # in-sample (training bars) performance
    cv_report:        BacktestReport  # cross-validation (test fold) performance
    cv_score:         float           # metric value on test fold
    has_enough_trades: bool           # cv_report.total_trades >= min_test_trades


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KFoldReport:
    """
    Aggregated K-fold cross-validation results.

    Key diagnostics:
    * ``cv_mean_score`` — mean cross-validated metric; primary performance estimate.
    * ``cv_std_score``  — std of fold scores; lower = more consistent strategy.
    * ``stability_score`` — fraction of folds where ``cv_score > 0``.
    * ``overfitting_gap`` — ``mean_is_score - cv_mean_score``; large positive
      values indicate in-sample overfitting.
    """

    symbol:           str
    timeframe:        str
    n_folds:          int
    config:           PurgedKFoldConfig
    metric:           str
    fold_results:     list[KFoldResult] = field(default_factory=list)
    # Aggregate CV metrics
    cv_mean_score:    float = 0.0
    cv_std_score:     float = 0.0
    stability_score:  float = 0.0   # fraction of folds with cv_score > 0
    mean_is_score:    float = 0.0   # mean IS score (overfitting reference)
    overfitting_gap:  float = 0.0   # mean_is_score - cv_mean_score
    mean_cv_pnl:      float = 0.0
    mean_cv_win_rate: float = 0.0
    total_cv_trades:  int   = 0
    total_cv_pnl:     float = 0.0

    def __str__(self) -> str:  # noqa: D105
        sep  = "=" * 62
        thin = "-" * 62
        lines = [
            sep,
            "  PURGED K-FOLD CROSS-VALIDATION REPORT",
            sep,
            f"  Symbol          : {self.symbol}",
            f"  Timeframe       : {self.timeframe}",
            f"  Folds           : {self.n_folds}",
            f"  Metric          : {self.metric}",
            f"  Purge bars      : {self.config.purge_bars}",
            f"  Embargo bars    : {self.config.embargo_bars}",
            thin,
            "  CROSS-VALIDATION AGGREGATE",
            thin,
            f"  CV Mean Score   : {self.cv_mean_score:.3f}",
            f"  CV Std Score    : {self.cv_std_score:.3f}",
            f"  Stability Score : {self.stability_score:.1%}",
            f"  Mean IS Score   : {self.mean_is_score:.3f}",
            f"  Overfitting Gap : {self.overfitting_gap:+.3f}"
            + ("  ⚠ high" if self.overfitting_gap > 0.5 else ""),
            f"  Mean CV PnL     : {self.mean_cv_pnl:+.2f}",
            f"  Mean CV Win Rate: {self.mean_cv_win_rate:.1%}",
            f"  Total CV Trades : {self.total_cv_trades}",
            f"  Total CV PnL    : {self.total_cv_pnl:+.2f}",
        ]

        if self.fold_results:
            lines += [
                thin,
                "  PER-FOLD SUMMARY",
                thin,
                f"  {'Fold':>4}  {'IS Score':>9}  {'CV Score':>9}  "
                f"{'CV Trades':>9}  {'CV PnL':>10}  {'Pass':>4}",
            ]
            for fr in self.fold_results:
                flag = "✓" if fr.has_enough_trades else "⚠"
                lines.append(
                    f"  {fr.fold_num:>4}  {fr.is_report.sharpe_ratio:>9.3f}  "
                    f"{fr.cv_score:>9.3f}  "
                    f"{fr.cv_report.total_trades:>9}  "
                    f"{fr.cv_report.total_pnl:>+10.2f}  {flag:>4}"
                )

        lines.append(sep)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class PurgedKFoldEngine:
    """
    Purged K-fold cross-validation at the strategy level.

    For each of the K folds:
    * **IS run**  — backtest on all training bars (purge/embargo applied).
    * **CV run**  — backtest on the held-out test fold.

    Fresh ``SimulatedExchange``, ``RiskManager``, and strategy instances are
    created per run (same pattern as ``WalkForwardEngine._run_fold``).
    All imports inside ``_run_fold`` are deferred to avoid circular dependencies.

    The ``overfitting_gap = mean_is_score - cv_mean_score`` diagnoses whether
    the strategy is over-tuned to historical data.
    """

    def __init__(
        self,
        cfg:          dict[str, Any],
        kf_config:    PurgedKFoldConfig,
        metric:       str = "sharpe_ratio",
    ) -> None:
        self._cfg       = cfg
        self._kf_config = kf_config
        self._metric    = metric

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self,
        bars:      pd.DataFrame,
        symbol:    str,
        timeframe: str,
    ) -> KFoldReport:
        """
        Run purged K-fold CV on *bars* and return an aggregate ``KFoldReport``.

        Parameters
        ----------
        bars      : Full OHLCV DataFrame sorted chronologically.
        symbol    : Trading pair identifier (e.g. ``"BTC/USDT"``).
        timeframe : Bar timeframe string (e.g. ``"1h"``).
        """
        kfc = self._kf_config
        n   = len(bars)

        fold_ranges = _compute_fold_ranges(n, kfc.n_folds)
        fold_results: list[KFoldResult] = []

        for fold_num, (test_start, test_end) in enumerate(fold_ranges, start=1):
            train_idx = _get_train_indices(
                n, test_start, test_end,
                kfc.purge_bars, kfc.embargo_bars,
            )

            if len(train_idx) < kfc.min_train_bars:
                log.warning(
                    f"[KFold] Fold {fold_num}: only {len(train_idx)} training bars "
                    f"(min={kfc.min_train_bars}) — skipping fold"
                )
                continue

            train_bars = bars.iloc[train_idx].reset_index(drop=True)
            test_bars  = bars.iloc[test_start:test_end].reset_index(drop=True)

            log.info(
                f"[KFold] Fold {fold_num}/{kfc.n_folds}: "
                f"test=[{test_start}–{test_end - 1}] ({len(test_bars)} bars), "
                f"train={len(train_bars)} bars "
                f"(purge={kfc.purge_bars}, embargo={kfc.embargo_bars})"
            )

            is_report = self._run_fold(train_bars, symbol, timeframe)
            cv_report = self._run_fold(test_bars,  symbol, timeframe)

            cv_score    = float(getattr(cv_report, self._metric, 0.0))
            has_enough  = cv_report.total_trades >= kfc.min_test_trades

            if not has_enough:
                log.warning(
                    f"[KFold] Fold {fold_num}: only {cv_report.total_trades} "
                    f"CV trades (min={kfc.min_test_trades}) — data quality flag set"
                )

            fold_results.append(KFoldResult(
                fold_num=fold_num,
                test_start=test_start,
                test_end=test_end,
                train_bars_used=len(train_bars),
                is_report=is_report,
                cv_report=cv_report,
                cv_score=cv_score,
                has_enough_trades=has_enough,
            ))

        if not fold_results:
            log.warning(
                f"[KFold] No folds completed for {symbol} {timeframe} "
                f"(bars={n}, n_folds={kfc.n_folds}). Returning empty report."
            )
            return KFoldReport(
                symbol=symbol, timeframe=timeframe,
                n_folds=0, config=kfc, metric=self._metric,
            )

        # ── Aggregate ─────────────────────────────────────────────────────────
        cv_scores    = [fr.cv_score for fr in fold_results]
        is_scores    = [getattr(fr.is_report, self._metric, 0.0) for fr in fold_results]

        cv_mean_score    = float(np.mean(cv_scores))
        cv_std_score     = float(np.std(cv_scores))
        stability_score  = sum(1 for s in cv_scores if s > 0) / len(cv_scores)
        mean_is_score    = float(np.mean(is_scores))
        overfitting_gap  = mean_is_score - cv_mean_score
        mean_cv_pnl      = float(np.mean([fr.cv_report.total_pnl for fr in fold_results]))
        mean_cv_win_rate = float(np.mean([fr.cv_report.win_rate   for fr in fold_results]))
        total_cv_trades  = sum(fr.cv_report.total_trades for fr in fold_results)
        total_cv_pnl     = sum(fr.cv_report.total_pnl    for fr in fold_results)

        report = KFoldReport(
            symbol=symbol,
            timeframe=timeframe,
            n_folds=len(fold_results),
            config=kfc,
            metric=self._metric,
            fold_results=fold_results,
            cv_mean_score=cv_mean_score,
            cv_std_score=cv_std_score,
            stability_score=stability_score,
            mean_is_score=mean_is_score,
            overfitting_gap=overfitting_gap,
            mean_cv_pnl=mean_cv_pnl,
            mean_cv_win_rate=mean_cv_win_rate,
            total_cv_trades=total_cv_trades,
            total_cv_pnl=total_cv_pnl,
        )
        log.info(
            f"[KFold] Complete: {len(fold_results)} folds, "
            f"cv_mean={cv_mean_score:.3f} ± {cv_std_score:.3f}, "
            f"stability={stability_score:.1%}, "
            f"overfitting_gap={overfitting_gap:+.3f}"
        )
        return report

    # ── Private ───────────────────────────────────────────────────────────────

    def _run_fold(
        self,
        bars:      pd.DataFrame,
        symbol:    str,
        timeframe: str,
    ) -> BacktestReport:
        """
        Run one ``BacktestEngine`` on *bars* with completely fresh state.

        Deferred imports prevent circular dependencies (kfold imports
        from main; main imports from kfold).
        """
        from privateye.backtesting.engine import BacktestEngine
        from privateye.backtesting.simulator import SimulatedExchange
        from privateye.main import _build_advanced_risk, _build_strategies
        from privateye.risk.asset_filter import AssetFilter
        from privateye.risk.manager import RiskManager

        cfg    = self._cfg
        bt_cfg = cfg.get("backtesting", {})
        initial_capital = float(bt_cfg.get("initial_capital", 10000.0))

        # Fresh instances per fold — no state bleeds between runs
        strategies   = _build_strategies(cfg)
        _, exposure_monitor, black_swan_guard = _build_advanced_risk(cfg)

        asset_filter_raw = cfg.get("risk", {}).get("advanced", {}).get("asset_filter", {})
        asset_filter     = AssetFilter.from_config(asset_filter_raw)
        asset_filter_obj = asset_filter if asset_filter.enabled else None

        risk_manager = RiskManager(
            cfg.get("risk", {}),
            exposure_monitor=exposure_monitor,
            asset_filter=asset_filter_obj,
        )
        exchange = SimulatedExchange(
            initial_capital=initial_capital,
            fee_maker=float(bt_cfg.get("fee_maker", 0.001)),
            fee_taker=float(bt_cfg.get("fee_taker", 0.001)),
            slippage_pct=float(bt_cfg.get("slippage_pct", 0.0005)),
            max_fill_pct_of_volume=float(bt_cfg.get("max_fill_pct_of_volume", 0.30)),
        )
        engine = BacktestEngine(
            strategies=strategies,
            risk_manager=risk_manager,
            exchange=exchange,
            config=cfg,
            black_swan_guard=black_swan_guard,
        )
        return engine.run(bars, symbol, timeframe)
