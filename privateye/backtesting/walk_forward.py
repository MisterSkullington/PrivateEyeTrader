"""
Walk-Forward Backtesting Engine.

Partitions historical bars into rolling IS/OOS windows using the existing
``walk_forward_splits()`` splitter, runs an independent ``BacktestEngine``
on each window, and aggregates out-of-sample results into a
``WalkForwardReport`` with stability score and efficiency ratio.

Usage::

    from privateye.backtesting.walk_forward import WalkForwardConfig, WalkForwardEngine

    wf_config = WalkForwardConfig(train_bars=6048, test_bars=1512)
    engine    = WalkForwardEngine(cfg, wf_config)
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
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardConfig:
    train_bars:      int = 6048   # ~252 trading days at 1h
    test_bars:       int = 1512   # ~63 trading days at 1h
    step_bars:       int = 1512   # advance by one test window each fold
    min_train_bars:  int = 500    # skip fold if IS window is shorter
    min_test_trades: int = 5      # flag OOS folds with too few trades

    @classmethod
    def from_config(cls, cfg: dict) -> "WalkForwardConfig":
        """Build from a ``walk_forward:`` settings dict (all keys optional)."""
        return cls(
            train_bars=int(cfg.get("train_bars", 6048)),
            test_bars=int(cfg.get("test_bars", 1512)),
            step_bars=int(cfg.get("step_bars", cfg.get("test_bars", 1512))),
            min_train_bars=int(cfg.get("min_train_bars", 500)),
            min_test_trades=int(cfg.get("min_test_trades", 5)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Per-fold result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardFoldResult:
    fold_num:          int
    is_start:          int          # first bar index in IS window
    is_end:            int          # last bar index in IS window (exclusive)
    oos_start:         int          # first bar index in OOS window
    oos_end:           int          # last bar index in OOS window (exclusive)
    is_report:         BacktestReport
    oos_report:        BacktestReport
    efficiency_ratio:  float        # oos_sharpe / is_sharpe; 0.0 when IS Sharpe = 0
    has_enough_trades: bool         # oos_report.total_trades >= min_test_trades


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardReport:
    symbol:               str
    timeframe:            str
    n_folds:              int
    config:               WalkForwardConfig
    fold_results:         list[WalkForwardFoldResult] = field(default_factory=list)
    # Aggregate OOS metrics
    mean_oos_sharpe:      float = 0.0
    std_oos_sharpe:       float = 0.0
    stability_score:      float = 0.0   # fraction of folds with OOS Sharpe > 0
    mean_efficiency_ratio: float = 0.0
    mean_oos_max_dd:      float = 0.0
    mean_oos_win_rate:    float = 0.0
    total_oos_trades:     int   = 0
    total_oos_pnl:        float = 0.0

    def __str__(self) -> str:  # noqa: D105
        sep = "=" * 60
        thin = "-" * 60
        lines = [
            sep,
            "  WALK-FORWARD REPORT",
            sep,
            f"  Symbol          : {self.symbol}",
            f"  Timeframe       : {self.timeframe}",
            f"  Folds           : {self.n_folds}",
            f"  Train bars      : {self.config.train_bars}",
            f"  Test bars       : {self.config.test_bars}",
            f"  Step bars       : {self.config.step_bars}",
            thin,
            "  OUT-OF-SAMPLE AGGREGATE",
            thin,
            f"  Mean OOS Sharpe : {self.mean_oos_sharpe:.3f}",
            f"  Std  OOS Sharpe : {self.std_oos_sharpe:.3f}",
            f"  Stability Score : {self.stability_score:.1%}",
            f"  Mean Efficiency : {self.mean_efficiency_ratio:.3f}",
            f"  Mean OOS Max DD : {self.mean_oos_max_dd:.1%}",
            f"  Mean OOS WinRate: {self.mean_oos_win_rate:.1%}",
            f"  Total OOS Trades: {self.total_oos_trades}",
            f"  Total OOS PnL   : {self.total_oos_pnl:+.2f}",
        ]

        if self.fold_results:
            lines += [thin, "  PER-FOLD SUMMARY", thin,
                      f"  {'Fold':>4}  {'IS Sharpe':>10}  {'OOS Sharpe':>10}  "
                      f"{'Efficiency':>10}  {'OOS Trades':>10}  {'OOS PnL':>10}"]
            for fr in self.fold_results:
                flag = "" if fr.has_enough_trades else " ⚠"
                lines.append(
                    f"  {fr.fold_num:>4}  {fr.is_report.sharpe_ratio:>10.3f}  "
                    f"{fr.oos_report.sharpe_ratio:>10.3f}  "
                    f"{fr.efficiency_ratio:>10.3f}  "
                    f"{fr.oos_report.total_trades:>10}  "
                    f"{fr.oos_report.total_pnl:>+10.2f}{flag}"
                )

        lines.append(sep)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardEngine:
    """
    Orchestrates walk-forward cross-validation at the strategy level.

    For each fold:
    * IS  (in-sample)  slice → fresh ``BacktestEngine`` run
    * OOS (out-of-sample) slice → fresh ``BacktestEngine`` run
    * Efficiency ratio = OOS Sharpe / IS Sharpe

    Fresh ``SimulatedExchange``, ``RiskManager``, and strategy instances are
    created per fold to guarantee zero state bleed.
    """

    def __init__(self, cfg: dict[str, Any], wf_config: WalkForwardConfig) -> None:
        self._cfg = cfg
        self._wf_config = wf_config

    # ── Public ────────────────────────────────────────────────────────────────

    def run(
        self,
        bars: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> WalkForwardReport:
        """
        Run walk-forward backtesting on *bars* and return an aggregate report.

        Parameters
        ----------
        bars:      Full OHLCV DataFrame sorted chronologically.
        symbol:    Trading pair identifier (e.g. ``"BTC/USDT"``).
        timeframe: Bar timeframe string (e.g. ``"1h"``).
        """
        from privateye.models.training import walk_forward_splits

        wfc = self._wf_config
        folds = walk_forward_splits(
            n=len(bars),
            train_window=wfc.train_bars,
            test_window=wfc.test_bars,
            step=wfc.step_bars,
            min_train=wfc.min_train_bars,
        )

        if not folds:
            log.warning(
                f"[WalkForward] No folds generated for {symbol} {timeframe} "
                f"(bars={len(bars)}, train={wfc.train_bars}, test={wfc.test_bars}). "
                "Returning empty report."
            )
            return WalkForwardReport(
                symbol=symbol,
                timeframe=timeframe,
                n_folds=0,
                config=wfc,
            )

        fold_results: list[WalkForwardFoldResult] = []

        for wf_fold in folds:
            fold_num = wf_fold.fold_num
            is_idx   = wf_fold.train_idx
            oos_idx  = wf_fold.test_idx

            is_bars  = bars.iloc[is_idx].reset_index(drop=True)
            oos_bars = bars.iloc[oos_idx].reset_index(drop=True)

            log.info(
                f"[WalkForward] Fold {fold_num}: "
                f"IS [{is_idx[0]}–{is_idx[-1]}] ({len(is_bars)} bars), "
                f"OOS [{oos_idx[0]}–{oos_idx[-1]}] ({len(oos_bars)} bars)"
            )

            is_report  = self._run_fold(is_bars,  symbol, timeframe)
            oos_report = self._run_fold(oos_bars, symbol, timeframe)

            is_sharpe  = is_report.sharpe_ratio
            oos_sharpe = oos_report.sharpe_ratio
            efficiency = oos_sharpe / is_sharpe if is_sharpe != 0 else 0.0

            has_enough = oos_report.total_trades >= wfc.min_test_trades
            if not has_enough:
                log.warning(
                    f"[WalkForward] Fold {fold_num}: only {oos_report.total_trades} OOS trades "
                    f"(min={wfc.min_test_trades}) — data quality flag set"
                )

            fold_results.append(WalkForwardFoldResult(
                fold_num=fold_num,
                is_start=int(is_idx[0]),
                is_end=int(is_idx[-1]) + 1,
                oos_start=int(oos_idx[0]),
                oos_end=int(oos_idx[-1]) + 1,
                is_report=is_report,
                oos_report=oos_report,
                efficiency_ratio=float(efficiency),
                has_enough_trades=has_enough,
            ))

        # ── Aggregate ─────────────────────────────────────────────────────────
        oos_sharpes = [fr.oos_report.sharpe_ratio for fr in fold_results]
        mean_oos_sharpe       = float(np.mean(oos_sharpes))
        std_oos_sharpe        = float(np.std(oos_sharpes))
        stability_score       = sum(1 for s in oos_sharpes if s > 0) / len(oos_sharpes)
        mean_efficiency_ratio = float(np.mean([fr.efficiency_ratio for fr in fold_results]))
        mean_oos_max_dd       = float(np.mean([fr.oos_report.max_drawdown_pct for fr in fold_results]))
        mean_oos_win_rate     = float(np.mean([fr.oos_report.win_rate for fr in fold_results]))
        total_oos_trades      = sum(fr.oos_report.total_trades for fr in fold_results)
        total_oos_pnl         = sum(fr.oos_report.total_pnl for fr in fold_results)

        report = WalkForwardReport(
            symbol=symbol,
            timeframe=timeframe,
            n_folds=len(fold_results),
            config=wfc,
            fold_results=fold_results,
            mean_oos_sharpe=mean_oos_sharpe,
            std_oos_sharpe=std_oos_sharpe,
            stability_score=stability_score,
            mean_efficiency_ratio=mean_efficiency_ratio,
            mean_oos_max_dd=mean_oos_max_dd,
            mean_oos_win_rate=mean_oos_win_rate,
            total_oos_trades=total_oos_trades,
            total_oos_pnl=total_oos_pnl,
        )
        log.info(
            f"[WalkForward] Complete: {len(fold_results)} folds, "
            f"mean OOS Sharpe={mean_oos_sharpe:.3f}, "
            f"stability={stability_score:.1%}"
        )
        return report

    # ── Private ───────────────────────────────────────────────────────────────

    def _run_fold(
        self,
        bars: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> BacktestReport:
        """
        Run one BacktestEngine on *bars* with completely fresh state.

        Deferred imports prevent circular dependencies (walk_forward imports
        from main; main imports from walk_forward).
        """
        from privateye.backtesting.engine import BacktestEngine
        from privateye.backtesting.simulator import SimulatedExchange
        from privateye.main import _build_advanced_risk, _build_strategies
        from privateye.risk.manager import RiskManager

        cfg = self._cfg
        bt_cfg = cfg.get("backtesting", {})
        initial_capital = float(bt_cfg.get("initial_capital", 10000.0))

        # Fresh instances per fold — no state bleeds between windows
        strategies   = _build_strategies(cfg)
        _, exposure_monitor, black_swan_guard = _build_advanced_risk(cfg)
        asset_filter_raw = cfg.get("risk", {}).get("advanced", {}).get("asset_filter", {})
        from privateye.risk.asset_filter import AssetFilter
        asset_filter = AssetFilter.from_config(asset_filter_raw)
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
