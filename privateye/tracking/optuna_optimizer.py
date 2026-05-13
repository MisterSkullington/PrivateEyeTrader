"""
Optuna-backed strategy parameter optimiser.

Drop-in replacement for ``GridSearchOptimizer`` that uses Bayesian (TPE)
search instead of exhaustive grid search — more efficient for large
parameter spaces.

Requires ``optuna`` (``pip install optuna`` or ``pip install privateye[phase0]``).

Usage::

    from privateye.tracking.optuna_optimizer import OptunaOptimizer
    from privateye.backtesting.optimizer import OptimizationConfig

    opt_config = OptimizationConfig(
        strategy_name="directional",
        param_grid={"macd_fast": [8, 10, 12, 14], "rsi_period": [10, 12, 14, 16, 18]},
        metric="sharpe_ratio",
        min_trades=5,
    )
    optimizer = OptunaOptimizer(cfg, opt_config)
    report = optimizer.run(bars, n_trials=50)   # 50 Bayesian trials vs 20 exhaustive
    print(report.best_params)
"""
from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from privateye.utils.logging import get_logger

log = get_logger()


class OptunaOptimizer:
    """Bayesian (TPE) strategy parameter optimiser powered by Optuna.

    Same ``OptimizationConfig`` / ``OptimizationReport`` interface as
    ``GridSearchOptimizer`` — fully interchangeable.

    Args:
        cfg:        Full settings dict (deep-copied per trial — never mutated).
        opt_config: ``OptimizationConfig`` describing the search space.

    Raises:
        ImportError: when ``optuna`` is not installed.
    """

    def __init__(self, cfg: dict[str, Any], opt_config: Any) -> None:
        try:
            import optuna as _optuna
            # Suppress optuna's default verbose INFO logs
            _optuna.logging.set_verbosity(_optuna.logging.WARNING)
            self._optuna = _optuna
        except ImportError as exc:
            raise ImportError(
                "optuna is required for OptunaOptimizer. "
                "Install with: pip install optuna  or  pip install privateye[phase0]"
            ) from exc
        self._cfg = cfg
        self._opt_config = opt_config

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        bars: pd.DataFrame,
        n_trials: int = 100,
    ) -> Any:
        """Run ``n_trials`` Bayesian trials and return an ``OptimizationReport``.

        The best trial corresponds to the maximum value of
        ``opt_config.metric``.
        """
        from privateye.backtesting.optimizer import CombinationResult, OptimizationReport

        if not self._opt_config.param_grid:
            return OptimizationReport(
                strategy_name=self._opt_config.strategy_name,
                metric=self._opt_config.metric,
                min_trades=self._opt_config.min_trades,
                symbol=self._opt_config.symbol,
                timeframe=self._opt_config.timeframe,
                n_combinations_total=0,
                n_combinations_passed=0,
                results=[],
                best_params={},
                best_score=float("nan"),
            )

        results: list[CombinationResult] = []

        def objective(trial: Any) -> float:
            params = {
                key: trial.suggest_categorical(key, vals)
                for key, vals in self._opt_config.param_grid.items()
            }
            report = self._run_combination(params, bars)
            score  = float(getattr(report, self._opt_config.metric, 0.0))
            passed = report.total_trades >= self._opt_config.min_trades
            results.append(CombinationResult(
                params=params, report=report, score=score, passed_min_trades=passed,
            ))
            log.debug(
                f"[OptunaOptimizer] Trial {trial.number}: "
                f"{params} → {self._opt_config.metric}={score:.4f}"
            )
            return score

        study = self._optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

        # Sort: passed first, then by score descending
        results.sort(key=lambda r: (not r.passed_min_trades, -r.score))
        passed_results = [r for r in results if r.passed_min_trades]

        return OptimizationReport(
            strategy_name=self._opt_config.strategy_name,
            metric=self._opt_config.metric,
            min_trades=self._opt_config.min_trades,
            symbol=self._opt_config.symbol,
            timeframe=self._opt_config.timeframe,
            n_combinations_total=len(results),
            n_combinations_passed=len(passed_results),
            results=results,
            best_params=results[0].params if passed_results else {},
            best_score=results[0].score if passed_results else float("nan"),
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_combination(
        self, params: dict[str, Any], bars: pd.DataFrame
    ) -> Any:
        """Run a fresh BacktestEngine with ``params`` overriding the strategy config.

        Mirrors the pattern in ``GridSearchOptimizer._run_combination``:
        deep-copy cfg, override strategy params, create fresh instances.
        """
        from privateye.backtesting.engine import BacktestEngine
        from privateye.backtesting.simulator import SimulatedExchange
        from privateye.risk.manager import RiskManager

        cfg = copy.deepcopy(self._cfg)
        sname = self._opt_config.strategy_name
        cfg.setdefault("strategies", {}).setdefault(sname, {}).update(params)

        from privateye.main import _build_strategies, _build_advanced_risk
        strategies = _build_strategies(cfg)
        exposure_monitor, asset_filter, black_swan_guard = _build_advanced_risk(cfg)
        rm = RiskManager(
            cfg.get("risk", {}),
            exposure_monitor=exposure_monitor,
            asset_filter=asset_filter,
        )
        bt_cfg = cfg.get("backtesting", {})
        sim = SimulatedExchange(
            initial_capital=float(bt_cfg.get("initial_capital", 10_000.0)),
            fee_maker=float(bt_cfg.get("fee_maker", 0.001)),
            fee_taker=float(bt_cfg.get("fee_taker", 0.001)),
            slippage_pct=float(bt_cfg.get("slippage_pct", 0.0005)),
            max_fill_pct_of_volume=float(bt_cfg.get("max_fill_pct_of_volume", 0.30)),
        )
        return BacktestEngine(strategies, rm, sim, cfg,
                              black_swan_guard=black_swan_guard).run(
            bars, self._opt_config.symbol, self._opt_config.timeframe
        )
