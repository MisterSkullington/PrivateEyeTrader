"""
Strategy Parameter Optimizer.

Exhaustively searches a cartesian-product parameter grid, running an
independent ``BacktestEngine`` per combination (fresh ``SimulatedExchange``,
``RiskManager``, and strategy instances — zero state bleed), and ranks
combinations by a user-chosen metric.

Usage::

    from privateye.backtesting.optimizer import GridSearchOptimizer, OptimizationConfig

    opt_config = OptimizationConfig(
        strategy_name="directional",
        param_grid={"macd_fast": [8, 10, 12], "rsi_period": [10, 14]},
        metric="sharpe_ratio",
    )
    optimizer = GridSearchOptimizer(cfg, opt_config)
    report    = optimizer.run(bars)
    print(report)  # ranked table + best params
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from itertools import product as itertools_product
from typing import Any

import pandas as pd

from privateye.backtesting.metrics import BacktestReport
from privateye.utils.logging import get_logger

log = get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptimizationConfig:
    """
    Describes the parameter space and scoring rule for one grid-search run.

    Parameters
    ----------
    strategy_name : str
        Key inside ``cfg["strategies"]`` to override — ``"directional"``
        or ``"mean_reversion"``.
    param_grid : dict[str, list]
        Mapping of parameter name → list of candidate values.
        e.g. ``{"macd_fast": [8, 10, 12], "rsi_period": [10, 14]}``.
    metric : str
        Field name on ``BacktestReport`` used for scoring.
        Common values: ``"sharpe_ratio"``, ``"total_pnl"``, ``"calmar_ratio"``.
    min_trades : int
        Combinations producing fewer completed trades are flagged
        (``passed_min_trades=False``).  They still appear in results but
        are sorted after passing combinations.
    symbol : str
        Trading pair identifier passed to ``BacktestEngine.run()``.
    timeframe : str
        Bar timeframe string passed to ``BacktestEngine.run()``.
    """

    strategy_name: str            = "directional"
    param_grid:    dict[str, list] = field(default_factory=dict)
    metric:        str            = "sharpe_ratio"
    min_trades:    int            = 5
    symbol:        str            = "BTC/USDT"
    timeframe:     str            = "1h"

    @classmethod
    def from_config(cls, cfg: dict) -> "OptimizationConfig":
        """
        Build from a ``settings.yaml`` ``optimization:`` section.

        The ``param_grid`` key in *cfg* contains a nested dict keyed by
        strategy name; ``from_config`` selects the sub-dict matching
        ``strategy_name``.

        Example YAML::

            optimization:
              strategy_name: "directional"
              metric: "sharpe_ratio"
              min_trades: 5
              param_grid:
                directional:
                  macd_fast: [8, 10, 12]
                  rsi_period: [10, 14]
        """
        strategy_name = str(cfg.get("strategy_name", "directional"))
        raw_grid      = cfg.get("param_grid", {}).get(strategy_name, {})
        return cls(
            strategy_name=strategy_name,
            param_grid=dict(raw_grid),
            metric=str(cfg.get("metric", "sharpe_ratio")),
            min_trades=int(cfg.get("min_trades", 5)),
            symbol=str(cfg.get("symbol", "BTC/USDT")),
            timeframe=str(cfg.get("timeframe", "1h")),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Per-combination result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CombinationResult:
    """Result for one parameter combination."""

    params:            dict[str, Any]   # e.g. {"macd_fast": 8, "rsi_period": 10}
    report:            BacktestReport
    score:             float            # value of the chosen metric
    passed_min_trades: bool             # report.total_trades >= opt_config.min_trades


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptimizationReport:
    """
    Aggregated results from a complete grid-search run.

    ``results`` is sorted: ``passed_min_trades=True`` first, then by
    ``score`` descending within each group.
    ``best_params`` and ``best_score`` reflect the top-ranked passed result,
    or ``{}`` / ``nan`` if every combination failed ``min_trades``.
    """

    strategy_name:         str
    metric:                str
    min_trades:            int
    symbol:                str
    timeframe:             str
    n_combinations_total:  int
    n_combinations_passed: int
    results:               list[CombinationResult] = field(default_factory=list)
    best_params:           dict[str, Any]          = field(default_factory=dict)
    best_score:            float                   = float("nan")

    def __str__(self) -> str:  # noqa: D105
        sep  = "=" * 68
        thin = "-" * 68
        lines = [
            sep,
            "  OPTIMIZATION REPORT",
            sep,
            f"  Strategy      : {self.strategy_name}",
            f"  Metric        : {self.metric}",
            f"  Symbol        : {self.symbol}",
            f"  Timeframe     : {self.timeframe}",
            f"  Combinations  : {self.n_combinations_total} total, "
            f"{self.n_combinations_passed} passed (min_trades={self.min_trades})",
            thin,
        ]

        if self.best_params:
            params_str = ", ".join(f"{k}={v}" for k, v in self.best_params.items())
            lines += [
                f"  Best params   : {params_str}",
                f"  Best score    : {self.best_score:.4f}",
                thin,
            ]
        else:
            lines += ["  No combinations passed min_trades filter.", thin]

        if self.results:
            # Build dynamic column headers from param keys of the first result
            param_keys = list(self.results[0].params.keys())
            col_w = max(10, max((len(k) for k in param_keys), default=0) + 2)
            header_cols = "".join(f"  {k:<{col_w}}" for k in param_keys)
            lines.append(
                f"  {'RANK':>4}  {'SCORE':>8}  {header_cols}  "
                f"{'TRADES':>6}  {'PASS':>4}"
            )
            lines.append(thin)
            for rank, res in enumerate(self.results, start=1):
                vals = "".join(
                    f"  {str(v):<{col_w}}" for v in res.params.values()
                )
                flag = "✓" if res.passed_min_trades else "⚠"
                lines.append(
                    f"  {rank:>4}  {res.score:>8.4f}  {vals}  "
                    f"{res.report.total_trades:>6}  {flag:>4}"
                )

        lines.append(sep)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Engine
# ─────────────────────────────────────────────────────────────────────────────

class GridSearchOptimizer:
    """
    Grid-search parameter optimizer for PrivateEyeTrader strategies.

    For each cartesian-product combination in ``opt_config.param_grid``:

    1. Deep-copies the master cfg.
    2. Overrides ``cfg["strategies"][strategy_name]`` keys with the combination.
    3. Runs a fresh ``BacktestEngine`` (same isolated-instance pattern as
       ``WalkForwardEngine._run_fold``).
    4. Scores the result by ``opt_config.metric``.

    Results are sorted: ``passed_min_trades=True`` first, then by score
    descending within each group.
    """

    def __init__(self, cfg: dict[str, Any], opt_config: OptimizationConfig) -> None:
        self._cfg = cfg
        self._opt_config = opt_config

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self, bars: pd.DataFrame) -> OptimizationReport:
        """
        Run the full grid search and return a ranked ``OptimizationReport``.

        Parameters
        ----------
        bars : pd.DataFrame
            Full OHLCV DataFrame sorted chronologically.  Passed as-is to
            each ``BacktestEngine.run()`` call.
        """
        oc = self._opt_config

        # ── Generate combinations ─────────────────────────────────────────────
        keys        = list(oc.param_grid.keys())
        value_lists = [oc.param_grid[k] for k in keys]
        if not keys:
            log.warning(
                "[Optimizer] param_grid is empty — nothing to optimize. "
                "Returning empty report."
            )
            return OptimizationReport(
                strategy_name=oc.strategy_name,
                metric=oc.metric,
                min_trades=oc.min_trades,
                symbol=oc.symbol,
                timeframe=oc.timeframe,
                n_combinations_total=0,
                n_combinations_passed=0,
            )

        combinations = [
            dict(zip(keys, combo))
            for combo in itertools_product(*value_lists)
        ]
        n_total = len(combinations)
        log.info(
            f"[Optimizer] Starting grid search: strategy={oc.strategy_name}, "
            f"metric={oc.metric}, {n_total} combinations"
        )

        # ── Evaluate each combination ─────────────────────────────────────────
        results: list[CombinationResult] = []

        for i, params in enumerate(combinations, start=1):
            report = self._run_combination(params, bars)
            score  = float(getattr(report, oc.metric, 0.0))
            passed = report.total_trades >= oc.min_trades
            results.append(
                CombinationResult(
                    params=params,
                    report=report,
                    score=score,
                    passed_min_trades=passed,
                )
            )
            log.info(
                f"[Optimizer] {i}/{n_total}: {params} → "
                f"{oc.metric}={score:.4f}, trades={report.total_trades}"
                + (" ⚠ low-trades" if not passed else "")
            )

        # ── Sort: passed first, then by score descending ──────────────────────
        results.sort(key=lambda r: (not r.passed_min_trades, -r.score))

        passed_results = [r for r in results if r.passed_min_trades]
        best_params    = passed_results[0].params if passed_results else {}
        best_score     = passed_results[0].score  if passed_results else float("nan")

        opt_report = OptimizationReport(
            strategy_name=oc.strategy_name,
            metric=oc.metric,
            min_trades=oc.min_trades,
            symbol=oc.symbol,
            timeframe=oc.timeframe,
            n_combinations_total=n_total,
            n_combinations_passed=len(passed_results),
            results=results,
            best_params=best_params,
            best_score=best_score,
        )
        log.info(
            f"[Optimizer] Complete: {n_total} combinations, "
            f"{len(passed_results)} passed. "
            f"Best {oc.metric}={best_score:.4f} at {best_params}"
        )
        return opt_report

    # ── Private ───────────────────────────────────────────────────────────────

    def _run_combination(
        self,
        params: dict[str, Any],
        bars: pd.DataFrame,
    ) -> BacktestReport:
        """
        Run one BacktestEngine on *bars* with *params* applied to the strategy
        config.  Uses the same isolated-instance pattern as
        ``WalkForwardEngine._run_fold()``.

        The master cfg is **never mutated** — a deep copy is taken first.
        All imports are deferred inside this method to avoid circular dependencies
        (optimizer.py ← main.py ← optimizer.py via ``_build_strategies``).
        """
        from privateye.backtesting.engine import BacktestEngine
        from privateye.backtesting.simulator import SimulatedExchange
        from privateye.main import _build_advanced_risk, _build_strategies
        from privateye.risk.asset_filter import AssetFilter
        from privateye.risk.manager import RiskManager

        cfg    = copy.deepcopy(self._cfg)
        oc     = self._opt_config

        # Override the target strategy's params with this combination
        cfg["strategies"][oc.strategy_name].update(params)

        bt_cfg          = cfg.get("backtesting", {})
        initial_capital = float(bt_cfg.get("initial_capital", 10000.0))

        # Fresh instances — no state bleeds between combinations
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
        return engine.run(bars, oc.symbol, oc.timeframe)
