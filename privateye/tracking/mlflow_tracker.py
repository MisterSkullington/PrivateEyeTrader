"""
Optional MLflow experiment tracker.

Wraps mlflow so that every BacktestEngine run, walk-forward fold, and
K-fold result is logged as an MLflow run. Gracefully disables itself when
mlflow is not installed or ``enabled: false`` in config.

Config (under ``phase0.mlflow``)::

    mlflow:
      enabled: false
      tracking_uri: "sqlite:///mlflow.db"
      experiment_name: "privateye-baseline"

Usage::

    tracker = MLflowTracker(cfg["phase0"]["mlflow"])
    tracker.log_backtest(report, run_name="backtest_BTC_1h", params={"seed": 42})
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from privateye.utils.logging import get_logger

if TYPE_CHECKING:
    from privateye.backtesting.metrics import BacktestReport
    from privateye.backtesting.walk_forward import WalkForwardReport
    from privateye.backtesting.kfold import KFoldReport

log = get_logger()


class MLflowTracker:
    """Optional MLflow wrapper.

    All public methods are no-ops when ``enabled=False`` or when mlflow
    is not installed — callers never need to guard with ``if tracker:``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled: bool = bool(config.get("enabled", False))
        self._tracking_uri: str = config.get("tracking_uri", "sqlite:///mlflow.db")
        self._experiment_name: str = config.get("experiment_name", "privateye-baseline")
        self._mlflow: Any = None

        if self.enabled:
            try:
                import mlflow as _mlflow
                _mlflow.set_tracking_uri(self._tracking_uri)
                _mlflow.set_experiment(self._experiment_name)
                self._mlflow = _mlflow
                log.info(
                    f"[MLflowTracker] Enabled — experiment='{self._experiment_name}' "
                    f"uri={self._tracking_uri}"
                )
            except ImportError:
                log.warning(
                    "[MLflowTracker] mlflow not installed — tracking disabled. "
                    "Install with: pip install mlflow"
                )
                self.enabled = False

    # ── Public API ────────────────────────────────────────────────────────────

    def log_backtest(
        self,
        report: "BacktestReport",
        run_name: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Log all scalar BacktestReport fields as MLflow metrics."""
        if not self.enabled or self._mlflow is None:
            return
        with self._mlflow.start_run(run_name=run_name):
            if params:
                self._mlflow.log_params(params)
            d = report.to_dict()
            metrics = {
                k: float(v)
                for k, v in d.items()
                if isinstance(v, (int, float)) and k != "equity_curve"
            }
            self._mlflow.log_metrics(metrics)
        log.info(
            f"[MLflowTracker] Logged '{run_name}' → experiment '{self._experiment_name}'"
        )

    def log_walk_forward(
        self,
        report: "WalkForwardReport",
        run_name: str,
    ) -> None:
        """Log mean_oos_sharpe, stability_score, mean_efficiency_ratio."""
        if not self.enabled or self._mlflow is None:
            return
        with self._mlflow.start_run(run_name=run_name):
            self._mlflow.log_metrics({
                "mean_oos_sharpe":       float(report.mean_oos_sharpe),
                "std_oos_sharpe":        float(report.std_oos_sharpe),
                "stability_score":       float(report.stability_score),
                "mean_efficiency_ratio": float(report.mean_efficiency_ratio),
                "mean_oos_max_dd":       float(report.mean_oos_max_dd),
                "total_oos_trades":      float(report.total_oos_trades),
            })

    def log_kfold(
        self,
        report: "KFoldReport",
        run_name: str,
    ) -> None:
        """Log cv_mean_score, cv_std_score, overfitting_gap."""
        if not self.enabled or self._mlflow is None:
            return
        with self._mlflow.start_run(run_name=run_name):
            self._mlflow.log_metrics({
                "cv_mean_score":   float(report.cv_mean_score),
                "cv_std_score":    float(report.cv_std_score),
                "stability_score": float(report.stability_score),
                "overfitting_gap": float(report.overfitting_gap),
                "total_cv_trades": float(report.total_cv_trades),
            })

    @contextmanager
    def start_run(self, run_name: str):
        """Context manager for manual metric logging.

        Usage::
            with tracker.start_run("my-run") as run:
                tracker._mlflow.log_metric("custom", 1.23)
        """
        if not self.enabled or self._mlflow is None:
            yield None
            return
        with self._mlflow.start_run(run_name=run_name) as run:
            yield run
