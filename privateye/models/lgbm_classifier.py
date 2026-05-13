"""
LightGBM Classifier: predicts whether the next trade entry will be profitable.

Drop-in parallel to GBMClassifier — same interface, same label construction,
different tree algorithm (leaf-wise growth, typically 3–5× faster on CPU).

Requires: lightgbm>=4.0
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from privateye.data.feature_extractor import FEATURE_NAMES, extract_features
from privateye.models.base import BaseModel
from privateye.models.training import make_direction_labels
from privateye.utils.logging import get_logger

log = get_logger()

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False


def _require_lgb() -> None:
    if not _LGB_AVAILABLE:
        raise ImportError(
            "lightgbm is required. Install with: pip install lightgbm>=4.0"
        )


class LGBMClassifier(BaseModel):
    """
    LightGBM binary gate classifier.

    Same training pipeline as GBMClassifier but using LightGBM's leaf-wise
    gradient boosting. Typically trains 3–5× faster than XGBoost on CPU and
    often achieves comparable or better OOS log-loss due to leaf-wise growth.

    Returns P(trade is profitable) in [0, 1].
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        gate_threshold: float = 0.45,
        label_horizon: int = 5,
        label_threshold: float = 0.002,
        artifacts_dir: str | Path = "privateye/models/artifacts",
    ) -> None:
        _require_lgb()
        super().__init__(artifacts_dir)
        self.n_estimators    = n_estimators
        self.max_depth       = max_depth
        self.learning_rate   = learning_rate
        self.num_leaves      = num_leaves
        self.gate_threshold  = gate_threshold
        self.label_horizon   = label_horizon
        self.label_threshold = label_threshold
        self._model = None  # lgb.LGBMClassifier

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, bars: pd.DataFrame, **kwargs) -> None:
        _require_lgb()
        log.info(f"[LGBMClassifier] Training on {len(bars)} bars.")

        features = extract_features(bars)
        labels   = make_direction_labels(bars, self.label_horizon, self.label_threshold)

        # Drop warmup rows (where all features are 0) and last label_horizon rows
        valid = ~np.all(features == 0, axis=1)
        valid[-self.label_horizon:] = False
        X, y = features[valid], labels[valid]

        # Train/val split (80/20, chronological)
        split = int(len(X) * 0.8)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        self._model = model
        self.is_fitted = True
        log.info(
            f"[LGBMClassifier] Training complete. "
            f"Best iter={model.best_iteration_}."
        )

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, bars: pd.DataFrame) -> tuple[str, float]:
        """
        Returns (direction, gate_prob).
        LightGBM doesn't predict direction — it gates other signals.
        """
        self._require_fitted()
        prob = self.predict_gate_prob(bars)
        direction = "long" if prob >= self.gate_threshold else "flat"
        return direction, float(prob)

    def predict_gate_prob(self, bars: pd.DataFrame) -> float:
        """Return P(trade is profitable) for the current bar. Range [0, 1]."""
        self._require_fitted()
        _require_lgb()
        features = extract_features(bars)
        x = features[-1:, :]
        return float(self._model.predict_proba(x)[0, 1])

    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Return top-n feature names by LightGBM importance score."""
        self._require_fitted()
        scores = self._model.feature_importances_
        # Trim to actual feature count (handles 68-feature and future 86-feature inputs)
        names = FEATURE_NAMES[: len(scores)]
        ranked = sorted(
            ((name, float(score)) for name, score in zip(names, scores)),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:n]

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        import joblib

        _require_lgb()
        self._require_fitted()
        model_path = self.artifacts_dir / "lgbm_classifier.pkl"
        meta_path  = self.artifacts_dir / "lgbm_config.json"
        joblib.dump(self._model, model_path)
        meta_path.write_text(
            json.dumps(
                {
                    "gate_threshold":  self.gate_threshold,
                    "label_horizon":   self.label_horizon,
                    "label_threshold": self.label_threshold,
                    "num_leaves":      self.num_leaves,
                },
                indent=2,
            )
        )
        log.info(f"[LGBMClassifier] Saved to {model_path}")

    def load(self) -> None:
        import joblib

        _require_lgb()
        model_path = self.artifacts_dir / "lgbm_classifier.pkl"
        meta_path  = self.artifacts_dir / "lgbm_config.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Artifact not found: {model_path}")
        self._model = joblib.load(model_path)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.gate_threshold  = meta.get("gate_threshold",  self.gate_threshold)
            self.label_horizon   = meta.get("label_horizon",   self.label_horizon)
            self.label_threshold = meta.get("label_threshold", self.label_threshold)
            self.num_leaves      = meta.get("num_leaves",      self.num_leaves)
        self.is_fitted = True
        log.info(f"[LGBMClassifier] Loaded from {model_path}")
