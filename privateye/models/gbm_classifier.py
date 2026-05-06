"""
GBM Classifier: predicts whether the next trade entry will be profitable.
Acts as a confidence gate in FusionStrategy — low-probability signals are vetoed.

Uses XGBoost with SHAP feature importance for XAI.
Requires: xgboost>=2.0, shap>=0.45
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from privateye.data.feature_extractor import FEATURE_NAMES, extract_features
from privateye.models.base import BaseModel
from privateye.models.training import make_direction_labels, walk_forward_splits
from privateye.utils.logging import get_logger

log = get_logger()

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    import shap as shap_lib
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


def _require_xgb() -> None:
    if not _XGB_AVAILABLE:
        raise ImportError("xgboost is required. Install with: pip install xgboost>=2.0")


class GBMClassifier(BaseModel):
    """
    XGBoost classifier that answers: 'Will this trade be profitable?'

    Uses walk-forward CV to generate training labels, ensuring the labels
    come from the same bar-level data but with a temporal offset.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        gate_threshold: float = 0.45,
        label_horizon: int = 5,
        label_threshold: float = 0.002,
        artifacts_dir: str | Path = "privateye/models/artifacts",
    ) -> None:
        _require_xgb()
        super().__init__(artifacts_dir)
        self.n_estimators    = n_estimators
        self.max_depth       = max_depth
        self.learning_rate   = learning_rate
        self.gate_threshold  = gate_threshold
        self.label_horizon   = label_horizon
        self.label_threshold = label_threshold
        self._model: xgb.XGBClassifier | None = None
        self._shap_values: np.ndarray | None = None

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, bars: pd.DataFrame, **kwargs) -> None:
        _require_xgb()
        log.info(f"[GBMClassifier] Training on {len(bars)} bars.")

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

        model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            early_stopping_rounds=50,
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self._model = model
        self.is_fitted = True

        # Compute SHAP values on val set for XAI
        if _SHAP_AVAILABLE:
            try:
                explainer = shap_lib.TreeExplainer(model)
                self._shap_values = explainer.shap_values(X_val[:200])
                log.info("[GBMClassifier] SHAP values computed.")
            except Exception as e:
                log.warning(f"[GBMClassifier] SHAP failed: {e}")

        log.info(f"[GBMClassifier] Training complete. Best iter={model.best_iteration}.")

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, bars: pd.DataFrame) -> tuple[str, float]:
        """
        Returns (direction='flat', confidence=P(profitable)).
        GBM doesn't predict direction — it gates other signals.
        Use predict_proba() directly in FusionStrategy.
        """
        self._require_fitted()
        prob = self.predict_gate_prob(bars)
        direction = "long" if prob >= self.gate_threshold else "flat"
        return direction, float(prob)

    def predict_gate_prob(self, bars: pd.DataFrame) -> float:
        """Return P(trade is profitable) for the current bar. Range [0, 1]."""
        self._require_fitted()
        _require_xgb()
        features = extract_features(bars)
        x = features[-1:, :]  # (1, 50)
        return float(self._model.predict_proba(x)[0, 1])

    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Return top-n feature names by XGBoost importance score."""
        self._require_fitted()
        scores = self._model.feature_importances_
        ranked = sorted(zip(FEATURE_NAMES, scores), key=lambda x: x[1], reverse=True)
        return ranked[:n]

    def get_shap_explanation(self, bars: pd.DataFrame) -> dict[str, float]:
        """Return SHAP values for the latest bar (requires shap library)."""
        if not _SHAP_AVAILABLE or self._shap_values is None:
            return {}
        self._require_fitted()
        features = extract_features(bars)
        x = features[-1:]
        try:
            explainer = shap_lib.TreeExplainer(self._model)
            sv = explainer.shap_values(x)[0]
            return {name: float(val) for name, val in zip(FEATURE_NAMES, sv)}
        except Exception:
            return {}

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        import joblib
        _require_xgb()
        self._require_fitted()
        model_path = self.artifacts_dir / "gbm_classifier.pkl"
        meta_path  = self.artifacts_dir / "gbm_feature_names.json"
        joblib.dump(self._model, model_path)
        meta_path.write_text(json.dumps({
            "feature_names": FEATURE_NAMES,
            "gate_threshold": self.gate_threshold,
            "label_horizon": self.label_horizon,
            "label_threshold": self.label_threshold,
        }, indent=2))
        log.info(f"[GBMClassifier] Saved to {model_path}")

    def load(self) -> None:
        import joblib
        _require_xgb()
        model_path = self.artifacts_dir / "gbm_classifier.pkl"
        meta_path  = self.artifacts_dir / "gbm_feature_names.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Artifact not found: {model_path}")
        self._model = joblib.load(model_path)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.gate_threshold  = meta.get("gate_threshold", self.gate_threshold)
            self.label_horizon   = meta.get("label_horizon", self.label_horizon)
            self.label_threshold = meta.get("label_threshold", self.label_threshold)
        self.is_fitted = True
        log.info(f"[GBMClassifier] Loaded from {model_path}")
