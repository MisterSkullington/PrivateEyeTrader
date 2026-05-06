"""
ModelEnsemble: loads all Phase-2 artifacts and provides unified per-bar inference.

Lazy loads each model only on first call; gracefully degrades if an artifact
is missing (model contributes 0.0 confidence and "flat" direction).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from privateye.utils.logging import get_logger

log = get_logger()


class ModelEnsemble:
    """
    Thin wrapper that aggregates all ML model predictions for a single bar.

    Returns a dict keyed by model name:
        {
            "regime":   (regime_int, probs_array),
            "lstm":     (direction, confidence),
            "gbm":      (gate_prob: float),
            "rl":       (direction, confidence),
        }
    """

    def __init__(self, artifacts_dir: str | Path = "privateye/models/artifacts") -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self._regime = None
        self._lstm   = None
        self._gbm    = None
        self._rl     = None
        self._loaded = False

    def load_all(self) -> None:
        """Load all available model artifacts. Missing artifacts are skipped."""
        self._regime = self._try_load_regime()
        self._lstm   = self._try_load_lstm()
        self._gbm    = self._try_load_gbm()
        self._rl     = self._try_load_rl()
        self._loaded = True
        loaded = [k for k, v in [("regime", self._regime), ("lstm", self._lstm),
                                  ("gbm", self._gbm), ("rl", self._rl)] if v]
        log.info(f"[ModelEnsemble] Loaded models: {loaded}")

    def predict(self, bars: pd.DataFrame) -> dict:
        """Run all loaded models on the latest bar of `bars`."""
        if not self._loaded:
            self.load_all()

        result: dict = {}

        if self._regime is not None:
            try:
                result["regime"] = self._regime.predict_regime(bars)
            except Exception as e:
                log.debug(f"[ModelEnsemble] regime failed: {e}")
                result["regime"] = (0, [0.25, 0.25, 0.25, 0.25])

        if self._lstm is not None:
            try:
                result["lstm"] = self._lstm.predict(bars)
            except Exception as e:
                log.debug(f"[ModelEnsemble] lstm failed: {e}")
                result["lstm"] = ("flat", 0.0)

        if self._gbm is not None:
            try:
                result["gbm"] = self._gbm.predict_gate_prob(bars)
            except Exception as e:
                log.debug(f"[ModelEnsemble] gbm failed: {e}")
                result["gbm"] = 0.5

        if self._rl is not None:
            try:
                result["rl"] = self._rl.predict(bars)
            except Exception as e:
                log.debug(f"[ModelEnsemble] rl failed: {e}")
                result["rl"] = ("flat", 0.0)

        # SHAP feature importances (XAI) — non-blocking; None when unavailable
        shap_vals: dict[str, float] | None = None
        if self._gbm is not None and self._gbm.is_fitted:
            try:
                raw = self._gbm.get_shap_explanation(bars)
                shap_vals = raw if raw else None
            except Exception as e:
                log.debug(f"[ModelEnsemble] SHAP explanation failed: {e}")
        result["shap"] = shap_vals

        return result

    # ── Lazy loaders ─────────────────────────────────────────────────────────

    def _try_load_regime(self):
        try:
            from privateye.models.regime_detector import RegimeDetector
            m = RegimeDetector(artifacts_dir=self.artifacts_dir)
            m.load()
            return m
        except Exception as e:
            log.debug(f"[ModelEnsemble] Could not load regime detector: {e}")
            return None

    def _try_load_lstm(self):
        try:
            from privateye.models.lstm_forecaster import LSTMForecaster
            m = LSTMForecaster(artifacts_dir=self.artifacts_dir)
            m.load()
            return m
        except Exception as e:
            log.debug(f"[ModelEnsemble] Could not load LSTM: {e}")
            return None

    def _try_load_gbm(self):
        try:
            from privateye.models.gbm_classifier import GBMClassifier
            m = GBMClassifier(artifacts_dir=self.artifacts_dir)
            m.load()
            return m
        except Exception as e:
            log.debug(f"[ModelEnsemble] Could not load GBM: {e}")
            return None

    def _try_load_rl(self):
        try:
            from privateye.models.rl_policy import RLPolicy
            m = RLPolicy(artifacts_dir=self.artifacts_dir)
            m.load()
            return m
        except Exception as e:
            log.debug(f"[ModelEnsemble] Could not load RL policy: {e}")
            return None

    @property
    def has_regime(self) -> bool:
        return self._regime is not None

    @property
    def has_lstm(self) -> bool:
        return self._lstm is not None

    @property
    def has_gbm(self) -> bool:
        return self._gbm is not None

    @property
    def has_rl(self) -> bool:
        return self._rl is not None
