"""
Regime Detector: classifies the current market into one of N_STATES regimes.

Uses GaussianHMM (hmmlearn) with a KMeans fallback.
Input: 5 volatility/trend features.  Output: (regime_int, probs_array).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from privateye.data.feature_extractor import extract_features
from privateye.models.base import BaseModel
from privateye.utils.logging import get_logger

log = get_logger()

# Column indices in the 50-feature matrix used for regime fitting
_REGIME_FEAT_INDICES = [
    18,  # atr_norm
    30,  # adx
    19,  # bb_width
    29,  # vol_zscore
    10,  # rsi_14
]
N_STATES = 4


def _try_hmm(n_states: int):
    try:
        from hmmlearn.hmm import GaussianHMM
        return GaussianHMM(n_components=n_states, covariance_type="full",
                           n_iter=100, random_state=42)
    except ImportError:
        return None


def _try_kmeans(n_states: int):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=n_states, n_init=10, random_state=42)


class RegimeDetector(BaseModel):
    """
    4-state market regime classifier (trending-bull, trending-bear,
    ranging-low-vol, ranging-high-vol).
    """

    def __init__(self, n_states: int = N_STATES, artifacts_dir: str | Path = "privateye/models/artifacts") -> None:
        super().__init__(artifacts_dir)
        self.n_states = n_states
        self._model = None
        self._use_hmm: bool = False
        self._centers: np.ndarray | None = None  # KMeans only

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, bars: pd.DataFrame, **kwargs) -> None:
        features = extract_features(bars)
        X = features[:, _REGIME_FEAT_INDICES]

        # Drop rows that were all-zero (warmup)
        valid = ~np.all(X == 0, axis=1)
        X_fit = X[valid]

        hmm = _try_hmm(self.n_states)
        if hmm is not None:
            try:
                hmm.fit(X_fit)
                self._model = hmm
                self._use_hmm = True
                log.info(f"[RegimeDetector] GaussianHMM fitted on {len(X_fit)} samples.")
            except Exception as exc:
                log.warning(f"[RegimeDetector] HMM failed ({exc}), falling back to KMeans.")
                hmm = None

        if hmm is None:
            km = _try_kmeans(self.n_states)
            km.fit(X_fit)
            self._model = km
            self._centers = km.cluster_centers_
            self._use_hmm = False
            log.info(f"[RegimeDetector] KMeans fitted on {len(X_fit)} samples.")

        self.is_fitted = True

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, bars: pd.DataFrame) -> tuple[str, float]:
        """Returns (direction='flat', confidence=max_regime_prob) — regimes don't emit direction."""
        self._require_fitted()
        _, probs = self.predict_regime(bars)
        return "flat", float(probs.max())

    def predict_regime(self, bars: pd.DataFrame) -> tuple[int, np.ndarray]:
        """Return (regime_int, probs_array shape (n_states,))."""
        self._require_fitted()
        features = extract_features(bars)
        x = features[-1:, _REGIME_FEAT_INDICES]  # (1, 5)

        if self._use_hmm:
            probs = self._model.predict_proba(x)[0]
            regime = int(np.argmax(probs))
        else:
            # KMeans: soft assignment via inverse distance
            dists = np.linalg.norm(self._centers - x[0], axis=1)
            inv   = 1.0 / (dists + 1e-9)
            probs = inv / inv.sum()
            regime = int(np.argmax(probs))

        return regime, probs.astype(np.float32)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        import joblib
        path = self.artifacts_dir / "regime_detector.pkl"
        meta = {"n_states": self.n_states, "use_hmm": self._use_hmm,
                "centers": self._centers.tolist() if self._centers is not None else None}
        joblib.dump({"model": self._model, "meta": meta}, path)
        log.info(f"[RegimeDetector] Saved to {path}")

    def load(self) -> None:
        import joblib
        path = self.artifacts_dir / "regime_detector.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        payload = joblib.load(path)
        self._model   = payload["model"]
        meta = payload["meta"]
        self.n_states = meta["n_states"]
        self._use_hmm = meta["use_hmm"]
        if meta["centers"] is not None:
            self._centers = np.array(meta["centers"])
        self.is_fitted = True
        log.info(f"[RegimeDetector] Loaded from {path}")
