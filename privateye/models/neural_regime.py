"""
Neural Regime Classifier: MLP 4-class regime classifier.

Replaces / supplements the HMM-based RegimeDetector with a learned
discriminative classifier. Uses the same 5 regime features as RegimeDetector
(atr_norm, adx, bb_width, vol_zscore, rsi_14).

Training: KMeans(k=4) assigns pseudo-labels on the training data, then the
MLP is trained supervised on those labels. Produces the same
(regime_int, probs[4]) output contract as RegimeDetector so downstream
regime gating is unchanged.

Architecture: Linear(5→64) + ReLU → LayerNorm → Linear(64→32) + ReLU
              → Linear(32→4) → Softmax

Requires: torch>=2.2, scikit-learn
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

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch is required for NeuralRegimeClassifier. "
            "Install with: pip install torch>=2.2"
        )


# Same 5 regime feature indices as RegimeDetector (column indices in 68-feature matrix)
_REGIME_FEAT_INDICES = [18, 30, 19, 29, 10]  # atr_norm, adx, bb_width, vol_zscore, rsi_14
_N_REGIME_FEATURES   = len(_REGIME_FEAT_INDICES)


class _RegimeMLP(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, n_states: int = 4, hidden: int = 64) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is not available")
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_N_REGIME_FEATURES, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_states),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class NeuralRegimeClassifier(BaseModel):
    """
    4-class MLP regime classifier.

    Uses the same 5 volatility/trend features as RegimeDetector to maintain
    compatibility with the existing regime-gate logic in FusionStrategy.

    Training steps:
      1. Extract 5 regime features and apply StandardScaler
      2. KMeans(k=4) assigns pseudo-labels
      3. MLP trained supervised on (scaled features, pseudo_labels)
    """

    def __init__(
        self,
        n_states: int = 4,
        hidden: int = 64,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 256,
        artifacts_dir: str | Path = "privateye/models/artifacts",
    ) -> None:
        _require_torch()
        super().__init__(artifacts_dir)
        self.n_states   = n_states
        self.hidden     = hidden
        self.epochs     = epochs
        self.lr         = lr
        self.batch_size = batch_size
        self._net = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Stored after fit() for inference-time scaling
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std:  np.ndarray | None = None

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, bars: pd.DataFrame, **kwargs) -> None:
        _require_torch()
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        log.info(f"[NeuralRegime] Training on {len(bars)} bars.")

        features = extract_features(bars)
        X = features[:, _REGIME_FEAT_INDICES].astype(np.float32)

        # Drop warmup rows (all-zero)
        valid = ~np.all(features == 0, axis=1)
        X_fit = X[valid]

        # Step 1: StandardScaler + KMeans pseudo-labels
        scaler    = StandardScaler()
        X_scaled  = scaler.fit_transform(X_fit).astype(np.float32)
        self._scaler_mean = scaler.mean_.astype(np.float32)
        self._scaler_std  = scaler.scale_.astype(np.float32)

        km = KMeans(n_clusters=self.n_states, n_init=10, random_state=42)
        pseudo_labels = km.fit_predict(X_scaled).astype(np.int64)

        # Step 2: Supervised MLP training
        net = _RegimeMLP(self.n_states, self.hidden).to(self._device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.from_numpy(X_scaled).to(self._device)
        y_t = torch.from_numpy(pseudo_labels).to(self._device)

        for epoch in range(1, self.epochs + 1):
            net.train()
            perm = torch.randperm(len(X_t), device=self._device)
            for i in range(0, len(X_t), self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad()
                loss = criterion(net(X_t[idx]), y_t[idx])
                loss.backward()
                opt.step()

        self._net      = net
        self.is_fitted = True
        log.info("[NeuralRegime] Training complete.")

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_regime(self, bars: pd.DataFrame) -> tuple[int, np.ndarray]:
        """Return (regime_int, probs_array shape (n_states,))."""
        if not self.is_fitted:
            return 0, np.full(self.n_states, 1.0 / self.n_states, dtype=np.float32)

        _require_torch()
        features = extract_features(bars)
        x = features[-1:, _REGIME_FEAT_INDICES].astype(np.float32)

        # Apply stored scaler
        x_scaled = (x - self._scaler_mean) / (self._scaler_std + 1e-9)
        x_t = torch.from_numpy(x_scaled).to(self._device)

        self._net.eval()
        with torch.no_grad():
            logits = self._net(x_t)
            probs  = torch.softmax(logits, dim=-1)[0].cpu().numpy().astype(np.float32)

        return int(np.argmax(probs)), probs

    def predict(self, bars: pd.DataFrame) -> tuple[str, float]:
        """BaseModel compliance — regime classifiers don't emit direction."""
        _, probs = self.predict_regime(bars)
        return "flat", float(probs.max())

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        _require_torch()
        self._require_fitted()
        pt_path  = self.artifacts_dir / "neural_regime.pt"
        cfg_path = self.artifacts_dir / "neural_regime_config.json"
        torch.save(
            {
                "net":          self._net.state_dict(),
                "scaler_mean":  self._scaler_mean.tolist(),
                "scaler_std":   self._scaler_std.tolist(),
            },
            pt_path,
        )
        cfg_path.write_text(
            json.dumps({"n_states": self.n_states, "hidden": self.hidden}, indent=2)
        )
        log.info(f"[NeuralRegime] Saved to {pt_path}")

    def load(self) -> None:
        _require_torch()
        pt_path  = self.artifacts_dir / "neural_regime.pt"
        cfg_path = self.artifacts_dir / "neural_regime_config.json"
        if not pt_path.exists():
            raise FileNotFoundError(f"Artifact not found: {pt_path}")
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            self.n_states = cfg.get("n_states", self.n_states)
            self.hidden   = cfg.get("hidden",   self.hidden)
        state = torch.load(pt_path, map_location=self._device, weights_only=True)
        net   = _RegimeMLP(self.n_states, self.hidden)
        net.load_state_dict(state["net"])
        net.to(self._device)
        self._net          = net
        self._scaler_mean  = np.array(state["scaler_mean"], dtype=np.float32)
        self._scaler_std   = np.array(state["scaler_std"],  dtype=np.float32)
        self.is_fitted     = True
        log.info(f"[NeuralRegime] Loaded from {pt_path}")
