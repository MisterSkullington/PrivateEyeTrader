"""
LSTM Forecaster: predicts next-bar direction from a 60-bar lookback window.

Architecture: Input (batch, 60, 50) → LSTM(256, 2) → LayerNorm →
              Linear(256→64) + ReLU + Dropout → Linear(64→2) → Softmax

Requires: torch>=2.2
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from privateye.data.feature_extractor import extract_features
from privateye.models.base import BaseModel
from privateye.models.training import (
    PurgedKFold, build_lstm_sequences, make_direction_labels,
    walk_forward_splits,
)
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
        raise ImportError("torch is required for LSTMForecaster. Install with: pip install torch>=2.2")


class _LSTMNet(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, input_size: int = 50, hidden_size: int = 256, num_layers: int = 2, dropout: float = 0.2):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is not available")
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])  # last timestep
        return self.head(out)


class LSTMForecaster(BaseModel):
    """
    Binary direction forecaster using a stacked LSTM.
    Predicts P(up) for the next `label_horizon` bars.
    """

    def __init__(
        self,
        lookback: int = 60,
        label_horizon: int = 5,
        label_threshold: float = 0.002,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        epochs: int = 50,
        patience: int = 10,
        batch_size: int = 64,
        lr: float = 1e-4,
        artifacts_dir: str | Path = "privateye/models/artifacts",
    ) -> None:
        _require_torch()
        super().__init__(artifacts_dir)
        self.lookback        = lookback
        self.label_horizon   = label_horizon
        self.label_threshold = label_threshold
        self.hidden_size     = hidden_size
        self.num_layers      = num_layers
        self.dropout         = dropout
        self.epochs          = epochs
        self.patience        = patience
        self.batch_size      = batch_size
        self.lr              = lr
        self._net: _LSTMNet | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, bars: pd.DataFrame, **kwargs) -> None:
        _require_torch()
        log.info(f"[LSTMForecaster] Training on {len(bars)} bars (device={self._device})")

        features = extract_features(bars)
        labels   = make_direction_labels(bars, self.label_horizon, self.label_threshold)
        X, y     = build_lstm_sequences(features, labels, self.lookback)

        # Class weights to handle imbalance
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        pos_w = (n_neg / (n_pos + 1e-9))
        weights = torch.tensor([1.0, float(pos_w)], device=self._device)

        net = _LSTMNet(features.shape[-1], self.hidden_size, self.num_layers, self.dropout).to(self._device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss(weight=weights)

        # Simple 80/20 chronological val split
        split = int(len(X) * 0.8)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        best_val_loss = float("inf")
        patience_left = self.patience
        best_state = None

        for epoch in range(1, self.epochs + 1):
            net.train()
            perm = torch.randperm(len(X_tr))
            epoch_loss = 0.0
            for i in range(0, len(X_tr), self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb  = torch.from_numpy(X_tr[idx]).to(self._device)
                yb  = torch.from_numpy(y_tr[idx].astype(np.int64)).to(self._device)
                opt.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                epoch_loss += loss.item()
            sched.step()

            # Validation
            net.eval()
            with torch.no_grad():
                xv = torch.from_numpy(X_val).to(self._device)
                yv = torch.from_numpy(y_val.astype(np.int64)).to(self._device)
                val_loss = criterion(net(xv), yv).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    log.info(f"[LSTMForecaster] Early stop at epoch {epoch}.")
                    break

            if epoch % 10 == 0:
                log.info(f"[LSTMForecaster] epoch={epoch} val_loss={val_loss:.4f}")

        if best_state:
            net.load_state_dict(best_state)
        self._net = net
        self.is_fitted = True
        log.info(f"[LSTMForecaster] Training complete. Best val_loss={best_val_loss:.4f}")

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, bars: pd.DataFrame) -> tuple[str, float]:
        """Returns (direction, confidence) for the latest bar."""
        self._require_fitted()
        _require_torch()

        features = extract_features(bars)
        if len(features) < self.lookback:
            return "flat", 0.0

        seq = features[-self.lookback:]  # (lookback, 50)
        x = torch.from_numpy(seq[np.newaxis]).to(self._device)  # (1, lookback, 50)

        self._net.eval()
        with torch.no_grad():
            logits = self._net(x)
            probs  = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        p_up   = float(probs[1])
        p_down = float(probs[0])
        if p_up > p_down:
            return "long",  float(p_up)
        else:
            return "short", float(p_down)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        _require_torch()
        self._require_fitted()
        pt_path   = self.artifacts_dir / "lstm_forecaster.pt"
        cfg_path  = self.artifacts_dir / "lstm_config.json"
        # Derive input_size from the trained network weights
        input_size = self._net.lstm.input_size
        torch.save(self._net.state_dict(), pt_path)
        cfg = dict(lookback=self.lookback, label_horizon=self.label_horizon,
                   label_threshold=self.label_threshold, hidden_size=self.hidden_size,
                   num_layers=self.num_layers, dropout=self.dropout,
                   input_size=input_size)
        cfg_path.write_text(json.dumps(cfg, indent=2))
        log.info(f"[LSTMForecaster] Saved to {pt_path}")

    def load(self) -> None:
        _require_torch()
        pt_path  = self.artifacts_dir / "lstm_forecaster.pt"
        cfg_path = self.artifacts_dir / "lstm_config.json"
        if not pt_path.exists():
            raise FileNotFoundError(f"Artifact not found: {pt_path}")
        cfg = json.loads(cfg_path.read_text())
        self.lookback        = cfg["lookback"]
        self.label_horizon   = cfg["label_horizon"]
        self.label_threshold = cfg["label_threshold"]
        self.hidden_size     = cfg["hidden_size"]
        self.num_layers      = cfg["num_layers"]
        self.dropout         = cfg["dropout"]
        input_size = cfg.get("input_size", 68)  # default 68 for Phase 3; 50 for old checkpoints
        net = _LSTMNet(input_size, self.hidden_size, self.num_layers, self.dropout)
        net.load_state_dict(torch.load(pt_path, map_location=self._device))
        net.to(self._device)
        self._net = net
        self.is_fitted = True
        log.info(f"[LSTMForecaster] Loaded from {pt_path}")
