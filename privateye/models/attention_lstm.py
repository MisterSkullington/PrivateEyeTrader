"""
AttentionLSTM: LSTM with multi-head self-attention for direction forecasting.

Architecture:
  Input  (batch, lookback, n_features)
  → LSTM(hidden=256, num_layers=2, dropout=0.2)
  → Self-attention (n_heads=4, embed_dim=256)   ← adds long-range context
  → LayerNorm(256)
  → Linear(256→64) + ReLU + Dropout(0.3)
  → Linear(64→2) → Softmax

The attention layer learns which timesteps in the lookback window are most
informative for the current prediction — addressing the primary weakness of
the plain LSTM on longer sequences.

Requires: torch>=2.2
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from privateye.data.feature_extractor import extract_features
from privateye.models.base import BaseModel
from privateye.models.training import build_lstm_sequences, make_direction_labels
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
            "torch is required for AttentionLSTM. Install with: pip install torch>=2.2"
        )


class _AttentionLSTMNet(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(
        self,
        input_size: int = 68,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_heads: int = 4,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is not available")
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})"
            )
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, seq, input_size)
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)  # self-attention
        # Take last timestep after add+norm residual
        last = self.norm(lstm_out[:, -1, :] + attn_out[:, -1, :])
        return self.head(last)


class AttentionLSTM(BaseModel):
    """
    LSTM + self-attention sequence forecaster.

    Same training pipeline as LSTMForecaster but with a MultiheadAttention
    layer between the LSTM output and classification head.
    """

    def __init__(
        self,
        lookback_bars: int = 60,
        label_horizon_bars: int = 5,
        label_threshold: float = 0.002,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_heads: int = 4,
        epochs: int = 50,
        patience: int = 10,
        batch_size: int = 64,
        lr: float = 1e-4,
        artifacts_dir: str | Path = "privateye/models/artifacts",
    ) -> None:
        _require_torch()
        super().__init__(artifacts_dir)
        self.lookback        = lookback_bars
        self.label_horizon   = label_horizon_bars
        self.label_threshold = label_threshold
        self.hidden_size     = hidden_size
        self.num_layers      = num_layers
        self.dropout         = dropout
        self.n_heads         = n_heads
        self.epochs          = epochs
        self.patience        = patience
        self.batch_size      = batch_size
        self.lr              = lr
        self._net = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._input_size: int = 68  # updated on first fit()

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, bars: pd.DataFrame, **kwargs) -> None:
        _require_torch()
        log.info(
            f"[AttentionLSTM] Training on {len(bars)} bars (device={self._device})"
        )

        features = extract_features(bars)
        labels   = make_direction_labels(bars, self.label_horizon, self.label_threshold)
        X, y     = build_lstm_sequences(features, labels, self.lookback)

        self._input_size = features.shape[-1]

        # Class weights to handle imbalance
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        pos_w = float(n_neg / (n_pos + 1e-9))
        weights = torch.tensor([1.0, pos_w], device=self._device)

        net = _AttentionLSTMNet(
            self._input_size,
            self.hidden_size,
            self.num_layers,
            self.dropout,
            self.n_heads,
        ).to(self._device)

        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss(weight=weights)

        split = int(len(X) * 0.8)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        best_val_loss = float("inf")
        patience_left = self.patience
        best_state = None

        for epoch in range(1, self.epochs + 1):
            net.train()
            perm = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb  = torch.from_numpy(X_tr[idx]).to(self._device)
                yb  = torch.from_numpy(y_tr[idx].astype(np.int64)).to(self._device)
                opt.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sched.step()

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
                    log.info(f"[AttentionLSTM] Early stop at epoch {epoch}.")
                    break

            if epoch % 10 == 0:
                log.info(f"[AttentionLSTM] epoch={epoch} val_loss={val_loss:.4f}")

        if best_state:
            net.load_state_dict(best_state)
        self._net     = net
        self.is_fitted = True
        log.info(
            f"[AttentionLSTM] Training complete. Best val_loss={best_val_loss:.4f}"
        )

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, bars: pd.DataFrame) -> tuple[str, float]:
        """Returns (direction, confidence) for the latest bar."""
        if not self.is_fitted:
            return "flat", 0.0
        _require_torch()

        features = extract_features(bars)
        if len(features) < self.lookback:
            return "flat", 0.0

        seq = features[-self.lookback:]  # (lookback, n_features)
        x = torch.from_numpy(seq[np.newaxis]).to(self._device)  # (1, lookback, n_features)

        self._net.eval()
        with torch.no_grad():
            logits = self._net(x)
            probs  = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        p_up   = float(probs[1])
        p_down = float(probs[0])
        if p_up > p_down:
            return "long", p_up
        else:
            return "flat", p_down

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        _require_torch()
        self._require_fitted()
        pt_path  = self.artifacts_dir / "attention_lstm.pt"
        cfg_path = self.artifacts_dir / "attention_lstm_config.json"
        torch.save(self._net.state_dict(), pt_path)
        cfg_path.write_text(
            json.dumps(
                {
                    "lookback":        self.lookback,
                    "label_horizon":   self.label_horizon,
                    "label_threshold": self.label_threshold,
                    "hidden_size":     self.hidden_size,
                    "num_layers":      self.num_layers,
                    "dropout":         self.dropout,
                    "n_heads":         self.n_heads,
                    "input_size":      self._input_size,
                },
                indent=2,
            )
        )
        log.info(f"[AttentionLSTM] Saved to {pt_path}")

    def load(self) -> None:
        _require_torch()
        pt_path  = self.artifacts_dir / "attention_lstm.pt"
        cfg_path = self.artifacts_dir / "attention_lstm_config.json"
        if not pt_path.exists():
            raise FileNotFoundError(f"Artifact not found: {pt_path}")
        cfg = json.loads(cfg_path.read_text())
        self.lookback        = cfg.get("lookback",        self.lookback)
        self.label_horizon   = cfg.get("label_horizon",   self.label_horizon)
        self.label_threshold = cfg.get("label_threshold", self.label_threshold)
        self.hidden_size     = cfg.get("hidden_size",     self.hidden_size)
        self.num_layers      = cfg.get("num_layers",      self.num_layers)
        self.dropout         = cfg.get("dropout",         self.dropout)
        self.n_heads         = cfg.get("n_heads",         self.n_heads)
        self._input_size     = cfg.get("input_size",      68)
        net = _AttentionLSTMNet(
            self._input_size,
            self.hidden_size,
            self.num_layers,
            self.dropout,
            self.n_heads,
        )
        net.load_state_dict(
            torch.load(pt_path, map_location=self._device, weights_only=True)
        )
        net.to(self._device)
        self._net     = net
        self.is_fitted = True
        log.info(f"[AttentionLSTM] Loaded from {pt_path}")
