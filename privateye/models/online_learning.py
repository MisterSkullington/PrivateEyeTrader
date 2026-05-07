"""
Online learning: incremental model retraining triggered by new bar arrivals.

On every MARKET_DATA event the latest bar is appended to a rolling buffer.
When `retrain_every_bars` new bars have accumulated, a background retrain is
scheduled for each configured model.  Before promotion the new model is
compared against the current one on a held-out validation slice; if the new
model regresses by more than `max_val_loss_regression` the update is rejected
and the checkpoint is restored.

Config keys (under robustness.online_learning):
  enabled                  (bool,  default False)
  buffer_bars              (int,   default 2000)   rolling bar buffer size
  retrain_every_bars       (int,   default 200)    new bars between retrains
  models                   (list,  default ["gbm","lstm","regime"])
  validation_gate          (bool,  default True)
  max_val_loss_regression  (float, default 0.10)
"""
from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from privateye.core.types import DataSnapshot, EventType
from privateye.models.checkpoint import load_checkpoint, prune_old_checkpoints, save_checkpoint
from privateye.utils.logging import get_logger

log = get_logger()

_MIN_BUFFER_ROWS = 50   # absolute minimum bars before any retrain attempt


class OnlineLearner:
    """
    Subscribes to MARKET_DATA events and incrementally retrains configured
    models on a rolling bar buffer.

    Usage::

        learner = OnlineLearner(bus, ensemble, cfg["robustness"]["online_learning"])
        bus.subscribe(EventType.MARKET_DATA, learner.on_market_data)
    """

    def __init__(
        self,
        bus: Any,
        ensemble: Any,
        config: dict[str, Any],
        artifacts_dir: str | Path = "privateye/models/artifacts",
    ) -> None:
        self.enabled: bool = config.get("enabled", False)
        self._buffer_size: int = config.get("buffer_bars", 2000)
        self._retrain_every: int = config.get("retrain_every_bars", 200)
        self._models_to_update: list[str] = config.get("models", ["gbm", "lstm", "regime"])
        self._validation_gate: bool = config.get("validation_gate", True)
        self._max_regression: float = config.get("max_val_loss_regression", 0.10)

        self._bus = bus
        self._ensemble = ensemble
        self._artifacts_dir = Path(artifacts_dir)
        self._checkpoint_dir = self._artifacts_dir / "checkpoints"

        self._buffer: deque[dict] = deque(maxlen=self._buffer_size)
        self._bars_since_retrain: int = 0
        self._retrain_lock: asyncio.Lock = asyncio.Lock()
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)

    # ── Public interface ──────────────────────────────────────────────────────

    async def on_market_data(self, payload: DataSnapshot) -> None:
        """Append latest bar to the rolling buffer; schedule retrain when due.

        Phase 13 (H-10): the counter is reset *before* spawning the retrain
        task. Previously the reset happened inside ``_run_retrain`` after the
        task was already created, leaving a window where two consecutive bars
        could both pass the trigger check and queue duplicate retrains.
        """
        if not self.enabled:
            return
        if not hasattr(payload, "bars") or len(payload.bars) == 0:
            return

        self._buffer.append(payload.bars.iloc[-1].to_dict())
        self._bars_since_retrain += 1

        # Heartbeat: visibility into buffer growth without per-bar spam.
        # Fires at 25/50/75/100% of the retrain window.
        heartbeat = max(1, self._retrain_every // 4)
        if self._bars_since_retrain % heartbeat == 0:
            log.debug(
                f"[OnlineLearner] Buffer {len(self._buffer)}/{self._buffer_size}, "
                f"retrain in {max(0, self._retrain_every - self._bars_since_retrain)} bars"
            )

        if (
            self._bars_since_retrain >= self._retrain_every
            and not self._retrain_lock.locked()
        ):
            # H-10: reset *before* task creation; prevents duplicate retrains
            # if a bar arrives between this check and the lock acquisition.
            self._bars_since_retrain = 0
            asyncio.create_task(self._run_retrain())

    # ── Retrain orchestrator ──────────────────────────────────────────────────

    async def _run_retrain(self) -> None:
        """Acquire lock, retrain all configured models, publish MODEL_UPDATED event.

        Phase 13 (H-10): the counter reset moved up to ``on_market_data`` so it
        happens atomically with task creation. This method no longer touches it.
        """
        if self._retrain_lock.locked():
            return
        async with self._retrain_lock:
            if len(self._buffer) < _MIN_BUFFER_ROWS:
                log.warning(
                    f"[OnlineLearner] Buffer only {len(self._buffer)} rows — "
                    "skipping retrain (need ≥ 50)"
                )
                return

            bars = _buffer_to_df(self._buffer)
            loop = asyncio.get_event_loop()
            models_updated: list[str] = []
            models_skipped: list[str] = []

            for model_name in self._models_to_update:
                retrain_fn = getattr(self, f"_retrain_{model_name}", None)
                if retrain_fn is None:
                    log.debug(f"[OnlineLearner] No retrain function for '{model_name}'")
                    continue

                model = self._get_model(model_name)
                if model is None:
                    log.warning(
                        f"[OnlineLearner] '{model_name}' not loaded in ensemble — skipping"
                    )
                    models_skipped.append(model_name)
                    continue

                try:
                    # Checkpoint current state before any mutation
                    save_checkpoint(model, len(self._buffer), self._checkpoint_dir)
                    prune_old_checkpoints(
                        model.__class__.__name__, self._checkpoint_dir, keep_n=3
                    )

                    # Run blocking training in thread pool
                    new_loss, old_loss = await loop.run_in_executor(
                        self._executor, retrain_fn, bars
                    )

                    # Validation gate: reject if new model regresses
                    if self._validation_gate and old_loss != float("inf"):
                        threshold = old_loss * (1.0 + self._max_regression)
                        if new_loss > threshold:
                            log.warning(
                                f"[OnlineLearner] {model_name} update rejected "
                                f"(val_loss regression: {old_loss:.4f} → {new_loss:.4f})"
                            )
                            _try_restore(model, self._checkpoint_dir, model_name)
                            models_skipped.append(model_name)
                            continue

                    # Promote: persist new in-memory state to disk
                    model.save()
                    log.info(
                        f"[OnlineLearner] {model_name} updated "
                        f"(val_loss: {old_loss:.4f} → {new_loss:.4f}, "
                        f"buffer={len(self._buffer)} bars)"
                    )
                    models_updated.append(model_name)

                except Exception as exc:
                    log.warning(f"[OnlineLearner] {model_name} retrain failed: {exc}")
                    models_skipped.append(model_name)
                    _try_restore(model, self._checkpoint_dir, model_name)

            await self._bus.publish(
                EventType.MODEL_UPDATED,
                {
                    "models_updated": models_updated,
                    "models_skipped": models_skipped,
                    "bars_seen": len(self._buffer),
                },
            )

    # ── Model accessor ────────────────────────────────────────────────────────

    def _get_model(self, name: str) -> Any:
        return {
            "gbm":    self._ensemble._gbm,
            "lstm":   self._ensemble._lstm,
            "regime": self._ensemble._regime,
            "rl":     self._ensemble._rl,
        }.get(name)

    # ── Per-model retrain functions (run in thread pool) ──────────────────────

    def _retrain_gbm(self, bars: pd.DataFrame) -> tuple[float, float]:
        """
        Full GBM retrain on rolling buffer.
        Returns (new_val_loss, old_val_loss) — lower logloss is better.
        """
        try:
            import xgboost as xgb
            from sklearn.metrics import log_loss as sk_log_loss
        except ImportError as exc:
            raise ImportError(
                "xgboost and scikit-learn are required for GBM online learning"
            ) from exc

        from privateye.data.feature_extractor import extract_features
        from privateye.models.training import make_direction_labels

        gbm = self._ensemble._gbm
        features = extract_features(bars)
        labels = make_direction_labels(bars, gbm.label_horizon, gbm.label_threshold)

        valid = ~np.all(features == 0, axis=1)
        valid[-gbm.label_horizon:] = False
        X, y = features[valid], labels[valid]

        if len(X) < _MIN_BUFFER_ROWS:
            return 0.0, float("inf")  # not enough valid rows → always promote

        split = int(len(X) * 0.8)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        # Old validation loss (before mutating the model)
        old_loss: float = float("inf")
        if gbm.is_fitted and gbm._model is not None:
            try:
                old_proba = gbm._model.predict_proba(X_val)
                old_loss = float(sk_log_loss(y_val, old_proba))
            except Exception:
                pass

        # Full retrain
        # Phase 13 (M-2): use_label_encoder was removed in xgboost 2.1+
        new_clf = xgb.XGBClassifier(
            n_estimators=gbm.n_estimators,
            max_depth=gbm.max_depth,
            learning_rate=gbm.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            early_stopping_rounds=50,
            random_state=42,
            verbosity=0,
        )
        new_clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        new_proba = new_clf.predict_proba(X_val)
        new_loss = float(sk_log_loss(y_val, new_proba))

        # Mutate model in-place; save() called by orchestrator on promotion
        gbm._model = new_clf
        gbm.is_fitted = True
        return new_loss, old_loss

    def _retrain_lstm(self, bars: pd.DataFrame) -> tuple[float, float]:
        """
        Fast LSTM retrain (5 epochs) on rolling buffer.
        Returns (new_val_loss, old_val_loss) — lower cross-entropy is better.
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError("torch is required for LSTM online learning") from exc

        from privateye.data.feature_extractor import extract_features
        from privateye.models.lstm_forecaster import _LSTMNet
        from privateye.models.training import build_lstm_sequences, make_direction_labels

        lstm = self._ensemble._lstm
        features = extract_features(bars)
        labels = make_direction_labels(bars, lstm.label_horizon, lstm.label_threshold)
        X, y = build_lstm_sequences(features, labels, lstm.lookback)

        if len(X) < 20:
            return 0.0, float("inf")

        split = int(len(X) * 0.8)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        device = lstm._device
        n_pos = float(y.sum())
        n_neg = float(len(y) - n_pos)
        weights = torch.tensor([1.0, n_neg / (n_pos + 1e-9)], device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        xv = torch.from_numpy(X_val).to(device)
        yv = torch.from_numpy(y_val.astype(np.int64)).to(device)

        # Old validation loss
        old_loss: float = float("inf")
        if lstm.is_fitted and lstm._net is not None:
            lstm._net.eval()
            with torch.no_grad():
                try:
                    old_loss = float(criterion(lstm._net(xv), yv).item())
                except Exception:
                    pass

        # Build and train new net (5 epochs — fast online update)
        input_size = features.shape[-1]
        new_net = _LSTMNet(
            input_size, lstm.hidden_size, lstm.num_layers, lstm.dropout
        ).to(device)
        opt = torch.optim.AdamW(new_net.parameters(), lr=lstm.lr, weight_decay=1e-5)

        best_val_loss = float("inf")
        best_state: dict | None = None

        for _ in range(5):
            new_net.train()
            perm = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), lstm.batch_size):
                idx = perm[i: i + lstm.batch_size]
                xb = torch.from_numpy(X_tr[idx]).to(device)
                yb = torch.from_numpy(y_tr[idx].astype(np.int64)).to(device)
                opt.zero_grad()
                loss = criterion(new_net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(new_net.parameters(), 1.0)
                opt.step()

            new_net.eval()
            with torch.no_grad():
                val_loss = float(criterion(new_net(xv), yv).item())
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in new_net.state_dict().items()}

        if best_state is not None:
            new_net.load_state_dict(best_state)

        # Mutate model in-place; save() called by orchestrator on promotion
        lstm._net = new_net.to(device)
        lstm.is_fitted = True
        return best_val_loss, old_loss

    def _retrain_regime(self, bars: pd.DataFrame) -> tuple[float, float]:
        """
        Full regime-detector retrain on rolling buffer.
        Returns (new_val_loss, old_val_loss) — lower negative log-likelihood is better.
        KMeans always promotes (no natural loss metric → returns (0.0, inf)).
        """
        from privateye.data.feature_extractor import extract_features
        from privateye.models.regime_detector import (
            _REGIME_FEAT_INDICES,
            _try_hmm,
            _try_kmeans,
        )

        regime = self._ensemble._regime
        features = extract_features(bars)
        X = features[:, _REGIME_FEAT_INDICES]
        valid = ~np.all(X == 0, axis=1)
        X_fit = X[valid]

        if len(X_fit) < 20:
            return 0.0, float("inf")

        split = int(len(X_fit) * 0.8)
        X_tr, X_val = X_fit[:split], X_fit[split:]

        # Old validation loss (HMM only; KMeans always promotes)
        old_loss: float = float("inf")
        if regime.is_fitted and regime._use_hmm and regime._model is not None:
            try:
                old_loss = float(-regime._model.score(X_val))
            except Exception:
                pass

        # Retrain with HMM first, KMeans as fallback
        new_model: Any = None
        new_use_hmm = False
        new_centers = None
        new_loss: float = 0.0

        hmm = _try_hmm(regime.n_states)
        if hmm is not None:
            try:
                hmm.fit(X_tr)
                new_model = hmm
                new_use_hmm = True
                try:
                    new_loss = float(-hmm.score(X_val))
                except Exception:
                    new_loss = 0.0
            except Exception as exc:
                log.warning(
                    f"[OnlineLearner] HMM retrain failed ({exc}), falling back to KMeans"
                )

        if new_model is None:
            km = _try_kmeans(regime.n_states)
            km.fit(X_tr)
            new_model = km
            new_use_hmm = False
            new_centers = km.cluster_centers_
            new_loss = 0.0
            old_loss = float("inf")  # KMeans: always promote

        # Mutate model in-place; save() called by orchestrator on promotion
        regime._model = new_model
        regime._use_hmm = new_use_hmm
        regime._centers = new_centers
        regime.is_fitted = True
        return new_loss, old_loss


# ── Helpers ───────────────────────────────────────────────────────────────────

def _buffer_to_df(buf: deque) -> pd.DataFrame:
    """Convert the rolling bar buffer to a DataFrame with a UTC timestamp column."""
    df = pd.DataFrame(list(buf))
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def _try_restore(model: Any, checkpoint_dir: Path, model_name: str) -> None:
    """Best-effort checkpoint restore — logs warning on failure, never raises."""
    try:
        load_checkpoint(model, checkpoint_dir, latest=True)
    except Exception as exc:
        log.warning(
            f"[OnlineLearner] Could not restore {model_name} from checkpoint: {exc}"
        )
