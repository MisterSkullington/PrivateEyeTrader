"""
StackingEnsemble: OOF-based stacking meta-learner.

Learns how to combine GBM, LGBM, LSTM, AttentionLSTM, and Regime predictions
via an out-of-fold LightGBM meta-learner. Replaces FusionStrategy's hand-tuned
Sharpe-weighted vote with a learned combination that adapts to market context.

Training (fit):
  1. Split bars into n_folds time-series folds (sklearn TimeSeriesSplit)
  2. For each fold: train lightweight base model instances on the train split,
     predict on the val split → collect OOF meta-feature predictions
  3. Build 11-feature meta-matrix from OOF predictions
  4. Build meta-labels: forward return > label_threshold → 1, else 0
  5. Train LightGBM meta-learner on (meta_features, meta_labels)

Inference (predict):
  Assemble live predictions from ModelEnsemble.predict() output,
  build the 11-feature vector, run through meta-learner.
  Returns ("long"|"flat", confidence).

Falls back to ("flat", 0.0) when not fitted (graceful degradation).

Requires: lightgbm>=4.0
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from privateye.utils.logging import get_logger

log = get_logger()

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

# ── Meta-feature layout (11 columns) ─────────────────────────────────────────
# [0]   gbm_gate_prob         float [0,1]
# [1]   lgbm_gate_prob        float [0,1]
# [2]   lstm_conf             float [0,1]
# [3]   attn_lstm_conf        float [0,1]
# [4]   lstm_dir_long         0 or 1
# [5]   lstm_dir_flat         0 or 1
# [6]   attn_lstm_dir_long    0 or 1
# [7]   regime_prob_0         float [0,1]
# [8]   regime_prob_1         float [0,1]
# [9]   regime_prob_2         float [0,1]
# [10]  regime_prob_3         float [0,1]
_N_META_FEATURES = 11

# Compact params for per-fold OOF base model instances (fast training)
_OOF_GBM_PARAMS  = {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.1}
_OOF_LGBM_PARAMS = {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.1, "num_leaves": 15}
_OOF_LSTM_PARAMS = {"epochs": 5, "patience": 3, "hidden_size": 64, "num_layers": 1}
_OOF_ATTN_PARAMS = {"epochs": 5, "patience": 3, "hidden_size": 64, "num_layers": 1, "n_heads": 4}
_OOF_NR_PARAMS   = {"epochs": 10, "hidden": 32}


class StackingEnsemble:
    """
    OOF stacking meta-learner that learns to weight base model outputs.

    Designed for use inside FusionStrategy: when is_fitted=True, on_data()
    bypasses the hand-tuned Sharpe-weighted vote and uses the learned
    meta-learner decision instead.
    """

    def __init__(
        self,
        n_folds: int = 5,
        label_horizon: int = 5,
        label_threshold: float = 0.002,
        meta_n_estimators: int = 100,
        artifacts_dir: str | Path = "privateye/models/artifacts",
        base_model_cfgs: dict[str, Any] | None = None,
    ) -> None:
        self.n_folds          = n_folds
        self.label_horizon    = label_horizon
        self.label_threshold  = label_threshold
        self.meta_n_estimators = meta_n_estimators
        self.artifacts_dir    = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._base_model_cfgs = base_model_cfgs or {}
        self._meta_learner    = None  # lgb.LGBMClassifier
        self._attribution:    dict[str, float] = {}  # Phase 4
        self.is_fitted        = False

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, bars: pd.DataFrame, cfg: dict[str, Any] | None = None) -> None:
        """
        Train the stacking meta-learner via OOF base model predictions.

        Parameters
        ----------
        bars : pd.DataFrame
            Full historical OHLCV bars used to generate OOF predictions.
        cfg : dict | None
            Optional full config dict; if provided, overrides base_model_cfgs.
        """
        if not _LGB_AVAILABLE:
            log.warning("[StackingEnsemble] lightgbm not installed — skipping fit().")
            return

        if cfg is not None:
            self._base_model_cfgs = cfg.get("ml", {})

        log.info(
            f"[StackingEnsemble] Starting OOF fit on {len(bars)} bars, "
            f"{self.n_folds} folds."
        )

        from sklearn.model_selection import TimeSeriesSplit

        from privateye.models.training import make_direction_labels

        tss = TimeSeriesSplit(n_splits=self.n_folds, gap=self.label_horizon)
        n   = len(bars)

        meta_X_rows: list[np.ndarray] = []
        meta_y_rows: list[int]        = []
        labels = make_direction_labels(bars, self.label_horizon, self.label_threshold)

        for fold_idx, (train_idx, val_idx) in enumerate(tss.split(np.arange(n))):
            log.info(
                f"[StackingEnsemble] Fold {fold_idx + 1}/{self.n_folds}: "
                f"train={len(train_idx)} val={len(val_idx)}"
            )
            train_bars = bars.iloc[train_idx].reset_index(drop=True)
            val_bars   = bars.iloc[val_idx].reset_index(drop=True)

            if len(train_bars) < 100 or len(val_bars) < 20:
                log.debug(f"[StackingEnsemble] Fold {fold_idx + 1} skipped — too few bars.")
                continue

            # Train per-fold base models
            fold_models = self._train_fold_models(train_bars, fold_idx)

            # Predict on val split — one prediction per val bar
            for bar_pos in range(len(val_bars)):
                # Use all bars up to and including this position for inference
                # (needs enough lookback context from the val slice itself)
                context_bars = val_bars.iloc[: bar_pos + 1]
                if len(context_bars) < 10:
                    continue

                meta_feat = self._predict_meta_features(context_bars, fold_models)
                # Label: was the forward return positive at the corresponding original index?
                orig_idx = val_idx[bar_pos]
                label = int(labels[orig_idx]) if orig_idx < len(labels) else 0

                meta_X_rows.append(meta_feat)
                meta_y_rows.append(label)

        if len(meta_X_rows) < 20:
            log.warning(
                "[StackingEnsemble] Too few OOF samples — meta-learner not trained."
            )
            return

        X_meta = np.vstack(meta_X_rows).astype(np.float32)
        y_meta = np.array(meta_y_rows, dtype=np.int32)

        log.info(
            f"[StackingEnsemble] Training meta-learner on {len(X_meta)} OOF samples. "
            f"Positive rate: {y_meta.mean():.1%}"
        )

        self._meta_learner = lgb.LGBMClassifier(
            n_estimators=self.meta_n_estimators,
            max_depth=3,
            learning_rate=0.05,
            num_leaves=15,
            random_state=42,
            verbosity=-1,
        )
        self._meta_learner.fit(X_meta, y_meta)
        self.is_fitted = True
        # Phase 4 — cache attribution after fit
        self._attribution = self.get_attribution()
        log.info("[StackingEnsemble] Meta-learner fit complete.")

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, base_preds: dict[str, Any]) -> tuple[str, float, dict[str, float]]:
        """
        Predict direction and confidence from base model outputs.

        Parameters
        ----------
        base_preds : dict
            Output of ModelEnsemble.predict() — must contain at minimum the
            same keys produced by ModelEnsemble. Missing keys yield 0.0.

        Returns
        -------
        (direction, confidence, attribution)
            direction  : "long" | "flat"
            confidence : float in [0, 1]
            attribution: dict mapping base-model group names to normalised
                         importance scores (sum ≈ 1.0); empty dict when not fitted.
        """
        if not self.is_fitted or self._meta_learner is None:
            return "flat", 0.0, {}

        meta_feat = self._build_meta_features(base_preds)
        X = meta_feat[np.newaxis, :]  # (1, 11)
        prob = float(self._meta_learner.predict_proba(X)[0, 1])
        direction = "long" if prob >= 0.5 else "flat"
        return direction, prob, self._attribution

    def predict_from_bars(self, bars: pd.DataFrame) -> tuple[str, float, dict[str, float]]:
        """Convenience: builds an ensemble and calls predict().  Lazy-loads models."""
        if not self.is_fitted:
            return "flat", 0.0, {}
        from privateye.models.inference import ModelEnsemble

        ens = ModelEnsemble(artifacts_dir=self.artifacts_dir)
        base_preds = ens.predict(bars)
        return self.predict(base_preds)

    # ── Phase 4: Attribution ──────────────────────────────────────────────────

    def get_attribution(self) -> dict[str, float]:
        """Return normalised feature group importances from the meta-learner.

        Maps the 11 meta-feature importances to 5 named groups:
          gbm, lgbm, lstm, attn_lstm, regime

        Returns
        -------
        dict[str, float]
            Attribution scores summing to approximately 1.0.
            Returns empty dict when meta-learner is not fitted.
        """
        if self._meta_learner is None or not self.is_fitted:
            return {}

        try:
            importances = self._meta_learner.feature_importances_
        except AttributeError:
            return {}

        if len(importances) < _N_META_FEATURES:
            return {}

        # Map 11 meta-features to 5 named groups:
        # [0] gbm_gate_prob       → gbm
        # [1] lgbm_gate_prob      → lgbm
        # [2,4,5] lstm_*          → lstm
        # [3,6]   attn_lstm_*     → attn_lstm
        # [7,8,9,10] regime_probs → regime
        group_map = {
            "gbm":       [0],
            "lgbm":      [1],
            "lstm":      [2, 4, 5],
            "attn_lstm": [3, 6],
            "regime":    [7, 8, 9, 10],
        }

        raw: dict[str, float] = {}
        for group, indices in group_map.items():
            raw[group] = float(sum(importances[i] for i in indices if i < len(importances)))

        total = sum(raw.values())
        if total <= 0:
            # Uniform fallback
            n = len(group_map)
            return {k: 1.0 / n for k in group_map}

        return {k: v / total for k, v in raw.items()}

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        import joblib

        if not self.is_fitted or self._meta_learner is None:
            raise RuntimeError("StackingEnsemble is not fitted. Call fit() first.")
        path = self.artifacts_dir / "stacking_meta.pkl"
        joblib.dump(
            {
                "meta_learner":    self._meta_learner,
                "n_folds":         self.n_folds,
                "label_horizon":   self.label_horizon,
                "label_threshold": self.label_threshold,
                "attribution":     self._attribution,   # Phase 4
            },
            path,
        )
        log.info(f"[StackingEnsemble] Saved to {path}")

    def load(self) -> None:
        import joblib

        path = self.artifacts_dir / "stacking_meta.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        payload = joblib.load(path)
        self._meta_learner   = payload["meta_learner"]
        self.n_folds         = payload.get("n_folds",         self.n_folds)
        self.label_horizon   = payload.get("label_horizon",   self.label_horizon)
        self.label_threshold = payload.get("label_threshold", self.label_threshold)
        self.is_fitted       = True
        # Phase 4 — restore attribution from saved payload or recompute
        self._attribution = payload.get("attribution") or self.get_attribution()
        log.info(f"[StackingEnsemble] Loaded from {path}")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _build_meta_features(self, base_preds: dict[str, Any]) -> np.ndarray:
        """Convert ModelEnsemble.predict() output dict → (11,) float32 array."""
        meta = np.zeros(_N_META_FEATURES, dtype=np.float32)

        # [0] gbm_gate_prob
        gbm = base_preds.get("gbm", 0.5)
        meta[0] = float(gbm) if isinstance(gbm, (int, float)) else 0.5

        # [1] lgbm_gate_prob
        lgbm = base_preds.get("lgbm", 0.0)
        if isinstance(lgbm, (int, float)):
            meta[1] = float(lgbm)
        elif isinstance(lgbm, tuple) and len(lgbm) == 2:
            meta[1] = float(lgbm[1])  # (direction, gate_prob) form

        # [2] lstm_conf, [4] lstm_dir_long, [5] lstm_dir_flat
        lstm = base_preds.get("lstm", ("flat", 0.0))
        if isinstance(lstm, tuple) and len(lstm) == 2:
            lstm_dir, lstm_conf = lstm
            meta[2] = float(lstm_conf)
            meta[4] = 1.0 if lstm_dir == "long" else 0.0
            meta[5] = 1.0 if lstm_dir == "flat" else 0.0

        # [3] attn_lstm_conf, [6] attn_lstm_dir_long
        attn = base_preds.get("attn_lstm", ("flat", 0.0))
        if isinstance(attn, tuple) and len(attn) == 2:
            attn_dir, attn_conf = attn
            meta[3] = float(attn_conf)
            meta[6] = 1.0 if attn_dir == "long" else 0.0

        # [7-10] regime_probs — prefer neural_regime over hmm regime
        regime_result = base_preds.get(
            "neural_regime", base_preds.get("regime")
        )
        if regime_result is not None and isinstance(regime_result, tuple) and len(regime_result) == 2:
            _, probs = regime_result
            if hasattr(probs, "__len__") and len(probs) >= 4:
                meta[7:11] = np.array(probs[:4], dtype=np.float32)
            else:
                meta[7:11] = 0.25
        else:
            meta[7:11] = 0.25

        return meta

    def _predict_meta_features(
        self,
        bars: pd.DataFrame,
        fold_models: dict[str, Any],
    ) -> np.ndarray:
        """Extract meta-features from per-fold lightweight models."""
        pseudo_preds: dict[str, Any] = {}

        if "gbm" in fold_models and fold_models["gbm"] is not None:
            try:
                pseudo_preds["gbm"] = fold_models["gbm"].predict_gate_prob(bars)
            except Exception:
                pseudo_preds["gbm"] = 0.5

        if "lgbm" in fold_models and fold_models["lgbm"] is not None:
            try:
                pseudo_preds["lgbm"] = fold_models["lgbm"].predict_gate_prob(bars)
            except Exception:
                pseudo_preds["lgbm"] = 0.0

        if "lstm" in fold_models and fold_models["lstm"] is not None:
            try:
                pseudo_preds["lstm"] = fold_models["lstm"].predict(bars)
            except Exception:
                pseudo_preds["lstm"] = ("flat", 0.0)

        if "attn_lstm" in fold_models and fold_models["attn_lstm"] is not None:
            try:
                pseudo_preds["attn_lstm"] = fold_models["attn_lstm"].predict(bars)
            except Exception:
                pseudo_preds["attn_lstm"] = ("flat", 0.0)

        if "neural_regime" in fold_models and fold_models["neural_regime"] is not None:
            try:
                pseudo_preds["neural_regime"] = fold_models["neural_regime"].predict_regime(bars)
            except Exception:
                pseudo_preds["neural_regime"] = (0, np.full(4, 0.25, dtype=np.float32))

        return self._build_meta_features(pseudo_preds)

    def _train_fold_models(
        self,
        train_bars: pd.DataFrame,
        fold_idx: int,
    ) -> dict[str, Any]:
        """Train lightweight per-fold base model instances."""
        import tempfile

        tmp_dir = Path(tempfile.mkdtemp())
        models: dict[str, Any] = {}

        # GBM (XGBoost)
        try:
            from privateye.models.gbm_classifier import GBMClassifier

            m = GBMClassifier(
                n_estimators=_OOF_GBM_PARAMS["n_estimators"],
                max_depth=_OOF_GBM_PARAMS["max_depth"],
                learning_rate=_OOF_GBM_PARAMS["learning_rate"],
                artifacts_dir=tmp_dir,
            )
            m.fit(train_bars)
            models["gbm"] = m
        except Exception as exc:
            log.debug(f"[StackingEnsemble] fold={fold_idx} GBM failed: {exc}")
            models["gbm"] = None

        # LightGBM
        try:
            from privateye.models.lgbm_classifier import LGBMClassifier

            m = LGBMClassifier(
                n_estimators=_OOF_LGBM_PARAMS["n_estimators"],
                max_depth=_OOF_LGBM_PARAMS["max_depth"],
                learning_rate=_OOF_LGBM_PARAMS["learning_rate"],
                num_leaves=_OOF_LGBM_PARAMS["num_leaves"],
                artifacts_dir=tmp_dir,
            )
            m.fit(train_bars)
            models["lgbm"] = m
        except Exception as exc:
            log.debug(f"[StackingEnsemble] fold={fold_idx} LGBM failed: {exc}")
            models["lgbm"] = None

        # LSTM Forecaster
        try:
            from privateye.models.lstm_forecaster import LSTMForecaster

            m = LSTMForecaster(
                epochs=_OOF_LSTM_PARAMS["epochs"],
                patience=_OOF_LSTM_PARAMS["patience"],
                hidden_size=_OOF_LSTM_PARAMS["hidden_size"],
                num_layers=_OOF_LSTM_PARAMS["num_layers"],
                artifacts_dir=tmp_dir,
            )
            m.fit(train_bars)
            models["lstm"] = m
        except Exception as exc:
            log.debug(f"[StackingEnsemble] fold={fold_idx} LSTM failed: {exc}")
            models["lstm"] = None

        # AttentionLSTM
        try:
            from privateye.models.attention_lstm import AttentionLSTM

            m = AttentionLSTM(
                epochs=_OOF_ATTN_PARAMS["epochs"],
                patience=_OOF_ATTN_PARAMS["patience"],
                hidden_size=_OOF_ATTN_PARAMS["hidden_size"],
                num_layers=_OOF_ATTN_PARAMS["num_layers"],
                n_heads=_OOF_ATTN_PARAMS["n_heads"],
                artifacts_dir=tmp_dir,
            )
            m.fit(train_bars)
            models["attn_lstm"] = m
        except Exception as exc:
            log.debug(f"[StackingEnsemble] fold={fold_idx} AttentionLSTM failed: {exc}")
            models["attn_lstm"] = None

        # NeuralRegimeClassifier
        try:
            from privateye.models.neural_regime import NeuralRegimeClassifier

            m = NeuralRegimeClassifier(
                epochs=_OOF_NR_PARAMS["epochs"],
                hidden=_OOF_NR_PARAMS["hidden"],
                artifacts_dir=tmp_dir,
            )
            m.fit(train_bars)
            models["neural_regime"] = m
        except Exception as exc:
            log.debug(f"[StackingEnsemble] fold={fold_idx} NeuralRegime failed: {exc}")
            models["neural_regime"] = None

        return models
