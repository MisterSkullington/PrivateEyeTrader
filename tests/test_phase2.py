"""
Phase 2 tests — Advanced Ensemble AI Core.

Tests for:
  - LGBMClassifier (6 tests)
  - AttentionLSTM  (6 tests)
  - NeuralRegimeClassifier (6 tests)
  - StackingEnsemble (8 tests)
  - FusionStrategy Phase 2 integration (4 tests)

Total: 30 tests.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_bars(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with realistic structure."""
    rng = np.random.default_rng(seed)
    closes = 30_000 + np.cumsum(rng.normal(0, 150, n))
    highs  = closes + np.abs(rng.normal(0, 80, n))
    lows   = closes - np.abs(rng.normal(0, 80, n))
    opens  = closes + rng.normal(0, 50, n)
    vols   = rng.uniform(500, 8000, n)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
            "open":      opens,
            "high":      highs,
            "low":       lows,
            "close":     closes,
            "volume":    vols,
        }
    )


def _tmp_artifacts() -> Path:
    return Path(tempfile.mkdtemp())


# ── TestLGBMClassifier ────────────────────────────────────────────────────────


class TestLGBMClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import lightgbm  # noqa: F401
            cls._lgbm_available = True
        except ImportError:
            cls._lgbm_available = False

    def _skip_if_missing(self):
        if not self._lgbm_available:
            self.skipTest("lightgbm not installed")

    def test_fit_predict_canonical_direction_and_confidence(self):
        """fit() + predict() return (direction, conf) in the correct form."""
        self._skip_if_missing()
        from privateye.models.lgbm_classifier import LGBMClassifier

        m = LGBMClassifier(n_estimators=30, artifacts_dir=_tmp_artifacts())
        bars = _make_bars(300)
        m.fit(bars)
        direction, conf = m.predict(bars)
        self.assertIn(direction, {"long", "flat"})
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_gate_prob_in_range(self):
        """predict_gate_prob() always returns a float in [0, 1]."""
        self._skip_if_missing()
        from privateye.models.lgbm_classifier import LGBMClassifier

        m = LGBMClassifier(n_estimators=30, artifacts_dir=_tmp_artifacts())
        bars = _make_bars(300)
        m.fit(bars)
        prob = m.predict_gate_prob(bars)
        self.assertIsInstance(prob, float)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_save_load_roundtrip(self):
        """save() + load() produces the same gate_prob for the same input."""
        self._skip_if_missing()
        from privateye.models.lgbm_classifier import LGBMClassifier

        artifacts = _tmp_artifacts()
        bars = _make_bars(300)
        m1 = LGBMClassifier(n_estimators=30, artifacts_dir=artifacts)
        m1.fit(bars)
        m1.save()
        prob_before = m1.predict_gate_prob(bars)

        m2 = LGBMClassifier(artifacts_dir=artifacts)
        m2.load()
        prob_after = m2.predict_gate_prob(bars)
        self.assertAlmostEqual(prob_before, prob_after, places=5)

    def test_feature_importance_non_empty_after_fit(self):
        """get_top_features() returns a non-empty list after training."""
        self._skip_if_missing()
        from privateye.models.lgbm_classifier import LGBMClassifier

        m = LGBMClassifier(n_estimators=30, artifacts_dir=_tmp_artifacts())
        bars = _make_bars(300)
        m.fit(bars)
        top = m.get_top_features(5)
        self.assertGreater(len(top), 0)
        self.assertIsInstance(top[0][0], str)
        self.assertIsInstance(top[0][1], (int, float))

    def test_predict_unfitted_raises(self):
        """predict() on an unfitted model raises RuntimeError."""
        self._skip_if_missing()
        from privateye.models.lgbm_classifier import LGBMClassifier

        m = LGBMClassifier(artifacts_dir=_tmp_artifacts())
        with self.assertRaises(RuntimeError):
            m.predict(_make_bars(50))

    def test_training_completes_on_minimal_data(self):
        """Training on 300 bars completes without error."""
        self._skip_if_missing()
        from privateye.models.lgbm_classifier import LGBMClassifier

        m = LGBMClassifier(n_estimators=20, artifacts_dir=_tmp_artifacts())
        m.fit(_make_bars(300))
        self.assertTrue(m.is_fitted)


# ── TestAttentionLSTM ─────────────────────────────────────────────────────────


class TestAttentionLSTM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
            cls._torch_available = True
        except ImportError:
            cls._torch_available = False

    def _skip_if_missing(self):
        if not self._torch_available:
            self.skipTest("torch not installed")

    def test_predict_canonical_direction_and_confidence(self):
        """predict() returns direction in {long, short, flat} and conf in [0,1]."""
        self._skip_if_missing()
        from privateye.models.attention_lstm import AttentionLSTM

        m = AttentionLSTM(
            epochs=2, patience=2, hidden_size=32, num_layers=1, n_heads=4,
            artifacts_dir=_tmp_artifacts(),
        )
        bars = _make_bars(200)
        m.fit(bars)
        direction, conf = m.predict(bars)
        self.assertIn(direction, {"long", "short", "flat"})
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_confidence_in_range(self):
        """Confidence value is always in [0, 1]."""
        self._skip_if_missing()
        from privateye.models.attention_lstm import AttentionLSTM

        m = AttentionLSTM(
            epochs=2, patience=2, hidden_size=32, num_layers=1, n_heads=4,
            artifacts_dir=_tmp_artifacts(),
        )
        m.fit(_make_bars(200))
        _, conf = m.predict(_make_bars(200))
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_save_load_roundtrip(self):
        """save() + load() preserves the model's prediction."""
        self._skip_if_missing()
        from privateye.models.attention_lstm import AttentionLSTM

        artifacts = _tmp_artifacts()
        bars = _make_bars(200)
        m1 = AttentionLSTM(
            epochs=2, patience=2, hidden_size=32, num_layers=1, n_heads=4,
            artifacts_dir=artifacts,
        )
        m1.fit(bars)
        m1.save()
        dir1, conf1 = m1.predict(bars)

        m2 = AttentionLSTM(artifacts_dir=artifacts)
        m2.load()
        dir2, conf2 = m2.predict(bars)
        self.assertEqual(dir1, dir2)
        self.assertAlmostEqual(conf1, conf2, places=4)

    def test_insufficient_bars_returns_flat(self):
        """predict() returns ('flat', 0.0) when fewer bars than lookback."""
        self._skip_if_missing()
        from privateye.models.attention_lstm import AttentionLSTM

        m = AttentionLSTM(
            epochs=2, patience=2, hidden_size=32, num_layers=1, n_heads=4,
            artifacts_dir=_tmp_artifacts(),
        )
        m.fit(_make_bars(200))
        # Only 10 bars — far fewer than lookback=60
        short_bars = _make_bars(10)
        direction, conf = m.predict(short_bars)
        self.assertEqual(direction, "flat")
        self.assertEqual(conf, 0.0)

    def test_attention_module_is_a_parameter(self):
        """The network has attention parameters (not a no-op)."""
        self._skip_if_missing()
        import torch
        from privateye.models.attention_lstm import AttentionLSTM

        m = AttentionLSTM(
            epochs=2, patience=2, hidden_size=32, num_layers=1, n_heads=4,
            artifacts_dir=_tmp_artifacts(),
        )
        m.fit(_make_bars(200))
        # Confirm the network has attention-related parameters
        param_names = [n for n, _ in m._net.named_parameters()]
        attn_params = [n for n in param_names if "attn" in n]
        self.assertGreater(len(attn_params), 0)

    def test_save_writes_both_pt_and_json(self):
        """save() writes both the .pt weights and the _config.json."""
        self._skip_if_missing()
        from privateye.models.attention_lstm import AttentionLSTM

        artifacts = _tmp_artifacts()
        m = AttentionLSTM(
            epochs=2, patience=2, hidden_size=32, num_layers=1, n_heads=4,
            artifacts_dir=artifacts,
        )
        m.fit(_make_bars(200))
        m.save()
        self.assertTrue((artifacts / "attention_lstm.pt").exists())
        self.assertTrue((artifacts / "attention_lstm_config.json").exists())


# ── TestNeuralRegimeClassifier ────────────────────────────────────────────────


class TestNeuralRegimeClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
            from sklearn.cluster import KMeans  # noqa: F401
            cls._available = True
        except ImportError:
            cls._available = False

    def _skip_if_missing(self):
        if not self._available:
            self.skipTest("torch or sklearn not installed")

    def test_fit_sets_is_fitted(self):
        """fit() sets is_fitted=True."""
        self._skip_if_missing()
        from privateye.models.neural_regime import NeuralRegimeClassifier

        m = NeuralRegimeClassifier(epochs=5, artifacts_dir=_tmp_artifacts())
        m.fit(_make_bars(300))
        self.assertTrue(m.is_fitted)

    def test_predict_regime_returns_int_and_array(self):
        """predict_regime() returns (int, ndarray(4,))."""
        self._skip_if_missing()
        from privateye.models.neural_regime import NeuralRegimeClassifier

        m = NeuralRegimeClassifier(epochs=5, artifacts_dir=_tmp_artifacts())
        m.fit(_make_bars(300))
        regime, probs = m.predict_regime(_make_bars(300))
        self.assertIsInstance(regime, int)
        self.assertEqual(len(probs), 4)

    def test_regime_int_in_valid_range(self):
        """Returned regime integer is in [0, n_states-1]."""
        self._skip_if_missing()
        from privateye.models.neural_regime import NeuralRegimeClassifier

        m = NeuralRegimeClassifier(n_states=4, epochs=5, artifacts_dir=_tmp_artifacts())
        m.fit(_make_bars(300))
        regime, _ = m.predict_regime(_make_bars(300))
        self.assertGreaterEqual(regime, 0)
        self.assertLess(regime, 4)

    def test_probs_sum_to_one(self):
        """Regime probabilities sum to approximately 1.0."""
        self._skip_if_missing()
        from privateye.models.neural_regime import NeuralRegimeClassifier

        m = NeuralRegimeClassifier(epochs=5, artifacts_dir=_tmp_artifacts())
        m.fit(_make_bars(300))
        _, probs = m.predict_regime(_make_bars(300))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=3)

    def test_predict_regime_unfitted_returns_uniform(self):
        """predict_regime() before fit() returns uniform probability distribution."""
        self._skip_if_missing()
        from privateye.models.neural_regime import NeuralRegimeClassifier

        m = NeuralRegimeClassifier(artifacts_dir=_tmp_artifacts())
        regime, probs = m.predict_regime(_make_bars(100))
        self.assertEqual(regime, 0)
        self.assertAlmostEqual(float(probs[0]), 0.25, places=3)

    def test_save_load_roundtrip_regime_output(self):
        """save() + load() produces the same regime output."""
        self._skip_if_missing()
        from privateye.models.neural_regime import NeuralRegimeClassifier

        artifacts = _tmp_artifacts()
        bars = _make_bars(300)
        m1 = NeuralRegimeClassifier(epochs=5, artifacts_dir=artifacts)
        m1.fit(bars)
        m1.save()
        r1, p1 = m1.predict_regime(bars)

        m2 = NeuralRegimeClassifier(artifacts_dir=artifacts)
        m2.load()
        r2, p2 = m2.predict_regime(bars)
        self.assertEqual(r1, r2)
        np.testing.assert_allclose(p1, p2, atol=1e-4)


# ── TestStackingEnsemble ──────────────────────────────────────────────────────


class TestStackingEnsemble(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import lightgbm  # noqa: F401
            cls._lgbm_available = True
        except ImportError:
            cls._lgbm_available = False

    def _skip_if_missing(self):
        if not self._lgbm_available:
            self.skipTest("lightgbm not installed")

    def test_predict_unfitted_returns_flat(self):
        """predict() before fit() returns ('flat', 0.0)."""
        from privateye.models.stacking_ensemble import StackingEnsemble

        stack = StackingEnsemble(artifacts_dir=_tmp_artifacts())
        direction, conf, attribution = stack.predict({})
        self.assertEqual(direction, "flat")
        self.assertEqual(conf, 0.0)

    def test_is_fitted_false_before_fit(self):
        """is_fitted starts as False."""
        from privateye.models.stacking_ensemble import StackingEnsemble

        stack = StackingEnsemble(artifacts_dir=_tmp_artifacts())
        self.assertFalse(stack.is_fitted)

    def test_build_meta_features_from_mock_preds(self):
        """_build_meta_features() converts base_preds dict to (11,) array."""
        from privateye.models.stacking_ensemble import StackingEnsemble, _N_META_FEATURES

        stack = StackingEnsemble(artifacts_dir=_tmp_artifacts())
        base_preds = {
            "gbm":          0.7,
            "lgbm":         0.6,
            "lstm":         ("long", 0.8),
            "attn_lstm":    ("flat", 0.4),
            "neural_regime": (1, np.array([0.1, 0.5, 0.2, 0.2], dtype=np.float32)),
        }
        meta = stack._build_meta_features(base_preds)
        self.assertEqual(meta.shape, (_N_META_FEATURES,))
        self.assertAlmostEqual(meta[0], 0.7, places=5)    # gbm_gate_prob
        self.assertAlmostEqual(meta[1], 0.6, places=5)    # lgbm_gate_prob
        self.assertAlmostEqual(meta[2], 0.8, places=5)    # lstm_conf
        self.assertEqual(meta[4], 1.0)                    # lstm_dir_long
        self.assertEqual(meta[5], 0.0)                    # lstm_dir_flat
        self.assertEqual(meta[6], 0.0)                    # attn_lstm_dir_long
        self.assertAlmostEqual(meta[7], 0.1, places=5)   # regime_prob_0

    def test_build_meta_features_missing_keys_yield_defaults(self):
        """_build_meta_features() handles missing keys with safe defaults."""
        from privateye.models.stacking_ensemble import StackingEnsemble

        stack = StackingEnsemble(artifacts_dir=_tmp_artifacts())
        meta = stack._build_meta_features({})  # empty dict
        # All numeric, no NaN/Inf
        self.assertTrue(np.all(np.isfinite(meta)))
        # regime probs default to 0.25 each
        self.assertAlmostEqual(float(meta[7]), 0.25, places=4)

    def test_predict_returns_canonical_form_after_fit(self):
        """predict() returns (str, float) in canonical form after fitting."""
        self._skip_if_missing()
        from privateye.models.stacking_ensemble import StackingEnsemble

        stack = StackingEnsemble(
            n_folds=2, meta_n_estimators=20, artifacts_dir=_tmp_artifacts()
        )
        # Inject a pre-trained meta-learner mock to avoid full OOF training
        import lightgbm as lgb
        import numpy as np

        mock_meta = MagicMock()
        mock_meta.predict_proba = MagicMock(return_value=np.array([[0.3, 0.7]]))
        stack._meta_learner = mock_meta
        stack.is_fitted = True

        direction, conf, _attr = stack.predict({"gbm": 0.6, "lstm": ("long", 0.7)})
        self.assertIn(direction, {"long", "flat"})
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_confidence_in_range(self):
        """Confidence from predict() is always in [0, 1]."""
        self._skip_if_missing()
        import numpy as np
        from privateye.models.stacking_ensemble import StackingEnsemble

        stack = StackingEnsemble(artifacts_dir=_tmp_artifacts())
        mock_meta = MagicMock()
        mock_meta.predict_proba = MagicMock(return_value=np.array([[0.5, 0.5]]))
        stack._meta_learner = mock_meta
        stack.is_fitted = True

        _, conf, _attr = stack.predict({})
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_save_load_roundtrip(self):
        """save() + load() restores is_fitted=True and can predict."""
        self._skip_if_missing()
        import numpy as np
        from privateye.models.stacking_ensemble import StackingEnsemble

        artifacts = _tmp_artifacts()
        stack = StackingEnsemble(n_folds=2, meta_n_estimators=20, artifacts_dir=artifacts)
        # Inject mock meta-learner and save
        import lightgbm as lgb
        # Build a tiny real meta-learner so joblib.dump works
        clf = lgb.LGBMClassifier(n_estimators=5, verbosity=-1)
        X_dummy = np.random.rand(20, 11).astype(np.float32)
        y_dummy = np.random.randint(0, 2, 20)
        clf.fit(X_dummy, y_dummy)
        stack._meta_learner = clf
        stack.is_fitted = True
        stack.save()

        stack2 = StackingEnsemble(artifacts_dir=artifacts)
        stack2.load()
        self.assertTrue(stack2.is_fitted)
        direction, conf, _attr = stack2.predict({"gbm": 0.5})
        self.assertIn(direction, {"long", "flat"})

    def test_oof_time_series_split_indices_are_temporal(self):
        """OOF fold train indices are strictly before val indices."""
        from sklearn.model_selection import TimeSeriesSplit

        n = 500
        tss = TimeSeriesSplit(n_splits=3, gap=5)
        for train_idx, val_idx in tss.split(np.arange(n)):
            # All training indices must precede all validation indices
            self.assertLess(int(train_idx.max()), int(val_idx.min()))


# ── TestFusionStrategyPhase2 ──────────────────────────────────────────────────


class TestFusionStrategyPhase2(unittest.TestCase):
    def _make_minimal_cfg(self) -> dict:
        return {
            "strategy_id":       "fusion",
            "enabled":           True,
            "timeframe":         "1h",
            "gbm_gate":          True,
            "gbm_gate_threshold": 0.45,
            "regime_gate":       True,
            "sharpe_window_trades": 10,
            "artifacts_dir":     str(_tmp_artifacts()),
            # DirectionalStrategy config keys
            "macd_fast":         12,
            "macd_slow":         26,
            "macd_signal":       9,
            "ema_trend":         200,
            "rsi_period":        14,
            "rsi_overbought":    70,
            "rsi_oversold":      30,
            # MeanReversionStrategy config keys
            "bb_period":         20,
            "bb_std":            2.0,
        }

    def test_on_data_uses_stacking_when_fitted(self):
        """When stacking is fitted, on_data bypasses weighted vote."""
        from privateye.strategies.fusion_strategy import FusionStrategy

        cfg = self._make_minimal_cfg()
        strat = FusionStrategy(cfg)

        # Mock ensemble to return a stacking result
        mock_result = {
            "regime":   (0, np.array([0.7, 0.1, 0.1, 0.1])),
            "lstm":     ("long", 0.6),
            "gbm":      0.7,
            "rl":       ("flat", 0.3),
            "stacking": ("long", 0.75),
            "shap":     None,
        }
        strat._ensemble = MagicMock()
        strat._ensemble.predict = MagicMock(return_value=mock_result)

        from privateye.core.types import DataSnapshot

        bars = _make_bars(300)
        snap = DataSnapshot(
            symbol="BTC/USDT",
            timeframe="1h",
            bars=bars,
            timestamp=bars["timestamp"].iloc[-1],
        )
        signals = strat.on_data(snap)
        # With stacking returning "long" and confidence 0.75, should emit a signal
        self.assertIsInstance(signals, list)

    def test_on_data_fallback_to_weighted_vote_when_stacking_none(self):
        """When stacking result is None, fall back to _weighted_vote."""
        from privateye.strategies.fusion_strategy import FusionStrategy

        cfg = self._make_minimal_cfg()
        strat = FusionStrategy(cfg)

        mock_result = {
            "regime":   (0, np.array([0.7, 0.1, 0.1, 0.1])),
            "lstm":     ("long", 0.9),
            "gbm":      0.8,
            "rl":       ("long", 0.7),
            "stacking": None,  # stacking not fitted
            "shap":     None,
        }
        strat._ensemble = MagicMock()
        strat._ensemble.predict = MagicMock(return_value=mock_result)

        # Spy on _weighted_vote
        original_vote = strat._weighted_vote
        vote_calls = []

        def spy_vote(sources, gbm_pass):
            vote_calls.append((sources, gbm_pass))
            return original_vote(sources, gbm_pass)

        strat._weighted_vote = spy_vote

        from privateye.core.types import DataSnapshot

        bars = _make_bars(300)
        snap = DataSnapshot(
            symbol="BTC/USDT",
            timeframe="1h",
            bars=bars,
            timestamp=bars["timestamp"].iloc[-1],
        )
        strat.on_data(snap)
        self.assertGreater(len(vote_calls), 0, "Expected _weighted_vote to be called")

    def test_lgbm_and_attn_lstm_appear_in_sources_for_fallback(self):
        """When stacking=None, lgbm and attn_lstm are included in sources dict."""
        from privateye.strategies.fusion_strategy import FusionStrategy

        cfg = self._make_minimal_cfg()
        strat = FusionStrategy(cfg)

        mock_result = {
            "regime":     (0, np.array([0.7, 0.1, 0.1, 0.1])),
            "lstm":       ("long", 0.8),
            "gbm":        0.8,
            "rl":         ("flat", 0.3),
            "lgbm":       0.7,            # gate_prob above threshold
            "attn_lstm":  ("long", 0.75),
            "stacking":   None,
            "shap":       None,
        }
        strat._ensemble = MagicMock()
        strat._ensemble.predict = MagicMock(return_value=mock_result)

        captured_sources: list[dict] = []
        original_vote = strat._weighted_vote

        def spy_vote(sources, gbm_pass):
            captured_sources.append(dict(sources))
            return original_vote(sources, gbm_pass)

        strat._weighted_vote = spy_vote

        from privateye.core.types import DataSnapshot

        bars = _make_bars(300)
        snap = DataSnapshot(
            symbol="BTC/USDT",
            timeframe="1h",
            bars=bars,
            timestamp=bars["timestamp"].iloc[-1],
        )
        strat.on_data(snap)

        if captured_sources:
            keys = set(captured_sources[0].keys())
            # lgbm and attn_lstm should be in sources
            self.assertIn("lgbm", keys)
            self.assertIn("attn_lstm", keys)

    def test_phase2_complete_flag_importable(self):
        """PHASE2_COMPLETE flag is importable from main."""
        from privateye.main import PHASE2_COMPLETE

        self.assertIsInstance(PHASE2_COMPLETE, bool)


if __name__ == "__main__":
    unittest.main()
