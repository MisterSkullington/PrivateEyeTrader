"""Tests for GBM classifier (requires xgboost)."""
import numpy as np
import pandas as pd
import pytest

xgb = pytest.importorskip("xgboost", reason="xgboost not installed")

from privateye.data.feature_extractor import FEATURE_NAMES, extract_features
from privateye.models.gbm_classifier import GBMClassifier
from privateye.models.training import make_direction_labels, walk_forward_splits


def _make_bars(n: int = 500, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 30_000 + np.cumsum(rng.normal(0, 100, n))
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open":   closes - 20,
        "high":   closes + 100,
        "low":    closes - 100,
        "close":  closes,
        "volume": rng.uniform(200, 1000, n),
    })


class TestWalkForwardSplits:
    def test_basic(self):
        splits = walk_forward_splits(1000, train_window=600, test_window=200, step=200)
        assert len(splits) > 0
        for fold in splits:
            assert len(fold.train_idx) > 0
            assert len(fold.test_idx) > 0
            # No overlap
            assert len(set(fold.train_idx) & set(fold.test_idx)) == 0
            # Chronological order
            assert fold.train_idx.max() < fold.test_idx.min()

    def test_empty_for_small_dataset(self):
        splits = walk_forward_splits(50, train_window=600, test_window=200)
        assert splits == []


class TestGBMClassifier:
    def test_fit_predict(self, tmp_path):
        bars = _make_bars(500)
        m = GBMClassifier(n_estimators=10, artifacts_dir=tmp_path)
        m.fit(bars)
        direction, conf = m.predict(bars)
        assert direction in ("long", "flat")
        assert 0.0 <= conf <= 1.0

    def test_gate_prob_range(self, tmp_path):
        bars = _make_bars(500)
        m = GBMClassifier(n_estimators=10, artifacts_dir=tmp_path)
        m.fit(bars)
        prob = m.predict_gate_prob(bars)
        assert 0.0 <= prob <= 1.0

    def test_save_load(self, tmp_path):
        bars = _make_bars(500)
        m = GBMClassifier(n_estimators=10, artifacts_dir=tmp_path)
        m.fit(bars)
        p1 = m.predict_gate_prob(bars)
        m.save()

        m2 = GBMClassifier(artifacts_dir=tmp_path)
        m2.load()
        p2 = m2.predict_gate_prob(bars)
        assert abs(p1 - p2) < 1e-5

    def test_top_features(self, tmp_path):
        bars = _make_bars(500)
        m = GBMClassifier(n_estimators=10, artifacts_dir=tmp_path)
        m.fit(bars)
        top = m.get_top_features(5)
        assert len(top) == 5
        for name, score in top:
            assert name in FEATURE_NAMES
            assert score >= 0.0

    def test_requires_fit_before_predict(self, tmp_path):
        m = GBMClassifier(artifacts_dir=tmp_path)
        bars = _make_bars(100)
        with pytest.raises(RuntimeError):
            m.predict(bars)
