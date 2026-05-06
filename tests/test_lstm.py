"""Tests for LSTM forecaster (requires torch)."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from privateye.models.lstm_forecaster import LSTMForecaster, _LSTMNet
from privateye.models.training import build_lstm_sequences, make_direction_labels
from privateye.data.feature_extractor import extract_features


def _make_bars(n: int = 400, seed: int = 1) -> pd.DataFrame:
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


class TestLSTMNet:
    def test_forward_shape(self):
        net = _LSTMNet(input_size=50, hidden_size=64, num_layers=1, dropout=0.0)
        x   = torch.randn(4, 60, 50)
        out = net(x)
        assert out.shape == (4, 2)

    def test_no_nan_output(self):
        net = _LSTMNet(input_size=50, hidden_size=64, num_layers=1, dropout=0.0)
        x   = torch.randn(8, 60, 50)
        out = net(x)
        assert not torch.isnan(out).any()


class TestBuildSequences:
    def test_shape(self):
        feats  = np.random.randn(200, 50).astype(np.float32)
        labels = np.zeros(200, dtype=np.int8)
        X, y   = build_lstm_sequences(feats, labels, lookback=60)
        assert X.shape == (140, 60, 50)
        assert y.shape == (140,)

    def test_dtypes(self):
        feats  = np.random.randn(100, 50).astype(np.float32)
        labels = np.ones(100, dtype=np.int8)
        X, y   = build_lstm_sequences(feats, labels, lookback=20)
        assert X.dtype == np.float32
        assert y.dtype == np.int64


class TestDirectionLabels:
    def test_shape(self):
        bars   = _make_bars(100)
        labels = make_direction_labels(bars, horizon=5)
        assert len(labels) == 100

    def test_last_horizon_rows_are_zero(self):
        bars   = _make_bars(100)
        labels = make_direction_labels(bars, horizon=5)
        assert all(labels[-5:] == 0)

    def test_binary(self):
        bars   = _make_bars(100)
        labels = make_direction_labels(bars, horizon=5)
        assert set(labels).issubset({0, 1})


class TestLSTMForecaster:
    def test_fit_predict(self, tmp_path):
        bars = _make_bars(500)
        m = LSTMForecaster(
            lookback=30, label_horizon=3, epochs=2,
            hidden_size=32, num_layers=1,
            artifacts_dir=tmp_path,
        )
        m.fit(bars)
        direction, conf = m.predict(bars)
        assert direction in ("long", "short")
        assert 0.0 <= conf <= 1.0

    def test_insufficient_bars_returns_flat(self, tmp_path):
        bars = _make_bars(500)
        m = LSTMForecaster(lookback=60, epochs=2, hidden_size=32, num_layers=1,
                           artifacts_dir=tmp_path)
        m.fit(bars)
        short_bars = bars.iloc[:30]
        d, c = m.predict(short_bars)
        assert d == "flat"
        assert c == 0.0

    def test_save_load(self, tmp_path):
        bars = _make_bars(500)
        m = LSTMForecaster(lookback=30, epochs=2, hidden_size=32, num_layers=1,
                           artifacts_dir=tmp_path)
        m.fit(bars)
        d1, c1 = m.predict(bars)
        m.save()

        m2 = LSTMForecaster(artifacts_dir=tmp_path)
        m2.load()
        d2, c2 = m2.predict(bars)
        assert d1 == d2
        assert abs(c1 - c2) < 1e-5

    def test_deterministic_inference(self, tmp_path):
        bars = _make_bars(500)
        m = LSTMForecaster(lookback=30, epochs=2, hidden_size=32, num_layers=1,
                           artifacts_dir=tmp_path)
        m.fit(bars)
        d1, c1 = m.predict(bars)
        d2, c2 = m.predict(bars)
        assert d1 == d2
        assert abs(c1 - c2) < 1e-5
