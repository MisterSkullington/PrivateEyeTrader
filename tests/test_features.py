"""Tests for the Phase-2/3 feature extractor (68 features)."""
import numpy as np
import pandas as pd
import pytest

from privateye.data.feature_extractor import (
    FEATURE_NAMES, extract_features, extract_latest,
)


def _make_bars(n: int = 300, seed: int = 0, with_alt_data: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 30_000 + np.cumsum(rng.normal(0, 100, n))
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "open":   closes - rng.uniform(10, 50, n),
        "high":   closes + rng.uniform(50, 200, n),
        "low":    closes - rng.uniform(50, 200, n),
        "close":  closes,
        "volume": rng.uniform(200, 1000, n),
    })
    if with_alt_data:
        df["funding_rate"]  = rng.uniform(-0.001, 0.001, n)
        df["open_interest"] = rng.uniform(1e8, 2e8, n)
        df["fear_greed"]    = rng.uniform(0, 100, n)
    return df


class TestFeatureNames:
    def test_count(self):
        assert len(FEATURE_NAMES) == 68

    def test_unique(self):
        assert len(set(FEATURE_NAMES)) == 68

    def test_has_mtf_features(self):
        assert "rsi_56" in FEATURE_NAMES
        assert "rsi_168" in FEATURE_NAMES
        assert "ema_96_ratio" in FEATURE_NAMES
        assert "trend_alignment" in FEATURE_NAMES

    def test_has_alt_data_features(self):
        assert "funding_rate_norm" in FEATURE_NAMES
        assert "fear_greed_norm" in FEATURE_NAMES
        assert "combined_sentiment" in FEATURE_NAMES


class TestExtractFeatures:
    def test_shape(self):
        df = _make_bars(300)
        out = extract_features(df)
        assert out.shape == (300, 68)

    def test_dtype(self):
        df = _make_bars(100)
        out = extract_features(df)
        assert out.dtype == np.float32

    def test_no_nan(self):
        df = _make_bars(300)
        out = extract_features(df)
        assert not np.isnan(out).any(), "NaN found in feature matrix"

    def test_no_inf(self):
        df = _make_bars(300)
        out = extract_features(df)
        assert not np.isinf(out).any(), "Inf found in feature matrix"

    def test_warmup_rows_are_zero(self):
        df = _make_bars(300)
        out = extract_features(df)
        # Row 0 has no prior data — many features should be 0
        assert np.count_nonzero(out[0]) < 25, "Too many non-zero features at row 0"

    def test_no_look_ahead(self):
        """Features at bar i must be identical whether computed on bars[:i+1] or full bars."""
        df = _make_bars(300)
        full  = extract_features(df)
        split = 150
        partial = extract_features(df.iloc[:split + 1])
        # Features at row `split` must match between full and partial
        assert np.allclose(full[split], partial[split], atol=1e-5), (
            "Look-ahead detected: features differ between full and partial computation"
        )

    def test_min_bars(self):
        df = _make_bars(5)
        out = extract_features(df)
        assert out.shape == (5, 68)

    def test_time_cyclical_range(self):
        df = _make_bars(300)
        out = extract_features(df)
        # Sin/cos features (indices 42-49) must be in [-1, 1]
        cyclic_indices = list(range(42, 50))
        cyclic = out[:, cyclic_indices]
        assert (cyclic >= -1.0 - 1e-6).all()
        assert (cyclic <=  1.0 + 1e-6).all()

    def test_rsi_range(self):
        df = _make_bars(300)
        out = extract_features(df)
        rsi14_col = FEATURE_NAMES.index("rsi_14")
        rsi_vals = out[50:, rsi14_col]  # skip warmup
        assert (rsi_vals >= 0.0 - 1e-6).all(), "RSI below 0"
        assert (rsi_vals <= 1.0 + 1e-6).all(), "RSI above 1 (should be normalised)"

    def test_consistent_across_calls(self):
        df = _make_bars(200)
        out1 = extract_features(df)
        out2 = extract_features(df)
        assert np.array_equal(out1, out2), "extract_features is not deterministic"

    def test_alt_data_zero_when_absent(self):
        """When bars has no alt-data columns, those features should be zero."""
        df = _make_bars(300)
        out = extract_features(df)
        fr_idx = FEATURE_NAMES.index("funding_rate_norm")
        fg_idx = FEATURE_NAMES.index("fear_greed_norm")
        # Without alt data columns: funding_rate_norm = 0/0.001 = 0
        assert np.all(out[:, fr_idx] == 0.0), "funding_rate_norm should be 0 when column absent"
        # fear_greed_norm from 0/100 = 0
        assert np.all(out[:, fg_idx] == 0.0), "fear_greed_norm should be 0 when column absent"

    def test_alt_data_columns_used(self):
        """When bars includes alt-data columns, those features should be non-trivially non-zero."""
        df = _make_bars(300, with_alt_data=True)
        out = extract_features(df)
        fr_idx = FEATURE_NAMES.index("funding_rate_norm")
        fg_idx = FEATURE_NAMES.index("fear_greed_norm")
        assert np.any(out[:, fr_idx] != 0.0), "funding_rate_norm all zero despite column present"
        assert np.any(out[:, fg_idx] != 0.0), "fear_greed_norm all zero despite column present"

    def test_mtf_rsi_range(self):
        df = _make_bars(300)
        out = extract_features(df)
        for name in ("rsi_56", "rsi_168"):
            idx = FEATURE_NAMES.index(name)
            vals = out[50:, idx]  # skip warmup
            assert (vals >= 0.0 - 1e-6).all(), f"{name} below 0"
            assert (vals <= 1.0 + 1e-6).all(), f"{name} above 1"

    def test_trend_alignment_values(self):
        df = _make_bars(300)
        out = extract_features(df)
        ta_idx = FEATURE_NAMES.index("trend_alignment")
        unique_vals = np.unique(out[:, ta_idx])
        # trend_alignment ∈ {-1, 0, 1}
        assert set(unique_vals).issubset({-1.0, 0.0, 1.0}), (
            f"trend_alignment has unexpected values: {unique_vals}"
        )

    def test_fear_greed_extreme_binary(self):
        df = _make_bars(300, with_alt_data=True)
        out = extract_features(df)
        fge_idx = FEATURE_NAMES.index("fear_greed_extreme")
        unique_vals = np.unique(out[:, fge_idx])
        assert set(unique_vals).issubset({0.0, 1.0}), (
            f"fear_greed_extreme should be binary, got: {unique_vals}"
        )


class TestExtractLatest:
    def test_shape(self):
        df = _make_bars(300)
        v = extract_latest(df)
        assert v.shape == (68,)

    def test_matches_last_row(self):
        df = _make_bars(300)
        full = extract_features(df)
        latest = extract_latest(df)
        assert np.array_equal(full[-1], latest)

    def test_no_timestamp_column(self):
        df = _make_bars(100)
        df = df.drop(columns=["timestamp"])
        out = extract_features(df)
        assert out.shape == (100, 68)
        assert not np.isnan(out).any()
