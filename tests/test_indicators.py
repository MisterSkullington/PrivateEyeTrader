"""Unit tests for the technical indicator library."""
import numpy as np
import pandas as pd
import pytest

from privateye.indicators.library import (
    atr, bollinger_bands, compute_all, ema, macd, obv, rsi, sma, stochastic,
)


def _make_bars(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 30000 + np.cumsum(rng.normal(0, 150, n))
    highs = closes + rng.uniform(50, 300, n)
    lows = closes - rng.uniform(50, 300, n)
    opens = closes - rng.normal(0, 80, n)
    volumes = rng.uniform(100, 1000, n)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": volumes,
    })


class TestEMA:
    def test_length(self):
        df = _make_bars()
        result = ema(df, 20)
        assert len(result) == len(df)

    def test_no_nan_after_warmup(self):
        df = _make_bars(100)
        result = ema(df, 20)
        assert result.iloc[20:].notna().all()

    def test_approaches_close(self):
        df = _make_bars(500)
        # EMA(9) on a trending series should be correlated with close
        e = ema(df, 9).dropna()
        c = df["close"].tail(len(e))
        corr = e.corr(c)
        assert corr > 0.9


class TestRSI:
    def test_bounds(self):
        df = _make_bars(200)
        r = rsi(df, 14).dropna()
        assert (r >= 0).all() and (r <= 100).all()

    def test_length(self):
        df = _make_bars(100)
        assert len(rsi(df, 14)) == len(df)

    def test_flat_market(self):
        df = _make_bars(100)
        df["close"] = 30000.0  # flat
        r = rsi(df, 14).dropna()
        # Flat market → RSI ~50 (or NaN if no changes)
        assert r.isna().any() or (r.between(40, 60)).all()


class TestMACD:
    def test_columns(self):
        df = _make_bars(200)
        result = macd(df)
        assert set(result.columns) == {"macd", "signal", "histogram"}

    def test_length(self):
        df = _make_bars(200)
        result = macd(df)
        assert len(result) == len(df)

    def test_histogram_equals_diff(self):
        df = _make_bars(200)
        m = macd(df)
        diff = m["macd"] - m["signal"]
        pd.testing.assert_series_equal(m["histogram"], diff, check_names=False)


class TestBollingerBands:
    def test_columns(self):
        df = _make_bars(100)
        bb = bollinger_bands(df)
        assert {"bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_pct"}.issubset(bb.columns)

    def test_upper_above_lower(self):
        df = _make_bars(200)
        bb = bollinger_bands(df).dropna()
        assert (bb["bb_upper"] > bb["bb_lower"]).all()

    def test_mid_between(self):
        df = _make_bars(200)
        bb = bollinger_bands(df).dropna()
        assert ((bb["bb_mid"] >= bb["bb_lower"]) & (bb["bb_mid"] <= bb["bb_upper"])).all()


class TestATR:
    def test_positive(self):
        df = _make_bars(100)
        a = atr(df, 14).dropna()
        assert (a > 0).all()

    def test_length(self):
        df = _make_bars(100)
        assert len(atr(df, 14)) == len(df)


class TestOBV:
    def test_monotone_up(self):
        df = _make_bars(50)
        df["close"] = range(30000, 30050)  # steadily rising
        o = obv(df)
        # OBV should be generally increasing
        assert o.iloc[-1] > o.iloc[0]

    def test_length(self):
        df = _make_bars(100)
        assert len(obv(df)) == len(df)


class TestComputeAll:
    def test_returns_dataframe(self):
        df = _make_bars(300)
        result = compute_all(df)
        assert isinstance(result, pd.DataFrame)

    def test_has_key_columns(self):
        df = _make_bars(300)
        result = compute_all(df)
        for col in ["ema_200", "rsi_14", "macd", "macd_hist", "bb_upper", "atr_14", "adx"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_future_leakage(self):
        """Slice at bar 250 — indicators should be identical to full-set indicators at that bar."""
        df = _make_bars(300)
        full = compute_all(df)
        partial = compute_all(df.iloc[:251])
        # EMA is computed causally — last value of partial must match full at index 250
        assert abs(float(partial["ema_20"].iloc[-1]) - float(full["ema_20"].iloc[250])) < 1e-6
