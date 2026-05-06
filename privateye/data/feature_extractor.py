"""
Feature extractor: builds a (N, 68) float32 matrix from a bars DataFrame.
All computations are causal — row i uses only bars[:i+1].
NaN-safe: warmup rows where indicators haven't converged fill with 0.0.

Features 0-49:  original 50 indicators (price, momentum, volatility, volume, trend, regime, time)
Features 50-59: multi-scale 1h-based MTF proxies (4h/1d surrogates)
Features 60-67: alt data (funding_rate, open_interest, fear_greed) — read from optional columns
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAMES: list[str] = [
    # Price-based (10)
    "ret_1", "ret_5", "ret_10", "ret_20",
    "close_vs_sma20", "close_vs_ema200",
    "hl_range_atr", "gap_atr", "upper_wick_atr", "lower_wick_atr",
    # Momentum (8)
    "rsi_14", "rsi_7", "macd_hist_norm", "macd_line_norm",
    "stoch_k", "stoch_d", "roc_10", "roc_20",
    # Volatility (6)
    "atr_norm", "bb_width", "bb_pct", "rvol_10", "rvol_20", "vol_regime",
    # Volume (6)
    "obv_zscore", "vol_ratio", "mfi_14", "vwap_dev", "cum_delta_norm", "vol_zscore",
    # Trend (8)
    "adx", "plus_di", "minus_di",
    "ema9_ema21", "ema21_ema55", "ema20_slope", "ema50_slope", "trend_strength",
    # Regime hints (4)
    "rolling_sharpe_20", "rolling_skew_20", "rolling_kurt_20", "autocorr_1",
    # Time cyclical (8)
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "woy_sin", "woy_cos", "month_sin", "month_cos",
    # Multi-scale MTF proxies (10) — computed from 1h bars, causal
    "rsi_56", "rsi_168", "ema_96_ratio", "ema_480_ratio", "adx_56",
    "macd_sign_48", "bb_pct_240", "rvol_60", "ret_60", "trend_alignment",
    # Alt data (8) — from optional bars columns; 0.0 when absent
    "funding_rate_norm", "funding_rate_zscore", "oi_change_pct", "oi_zscore",
    "fear_greed_norm", "fear_greed_zscore", "fear_greed_extreme", "combined_sentiment",
]

assert len(FEATURE_NAMES) == 68


def extract_features(bars: pd.DataFrame) -> np.ndarray:
    """
    Compute the 68-feature matrix from a OHLCV bars DataFrame.

    Args:
        bars: DataFrame with columns [timestamp, open, high, low, close, volume].
              May also contain optional alt-data columns:
              funding_rate, open_interest, fear_greed.
              Latest row = current bar. Must have at least 1 row.

    Returns:
        np.ndarray of shape (N, 68), dtype float32. NaN/Inf replaced with 0.
    """
    feats = pd.DataFrame(index=bars.index)

    close = bars["close"].astype(float)
    high  = bars["high"].astype(float)
    low   = bars["low"].astype(float)
    open_ = bars["open"].astype(float)
    vol   = bars["volume"].astype(float)

    # ATR(14) — used by many features below
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low  - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()
    atr14s = atr14.replace(0, np.nan)  # safe divisor

    # ── Price-based (10) ─────────────────────────────────────────────────────
    feats["ret_1"]  = close.pct_change(1)
    feats["ret_5"]  = close.pct_change(5)
    feats["ret_10"] = close.pct_change(10)
    feats["ret_20"] = close.pct_change(20)

    sma20   = close.rolling(20).mean()
    ema200  = close.ewm(span=200, adjust=False).mean()
    feats["close_vs_sma20"]  = close / sma20.replace(0, np.nan) - 1
    feats["close_vs_ema200"] = close / ema200.replace(0, np.nan) - 1
    feats["hl_range_atr"]    = hl / atr14s
    feats["gap_atr"]         = (open_ - close.shift()) / atr14s
    body_top = pd.concat([close, open_], axis=1).max(axis=1)
    body_bot = pd.concat([close, open_], axis=1).min(axis=1)
    feats["upper_wick_atr"] = (high - body_top) / atr14s
    feats["lower_wick_atr"] = (body_bot - low)  / atr14s

    # ── Momentum (8) ─────────────────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    ag14  = gain.ewm(alpha=1 / 14, adjust=False).mean()
    al14  = loss.ewm(alpha=1 / 14, adjust=False).mean()
    ag7   = gain.ewm(alpha=1 / 7,  adjust=False).mean()
    al7   = loss.ewm(alpha=1 / 7,  adjust=False).mean()
    feats["rsi_14"] = (100 - 100 / (1 + ag14 / al14.replace(0, np.nan))) / 100.0
    feats["rsi_7"]  = (100 - 100 / (1 + ag7  / al7.replace(0, np.nan)))  / 100.0

    ema12    = close.ewm(span=12, adjust=False).mean()
    ema26    = close.ewm(span=26, adjust=False).mean()
    macd_l   = ema12 - ema26
    macd_sig = macd_l.ewm(span=9, adjust=False).mean()
    feats["macd_hist_norm"] = (macd_l - macd_sig) / atr14s
    feats["macd_line_norm"] = macd_l / close.replace(0, np.nan) * 100

    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_k = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    feats["stoch_k"] = stoch_k / 100.0
    feats["stoch_d"] = stoch_k.rolling(3).mean() / 100.0

    feats["roc_10"] = close.pct_change(10)
    feats["roc_20"] = close.pct_change(20)

    # ── Volatility (6) ───────────────────────────────────────────────────────
    feats["atr_norm"] = atr14 / close.replace(0, np.nan)

    bb_std   = close.rolling(20).std()
    bb_mid   = sma20
    bb_range = (4 * bb_std).replace(0, np.nan)
    feats["bb_width"] = (2 * bb_std) / bb_mid.replace(0, np.nan)
    feats["bb_pct"]   = (close - (bb_mid - 2 * bb_std)) / bb_range

    log_ret = np.log(close / close.shift(1))
    feats["rvol_10"]   = log_ret.rolling(10).std()
    feats["rvol_20"]   = log_ret.rolling(20).std()
    atr100 = tr.ewm(alpha=1 / 100, adjust=False).mean()
    feats["vol_regime"] = atr14 / atr100.replace(0, np.nan)

    # ── Volume (6) ───────────────────────────────────────────────────────────
    obv_v   = (np.sign(close.diff()).fillna(0) * vol).cumsum()
    obv_m   = obv_v.rolling(20).mean()
    obv_s   = obv_v.rolling(20).std().replace(0, np.nan)
    feats["obv_zscore"] = (obv_v - obv_m) / obv_s

    vol_sma20 = vol.rolling(20).mean().replace(0, np.nan)
    feats["vol_ratio"] = vol / vol_sma20

    typical = (high + low + close) / 3
    raw_mf  = typical * vol
    tp_dir  = typical.diff()
    pos_mf  = raw_mf.where(tp_dir > 0,  0).rolling(14).sum()
    neg_mf  = raw_mf.where(tp_dir <= 0, 0).rolling(14).sum()
    feats["mfi_14"] = (100 - 100 / (1 + pos_mf / neg_mf.replace(0, np.nan))) / 100.0

    if "timestamp" in bars.columns:
        try:
            ts       = pd.to_datetime(bars["timestamp"])
            date_grp = ts.dt.date
        except Exception:
            date_grp = pd.Series(range(len(bars)), index=bars.index)
    else:
        date_grp = pd.Series(range(len(bars)), index=bars.index)
    vwap_v = (typical * vol).groupby(date_grp).cumsum() / vol.groupby(date_grp).cumsum().replace(0, np.nan)
    feats["vwap_dev"] = (close - vwap_v) / atr14s

    cum_d  = (np.sign(close - open_) * vol).cumsum()
    cd_m   = cum_d.rolling(20).mean()
    cd_s   = cum_d.rolling(20).std().replace(0, np.nan)
    feats["cum_delta_norm"] = (cum_d - cd_m) / cd_s

    vol_m = vol.rolling(20).mean()
    vol_s = vol.rolling(20).std().replace(0, np.nan)
    feats["vol_zscore"] = (vol - vol_m) / vol_s

    # ── Trend (8) ────────────────────────────────────────────────────────────
    pdm = high.diff().clip(lower=0)
    mdm = (-low.diff()).clip(lower=0)
    pdm_c = pdm.where(pdm > mdm, 0)
    mdm_c = mdm.where(mdm > pdm, 0)
    plus_di  = 100 * pdm_c.ewm(alpha=1 / 14, adjust=False).mean() / atr14s
    minus_di = 100 * mdm_c.ewm(alpha=1 / 14, adjust=False).mean() / atr14s
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_v    = dx.ewm(alpha=1 / 14, adjust=False).mean()
    feats["adx"]      = adx_v   / 100.0
    feats["plus_di"]  = plus_di  / 100.0
    feats["minus_di"] = minus_di / 100.0

    ema9  = close.ewm(span=9,  adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema55 = close.ewm(span=55, adjust=False).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    feats["ema9_ema21"]  = ema9  / ema21.replace(0, np.nan) - 1
    feats["ema21_ema55"] = ema21 / ema55.replace(0, np.nan) - 1

    e20s = ema20.shift(5).replace(0, np.nan)
    e50s = ema50.shift(5).replace(0, np.nan)
    feats["ema20_slope"]    = (ema20 - ema20.shift(5)) / (e20s * 5)
    feats["ema50_slope"]    = (ema50 - ema50.shift(5)) / (e50s * 5)
    feats["trend_strength"] = (ema9 - ema200) / atr14s

    # ── Regime hints (4) ─────────────────────────────────────────────────────
    r1   = close.pct_change(1)
    r_m  = r1.rolling(20).mean()
    r_s  = r1.rolling(20).std().replace(0, np.nan)
    feats["rolling_sharpe_20"] = (r_m / r_s * np.sqrt(20)).clip(-5, 5)
    feats["rolling_skew_20"]   = r1.rolling(20).skew().clip(-5, 5)
    feats["rolling_kurt_20"]   = (r1.rolling(20).kurt() / 10.0).clip(-1, 1)

    r1_lag = r1.shift(1)
    feats["autocorr_1"] = r1.rolling(20).corr(r1_lag).fillna(0).clip(-1, 1)

    # ── Time cyclical (8) ────────────────────────────────────────────────────
    if "timestamp" in bars.columns:
        try:
            ts_dt = pd.to_datetime(bars["timestamp"])
            hour  = ts_dt.dt.hour.astype(float)
            dow   = ts_dt.dt.dayofweek.astype(float)
            woy   = ts_dt.dt.isocalendar().week.astype(float)
            month = ts_dt.dt.month.astype(float)
        except Exception:
            hour = dow = woy = month = pd.Series(0.0, index=bars.index)
    else:
        hour = dow = woy = month = pd.Series(0.0, index=bars.index)

    feats["hour_sin"] = np.sin(2 * np.pi * hour  / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour  / 24)
    feats["dow_sin"]  = np.sin(2 * np.pi * dow   / 7)
    feats["dow_cos"]  = np.cos(2 * np.pi * dow   / 7)
    feats["woy_sin"]  = np.sin(2 * np.pi * woy   / 52)
    feats["woy_cos"]  = np.cos(2 * np.pi * woy   / 52)
    feats["month_sin"] = np.sin(2 * np.pi * month / 12)
    feats["month_cos"] = np.cos(2 * np.pi * month / 12)

    # ── Multi-scale MTF proxies (10) — causal 4h/1d surrogates ───────────────
    # RSI(56) ≈ 4h RSI proxy (4×14), RSI(168) ≈ 1d RSI proxy (7×24)
    ag56 = gain.ewm(alpha=1 / 56, adjust=False).mean()
    al56 = loss.ewm(alpha=1 / 56, adjust=False).mean()
    ag168 = gain.ewm(alpha=1 / 168, adjust=False).mean()
    al168 = loss.ewm(alpha=1 / 168, adjust=False).mean()
    feats["rsi_56"]  = (100 - 100 / (1 + ag56  / al56.replace(0, np.nan)))  / 100.0
    feats["rsi_168"] = (100 - 100 / (1 + ag168 / al168.replace(0, np.nan))) / 100.0

    ema96  = close.ewm(span=96,  adjust=False).mean()
    ema480 = close.ewm(span=480, adjust=False).mean()
    feats["ema_96_ratio"]  = close / ema96.replace(0, np.nan) - 1
    feats["ema_480_ratio"] = close / ema480.replace(0, np.nan) - 1

    # ADX(56) ≈ 4h trend strength
    pdm56 = pdm_c.ewm(alpha=1 / 56, adjust=False).mean()
    mdm56 = mdm_c.ewm(alpha=1 / 56, adjust=False).mean()
    atr56 = tr.ewm(alpha=1 / 56, adjust=False).mean().replace(0, np.nan)
    pdi56 = 100 * pdm56 / atr56
    mdi56 = 100 * mdm56 / atr56
    dx56  = 100 * (pdi56 - mdi56).abs() / (pdi56 + mdi56).replace(0, np.nan)
    feats["adx_56"] = dx56.ewm(alpha=1 / 56, adjust=False).mean() / 100.0

    # MACD sign(48, 104, 36) ≈ 4h MACD direction
    ema48  = close.ewm(span=48,  adjust=False).mean()
    ema104 = close.ewm(span=104, adjust=False).mean()
    macd48 = ema48 - ema104
    macd48_sig = macd48.ewm(span=36, adjust=False).mean()
    feats["macd_sign_48"] = np.sign(macd48 - macd48_sig)

    # BB %b over 240 bars (≈10-day)
    bb240_mid = close.rolling(240).mean()
    bb240_std = close.rolling(240).std()
    bb240_rng = (4 * bb240_std).replace(0, np.nan)
    feats["bb_pct_240"] = (close - (bb240_mid - 2 * bb240_std)) / bb240_rng

    feats["rvol_60"] = log_ret.rolling(60).std()
    feats["ret_60"]  = close.pct_change(60)

    # Trend alignment: sign(ret_5) × sign(ret_20) × sign(ret_60) ∈ {-1, 0, 1}
    feats["trend_alignment"] = (
        np.sign(close.pct_change(5)) *
        np.sign(close.pct_change(20)) *
        np.sign(close.pct_change(60))
    )

    # ── Alt data (8) — read optional columns; default 0 if absent ────────────
    def _get_col(name: str) -> pd.Series:
        if name in bars.columns:
            return bars[name].astype(float).fillna(0.0)
        return pd.Series(0.0, index=bars.index)

    fr  = _get_col("funding_rate")
    oi  = _get_col("open_interest")
    fg  = _get_col("fear_greed")

    feats["funding_rate_norm"] = fr / 0.001  # 0.1% normalization

    fr_m = fr.rolling(720, min_periods=1).mean()   # ~30 days of 1h bars
    fr_s = fr.rolling(720, min_periods=1).std().replace(0, np.nan)
    feats["funding_rate_zscore"] = (fr - fr_m) / fr_s

    feats["oi_change_pct"] = oi.pct_change(4)

    oi_m = oi.rolling(720, min_periods=1).mean()
    oi_s = oi.rolling(720, min_periods=1).std().replace(0, np.nan)
    feats["oi_zscore"] = (oi - oi_m) / oi_s

    feats["fear_greed_norm"] = fg / 100.0

    fg_m = fg.rolling(720, min_periods=1).mean()
    fg_s = fg.rolling(720, min_periods=1).std().replace(0, np.nan)
    feats["fear_greed_zscore"] = (fg - fg_m) / fg_s

    feats["fear_greed_extreme"] = ((fg < 20) | (fg > 80)).astype(float)

    # Contrarian: negative funding (shorts pay) + high fear = potential long
    feats["combined_sentiment"] = np.sign(-fr) * (fg / 100.0)

    # ── Assemble ─────────────────────────────────────────────────────────────
    arr = feats[FEATURE_NAMES].values.astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def extract_latest(bars: pd.DataFrame) -> np.ndarray:
    """Return features for the most recent bar only. Shape: (68,)."""
    return extract_features(bars)[-1]
