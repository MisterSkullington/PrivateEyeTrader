"""
Technical indicator library. All functions accept a DataFrame with columns:
    timestamp, open, high, low, close, volume

and return the indicator Series/DataFrame computed on that slice only.
No future data is ever referenced — safe to call on bars[:i] during replay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Trend ──────────────────────────────────────────────────────────────────

def ema(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
    return df[col].ewm(span=period, adjust=False).mean()


def sma(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
    return df[col].rolling(period).mean()


def wma(df: pd.DataFrame, period: int, col: str = "close") -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return df[col].rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def dema(df: pd.DataFrame, period: int) -> pd.Series:
    e = ema(df, period)
    return 2 * e - e.ewm(span=period, adjust=False).mean()


def tema(df: pd.DataFrame, period: int) -> pd.Series:
    e1 = ema(df, period)
    e2 = e1.ewm(span=period, adjust=False).mean()
    e3 = e2.ewm(span=period, adjust=False).mean()
    return 3 * e1 - 3 * e2 + e3


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP (resets each day)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]
    # Group by date for daily reset
    date_group = df["timestamp"].dt.date if "timestamp" in df.columns else pd.Series(range(len(df)))
    cum_pv = pv.groupby(date_group).cumsum()
    cum_vol = df["volume"].groupby(date_group).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


# ── Momentum ───────────────────────────────────────────────────────────────

def rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> pd.Series:
    delta = df[col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, col: str = "close"
) -> pd.DataFrame:
    fast_ema = df[col].ewm(span=fast, adjust=False).mean()
    slow_ema = df[col].ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})


def stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def roc(df: pd.DataFrame, period: int = 12, col: str = "close") -> pd.Series:
    return df[col].pct_change(period) * 100


def momentum(df: pd.DataFrame, period: int = 10, col: str = "close") -> pd.Series:
    return df[col] - df[col].shift(period)


def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_max = df["high"].rolling(period).max()
    low_min = df["low"].rolling(period).min()
    return -100 * (high_max - df["close"]) / (high_max - low_min).replace(0, np.nan)


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    mean = typical.rolling(period).mean()
    mad = typical.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    return (typical - mean) / (0.015 * mad.replace(0, np.nan))


# ── Volatility ─────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger_bands(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, col: str = "close"
) -> pd.DataFrame:
    mid = df[col].rolling(period).mean()
    std = df[col].rolling(period).std()
    return pd.DataFrame({
        "bb_upper": mid + std_dev * std,
        "bb_mid": mid,
        "bb_lower": mid - std_dev * std,
        "bb_width": (std_dev * 2 * std) / mid.replace(0, np.nan),
        "bb_pct": (df[col] - (mid - std_dev * std)) / (std_dev * 2 * std).replace(0, np.nan),
    })


def keltner_channels(
    df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0
) -> pd.DataFrame:
    mid = ema(df, ema_period)
    a = atr(df, atr_period)
    return pd.DataFrame({
        "kc_upper": mid + multiplier * a,
        "kc_mid": mid,
        "kc_lower": mid - multiplier * a,
    })


def historical_volatility(df: pd.DataFrame, period: int = 20, col: str = "close") -> pd.Series:
    log_ret = np.log(df[col] / df[col].shift(1))
    return log_ret.rolling(period).std() * np.sqrt(252)


# ── Volume ─────────────────────────────────────────────────────────────────

def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["volume"].rolling(period).mean()


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    raw_mf = typical * df["volume"]
    direction = typical.diff()
    pos_mf = raw_mf.where(direction > 0, 0).rolling(period).sum()
    neg_mf = raw_mf.where(direction <= 0, 0).rolling(period).sum()
    mfr = pos_mf / neg_mf.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def vwma(df: pd.DataFrame, period: int = 20) -> pd.Series:
    pv = df["close"] * df["volume"]
    return pv.rolling(period).sum() / df["volume"].rolling(period).sum().replace(0, np.nan)


# ── Trend strength ─────────────────────────────────────────────────────────

def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    plus_dm = df["high"].diff().clip(lower=0)
    minus_dm = (-df["low"].diff()).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    a = atr(df, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / a.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / a.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()
    return pd.DataFrame({"adx": adx_val, "plus_di": plus_di, "minus_di": minus_di})


def aroon(df: pd.DataFrame, period: int = 25) -> pd.DataFrame:
    aroon_up = df["high"].rolling(period + 1).apply(
        lambda x: (period - x[::-1].argmax()) / period * 100, raw=True
    )
    aroon_down = df["low"].rolling(period + 1).apply(
        lambda x: (period - x[::-1].argmin()) / period * 100, raw=True
    )
    return pd.DataFrame({"aroon_up": aroon_up, "aroon_down": aroon_down,
                         "aroon_osc": aroon_up - aroon_down})


# ── Support / Resistance ────────────────────────────────────────────────────

def pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    prev = df.shift(1)
    pivot = (prev["high"] + prev["low"] + prev["close"]) / 3
    r1 = 2 * pivot - prev["low"]
    s1 = 2 * pivot - prev["high"]
    r2 = pivot + (prev["high"] - prev["low"])
    s2 = pivot - (prev["high"] - prev["low"])
    return pd.DataFrame({"pivot": pivot, "r1": r1, "r2": r2, "s1": s1, "s2": s2})


def donchian_channels(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    return pd.DataFrame({
        "dc_upper": df["high"].rolling(period).max(),
        "dc_lower": df["low"].rolling(period).min(),
        "dc_mid": (df["high"].rolling(period).max() + df["low"].rolling(period).min()) / 2,
    })


# ── Composite ──────────────────────────────────────────────────────────────

def compute_all(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """
    Compute a standard suite of indicators and append as columns.
    Returns a copy of df with all indicator columns added.
    """
    cfg = cfg or {}
    out = df.copy()

    # Trend
    for p in [9, 20, 50, 100, 200]:
        out[f"ema_{p}"] = ema(df, p)
        out[f"sma_{p}"] = sma(df, p)

    # MACD
    macd_df = macd(df, fast=cfg.get("macd_fast", 12), slow=cfg.get("macd_slow", 26),
                   signal=cfg.get("macd_signal", 9))
    out[["macd", "macd_signal", "macd_hist"]] = macd_df.values

    # RSI
    out["rsi_14"] = rsi(df, 14)
    out["rsi_7"] = rsi(df, 7)

    # Stochastic
    stoch_df = stochastic(df)
    out[["stoch_k", "stoch_d"]] = stoch_df.values

    # BB
    bb_df = bollinger_bands(df)
    out[["bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_pct"]] = bb_df.values

    # ATR
    out["atr_14"] = atr(df, 14)
    out["atr_7"] = atr(df, 7)

    # ADX
    adx_df = adx(df)
    out[["adx", "plus_di", "minus_di"]] = adx_df.values

    # Volume
    out["obv"] = obv(df)
    out["volume_sma_20"] = volume_sma(df, 20)
    out["mfi_14"] = mfi(df, 14)

    # Volatility
    out["hv_20"] = historical_volatility(df, 20)

    # Momentum
    out["roc_12"] = roc(df, 12)
    out["cci_20"] = cci(df, 20)
    out["williams_r_14"] = williams_r(df, 14)

    # Donchian
    dc_df = donchian_channels(df, 20)
    out[["dc_upper", "dc_lower", "dc_mid"]] = dc_df.values

    # Derived signals
    out["close_vs_ema200"] = df["close"] / out["ema_200"] - 1
    out["close_vs_bb_mid"] = (df["close"] - out["bb_mid"]) / out["bb_mid"]
    out["volume_ratio"] = df["volume"] / out["volume_sma_20"]
    out["candle_body"] = (df["close"] - df["open"]).abs() / df["open"]
    out["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / df["open"]
    out["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / df["open"]

    return out
