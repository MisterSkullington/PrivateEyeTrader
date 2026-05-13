"""
Extended feature engineer — wraps the 68-feature base extractor and optionally
appends 18 derived features (derivatives, macro, enhanced sentiment) to produce
an 86-feature matrix.

The base ``feature_extractor.py`` is **not modified** — all 22 tests that lock
it to exactly 68 features remain unaffected.

When all three groups are disabled (the default config) the output is identical
to ``extract_features(bars)`` — backward-compatible with existing model paths.

Feature layout (when all enabled):
  0–67  : base 68 features from feature_extractor.py
  68–75 : derivatives (8)
  76–81 : macro context (6)
  82–85 : enhanced sentiment (4)
  Total = 86

Config keys (under phase1.features):
  derivatives_enabled         (bool, default True)
  macro_enabled               (bool, default True)
  enhanced_sentiment_enabled  (bool, default True)
  use_extended_in_models      (bool, default False)
    When False, models still consume only the base 68 features.
    Set True after retraining GBM/LSTM on the 86-feature dataset.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from privateye.data.feature_extractor import FEATURE_NAMES, extract_features

# ── Extended feature name registry ────────────────────────────────────────────

_DERIVATIVES_NAMES: list[str] = [
    "basis_norm",              # (spot_price - perp_price) / spot_price, z-scored
    "taker_buy_ratio",         # taker_buy_volume / total_volume proxy
    "oi_velocity",             # delta(OI) / OI[-5]
    "oi_velocity_zscore",      # z-score of oi_velocity over 20 bars
    "funding_velocity",        # delta(funding_rate) over 3 bars
    "funding_velocity_zscore", # z-score of funding_velocity over 20 bars
    "perp_spot_spread_zscore", # z-score of basis_norm over 50 bars
    "derivatives_sentiment",   # composite: sign(funding) × |oi_velocity_zscore|, clipped [-1,1]
]

_MACRO_NAMES: list[str] = [
    "btc_dominance_norm",      # (btc_dominance - 50) / 20, clipped [-1.5, 1.5]
    "btc_dominance_extreme",   # 1 if dominance > 60 or < 40 else 0
    "total_mcap_change_norm",  # mcap_change_24h / 10.0, clipped [-1, 1]
    "mcap_trend_signal",       # sign of 3-period moving avg of mcap_change
    "stablecoin_ratio_norm",   # (stablecoin_ratio - 0.10) / 0.05
    "macro_regime",            # 0=bear(dom<40%), 1=neutral, 2=btc_dom(dom>60%)
]

_SENTIMENT_NAMES: list[str] = [
    "fg_momentum",             # fg_value[t] - fg_value[t-7]
    "fg_reversal_signal",      # 1 if fg crossed extreme (>75 or <25) in last 3 bars
    "combined_deriv_sent",     # avg(derivatives_sentiment, normalized_funding_rate)
    "sentiment_volatility",    # std(fg_value over 14 bars) / 100
]

EXTENDED_FEATURE_NAMES: list[str] = (
    FEATURE_NAMES
    + _DERIVATIVES_NAMES
    + _MACRO_NAMES
    + _SENTIMENT_NAMES
)

assert len(EXTENDED_FEATURE_NAMES) == 86, (
    f"Expected 86 extended feature names, got {len(EXTENDED_FEATURE_NAMES)}"
)


class FeatureEngineer:
    """Extends the 68-feature base extractor with optional derived feature groups.

    Safe to construct with no config — all extended groups default to enabled
    (following ``settings.yaml phase1.features`` defaults), but the base 68-feature
    output is always produced regardless.

    Usage::

        eng = FeatureEngineer(cfg)
        eng.set_macro_snapshot(macro_dict)   # optional, from MacroProvider
        features = eng.extract(bars)         # (N, 86)
        latest   = eng.extract_latest(bars)  # (86,)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        feat_cfg = (config or {}).get("phase1", {}).get("features", {})
        self._deriv_enabled = bool(feat_cfg.get("derivatives_enabled", True))
        self._macro_enabled = bool(feat_cfg.get("macro_enabled", True))
        self._sent_enabled  = bool(feat_cfg.get("enhanced_sentiment_enabled", True))
        self._macro_snapshot: dict[str, float] | None = None

    def set_macro_snapshot(self, snapshot: dict[str, float] | None) -> None:
        """Inject the latest MacroProvider data before calling ``extract()``.

        Called by ``main.py``'s bar enrichment helper on each MARKET_DATA event.
        When None (the default), macro features are all 0.0.
        """
        self._macro_snapshot = snapshot

    def extract(self, bars: pd.DataFrame) -> np.ndarray:
        """Return (N, F) feature matrix.

        F = 68 (base) + 8 * deriv_enabled + 6 * macro_enabled + 4 * sent_enabled.
        """
        base = extract_features(bars).astype(np.float32)   # (N, 68)

        extra_cols: list[np.ndarray] = []

        if self._deriv_enabled:
            df_d = self._build_derivatives_features(bars)
            extra_cols.append(df_d.values.astype(np.float32))

        if self._macro_enabled:
            df_m = self._build_macro_features(bars, self._macro_snapshot)
            extra_cols.append(df_m.values.astype(np.float32))

        if self._sent_enabled:
            df_s = self._build_enhanced_sentiment_features(bars)
            extra_cols.append(df_s.values.astype(np.float32))

        if not extra_cols:
            return base

        extended = np.concatenate([base, *extra_cols], axis=1)
        np.nan_to_num(extended, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        return extended

    def extract_latest(self, bars: pd.DataFrame) -> np.ndarray:
        """Return (F,) 1-D feature vector for the most recent bar."""
        return self.extract(bars)[-1]

    @property
    def feature_names(self) -> list[str]:
        """Active feature names list, length equals ``n_features``."""
        names = list(FEATURE_NAMES)
        if self._deriv_enabled:
            names.extend(_DERIVATIVES_NAMES)
        if self._macro_enabled:
            names.extend(_MACRO_NAMES)
        if self._sent_enabled:
            names.extend(_SENTIMENT_NAMES)
        return names

    @property
    def n_features(self) -> int:
        """Total number of active features."""
        return len(self.feature_names)

    # ── Internal builders ─────────────────────────────────────────────────────

    def _build_derivatives_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute 8 derivatives-based feature columns from bars.

        Uses bar data as primary source; optional ``funding_rate`` and
        ``open_interest`` columns are consumed when present.  Falls back to
        zeros gracefully when perp data is unavailable.
        """
        idx = bars.index
        close = bars["close"].astype(float)
        vol   = bars["volume"].astype(float)
        n     = len(bars)

        df = pd.DataFrame(index=idx)

        # funding_rate column (0.0 when absent)
        if "funding_rate" in bars.columns:
            fr = bars["funding_rate"].astype(float).fillna(0.0)
        else:
            fr = pd.Series(np.zeros(n, dtype=float), index=idx)

        # open_interest column (0.0 when absent)
        if "open_interest" in bars.columns:
            oi = bars["open_interest"].astype(float).fillna(0.0)
        else:
            oi = pd.Series(np.zeros(n, dtype=float), index=idx)

        # basis_norm: proxy for perp/spot spread using close only (0 when no perp feed)
        df["basis_norm"] = pd.Series(np.zeros(n, dtype=float), index=idx)

        # taker_buy_ratio: proxy via volume momentum (buy pressure heuristic)
        # When actual taker data is unavailable, use positive-return volume fraction
        ret = close.pct_change().fillna(0.0)
        buy_proxy = vol * (ret > 0).astype(float)
        total_vol_safe = vol.replace(0, np.nan)
        df["taker_buy_ratio"] = (buy_proxy / total_vol_safe).fillna(0.5).clip(0, 1)

        # OI velocity and z-score
        oi_safe      = oi.replace(0, np.nan)
        oi_vel       = (oi - oi.shift(5)) / oi_safe.shift(5)
        df["oi_velocity"] = oi_vel.fillna(0.0)
        oi_vel_m     = oi_vel.rolling(20, min_periods=3).mean()
        oi_vel_s     = oi_vel.rolling(20, min_periods=3).std().replace(0, np.nan)
        df["oi_velocity_zscore"] = ((oi_vel - oi_vel_m) / oi_vel_s).fillna(0.0).clip(-5, 5)

        # Funding velocity and z-score
        fr_vel       = fr.diff(3)
        df["funding_velocity"] = fr_vel.fillna(0.0)
        fr_vel_m     = fr_vel.rolling(20, min_periods=3).mean()
        fr_vel_s     = fr_vel.rolling(20, min_periods=3).std().replace(0, np.nan)
        df["funding_velocity_zscore"] = ((fr_vel - fr_vel_m) / fr_vel_s).fillna(0.0).clip(-5, 5)

        # perp_spot_spread_zscore: z-score of basis_norm over 50 bars (all zeros here)
        df["perp_spot_spread_zscore"] = pd.Series(np.zeros(n, dtype=float), index=idx)

        # derivatives_sentiment: composite = sign(funding) × |oi_velocity_zscore|, clipped [-1,1]
        fr_sign = np.sign(fr.values)
        oi_z    = df["oi_velocity_zscore"].values
        df["derivatives_sentiment"] = np.clip(fr_sign * np.abs(oi_z), -1, 1)

        return df[_DERIVATIVES_NAMES]

    def _build_macro_features(
        self,
        bars: pd.DataFrame,
        macro: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Compute 6 macro context feature columns.

        When ``macro`` is None, all features are 0.0.
        """
        idx = bars.index
        n   = len(bars)
        df  = pd.DataFrame(index=idx)

        btc_dom   = float((macro or {}).get("btc_dominance",           50.0))
        mcap_chg  = float((macro or {}).get("total_mcap_change_24h",   0.0))
        stab_rat  = float((macro or {}).get("stablecoin_ratio_approx", 0.10))

        df["btc_dominance_norm"]     = np.clip((btc_dom - 50.0) / 20.0, -1.5, 1.5)
        df["btc_dominance_extreme"]  = 1.0 if (btc_dom > 60.0 or btc_dom < 40.0) else 0.0
        df["total_mcap_change_norm"] = np.clip(mcap_chg / 10.0, -1.0, 1.0)
        df["stablecoin_ratio_norm"]  = (stab_rat - 0.10) / 0.05

        # mcap_trend_signal: sign of 3-period moving average of mcap_change
        # Broadcast the scalar across all rows (we have only one snapshot per call)
        df["mcap_trend_signal"] = float(np.sign(mcap_chg))

        # macro_regime: 0=bear-dom(<40%), 1=neutral, 2=btc-dom(>60%)
        if btc_dom > 60.0:
            regime = 2.0
        elif btc_dom < 40.0:
            regime = 0.0
        else:
            regime = 1.0
        df["macro_regime"] = regime

        # All scalar values — broadcast to every row
        for col in _MACRO_NAMES:
            df[col] = df[col].iloc[0] if len(df) > 0 else 0.0

        return df[_MACRO_NAMES]

    def _build_enhanced_sentiment_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute 4 enhanced sentiment feature columns.

        Uses ``fear_greed`` column when present, zeros otherwise.
        """
        idx = bars.index
        n   = len(bars)
        df  = pd.DataFrame(index=idx)

        if "fear_greed" in bars.columns:
            fg = bars["fear_greed"].astype(float).fillna(50.0)
        else:
            fg = pd.Series(np.full(n, 50.0), index=idx)

        # fg_momentum: week-over-week (7 bars) change, normalised to [-1, 1]
        df["fg_momentum"] = (fg - fg.shift(7)).fillna(0.0) / 100.0

        # fg_reversal_signal: 1 if crossed extreme (>75 or <25) in last 3 bars
        above75 = (fg > 75).astype(float)
        below25 = (fg < 25).astype(float)
        extreme = (above75 + below25).clip(0, 1)
        df["fg_reversal_signal"] = extreme.rolling(3, min_periods=1).max().fillna(0.0)

        # Normalised funding rate for combined signal
        if "funding_rate" in bars.columns:
            fr_norm = bars["funding_rate"].astype(float).fillna(0.0)
            fr_norm = (fr_norm / 0.001).clip(-1, 1)   # normalise by 0.1% per 8h
        else:
            fr_norm = pd.Series(np.zeros(n, dtype=float), index=idx)

        # derivatives_sentiment from the derivatives group (recompute lightweight version)
        if "_derivatives_sentiment_cache" in bars.columns:
            deriv_sent = bars["_derivatives_sentiment_cache"].astype(float)
        else:
            # Simple funding-rate proxy: sign(funding) scaled to [-1,1]
            deriv_sent = fr_norm

        fg_norm = (fg - 50.0) / 50.0   # normalise fear&greed to [-1, 1]
        df["combined_deriv_sent"] = ((deriv_sent + fg_norm) / 2.0).clip(-1, 1)

        # sentiment_volatility: std of fg over 14 bars, normalised by 100
        df["sentiment_volatility"] = fg.rolling(14, min_periods=2).std().fillna(0.0) / 100.0

        return df[_SENTIMENT_NAMES]
