"""
SlippagePredictor — square-root law market impact model.

Replaces the fixed 5bps slippage assumption used in BacktestEngine and
PreTradeCostAnalyzer with an order-size-aware prediction calibrated to
current market volatility.

Formula:
    impact = alpha * sqrt(notional / avg_daily_volume_usd) * atr_pct

Where:
    notional            = order_qty * price (USD)
    avg_daily_volume_usd = config value (500M USD/day BTC/USDT default)
    atr_pct             = ATR(14).iloc[-1] / close.iloc[-1]
    alpha               = 0.1 (calibration constant)

LightGBM residual correction is deferred to Phase 4 (continuous adaptation).
record_fill() accumulates ShadowFillRecord data for future training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from privateye.utils.logging import get_logger

if TYPE_CHECKING:
    from privateye.execution.shadow_tracker import ShadowFillRecord

log = get_logger()


@dataclass
class SlippagePrediction:
    slippage_pct: float     # expected slippage as a decimal (0.0003 = 3 bps)
    is_modeled: bool        # True when LightGBM residual active; False = pure formula
    impact_bps: float       # same value in basis points
    confidence_lo: float    # lower bound (slippage_pct × 0.5)
    confidence_hi: float    # upper bound (slippage_pct × 2.0)


class SlippagePredictor:
    """Square-root law market impact model.

    Design notes:
    - Deterministic pure-math; no I/O; never raises.
    - Falls back to min_slippage_pct on bad inputs (zero notional, no bars, etc.).
    - record_fill() accumulates ShadowFillRecord data for future LightGBM training.
    - Uses privateye.indicators.library.atr() if bars are long enough (≥ 14 rows);
      falls back to a 0.2% default volatility proxy otherwise.
    """

    _FALLBACK_ATR_PCT: float = 0.002  # 0.2% — conservative fallback when ATR unavailable

    def __init__(
        self,
        alpha: float = 0.1,
        avg_daily_volume_usd: float = 5e8,   # 500M USD/day (BTC/USDT default)
        min_slippage_pct: float = 0.0001,    # 1 bps floor
        max_slippage_pct: float = 0.005,     # 50 bps ceiling
    ) -> None:
        self.alpha                = alpha
        self.avg_daily_volume_usd = max(avg_daily_volume_usd, 1.0)  # guard against zero
        self.min_slippage_pct     = min_slippage_pct
        self.max_slippage_pct     = max_slippage_pct
        self._fill_history: list = []  # list[ShadowFillRecord] for future training

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        order_notional: float,
        bars: pd.DataFrame | None = None,
    ) -> SlippagePrediction:
        """Compute predicted one-way slippage for an order.

        Parameters
        ----------
        order_notional : float
            Order size in USD (qty × price).
        bars : pd.DataFrame | None
            Recent OHLCV bars used to compute ATR-based volatility.
            If None or fewer than 14 rows, a 0.2% volatility proxy is used.

        Returns
        -------
        SlippagePrediction
            Predicted slippage and confidence interval.
        """
        if order_notional <= 0:
            return self._zero_prediction()

        atr_pct = self._compute_atr_pct(bars)
        impact  = self.alpha * (order_notional / self.avg_daily_volume_usd) ** 0.5 * atr_pct
        impact  = float(np.clip(impact, self.min_slippage_pct, self.max_slippage_pct))

        return SlippagePrediction(
            slippage_pct  = impact,
            is_modeled    = False,            # LightGBM residual not yet active
            impact_bps    = impact * 10_000,
            confidence_lo = impact * 0.5,
            confidence_hi = min(impact * 2.0, self.max_slippage_pct),
        )

    # ── Fill recording (future training) ─────────────────────────────────────

    def record_fill(self, record: "ShadowFillRecord") -> None:
        """Accumulate ShadowFillRecord data for future LightGBM residual training."""
        self._fill_history.append(record)

    @property
    def n_fills_recorded(self) -> int:
        return len(self._fill_history)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_atr_pct(self, bars: pd.DataFrame | None) -> float:
        """Return ATR(14) / close as a volatility proxy.  Falls back to 0.2%."""
        if bars is None or len(bars) < 14:
            return self._FALLBACK_ATR_PCT
        try:
            from privateye.indicators.library import atr as compute_atr
            close = float(bars["close"].iloc[-1])
            if close <= 0:
                return self._FALLBACK_ATR_PCT
            atr_val = float(compute_atr(bars, 14).iloc[-1])
            return max(atr_val / close, self._FALLBACK_ATR_PCT)
        except Exception as exc:
            log.debug(f"[SlippagePredictor] ATR computation failed: {exc}; using fallback")
            return self._FALLBACK_ATR_PCT

    def _zero_prediction(self) -> SlippagePrediction:
        return SlippagePrediction(
            slippage_pct  = self.min_slippage_pct,
            is_modeled    = False,
            impact_bps    = self.min_slippage_pct * 10_000,
            confidence_lo = 0.0,
            confidence_hi = self.min_slippage_pct,
        )
