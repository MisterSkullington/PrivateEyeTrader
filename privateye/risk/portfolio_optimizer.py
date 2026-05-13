"""
Portfolio position-size optimizers for Phase 5.

Each optimizer returns a dict[symbol → weight] where weights sum to 1.0.
The weights are used by RiskManager as per-symbol max_position_notional_pct
caps — they represent the maximum fraction of equity to allocate to each
symbol, not an instruction to be 100% invested.

Two implementations:
  EqualWeightOptimizer — trivial N-way split (baseline)
  RiskParityOptimizer  — inverse-volatility weighting (diagonal risk parity)

Usage in RiskManager._compute_size():
  weights = optimizer.compute_weights(symbols, bars_by_symbol)
  effective_max_pct = min(config_max_pct, weights[symbol])
  qty = cap_to_max_notional(qty, price, equity, effective_max_pct)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from privateye.utils.logging import get_logger

log = get_logger()


class PortfolioOptimizer(ABC):
    """Abstract base class for portfolio weighting strategies."""

    @abstractmethod
    def compute_weights(
        self,
        symbols: list[str],
        bars_by_symbol: dict[str, pd.DataFrame],
        lookback: int = 60,
    ) -> dict[str, float]:
        """
        Return {symbol: weight} where sum(weights.values()) ≈ 1.0.

        Falls back to equal weights on any error or insufficient data.
        Never raises.
        """


class EqualWeightOptimizer(PortfolioOptimizer):
    """Baseline: all symbols receive equal notional allocation.

    With N symbols, each gets weight = 1/N.
    """

    def compute_weights(
        self,
        symbols: list[str],
        bars_by_symbol: dict[str, pd.DataFrame],
        lookback: int = 60,
    ) -> dict[str, float]:
        n = len(symbols)
        if n == 0:
            return {}
        w = 1.0 / n
        return {s: w for s in symbols}


class RiskParityOptimizer(PortfolioOptimizer):
    """Inverse-volatility weighting (diagonal risk-parity approximation).

    Each symbol's weight is proportional to 1 / σ_i, where σ_i is the
    rolling annualised daily volatility of log-returns over `lookback` bars.

    Lower-volatility assets receive higher allocations; higher-volatility
    assets receive lower allocations, so each position contributes equally
    to total portfolio risk when correlations are ignored.

    Falls back to equal weights when:
      - fewer than `min_bars` rows are available for any symbol
      - computed volatility is zero for any symbol
      - any unexpected error occurs during computation
    """

    def __init__(
        self,
        min_bars: int = 30,
        use_correlation: bool = False,
    ) -> None:
        self._min_bars = min_bars
        self._use_correlation = use_correlation  # reserved for future extension

    def compute_weights(
        self,
        symbols: list[str],
        bars_by_symbol: dict[str, pd.DataFrame],
        lookback: int = 60,
    ) -> dict[str, float]:
        if not symbols:
            return {}
        if len(symbols) == 1:
            return {symbols[0]: 1.0}

        n = len(symbols)
        equal = {s: 1.0 / n for s in symbols}

        try:
            inv_vols: dict[str, float] = {}
            for sym in symbols:
                bars = bars_by_symbol.get(sym)
                if bars is None or len(bars) < self._min_bars:
                    log.debug(
                        f"[RiskParityOptimizer] {sym}: insufficient bars "
                        f"({0 if bars is None else len(bars)} < {self._min_bars}) "
                        "— falling back to equal weights"
                    )
                    return equal  # need all symbols; any gap → fallback

                vol = self._compute_daily_vol(bars, lookback)
                if vol <= 0.0:
                    log.debug(
                        f"[RiskParityOptimizer] {sym}: vol=0 — falling back to equal weights"
                    )
                    return equal

                inv_vols[sym] = 1.0 / vol

            total_inv = sum(inv_vols.values())
            if total_inv <= 0.0:
                return equal

            return {sym: inv_vols[sym] / total_inv for sym in symbols}

        except Exception as exc:  # pragma: no cover
            log.warning(f"[RiskParityOptimizer] Error computing weights: {exc} — using equal")
            return equal

    def _compute_daily_vol(self, bars: pd.DataFrame, lookback: int) -> float:
        """Rolling std of log-returns on the last `lookback` bars. Returns 0.0 on error."""
        try:
            closes = bars["close"].iloc[-lookback:].astype(float)
            if len(closes) < 2:
                return 0.0
            log_returns = np.log(closes / closes.shift(1)).dropna()
            if len(log_returns) < 2:
                return 0.0
            return float(log_returns.std())
        except Exception:  # pragma: no cover
            return 0.0


def build_portfolio_optimizer(cfg: dict[str, Any]) -> PortfolioOptimizer:
    """Factory: reads cfg['portfolio_backtest']['optimizer'] to select implementation.

    Returns:
        RiskParityOptimizer  when optimizer == "risk_parity"
        EqualWeightOptimizer for any other value (default)
    """
    pb_cfg = cfg.get("portfolio_backtest", {})
    optimizer_type = pb_cfg.get("optimizer", "equal_weight")
    if optimizer_type == "risk_parity":
        return RiskParityOptimizer(
            min_bars=int(pb_cfg.get("min_bars_for_optimizer", 30)),
            use_correlation=bool(pb_cfg.get("use_correlation", False)),
        )
    return EqualWeightOptimizer()
