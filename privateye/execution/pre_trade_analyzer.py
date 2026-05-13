"""
PreTradeCostAnalyzer — Gate 11 for RiskManager.

Hard cost filter that rejects entry signals whose expected edge, after
accounting for round-trip fees and predicted slippage, falls below a
configurable threshold (default 0.25%).

Cost model (round-trip):
    fee_cost_pct      = 2 × fee_taker
    slippage_cost_pct = 2 × predicted_one_way_slippage
    net_edge_pct      = expected_return_pct − fee_cost_pct − slippage_cost_pct

Expected return estimate:
    |target_price − entry_price| / entry_price × signal.confidence

When target_price is unavailable or zero, the model falls back to:
    |stop_price − entry_price| / entry_price × 2.0 × signal.confidence
    (assumes a 2:1 reward-to-risk ratio for the stop-loss distance)

Disabled by default — enable via:
    execution.pre_trade_analysis.enabled: true
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from privateye.utils.logging import get_logger

if TYPE_CHECKING:
    from privateye.core.types import PortfolioState, TradingSignal
    from privateye.execution.slippage_predictor import SlippagePredictor

log = get_logger()


@dataclass
class PreTradeResult:
    expected_return_pct: float  # signal.confidence × |target − entry| / entry
    fee_cost_pct: float         # 2 × fee_taker (round-trip)
    slippage_cost_pct: float    # 2 × predicted one-way slippage (round-trip)
    net_edge_pct: float         # expected − fee − slippage
    approved: bool              # net_edge_pct >= min_net_edge_pct
    reason: str                 # human-readable explanation


class PreTradeCostAnalyzer:
    """Pre-trade net edge calculator — plugs into RiskManager as Gate 11.

    Parameters
    ----------
    slippage_predictor : SlippagePredictor
    min_net_edge_pct : float
        Minimum acceptable net edge after round-trip costs (default 0.25%).
    fee_taker : float
        Taker fee rate (default 0.1%).
    """

    def __init__(
        self,
        slippage_predictor: SlippagePredictor,
        min_net_edge_pct: float = 0.0025,
        fee_taker: float = 0.001,
    ) -> None:
        self._predictor      = slippage_predictor
        self.min_net_edge_pct = min_net_edge_pct
        self._fee_taker      = fee_taker

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(
        self,
        signal: TradingSignal,
        order_qty: float,
        portfolio: PortfolioState,
        bars: pd.DataFrame | None = None,
    ) -> PreTradeResult:
        """Compute expected return, costs, and net edge for *signal*.

        Never raises — returns approved=True with zeroed costs when inputs are
        degenerate (zero entry price, missing bars, etc.).
        """
        try:
            return self._analyze_impl(signal, order_qty, portfolio, bars)
        except Exception as exc:
            log.debug(f"[PreTradeCostAnalyzer] Analysis failed ({exc}); approving by default")
            return PreTradeResult(
                expected_return_pct = 0.0,
                fee_cost_pct        = 0.0,
                slippage_cost_pct   = 0.0,
                net_edge_pct        = 0.0,
                approved            = True,
                reason              = f"analysis_error: {exc}",
            )

    # ── Private ───────────────────────────────────────────────────────────────

    def _analyze_impl(
        self,
        signal: TradingSignal,
        order_qty: float,
        portfolio: PortfolioState,
        bars: pd.DataFrame | None,
    ) -> PreTradeResult:
        entry  = float(signal.entry_price)
        target = float(getattr(signal, "target_price", 0.0) or 0.0)
        stop   = float(getattr(signal, "stop_price",  0.0) or 0.0)
        conf   = float(getattr(signal, "confidence",  0.5))

        if entry <= 0:
            # Cannot compute without a valid entry price
            return PreTradeResult(0.0, 0.0, 0.0, 0.0, True, "entry_price_zero")

        # Expected return
        if target > 0 and abs(target - entry) > 0:
            raw_move = abs(target - entry) / entry
        elif stop > 0 and abs(stop - entry) > 0:
            # 2:1 reward-to-risk proxy
            raw_move = abs(stop - entry) / entry * 2.0
        else:
            # Absolute minimum — assume 0.5% move
            raw_move = 0.005

        expected_return_pct = conf * raw_move

        # Fee cost (round-trip: entry + exit)
        fee_cost_pct = 2.0 * self._fee_taker

        # Slippage cost (round-trip: entry + exit, symmetric)
        notional = order_qty * entry
        pred     = self._predictor.predict(notional, bars)
        slippage_cost_pct = 2.0 * pred.slippage_pct

        # Net edge
        net_edge_pct = expected_return_pct - fee_cost_pct - slippage_cost_pct
        approved     = net_edge_pct >= self.min_net_edge_pct

        reason = (
            f"expected={expected_return_pct:.4%} "
            f"fee={fee_cost_pct:.4%} "
            f"slip={slippage_cost_pct:.4%} "
            f"net={net_edge_pct:.4%} "
            f"({'OK' if approved else 'REJECTED'})"
        )

        return PreTradeResult(
            expected_return_pct = expected_return_pct,
            fee_cost_pct        = fee_cost_pct,
            slippage_cost_pct   = slippage_cost_pct,
            net_edge_pct        = net_edge_pct,
            approved            = approved,
            reason              = reason,
        )
