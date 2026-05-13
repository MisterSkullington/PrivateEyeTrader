"""
SmartOrderRouter — venue-aware order routing with dynamic order-type selection.

Single-venue mode (default — Bybit disabled):
    Acts as an improved FeeOptimizer, using SlippagePredictor instead of a
    fixed spread check for the MARKET/LIMIT decision.

Multi-venue mode (execution.smart_routing.enabled: true + Bybit enabled):
    Scores each venue by (predicted_slippage + fee_rate) and routes to the
    cheapest option when the cost difference exceeds min_score_diff_bps.

Order-type selection logic:
    urgency >= urgency_threshold (default 0.85) → MARKET (fill certainty)
    else                                         → LIMIT  (post_only=True for rebate)
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from privateye.core.types import Order, OrderType
from privateye.utils.logging import get_logger

if TYPE_CHECKING:
    from privateye.execution.slippage_predictor import SlippagePredictor

log = get_logger()


@dataclass
class RoutingDecision:
    venue: str                      # exchange name: "binance" | "bybit"
    order_type: OrderType           # MARKET | LIMIT
    post_only: bool
    predicted_slippage_pct: float
    total_cost_pct: float           # slippage + fee (one-way)
    reason: str


class SmartOrderRouter:
    """Multi-venue cost-minimising order router.

    Parameters
    ----------
    slippage_predictor : SlippagePredictor
    enabled_venues : list[str]
        Exchange names that are currently enabled, e.g. ["binance"].
    fee_rates : dict[str, float]
        Per-venue taker fee rate, e.g. {"binance": 0.001, "bybit": 0.001}.
    enabled : bool
        Multi-venue routing switch.  False = single-venue (passthrough) mode.
    urgency_threshold : float
        Signal confidence above which MARKET order is preferred.
    min_score_diff_bps : float
        Minimum cost difference in bps to trigger a venue switch.
    """

    def __init__(
        self,
        slippage_predictor: SlippagePredictor,
        enabled_venues: list[str],
        fee_rates: dict[str, float],
        enabled: bool = False,
        urgency_threshold: float = 0.85,
        min_score_diff_bps: float = 0.5,
    ) -> None:
        self._predictor         = slippage_predictor
        self._venues            = list(enabled_venues) if enabled_venues else ["binance"]
        self._fee_rates         = fee_rates
        self._enabled           = enabled
        self._urgency_threshold = urgency_threshold
        self._min_diff_bps      = min_score_diff_bps

    # ── Public API ────────────────────────────────────────────────────────────

    def route(
        self,
        order: Order,
        urgency: float = 0.5,
        bars: pd.DataFrame | None = None,
    ) -> RoutingDecision:
        """Determine the best venue and order type for *order*.

        Returns a RoutingDecision describing the chosen venue, order type,
        predicted slippage, and total one-way cost.
        """
        notional = order.quantity * order.price if order.price > 0 else 0.0
        prediction = self._predictor.predict(notional, bars)
        slippage   = prediction.slippage_pct

        # ── Venue selection ───────────────────────────────────────────────────
        if self._enabled and len(self._venues) > 1:
            venue = self._pick_best_venue(slippage)
            reason_prefix = f"multi-venue({len(self._venues)})"
        else:
            venue = self._venues[0] if self._venues else "binance"
            reason_prefix = "single-venue"

        fee_rate    = self._fee_rates.get(venue, 0.001)
        total_cost  = slippage + fee_rate

        # ── Order type selection ──────────────────────────────────────────────
        if urgency >= self._urgency_threshold:
            order_type = OrderType.MARKET
            post_only  = False
            reason     = f"{reason_prefix}: urgency={urgency:.2f} >= {self._urgency_threshold}"
        else:
            order_type = OrderType.LIMIT
            post_only  = True
            reason     = (
                f"{reason_prefix}: urgency={urgency:.2f} < {self._urgency_threshold}; "
                f"predicted_slip={slippage:.4%} fee={fee_rate:.3%}"
            )

        return RoutingDecision(
            venue                  = venue,
            order_type             = order_type,
            post_only              = post_only,
            predicted_slippage_pct = slippage,
            total_cost_pct         = total_cost,
            reason                 = reason,
        )

    def annotate_order(self, order: Order, decision: RoutingDecision) -> Order:
        """Return a copy of *order* with order_type and post_only from *decision*."""
        return dataclasses.replace(
            order,
            order_type = decision.order_type,
            post_only  = decision.post_only,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _pick_best_venue(self, baseline_slippage: float) -> str:
        """Score each venue by (slippage + fee). Return cheapest.

        Uses the same baseline slippage for all venues (venue-specific
        slippage calibration deferred to Phase 4).  Only switches venues
        when the cost difference exceeds min_score_diff_bps.
        """
        scores: dict[str, float] = {}
        for v in self._venues:
            fee = self._fee_rates.get(v, 0.001)
            scores[v] = baseline_slippage + fee

        best    = min(scores, key=scores.__getitem__)
        default = self._venues[0]

        diff_bps = abs(scores[default] - scores[best]) * 10_000
        if diff_bps >= self._min_diff_bps and best != default:
            log.debug(
                f"[SmartOrderRouter] Routing to {best} "
                f"(cost diff={diff_bps:.2f}bps vs {default})"
            )
            return best
        return default
