"""
FeeOptimizer — chooses order type (LIMIT / MARKET) per signal and screens marginal trades.

Decision logic:
  - If expected_return < min_net_return → return None (screen out the trade entirely)
  - If confidence > 0.85 (high urgency) or spread too wide → return MARKET
  - Otherwise → return LIMIT (capture maker fee savings)
"""
from __future__ import annotations

import dataclasses

from privateye.core.types import Order, OrderType, TradingSignal
from privateye.utils.logging import get_logger

log = get_logger()


class FeeOptimizer:
    def __init__(
        self,
        fee_maker: float = 0.001,
        fee_taker: float = 0.001,
        slippage_pct: float = 0.0005,
        min_net_return: float = 0.003,  # minimum 0.3% expected return after all costs
        limit_patience_bars: int = 3,
        urgency_confidence_threshold: float = 0.85,
        wide_spread_threshold: float = 0.005,  # 0.5% spread = use MARKET
    ) -> None:
        self.fee_maker = fee_maker
        self.fee_taker = fee_taker
        self.slippage_pct = slippage_pct
        self.min_net_return = min_net_return
        self.limit_patience_bars = limit_patience_bars
        self.urgency_confidence_threshold = urgency_confidence_threshold
        self.wide_spread_threshold = wide_spread_threshold

    def choose_order_type(
        self,
        signal: TradingSignal,
        spread_pct: float = 0.001,
    ) -> OrderType | None:
        """
        Returns the recommended OrderType for this signal, or None to screen it out.

        None means the trade's expected return does not cover round-trip costs.
        """
        if signal.entry_price <= 0:
            return None

        expected_return = abs(signal.target_price - signal.entry_price) / signal.entry_price

        # Screen out marginal trades where net return < min threshold
        if expected_return < self.min_net_return:
            log.debug(
                f"[FeeOptimizer] Signal screened: expected_return={expected_return:.4f} "
                f"< min_net_return={self.min_net_return:.4f} [{signal.symbol}]"
            )
            return None

        # High-urgency or wide spread: use MARKET to guarantee execution
        if signal.confidence >= self.urgency_confidence_threshold or spread_pct >= self.wide_spread_threshold:
            return OrderType.MARKET

        return OrderType.LIMIT

    def annotate_order(self, order: Order, order_type: OrderType) -> Order:
        """Return a new Order with updated order_type and post_only flag."""
        return dataclasses.replace(
            order,
            order_type=order_type,
            post_only=(order_type == OrderType.LIMIT),
        )
