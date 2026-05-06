"""
TWAP (Time-Weighted Average Price) order slicer.

Splits a large order into N equal-quantity sub-orders to reduce market impact.
Each slice gets a unique ID suffix. Intended for use with ExecutionEngine when
order notional exceeds the configured threshold.
"""
from __future__ import annotations

import dataclasses

from privateye.core.types import Order
from privateye.utils.logging import get_logger

log = get_logger()


class TWAPExecutor:
    def __init__(self, n_slices: int = 4) -> None:
        if n_slices < 1:
            raise ValueError(f"n_slices must be >= 1, got {n_slices}")
        self.n_slices = n_slices

    def split(self, order: Order) -> list[Order]:
        """
        Split `order` into self.n_slices equal-quantity sub-orders.

        Each slice has quantity = order.quantity / n_slices and an ID suffix _t0, _t1, ...
        All other fields (symbol, side, type, price, stop_price, strategy_id) are preserved.
        """
        slice_qty = order.quantity / self.n_slices
        if slice_qty <= 0:
            log.warning(f"[TWAPExecutor] Cannot split order with qty={order.quantity}")
            return [order]

        slices = [
            dataclasses.replace(order, id=f"{order.id}_t{i}", quantity=slice_qty)
            for i in range(self.n_slices)
        ]
        log.debug(
            f"[TWAPExecutor] Split order {order.id} into {self.n_slices} slices "
            f"of qty={slice_qty:.6f}"
        )
        return slices
