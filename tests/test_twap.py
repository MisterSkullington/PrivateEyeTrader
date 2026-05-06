"""Tests for TWAPExecutor order slicer."""
from __future__ import annotations

import pytest

from privateye.core.types import Order, OrderSide, OrderType


def _make_order(qty: float = 1.0, symbol: str = "BTC/USDT") -> Order:
    return Order(
        id="ord_001",
        symbol=symbol,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=qty,
        price=50000.0,
        stop_price=49000.0,
        strategy_id="test",
    )


class TestTWAPExecutor:
    def test_import(self):
        from privateye.execution.twap import TWAPExecutor
        ex = TWAPExecutor(n_slices=4)
        assert ex.n_slices == 4

    def test_split_count(self):
        from privateye.execution.twap import TWAPExecutor
        ex = TWAPExecutor(n_slices=4)
        order = _make_order(qty=1.0)
        slices = ex.split(order)
        assert len(slices) == 4

    def test_split_quantity_sum(self):
        from privateye.execution.twap import TWAPExecutor
        ex = TWAPExecutor(n_slices=4)
        order = _make_order(qty=1.0)
        slices = ex.split(order)
        total = sum(s.quantity for s in slices)
        assert total == pytest.approx(1.0, rel=1e-9)

    def test_split_equal_quantities(self):
        from privateye.execution.twap import TWAPExecutor
        ex = TWAPExecutor(n_slices=4)
        order = _make_order(qty=1.0)
        slices = ex.split(order)
        for s in slices:
            assert s.quantity == pytest.approx(0.25, rel=1e-9)

    def test_split_unique_ids(self):
        from privateye.execution.twap import TWAPExecutor
        ex = TWAPExecutor(n_slices=4)
        order = _make_order()
        slices = ex.split(order)
        ids = [s.id for s in slices]
        assert len(set(ids)) == 4

    def test_split_id_format(self):
        from privateye.execution.twap import TWAPExecutor
        ex = TWAPExecutor(n_slices=3)
        order = _make_order()
        slices = ex.split(order)
        assert slices[0].id == "ord_001_t0"
        assert slices[1].id == "ord_001_t1"
        assert slices[2].id == "ord_001_t2"

    def test_split_preserves_fields(self):
        from privateye.execution.twap import TWAPExecutor
        ex = TWAPExecutor(n_slices=2)
        order = _make_order(qty=0.5)
        slices = ex.split(order)
        for s in slices:
            assert s.symbol == order.symbol
            assert s.side == order.side
            assert s.order_type == order.order_type
            assert s.price == order.price
            assert s.stop_price == order.stop_price
            assert s.strategy_id == order.strategy_id

    def test_single_slice(self):
        from privateye.execution.twap import TWAPExecutor
        ex = TWAPExecutor(n_slices=1)
        order = _make_order(qty=0.5)
        slices = ex.split(order)
        assert len(slices) == 1
        assert slices[0].quantity == pytest.approx(0.5)

    def test_invalid_n_slices(self):
        from privateye.execution.twap import TWAPExecutor
        with pytest.raises(ValueError):
            TWAPExecutor(n_slices=0)

    def test_zero_quantity_returns_original(self):
        from privateye.execution.twap import TWAPExecutor
        ex = TWAPExecutor(n_slices=4)
        order = _make_order(qty=0.0)
        slices = ex.split(order)
        assert len(slices) == 1
        assert slices[0].id == order.id  # unchanged original

    def test_small_quantity(self):
        from privateye.execution.twap import TWAPExecutor
        ex = TWAPExecutor(n_slices=5)
        order = _make_order(qty=0.001)
        slices = ex.split(order)
        assert len(slices) == 5
        assert sum(s.quantity for s in slices) == pytest.approx(0.001, rel=1e-6)
