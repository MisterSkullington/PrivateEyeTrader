"""
Simulated exchange for backtesting and paper trading.

Models:
  - Maker/taker fees (configurable)
  - Percentage slippage applied to market orders only
  - Partial fills: capped at max_fill_pct_of_volume × bar volume
  - Market orders fill at close + slippage of the bar they are submitted on
  - Limit orders fill when bar crosses the limit price (no slippage, maker fee)
  - Unfilled limit orders are re-queued up to limit_patience_bars then cancelled
"""
from __future__ import annotations

import dataclasses
import uuid
from datetime import datetime, timezone
from typing import Any

from privateye.core.types import (
    Direction, Fill, Order, OrderSide, OrderStatus, OrderType,
    PortfolioState, Position, TradeRecord,
)
from privateye.utils.logging import get_logger

log = get_logger()


class SimulatedExchange:
    def __init__(
        self,
        initial_capital: float,
        fee_maker: float = 0.001,
        fee_taker: float = 0.001,
        slippage_pct: float = 0.0005,
        max_fill_pct_of_volume: float = 0.30,
        limit_patience_bars: int = 3,
    ) -> None:
        self.fee_maker = fee_maker
        self.fee_taker = fee_taker
        self.slippage_pct = slippage_pct
        self.max_fill_pct_of_volume = max_fill_pct_of_volume
        self.limit_patience_bars = limit_patience_bars

        self.cash: float = initial_capital
        self.positions: dict[str, Position] = {}
        self.equity: float = initial_capital
        self.peak_equity: float = initial_capital
        self.daily_pnl: float = 0.0
        self.daily_start_equity: float = initial_capital
        self.trade_records: list[TradeRecord] = []
        self.fills: list[Fill] = []
        self._pending_orders: list[Order] = []
        self._last_date: str = ""

        # Age tracking for limit orders (order_id → bars waiting)
        self._order_ages: dict[str, int] = {}

        # Cost stats for BacktestReport
        self._limit_orders_placed: int = 0
        self._limit_orders_with_fill: set[str] = set()  # base order IDs that got at least one fill
        self._total_slippage_cost: float = 0.0

    # ── Order lifecycle ─────────────────────────────────────────────────────

    def submit_order(self, order: Order) -> None:
        if order.order_type == OrderType.LIMIT:
            self._limit_orders_placed += 1
            self._order_ages[order.id] = 0
        self._pending_orders.append(order)

    def process_bar(self, symbol: str, bar: dict[str, Any]) -> list[Fill]:
        """
        Process all pending orders for a symbol against the current bar.
        Returns list of fills generated this bar.
        """
        bar_date = str(bar.get("timestamp", ""))[:10]
        if bar_date != self._last_date:
            self.daily_start_equity = self.equity
            self.daily_pnl = 0.0
            self._last_date = bar_date

        new_fills: list[Fill] = []
        remaining: list[Order] = []

        for order in self._pending_orders:
            if order.symbol != symbol:
                remaining.append(order)
                continue

            fill = self._try_fill(order, bar)

            if fill:
                new_fills.append(fill)
                self.fills.append(fill)
                self._apply_fill(fill, order)

                # Track limit order fills by base ID (strip _r suffix from partial re-queues)
                if order.order_type == OrderType.LIMIT:
                    base_id = order.id.split("_r")[0]
                    self._limit_orders_with_fill.add(base_id)

                # Partial fill: requeue remainder for limit orders
                remaining_qty = order.quantity - fill.quantity
                if order.order_type == OrderType.LIMIT and remaining_qty > order.quantity * 0.01:
                    remainder = dataclasses.replace(
                        order,
                        quantity=remaining_qty,
                        status=OrderStatus.PARTIALLY_FILLED,
                        id=order.id + "_r",
                        filled_quantity=0.0,
                        average_fill_price=0.0,
                    )
                    # Inherit parent age
                    self._order_ages[remainder.id] = self._order_ages.get(order.id, 0)
                    remaining.append(remainder)

            else:
                if order.order_type == OrderType.LIMIT:
                    # Age the limit order; cancel only when patience exceeded
                    age = self._order_ages.get(order.id, 0) + 1
                    if age >= self.limit_patience_bars:
                        order.status = OrderStatus.CANCELLED
                        log.debug(f"Limit order expired after {age} bars: {order.id}")
                        self._order_ages.pop(order.id, None)
                    else:
                        self._order_ages[order.id] = age
                        remaining.append(order)
                else:
                    order.status = OrderStatus.CANCELLED
                    log.debug(f"Order cancelled (insufficient volume): {order.id}")

        self._pending_orders = remaining
        self._update_equity(symbol, bar["close"])
        return new_fills

    def _try_fill(self, order: Order, bar: dict[str, Any]) -> Fill | None:
        close = float(bar["close"])
        volume = float(bar["volume"])
        ts = bar.get("timestamp", datetime.now(timezone.utc))
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)

        if order.order_type == OrderType.LIMIT:
            bar_low  = float(bar["low"])
            bar_high = float(bar["high"])
            if order.side == OrderSide.BUY and bar_low <= order.price:
                fill_price = order.price
                fee_rate   = self.fee_maker
            elif order.side == OrderSide.SELL and bar_high >= order.price:
                fill_price = order.price
                fee_rate   = self.fee_maker
            else:
                return None  # price not reached — stays in queue
        else:
            # MARKET order: fill at close + slippage
            if order.side == OrderSide.BUY:
                fill_price = close * (1 + self.slippage_pct)
            else:
                fill_price = close * (1 - self.slippage_pct)
            fee_rate = self.fee_taker
            # Track slippage cost (vs theoretical mid)
            self._total_slippage_cost += abs(fill_price - close) * order.quantity

        # Volume cap (partial fills)
        max_qty = volume * self.max_fill_pct_of_volume / fill_price if fill_price > 0 else 0
        fill_qty = min(order.quantity, max_qty)

        if fill_qty < order.quantity * 0.01:  # less than 1% fill — reject
            return None

        fee = fill_qty * fill_price * fee_rate
        return Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_qty,
            price=fill_price,
            fee=fee,
            strategy_id=order.strategy_id,
            timestamp=ts,
        )

    def _apply_fill(self, fill: Fill, order: Order) -> None:
        order.status = OrderStatus.FILLED
        order.filled_quantity = fill.quantity
        order.average_fill_price = fill.price

        symbol = fill.symbol
        if fill.side == OrderSide.BUY:
            cost = fill.quantity * fill.price + fill.fee
            if cost > self.cash:
                log.warning(f"Overspend on fill — capping. cost={cost:.2f} cash={self.cash:.2f}")
            self.cash = max(0.0, self.cash - cost)
            if symbol not in self.positions:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=Direction.LONG,
                    quantity=fill.quantity,
                    entry_price=fill.price,
                    stop_price=order.stop_price,
                    target_price=0.0,
                    strategy_id=fill.strategy_id,
                    entry_time=fill.timestamp,
                )
        else:  # SELL
            pos = self.positions.get(symbol)
            if pos and pos.side == Direction.LONG:
                realised_pnl = (fill.price - pos.entry_price) * fill.quantity - fill.fee
                self.cash += fill.quantity * fill.price - fill.fee
                fill.realised_pnl = realised_pnl
                self.daily_pnl += realised_pnl
                trade = TradeRecord(
                    symbol=symbol,
                    side=Direction.LONG,
                    entry_price=pos.entry_price,
                    exit_price=fill.price,
                    quantity=fill.quantity,
                    entry_time=pos.entry_time,
                    exit_time=fill.timestamp,
                    pnl=realised_pnl,
                    pnl_pct=realised_pnl / (pos.entry_price * fill.quantity),
                    fees=fill.fee,
                    strategy_id=fill.strategy_id,
                    exit_reason=order.strategy_id,
                    bars_held=pos.bars_held,
                )
                self.trade_records.append(trade)
                if fill.quantity >= pos.quantity * 0.95:
                    del self.positions[symbol]
                else:
                    pos.quantity -= fill.quantity

    def _update_equity(self, symbol: str, current_price: float) -> None:
        pos_value = sum(
            p.quantity * current_price if s == symbol else p.quantity * p.entry_price
            for s, p in self.positions.items()
        )
        self.equity = self.cash + pos_value
        self.peak_equity = max(self.peak_equity, self.equity)

    def get_portfolio_state(self) -> PortfolioState:
        dd = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        daily_dd = (self.daily_start_equity - self.equity) / self.daily_start_equity \
            if self.daily_start_equity > 0 else 0.0
        return PortfolioState(
            equity=self.equity,
            cash=self.cash,
            positions=dict(self.positions),
            daily_pnl=self.daily_pnl,
            daily_drawdown_pct=max(0.0, daily_dd),
            peak_equity=self.peak_equity,
            total_trades=len(self.trade_records),
        )

    def get_cost_stats(self) -> dict:
        """Return cost statistics for BacktestReport trade cost analysis."""
        total_fees = sum(t.fees for t in self.trade_records)
        return {
            "total_fees_paid": total_fees,
            "limit_orders_placed": self._limit_orders_placed,
            "limit_orders_filled": len(self._limit_orders_with_fill),
            "total_slippage_cost": self._total_slippage_cost,
        }
