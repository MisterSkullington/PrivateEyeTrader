"""
ExecutionEngine — routes orders to either a SimulatedExchange or a live ExchangeAdapter.

Responsibilities:
  - submit_order: validate, optionally TWAP-slice large orders, route to correct backend
  - flatten_all: emergency close all positions (kill switch)
  - get_portfolio_state: reconcile live balance + open positions → PortfolioState
"""
from __future__ import annotations

import asyncio
from typing import Any

from privateye.core.event_bus import AsyncEventBus
from privateye.core.types import (
    Direction, EventType, Fill, Order, OrderSide, OrderType,
    PortfolioState, Position,
)
from privateye.utils.logging import get_logger
from privateye.utils.time import now_utc

log = get_logger()


class ExecutionEngine:
    def __init__(
        self,
        bus: AsyncEventBus,
        adapter: Any = None,           # ExchangeAdapter for live, None for paper
        simulator: Any = None,         # SimulatedExchange for paper/backtest
        exec_config: dict | None = None,
    ) -> None:
        self.bus = bus
        self.adapter = adapter
        self.simulator = simulator
        self._live_mode = adapter is not None and simulator is None
        self._fills: list[Fill] = []
        self._kill_active = False
        self._exec_cfg: dict = exec_config or {}

    async def submit_order(self, order: Order) -> Fill | None:
        if self._kill_active:
            log.warning(f"Kill switch active — order rejected: {order.symbol}")
            return None

        log.info(f"[ExecutionEngine] Submit: {order.side.value} {order.quantity:.6f} {order.symbol}")

        # Auto-TWAP: slice large orders if enabled
        use_twap   = self._exec_cfg.get("use_twap", False)
        twap_thr   = float(self._exec_cfg.get("twap_threshold_usd", 1000))
        twap_n     = int(self._exec_cfg.get("twap_slices", 4))
        notional   = order.quantity * order.price

        if use_twap and notional > twap_thr:
            from privateye.execution.twap import TWAPExecutor
            slices = TWAPExecutor(twap_n).split(order)
            last_fill = None
            for s in slices:
                last_fill = await self._submit_single(s)
            return last_fill

        return await self._submit_single(order)

    async def _submit_single(self, order: Order) -> Fill | None:
        if self._live_mode and self.adapter:
            if order.order_type == OrderType.LIMIT:
                fill = await self.adapter.create_limit_order(order)
            else:
                fill = await self.adapter.create_market_order(order)
        elif self.simulator:
            self.simulator.submit_order(order)
            return None  # fill comes async via process_bar
        else:
            log.error("No exchange backend configured")
            return None

        if fill:
            self._fills.append(fill)
            await self.bus.publish(EventType.FILL, fill)
        return fill

    async def flatten_all(self, current_prices: dict[str, float], reason: str = "kill") -> None:
        """Close all open positions at market. Used by kill switch."""
        self._kill_active = True
        log.critical(f"[ExecutionEngine] FLATTEN ALL — reason: {reason}")

        if self.simulator:
            positions = dict(self.simulator.positions)
            for symbol, pos in positions.items():
                price = current_prices.get(symbol, pos.entry_price)
                exit_side = OrderSide.SELL if pos.side == Direction.LONG else OrderSide.BUY
                order = Order(
                    symbol=symbol,
                    side=exit_side,
                    order_type=OrderType.MARKET,
                    quantity=pos.quantity,
                    price=price,
                    stop_price=0.0,
                    strategy_id="kill_switch",
                )
                self.simulator.submit_order(order)
                bar = {"timestamp": now_utc(), "open": price, "high": price,
                       "low": price, "close": price, "volume": 1e9}
                fills = self.simulator.process_bar(symbol, bar)
                for fill in fills:
                    await self.bus.publish(EventType.FILL, fill)

        elif self.adapter:
            open_orders = await self.adapter.fetch_open_orders()
            for o in open_orders:
                await self.adapter.cancel_order(o["id"], o["symbol"])
            balance = await self.adapter.fetch_balance()
            for asset, info in balance.get("total", {}).items():
                if asset == "USDT" or float(info or 0) < 1e-8:
                    continue
                symbol = f"{asset}/USDT"
                ticker = await self.adapter.fetch_ticker(symbol)
                price = ticker.get("last", 0)
                qty = float(info)
                if qty > 0 and price > 0:
                    order = Order(
                        symbol=symbol, side=OrderSide.SELL, order_type=OrderType.MARKET,
                        quantity=qty, price=price, stop_price=0.0, strategy_id="kill_switch",
                    )
                    await self.adapter.create_market_order(order)

        await self.bus.publish(EventType.KILL, {"reason": reason})
        log.critical("[ExecutionEngine] All positions closed — trading halted")

    async def get_portfolio_state(self, symbols: list[str] | None = None) -> PortfolioState:
        if self.simulator:
            return self.simulator.get_portfolio_state()

        if self.adapter:
            balance = await self.adapter.fetch_balance()
            usdt = float(balance.get("free", {}).get("USDT", 0))
            positions: dict[str, Position] = {}
            total_value = usdt
            for asset, qty_info in balance.get("total", {}).items():
                qty = float(qty_info or 0)
                if asset == "USDT" or qty < 1e-8:
                    continue
                symbol = f"{asset}/USDT"
                ticker = await self.adapter.fetch_ticker(symbol)
                price = ticker.get("last", 0)
                mtm = qty * price
                total_value += mtm
                positions[symbol] = Position(
                    symbol=symbol, side=Direction.LONG, quantity=qty,
                    entry_price=price, stop_price=0.0, target_price=0.0,
                    strategy_id="live",
                )
            return PortfolioState(equity=total_value, cash=usdt, positions=positions)

        return PortfolioState(equity=0.0, cash=0.0)

    def resume(self) -> None:
        self._kill_active = False
        log.info("[ExecutionEngine] Kill switch lifted — trading resumed")
