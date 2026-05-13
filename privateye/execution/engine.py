"""
ExecutionEngine — routes orders to either a SimulatedExchange or a live ExchangeAdapter.

Responsibilities:
  - submit_order: validate, optionally TWAP-slice large orders, route to correct backend
  - flatten_all: emergency close all positions (kill switch)
  - get_portfolio_state: reconcile live balance + open positions → PortfolioState
"""
from __future__ import annotations

import asyncio
import dataclasses
from datetime import timezone
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from privateye.core.event_bus import AsyncEventBus
from privateye.core.types import (
    Direction, EventType, Fill, Order, OrderSide, OrderType,
    PortfolioState, Position,
)
from privateye.execution.twap import TWAPExecutor
from privateye.utils.logging import get_logger
from privateye.utils.time import now_utc

if TYPE_CHECKING:
    from privateye.execution.smart_order_router import SmartOrderRouter
    from privateye.execution.vwap import VWAPExecutor

log = get_logger()


class ExecutionEngine:
    def __init__(
        self,
        bus: AsyncEventBus,
        adapter: Any = None,           # ExchangeAdapter for live, None for paper
        simulator: Any = None,         # SimulatedExchange for paper/backtest
        exec_config: dict | None = None,
        smart_router: "SmartOrderRouter | None" = None,  # Phase 3: venue-aware router
    ) -> None:
        self.bus = bus
        self.adapter = adapter
        self.simulator = simulator
        self._live_mode = adapter is not None and simulator is None
        self._fills: list[Fill] = []
        self._kill_active = False
        self._exec_cfg: dict = exec_config or {}

        # Phase 3: VWAP executor (set by fit_volume_profile), SmartOrderRouter, conservative mode
        self._vwap_executor: VWAPExecutor | None = None
        self._smart_router: SmartOrderRouter | None = smart_router
        self._conservative_mode: bool = False
        self._conservative_size_multiplier: float = float(
            self._exec_cfg.get("shadow_mode_2", {}).get("conservative_size_multiplier", 0.5)
        )
        bus.subscribe(EventType.CONSERVATIVE_MODE, self._on_conservative_mode)

    async def submit_order(
        self,
        order: Order,
        bars: pd.DataFrame | None = None,
    ) -> Fill | None:
        if self._kill_active:
            log.warning(f"Kill switch active — order rejected: {order.symbol}")
            return None

        # Phase 3: Apply conservative size multiplier when Shadow Mode 2.0 is active
        if self._conservative_mode:
            reduced_qty = order.quantity * self._conservative_size_multiplier
            log.info(
                f"[ExecutionEngine] Conservative mode: reducing qty "
                f"{order.quantity:.6f} → {reduced_qty:.6f} for {order.symbol}"
            )
            order = dataclasses.replace(order, quantity=reduced_qty)

        log.info(f"[ExecutionEngine] Submit: {order.side.value} {order.quantity:.6f} {order.symbol}")

        # Auto-slice large orders using VWAP (when fitted) or TWAP fallback
        use_twap   = self._exec_cfg.get("use_twap", False)
        twap_thr   = float(self._exec_cfg.get("twap_threshold_usd", 1000))
        twap_n     = int(self._exec_cfg.get("twap_slices", 4))
        notional   = order.quantity * order.price

        if use_twap and notional > twap_thr:
            if self._vwap_executor is not None:
                current_hour = datetime.now(timezone.utc).hour
                slices = self._vwap_executor.split(order, current_hour)
                log.debug(f"[ExecutionEngine] VWAP-sliced into {len(slices)} sub-orders")
            else:
                slices = TWAPExecutor(twap_n).split(order)
                log.debug(f"[ExecutionEngine] TWAP-sliced into {len(slices)} sub-orders")

            last_fill = None
            for s in slices:
                last_fill = await self._submit_single(s, bars=bars)
            return last_fill

        return await self._submit_single(order, bars=bars)

    async def _submit_single(
        self,
        order: Order,
        bars: pd.DataFrame | None = None,
    ) -> Fill | None:
        # Phase 3: Apply SmartOrderRouter annotation (order type + venue selection)
        # Only annotate when bars are available; without bars the slippage
        # prediction is a fallback guess and the order type should stay as-is.
        if self._smart_router is not None and bars is not None:
            urgency = getattr(order, "_urgency", 0.5)
            decision = self._smart_router.route(order, urgency=urgency, bars=bars)
            order = self._smart_router.annotate_order(order, decision)
            log.debug(
                f"[ExecutionEngine] SOR decision: venue={decision.venue} "
                f"type={decision.order_type.value} "
                f"slip={decision.predicted_slippage_pct:.4%}"
            )

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

    # ── Phase 3: VWAP + conservative mode ────────────────────────────────────

    def fit_volume_profile(self, bars: pd.DataFrame) -> None:
        """Fit a VWAPExecutor on historical bars so submit_order can use VWAP slicing.

        Once called, VWAP splitting is used in place of TWAP whenever
        `use_twap=True` and the order notional exceeds `twap_threshold_usd`.
        """
        from privateye.execution.vwap import VWAPExecutor
        n_slices = int(self._exec_cfg.get("twap_slices", 4))
        self._vwap_executor = VWAPExecutor(n_slices)
        self._vwap_executor.fit_volume_profile(bars)
        log.info("[ExecutionEngine] VWAP volume profile fitted")

    async def _on_conservative_mode(self, payload: Any) -> None:
        """Handler for CONSERVATIVE_MODE events from ShadowTracker."""
        self._conservative_mode = True
        score = payload.get("score", 0.0) if isinstance(payload, dict) else 0.0
        reason = payload.get("reason", "unknown") if isinstance(payload, dict) else str(payload)
        log.warning(
            f"[ExecutionEngine] Conservative mode ACTIVATED by event "
            f"(reason={reason}, score={score:.3f}) — "
            f"position sizes will be multiplied by {self._conservative_size_multiplier}"
        )
