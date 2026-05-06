"""
ExchangeAdapter — thin CCXT wrapper for live order execution.

Supports multiple exchanges via a factory pattern keyed on `exchange_id`.
Exposes:
  - create_market_order / create_limit_order
  - cancel_order
  - fetch_open_orders
  - fetch_balance
  - fetch_ticker

API keys require only trading permissions — no withdrawal rights needed.
"""
from __future__ import annotations

import asyncio
from typing import Any

import ccxt

from privateye.core.exceptions import ExchangeError
from privateye.core.types import Fill, Order, OrderSide, OrderStatus, OrderType
from privateye.utils.logging import get_logger
from privateye.utils.time import now_utc

log = get_logger()

CCXT_MAP: dict[str, type] = {
    "binance": ccxt.binance,
    "bybit":   ccxt.bybit,
    "okx":     ccxt.okx,
}


class ExchangeAdapter:
    def __init__(self, config: dict[str, Any]) -> None:
        exchange_id = config.get("exchange_id", "binance")
        cls = CCXT_MAP.get(exchange_id, ccxt.binance)
        self._exchange = cls({
            "apiKey": config.get("api_key", ""),
            "secret": config.get("api_secret", ""),
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        if config.get("sandbox", True):
            self._exchange.set_sandbox_mode(True)
            log.info(f"ExchangeAdapter: {exchange_id} sandbox mode enabled")
        else:
            log.warning(f"ExchangeAdapter: {exchange_id} LIVE MODE — real orders will be placed")

    async def create_market_order(self, order: Order) -> Fill | None:
        side = "buy" if order.side == OrderSide.BUY else "sell"
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._exchange.create_market_order(
                    order.symbol, side, order.quantity
                ),
            )
            fill_price = result.get("average") or result.get("price") or order.price
            fill_qty   = result.get("filled", order.quantity)
            fee_info   = result.get("fee") or {}
            fee        = fee_info.get("cost", fill_qty * fill_price * 0.001)
            order.status = OrderStatus.FILLED
            return Fill(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                quantity=fill_qty,
                price=fill_price,
                fee=fee,
                strategy_id=order.strategy_id,
                timestamp=now_utc(),
            )
        except Exception as e:
            log.error(f"ExchangeAdapter market order failed [{order.symbol} {side}]: {e}")
            order.status = OrderStatus.REJECTED
            raise ExchangeError(str(e)) from e

    async def create_limit_order(self, order: Order) -> Fill | None:
        side = "buy" if order.side == OrderSide.BUY else "sell"
        params: dict[str, Any] = {}
        if order.post_only:
            params["postOnly"] = True
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._exchange.create_order(
                    order.symbol, "limit", side, order.quantity, order.price, params
                ),
            )
            fill_price = result.get("average") or result.get("price") or order.price
            fill_qty   = result.get("filled", 0.0)
            fee_info   = result.get("fee") or {}
            fee        = fee_info.get("cost", fill_qty * fill_price * 0.001) if fill_qty > 0 else 0.0
            order.status = OrderStatus.OPEN  # limit orders may not fill immediately
            if fill_qty <= 0:
                return None  # order placed but not filled yet
            return Fill(
                order_id=order.id,
                symbol=order.symbol,
                side=order.side,
                quantity=fill_qty,
                price=fill_price,
                fee=fee,
                strategy_id=order.strategy_id,
                timestamp=now_utc(),
            )
        except Exception as e:
            log.error(f"ExchangeAdapter limit order failed [{order.symbol} {side}]: {e}")
            order.status = OrderStatus.REJECTED
            raise ExchangeError(str(e)) from e

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._exchange.cancel_order(order_id, symbol)
            )
            return True
        except Exception as e:
            log.warning(f"Cancel order failed [{order_id}]: {e}")
            return False

    async def fetch_balance(self) -> dict[str, Any]:
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._exchange.fetch_balance
            )
        except Exception as e:
            log.warning(f"fetch_balance error: {e}")
            return {}

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._exchange.fetch_ticker(symbol)
            )
        except Exception as e:
            log.warning(f"fetch_ticker error [{symbol}]: {e}")
            return {}

    async def fetch_open_orders(self, symbol: str | None = None) -> list[dict]:
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._exchange.fetch_open_orders(symbol)
            )
        except Exception as e:
            log.warning(f"fetch_open_orders error: {e}")
            return []
