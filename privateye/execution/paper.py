"""
PaperTrader — wraps SimulatedExchange for live paper trading mode.

Subscribes to MARKET_DATA events from the DataPipeline, processes each
bar through the SimulatedExchange, and publishes fills back to the bus.
The fill/execution logic is identical to backtest — ensuring paper ≈ live.
"""
from __future__ import annotations

from typing import Any

from privateye.backtesting.simulator import SimulatedExchange
from privateye.core.event_bus import AsyncEventBus
from privateye.core.types import DataSnapshot, EventType
from privateye.utils.logging import get_logger

log = get_logger()


class PaperTrader:
    def __init__(
        self,
        bus: AsyncEventBus,
        exchange: SimulatedExchange,
    ) -> None:
        self.bus = bus
        self.exchange = exchange
        bus.subscribe(EventType.MARKET_DATA, self._on_market_data)

    async def _on_market_data(self, payload: Any) -> None:
        if not isinstance(payload, DataSnapshot):
            return
        snapshot: DataSnapshot = payload
        bar = {
            "timestamp": snapshot.timestamp,
            "open": float(snapshot.bars["open"].iloc[-1]),
            "high": float(snapshot.bars["high"].iloc[-1]),
            "low": float(snapshot.bars["low"].iloc[-1]),
            "close": snapshot.close,
            "volume": snapshot.volume,
        }
        fills = self.exchange.process_bar(snapshot.symbol, bar)
        for fill in fills:
            await self.bus.dispatch(EventType.FILL, fill)
            log.info(
                f"[PaperTrader] Fill: {fill.side.value} {fill.quantity:.6f} {fill.symbol} "
                f"@ {fill.price:.2f} pnl={fill.realised_pnl:+.2f}"
            )

    def get_portfolio_state(self):
        return self.exchange.get_portfolio_state()
