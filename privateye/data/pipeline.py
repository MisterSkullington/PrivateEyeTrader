"""
DataPipeline — ingests bars from any provider and publishes DataSnapshot events to the event bus.

For backtest: replays CSV/SQLite rows bar-by-bar.
For paper/live: polls Binance provider on an interval.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from privateye.core.event_bus import AsyncEventBus
from privateye.core.types import DataSnapshot, EventType
from privateye.utils.logging import get_logger
from privateye.utils.time import now_utc

log = get_logger()


class DataPipeline:
    def __init__(
        self,
        bus: AsyncEventBus,
        bar_window: int = 500,
    ) -> None:
        self.bus = bus
        self.bar_window = bar_window
        # Rolling bar store: (symbol, timeframe) → DataFrame
        self._bars: dict[tuple[str, str], pd.DataFrame] = {}

    async def push_bars(self, symbol: str, timeframe: str, new_bars: pd.DataFrame) -> None:
        """
        Update the rolling window for (symbol, timeframe) and publish a DataSnapshot.
        new_bars is appended; oldest bars are dropped to maintain bar_window length.
        """
        key = (symbol, timeframe)
        existing = self._bars.get(key, pd.DataFrame())

        if existing.empty:
            combined = new_bars
        else:
            combined = pd.concat([existing, new_bars], ignore_index=True)
            combined = combined.drop_duplicates("timestamp").sort_values("timestamp")

        combined = combined.tail(self.bar_window).reset_index(drop=True)
        self._bars[key] = combined

        if len(combined) < 2:
            return

        snapshot = DataSnapshot(
            symbol=symbol,
            timeframe=timeframe,
            bars=combined.copy(),
            timestamp=now_utc(),
        )
        await self.bus.publish(EventType.MARKET_DATA, snapshot)

    async def push_bar(self, symbol: str, timeframe: str, bar: dict[str, Any]) -> None:
        """Push a single bar dict."""
        df = pd.DataFrame([bar])
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        await self.push_bars(symbol, timeframe, df)

    def get_snapshot(self, symbol: str, timeframe: str) -> DataSnapshot | None:
        key = (symbol, timeframe)
        bars = self._bars.get(key)
        if bars is None or bars.empty:
            return None
        return DataSnapshot(symbol=symbol, timeframe=timeframe, bars=bars.copy(), timestamp=now_utc())

    def reset(self) -> None:
        self._bars.clear()
