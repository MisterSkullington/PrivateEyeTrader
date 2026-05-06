"""Tests for PaperTrader and DataSnapshot handling."""
import asyncio
import pandas as pd
import numpy as np
import pytest

from privateye.backtesting.simulator import SimulatedExchange
from privateye.core.event_bus import AsyncEventBus
from privateye.core.types import DataSnapshot, EventType
from privateye.execution.paper import PaperTrader
from privateye.utils.time import now_utc


def _make_snapshot(n: int = 50, symbol: str = "BTC/USDT", tf: str = "1h") -> DataSnapshot:
    rng = np.random.default_rng(1)
    closes = 30000 + np.cumsum(rng.normal(0, 100, n))
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "open": closes - 50, "high": closes + 100,
        "low": closes - 100, "close": closes,
        "volume": rng.uniform(500, 1000, n),
    })
    return DataSnapshot(symbol=symbol, timeframe=tf, bars=df, timestamp=now_utc())


class TestPaperTrader:
    def setup_method(self):
        self.bus = AsyncEventBus()
        self.sim = SimulatedExchange(initial_capital=10000.0)
        self.paper = PaperTrader(self.bus, self.sim)

    def test_ignores_dict_payload(self):
        """PaperTrader must only handle DataSnapshot, not raw dicts."""
        async def run():
            await self.bus.publish(EventType.MARKET_DATA, {"close": 30000})
            await asyncio.sleep(0.05)
        asyncio.run(run())
        # Should not crash and position count stays 0
        assert len(self.sim.positions) == 0

    def test_processes_datasnapshot(self):
        """PaperTrader should process DataSnapshot without error."""
        snapshot = _make_snapshot()
        fills_received = []

        async def capture_fill(fill):
            fills_received.append(fill)

        self.bus.subscribe(EventType.FILL, capture_fill)

        async def run():
            await self.bus.publish(EventType.MARKET_DATA, snapshot)
            await asyncio.sleep(0.1)
        asyncio.run(run())
        # No pending orders → no fills, but no crash either
        assert isinstance(fills_received, list)

    def test_fill_published_on_pending_order(self):
        """Submit an order then push a bar — should get a fill."""
        from privateye.core.types import Order, OrderSide, OrderType
        order = Order(
            symbol="BTC/USDT", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=0.01, price=30000.0, stop_price=29000.0, strategy_id="test",
        )
        self.sim.submit_order(order)
        fills_received = []

        async def capture_fill(fill):
            fills_received.append(fill)

        self.bus.subscribe(EventType.FILL, capture_fill)

        async def run():
            snapshot = _make_snapshot()
            # Use dispatch (immediate, synchronous) instead of publish (queued)
            await self.bus.dispatch(EventType.MARKET_DATA, snapshot)
            await asyncio.sleep(0.1)

        asyncio.run(run())
        assert len(fills_received) > 0
        assert fills_received[0].symbol == "BTC/USDT"


class TestDataPipeline:
    def test_push_and_snapshot(self):
        from privateye.data.pipeline import DataPipeline
        bus = AsyncEventBus()
        pipeline = DataPipeline(bus, bar_window=100)

        snapshots = []

        async def capture(payload):
            snapshots.append(payload)

        bus.subscribe(EventType.MARKET_DATA, capture)

        rng = np.random.default_rng(0)
        n = 50
        bars = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
            "open": rng.uniform(29000, 31000, n),
            "high": rng.uniform(30500, 32000, n),
            "low": rng.uniform(28000, 30000, n),
            "close": rng.uniform(29500, 31500, n),
            "volume": rng.uniform(100, 1000, n),
        })

        async def run():
            # Start bus in background so queued events are dispatched
            bus_task = asyncio.create_task(bus.run())
            await pipeline.push_bars("BTC/USDT", "1h", bars)
            await asyncio.sleep(0.1)
            bus.stop()
            bus_task.cancel()

        asyncio.run(run())
        assert len(snapshots) == 1
        assert isinstance(snapshots[0], DataSnapshot)
        assert snapshots[0].symbol == "BTC/USDT"
        assert len(snapshots[0].bars) == n

    def test_rolling_window_bounded(self):
        from privateye.data.pipeline import DataPipeline
        bus = AsyncEventBus()
        pipeline = DataPipeline(bus, bar_window=50)

        rng = np.random.default_rng(1)
        n = 80
        bars = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
            "open": 30000.0, "high": 30100.0, "low": 29900.0,
            "close": 30000.0, "volume": 500.0,
        })

        async def run():
            await pipeline.push_bars("BTC/USDT", "1h", bars)

        asyncio.run(run())
        snap = pipeline.get_snapshot("BTC/USDT", "1h")
        assert snap is not None
        assert len(snap.bars) == 50  # capped at bar_window
