"""
Phase 13 test suite — SHORT support, MTM, fee accounting, bar-date robustness.

Twelve tests covering audit fixes:
  C-1: SHORT positions are opened/closed correctly
  C-2: Mark-to-market handles SHORT (equity rises when price falls)
  H-1: Total fees include both entry and exit
  H-4: Bar-date extraction works on every common timestamp type
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from privateye.backtesting.simulator import SimulatedExchange, _bar_date_iso
from privateye.core.types import (
    Direction, Order, OrderSide, OrderStatus, OrderType, Position,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar(close: float = 40_000.0, ts=None, volume: float = 1e9,
         high: float | None = None, low: float | None = None) -> dict:
    return {
        "timestamp": ts or datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
        "open": close, "close": close,
        "high": high if high is not None else close * 1.01,
        "low":  low  if low  is not None else close * 0.99,
        "volume": volume,
    }


def _market_order(side: OrderSide, qty: float, price: float,
                  symbol: str = "BTC/USDT") -> Order:
    return Order(
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=qty,
        price=price,
        stop_price=0.0,
        strategy_id="test",
    )


# ── 1. SHORT path ─────────────────────────────────────────────────────────────

class TestShortOpenAndClose:

    def test_short_open_creates_short_position(self):
        sim = SimulatedExchange(initial_capital=10_000.0)
        sim.submit_order(_market_order(OrderSide.SELL, 0.1, 40_000.0))
        sim.process_bar("BTC/USDT", _bar(close=40_000.0))
        pos = sim.positions["BTC/USDT"]
        assert pos.side == Direction.SHORT
        assert pos.quantity == pytest.approx(0.1)
        assert pos.entry_price > 0
        assert pos.entry_fees > 0   # entry fee tracked

    def test_short_open_credits_proceeds_to_cash(self):
        sim = SimulatedExchange(initial_capital=10_000.0)
        sim.submit_order(_market_order(OrderSide.SELL, 0.1, 40_000.0))
        sim.process_bar("BTC/USDT", _bar(close=40_000.0))
        # Proceeds added: ~0.1 * fill_price - fee. Cash should rise above 10k.
        assert sim.cash > 10_000.0

    def test_short_close_emits_trade_record(self):
        sim = SimulatedExchange(initial_capital=10_000.0)
        # Open short at 40k
        sim.submit_order(_market_order(OrderSide.SELL, 0.1, 40_000.0))
        sim.process_bar("BTC/USDT", _bar(close=40_000.0))
        # Close short by buying at 38k (price fell, profit)
        sim.submit_order(_market_order(OrderSide.BUY, 0.1, 38_000.0))
        sim.process_bar("BTC/USDT", _bar(close=38_000.0))
        assert "BTC/USDT" not in sim.positions
        assert len(sim.trade_records) == 1
        trade = sim.trade_records[0]
        assert trade.side == Direction.SHORT
        assert trade.pnl > 0   # profitable: entry > exit for shorts

    def test_short_close_after_price_rise_is_loss(self):
        sim = SimulatedExchange(initial_capital=10_000.0)
        sim.submit_order(_market_order(OrderSide.SELL, 0.1, 40_000.0))
        sim.process_bar("BTC/USDT", _bar(close=40_000.0))
        sim.submit_order(_market_order(OrderSide.BUY, 0.1, 42_000.0))
        sim.process_bar("BTC/USDT", _bar(close=42_000.0))
        assert sim.trade_records[0].pnl < 0


# ── 2. SHORT mark-to-market (C-2) ─────────────────────────────────────────────

class TestShortMTM:

    def test_short_equity_rises_when_price_falls(self):
        sim = SimulatedExchange(initial_capital=10_000.0)
        sim.submit_order(_market_order(OrderSide.SELL, 0.1, 40_000.0))
        sim.process_bar("BTC/USDT", _bar(close=40_000.0))
        equity_at_entry = sim.equity
        # Price falls; same position
        sim.process_bar("BTC/USDT", _bar(close=38_000.0))
        assert sim.equity > equity_at_entry, "Short equity must rise when price falls"

    def test_short_equity_falls_when_price_rises(self):
        sim = SimulatedExchange(initial_capital=10_000.0)
        sim.submit_order(_market_order(OrderSide.SELL, 0.1, 40_000.0))
        sim.process_bar("BTC/USDT", _bar(close=40_000.0))
        equity_at_entry = sim.equity
        sim.process_bar("BTC/USDT", _bar(close=42_000.0))
        assert sim.equity < equity_at_entry


# ── 3. Fee accounting (H-1) ───────────────────────────────────────────────────

class TestFeeAccounting:

    def test_long_round_trip_fees_include_both_legs(self):
        sim = SimulatedExchange(initial_capital=100_000.0,
                                fee_taker=0.001, slippage_pct=0.0)
        # Buy 1 BTC at 40k → entry fee ~40
        sim.submit_order(_market_order(OrderSide.BUY, 1.0, 40_000.0))
        sim.process_bar("BTC/USDT", _bar(close=40_000.0))
        # Sell 1 BTC at 41k → exit fee ~41
        sim.submit_order(_market_order(OrderSide.SELL, 1.0, 41_000.0))
        sim.process_bar("BTC/USDT", _bar(close=41_000.0))
        trade = sim.trade_records[0]
        # Both fees should be present (~81 total). Was previously ~41 (exit only).
        assert trade.fees > 70.0
        assert trade.fees < 100.0

    def test_short_round_trip_fees_include_both_legs(self):
        sim = SimulatedExchange(initial_capital=100_000.0,
                                fee_taker=0.001, slippage_pct=0.0)
        sim.submit_order(_market_order(OrderSide.SELL, 1.0, 40_000.0))
        sim.process_bar("BTC/USDT", _bar(close=40_000.0))
        sim.submit_order(_market_order(OrderSide.BUY, 1.0, 39_000.0))
        sim.process_bar("BTC/USDT", _bar(close=39_000.0))
        trade = sim.trade_records[0]
        # Both fees included
        assert trade.fees > 70.0


# ── 4. Bar-date robustness (H-4) ──────────────────────────────────────────────

class TestBarDateExtraction:

    def test_bar_date_from_datetime(self):
        ts = datetime(2024, 6, 15, 14, 30, tzinfo=timezone.utc)
        assert _bar_date_iso(ts) == "2024-06-15"

    def test_bar_date_from_pandas_timestamp(self):
        ts = pd.Timestamp("2024-06-15 14:30", tz="UTC")
        assert _bar_date_iso(ts) == "2024-06-15"

    def test_bar_date_from_int_milliseconds(self):
        # 2024-06-15 14:00:00 UTC in milliseconds
        ts_ms = int(datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc).timestamp() * 1000)
        assert _bar_date_iso(ts_ms) == "2024-06-15"

    def test_bar_date_from_iso_string(self):
        assert _bar_date_iso("2024-06-15T14:30:00+00:00") == "2024-06-15"

    def test_daily_dd_resets_with_int_ms_timestamps(self):
        """The original bug: int ms timestamps all share first 10 chars,
        so daily-DD reset never fired. Verify it does now."""
        sim = SimulatedExchange(initial_capital=10_000.0)
        ts_day1 = int(datetime(2024, 6, 15, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        ts_day2 = int(datetime(2024, 6, 16, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        sim.process_bar("BTC/USDT", _bar(close=40_000.0, ts=ts_day1))
        # Synthetic loss to set daily_pnl ≠ 0
        sim.daily_pnl = -100.0
        sim.process_bar("BTC/USDT", _bar(close=40_000.0, ts=ts_day2))
        assert sim.daily_pnl == 0.0   # reset must have fired
