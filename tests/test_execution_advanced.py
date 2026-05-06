"""
Phase 4 advanced execution tests.

Covers:
  - Limit order fill logic (BUY fills when bar.low <= price, SELL when bar.high >= price)
  - Limit order pending queue (unfilled orders persist, expire after patience bars)
  - Partial fills for limit orders requeue the remainder
  - Maker fee applied to limit fills, taker fee to market fills
  - FeeOptimizer: screens marginal trades, chooses LIMIT vs MARKET
  - VWAPExecutor: volume profile fitting and weighted splitting
  - BybitProvider: class exists, correct interface
  - ExchangeAdapter: factory pattern, create_limit_order present
  - BacktestReport: cost stats fields populated
"""
from __future__ import annotations

import dataclasses
import datetime

import numpy as np
import pandas as pd
import pytest

from privateye.core.types import (
    Direction, Fill, Order, OrderSide, OrderStatus, OrderType,
    TradingSignal,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_order(
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.LIMIT,
    price: float = 100.0,
    qty: float = 1.0,
    stop: float = 90.0,
) -> Order:
    return Order(
        symbol="BTC/USDT",
        side=side,
        order_type=order_type,
        quantity=qty,
        price=price,
        stop_price=stop,
        strategy_id="test",
    )


def _make_bar(
    close: float = 100.0,
    high: float = 105.0,
    low: float = 95.0,
    volume: float = 10_000.0,
    timestamp: str = "2024-01-01",
) -> dict:
    return {"timestamp": timestamp, "open": close, "high": high, "low": low,
            "close": close, "volume": volume}


def _make_exchange(capital: float = 10_000.0, patience: int = 3):
    from privateye.backtesting.simulator import SimulatedExchange
    return SimulatedExchange(
        initial_capital=capital,
        fee_maker=0.001,
        fee_taker=0.002,
        slippage_pct=0.0005,
        limit_patience_bars=patience,
    )


# ── Limit order fill logic ───────────────────────────────────────────────────

def test_buy_limit_fills_when_bar_crosses():
    ex = _make_exchange()
    order = _make_order(side=OrderSide.BUY, price=98.0)
    ex.submit_order(order)
    # bar.low = 95 < limit=98 → should fill
    fills = ex.process_bar("BTC/USDT", _make_bar(close=100.0, high=105.0, low=95.0))
    assert len(fills) == 1
    assert fills[0].price == 98.0


def test_buy_limit_no_fill_when_bar_above():
    ex = _make_exchange()
    order = _make_order(side=OrderSide.BUY, price=90.0)
    ex.submit_order(order)
    # bar.low = 95 > limit=90 → should NOT fill
    fills = ex.process_bar("BTC/USDT", _make_bar(close=100.0, high=105.0, low=95.0))
    assert len(fills) == 0


def test_sell_limit_fills_when_bar_crosses():
    ex = _make_exchange()
    # First buy to open a position
    buy = _make_order(side=OrderSide.BUY, order_type=OrderType.MARKET, price=100.0)
    ex.submit_order(buy)
    ex.process_bar("BTC/USDT", _make_bar())

    sell = _make_order(side=OrderSide.SELL, price=108.0, stop=90.0)
    ex.submit_order(sell)
    # bar.high = 110 > limit=108 → should fill
    fills = ex.process_bar("BTC/USDT", _make_bar(close=109.0, high=110.0, low=107.0))
    assert len(fills) == 1
    assert fills[0].price == 108.0


def test_sell_limit_no_fill_when_bar_below():
    ex = _make_exchange()
    buy = _make_order(side=OrderSide.BUY, order_type=OrderType.MARKET, price=100.0)
    ex.submit_order(buy)
    ex.process_bar("BTC/USDT", _make_bar())

    sell = _make_order(side=OrderSide.SELL, price=115.0, stop=90.0)
    ex.submit_order(sell)
    # bar.high = 110 < limit=115 → no fill
    fills = ex.process_bar("BTC/USDT", _make_bar(close=109.0, high=110.0, low=107.0))
    assert len(fills) == 0


def test_limit_fill_uses_maker_fee():
    ex = _make_exchange()
    order = _make_order(side=OrderSide.BUY, price=98.0)
    ex.submit_order(order)
    fills = ex.process_bar("BTC/USDT", _make_bar(close=100.0, high=105.0, low=95.0))
    assert len(fills) == 1
    expected_fee = 1.0 * 98.0 * 0.001  # qty * price * fee_maker
    assert abs(fills[0].fee - expected_fee) < 1e-9


def test_market_order_uses_taker_fee():
    ex = _make_exchange()
    order = _make_order(order_type=OrderType.MARKET, price=100.0)
    ex.submit_order(order)
    fills = ex.process_bar("BTC/USDT", _make_bar())
    assert len(fills) == 1
    # fill_price = 100 * (1 + 0.0005) = 100.05
    fill_price = 100.0 * (1 + 0.0005)
    expected_fee = 1.0 * fill_price * 0.002  # taker fee
    assert abs(fills[0].fee - expected_fee) < 1e-6


# ── Pending queue behaviour ──────────────────────────────────────────────────

def test_unfilled_limit_stays_in_queue():
    ex = _make_exchange(patience=3)
    order = _make_order(price=90.0)  # below bar — won't fill
    ex.submit_order(order)
    ex.process_bar("BTC/USDT", _make_bar(low=95.0))
    assert len(ex._pending_orders) == 1


def test_unfilled_limit_expires_after_patience():
    ex = _make_exchange(patience=2)
    order = _make_order(price=90.0)
    ex.submit_order(order)
    ex.process_bar("BTC/USDT", _make_bar(low=95.0))  # age=1, kept
    ex.process_bar("BTC/USDT", _make_bar(low=95.0))  # age=2 >= patience=2, cancelled
    assert len(ex._pending_orders) == 0


def test_limit_fills_after_one_bar_wait():
    ex = _make_exchange(patience=3)
    order = _make_order(price=90.0)
    ex.submit_order(order)
    ex.process_bar("BTC/USDT", _make_bar(low=95.0))  # no fill, age=1
    fills = ex.process_bar("BTC/USDT", _make_bar(low=88.0))  # low crosses — fills
    assert len(fills) == 1
    assert fills[0].price == 90.0


def test_market_order_cancelled_when_volume_zero():
    ex = _make_exchange()
    order = _make_order(order_type=OrderType.MARKET, qty=1.0)
    ex.submit_order(order)
    # volume=0 → max_qty=0 → reject
    fills = ex.process_bar("BTC/USDT", _make_bar(volume=0.0))
    assert len(fills) == 0
    assert len(ex._pending_orders) == 0  # market order cancelled (not re-queued)


# ── Cost stats ───────────────────────────────────────────────────────────────

def test_cost_stats_limit_fill_tracking():
    ex = _make_exchange()
    o1 = _make_order(price=98.0)
    o2 = _make_order(price=90.0)  # won't fill (bar.low=95)
    ex.submit_order(o1)
    ex.submit_order(o2)
    ex.process_bar("BTC/USDT", _make_bar(low=95.0))  # o1 fills, o2 stays

    stats = ex.get_cost_stats()
    assert stats["limit_orders_placed"] == 2
    assert stats["limit_orders_filled"] == 1


def test_cost_stats_slippage_tracked_for_market():
    ex = _make_exchange()
    order = _make_order(order_type=OrderType.MARKET, price=100.0)
    ex.submit_order(order)
    ex.process_bar("BTC/USDT", _make_bar())
    stats = ex.get_cost_stats()
    assert stats["total_slippage_cost"] > 0


def test_cost_stats_no_slippage_for_limit():
    ex = _make_exchange()
    order = _make_order(price=98.0)
    ex.submit_order(order)
    ex.process_bar("BTC/USDT", _make_bar(low=95.0))
    stats = ex.get_cost_stats()
    assert stats["total_slippage_cost"] == 0.0


# ── FeeOptimizer ─────────────────────────────────────────────────────────────

def _make_signal(entry: float = 100.0, target: float = 104.0, stop: float = 98.0,
                 confidence: float = 0.65) -> TradingSignal:
    return TradingSignal(
        symbol="BTC/USDT",
        direction=Direction.LONG,
        confidence=confidence,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        strategy_id="test",
        timeframe="1h",
        timestamp=datetime.datetime.now(datetime.timezone.utc),
    )


def test_fee_optimizer_screens_low_return():
    from privateye.execution.fee_optimizer import FeeOptimizer
    fo = FeeOptimizer(min_net_return=0.03)  # 3% min
    sig = _make_signal(entry=100.0, target=101.0)  # 1% return < 3% min
    result = fo.choose_order_type(sig)
    assert result is None


def test_fee_optimizer_returns_limit_normal():
    from privateye.execution.fee_optimizer import FeeOptimizer
    fo = FeeOptimizer(min_net_return=0.003)
    sig = _make_signal(entry=100.0, target=104.0, confidence=0.65)  # 4% return, low conf
    result = fo.choose_order_type(sig, spread_pct=0.001)
    assert result == OrderType.LIMIT


def test_fee_optimizer_returns_market_high_confidence():
    from privateye.execution.fee_optimizer import FeeOptimizer
    fo = FeeOptimizer(min_net_return=0.003)
    sig = _make_signal(entry=100.0, target=104.0, confidence=0.90)  # high conf → MARKET
    result = fo.choose_order_type(sig, spread_pct=0.001)
    assert result == OrderType.MARKET


def test_fee_optimizer_returns_market_wide_spread():
    from privateye.execution.fee_optimizer import FeeOptimizer
    fo = FeeOptimizer(min_net_return=0.003)
    sig = _make_signal(entry=100.0, target=104.0, confidence=0.65)
    # spread > wide_spread_threshold (0.5%) → MARKET
    result = fo.choose_order_type(sig, spread_pct=0.006)
    assert result == OrderType.MARKET


def test_fee_optimizer_annotate_order():
    from privateye.execution.fee_optimizer import FeeOptimizer
    fo = FeeOptimizer()
    order = _make_order(order_type=OrderType.MARKET)
    annotated = fo.annotate_order(order, OrderType.LIMIT)
    assert annotated.order_type == OrderType.LIMIT
    assert annotated.post_only is True
    assert order.order_type == OrderType.MARKET  # original unchanged


# ── VWAPExecutor ─────────────────────────────────────────────────────────────

def _make_bars_with_volume(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h")
    volumes = rng.uniform(100, 1000, n)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": 50000.0,
        "high": 50100.0,
        "low": 49900.0,
        "close": 50000.0,
        "volume": volumes,
    })


def test_vwap_split_returns_n_slices():
    from privateye.execution.vwap import VWAPExecutor
    bars = _make_bars_with_volume()
    ex = VWAPExecutor(n_slices=4)
    ex.fit_volume_profile(bars)
    order = _make_order(qty=2.0, order_type=OrderType.MARKET)
    slices = ex.split(order, current_hour=14)
    assert len(slices) == 4


def test_vwap_split_total_qty_preserved():
    from privateye.execution.vwap import VWAPExecutor
    bars = _make_bars_with_volume()
    ex = VWAPExecutor(n_slices=4)
    ex.fit_volume_profile(bars)
    order = _make_order(qty=1.0, order_type=OrderType.MARKET)
    slices = ex.split(order, current_hour=10)
    total = sum(s.quantity for s in slices)
    assert abs(total - 1.0) < 1e-6


def test_vwap_quantities_not_equal_with_profile():
    from privateye.execution.vwap import VWAPExecutor
    # Create bars with very uneven volume distribution
    timestamps = pd.date_range("2024-01-01", periods=240, freq="1h")
    # Hour 0 has 10x volume of hour 1
    volumes = [1000.0 if t.hour == 0 else 100.0 for t in timestamps]
    bars = pd.DataFrame({
        "timestamp": timestamps,
        "open": 50000.0, "high": 50100.0, "low": 49900.0,
        "close": 50000.0, "volume": volumes,
    })
    ex = VWAPExecutor(n_slices=2)
    ex.fit_volume_profile(bars)
    order = _make_order(qty=1.0)
    slices = ex.split(order, current_hour=0)
    # slice at hour 0 should have more qty than slice at hour 1
    qtys = [s.quantity for s in slices]
    assert qtys[0] != qtys[1]


def test_vwap_falls_back_to_uniform_without_profile():
    from privateye.execution.vwap import VWAPExecutor
    ex = VWAPExecutor(n_slices=3)
    order = _make_order(qty=3.0)
    slices = ex.split(order, current_hour=None)  # no hour provided → uniform
    assert len(slices) == 3
    for s in slices:
        assert abs(s.quantity - 1.0) < 1e-6


def test_vwap_invalid_n_slices():
    from privateye.execution.vwap import VWAPExecutor
    with pytest.raises(ValueError):
        VWAPExecutor(n_slices=0)


# ── BybitProvider ────────────────────────────────────────────────────────────

def test_bybit_provider_importable():
    from privateye.data.providers.bybit import BybitProvider
    bp = BybitProvider(api_key="", api_secret="", sandbox=True)
    assert bp is not None
    assert bp.bar_limit == 500
    assert bp._circuit_open is False


def test_bybit_provider_has_expected_methods():
    from privateye.data.providers.bybit import BybitProvider
    assert hasattr(BybitProvider, "start")
    assert hasattr(BybitProvider, "stop")
    assert hasattr(BybitProvider, "get_bars")
    assert hasattr(BybitProvider, "reset_circuit_breaker")
    assert hasattr(BybitProvider, "fetch_ticker")


def test_bybit_provider_get_bars_returns_empty_df():
    from privateye.data.providers.bybit import BybitProvider
    bp = BybitProvider()
    df = bp.get_bars("BTC/USDT", "1h")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_bybit_provider_reset_circuit_breaker():
    from privateye.data.providers.bybit import BybitProvider
    bp = BybitProvider()
    bp._circuit_open = True
    bp._consecutive_failures = 15
    bp.reset_circuit_breaker()
    assert bp._circuit_open is False
    assert bp._consecutive_failures == 0


# ── ExchangeAdapter factory ──────────────────────────────────────────────────

def test_adapter_has_create_limit_order():
    from privateye.execution.adapter import ExchangeAdapter
    assert hasattr(ExchangeAdapter, "create_limit_order")


def test_adapter_ccxt_map_contains_exchanges():
    from privateye.execution.adapter import CCXT_MAP
    assert "binance" in CCXT_MAP
    assert "bybit" in CCXT_MAP
    assert "okx" in CCXT_MAP


# ── BacktestReport cost fields ───────────────────────────────────────────────

def test_backtest_report_has_cost_fields():
    from privateye.backtesting.metrics import BacktestReport
    r = BacktestReport()
    assert hasattr(r, "total_fees_paid")
    assert hasattr(r, "fee_drag_pct")
    assert hasattr(r, "slippage_cost_pct")
    assert hasattr(r, "limit_fill_rate")
    assert hasattr(r, "limit_orders_placed")
    assert hasattr(r, "limit_orders_filled")


def test_compute_metrics_populates_cost_stats():
    from privateye.backtesting.metrics import compute_metrics
    from privateye.core.types import TradeRecord
    trades = [
        TradeRecord(
            symbol="BTC/USDT", side=Direction.LONG,
            entry_price=100.0, exit_price=105.0, quantity=1.0,
            entry_time=datetime.datetime.now(datetime.timezone.utc),
            exit_time=datetime.datetime.now(datetime.timezone.utc),
            pnl=4.9, pnl_pct=0.049, fees=0.1,
            strategy_id="test", exit_reason="target", bars_held=5,
        )
    ]
    equity_curve = [10000.0, 10005.0, 10010.0, 10005.0, 10015.0]
    cost_stats = {
        "total_fees_paid": 0.2,
        "limit_orders_placed": 3,
        "limit_orders_filled": 2,
        "total_slippage_cost": 5.0,
    }
    report = compute_metrics(trades, equity_curve, 10000.0, cost_stats=cost_stats)
    assert report.total_fees_paid == 0.2
    assert report.limit_orders_placed == 3
    assert report.limit_orders_filled == 2
    assert abs(report.limit_fill_rate - 2 / 3) < 1e-9
    assert report.slippage_cost_pct == pytest.approx(0.05, abs=1e-6)


def test_types_post_only_default():
    order = _make_order()
    assert order.post_only is False


def test_types_stop_limit_enum():
    assert OrderType.STOP_LIMIT.value == "stop_limit"
