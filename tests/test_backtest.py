"""Unit tests for the backtest engine and SimulatedExchange."""
import numpy as np
import pandas as pd
import pytest

from privateye.backtesting.metrics import BacktestReport, compute_metrics
from privateye.backtesting.simulator import SimulatedExchange
from privateye.core.types import Direction, Order, OrderSide, OrderType, TradeRecord
from privateye.utils.time import now_utc


def _make_bars(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 30000 + np.cumsum(rng.normal(0, 200, n))
    highs = closes + rng.uniform(100, 400, n)
    lows = closes - rng.uniform(100, 400, n)
    opens = closes - rng.normal(0, 100, n)
    vols = rng.uniform(500, 5000, n)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": opens, "high": highs,
        "low": lows, "close": closes, "volume": vols,
    })


def _bar(price: float = 30000.0, volume: float = 10000.0) -> dict:
    return {
        "timestamp": now_utc(), "open": price, "high": price + 100,
        "low": price - 100, "close": price, "volume": volume,
    }


class TestSimulatedExchange:
    def setup_method(self):
        self.ex = SimulatedExchange(
            initial_capital=10000.0,
            fee_maker=0.001, fee_taker=0.001,
            slippage_pct=0.0005,
            max_fill_pct_of_volume=0.30,
        )

    def test_initial_state(self):
        state = self.ex.get_portfolio_state()
        assert state.equity == 10000.0
        assert state.cash == 10000.0
        assert len(state.positions) == 0

    def test_buy_reduces_cash(self):
        order = Order(
            symbol="BTC/USDT", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=0.1, price=30000.0, stop_price=29000.0, strategy_id="test",
        )
        self.ex.submit_order(order)
        self.ex.process_bar("BTC/USDT", _bar(30000.0, 10000.0))
        assert self.ex.cash < 10000.0

    def test_buy_then_sell_closes_position(self):
        order_buy = Order(
            symbol="BTC/USDT", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=0.1, price=30000.0, stop_price=29000.0, strategy_id="test",
        )
        self.ex.submit_order(order_buy)
        self.ex.process_bar("BTC/USDT", _bar(30000.0))
        assert "BTC/USDT" in self.ex.positions

        # Sell exactly the filled position quantity (not the original order qty)
        pos_qty = self.ex.positions["BTC/USDT"].quantity
        order_sell = Order(
            symbol="BTC/USDT", side=OrderSide.SELL, order_type=OrderType.MARKET,
            quantity=pos_qty, price=31000.0, stop_price=0.0, strategy_id="test",
        )
        self.ex.submit_order(order_sell)
        fills = self.ex.process_bar("BTC/USDT", _bar(31000.0))
        assert len(fills) > 0
        assert "BTC/USDT" not in self.ex.positions
        assert len(self.ex.trade_records) == 1

    def test_profitable_trade_increases_equity(self):
        order = Order(
            symbol="BTC/USDT", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=0.1, price=30000.0, stop_price=29000.0, strategy_id="test",
        )
        self.ex.submit_order(order)
        self.ex.process_bar("BTC/USDT", _bar(30000.0))
        order_sell = Order(
            symbol="BTC/USDT", side=OrderSide.SELL, order_type=OrderType.MARKET,
            quantity=0.1, price=35000.0, stop_price=0.0, strategy_id="test",
        )
        self.ex.submit_order(order_sell)
        self.ex.process_bar("BTC/USDT", _bar(35000.0))
        assert self.ex.equity > 10000.0

    def test_volume_cap_partial_fill(self):
        """Order larger than 30% of bar volume should be partially filled or rejected."""
        # Bar volume = 10, 30% = 3; at price 30000 that's 0.0001 BTC max
        small_vol_bar = {"timestamp": now_utc(), "open": 30000, "high": 30100,
                         "low": 29900, "close": 30000, "volume": 10.0}
        order = Order(
            symbol="BTC/USDT", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=10.0, price=30000.0, stop_price=29000.0, strategy_id="test",
        )
        self.ex.submit_order(order)
        fills = self.ex.process_bar("BTC/USDT", small_vol_bar)
        # Should be cancelled (volume too small) or filled at reduced qty
        # Either way, we should not have bought 10 BTC (cash would go very negative)
        if fills:
            assert fills[0].quantity < 10.0
        assert self.ex.cash >= 0  # never go deeply negative

    def test_slippage_applied(self):
        order = Order(
            symbol="BTC/USDT", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=0.01, price=30000.0, stop_price=29000.0, strategy_id="test",
        )
        self.ex.submit_order(order)
        fills = self.ex.process_bar("BTC/USDT", _bar(30000.0))
        if fills:
            # Fill price should be slightly above close (buy-side slippage)
            assert fills[0].price >= 30000.0


class TestMetrics:
    def _make_trade(self, pnl: float, bars: int = 5) -> TradeRecord:
        return TradeRecord(
            symbol="BTC/USDT", side=Direction.LONG,
            entry_price=30000.0, exit_price=30000.0 + pnl / 0.1,
            quantity=0.1, entry_time=now_utc(), exit_time=now_utc(),
            pnl=pnl, pnl_pct=pnl / 3000.0, fees=3.0,
            strategy_id="test", exit_reason="target", bars_held=bars,
        )

    def test_win_rate(self):
        trades = [self._make_trade(100), self._make_trade(100), self._make_trade(-50)]
        curve = [10000, 10100, 10200, 10150]
        report = compute_metrics(trades, curve, 10000)
        assert abs(report.win_rate - 2 / 3) < 1e-6

    def test_profit_factor(self):
        trades = [self._make_trade(200), self._make_trade(-100)]
        curve = [10000, 10200, 10100]
        report = compute_metrics(trades, curve, 10000)
        assert abs(report.profit_factor - 2.0) < 1e-6

    def test_max_drawdown(self):
        # Equity rises then falls — expect drawdown detected
        curve = [10000, 11000, 12000, 10000, 9000]
        trades = [self._make_trade(-1000)]
        report = compute_metrics(trades, curve, 10000)
        assert report.max_drawdown_pct > 0

    def test_no_trades(self):
        report = compute_metrics([], [10000], 10000)
        assert report.total_trades == 0

    def test_str_output(self):
        trades = [self._make_trade(100), self._make_trade(-50)]
        report = compute_metrics(trades, [10000, 10100, 10050], 10000)
        output = str(report)
        assert "Win Rate" in output
        assert "Sharpe" in output
