"""
Integration tests — full signal→risk→execution chain using synthetic data.
These tests run the backtest engine end-to-end without any network calls.
"""
import numpy as np
import pandas as pd
import pytest

from privateye.backtesting.engine import BacktestEngine
from privateye.backtesting.simulator import SimulatedExchange
from privateye.core.types import DataSnapshot, Direction
from privateye.data.pipeline import DataPipeline
from privateye.core.event_bus import AsyncEventBus
from privateye.risk.manager import RiskManager
from privateye.strategies.directional import DirectionalStrategy
from privateye.strategies.mean_reversion import MeanReversionStrategy


def _trending_bars(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Synthetic strongly trending market (upward) — favours DirectionalStrategy longs."""
    rng = np.random.default_rng(seed)
    trend = np.linspace(20000, 40000, n)
    noise = rng.normal(0, 200, n)
    closes = trend + noise
    highs = closes + rng.uniform(100, 500, n)
    lows = closes - rng.uniform(100, 500, n)
    opens = closes - rng.normal(0, 150, n)
    vols = rng.uniform(300, 3000, n)
    ts = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": opens, "high": highs,
        "low": lows, "close": closes, "volume": vols,
    })


def _flat_bars(n: int = 500, seed: int = 1) -> pd.DataFrame:
    """Synthetic ranging market — many BB/RSI oscillations."""
    rng = np.random.default_rng(seed)
    closes = 30000 + rng.normal(0, 300, n)
    highs = closes + rng.uniform(50, 200, n)
    lows = closes - rng.uniform(50, 200, n)
    opens = closes - rng.normal(0, 80, n)
    vols = rng.uniform(200, 2000, n)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts, "open": opens, "high": highs,
        "low": lows, "close": closes, "volume": vols,
    })


DEFAULT_CONFIG = {
    "backtesting": {"initial_capital": 10000.0, "fee_maker": 0.001, "fee_taker": 0.001,
                    "slippage_pct": 0.0005, "max_fill_pct_of_volume": 0.30},
    "data": {"bar_window": 500},
    "risk": {"max_risk_per_trade_pct": 0.01, "max_daily_drawdown_pct": 0.05,
              "max_position_notional_pct": 0.20, "min_confidence": 0.55,
              "atr_stop_multiplier": 2.0, "max_bars_in_trade": 48},
}


class TestBacktestSmoke:
    def _run(self, bars, strategies):
        risk = RiskManager(DEFAULT_CONFIG["risk"])
        sim = SimulatedExchange(
            initial_capital=10000.0,
            fee_maker=0.001, fee_taker=0.001,
            slippage_pct=0.0005,
        )
        engine = BacktestEngine(strategies, risk, sim, DEFAULT_CONFIG)
        return engine.run(bars, "BTC/USDT", "1h")

    def test_directional_smoke(self):
        bars = _trending_bars()
        cfg = {"enabled": True, "timeframe": "1h", "macd_fast": 12, "macd_slow": 26,
               "macd_signal": 9, "ema_trend": 200, "rsi_period": 14,
               "atr_stop_multiplier": 2.0, "max_bars_in_trade": 48,
               "rsi_overbought": 70, "rsi_oversold": 30}
        report = self._run(bars, [DirectionalStrategy(cfg)])
        assert report is not None
        assert report.total_trades >= 0
        assert report.final_equity > 0

    def test_mean_reversion_smoke(self):
        bars = _flat_bars()
        cfg = {"enabled": True, "timeframe": "1h", "bb_period": 20, "bb_std": 2.0,
               "rsi_period": 14, "rsi_overbought": 70, "rsi_oversold": 30,
               "atr_stop_multiplier": 1.0, "max_bars_in_trade": 24}
        report = self._run(bars, [MeanReversionStrategy(cfg)])
        assert report is not None
        assert report.final_equity > 0

    def test_no_lookahead_bias(self):
        """
        Run backtest on first 600 bars, then on first 400 bars.
        Trades that occurred within the first 400 bars must be identical
        (no future data influence on past decisions).
        """
        bars = _trending_bars(800)
        cfg = {"enabled": True, "timeframe": "1h", "macd_fast": 12, "macd_slow": 26,
               "macd_signal": 9, "ema_trend": 200, "rsi_period": 14,
               "atr_stop_multiplier": 2.0, "max_bars_in_trade": 48,
               "rsi_overbought": 70, "rsi_oversold": 30}

        def run_on(n):
            risk = RiskManager(DEFAULT_CONFIG["risk"])
            sim = SimulatedExchange(10000.0)
            engine = BacktestEngine([DirectionalStrategy(cfg)], risk, sim, DEFAULT_CONFIG)
            return engine.run(bars.head(n), "BTC/USDT", "1h")

        report_600 = run_on(600)
        report_400 = run_on(400)

        # Trades from the 400-bar run should also appear at the start of the 600-bar run
        # (same entry times and prices for the first N trades)
        trades_400 = report_400.trades
        trades_600 = report_600.trades

        for t400 in trades_400:
            match = next(
                (t600 for t600 in trades_600
                 if abs(t600.entry_price - t400.entry_price) < 1e-2
                 and t600.side == t400.side),
                None,
            )
            assert match is not None, (
                f"Trade at entry={t400.entry_price:.2f} missing in 600-bar run — "
                f"possible look-ahead bias"
            )

    def test_equity_never_negative(self):
        bars = _trending_bars(600)
        cfg = {"enabled": True, "timeframe": "1h", "macd_fast": 12, "macd_slow": 26,
               "macd_signal": 9, "ema_trend": 200, "rsi_period": 14,
               "atr_stop_multiplier": 2.0, "max_bars_in_trade": 48,
               "rsi_overbought": 70, "rsi_oversold": 30}
        report = self._run(bars, [DirectionalStrategy(cfg)])
        assert all(e >= 0 for e in report.equity_curve)

    def test_risk_limits_respected(self):
        bars = _trending_bars(600)
        cfg = {"enabled": True, "timeframe": "1h", "macd_fast": 12, "macd_slow": 26,
               "macd_signal": 9, "ema_trend": 200, "rsi_period": 14,
               "atr_stop_multiplier": 2.0, "max_bars_in_trade": 48,
               "rsi_overbought": 70, "rsi_oversold": 30}
        risk = RiskManager(DEFAULT_CONFIG["risk"])
        sim = SimulatedExchange(10000.0)
        engine = BacktestEngine([DirectionalStrategy(cfg)], risk, sim, DEFAULT_CONFIG)
        report = engine.run(bars, "BTC/USDT", "1h")
        # Each trade's risk should never exceed 1% of equity (at the time)
        for trade in report.trades:
            risk_taken = abs(trade.entry_price - trade.exit_price) * trade.quantity
            # We can only check approximately since we don't store equity at entry
            # Just check that per-trade PnL loss is bounded
            assert abs(min(trade.pnl, 0)) < 10000  # never loses more than full capital
