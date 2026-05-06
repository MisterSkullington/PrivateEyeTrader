"""Unit tests for risk manager and position sizing."""
import pytest

from privateye.core.types import Direction, PortfolioState, TradingSignal
from privateye.risk.manager import RiskManager
from privateye.risk.sizing import cap_to_max_notional, fixed_risk_size, volatility_targeted_size


def _make_portfolio(equity=10000.0, cash=10000.0, positions=None, daily_dd=0.0) -> PortfolioState:
    return PortfolioState(
        equity=equity, cash=cash,
        positions=positions or {},
        daily_pnl=0.0,
        daily_drawdown_pct=daily_dd,
        peak_equity=equity,
    )


def _make_signal(direction=Direction.LONG, confidence=0.7, entry=30000.0, stop=29400.0) -> TradingSignal:
    return TradingSignal(
        symbol="BTC/USDT",
        direction=direction,
        confidence=confidence,
        entry_price=entry,
        stop_price=stop,
        target_price=31200.0,
        strategy_id="test",
        timeframe="1h",
    )


class TestFixedRiskSizing:
    def test_basic(self):
        qty = fixed_risk_size(10000, 0.01, 30000, 29400)
        # risk = 10000 * 0.01 = 100; stop dist = 600; qty = 100/600 ≈ 0.1667
        assert abs(qty - 100 / 600) < 1e-6

    def test_zero_stop_distance(self):
        qty = fixed_risk_size(10000, 0.01, 30000, 30000)
        assert qty == 0.0

    def test_zero_equity(self):
        qty = fixed_risk_size(0, 0.01, 30000, 29400)
        assert qty == 0.0


class TestCapToMaxNotional:
    def test_cap_applies(self):
        # max 20% of 10000 = 2000; at price 30000 → max 0.0667
        qty = cap_to_max_notional(1.0, 30000, 10000, 0.20)
        assert abs(qty - 2000 / 30000) < 1e-6

    def test_no_cap_when_below(self):
        qty = cap_to_max_notional(0.01, 30000, 10000, 0.20)
        assert qty == 0.01


class TestRiskManager:
    def setup_method(self):
        self.mgr = RiskManager({
            "max_risk_per_trade_pct": 0.01,
            "max_daily_drawdown_pct": 0.05,
            "max_position_notional_pct": 0.20,
            "min_confidence": 0.55,
        })

    def test_approves_valid_signal(self):
        portfolio = _make_portfolio()
        signal = _make_signal()
        approved, reason, order = self.mgr.evaluate_signal(signal, portfolio)
        assert approved
        assert order is not None
        assert order.quantity > 0

    def test_rejects_low_confidence(self):
        portfolio = _make_portfolio()
        signal = _make_signal(confidence=0.3)
        approved, reason, _ = self.mgr.evaluate_signal(signal, portfolio)
        assert not approved
        assert "confidence" in reason.lower()

    def test_rejects_when_halted(self):
        portfolio = _make_portfolio()
        self.mgr.halt("test halt")
        signal = _make_signal()
        approved, reason, _ = self.mgr.evaluate_signal(signal, portfolio)
        assert not approved
        assert "halted" in reason.lower()

    def test_circuit_breaker_triggers(self):
        portfolio = _make_portfolio(daily_dd=0.06)
        signal = _make_signal()
        approved, reason, _ = self.mgr.evaluate_signal(signal, portfolio)
        assert not approved
        assert self.mgr.is_halted()

    def test_approves_flat_signal_when_halted(self):
        self.mgr.halt("test")
        portfolio = _make_portfolio()
        signal = _make_signal(direction=Direction.FLAT)
        approved, reason, _ = self.mgr.evaluate_signal(signal, portfolio)
        assert approved  # exits always go through

    def test_resume_clears_halt(self):
        self.mgr.halt("test")
        assert self.mgr.is_halted()
        self.mgr.resume()
        assert not self.mgr.is_halted()

    def test_position_size_respects_risk_pct(self):
        portfolio = _make_portfolio(equity=10000, cash=10000)
        signal = _make_signal(entry=30000.0, stop=29400.0)  # stop dist = 600
        approved, _, order = self.mgr.evaluate_signal(signal, portfolio)
        assert approved
        # Risk = qty × 600 <= 10000 × 0.01 = 100
        risk_taken = order.quantity * abs(signal.entry_price - signal.stop_price)
        assert risk_taken <= 100 * 1.001  # allow tiny float tolerance

    def test_insufficient_cash_rejected(self):
        portfolio = _make_portfolio(equity=10000, cash=1.0)  # almost no cash
        signal = _make_signal(entry=30000.0, stop=29000.0)
        approved, reason, _ = self.mgr.evaluate_signal(signal, portfolio)
        assert not approved
        assert "cash" in reason.lower()
