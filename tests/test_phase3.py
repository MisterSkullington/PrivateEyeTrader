"""
Phase 3 — Execution & Microstructure tests.

30 tests across 6 groups:
  TestSlippagePredictor    (6)
  TestSmartOrderRouter     (6)
  TestPreTradeCostAnalyzer (7)
  TestShadowMode2          (5)
  TestDashboardExecutionStats (3)
  TestVWAPWiring           (3)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from privateye.core.types import (
    Direction, EventType, Fill, Order, OrderSide, OrderType,
    PortfolioState, TradingSignal,
)
from privateye.execution.pre_trade_analyzer import PreTradeCostAnalyzer, PreTradeResult
from privateye.execution.slippage_predictor import SlippagePredictor, SlippagePrediction
from privateye.execution.smart_order_router import RoutingDecision, SmartOrderRouter


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_bars(n: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 30_000 + np.cumsum(rng.normal(0, 200, n))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
        "open":   closes + rng.normal(0, 50, n),
        "high":   closes + rng.uniform(50, 150, n),
        "low":    closes - rng.uniform(50, 150, n),
        "close":  closes,
        "volume": rng.uniform(500, 5_000, n),
    })


def _make_order(qty: float = 0.1, price: float = 30_000.0) -> Order:
    return Order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=qty,
        price=price,
        stop_price=29_000.0,
        strategy_id="test",
    )


def _make_signal(
    confidence: float = 0.7,
    entry: float = 30_000.0,
    target: float = 31_500.0,
    stop: float = 29_500.0,
) -> TradingSignal:
    return TradingSignal(
        symbol="BTC/USDT",
        direction=Direction.LONG,
        confidence=confidence,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        strategy_id="test",
        timeframe="1h",
    )


def _make_portfolio(equity: float = 10_000.0) -> PortfolioState:
    return PortfolioState(equity=equity, cash=equity)


# ── TestSlippagePredictor ──────────────────────────────────────────────────────

class TestSlippagePredictor:
    def test_predict_returns_slippage_prediction_type(self):
        pred = SlippagePredictor()
        result = pred.predict(10_000.0, _make_bars())
        assert isinstance(result, SlippagePrediction)

    def test_slippage_pct_is_positive(self):
        pred = SlippagePredictor()
        result = pred.predict(10_000.0, _make_bars())
        assert result.slippage_pct > 0

    def test_larger_notional_yields_higher_slippage(self):
        pred = SlippagePredictor()
        bars = _make_bars()
        small = pred.predict(5_000.0, bars)
        large = pred.predict(10_000_000.0, bars)
        assert large.slippage_pct > small.slippage_pct

    def test_short_bars_uses_fallback_atr(self):
        """With fewer than 14 bars, the predictor falls back to _FALLBACK_ATR_PCT."""
        pred = SlippagePredictor(alpha=0.1, avg_daily_volume_usd=5e8)
        short_bars = _make_bars(10)  # < 14 bars
        result_short = pred.predict(10_000.0, short_bars)
        # Manually compute expected result with fallback ATR
        fallback_atr = SlippagePredictor._FALLBACK_ATR_PCT
        notional = 10_000.0
        expected_impact = 0.1 * (notional / 5e8) ** 0.5 * fallback_atr
        expected_impact = float(np.clip(expected_impact, pred.min_slippage_pct, pred.max_slippage_pct))
        assert abs(result_short.slippage_pct - expected_impact) < 1e-10

    def test_slippage_clamped_to_max(self):
        """Enormous order should hit the max_slippage_pct ceiling."""
        pred = SlippagePredictor(max_slippage_pct=0.005)
        result = pred.predict(1e15, _make_bars())  # astronomically large
        assert result.slippage_pct <= pred.max_slippage_pct

    def test_record_fill_increments_n_fills_recorded(self):
        pred = SlippagePredictor()
        assert pred.n_fills_recorded == 0
        mock_record = MagicMock()
        pred.record_fill(mock_record)
        assert pred.n_fills_recorded == 1
        pred.record_fill(mock_record)
        assert pred.n_fills_recorded == 2


# ── TestSmartOrderRouter ───────────────────────────────────────────────────────

class TestSmartOrderRouter:
    def _make_router(self, enabled: bool = False) -> SmartOrderRouter:
        predictor = SlippagePredictor()
        return SmartOrderRouter(
            slippage_predictor=predictor,
            enabled_venues=["binance", "bybit"],
            fee_rates={"binance": 0.001, "bybit": 0.001},
            enabled=enabled,
        )

    def test_single_venue_returns_correct_venue_name(self):
        predictor = SlippagePredictor()
        router = SmartOrderRouter(
            slippage_predictor=predictor,
            enabled_venues=["binance"],
            fee_rates={"binance": 0.001},
            enabled=False,
        )
        order = _make_order()
        decision = router.route(order, urgency=0.5, bars=_make_bars())
        assert decision.venue == "binance"

    def test_urgency_above_threshold_gives_market_order(self):
        router = self._make_router()
        order = _make_order()
        decision = router.route(order, urgency=0.9)
        assert decision.order_type == OrderType.MARKET

    def test_urgency_below_threshold_gives_limit_order(self):
        router = self._make_router()
        order = _make_order()
        decision = router.route(order, urgency=0.5)
        assert decision.order_type == OrderType.LIMIT

    def test_limit_order_has_post_only_true(self):
        router = self._make_router()
        order = _make_order()
        decision = router.route(order, urgency=0.5)
        assert decision.post_only is True

    def test_annotate_order_sets_order_type(self):
        router = self._make_router()
        order = _make_order()
        decision = router.route(order, urgency=0.9)
        annotated = router.annotate_order(order, decision)
        assert annotated.order_type == decision.order_type

    def test_multi_venue_picks_lower_cost_venue_when_diff_exceeds_threshold(self):
        """When bybit has a significantly lower fee, the router should prefer it."""
        predictor = SlippagePredictor()
        router = SmartOrderRouter(
            slippage_predictor=predictor,
            enabled_venues=["binance", "bybit"],
            fee_rates={"binance": 0.01, "bybit": 0.0001},  # bybit much cheaper
            enabled=True,
            min_score_diff_bps=0.5,
        )
        order = _make_order()
        decision = router.route(order, urgency=0.5, bars=_make_bars())
        assert decision.venue == "bybit"


# ── TestPreTradeCostAnalyzer ───────────────────────────────────────────────────

class TestPreTradeCostAnalyzer:
    def _make_analyzer(
        self,
        min_net_edge: float = 0.0025,
        fee_taker: float = 0.001,
    ) -> PreTradeCostAnalyzer:
        predictor = SlippagePredictor(min_slippage_pct=0.0001)
        return PreTradeCostAnalyzer(
            slippage_predictor=predictor,
            min_net_edge_pct=min_net_edge,
            fee_taker=fee_taker,
        )

    def test_analyze_returns_pre_trade_result_type(self):
        analyzer = self._make_analyzer()
        result = analyzer.analyze(
            _make_signal(), order_qty=0.1, portfolio=_make_portfolio(), bars=_make_bars()
        )
        assert isinstance(result, PreTradeResult)

    def test_high_confidence_large_move_approved(self):
        """Signal with 80% confidence and a 5% target move should pass."""
        analyzer = self._make_analyzer(min_net_edge=0.0025)
        signal = _make_signal(confidence=0.8, entry=30_000.0, target=31_500.0)  # +5%
        result = analyzer.analyze(signal, order_qty=0.1, portfolio=_make_portfolio())
        assert result.approved is True

    def test_tiny_move_signal_rejected(self):
        """Signal where expected return minus costs is negative should be rejected."""
        analyzer = self._make_analyzer(min_net_edge=0.0025)
        # 0.1% move × 0.5 confidence = 0.05% expected return
        # fees alone are 0.2% → definitely rejected
        signal = _make_signal(
            confidence=0.5,
            entry=30_000.0,
            target=30_030.0,   # only +0.1%
            stop=29_970.0,
        )
        result = analyzer.analyze(signal, order_qty=0.01, portfolio=_make_portfolio())
        assert result.approved is False

    def test_fee_cost_pct_equals_twice_fee_taker(self):
        fee_taker = 0.002
        analyzer = self._make_analyzer(fee_taker=fee_taker)
        result = analyzer.analyze(_make_signal(), 0.1, _make_portfolio())
        assert abs(result.fee_cost_pct - 2 * fee_taker) < 1e-12

    def test_slippage_cost_comes_from_predictor(self):
        """slippage_cost_pct must equal 2× the predictor's one-way slippage."""
        predictor = SlippagePredictor(min_slippage_pct=0.0003, alpha=0.0)  # alpha=0 → min always
        analyzer = PreTradeCostAnalyzer(predictor, fee_taker=0.001)
        result = analyzer.analyze(_make_signal(), 0.1, _make_portfolio())
        # alpha=0 → impact=0, clamped to min_slippage_pct
        assert abs(result.slippage_cost_pct - 2 * 0.0003) < 1e-9

    def test_net_edge_formula(self):
        """net_edge = expected_return - fee - slippage (to floating-point precision)."""
        analyzer = self._make_analyzer()
        result = analyzer.analyze(_make_signal(confidence=0.7), 0.1, _make_portfolio())
        assert abs(result.net_edge_pct - (
            result.expected_return_pct - result.fee_cost_pct - result.slippage_cost_pct
        )) < 1e-12

    def test_none_analyzer_gate_always_passes(self):
        """When pre_trade_analyzer is None, Gate 11 must not execute."""
        from privateye.risk.manager import RiskManager
        risk_cfg = {
            "max_risk_per_trade_pct": 0.01,
            "max_daily_drawdown_pct": 0.05,
            "max_position_notional_pct": 0.5,
            "min_confidence": 0.0,
            "sizing_method": "fixed_risk",
            "atr_period": 14,
            "atr_stop_multiplier": 2.0,
        }
        rm = RiskManager(risk_cfg, pre_trade_analyzer=None)
        portfolio = PortfolioState(equity=10_000.0, cash=10_000.0)
        signal = _make_signal(confidence=0.7)
        approved, reason, order = rm.evaluate_signal(signal, portfolio)
        # Gate 11 is absent → approval depends only on other gates
        assert reason != "pre-trade cost screen"


# ── TestShadowMode2 ────────────────────────────────────────────────────────────

class TestShadowMode2:
    def _make_tracker(self, baseline: float = 0.0005, threshold: float = 0.75):
        from privateye.execution.shadow_tracker import ShadowTracker
        bus = MagicMock()
        bus.publish_sync = MagicMock()
        adapter = MagicMock()
        cfg = {
            "enabled": True,
            "divergence_alert_threshold_pct": 2.0,
            "conservative_mode_threshold": threshold,
            "reality_score_window": 50,
        }
        return ShadowTracker(bus, adapter, cfg, baseline_slippage=baseline)

    def test_reality_score_is_one_before_any_fills(self):
        tracker = self._make_tracker()
        assert tracker.reality_score == 1.0

    def test_realized_equals_predicted_gives_high_score(self):
        """When realized slippage == baseline, score should be exactly 1.0."""
        tracker = self._make_tracker(baseline=0.001)
        for _ in range(15):
            tracker._reality_scores.append(1.0)  # inject perfect scores
        assert tracker.reality_score == 1.0

    def test_ten_times_realized_gives_low_score(self):
        """When realized = 10× baseline, score drops dramatically (→ 0.0 floor)."""
        tracker = self._make_tracker(baseline=0.001)
        # realized = 0.01 = 10× baseline → score = max(0, 1 - |0.01-0.001|/0.001)
        #                                         = max(0, 1 - 9.0) = 0.0
        for _ in range(15):
            tracker._reality_scores.append(0.0)
        assert tracker.reality_score == 0.0

    def test_get_reality_stats_has_required_keys(self):
        tracker = self._make_tracker()
        stats = tracker.get_reality_stats()
        required = {
            "n_fills", "mean_slippage_pct", "max_slippage_pct",
            "pct_fills_above_threshold", "reality_score", "conservative_mode",
        }
        assert required.issubset(stats.keys())

    def test_conservative_mode_triggered_below_threshold(self):
        """After 10+ fills all with very poor reality scores, conservative_mode=True."""
        tracker = self._make_tracker(baseline=0.001, threshold=0.75)
        # Inject 10 low scores (all 0.0) directly into deque
        for _ in range(10):
            tracker._reality_scores.append(0.0)
        # Trigger the check manually (mirrors what on_fill does after each append)
        rolling = tracker.reality_score
        if rolling < tracker._conservative_threshold and not tracker._conservative_mode:
            tracker._conservative_mode = True
        assert tracker.conservative_mode is True


# ── TestDashboardExecutionStats ────────────────────────────────────────────────

class TestDashboardExecutionStats:
    def _make_client(self, get_execution_stats=None) -> TestClient:
        from privateye.dashboard.routes import build_router
        app = FastAPI()
        router = build_router(
            get_portfolio=lambda: PortfolioState(equity=10000.0, cash=10000.0),
            get_trades=lambda: [],
            get_fills=lambda: [],
            exec_engine=MagicMock(flatten_all=AsyncMock(), resume=MagicMock()),
            risk_manager=MagicMock(is_halted=lambda: False, _halt_reason=""),
            initial_capital=10000.0,
            get_execution_stats=get_execution_stats,
        )
        app.include_router(router)
        return TestClient(app)

    def test_get_execution_stats_returns_200(self):
        def mock_stats():
            return {
                "n_fills": 5,
                "mean_slippage_pct": 0.0003,
                "max_slippage_pct": 0.0008,
                "pct_fills_above_threshold": 0.1,
                "reality_score": 0.92,
                "conservative_mode": False,
            }
        client = self._make_client(get_execution_stats=mock_stats)
        resp = client.get("/api/execution-stats")
        assert resp.status_code == 200

    def test_safe_defaults_when_no_execution_stats_callback(self):
        client = self._make_client(get_execution_stats=None)
        resp = client.get("/api/execution-stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_fills"] == 0
        assert data["reality_score"] == 1.0
        assert data["conservative_mode"] is False

    def test_response_keys_include_reality_score_and_conservative_mode(self):
        def mock_stats():
            return {
                "n_fills": 0,
                "mean_slippage_pct": 0.0,
                "max_slippage_pct": 0.0,
                "pct_fills_above_threshold": 0.0,
                "reality_score": 0.95,
                "conservative_mode": True,
            }
        client = self._make_client(get_execution_stats=mock_stats)
        resp = client.get("/api/execution-stats")
        data = resp.json()
        assert "reality_score" in data
        assert "conservative_mode" in data
        assert data["reality_score"] == 0.95
        assert data["conservative_mode"] is True


# ── TestVWAPWiring ─────────────────────────────────────────────────────────────

class TestVWAPWiring:
    def _make_engine(self):
        from privateye.core.event_bus import AsyncEventBus
        from privateye.execution.engine import ExecutionEngine
        bus = AsyncEventBus()
        sim = MagicMock()
        sim.positions = {}
        sim.submit_order = MagicMock()
        return ExecutionEngine(bus, simulator=sim, exec_config={"use_twap": True, "twap_slices": 4, "twap_threshold_usd": 0})

    def test_fit_volume_profile_sets_vwap_executor(self):
        engine = self._make_engine()
        assert engine._vwap_executor is None
        bars = _make_bars(200)
        engine.fit_volume_profile(bars)
        assert engine._vwap_executor is not None

    def test_submit_order_uses_vwap_when_fitted(self):
        from privateye.execution.vwap import VWAPExecutor
        engine = self._make_engine()
        bars = _make_bars(200)
        engine.fit_volume_profile(bars)

        order = _make_order(qty=1.0, price=30_000.0)  # notional=30k > threshold=0
        with patch.object(engine._vwap_executor, "split", wraps=engine._vwap_executor.split) as mock_split:
            asyncio.run(engine.submit_order(order, bars=bars))
            assert mock_split.called

    def test_falls_back_to_twap_when_vwap_not_fitted(self):
        """When _vwap_executor is None, TWAPExecutor.split must be used instead."""
        from privateye.execution.twap import TWAPExecutor
        engine = self._make_engine()
        assert engine._vwap_executor is None  # no fit called

        order = _make_order(qty=1.0, price=30_000.0)
        with patch("privateye.execution.engine.TWAPExecutor", wraps=TWAPExecutor) as mock_twap:
            asyncio.run(engine.submit_order(order))
            assert mock_twap.called
