"""
Phase 7 test suite — Performance Analytics, XAI & Tax Reporting.

30 tests across:
  TestAdvancedMetrics     (7)  — ulcer, VaR, CVaR, streaks, omega, to_dict
  TestMonteCarloWired     (5)  — trade permutation, block bootstrap, ruin, seed
  TestTaxReporting        (8)  — FIFO gain/loss, long-term, fees, year filter, CSV
  TestXAI                 (5)  — SHAP key in ensemble, top_features in fusion
  TestDashboardEquityCurve (5) — equity/drawdown curve endpoints
"""
from __future__ import annotations

import csv
import io
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from privateye.backtesting.metrics import BacktestReport, compute_metrics
from privateye.core.types import Direction, TradeRecord
from privateye.dashboard.routes import build_router
from privateye.models.monte_carlo import run_block_bootstrap, run_trade_permutation
from privateye.reports.tax import AnnualTaxReport, compute_tax_lots, export_tax_csv


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dt(days_offset: int = 0) -> datetime:
    """UTC datetime offset from a fixed base."""
    return datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=days_offset)


def _make_trade(
    pnl: float,
    pnl_pct: float | None = None,
    side: Direction = Direction.LONG,
    entry_price: float = 40_000.0,
    exit_price: float | None = None,
    quantity: float = 0.1,
    fees: float = 0.0,
    entry_days: int = 0,
    exit_days: int = 1,
) -> TradeRecord:
    if pnl_pct is None:
        pnl_pct = (pnl / (entry_price * quantity)) * 100
    if exit_price is None:
        exit_price = entry_price + pnl / quantity
    return TradeRecord(
        symbol="BTC/USDT",
        side=side,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        entry_time=_dt(entry_days),
        exit_time=_dt(exit_days),
        pnl=pnl,
        pnl_pct=pnl_pct,
        fees=fees,
        strategy_id="test",
        exit_reason="signal_reversal",
        bars_held=exit_days - entry_days,
    )


def _make_equity_curve(trades: list[TradeRecord], initial: float = 10_000.0) -> list[float]:
    eq = initial
    curve = [eq]
    for t in trades:
        eq += t.pnl
        curve.append(eq)
    return curve


def _make_report_with_trades(trades: list[TradeRecord], initial: float = 10_000.0) -> BacktestReport:
    equity_curve = _make_equity_curve(trades, initial)
    return compute_metrics(trades, equity_curve, initial_capital=initial)


# ── Test Group 1: Advanced Metrics ───────────────────────────────────────────

class TestAdvancedMetrics:

    def test_ulcer_index_nonzero_when_drawdown_present(self):
        """Ulcer index must be > 0 when the equity curve dips below its peak."""
        trades = [
            _make_trade(pnl=200.0),
            _make_trade(pnl=-300.0, exit_days=2),   # drawdown
            _make_trade(pnl=100.0, exit_days=3),
        ]
        report = _make_report_with_trades(trades)
        assert report.ulcer_index > 0.0

    def test_var_95_equals_fifth_percentile(self):
        """VaR 95% should equal the 5th percentile of per-trade returns."""
        pnl_pcts = [1.0, -2.0, 3.0, -4.0, 0.5, -1.5, 2.5, -0.5, 1.2, -3.0]
        trades = [
            _make_trade(pnl=p * 4, pnl_pct=p, exit_days=i + 1)
            for i, p in enumerate(pnl_pcts)
        ]
        report = _make_report_with_trades(trades)
        expected_var = float(np.percentile(pnl_pcts, 5))
        assert abs(report.var_95_pct - expected_var) < 1e-9

    def test_cvar_is_worse_than_or_equal_to_var(self):
        """CVaR (expected shortfall) must be <= VaR (further into the tail)."""
        trades = [_make_trade(pnl=p * 4, pnl_pct=p, exit_days=i + 1)
                  for i, p in enumerate([1, -2, 3, -4, 0.5, -1.5, 2.5, -3.5, 1.2, -3.0])]
        report = _make_report_with_trades(trades)
        # CVaR should be <= VaR (more negative means worse loss)
        assert report.cvar_95_pct <= report.var_95_pct + 1e-9

    def test_max_consecutive_wins(self):
        """Streak counter must correctly identify the longest winning run."""
        # Pattern: W W W L W W → max_wins = 3
        pnls = [100, 200, 150, -50, 80, 90]
        trades = [_make_trade(pnl=p, exit_days=i + 1) for i, p in enumerate(pnls)]
        report = _make_report_with_trades(trades)
        assert report.max_consecutive_wins == 3

    def test_max_consecutive_losses(self):
        """Streak counter must correctly identify the longest losing run."""
        # Pattern: L L L L W L → max_losses = 4
        pnls = [-50, -30, -70, -20, 100, -40]
        trades = [_make_trade(pnl=p, exit_days=i + 1) for i, p in enumerate(pnls)]
        report = _make_report_with_trades(trades)
        assert report.max_consecutive_losses == 4

    def test_omega_ratio_greater_than_one_for_profitable_system(self):
        """Omega ratio > 1 when gains outweigh losses."""
        # 3 winning trades of +200%, 1 losing trade of -50%
        trades = [
            _make_trade(pnl=200, pnl_pct=2.0, exit_days=1),
            _make_trade(pnl=200, pnl_pct=2.0, exit_days=2),
            _make_trade(pnl=200, pnl_pct=2.0, exit_days=3),
            _make_trade(pnl=-50,  pnl_pct=-0.5, exit_days=4),
        ]
        report = _make_report_with_trades(trades)
        assert report.omega_ratio > 1.0

    def test_to_dict_contains_all_expected_keys_and_excludes_trades(self):
        """to_dict() must include all scalar fields but omit the raw trades list."""
        trades = [_make_trade(pnl=100), _make_trade(pnl=-50, exit_days=2)]
        report = _make_report_with_trades(trades)
        d = report.to_dict()

        required_keys = {
            "total_trades", "win_rate", "profit_factor", "sharpe_ratio",
            "max_drawdown_pct", "ulcer_index", "var_95_pct", "cvar_95_pct",
            "max_consecutive_wins", "max_consecutive_losses", "omega_ratio",
            "equity_curve",
        }
        for key in required_keys:
            assert key in d, f"Missing key: {key}"
        assert "trades" not in d, "Raw trades list must be excluded from to_dict()"
        assert isinstance(d["equity_curve"], list)


# ── Test Group 2: Monte Carlo ─────────────────────────────────────────────────

class TestMonteCarloWired:

    def test_run_backtest_calls_monte_carlo_when_enabled(self):
        """run_backtest() must invoke run_trade_permutation when mc enabled."""
        from privateye.main import run_backtest

        fake_trade = _make_trade(pnl=500.0)
        fake_report = MagicMock()
        fake_report.trades = [fake_trade]

        cfg = {
            "backtesting": {"initial_capital": 10000.0, "data_dir": "data/historical",
                            "fee_maker": 0.001, "fee_taker": 0.001,
                            "slippage_pct": 0.0005, "max_fill_pct_of_volume": 0.30},
            "symbols": ["BTC/USDT"],
            "primary_timeframe": "1h",
            "risk": {},
            "ml": {"monte_carlo": {"enabled": True, "n_simulations": 10, "seed": 1}},
            "alt_data": {"enabled": False},
            "data": {"sqlite_path": ":memory:"},
        }

        # Imports inside run_backtest() are resolved at call time via sys.modules,
        # so patch the source modules and the Monte Carlo function directly.
        with patch("privateye.backtesting.engine.BacktestEngine") as mock_eng_cls, \
             patch("privateye.data.providers.csv_provider.CSVProvider") as mock_prov_cls, \
             patch("privateye.risk.manager.RiskManager"), \
             patch("privateye.backtesting.simulator.SimulatedExchange"), \
             patch("privateye.main._build_strategies", return_value=[]), \
             patch("privateye.main._build_advanced_risk", return_value=(None, None, None)), \
             patch("privateye.main.asyncio.run"), \
             patch("privateye.models.monte_carlo.run_trade_permutation",
                   return_value=MagicMock()) as mock_mc:

            mock_eng = MagicMock()
            mock_eng.run.return_value = fake_report
            mock_eng_cls.return_value = mock_eng

            mock_prov = MagicMock()
            bars = pd.DataFrame({
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC"),
                "open": [1.0] * 5, "high": [1.1] * 5,
                "low": [0.9] * 5, "close": [1.0] * 5, "volume": [100.0] * 5,
            })
            mock_prov.load.return_value = bars
            mock_prov_cls.return_value = mock_prov

            run_backtest(cfg)
            mock_mc.assert_called_once()

    def test_run_trade_permutation_ruin_probability_in_range(self):
        """ruin_probability must be in [0.0, 1.0]."""
        pnl = [100, -50, 200, -80, 150, -30, 90, -20, 120, -60]
        report = run_trade_permutation(pnl, n_simulations=100, rng_seed=1)
        assert 0.0 <= report.ruin_probability <= 1.0

    def test_run_trade_permutation_empty_trades_handled(self):
        """Empty PnL sequence must not raise; returns zero ruin probability."""
        report = run_trade_permutation([], n_simulations=50, rng_seed=42)
        assert report.ruin_probability == 0.0
        assert report.n_simulations == 50

    def test_run_trade_permutation_seed_deterministic(self):
        """Same seed must produce identical results across two calls."""
        pnl = [100, -50, 200, -80, 150]
        r1 = run_trade_permutation(pnl, n_simulations=200, rng_seed=99)
        r2 = run_trade_permutation(pnl, n_simulations=200, rng_seed=99)
        assert r1.p50_sharpe == r2.p50_sharpe
        assert r1.ruin_probability == r2.ruin_probability

    def test_run_block_bootstrap_returns_monte_carlo_report(self):
        """Block bootstrap must return a MonteCarloReport with mode=block_bootstrap."""
        from privateye.models.monte_carlo import MonteCarloReport
        prices = np.cumsum(np.abs(np.random.default_rng(7).normal(0, 100, 200))) + 30_000

        def dummy_strategy(close: np.ndarray) -> np.ndarray:
            returns = np.diff(close) / close[:-1]
            return returns * 1000.0  # fake PnL

        report = run_block_bootstrap(
            prices, dummy_strategy, n_simulations=20, block_size=10, rng_seed=5
        )
        assert isinstance(report, MonteCarloReport)
        assert report.mode == "block_bootstrap"
        assert 0.0 <= report.ruin_probability <= 1.0


# ── Test Group 3: Tax Reporting ───────────────────────────────────────────────

class TestTaxReporting:

    def _long_trade(self, entry: float, exit: float, qty: float = 0.1,
                    entry_days: int = 0, exit_days: int = 1, fees: float = 0.0):
        pnl = (exit - entry) * qty - fees
        return _make_trade(
            pnl=pnl, pnl_pct=pnl / (entry * qty) * 100,
            side=Direction.LONG, entry_price=entry, exit_price=exit,
            quantity=qty, fees=fees, entry_days=entry_days, exit_days=exit_days,
        )

    def test_basic_long_gain(self):
        """LONG trade: gain = (exit - entry) × qty − fees."""
        trade = self._long_trade(40_000, 45_000, qty=0.1)
        report = compute_tax_lots([trade])
        assert len(report.all_events) == 1
        event = report.all_events[0]
        assert abs(event.gain_loss - 500.0) < 1e-6
        assert event.proceeds == pytest.approx(4_500.0)
        assert event.cost_basis == pytest.approx(4_000.0)

    def test_basic_long_loss(self):
        """LONG trade at a loss should be captured as a negative gain_loss."""
        trade = self._long_trade(40_000, 38_000, qty=0.1)
        report = compute_tax_lots([trade])
        assert report.all_events[0].gain_loss < 0.0
        assert report.total_realized_loss < 0.0
        assert report.total_realized_gain == 0.0

    def test_long_term_flag_for_trade_held_over_365_days(self):
        """Trades held ≥ 365 days must be flagged is_long_term=True."""
        trade = self._long_trade(40_000, 45_000, entry_days=0, exit_days=400)
        report = compute_tax_lots([trade])
        assert report.all_events[0].is_long_term is True
        assert len(report.long_term_gains) == 1
        assert len(report.short_term_gains) == 0

    def test_short_term_flag_for_trade_held_under_365_days(self):
        """Trades held < 365 days must be flagged is_long_term=False."""
        trade = self._long_trade(40_000, 45_000, entry_days=0, exit_days=100)
        report = compute_tax_lots([trade])
        assert report.all_events[0].is_long_term is False
        assert len(report.short_term_gains) == 1
        assert len(report.long_term_gains) == 0

    def test_fees_deducted_from_gain_loss(self):
        """Fees must reduce the net gain_loss."""
        fees = 10.0
        trade = self._long_trade(40_000, 45_000, qty=0.1, fees=fees)
        report = compute_tax_lots([trade])
        assert abs(report.all_events[0].gain_loss - (500.0 - fees)) < 1e-6

    def test_annual_year_filter(self):
        """Only events with exit_time in the specified year should be included."""
        t2023 = _make_trade(pnl=100, entry_days=0, exit_days=1)
        t2023 = TradeRecord(
            symbol="BTC/USDT", side=Direction.LONG,
            entry_price=40_000, exit_price=40_100,
            quantity=0.1,
            entry_time=datetime(2023, 6, 1, tzinfo=timezone.utc),
            exit_time=datetime(2023, 12, 31, tzinfo=timezone.utc),
            pnl=10.0, pnl_pct=0.025, fees=0.0,
            strategy_id="test", exit_reason="signal_reversal", bars_held=1,
        )
        t2024 = TradeRecord(
            symbol="BTC/USDT", side=Direction.LONG,
            entry_price=40_000, exit_price=45_000,
            quantity=0.1,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            exit_time=datetime(2024, 6, 1, tzinfo=timezone.utc),
            pnl=500.0, pnl_pct=1.25, fees=0.0,
            strategy_id="test", exit_reason="signal_reversal", bars_held=1,
        )
        report_2024 = compute_tax_lots([t2023, t2024], year=2024)
        assert len(report_2024.all_events) == 1
        assert report_2024.all_events[0].exit_time.year == 2024

    def test_net_gain_loss_equals_total_gain_plus_total_loss(self):
        """net_gain_loss must equal total_realized_gain + total_realized_loss."""
        trades = [
            self._long_trade(40_000, 45_000),
            self._long_trade(45_000, 43_000, exit_days=2),
        ]
        report = compute_tax_lots(trades)
        assert abs(report.net_gain_loss
                   - (report.total_realized_gain + report.total_realized_loss)) < 1e-9

    def test_export_tax_csv_valid_format(self):
        """CSV export must produce a file with correct columns and row count."""
        trades = [
            self._long_trade(40_000, 45_000),
            self._long_trade(45_000, 43_000, exit_days=2),
        ]
        report = compute_tax_lots(trades)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            tmp_path = Path(f.name)

        try:
            export_tax_csv(report, tmp_path)
            with tmp_path.open() as fh:
                rows = list(csv.DictReader(fh))
            expected_cols = {
                "symbol", "quantity", "proceeds", "cost_basis", "gain_loss",
                "entry_time", "exit_time", "holding_days", "is_long_term", "fees",
            }
            assert set(rows[0].keys()) == expected_cols
            assert len(rows) == 2
        finally:
            tmp_path.unlink(missing_ok=True)


# ── Test Group 4: XAI ─────────────────────────────────────────────────────────

class TestXAI:

    def _make_bars(self, n: int = 120) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        closes = 30_000 + np.cumsum(rng.normal(0, 200, n))
        return pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
            "open": closes, "high": closes + 100, "low": closes - 100,
            "close": closes, "volume": rng.uniform(500, 5000, n),
        })

    def test_ensemble_predict_always_has_shap_key(self):
        """predict() must include 'shap' key regardless of whether GBM is loaded."""
        from privateye.models.inference import ModelEnsemble
        ens = ModelEnsemble.__new__(ModelEnsemble)
        ens.artifacts_dir = Path("nonexistent")
        ens._regime         = None
        ens._lstm           = None
        ens._gbm            = None
        ens._rl             = None
        # Phase 2 models (not present in artifacts)
        ens._lgbm           = None
        ens._attn_lstm      = None
        ens._neural_regime  = None
        ens._stacking       = None
        ens._loaded         = True

        bars = self._make_bars()
        result = ens.predict(bars)
        assert "shap" in result

    def test_shap_is_none_when_gbm_not_loaded(self):
        """When GBM is absent, result['shap'] must be None."""
        from privateye.models.inference import ModelEnsemble
        ens = ModelEnsemble.__new__(ModelEnsemble)
        ens.artifacts_dir = Path("nonexistent")
        ens._regime         = None
        ens._lstm           = None
        ens._gbm            = None
        ens._rl             = None
        # Phase 2 models (not present in artifacts)
        ens._lgbm           = None
        ens._attn_lstm      = None
        ens._neural_regime  = None
        ens._stacking       = None
        ens._loaded         = True

        bars = self._make_bars()
        result = ens.predict(bars)
        assert result["shap"] is None

    def test_shap_is_none_when_gbm_returns_empty_dict(self):
        """get_shap_explanation returning {} must produce result['shap'] = None."""
        from privateye.models.inference import ModelEnsemble
        mock_gbm = MagicMock()
        mock_gbm.is_fitted = True
        mock_gbm.get_shap_explanation.return_value = {}

        ens = ModelEnsemble.__new__(ModelEnsemble)
        ens.artifacts_dir = Path("nonexistent")
        ens._regime         = None
        ens._lstm           = None
        ens._gbm            = mock_gbm
        ens._rl             = None
        # Phase 2 models (not present in artifacts)
        ens._lgbm           = None
        ens._attn_lstm      = None
        ens._neural_regime  = None
        ens._stacking       = None
        ens._loaded         = True

        bars = self._make_bars()
        result = ens.predict(bars)
        assert result["shap"] is None

    def test_fusion_signal_has_top_features_key(self):
        """FusionStrategy signal metadata must always contain 'top_features'."""
        from privateye.core.types import DataSnapshot, Direction as Dir
        from privateye.strategies.fusion_strategy import FusionStrategy

        cfg = {
            "strategy_id": "fusion", "timeframe": "1h",
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "ema_trend": 200, "rsi_period": 14,
            "rsi_overbought": 70, "rsi_oversold": 30,
            "gbm_gate": False, "regime_gate": False,
            "artifacts_dir": "nonexistent",
        }
        strategy = FusionStrategy(cfg)

        # Inject a mock ensemble that always returns a LONG signal with no SHAP
        mock_ens = MagicMock()
        mock_ens.predict.return_value = {
            "regime": (0, [0.25, 0.25, 0.25, 0.25]),
            "lstm": ("long", 0.8),
            "gbm": 0.7,
            "shap": None,
        }
        strategy._ensemble = mock_ens

        rng = np.random.default_rng(1)
        n = 300
        closes = 30_000 + np.cumsum(rng.normal(0, 200, n))
        bars = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
            "open": closes, "high": closes + 200, "low": closes - 200,
            "close": closes, "volume": rng.uniform(500, 5000, n),
        })
        snapshot = DataSnapshot(symbol="BTC/USDT", timeframe="1h", bars=bars)
        signals = strategy.on_data(snapshot)

        assert len(signals) > 0
        assert "top_features" in signals[0].metadata

    def test_top_features_has_at_most_five_entries(self):
        """top_features must contain ≤ 5 entries even if SHAP returns many."""
        from privateye.core.types import DataSnapshot
        from privateye.strategies.fusion_strategy import FusionStrategy

        cfg = {
            "strategy_id": "fusion", "timeframe": "1h",
            "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
            "ema_trend": 200, "rsi_period": 14,
            "rsi_overbought": 70, "rsi_oversold": 30,
            "gbm_gate": False, "regime_gate": False,
            "artifacts_dir": "nonexistent",
        }
        strategy = FusionStrategy(cfg)

        # 10 SHAP features
        shap_data = {f"feat_{i}": float(i * 0.1) for i in range(10)}
        mock_ens = MagicMock()
        mock_ens.predict.return_value = {
            "regime": (0, [0.25, 0.25, 0.25, 0.25]),
            "lstm": ("long", 0.9),
            "gbm": 0.8,
            "shap": shap_data,
        }
        strategy._ensemble = mock_ens

        rng = np.random.default_rng(2)
        n = 300
        closes = 30_000 + np.cumsum(rng.normal(0, 200, n))
        bars = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
            "open": closes, "high": closes + 200, "low": closes - 200,
            "close": closes, "volume": rng.uniform(500, 5000, n),
        })
        snapshot = DataSnapshot(symbol="BTC/USDT", timeframe="1h", bars=bars)
        signals = strategy.on_data(snapshot)

        assert len(signals) > 0
        top_features = signals[0].metadata["top_features"]
        assert len(top_features) <= 5


# ── Test Group 5: Dashboard Equity/Drawdown Curves ────────────────────────────

class TestDashboardEquityCurve:
    """Tests for GET /api/equity-curve and GET /api/drawdown-curve."""

    INITIAL = 10_000.0

    def _make_app(self, trades: list) -> TestClient:
        app = FastAPI()
        portfolio_mock = MagicMock()
        portfolio_mock.equity = self.INITIAL
        portfolio_mock.cash = self.INITIAL
        portfolio_mock.daily_pnl = 0.0
        portfolio_mock.daily_drawdown_pct = 0.0
        portfolio_mock.drawdown_pct = 0.0
        portfolio_mock.peak_equity = self.INITIAL
        portfolio_mock.total_trades = len(trades)
        portfolio_mock.invested = 0.0
        portfolio_mock.positions = {}

        risk_mock = MagicMock()
        risk_mock.is_halted.return_value = False
        risk_mock._halt_reason = None

        exec_mock = MagicMock()

        router = build_router(
            get_portfolio=lambda: portfolio_mock,
            get_trades=lambda: trades,
            get_fills=lambda: [],
            exec_engine=exec_mock,
            risk_manager=risk_mock,
            initial_capital=self.INITIAL,
        )
        app.include_router(router)
        return TestClient(app)

    def test_equity_curve_empty_trades_returns_baseline(self):
        """No trades → single point with equity = initial_capital."""
        client = self._make_app([])
        resp = client.get("/api/equity-curve")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["equity"] == pytest.approx(self.INITIAL)
        assert "ts" in data[0]

    def test_equity_curve_tracks_cumulative_pnl(self):
        """Each point must equal initial_capital + cumulative PnL."""
        trades = [
            _make_trade(pnl=500.0, exit_days=1),
            _make_trade(pnl=-200.0, exit_days=2),
        ]
        client = self._make_app(trades)
        resp = client.get("/api/equity-curve")
        assert resp.status_code == 200
        data = resp.json()
        assert data[0]["equity"] == pytest.approx(self.INITIAL + 500.0)
        assert data[1]["equity"] == pytest.approx(self.INITIAL + 500.0 - 200.0)

    def test_equity_curve_length_matches_trade_count(self):
        """Number of data points must equal number of trades."""
        trades = [_make_trade(pnl=100.0, exit_days=i + 1) for i in range(5)]
        client = self._make_app(trades)
        resp = client.get("/api/equity-curve")
        assert resp.status_code == 200
        assert len(resp.json()) == 5

    def test_drawdown_curve_values_are_non_positive(self):
        """drawdown_pct must be ≤ 0.0 at every point."""
        trades = [
            _make_trade(pnl=1000.0, exit_days=1),
            _make_trade(pnl=-500.0, exit_days=2),
            _make_trade(pnl=-300.0, exit_days=3),
            _make_trade(pnl=800.0,  exit_days=4),
        ]
        client = self._make_app(trades)
        resp = client.get("/api/drawdown-curve")
        assert resp.status_code == 200
        for point in resp.json():
            assert point["drawdown_pct"] <= 0.0 + 1e-9

    def test_drawdown_curve_zero_at_new_peak(self):
        """drawdown_pct must be 0.0 immediately after a new equity high."""
        # First trade sets new peak → drawdown_pct = 0
        trades = [
            _make_trade(pnl=500.0, exit_days=1),
            _make_trade(pnl=-100.0, exit_days=2),
            _make_trade(pnl=600.0, exit_days=3),   # new peak → dd = 0
        ]
        client = self._make_app(trades)
        resp = client.get("/api/drawdown-curve")
        assert resp.status_code == 200
        data = resp.json()
        assert data[0]["drawdown_pct"] == pytest.approx(0.0)   # first trade = new peak
        assert data[2]["drawdown_pct"] == pytest.approx(0.0)   # third trade = new peak
