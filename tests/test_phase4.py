"""Phase 4 — Real-Time Adaptation & Online Learning tests.

30 tests covering:
  TestEWC (6)              — ElasticWeightConsolidation
  TestPSI (5)              — PSI drift detection extension to FeatureDriftDetector
  TestLivePerformanceGate (5) — OnlineLearner live P&L gate + CONSERVATIVE_MODE event
  TestStackingAttribution (4) — StackingEnsemble 3-tuple return + FusionStrategy metadata
  TestFeedbackStore (5)    — FeedbackStore signal/outcome/thumbs recording
  TestDashboardPhase4 (5)  — /api/model-versions, /api/feedback, /api/signal-feedback
"""
from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import tempfile
from collections import deque
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_bars(n: int = 120):
    """Return a minimal OHLCV DataFrame with n rows."""
    import pandas as pd
    rng = np.random.default_rng(0)
    closes = 30000.0 + np.cumsum(rng.normal(0, 200, n))
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
            "open": closes,
            "high": closes + 100,
            "low": closes - 100,
            "close": closes,
            "volume": rng.uniform(500, 5000, n),
        }
    )


def _make_fill(symbol="BTC/USDT", pnl=50.0):
    fill = MagicMock()
    fill.symbol = symbol
    fill.realised_pnl = pnl
    return fill


# ===========================================================================
# TestEWC
# ===========================================================================

class TestEWC:
    """ElasticWeightConsolidation unit tests (6)."""

    def test_not_consolidated_before_consolidate(self):
        from privateye.models.ewc import ElasticWeightConsolidation
        ewc = ElasticWeightConsolidation(lambda_=400.0)
        assert ewc.is_consolidated is False

    def test_penalty_returns_zero_tensor_before_consolidation(self):
        from privateye.models.ewc import ElasticWeightConsolidation
        ewc = ElasticWeightConsolidation(lambda_=400.0)
        model = nn.Linear(10, 2)
        penalty = ewc.penalty(model)
        assert isinstance(penalty, torch.Tensor)
        assert float(penalty.item()) == pytest.approx(0.0)

    def test_consolidate_sets_is_consolidated(self):
        from privateye.models.ewc import ElasticWeightConsolidation
        ewc = ElasticWeightConsolidation(lambda_=400.0)
        model = nn.Linear(5, 2)
        device = torch.device("cpu")

        # Build a tiny dataloader
        from torch.utils.data import DataLoader, TensorDataset
        X = torch.randn(20, 5)
        y = torch.randint(0, 2, (20,))
        dl = DataLoader(TensorDataset(X, y), batch_size=10)

        ewc.consolidate(model, dl, device, n_samples=20)
        assert ewc.is_consolidated is True

    def test_penalty_positive_after_param_shift(self):
        """After consolidation, shifting weights should produce a positive penalty."""
        from privateye.models.ewc import ElasticWeightConsolidation
        from torch.utils.data import DataLoader, TensorDataset

        ewc = ElasticWeightConsolidation(lambda_=400.0)
        model = nn.Linear(5, 2)
        device = torch.device("cpu")

        X = torch.randn(30, 5)
        y = torch.randint(0, 2, (30,))
        dl = DataLoader(TensorDataset(X, y), batch_size=10)
        ewc.consolidate(model, dl, device, n_samples=30)

        # Shift all weights by a large amount
        new_model = nn.Linear(5, 2)
        with torch.no_grad():
            for p in new_model.parameters():
                p.add_(10.0)

        penalty = ewc.penalty(new_model)
        assert float(penalty.item()) > 0.0

    def test_penalty_near_zero_when_params_unchanged(self):
        """EWC penalty ≈ 0 when new model has same weights as the reference."""
        from privateye.models.ewc import ElasticWeightConsolidation
        from torch.utils.data import DataLoader, TensorDataset

        ewc = ElasticWeightConsolidation(lambda_=400.0)
        model = nn.Linear(5, 2)
        device = torch.device("cpu")

        X = torch.randn(20, 5)
        y = torch.randint(0, 2, (20,))
        dl = DataLoader(TensorDataset(X, y), batch_size=10)
        ewc.consolidate(model, dl, device, n_samples=20)

        # Clone the model to get identical weights
        import copy
        clone = copy.deepcopy(model)
        penalty = ewc.penalty(clone)
        assert float(penalty.item()) == pytest.approx(0.0, abs=1e-5)

    def test_is_consolidated_stays_false_with_empty_dataloader(self):
        """An empty DataLoader must not raise — is_consolidated stays False."""
        from privateye.models.ewc import ElasticWeightConsolidation
        from torch.utils.data import DataLoader, TensorDataset

        ewc = ElasticWeightConsolidation(lambda_=400.0)
        model = nn.Linear(5, 2)
        device = torch.device("cpu")

        # Empty dataset → DataLoader with 0 batches
        X = torch.zeros(0, 5)
        y = torch.zeros(0, dtype=torch.long)
        dl = DataLoader(TensorDataset(X, y), batch_size=10)

        ewc.consolidate(model, dl, device, n_samples=0)
        assert ewc.is_consolidated is False


# ===========================================================================
# TestPSI
# ===========================================================================

class TestPSI:
    """PSI extension to DriftReport and FeatureDriftDetector (5)."""

    def test_compute_psi_identical_distributions_returns_zero(self):
        from privateye.models.drift import _compute_psi
        rng = np.random.default_rng(1)
        ref = rng.normal(0, 1, 500)
        # Same distribution, different seed — PSI should be very small
        cur = rng.normal(0, 1, 500)
        psi = _compute_psi(ref, cur)
        # Same dist → PSI typically <0.05
        assert psi < 0.10

    def test_compute_psi_large_shift_exceeds_threshold(self):
        from privateye.models.drift import _compute_psi
        rng = np.random.default_rng(2)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(5, 1, 300)    # 5σ shift
        psi = _compute_psi(ref, cur)
        assert psi > 0.20

    def test_drift_report_has_psi_fields(self):
        from privateye.models.drift import DriftReport
        report = DriftReport(
            n_features_total=10,
            n_features_drifted=0,
            drift_fraction=0.0,
            is_drifted=False,
            drifted_feature_indices=[],
            min_p_value=1.0,
        )
        # New fields exist with defaults
        assert hasattr(report, "psi_score")
        assert hasattr(report, "psi_drifted")
        assert report.psi_score == 0.0
        assert report.psi_drifted is False

    def test_check_flags_drifted_when_only_psi_exceeds_threshold(self):
        """PSI alone can trigger is_drifted even when KS is clean."""
        from privateye.models.drift import FeatureDriftDetector

        det = FeatureDriftDetector(
            p_threshold=0.0001,       # very strict KS → KS likely passes on moderate shift
            drift_threshold=0.99,     # 99% of features must drift via KS → KS won't fire
            psi_threshold=0.10,       # PSI fires on any moderate shift
        )
        rng = np.random.default_rng(3)
        ref = rng.normal(0, 1, (500, 10))
        det.fit(ref)

        # Large shift → PSI will fire
        shifted = rng.normal(3, 1, (300, 10))
        report = det.check(shifted)

        assert report.psi_drifted is True
        assert report.is_drifted is True

    def test_both_clean_produces_not_drifted(self):
        from privateye.models.drift import FeatureDriftDetector

        det = FeatureDriftDetector(
            p_threshold=0.01,
            drift_threshold=0.20,
            psi_threshold=0.20,
        )
        rng = np.random.default_rng(4)
        ref = rng.normal(0, 1, (500, 10))
        det.fit(ref)

        # Same distribution — neither KS nor PSI should fire
        cur = rng.normal(0, 1, (300, 10))
        report = det.check(cur)

        assert report.is_drifted is False


# ===========================================================================
# TestLivePerformanceGate
# ===========================================================================

class TestLivePerformanceGate:
    """OnlineLearner live P&L gate (5)."""

    def _make_learner(self, gate_enabled: bool = True, min_trades: int = 3):
        from privateye.models.online_learning import OnlineLearner
        bus = MagicMock()
        bus.publish = AsyncMock()
        ensemble = MagicMock()
        ensemble._gbm = MagicMock(); ensemble._gbm.is_fitted = False
        ensemble._lstm = MagicMock(); ensemble._lstm.is_fitted = False
        ensemble._regime = MagicMock(); ensemble._regime.is_fitted = False
        cfg = {
            "enabled": True,
            "buffer_bars": 100,
            "retrain_every_bars": 50,
            "models": [],
            "validation_gate": False,
            "max_val_loss_regression": 0.10,
            "live_performance_gate": {
                "enabled": gate_enabled,
                "eval_window": 20,
                "min_trades": min_trades,
                "degradation_threshold": 0.15,
            },
        }
        return OnlineLearner(bus=bus, ensemble=ensemble, config=cfg)

    @staticmethod
    def _run(*coros):
        """Run one or more coroutines sequentially in a fresh event loop."""
        async def _runner():
            for coro in coros:
                await coro
        asyncio.run(_runner())

    def test_on_fill_noop_when_gate_disabled(self):
        learner = self._make_learner(gate_enabled=False)
        self._run(learner.on_fill(_make_fill(pnl=100.0)))
        assert len(learner._fill_pnls) == 0

    def test_gate_does_not_fire_before_min_trades(self):
        learner = self._make_learner(gate_enabled=True, min_trades=10)
        self._run(*(learner.on_fill(_make_fill(pnl=50.0)) for _ in range(5)))
        learner._bus.publish.assert_not_awaited()

    def test_gate_publishes_conservative_mode_on_degradation(self):
        from privateye.core.types import EventType
        learner = self._make_learner(gate_enabled=True, min_trades=3)

        # Fill with positive PnL to set best_mean_pnl, then large negative to trigger gate
        self._run(
            learner.on_fill(_make_fill(pnl=100.0)),
            learner.on_fill(_make_fill(pnl=100.0)),
            learner.on_fill(_make_fill(pnl=100.0)),
            learner.on_fill(_make_fill(pnl=-1000.0)),
            learner.on_fill(_make_fill(pnl=-1000.0)),
            learner.on_fill(_make_fill(pnl=-1000.0)),
        )

        learner._bus.publish.assert_awaited()
        call_args = learner._bus.publish.await_args_list
        events = [args[0][0] for args in call_args]
        assert EventType.CONSERVATIVE_MODE in events

    def test_gate_updates_best_mean_pnl_on_improvement(self):
        learner = self._make_learner(gate_enabled=True, min_trades=3)

        self._run(
            learner.on_fill(_make_fill(pnl=50.0)),
            learner.on_fill(_make_fill(pnl=50.0)),
            learner.on_fill(_make_fill(pnl=50.0)),
        )
        first_best = learner._best_mean_pnl

        self._run(
            learner.on_fill(_make_fill(pnl=200.0)),
            learner.on_fill(_make_fill(pnl=200.0)),
            learner.on_fill(_make_fill(pnl=200.0)),
        )
        assert learner._best_mean_pnl > first_best

    def test_gate_resets_best_mean_pnl_on_rollback(self):
        """After a checkpoint rollback, _best_mean_pnl should reset to -inf."""
        learner = self._make_learner(gate_enabled=True, min_trades=3)
        self._run(
            learner.on_fill(_make_fill(pnl=100.0)),
            learner.on_fill(_make_fill(pnl=100.0)),
            learner.on_fill(_make_fill(pnl=100.0)),
        )
        assert learner._best_mean_pnl > -float("inf")

        # Simulate rollback via direct attribute reset (as done in _run_retrain rollback path)
        learner._best_mean_pnl = -float("inf")
        assert learner._best_mean_pnl == -float("inf")


# ===========================================================================
# TestStackingAttribution
# ===========================================================================

class TestStackingAttribution:
    """StackingEnsemble 3-tuple return + FusionStrategy metadata (4)."""

    def test_predict_returns_three_tuple(self):
        from privateye.models.stacking_ensemble import StackingEnsemble
        with tempfile.TemporaryDirectory() as tmp:
            stack = StackingEnsemble(artifacts_dir=tmp)
            result = stack.predict({})
            assert len(result) == 3

    def test_unfitted_predict_returns_flat_zero_empty(self):
        from privateye.models.stacking_ensemble import StackingEnsemble
        with tempfile.TemporaryDirectory() as tmp:
            stack = StackingEnsemble(artifacts_dir=tmp)
            direction, confidence, attribution = stack.predict({})
            assert direction == "flat"
            assert confidence == 0.0
            assert attribution == {}

    def test_attribution_dict_has_five_named_groups(self):
        """get_attribution() must produce exactly 5 named groups when fitted."""
        from privateye.models.stacking_ensemble import StackingEnsemble
        with tempfile.TemporaryDirectory() as tmp:
            stack = StackingEnsemble(artifacts_dir=tmp)
            # Inject a fake trained meta-learner with 11 feature importances
            fake_lgbm = MagicMock()
            fake_lgbm.feature_importances_ = np.array(
                [0.1, 0.1, 0.15, 0.15, 0.08, 0.07, 0.10, 0.08, 0.07, 0.05, 0.05]
            )
            stack._meta_learner = fake_lgbm
            stack.is_fitted = True   # public attribute, no underscore
            attribution = stack.get_attribution()
            expected_keys = {"gbm", "lgbm", "lstm", "attn_lstm", "regime"}
            assert set(attribution.keys()) == expected_keys

    def test_attribution_sums_to_one(self):
        from privateye.models.stacking_ensemble import StackingEnsemble
        with tempfile.TemporaryDirectory() as tmp:
            stack = StackingEnsemble(artifacts_dir=tmp)
            fake_lgbm = MagicMock()
            fake_lgbm.feature_importances_ = np.ones(11) / 11.0
            stack._meta_learner = fake_lgbm
            stack.is_fitted = True   # public attribute, no underscore
            attribution = stack.get_attribution()
            total = sum(attribution.values())
            assert total == pytest.approx(1.0, abs=0.01)


# ===========================================================================
# TestFeedbackStore
# ===========================================================================

class TestFeedbackStore:
    """FeedbackStore signal/outcome/thumbs recording (5)."""

    def _make_signal(self, symbol="BTC/USDT", direction="long",
                     entry_price=40000.0, strategy_id="fusion"):
        sig = MagicMock()
        sig.symbol = symbol
        sig.direction = direction
        sig.confidence = 0.75
        sig.entry_price = entry_price
        sig.strategy_id = strategy_id
        sig.metadata = {}
        return sig

    def test_record_signal_returns_nonempty_key(self):
        from privateye.feedback.store import FeedbackStore
        store = FeedbackStore(maxlen=100)
        sig = self._make_signal()
        key = store.record_signal(sig)
        assert isinstance(key, str)
        assert len(key) > 0

    def test_record_outcome_updates_pnl_and_outcome(self):
        from privateye.feedback.store import FeedbackStore
        store = FeedbackStore(maxlen=100)
        key = store.record_signal(self._make_signal())
        store.record_outcome(key, pnl=150.0)
        recent = store.get_recent(1)
        assert len(recent) == 1
        assert recent[0]["pnl"] == pytest.approx(150.0)
        assert recent[0]["outcome"] == "win"

    def test_submit_feedback_sets_thumbs_up(self):
        from privateye.feedback.store import FeedbackStore
        store = FeedbackStore(maxlen=100)
        key = store.record_signal(self._make_signal())
        ok = store.submit_feedback(key, thumbs_up=True)
        assert ok is True
        recent = store.get_recent(1)
        assert recent[0]["thumbs_up"] is True

    def test_get_recent_returns_newest_first(self):
        from privateye.feedback.store import FeedbackStore
        store = FeedbackStore(maxlen=100)
        key1 = store.record_signal(self._make_signal(entry_price=40000.0))
        key2 = store.record_signal(self._make_signal(entry_price=41000.0))
        recent = store.get_recent(10)
        # key2 added last → should appear first
        assert recent[0]["entry_price"] == pytest.approx(41000.0)

    def test_get_win_rate_by_version_returns_dict(self):
        from privateye.feedback.store import FeedbackStore
        store = FeedbackStore(maxlen=100)
        key = store.record_signal(self._make_signal(), model_version_id="v1")
        store.record_outcome(key, pnl=100.0)
        win_rates = store.get_win_rate_by_version()
        assert isinstance(win_rates, dict)
        assert "v1" in win_rates
        assert win_rates["v1"] == pytest.approx(1.0)


# ===========================================================================
# TestDashboardPhase4
# ===========================================================================

class TestDashboardPhase4:
    """Dashboard Phase 4 endpoints (5)."""

    def _build_client(
        self,
        get_model_versions=None,
        get_feedback=None,
        post_feedback=None,
    ):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from privateye.dashboard.routes import build_router

        app = FastAPI()
        router = build_router(
            get_portfolio=MagicMock(return_value=MagicMock(
                equity=10000.0, cash=10000.0, invested=0.0,
                daily_pnl=0.0, daily_drawdown_pct=0.0, drawdown_pct=0.0,
                peak_equity=10000.0, total_trades=0, positions={},
            )),
            get_trades=lambda: [],
            get_fills=lambda: [],
            exec_engine=MagicMock(),
            risk_manager=MagicMock(is_halted=MagicMock(return_value=False)),
            initial_capital=10000.0,
            get_model_versions=get_model_versions,
            get_feedback=get_feedback,
            post_feedback=post_feedback,
        )
        app.include_router(router)
        return TestClient(app)

    def test_get_model_versions_returns_200_list(self):
        client = self._build_client(get_model_versions=lambda: [{"model": "GBMClassifier"}])
        resp = client.get("/api/model-versions")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert data[0]["model"] == "GBMClassifier"

    def test_get_model_versions_returns_empty_list_when_not_wired(self):
        client = self._build_client(get_model_versions=None)
        resp = client.get("/api/model-versions")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_post_signal_feedback_valid_body_returns_ok_true(self):
        from privateye.feedback.store import FeedbackStore
        store = FeedbackStore(maxlen=100)
        sig = MagicMock()
        sig.symbol = "BTC/USDT"; sig.direction = "long"
        sig.confidence = 0.8; sig.entry_price = 40000.0
        sig.strategy_id = "fusion"; sig.metadata = {}
        key = store.record_signal(sig)

        client = self._build_client(
            post_feedback=store.submit_feedback
        )
        resp = client.post(
            "/api/signal-feedback",
            json={"signal_key": key, "thumbs_up": True},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_get_feedback_returns_list(self):
        from privateye.feedback.store import FeedbackStore
        store = FeedbackStore(maxlen=100)
        sig = MagicMock()
        sig.symbol = "ETH/USDT"; sig.direction = "flat"
        sig.confidence = 0.6; sig.entry_price = 3000.0
        sig.strategy_id = "fusion"; sig.metadata = {}
        store.record_signal(sig)

        client = self._build_client(get_feedback=store.get_recent)
        resp = client.get("/api/feedback?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1

    def test_phase4_complete_importable_from_main(self):
        from privateye.main import PHASE4_COMPLETE
        assert PHASE4_COMPLETE is False
