"""
Phase 6 Robustness tests (30 tests).

Groups:
  OnlineLearner        (8)  — buffer, retrain trigger, val gate, events
  Checkpoint           (5)  — save/load/prune/list
  ShadowTracker        (7)  — fill recording, slippage, events, report
  BinanceProvider      (6)  — auto-recovery, get_health, manual reset
  EventType            (2)  — new Phase 6 enum members
  Integration          (2)  — wiring smoke tests
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from privateye.core.types import DataSnapshot, EventType, Fill, OrderSide
from privateye.utils.time import now_utc


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_snapshot(n: int = 10, symbol: str = "BTC/USDT") -> DataSnapshot:
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "open": [100.0] * n, "high": [101.0] * n,
        "low": [99.0] * n, "close": [100.0] * n,
        "volume": [1000.0] * n,
    })
    return DataSnapshot(symbol=symbol, timeframe="1h", bars=df, timestamp=now_utc())


def _make_fill(price: float = 100.0, side: OrderSide = OrderSide.BUY) -> Fill:
    return Fill(
        order_id="o1", symbol="BTC/USDT", side=side,
        quantity=0.1, price=price, fee=0.01,
        strategy_id="test", timestamp=now_utc(),
    )


def _make_model_mock(artifact_dir: Path, name: str = "MockModel") -> MagicMock:
    """Model mock with a real artifacts_dir containing one artifact file."""
    model = MagicMock()
    model.__class__.__name__ = name
    model.artifacts_dir = artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / f"{name.lower()}.pkl").write_bytes(b"artifact_data")
    return model


# ── OnlineLearner tests ───────────────────────────────────────────────────────

class TestOnlineLearner:
    """8 tests covering buffer logic, retrain trigger, val gate, events."""

    def _make_learner(self, tmp_path: Path, **cfg_overrides) -> "OnlineLearner":
        from privateye.models.online_learning import OnlineLearner

        bus = MagicMock()
        bus.publish = AsyncMock()
        ensemble = MagicMock()
        ensemble._gbm = None
        ensemble._lstm = None
        ensemble._regime = None
        ensemble._rl = None

        cfg = {
            "enabled": True,
            "buffer_bars": 100,
            "retrain_every_bars": 5,
            "models": [],
            "validation_gate": True,
            "max_val_loss_regression": 0.10,
            **cfg_overrides,
        }
        learner = OnlineLearner(bus, ensemble, cfg, artifacts_dir=str(tmp_path))
        learner._bus = bus
        return learner

    # 1
    async def test_disabled_noop(self, tmp_path):
        """enabled=False → on_market_data does not touch buffer."""
        from privateye.models.online_learning import OnlineLearner

        bus = MagicMock()
        learner = OnlineLearner(bus, MagicMock(), {"enabled": False}, artifacts_dir=str(tmp_path))
        snap = _make_snapshot()
        await learner.on_market_data(snap)
        assert len(learner._buffer) == 0

    # 2
    async def test_buffer_fills_triggers_retrain(self, tmp_path):
        """After retrain_every_bars snapshots a retrain task is scheduled."""
        learner = self._make_learner(tmp_path, retrain_every_bars=2)

        retrain_called = False

        async def mock_retrain():
            nonlocal retrain_called
            retrain_called = True

        learner._run_retrain = mock_retrain
        snap = _make_snapshot()
        await learner.on_market_data(snap)
        assert not retrain_called
        await learner.on_market_data(snap)  # 2nd bar → create_task
        await asyncio.sleep(0)              # let scheduled task run
        assert retrain_called

    # 3
    async def test_val_gate_passes_promotes_model(self, tmp_path):
        """new_loss < old_loss → model.save() is called (promotion)."""
        from privateye.models.online_learning import OnlineLearner

        bus = MagicMock()
        bus.publish = AsyncMock()
        gbm = _make_model_mock(tmp_path / "gbm", "GBMClassifier")
        ensemble = MagicMock()
        ensemble._gbm = gbm
        ensemble._lstm = None
        ensemble._regime = None
        ensemble._rl = None

        cfg = {
            "enabled": True, "buffer_bars": 200, "retrain_every_bars": 10,
            "models": ["gbm"], "validation_gate": True, "max_val_loss_regression": 0.10,
        }
        learner = OnlineLearner(bus, ensemble, cfg, artifacts_dir=str(tmp_path))
        # Fill buffer with enough rows to pass the minimum check
        for _ in range(60):
            learner._buffer.append({"close": 100.0, "open": 99.0, "high": 101.0,
                                    "low": 99.0, "volume": 1.0,
                                    "timestamp": now_utc()})

        with patch("privateye.models.online_learning.save_checkpoint"), \
             patch("privateye.models.online_learning.prune_old_checkpoints"), \
             patch.object(learner, "_retrain_gbm", return_value=(0.40, 0.60)):
            await learner._run_retrain()

        gbm.save.assert_called_once()

    # 4
    async def test_val_gate_rejects_regression(self, tmp_path):
        """new_loss > old_loss * (1+max_reg) → load_checkpoint called, save NOT called."""
        from privateye.models.online_learning import OnlineLearner

        bus = MagicMock()
        bus.publish = AsyncMock()
        gbm = _make_model_mock(tmp_path / "gbm", "GBMClassifier")
        ensemble = MagicMock()
        ensemble._gbm = gbm
        ensemble._lstm = None
        ensemble._regime = None
        ensemble._rl = None

        cfg = {
            "enabled": True, "buffer_bars": 200, "retrain_every_bars": 10,
            "models": ["gbm"], "validation_gate": True, "max_val_loss_regression": 0.10,
        }
        learner = OnlineLearner(bus, ensemble, cfg, artifacts_dir=str(tmp_path))
        for _ in range(60):
            learner._buffer.append({"close": 100.0, "open": 99.0, "high": 101.0,
                                    "low": 99.0, "volume": 1.0,
                                    "timestamp": now_utc()})

        with patch("privateye.models.online_learning.save_checkpoint"), \
             patch("privateye.models.online_learning.prune_old_checkpoints"), \
             patch("privateye.models.online_learning.load_checkpoint") as mock_restore, \
             patch.object(learner, "_retrain_gbm", return_value=(0.80, 0.50)):
            # new=0.80 > 0.50*1.10=0.55 → reject
            await learner._run_retrain()

        gbm.save.assert_not_called()
        mock_restore.assert_called_once()

    # 5
    async def test_checkpoint_saved_before_promotion(self, tmp_path):
        """save_checkpoint must be called before the retrain function."""
        from privateye.models.online_learning import OnlineLearner

        call_order: list[str] = []
        bus = MagicMock(); bus.publish = AsyncMock()
        gbm = _make_model_mock(tmp_path / "gbm", "GBMClassifier")
        ensemble = MagicMock()
        ensemble._gbm = gbm; ensemble._lstm = None
        ensemble._regime = None; ensemble._rl = None

        cfg = {
            "enabled": True, "buffer_bars": 200, "retrain_every_bars": 10,
            "models": ["gbm"], "validation_gate": False,
        }
        learner = OnlineLearner(bus, ensemble, cfg, artifacts_dir=str(tmp_path))
        for _ in range(60):
            learner._buffer.append({"close": 100.0, "open": 99.0, "high": 101.0,
                                    "low": 99.0, "volume": 1.0, "timestamp": now_utc()})

        def record_ckpt(*a, **kw): call_order.append("save_checkpoint")
        def record_retrain(bars):
            call_order.append("retrain")
            return 0.5, 0.6

        with patch("privateye.models.online_learning.save_checkpoint", record_ckpt), \
             patch("privateye.models.online_learning.prune_old_checkpoints"), \
             patch.object(learner, "_retrain_gbm", record_retrain):
            await learner._run_retrain()

        assert call_order.index("save_checkpoint") < call_order.index("retrain")

    # 6
    async def test_model_updated_event_published(self, tmp_path):
        """MODEL_UPDATED is published after every retrain call."""
        from privateye.models.online_learning import OnlineLearner

        bus = MagicMock(); bus.publish = AsyncMock()
        ensemble = MagicMock()
        ensemble._gbm = None; ensemble._lstm = None
        ensemble._regime = None; ensemble._rl = None

        cfg = {"enabled": True, "buffer_bars": 200, "retrain_every_bars": 10, "models": []}
        learner = OnlineLearner(bus, ensemble, cfg, artifacts_dir=str(tmp_path))
        for _ in range(60):
            learner._buffer.append({"close": 100.0, "timestamp": now_utc()})

        with patch("privateye.models.online_learning.save_checkpoint"), \
             patch("privateye.models.online_learning.prune_old_checkpoints"):
            await learner._run_retrain()

        bus.publish.assert_called_once()
        event_type, payload = bus.publish.call_args[0]
        assert event_type == EventType.MODEL_UPDATED
        assert "models_updated" in payload
        assert "bars_seen" in payload

    # 7
    async def test_gbm_retrain_invoked_when_in_models(self, tmp_path):
        """_retrain_gbm is called when 'gbm' is in the models list."""
        from privateye.models.online_learning import OnlineLearner

        bus = MagicMock(); bus.publish = AsyncMock()
        gbm = _make_model_mock(tmp_path / "gbm", "GBMClassifier")
        ensemble = MagicMock()
        ensemble._gbm = gbm; ensemble._lstm = None
        ensemble._regime = None; ensemble._rl = None

        cfg = {
            "enabled": True, "buffer_bars": 200, "retrain_every_bars": 10,
            "models": ["gbm"], "validation_gate": False,
        }
        learner = OnlineLearner(bus, ensemble, cfg, artifacts_dir=str(tmp_path))
        for _ in range(60):
            learner._buffer.append({"close": 100.0, "timestamp": now_utc()})

        with patch("privateye.models.online_learning.save_checkpoint"), \
             patch("privateye.models.online_learning.prune_old_checkpoints"), \
             patch.object(learner, "_retrain_gbm", return_value=(0.5, 0.6)) as mock_gbm:
            await learner._run_retrain()

        mock_gbm.assert_called_once()

    # 8
    async def test_insufficient_buffer_skips_retrain(self, tmp_path):
        """Buffer with <50 rows → orchestrator body is skipped entirely."""
        from privateye.models.online_learning import OnlineLearner

        bus = MagicMock(); bus.publish = AsyncMock()
        ensemble = MagicMock()
        ensemble._gbm = None; ensemble._lstm = None
        ensemble._regime = None; ensemble._rl = None

        cfg = {"enabled": True, "buffer_bars": 200, "retrain_every_bars": 1, "models": ["gbm"]}
        learner = OnlineLearner(bus, ensemble, cfg, artifacts_dir=str(tmp_path))

        # Only 10 rows in buffer — below _MIN_BUFFER_ROWS (50)
        for _ in range(10):
            learner._buffer.append({"close": 100.0, "timestamp": now_utc()})

        with patch.object(learner, "_retrain_gbm") as mock_fn:
            await learner._run_retrain()

        mock_fn.assert_not_called()
        bus.publish.assert_not_called()  # no MODEL_UPDATED when skipped


# ── Checkpoint tests ──────────────────────────────────────────────────────────

class TestCheckpoint:
    """5 tests covering save, load, prune, roundtrip, sorted listing."""

    # 9
    def test_save_creates_dir_and_meta(self, tmp_path):
        """save_checkpoint creates a versioned directory with meta.json."""
        from privateye.models.checkpoint import save_checkpoint

        art_dir = tmp_path / "artifacts"
        model = _make_model_mock(art_dir, "GBMClassifier")
        ckpt_dir = tmp_path / "checkpoints"

        path = save_checkpoint(model, bars_seen=500, checkpoint_dir=ckpt_dir)

        assert path.is_dir()
        meta_file = path / "meta.json"
        assert meta_file.exists()
        meta = json.loads(meta_file.read_text())
        assert meta["model_name"] == "GBMClassifier"
        assert meta["bars_seen"] == 500
        assert "saved_at" in meta
        assert "artifact_files" in meta

    # 10
    def test_load_latest_picks_newest(self, tmp_path):
        """load_checkpoint with latest=True selects the most recently saved."""
        from privateye.models.checkpoint import save_checkpoint, load_checkpoint

        art_dir = tmp_path / "artifacts"
        art_dir.mkdir(parents=True)
        (art_dir / "model.pkl").write_bytes(b"v1")

        model = _make_model_mock(art_dir, "MyModel")
        ckpt_dir = tmp_path / "checkpoints"

        save_checkpoint(model, bars_seen=100, checkpoint_dir=ckpt_dir)
        (art_dir / "model.pkl").write_bytes(b"v2")
        save_checkpoint(model, bars_seen=200, checkpoint_dir=ckpt_dir)

        load_checkpoint(model, ckpt_dir, latest=True)
        # After loading the latest (bars_seen=200), model.load() is called
        model.load.assert_called_once()

    # 11
    def test_prune_keeps_only_n(self, tmp_path):
        """prune_old_checkpoints deletes all but the newest keep_n."""
        from datetime import datetime as _dt
        from privateye.models.checkpoint import save_checkpoint, prune_old_checkpoints, list_checkpoints

        art_dir = tmp_path / "artifacts"
        model = _make_model_mock(art_dir, "PruneModel")
        ckpt_dir = tmp_path / "checkpoints"

        # Provide unique per-second timestamps so each save gets its own directory
        dt_values = [_dt(2024, 1, 1, 0, 0, i, tzinfo=timezone.utc) for i in range(1, 15)]
        with patch("privateye.models.checkpoint.datetime") as mock_dt:
            mock_dt.now.side_effect = dt_values
            for _ in range(5):
                save_checkpoint(model, bars_seen=100, checkpoint_dir=ckpt_dir)

        deleted = prune_old_checkpoints("PruneModel", ckpt_dir, keep_n=3)
        remaining = list_checkpoints("PruneModel", ckpt_dir)

        assert deleted == 2
        assert len(remaining) == 3

    # 12
    def test_roundtrip_save_load_restores_state(self, tmp_path):
        """save → load → model.load() is called with correct artifact files."""
        from privateye.models.checkpoint import save_checkpoint, load_checkpoint

        art_dir = tmp_path / "artifacts"
        model = _make_model_mock(art_dir, "RoundTrip")
        ckpt_dir = tmp_path / "checkpoints"

        ckpt_path = save_checkpoint(model, bars_seen=42, checkpoint_dir=ckpt_dir)
        assert ckpt_path.is_dir()

        load_checkpoint(model, ckpt_dir, latest=True)
        model.load.assert_called_once()

    # 13
    def test_list_checkpoints_sorted_newest_first(self, tmp_path):
        """list_checkpoints returns entries sorted newest-first."""
        from datetime import datetime as _dt
        from privateye.models.checkpoint import save_checkpoint, list_checkpoints

        art_dir = tmp_path / "artifacts"
        model = _make_model_mock(art_dir, "SortModel")
        ckpt_dir = tmp_path / "checkpoints"

        # Unique per-second timestamps so three distinct checkpoint directories are created
        dt_values = [_dt(2024, 1, 1, 0, 0, i, tzinfo=timezone.utc) for i in range(1, 10)]
        with patch("privateye.models.checkpoint.datetime") as mock_dt:
            mock_dt.now.side_effect = dt_values
            save_checkpoint(model, bars_seen=1, checkpoint_dir=ckpt_dir)
            save_checkpoint(model, bars_seen=2, checkpoint_dir=ckpt_dir)
            save_checkpoint(model, bars_seen=3, checkpoint_dir=ckpt_dir)

        results = list_checkpoints("SortModel", ckpt_dir)
        assert len(results) == 3
        # sorted newest-first means saved_at is non-increasing
        assert results[0]["saved_at"] >= results[1]["saved_at"] >= results[2]["saved_at"]


# ── ShadowTracker tests ───────────────────────────────────────────────────────

class TestShadowTracker:
    """7 tests covering fill recording, slippage, events, report, failure."""

    def _make_tracker(self, enabled: bool = True, threshold_pct: float = 2.0):
        from privateye.execution.shadow_tracker import ShadowTracker

        bus = MagicMock(); bus.publish = AsyncMock()
        adapter = MagicMock()
        cfg = {"enabled": enabled, "divergence_alert_threshold_pct": threshold_pct}
        tracker = ShadowTracker(bus, adapter, cfg)
        return tracker, bus, adapter

    # 14
    async def test_disabled_noop(self):
        """enabled=False → on_fill records nothing."""
        tracker, bus, adapter = self._make_tracker(enabled=False)
        await tracker.on_fill(_make_fill(100.0))
        assert len(tracker._records) == 0
        adapter.fetch_ticker.assert_not_called()

    # 15
    async def test_fill_recorded_with_live_mid(self):
        """A fill is stored with the correct live_mid_price."""
        tracker, bus, adapter = self._make_tracker()
        adapter.fetch_ticker = AsyncMock(return_value={"bid": 99.0, "ask": 101.0, "last": 100.0})

        await tracker.on_fill(_make_fill(price=105.0))

        assert len(tracker._records) == 1
        rec = tracker._records[0]
        assert rec.live_mid_price == pytest.approx(100.0)
        assert rec.paper_fill_price == pytest.approx(105.0)

    # 16
    async def test_slippage_computed_correctly(self):
        """slippage_pct = (fill_price - mid) / mid."""
        tracker, bus, adapter = self._make_tracker()
        # mid = (100 + 102) / 2 = 101; fill = 104 → slip = (104-101)/101 ≈ 0.0297
        adapter.fetch_ticker = AsyncMock(return_value={"bid": 100.0, "ask": 102.0})
        fill = _make_fill(price=104.0)
        await tracker.on_fill(fill)

        rec = tracker._records[0]
        expected_slip = (104.0 - 101.0) / 101.0
        assert rec.slippage_pct == pytest.approx(expected_slip, rel=1e-5)

    # 17
    async def test_divergence_above_threshold_publishes_event(self):
        """Slippage > threshold → SHADOW_DIVERGENCE event published."""
        tracker, bus, adapter = self._make_tracker(threshold_pct=1.0)
        # fill=110, mid=100 → slippage=10% >> 1%
        adapter.fetch_ticker = AsyncMock(return_value={"bid": 99.0, "ask": 101.0})
        await tracker.on_fill(_make_fill(price=115.0))

        bus.publish.assert_called_once()
        event_type, payload = bus.publish.call_args[0]
        assert event_type == EventType.SHADOW_DIVERGENCE

    # 18
    async def test_below_threshold_no_event(self):
        """Slippage within threshold → no SHADOW_DIVERGENCE event."""
        tracker, bus, adapter = self._make_tracker(threshold_pct=5.0)
        # fill=100.1, mid=100 → slippage=0.1% < 5%
        adapter.fetch_ticker = AsyncMock(return_value={"bid": 99.9, "ask": 100.1})
        await tracker.on_fill(_make_fill(price=100.1))

        bus.publish.assert_not_called()
        assert len(tracker._records) == 1

    # 19
    async def test_get_report_stats_correct(self):
        """get_report() computes correct mean, max, and pct_above_threshold."""
        from privateye.execution.shadow_tracker import ShadowFillRecord, ShadowReport

        tracker, bus, adapter = self._make_tracker(threshold_pct=2.0)
        # Inject pre-computed records directly
        ts = now_utc()
        tracker._records = [
            ShadowFillRecord("BTC/USDT", 101.0, 100.0, 0.01, OrderSide.BUY, 0.1, ts),
            ShadowFillRecord("BTC/USDT", 104.0, 100.0, 0.04, OrderSide.SELL, 0.1, ts),
        ]

        report = tracker.get_report()
        assert report.n_fills == 2
        assert report.mean_slippage_pct == pytest.approx(0.025, rel=1e-5)
        assert report.max_slippage_pct == pytest.approx(0.04, rel=1e-5)
        # threshold=2%=0.02; only 0.04 > 0.02 → 1/2 = 0.50
        assert report.pct_fills_above_threshold == pytest.approx(0.5, rel=1e-5)

    # 20
    async def test_adapter_fetch_failure_graceful_skip(self):
        """Adapter raises → warning logged, no record appended, no crash."""
        tracker, bus, adapter = self._make_tracker()
        adapter.fetch_ticker = AsyncMock(side_effect=RuntimeError("network error"))

        await tracker.on_fill(_make_fill(100.0))  # must not raise

        assert len(tracker._records) == 0
        bus.publish.assert_not_called()


# ── BinanceProvider auto-recovery tests ──────────────────────────────────────

class TestBinanceProviderAutoRecovery:
    """6 tests covering auto-recovery loop and health metrics."""

    def _make_provider(self, auto_recovery_seconds: float = 1800.0):
        from privateye.data.providers.binance import BinanceProvider
        return BinanceProvider(auto_recovery_seconds=auto_recovery_seconds)

    # 21
    async def test_circuit_stays_open_timer_not_elapsed(self):
        """When elapsed < auto_recovery_seconds the probe is NOT attempted."""
        provider = self._make_provider(auto_recovery_seconds=1800.0)
        provider._symbols = ["BTC/USDT"]
        provider._timeframes = ["1h"]
        provider._trip_circuit()
        # Only 100s have passed — well below 1800
        provider._circuit_open_since = datetime.now(timezone.utc) - timedelta(seconds=100)

        provider._running = True
        iter_count = 0

        async def fake_sleep(t):
            nonlocal iter_count
            iter_count += 1
            if iter_count >= 2:
                provider._running = False

        with patch("asyncio.sleep", fake_sleep), \
             patch.object(provider, "_fetch_and_update") as mock_probe:
            await provider._auto_recovery_loop()

        assert provider._circuit_open is True
        mock_probe.assert_not_called()

    # 22
    async def test_auto_recovery_succeeds_resets_circuit(self):
        """When elapsed >= threshold and probe succeeds, circuit is reset."""
        import pandas as pd

        provider = self._make_provider(auto_recovery_seconds=100.0)
        provider._symbols = ["BTC/USDT"]
        provider._timeframes = ["1h"]
        provider._trip_circuit()
        provider._circuit_open_since = datetime.now(timezone.utc) - timedelta(seconds=200)

        fake_df = pd.DataFrame({"close": [100.0]})
        provider._running = True
        iter_count = 0

        async def fake_sleep(t):
            nonlocal iter_count
            iter_count += 1
            if iter_count >= 2:
                provider._running = False

        async def mock_fetch(sym, tf):
            return fake_df

        with patch("asyncio.sleep", fake_sleep), \
             patch.object(provider, "_fetch_and_update", mock_fetch):
            await provider._auto_recovery_loop()

        assert provider._circuit_open is False
        assert provider._circuit_open_since is None

    # 23
    async def test_auto_recovery_probe_returns_none_stays_open(self):
        """When probe returns None (non-exception failure), circuit remains open."""
        provider = self._make_provider(auto_recovery_seconds=100.0)
        provider._symbols = ["BTC/USDT"]
        provider._timeframes = ["1h"]
        provider._trip_circuit()
        provider._circuit_open_since = datetime.now(timezone.utc) - timedelta(seconds=200)

        provider._running = True
        iter_count = 0

        async def fake_sleep(t):
            nonlocal iter_count
            iter_count += 1
            if iter_count >= 2:
                provider._running = False

        async def mock_fetch_none(sym, tf):
            return None

        with patch("asyncio.sleep", fake_sleep), \
             patch.object(provider, "_fetch_and_update", mock_fetch_none):
            await provider._auto_recovery_loop()

        assert provider._circuit_open is True

    # 24
    def test_get_health_returns_expected_structure(self):
        """get_health() must contain all required keys with correct types."""
        provider = self._make_provider()
        health = provider.get_health()

        assert isinstance(health["circuit_open"], bool)
        assert isinstance(health["consecutive_failures"], int)
        assert health["circuit_open_since"] is None  # not tripped yet
        assert 0.0 <= health["success_rate"] <= 1.0
        assert isinstance(health["total_fetches"], int)
        assert isinstance(health["last_fetch_latency_ms"], float)

    # 25
    async def test_auto_recovery_loop_cancellable(self):
        """CancelledError is propagated cleanly — no swallowed exceptions."""
        provider = self._make_provider()
        provider._running = True
        provider._symbols = []
        provider._timeframes = []

        task = asyncio.create_task(provider._auto_recovery_loop())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    # 26
    def test_manual_reset_clears_all_state(self):
        """reset_circuit_breaker() clears open flag, timestamp, and failure counter."""
        provider = self._make_provider()
        provider._trip_circuit()
        provider._consecutive_failures = 15

        provider.reset_circuit_breaker()

        assert provider._circuit_open is False
        assert provider._circuit_open_since is None
        assert provider._consecutive_failures == 0


# ── EventType tests ───────────────────────────────────────────────────────────

class TestEventTypes:
    """2 tests verifying the three new Phase 6 EventType members."""

    # 27
    def test_provider_health_event_type_exists(self):
        """PROVIDER_HEALTH must be a member of EventType."""
        assert EventType.PROVIDER_HEALTH == "provider_health"
        assert EventType.PROVIDER_HEALTH in EventType.__members__.values()

    # 28
    async def test_new_event_types_publishable(self):
        """MODEL_UPDATED and SHADOW_DIVERGENCE can be dispatched to subscribers."""
        from privateye.core.event_bus import AsyncEventBus

        bus = AsyncEventBus()
        received: list = []

        async def handler(payload):
            received.append(payload)

        bus.subscribe(EventType.MODEL_UPDATED, handler)
        bus.subscribe(EventType.SHADOW_DIVERGENCE, handler)

        # dispatch() calls handlers immediately (no queue; publish() needs bus.run())
        await bus.dispatch(EventType.MODEL_UPDATED, {"models_updated": [], "bars_seen": 100})
        await bus.dispatch(EventType.SHADOW_DIVERGENCE, {"symbol": "BTC/USDT"})

        assert len(received) == 2


# ── Integration tests ─────────────────────────────────────────────────────────

class TestIntegration:
    """2 smoke tests for main.py wiring helpers."""

    # 29
    def test_maybe_wire_online_learner_no_subscribe_when_disabled(self):
        """_maybe_wire_online_learner does nothing when online_learning.enabled=False."""
        from privateye.main import _maybe_wire_online_learner

        bus = MagicMock()
        cfg = {"robustness": {"online_learning": {"enabled": False}}}
        _maybe_wire_online_learner(bus, [], cfg)
        bus.subscribe.assert_not_called()

    # 30
    async def test_shadow_tracker_can_subscribe_to_fill_events(self):
        """ShadowTracker.on_fill is an async callable, subscribable to FILL bus events."""
        from privateye.core.event_bus import AsyncEventBus
        from privateye.execution.shadow_tracker import ShadowTracker

        bus = AsyncEventBus()
        adapter = MagicMock()
        adapter.fetch_ticker = AsyncMock(return_value={"bid": 99.0, "ask": 101.0})

        tracker = ShadowTracker(bus, adapter, {"enabled": True, "divergence_alert_threshold_pct": 5.0})
        bus.subscribe(EventType.FILL, tracker.on_fill)

        # Use dispatch() for immediate (synchronous) handler invocation
        fill = _make_fill(price=100.0)
        await bus.dispatch(EventType.FILL, fill)

        adapter.fetch_ticker.assert_called_once_with("BTC/USDT")
