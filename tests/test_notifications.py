"""
Phase 9 — Notification Intelligence
Tests for AlertRecord, _build_notifier_config, AlertManager, Notifier,
and the /api/alerts dashboard endpoint.

Target: 316 total tests (286 existing + 30 new).
"""
from __future__ import annotations

import asyncio
import os
from collections import deque
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from privateye.alerts.alert_manager import AlertManager, AlertRecord, _build_notifier_config
from privateye.alerts.notifier import Notifier
from privateye.core.event_bus import AsyncEventBus
from privateye.core.types import AlertLevel, EventType
from privateye.dashboard.routes import build_router


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_notifier(min_level: str = "INFO") -> Notifier:
    return Notifier({"min_level": min_level})


def _make_manager(
    notifier: Notifier | None = None,
    rate_limit: int = 0,
    history_size: int = 100,
) -> AlertManager:
    bus = AsyncEventBus()
    n = notifier or _make_notifier()
    return AlertManager(n, bus, {
        "rate_limit_seconds": rate_limit,
        "history_size": history_size,
    })


def _make_fill(symbol: str = "BTC/USDT", price: float = 50000.0, pnl: float = 10.0):
    fill = MagicMock()
    fill.symbol = symbol
    fill.side.value = "buy"
    fill.quantity = 0.01
    fill.price = price
    fill.fee = 0.001
    fill.realised_pnl = pnl
    return fill


# ─────────────────────────────────────────────────────────────────────────────
# Group 1: TestAlertRecord  (3 tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestAlertRecord:
    def test_fields_stored(self):
        now = datetime.now(timezone.utc)
        rec = AlertRecord(
            level=AlertLevel.INFO,
            message="hello",
            timestamp=now,
            event_type="fill",
        )
        assert rec.level == AlertLevel.INFO
        assert rec.message == "hello"
        assert rec.timestamp == now
        assert rec.event_type == "fill"

    def test_timestamp_is_datetime(self):
        now = datetime.now(timezone.utc)
        rec = AlertRecord(AlertLevel.CRITICAL, "boom", now, "kill")
        assert isinstance(rec.timestamp, datetime)

    def test_event_type_is_string(self):
        now = datetime.now(timezone.utc)
        rec = AlertRecord(AlertLevel.DEBUG, "debug msg", now, "market_data")
        assert isinstance(rec.event_type, str)


# ─────────────────────────────────────────────────────────────────────────────
# Group 2: TestNotifier  (7 tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestNotifier:
    def test_send_skips_below_min_level(self):
        n = _make_notifier("CRITICAL")
        n._tg_available = False
        n._email_available = False
        # Should not raise; INFO is below CRITICAL min_level
        asyncio.get_event_loop().run_until_complete(n.send(AlertLevel.INFO, "quiet"))

    def test_send_passes_at_min_level(self):
        n = _make_notifier("INFO")
        n._tg_available = False
        n._email_available = False
        asyncio.get_event_loop().run_until_complete(n.send(AlertLevel.INFO, "msg"))
        # No exception = pass

    def test_critical_would_trigger_email(self):
        n = _make_notifier("INFO")
        n._tg_available = False
        n._email_available = True
        with patch.object(n, "_send_email", new_callable=AsyncMock) as mock_email:
            asyncio.get_event_loop().run_until_complete(
                n.send(AlertLevel.CRITICAL, "breach!")
            )
            mock_email.assert_called_once()

    def test_info_skips_email(self):
        n = _make_notifier("INFO")
        n._tg_available = False
        n._email_available = True
        with patch.object(n, "_send_email", new_callable=AsyncMock) as mock_email:
            asyncio.get_event_loop().run_until_complete(
                n.send(AlertLevel.INFO, "just info")
            )
            mock_email.assert_not_called()

    def test_send_telegram_skips_when_no_token(self):
        n = Notifier({"min_level": "INFO", "telegram_token": "", "chat_id": ""})
        assert not n._tg_available
        # send() should complete without attempting Telegram
        asyncio.get_event_loop().run_until_complete(n.send(AlertLevel.INFO, "no tg"))

    def test_notify_fill_calls_send_info(self):
        n = _make_notifier("INFO")
        n._tg_available = False
        n._email_available = False
        fill = _make_fill()
        # Patch at class level so bound method picks up the mock
        with patch.object(Notifier, "send", new_callable=AsyncMock) as mock_send:
            asyncio.get_event_loop().run_until_complete(n.notify_fill(fill))
            mock_send.assert_called_once()
            args = mock_send.call_args[0]
            assert args[0] == AlertLevel.INFO

    def test_notify_error_calls_send_critical(self):
        n = _make_notifier("INFO")
        n._tg_available = False
        n._email_available = False
        with patch.object(n, "send", new_callable=AsyncMock) as mock_send:
            asyncio.get_event_loop().run_until_complete(
                n.notify_error("TestComponent", "oops")
            )
            mock_send.assert_called_once()
            args = mock_send.call_args[0]
            assert args[0] == AlertLevel.CRITICAL


# ─────────────────────────────────────────────────────────────────────────────
# Group 3: TestAlertManager  (12 tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestAlertManager:
    def test_subscribe_all_wires_fill_handler(self):
        bus = AsyncEventBus()
        n = _make_notifier()
        mgr = AlertManager(n, bus, {"rate_limit_seconds": 0})
        mgr.subscribe_all()
        assert EventType.FILL in bus._handlers

    def test_subscribe_all_wires_shadow_divergence(self):
        bus = AsyncEventBus()
        mgr = AlertManager(_make_notifier(), bus, {})
        mgr.subscribe_all()
        assert EventType.SHADOW_DIVERGENCE in bus._handlers

    def test_subscribe_all_wires_model_updated(self):
        bus = AsyncEventBus()
        mgr = AlertManager(_make_notifier(), bus, {})
        mgr.subscribe_all()
        assert EventType.MODEL_UPDATED in bus._handlers

    def test_subscribe_all_wires_provider_health(self):
        bus = AsyncEventBus()
        mgr = AlertManager(_make_notifier(), bus, {})
        mgr.subscribe_all()
        assert EventType.PROVIDER_HEALTH in bus._handlers

    def test_on_fill_calls_notifier(self):
        n = _make_notifier()
        n._tg_available = False
        mgr = _make_manager(n, rate_limit=0)
        fill = _make_fill()
        with patch.object(n, "notify_fill", new_callable=AsyncMock) as mock_nf:
            asyncio.get_event_loop().run_until_complete(mgr._on_fill(fill))
            mock_nf.assert_called_once_with(fill)

    def test_on_shadow_divergence_calls_send(self):
        n = _make_notifier()
        mgr = _make_manager(n, rate_limit=0)
        record = MagicMock()
        record.symbol = "ETH/USDT"
        record.slippage_pct = 0.03
        with patch.object(n, "send", new_callable=AsyncMock) as mock_send:
            asyncio.get_event_loop().run_until_complete(mgr._on_shadow_divergence(record))
            mock_send.assert_called_once()
            assert mock_send.call_args[0][0] == AlertLevel.INFO

    def test_on_model_updated_calls_send(self):
        n = _make_notifier()
        mgr = _make_manager(n, rate_limit=0)
        payload = {"models_updated": ["gbm", "lstm"], "bars_seen": 200}
        with patch.object(n, "send", new_callable=AsyncMock) as mock_send:
            asyncio.get_event_loop().run_until_complete(mgr._on_model_updated(payload))
            mock_send.assert_called_once()

    def test_on_provider_health_dispatches_on_circuit_open(self):
        n = _make_notifier()
        mgr = _make_manager(n, rate_limit=0)
        health = {"circuit_open": True, "consecutive_failures": 5, "circuit_open_since": "2026-05-01T00:00:00Z"}
        with patch.object(n, "send", new_callable=AsyncMock) as mock_send:
            asyncio.get_event_loop().run_until_complete(mgr._on_provider_health(health))
            mock_send.assert_called_once()
            assert mock_send.call_args[0][0] == AlertLevel.CRITICAL

    def test_on_provider_health_skips_when_circuit_closed(self):
        n = _make_notifier()
        mgr = _make_manager(n, rate_limit=0)
        health = {"circuit_open": False}
        with patch.object(n, "send", new_callable=AsyncMock) as mock_send:
            asyncio.get_event_loop().run_until_complete(mgr._on_provider_health(health))
            mock_send.assert_not_called()

    def test_rate_limit_suppresses_second_same_event(self):
        n = _make_notifier()
        mgr = _make_manager(n, rate_limit=60)  # 60 second window
        fill = _make_fill()
        with patch.object(n, "notify_fill", new_callable=AsyncMock) as mock_nf:
            asyncio.get_event_loop().run_until_complete(mgr._on_fill(fill))
            asyncio.get_event_loop().run_until_complete(mgr._on_fill(fill))
            # Only first call should go through
            mock_nf.assert_called_once()

    def test_rate_limit_allows_after_window_expires(self):
        n = _make_notifier()
        mgr = _make_manager(n, rate_limit=1)  # 1 second
        fill = _make_fill()
        # Pre-populate last_sent with a timestamp 2 seconds ago
        mgr._last_sent[str(EventType.FILL)] = (
            datetime.now(timezone.utc) - timedelta(seconds=2)
        )
        with patch.object(n, "notify_fill", new_callable=AsyncMock) as mock_nf:
            asyncio.get_event_loop().run_until_complete(mgr._on_fill(fill))
            mock_nf.assert_called_once()

    def test_dispatch_records_to_history(self):
        n = _make_notifier()
        mgr = _make_manager(n, rate_limit=0)
        fill = _make_fill()
        with patch.object(n, "notify_fill", new_callable=AsyncMock):
            asyncio.get_event_loop().run_until_complete(mgr._on_fill(fill))
        assert len(mgr._history) == 1
        assert mgr._history[0].event_type == str(EventType.FILL)


# ─────────────────────────────────────────────────────────────────────────────
# Group 4: TestGetRecentAlerts  (4 tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestGetRecentAlerts:
    def test_empty_initially(self):
        mgr = _make_manager()
        assert mgr.get_recent_alerts() == []

    def test_newest_first_order(self):
        mgr = _make_manager()
        now = datetime.now(timezone.utc)
        mgr._history.append(AlertRecord(AlertLevel.INFO, "first", now - timedelta(seconds=2), "fill"))
        mgr._history.append(AlertRecord(AlertLevel.INFO, "second", now - timedelta(seconds=1), "kill"))
        mgr._history.append(AlertRecord(AlertLevel.INFO, "third", now, "fill"))
        result = mgr.get_recent_alerts()
        assert result[0].message == "third"
        assert result[1].message == "second"
        assert result[2].message == "first"

    def test_limit_respected(self):
        mgr = _make_manager()
        now = datetime.now(timezone.utc)
        for i in range(20):
            mgr._history.append(
                AlertRecord(AlertLevel.INFO, f"msg{i}", now + timedelta(seconds=i), "fill")
            )
        result = mgr.get_recent_alerts(limit=5)
        assert len(result) == 5

    def test_history_size_cap_via_deque_maxlen(self):
        mgr = _make_manager(history_size=3)
        now = datetime.now(timezone.utc)
        for i in range(10):
            mgr._history.append(
                AlertRecord(AlertLevel.INFO, f"msg{i}", now + timedelta(seconds=i), "fill")
            )
        # deque maxlen caps at 3
        assert len(mgr._history) == 3
        # Most recent 3 kept
        messages = [r.message for r in mgr._history]
        assert "msg9" in messages


# ─────────────────────────────────────────────────────────────────────────────
# Group 5: TestDashboardAlerts  (4 tests)
# ─────────────────────────────────────────────────────────────────────────────

def _make_test_app(get_alerts=None) -> TestClient:
    app = FastAPI()
    router = build_router(
        get_portfolio=MagicMock(),
        get_trades=lambda: [],
        get_fills=lambda: [],
        exec_engine=None,
        risk_manager=MagicMock(),
        get_alerts=get_alerts,
    )
    app.include_router(router)
    return TestClient(app)


class TestDashboardAlerts:
    def test_api_alerts_returns_list(self):
        mgr = _make_manager()
        client = _make_test_app(get_alerts=mgr.get_recent_alerts)
        resp = client.get("/api/alerts")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_empty_when_get_alerts_is_none(self):
        client = _make_test_app(get_alerts=None)
        resp = client.get("/api/alerts")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_limit_query_param(self):
        mgr = _make_manager()
        now = datetime.now(timezone.utc)
        for i in range(20):
            mgr._history.append(
                AlertRecord(AlertLevel.INFO, f"msg{i}", now + timedelta(seconds=i), "fill")
            )
        client = _make_test_app(get_alerts=mgr.get_recent_alerts)
        resp = client.get("/api/alerts?limit=3")
        assert resp.status_code == 200
        assert len(resp.json()) == 3

    def test_structure_has_required_keys(self):
        mgr = _make_manager(rate_limit=0)
        now = datetime.now(timezone.utc)
        mgr._history.append(
            AlertRecord(AlertLevel.CRITICAL, "test message", now, "kill")
        )
        client = _make_test_app(get_alerts=mgr.get_recent_alerts)
        resp = client.get("/api/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        item = data[0]
        assert "level" in item
        assert "message" in item
        assert "ts" in item
        assert "event_type" in item
        assert item["level"] == "CRITICAL"
        assert item["message"] == "test message"
        assert item["event_type"] == "kill"


# ─────────────────────────────────────────────────────────────────────────────
# Group 6: TestBuildNotifierConfig  (3 tests — part of bonus coverage)
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildNotifierConfig:
    def test_reads_env_vars(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "456")
        monkeypatch.setenv("SMTP_PORT", "465")
        cfg = _build_notifier_config({"min_level": "CRITICAL"})
        assert cfg["telegram_token"] == "tok123"
        assert cfg["chat_id"] == "456"
        assert cfg["smtp_port"] == 465
        assert cfg["min_level"] == "CRITICAL"

    def test_defaults_when_env_absent(self, monkeypatch):
        for var in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "SMTP_HOST",
                    "SMTP_USER", "SMTP_PASS", "ALERT_EMAIL_TO"):
            monkeypatch.delenv(var, raising=False)
        cfg = _build_notifier_config({})
        assert cfg["telegram_token"] == ""
        assert cfg["min_level"] == "INFO"
        assert cfg["smtp_port"] == 587

    def test_notifier_constructed_from_result(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        cfg = _build_notifier_config({"min_level": "DEBUG"})
        n = Notifier(cfg)
        assert n.min_level == AlertLevel.DEBUG
        assert not n._tg_available  # no token
