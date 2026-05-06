"""
Alert Manager — subscribes to the event bus and dispatches contextual,
rate-limited notifications via the Notifier.

Every alert-worthy event (fill, risk breach, model update, etc.) is recorded
in an in-memory deque regardless of rate limiting, so the dashboard can surface
recent alert history via ``GET /api/alerts``.

Usage::

    from privateye.alerts.alert_manager import AlertManager, _build_notifier_config
    from privateye.alerts.notifier import Notifier

    notifier = Notifier(_build_notifier_config(cfg.get("alerts", {})))
    manager  = AlertManager(notifier, bus, cfg.get("alerts", {}))
    manager.subscribe_all()
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from privateye.core.types import AlertLevel, EventType
from privateye.utils.logging import get_logger

log = get_logger()


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlertRecord:
    level:      AlertLevel
    message:    str
    timestamp:  datetime
    event_type: str   # EventType value string, e.g. "fill"


# ─────────────────────────────────────────────────────────────────────────────
# Config helper
# ─────────────────────────────────────────────────────────────────────────────

def _build_notifier_config(alerts_cfg: dict) -> dict:
    """
    Merge a ``alerts:`` settings dict with .env credentials.

    Reads:
        TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
        SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, ALERT_EMAIL_TO

    Returns a dict shaped for ``Notifier.__init__()``.
    All missing env vars default to empty strings (Notifier degrades gracefully).
    """
    return {
        "min_level":      alerts_cfg.get("min_level", "INFO"),
        "telegram_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
        "chat_id":        os.getenv("TELEGRAM_CHAT_ID", ""),
        "smtp_host":      os.getenv("SMTP_HOST", ""),
        "smtp_port":      int(os.getenv("SMTP_PORT", "587")),
        "smtp_user":      os.getenv("SMTP_USER", ""),
        "smtp_pass":      os.getenv("SMTP_PASS", ""),
        "email_to":       os.getenv("ALERT_EMAIL_TO", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# AlertManager
# ─────────────────────────────────────────────────────────────────────────────

class AlertManager:
    """
    Event-bus subscriber that converts trading events into rate-limited alerts.

    Responsibilities:
    * Subscribe to all alert-relevant ``EventType`` values.
    * Format human-readable messages from event payloads.
    * Rate-limit repetitive alerts of the same event type.
    * Record every event (including suppressed ones) in a bounded history deque.
    * Expose ``get_recent_alerts()`` for the dashboard ``/api/alerts`` endpoint.

    Parameters
    ----------
    notifier : Notifier
        Pre-constructed notifier instance (Telegram + email).
    bus      : AsyncEventBus
        The application event bus.
    config   : dict
        ``alerts:`` section from ``settings.yaml``.  Supported keys:
        ``rate_limit_seconds`` (int, default 60),
        ``history_size``       (int, default 100).
    """

    def __init__(
        self,
        notifier: Any,
        bus: Any,
        config: dict,
    ) -> None:
        self._notifier = notifier
        self._bus = bus
        self._rate_limit: int = int(config.get("rate_limit_seconds", 60))
        self._history: deque[AlertRecord] = deque(
            maxlen=int(config.get("history_size", 100))
        )
        self._last_sent: dict[str, datetime] = {}

    # ── Wiring ────────────────────────────────────────────────────────────────

    def subscribe_all(self) -> None:
        """Subscribe handlers to all alert-relevant event types."""
        self._bus.subscribe(EventType.FILL,              self._on_fill)
        self._bus.subscribe(EventType.SHADOW_DIVERGENCE, self._on_shadow_divergence)
        self._bus.subscribe(EventType.MODEL_UPDATED,     self._on_model_updated)
        self._bus.subscribe(EventType.PROVIDER_HEALTH,   self._on_provider_health)
        self._bus.subscribe(EventType.RISK_BREACH,       self._on_risk_breach)
        self._bus.subscribe(EventType.KILL,              self._on_kill)
        log.info(
            "[AlertManager] Subscribed to 6 event types "
            f"(rate_limit={self._rate_limit}s, history={self._history.maxlen})"
        )

    # ── Event handlers ────────────────────────────────────────────────────────

    async def _on_fill(self, fill: Any) -> None:
        msg = (
            f"Fill: {fill.side.value.upper()} {fill.quantity:.6f} {fill.symbol}"
            f" @ {fill.price:.2f} | pnl={fill.realised_pnl:+.2f}"
        )
        await self._dispatch(
            EventType.FILL,
            AlertLevel.INFO,
            msg,
            lambda: self._notifier.notify_fill(fill),
        )

    async def _on_shadow_divergence(self, record: Any) -> None:
        msg = (
            f"Shadow divergence: {record.symbol} "
            f"slippage={record.slippage_pct:.2%}"
        )
        await self._dispatch(
            EventType.SHADOW_DIVERGENCE,
            AlertLevel.INFO,
            msg,
            lambda: self._notifier.send(AlertLevel.INFO, msg),
        )

    async def _on_model_updated(self, payload: dict) -> None:
        updated = payload.get("models_updated", [])
        msg = (
            f"Models updated: {', '.join(updated)} "
            f"({payload.get('bars_seen', 0)} bars)"
        )
        await self._dispatch(
            EventType.MODEL_UPDATED,
            AlertLevel.INFO,
            msg,
            lambda: self._notifier.send(AlertLevel.INFO, msg),
        )

    async def _on_provider_health(self, health: dict) -> None:
        if not health.get("circuit_open", False):
            return   # only alert when circuit trips
        msg = (
            f"Provider circuit open! "
            f"failures={health.get('consecutive_failures', '?')}, "
            f"since={health.get('circuit_open_since', 'unknown')}"
        )
        await self._dispatch(
            EventType.PROVIDER_HEALTH,
            AlertLevel.CRITICAL,
            msg,
            lambda: self._notifier.send(AlertLevel.CRITICAL, msg),
        )

    async def _on_risk_breach(self, payload: Any) -> None:
        reason = (
            payload.get("reason", str(payload))
            if isinstance(payload, dict)
            else str(payload)
        )
        await self._dispatch(
            EventType.RISK_BREACH,
            AlertLevel.CRITICAL,
            reason,
            lambda: self._notifier.notify_risk_breach(reason),
        )

    async def _on_kill(self, payload: Any) -> None:
        msg = "Kill switch activated — all positions closed"
        await self._dispatch(
            EventType.KILL,
            AlertLevel.CRITICAL,
            msg,
            lambda: self._notifier.notify_kill_switch(),
        )

    # ── Dispatch ──────────────────────────────────────────────────────────────

    async def _dispatch(
        self,
        event_type: EventType,
        level: AlertLevel,
        message: str,
        notify_factory: Any,
    ) -> None:
        """
        Rate-limited dispatch.

        The alert record is always appended to ``_history`` (so suppressed events
        are still visible in the dashboard).  ``notify_factory`` is a zero-argument
        callable that returns a coroutine — it is only called (and awaited) when
        the same event type has not been dispatched within ``rate_limit_seconds``.
        Lazy evaluation prevents spurious calls to the notifier when rate-limited.
        """
        key = str(event_type)
        now = datetime.now(timezone.utc)
        self._history.append(
            AlertRecord(level=level, message=message, timestamp=now, event_type=key)
        )
        last = self._last_sent.get(key)
        if last and (now - last).total_seconds() < self._rate_limit:
            return   # rate-limited — recorded in history but notification suppressed
        self._last_sent[key] = now
        await notify_factory()

    # ── History access ────────────────────────────────────────────────────────

    def get_recent_alerts(self, limit: int = 50) -> list[AlertRecord]:
        """Return up to *limit* most recent alerts, newest first."""
        items = list(self._history)
        return list(reversed(items))[:limit]
