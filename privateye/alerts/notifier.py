"""
Alert notifier — dispatches contextual Telegram and email alerts.

Importance levels (lowest to highest):
  DEBUG    — suppressed by default
  INFO     — daily summary, fills
  CRITICAL — risk breaches, kill switch, exchange errors

Only alerts at or above `min_level` are sent.
Each channel (Telegram, email) degrades gracefully if not configured.
"""
from __future__ import annotations

import asyncio
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from typing import Any

from privateye.core.types import AlertLevel, Fill, TradeRecord
from privateye.utils.logging import get_logger

log = get_logger()

_LEVEL_ORDER = {AlertLevel.DEBUG: 0, AlertLevel.INFO: 1, AlertLevel.CRITICAL: 2}


class Notifier:
    def __init__(self, config: dict[str, Any]) -> None:
        self.min_level = AlertLevel(config.get("min_level", "INFO"))
        self._tg_token: str = config.get("telegram_token", "")
        self._tg_chat: str = config.get("chat_id", "")
        self._smtp_host: str = config.get("smtp_host", "")
        self._smtp_port: int = config.get("smtp_port", 587)
        self._smtp_user: str = config.get("smtp_user", "")
        self._smtp_pass: str = config.get("smtp_pass", "")
        self._email_to: str = config.get("email_to", "")
        self._tg_available = bool(self._tg_token and self._tg_chat)
        self._email_available = bool(self._smtp_host and self._smtp_user and self._email_to)

    # ── Public API ──────────────────────────────────────────────────────────

    async def send(self, level: AlertLevel, message: str) -> None:
        if _LEVEL_ORDER[level] < _LEVEL_ORDER[self.min_level]:
            return
        prefix = {AlertLevel.DEBUG: "🔍", AlertLevel.INFO: "ℹ️", AlertLevel.CRITICAL: "🚨"}[level]
        formatted = f"{prefix} [{level.value}] {datetime.utcnow().strftime('%H:%M:%S UTC')}\n{message}"
        log.info(f"[Alert/{level.value}] {message}")
        tasks = []
        if self._tg_available:
            tasks.append(self._send_telegram(formatted))
        if level == AlertLevel.CRITICAL and self._email_available:
            tasks.append(self._send_email(f"[PrivateEyeTrader CRITICAL] {message[:80]}", formatted))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def notify_fill(self, fill: Fill) -> None:
        await self.send(
            AlertLevel.INFO,
            f"Fill: {fill.side.value.upper()} {fill.quantity:.6f} {fill.symbol} "
            f"@ {fill.price:.2f} | fee={fill.fee:.4f} | pnl={fill.realised_pnl:+.2f}",
        )

    async def notify_risk_breach(self, reason: str) -> None:
        await self.send(AlertLevel.CRITICAL, f"RISK BREACH — {reason}")

    async def notify_kill_switch(self) -> None:
        await self.send(AlertLevel.CRITICAL, "KILL SWITCH ACTIVATED — all positions closed, trading halted")

    async def notify_daily_summary(self, portfolio: Any, trades: list[TradeRecord]) -> None:
        today_trades = [t for t in trades if t.exit_time.date() == datetime.utcnow().date()]
        total_pnl = sum(t.pnl for t in today_trades)
        msg = (
            f"Daily Summary\n"
            f"Equity: ${portfolio.equity:.2f}\n"
            f"Daily PnL: {total_pnl:+.2f}\n"
            f"Trades today: {len(today_trades)}\n"
            f"Drawdown: {portfolio.daily_drawdown_pct * 100:.2f}%"
        )
        await self.send(AlertLevel.INFO, msg)

    async def notify_error(self, component: str, error: str) -> None:
        await self.send(AlertLevel.CRITICAL, f"ERROR in {component}: {error}")

    # ── Telegram ────────────────────────────────────────────────────────────

    async def _send_telegram(self, text: str) -> None:
        try:
            import aiohttp
            url = f"https://api.telegram.org/bot{self._tg_token}/sendMessage"
            payload = {"chat_id": self._tg_chat, "text": text[:4096], "parse_mode": ""}
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        log.warning(f"Telegram send failed: {resp.status}")
        except Exception as e:
            log.warning(f"Telegram error: {e}")

    # ── Email ───────────────────────────────────────────────────────────────

    async def _send_email(self, subject: str, body: str) -> None:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self._send_email_sync(subject, body))
        except Exception as e:
            log.warning(f"Email error: {e}")

    def _send_email_sync(self, subject: str, body: str) -> None:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = self._smtp_user
        msg["To"] = self._email_to
        with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=10) as server:
            server.starttls()
            server.login(self._smtp_user, self._smtp_pass)
            server.sendmail(self._smtp_user, [self._email_to], msg.as_string())
