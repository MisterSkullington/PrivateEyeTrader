"""Dashboard REST + WebSocket routes."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Callable

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from privateye.utils.logging import get_logger

log = get_logger()
TEMPLATES_DIR = Path(__file__).parent / "templates"


def build_router(
    get_portfolio: Callable,
    get_trades: Callable,
    get_fills: Callable,
    exec_engine: Any,
    risk_manager: Any,
    initial_capital: float = 10000.0,
    get_health: Callable | None = None,
    get_alerts: Callable | None = None,
    get_compliance: Callable | None = None,
) -> APIRouter:
    router = APIRouter()
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    _ws_clients: list[WebSocket] = []

    # ── Pages ───────────────────────────────────────────────────────────────

    @router.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    # ── REST API ────────────────────────────────────────────────────────────

    @router.get("/api/portfolio")
    async def api_portfolio():
        p = get_portfolio()
        return {
            "equity": p.equity,
            "cash": p.cash,
            "invested": p.invested,
            "daily_pnl": p.daily_pnl,
            "daily_drawdown_pct": p.daily_drawdown_pct,
            "drawdown_pct": p.drawdown_pct,
            "peak_equity": p.peak_equity,
            "total_trades": p.total_trades,
            "timestamp": datetime.utcnow().isoformat(),
            "halted": risk_manager.is_halted(),
        }

    @router.get("/api/positions")
    async def api_positions():
        p = get_portfolio()
        return [
            {
                "symbol": pos.symbol,
                "side": pos.side.value,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "stop_price": pos.stop_price,
                "target_price": pos.target_price,
                "trailing_stop": pos.trailing_stop,
                "bars_held": pos.bars_held,
                "strategy_id": pos.strategy_id,
            }
            for pos in p.positions.values()
        ]

    @router.get("/api/trades")
    async def api_trades(limit: int = 100):
        trades = get_trades()
        recent = trades[-limit:] if len(trades) > limit else trades
        return [
            {
                "symbol": t.symbol,
                "side": t.side.value,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "fees": t.fees,
                "bars_held": t.bars_held,
                "exit_reason": t.exit_reason,
                "strategy_id": t.strategy_id,
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
            }
            for t in recent
        ]

    @router.get("/api/metrics")
    async def api_metrics():
        trades = get_trades()
        if not trades:
            return {"message": "No completed trades yet"}
        from privateye.backtesting.metrics import compute_metrics
        p = get_portfolio()
        report = compute_metrics(trades, [p.equity], p.equity - p.daily_pnl)
        return {
            "total_trades": report.total_trades,
            "win_rate": report.win_rate,
            "profit_factor": report.profit_factor,
            "total_pnl": report.total_pnl,
            "max_drawdown_pct": report.max_drawdown_pct,
            "sharpe_ratio": report.sharpe_ratio,
            "avg_win": report.avg_win,
            "avg_loss": report.avg_loss,
        }

    @router.get("/api/signals")
    async def api_signals():
        fills = get_fills()
        recent = fills[-50:] if len(fills) > 50 else fills
        return [
            {
                "symbol": f.symbol,
                "side": f.side.value,
                "quantity": f.quantity,
                "price": f.price,
                "fee": f.fee,
                "realised_pnl": f.realised_pnl,
                "strategy_id": f.strategy_id,
                "timestamp": f.timestamp.isoformat(),
            }
            for f in recent
        ]

    @router.get("/api/status")
    async def api_status():
        return {
            "halted": risk_manager.is_halted(),
            "halt_reason": risk_manager._halt_reason if risk_manager.is_halted() else None,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ── Equity / drawdown curves ────────────────────────────────────────────

    @router.get("/api/equity-curve")
    async def api_equity_curve():
        """
        Cumulative equity curve computed from completed trades.

        Returns a list of ``{ts, equity}`` dicts where *equity* is the
        running portfolio value starting from *initial_capital*.
        Returns a single baseline point when no trades are recorded yet.
        """
        trades = get_trades()
        if not trades:
            return [{"ts": datetime.utcnow().isoformat(), "equity": initial_capital}]

        equity = initial_capital
        curve = []
        for t in trades:
            equity += t.pnl
            curve.append({
                "ts": t.exit_time.isoformat(),
                "equity": round(equity, 4),
            })
        return curve

    @router.get("/api/drawdown-curve")
    async def api_drawdown_curve():
        """
        Drawdown curve relative to the running equity peak.

        Returns a list of ``{ts, drawdown_pct}`` dicts where *drawdown_pct*
        is 0.0 at equity peaks and negative in troughs (e.g. -0.05 = −5%).
        """
        trades = get_trades()
        if not trades:
            return [{"ts": datetime.utcnow().isoformat(), "drawdown_pct": 0.0}]

        equity = initial_capital
        peak   = initial_capital
        curve  = []
        for t in trades:
            equity += t.pnl
            if equity > peak:
                peak = equity
            dd = (equity - peak) / peak if peak > 0 else 0.0
            curve.append({
                "ts": t.exit_time.isoformat(),
                "drawdown_pct": round(dd, 6),
            })
        return curve

    # ── Alerts history ──────────────────────────────────────────────────────

    @router.get("/api/alerts")
    async def api_alerts(limit: int = 50):
        """
        Recent alert history, newest first.

        Returns an empty list when ``AlertManager`` is not wired (i.e. when
        ``get_alerts`` was not supplied to ``build_router``).
        """
        if get_alerts is None:
            return []
        alerts = get_alerts(limit)
        return [
            {
                "level":      a.level.value,
                "message":    a.message,
                "ts":         a.timestamp.isoformat(),
                "event_type": a.event_type,
            }
            for a in alerts
        ]

    # ── Compliance status ───────────────────────────────────────────────────

    @router.get("/api/compliance")
    async def api_compliance():
        """
        Current compliance status: jurisdiction, sanctions setting, blocked
        symbols seen this session, and wash-sale flags (populated after a
        post-trade analysis run).

        Returns a minimal empty structure when ``ComplianceEngine`` is not
        wired (i.e. ``get_compliance`` was not supplied to ``build_router``).
        """
        if get_compliance is None:
            return {
                "jurisdiction":          None,
                "sanctions_enabled":     False,
                "blocked_symbols":       [],
                "wash_sale_flags":       [],
                "total_disallowed_loss": 0.0,
            }
        return get_compliance()

    # ── Kill switch ─────────────────────────────────────────────────────────

    @router.post("/api/kill")
    async def api_kill():
        log.critical("[Dashboard] Kill switch activated via API")
        p = get_portfolio()
        prices = {sym: pos.entry_price for sym, pos in p.positions.items()}
        await exec_engine.flatten_all(prices, reason="manual_kill_switch")
        risk_manager.halt("kill switch activated from dashboard")
        await _broadcast({"event": "kill", "message": "Kill switch activated — all positions closed"})
        return {"status": "kill_switch_activated", "positions_closed": len(p.positions)}

    @router.post("/api/resume")
    async def api_resume():
        exec_engine.resume()
        risk_manager.resume()
        return {"status": "resumed"}

    # ── WebSocket ────────────────────────────────────────────────────────────

    @router.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        _ws_clients.append(ws)
        log.debug(f"WebSocket client connected ({len(_ws_clients)} total)")
        try:
            while True:
                p = get_portfolio()
                trades = get_trades()
                data = {
                    "event": "update",
                    "equity": p.equity,
                    "cash": p.cash,
                    "daily_pnl": p.daily_pnl,
                    "drawdown_pct": p.daily_drawdown_pct,
                    "halted": risk_manager.is_halted(),
                    "open_positions": len(p.positions),
                    "total_trades": len(trades),
                    "timestamp": datetime.utcnow().isoformat(),
                }
                await ws.send_json(data)
                await asyncio.sleep(5)
        except (WebSocketDisconnect, Exception):
            _ws_clients.remove(ws)
            log.debug(f"WebSocket client disconnected ({len(_ws_clients)} remaining)")

    async def _broadcast(data: dict) -> None:
        for client in list(_ws_clients):
            try:
                await client.send_json(data)
            except Exception:
                _ws_clients.remove(client)

    return router
