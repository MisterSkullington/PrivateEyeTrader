"""FastAPI dashboard server with WebSocket real-time push."""
from __future__ import annotations

import asyncio
from typing import Any, Callable

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from privateye.dashboard.routes import build_router
from privateye.utils.logging import get_logger

log = get_logger()

_app: FastAPI | None = None


def create_app(
    get_portfolio: Callable,
    get_trades: Callable,
    get_fills: Callable,
    exec_engine: Any,
    risk_manager: Any,
    initial_capital: float = 10000.0,
    get_health: Callable | None = None,
    get_alerts: Callable | None = None,
    get_compliance: Callable | None = None,
) -> FastAPI:
    global _app
    app = FastAPI(title="PrivateEyeTrader Dashboard", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    router = build_router(
        get_portfolio, get_trades, get_fills, exec_engine, risk_manager,
        initial_capital=initial_capital,
        get_health=get_health,
        get_alerts=get_alerts,
        get_compliance=get_compliance,
    )
    app.include_router(router)
    _app = app
    return app


async def start_dashboard(
    host: str,
    port: int,
    get_portfolio: Callable,
    get_trades: Callable,
    get_fills: Callable,
    exec_engine: Any,
    risk_manager: Any,
    initial_capital: float = 10000.0,
    get_health: Callable | None = None,
    get_alerts: Callable | None = None,
    get_compliance: Callable | None = None,
) -> None:
    app = create_app(
        get_portfolio, get_trades, get_fills, exec_engine, risk_manager,
        initial_capital=initial_capital,
        get_health=get_health,
        get_alerts=get_alerts,
        get_compliance=get_compliance,
    )
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", loop="asyncio")
    server = uvicorn.Server(config)
    log.info(f"Dashboard starting at http://{host}:{port}")
    await server.serve()
