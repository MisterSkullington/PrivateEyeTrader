"""
Dashboard API key authentication (Phase 13, audit fix C-4).

The dashboard exposes endpoints that can flatten all positions
(``POST /api/kill``) and disclose live equity / positions / trades. With the
default ``host: "127.0.0.1"`` only loopback connections are accepted, but the
configurable host (e.g. ``0.0.0.0`` for Docker port-forward) immediately
exposes those endpoints to anyone on the LAN.

This module provides a lightweight FastAPI dependency that requires a
``X-API-Key`` header matching the value of the ``DASHBOARD_API_KEY``
environment variable.

Auth is **disabled** when ``DASHBOARD_API_KEY`` is unset — this is a dev
convenience (so unit tests and local exploration don't break), but
``run_live`` refuses to start when ``dashboard.require_auth: true`` and
no key is configured.

Typical wiring::

    from fastapi import APIRouter, Depends
    from privateye.dashboard.auth import require_api_key

    router = APIRouter()
    @router.get("/api/portfolio", dependencies=[Depends(require_api_key)])
    async def portfolio(): ...
"""
from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, status

API_KEY_HEADER = "X-API-Key"


def _expected_api_key() -> str | None:
    """Return the configured API key, or ``None`` when auth is disabled."""
    key = os.getenv("DASHBOARD_API_KEY", "").strip()
    return key or None


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> None:
    """FastAPI dependency: 401 unless ``X-API-Key`` matches the env var.

    When ``DASHBOARD_API_KEY`` is not set, the dependency is a no-op so that
    paper-mode dev sessions and unit tests work without ceremony.

    Comparison uses :func:`hmac.compare_digest` to avoid timing leaks.
    """
    expected = _expected_api_key()
    if expected is None:
        return  # auth disabled

    if not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )


def auth_enabled() -> bool:
    """``True`` iff a non-empty ``DASHBOARD_API_KEY`` is configured."""
    return _expected_api_key() is not None
