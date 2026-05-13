"""
Free CoinGecko Global Markets macro provider.

Fetches top-level market statistics (BTC dominance, total market cap,
24h change, stablecoin ratio proxy) with an in-process TTL cache.

No API key required. Rate limit: ~10–50 calls/minute on the public endpoint.
Uses only the stdlib ``urllib.request`` — zero new dependencies.

Config keys (under phase1.macro_provider):
  cache_ttl_seconds (float, default 60.0)  — seconds before cache expires
  timeout_seconds   (float, default 10.0)  — HTTP read/connect timeout
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from privateye.utils.logging import get_logger

log = get_logger()

COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"

NEUTRAL_DEFAULTS: dict[str, float] = {
    "btc_dominance":           50.0,   # % of total market cap
    "total_mcap_usd":          0.0,
    "total_mcap_change_24h":   0.0,
    "stablecoin_ratio_approx": 0.10,   # ~10% historically
    "btc_mcap_usd":            0.0,
}


class MacroProvider:
    """Fetches CoinGecko global market data for macro context features.

    Caches the last successful response for ``cache_ttl_seconds``.
    On any network or parse error, returns ``NEUTRAL_DEFAULTS`` and logs a
    warning — never raises. This ensures ``FeatureEngineer`` always gets a
    complete dict even when the API is unreachable.

    Usage::

        provider = MacroProvider(cache_ttl_seconds=60)
        data = await provider.fetch_latest()
        # data = {"btc_dominance": 52.3, "total_mcap_usd": 2.4e12, ...}
    """

    def __init__(
        self,
        timeout: float = 10.0,
        cache_ttl_seconds: float = 60.0,
    ) -> None:
        self._timeout = float(timeout)
        self._ttl = float(cache_ttl_seconds)
        self._cache: dict[str, float] | None = None
        self._cache_ts: float = 0.0   # time.monotonic() of last successful fetch

    # ── Public API ────────────────────────────────────────────────────────────

    async def fetch_latest(self) -> dict[str, float]:
        """Return macro snapshot dict.

        Keys: btc_dominance, total_mcap_usd, total_mcap_change_24h,
              stablecoin_ratio_approx, btc_mcap_usd.

        Returns ``NEUTRAL_DEFAULTS`` on any error; logs a warning in that case.
        """
        now = time.monotonic()
        if self._cache is not None and (now - self._cache_ts) < self._ttl:
            return dict(self._cache)

        try:
            raw = await _fetch_url(COINGECKO_GLOBAL_URL, timeout=self._timeout)
            parsed = self._parse_response(raw)
            self._cache = parsed
            self._cache_ts = time.monotonic()
            log.debug(
                f"[MacroProvider] Fetched: BTC dom={parsed['btc_dominance']:.1f}% "
                f"mcap=${parsed['total_mcap_usd']/1e9:.0f}B"
            )
            return dict(parsed)
        except Exception as exc:
            log.warning(f"[MacroProvider] Fetch failed ({exc!r}) — returning neutral defaults")
            return dict(NEUTRAL_DEFAULTS)

    def _parse_response(self, raw: str) -> dict[str, float]:
        """Parse the CoinGecko /api/v3/global JSON payload.

        Gracefully handles missing keys by substituting neutral values.
        """
        data = json.loads(raw).get("data", {})

        btc_dominance = float(
            data.get("market_cap_percentage", {}).get("btc", NEUTRAL_DEFAULTS["btc_dominance"])
        )
        total_mcap_usd = float(
            data.get("total_market_cap", {}).get("usd", NEUTRAL_DEFAULTS["total_mcap_usd"])
        )
        total_mcap_change_24h = float(
            data.get(
                "market_cap_change_percentage_24h_usd",
                NEUTRAL_DEFAULTS["total_mcap_change_24h"],
            )
        )
        total_volume_usd = float(
            data.get("total_volume", {}).get("usd", 0.0)
        )

        btc_mcap_usd = total_mcap_usd * btc_dominance / 100.0

        # Stablecoin ratio proxy: volume/mcap ratio (heuristic).
        # When mcap is 0 (parse failure) fall back to the neutral default.
        if total_mcap_usd > 0:
            stablecoin_ratio_approx = min(1.0, max(0.0, total_volume_usd / total_mcap_usd))
        else:
            stablecoin_ratio_approx = NEUTRAL_DEFAULTS["stablecoin_ratio_approx"]

        return {
            "btc_dominance":           btc_dominance,
            "total_mcap_usd":          total_mcap_usd,
            "total_mcap_change_24h":   total_mcap_change_24h,
            "stablecoin_ratio_approx": stablecoin_ratio_approx,
            "btc_mcap_usd":            btc_mcap_usd,
        }


# ── Private helpers ───────────────────────────────────────────────────────────

async def _fetch_url(url: str, timeout: float) -> str:
    """Non-blocking HTTP GET via asyncio executor (keeps the event loop free)."""
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: _sync_fetch(url, timeout))


def _sync_fetch(url: str, timeout: float) -> str:
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "PrivateEyeTrader/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")
