"""
Fear & Greed Index provider via the alternative.me free API.

Endpoint: https://api.alternative.me/fng/?limit=N&format=json
No API key required. Daily resolution (one reading per day).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from privateye.utils.logging import get_logger

log = get_logger()

API_URL = "https://api.alternative.me/fng/"


class FearGreedProvider:
    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout

    async def fetch_latest(self) -> dict[str, Any]:
        """Return {timestamp, fear_greed, classification} for today."""
        data = await self._get(limit=1)
        if not data:
            return {"timestamp": datetime.now(timezone.utc), "fear_greed": 50.0, "classification": "Neutral"}
        row = data[0]
        return {
            "timestamp": datetime.fromtimestamp(int(row["timestamp"]), tz=timezone.utc),
            "fear_greed": float(row["value"]),
            "classification": row.get("value_classification", "Neutral"),
        }

    async def fetch_history(self, limit: int = 365) -> pd.DataFrame:
        """
        Fetch up to `limit` days of Fear & Greed history.

        Returns DataFrame with columns: timestamp (UTC), fear_greed, classification.
        """
        data = await self._get(limit=limit)
        if not data:
            return _empty_fg_df()

        records = []
        for row in data:
            try:
                records.append({
                    "timestamp": pd.Timestamp(int(row["timestamp"]), unit="s", tz="UTC"),
                    "fear_greed": float(row["value"]),
                    "classification": str(row.get("value_classification", "Neutral")),
                })
            except (KeyError, ValueError):
                continue

        if not records:
            return _empty_fg_df()

        df = pd.DataFrame(records)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        log.info(f"[FearGreedProvider] Fetched {len(df)} Fear & Greed records")
        return df

    async def _get(self, limit: int) -> list[dict]:
        """Raw HTTP call — returns list of API rows."""
        loop = asyncio.get_event_loop()
        url = f"{API_URL}?limit={limit}&format=json"
        try:
            import urllib.request
            resp = await loop.run_in_executor(None, lambda: _http_get(url, self.timeout))
            import json
            parsed = json.loads(resp)
            return parsed.get("data", [])
        except Exception as e:
            log.warning(f"[FearGreedProvider] HTTP error: {e}")
            return []


def _http_get(url: str, timeout: float) -> str:
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8")


def _empty_fg_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "fear_greed", "classification"])
