"""
Binance perpetual funding rate + open interest provider.

Uses CCXT futures mode. Returns DataFrames suitable for pd.merge_asof
joining onto hourly OHLCV bars before backtesting.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from privateye.utils.logging import get_logger

log = get_logger()


class FundingRateProvider:
    def __init__(
        self,
        symbol: str = "BTC/USDT",
        api_key: str = "",
        api_secret: str = "",
        sandbox: bool = False,
    ) -> None:
        self.symbol = symbol
        self._api_key = api_key
        self._api_secret = api_secret
        self._sandbox = sandbox
        self._exchange: Any = None

    def _get_exchange(self) -> Any:
        if self._exchange is None:
            import ccxt
            self._exchange = ccxt.binance({
                "apiKey": self._api_key,
                "secret": self._api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            })
            if self._sandbox:
                self._exchange.set_sandbox_mode(True)
        return self._exchange

    async def fetch_latest(self) -> dict[str, Any]:
        """Return {timestamp, funding_rate, open_interest} for the current bar."""
        loop = asyncio.get_event_loop()
        ex = self._get_exchange()
        try:
            fr_data = await loop.run_in_executor(
                None, lambda: ex.fetch_funding_rate(self.symbol)
            )
            oi_data = await loop.run_in_executor(
                None, lambda: ex.fetch_open_interest(self.symbol)
            )
            return {
                "timestamp": datetime.now(timezone.utc),
                "funding_rate": float(fr_data.get("fundingRate", 0.0) or 0.0),
                "open_interest": float(oi_data.get("openInterestAmount", 0.0) or 0.0),
            }
        except Exception as e:
            log.warning(f"[FundingRateProvider] fetch_latest error: {e}")
            return {
                "timestamp": datetime.now(timezone.utc),
                "funding_rate": 0.0,
                "open_interest": 0.0,
            }

    async def fetch_history(
        self, since_ms: int | None = None, limit: int = 500
    ) -> pd.DataFrame:
        """
        Fetch historical funding rates.

        Returns DataFrame with columns: timestamp (UTC), funding_rate, open_interest.
        """
        loop = asyncio.get_event_loop()
        ex = self._get_exchange()
        try:
            kwargs: dict[str, Any] = {"symbol": self.symbol, "limit": limit}
            if since_ms is not None:
                kwargs["since"] = since_ms

            rows = await loop.run_in_executor(
                None, lambda: ex.fetch_funding_rate_history(**kwargs)
            )
            if not rows:
                return _empty_funding_df()

            records = []
            for r in rows:
                records.append({
                    "timestamp": pd.Timestamp(r["timestamp"], unit="ms", tz="UTC"),
                    "funding_rate": float(r.get("fundingRate", 0.0) or 0.0),
                    "open_interest": 0.0,  # not included in history endpoint
                })

            df = pd.DataFrame(records)
            df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
            log.info(f"[FundingRateProvider] Fetched {len(df)} funding rate records for {self.symbol}")
            return df

        except Exception as e:
            log.warning(f"[FundingRateProvider] fetch_history error: {e}")
            return _empty_funding_df()


def _empty_funding_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "funding_rate", "open_interest"])
