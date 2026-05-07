"""
Bybit OHLCV provider (live/paper mode).

Mirrors BinanceProvider exactly — same interface, same self-healing pattern.
Uses ccxt.bybit for data fetching.

Self-healing:
  - Exponential backoff retry: 2^n seconds (max 5 attempts)
  - Circuit breaker: opens after 10 consecutive failures
  - Auto-recovery (Phase 6): probes after auto_recovery_seconds of open circuit
  - Health metrics (Phase 6): get_health() returns success_rate, latency, circuit state
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

import ccxt
import pandas as pd

from privateye.utils.logging import get_logger
from privateye.utils.time import now_utc

log = get_logger()

_MAX_RETRIES = 5
_CIRCUIT_BREAKER_THRESHOLD = 10


class BybitProvider:
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        sandbox: bool = True,
        poll_interval: float = 60.0,
        bar_limit: int = 500,
        auto_recovery_seconds: float = 1800.0,
    ) -> None:
        self.poll_interval = poll_interval
        self.bar_limit = bar_limit
        self._exchange = ccxt.bybit({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
        })
        if sandbox:
            self._exchange.set_sandbox_mode(True)
        self._bars: dict[tuple[str, str], pd.DataFrame] = {}
        self._running = False
        self._consecutive_failures: int = 0
        self._circuit_open: bool = False

        # Phase 6: auto-recovery + health tracking
        self._circuit_open_since: datetime | None = None
        self._auto_recovery_seconds: float = auto_recovery_seconds
        self._success_count: int = 0
        self._failure_count: int = 0
        self._last_fetch_latency_ms: float = 0.0
        self._recovery_task: asyncio.Task | None = None
        self._symbols: list[str] = []
        self._timeframes: list[str] = []

    async def start(
        self,
        symbols: list[str],
        timeframes: list[str],
        on_bar: Any,
    ) -> None:
        self._running = True
        self._symbols = list(symbols)
        self._timeframes = list(timeframes)
        log.info(f"BybitProvider starting: {symbols} {timeframes}")

        self._recovery_task = asyncio.create_task(self._auto_recovery_loop())

        for symbol in symbols:
            for tf in timeframes:
                await self._fetch_and_update(symbol, tf)
        while self._running:
            for symbol in symbols:
                for tf in timeframes:
                    bars = await self._fetch_and_update(symbol, tf)
                    if bars is not None:
                        await on_bar(symbol, tf, bars)
            await asyncio.sleep(self.poll_interval)

    def stop(self) -> None:
        self._running = False
        if self._recovery_task is not None:
            self._recovery_task.cancel()
            self._recovery_task = None
        log.info("BybitProvider stopped")

    async def _fetch_and_update(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        if self._circuit_open:
            log.warning(
                f"[BybitProvider] Circuit breaker OPEN after {_CIRCUIT_BREAKER_THRESHOLD} "
                "consecutive failures — skipping fetch"
            )
            return None

        t0 = time.monotonic()

        for attempt in range(_MAX_RETRIES):
            try:
                raw = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._exchange.fetch_ohlcv(symbol, timeframe, limit=self.bar_limit),
                )
                if not raw:
                    return None
                df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.astype({
                    "open": float, "high": float, "low": float, "close": float, "volume": float
                })
                df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
                df = df.iloc[:-1].reset_index(drop=True)  # drop incomplete current bar
                self._bars[(symbol, timeframe)] = df
                self._consecutive_failures = 0
                self._success_count += 1
                self._last_fetch_latency_ms = (time.monotonic() - t0) * 1000
                log.debug(
                    f"[BybitProvider] Fetched {symbol} {timeframe}: "
                    f"{len(df)} closed bars in {self._last_fetch_latency_ms:.0f}ms"
                )
                return df

            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
                wait = 2 ** attempt
                if attempt < _MAX_RETRIES - 1:
                    log.warning(
                        f"[BybitProvider] Transient error [{symbol} {timeframe}] "
                        f"(attempt {attempt+1}/{_MAX_RETRIES}): {e} — retrying in {wait}s"
                    )
                    await asyncio.sleep(wait)
                else:
                    log.error(
                        f"[BybitProvider] All {_MAX_RETRIES} retries exhausted "
                        f"[{symbol} {timeframe}]: {e}"
                    )
                    self._failure_count += 1
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
                        self._trip_circuit()
                    return None

            except Exception as e:
                log.warning(f"[BybitProvider] Non-retryable error [{symbol} {timeframe}]: {e}")
                self._failure_count += 1
                self._consecutive_failures += 1
                if self._consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
                    self._trip_circuit()
                return None

        return None

    def _trip_circuit(self) -> None:
        self._circuit_open = True
        self._circuit_open_since = datetime.now(timezone.utc)
        log.critical(
            f"[BybitProvider] Circuit breaker TRIPPED after "
            f"{self._consecutive_failures} consecutive failures"
        )

    async def _auto_recovery_loop(self) -> None:
        while self._running:
            await asyncio.sleep(60)
            if not self._circuit_open or self._circuit_open_since is None:
                continue
            elapsed = (datetime.now(timezone.utc) - self._circuit_open_since).total_seconds()
            if elapsed < self._auto_recovery_seconds:
                continue
            probe_symbol = self._symbols[0] if self._symbols else None
            probe_tf = self._timeframes[0] if self._timeframes else None
            if probe_symbol is None or probe_tf is None:
                continue
            log.info(
                f"[BybitProvider] Auto-recovery probe after {elapsed:.0f}s "
                f"— testing {probe_symbol} {probe_tf}"
            )
            try:
                self._circuit_open = False
                result = await self._fetch_and_update(probe_symbol, probe_tf)
                if result is not None:
                    self.reset_circuit_breaker()
                    log.info("[BybitProvider] Auto-recovery succeeded — circuit reset")
                else:
                    self._circuit_open = True
            except Exception:
                self._circuit_open = True
                self._circuit_open_since = datetime.now(timezone.utc)
                log.warning("[BybitProvider] Auto-recovery probe failed — circuit remains open")

    def reset_circuit_breaker(self) -> None:
        self._circuit_open = False
        self._circuit_open_since = None
        self._consecutive_failures = 0
        log.info("[BybitProvider] Circuit breaker reset")

    def get_health(self) -> dict[str, Any]:
        """Return a snapshot of provider health metrics."""
        total = self._success_count + self._failure_count
        return {
            "circuit_open": self._circuit_open,
            "consecutive_failures": self._consecutive_failures,
            "circuit_open_since": (
                self._circuit_open_since.isoformat()
                if self._circuit_open_since else None
            ),
            "success_rate": self._success_count / max(1, total),
            "total_fetches": total,
            "last_fetch_latency_ms": round(self._last_fetch_latency_ms, 2),
        }

    def get_bars(self, symbol: str, timeframe: str) -> pd.DataFrame:
        return self._bars.get((symbol, timeframe), pd.DataFrame())

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._exchange.fetch_ticker(symbol)
            )
        except Exception as e:
            log.warning(f"fetch_ticker error [{symbol}]: {e}")
            return {}

    async def fetch_balance(self) -> dict[str, Any]:
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._exchange.fetch_balance
            )
        except Exception as e:
            log.warning(f"fetch_balance error: {e}")
            return {}
