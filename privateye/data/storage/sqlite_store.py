"""SQLite-backed time-series store for OHLCV bars and trade records."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite
import pandas as pd

from privateye.utils.logging import get_logger

log = get_logger()

CREATE_OHLCV = """
CREATE TABLE IF NOT EXISTS ohlcv (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol    TEXT    NOT NULL,
    timeframe TEXT    NOT NULL,
    timestamp INTEGER NOT NULL,
    open      REAL    NOT NULL,
    high      REAL    NOT NULL,
    low       REAL    NOT NULL,
    close     REAL    NOT NULL,
    volume    REAL    NOT NULL,
    UNIQUE(symbol, timeframe, timestamp)
)
"""

CREATE_TRADES = """
CREATE TABLE IF NOT EXISTS trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT    NOT NULL,
    side         TEXT    NOT NULL,
    entry_price  REAL    NOT NULL,
    exit_price   REAL    NOT NULL,
    quantity     REAL    NOT NULL,
    entry_time   INTEGER NOT NULL,
    exit_time    INTEGER NOT NULL,
    pnl          REAL    NOT NULL,
    pnl_pct      REAL    NOT NULL,
    fees         REAL    NOT NULL,
    strategy_id  TEXT    NOT NULL,
    exit_reason  TEXT    NOT NULL,
    bars_held    INTEGER NOT NULL
)
"""

CREATE_IDX_OHLCV = "CREATE INDEX IF NOT EXISTS idx_ohlcv ON ohlcv(symbol, timeframe, timestamp)"


class SQLiteStore:
    def __init__(self, db_path: str | Path = "data/privateye.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: aiosqlite.Connection | None = None

    async def open(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute(CREATE_OHLCV)
        await self._db.execute(CREATE_TRADES)
        await self._db.execute(CREATE_IDX_OHLCV)
        await self._db.commit()
        log.debug(f"SQLiteStore opened: {self.db_path}")

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def upsert_ohlcv(self, symbol: str, timeframe: str, bars: list[dict[str, Any]]) -> int:
        """Insert or ignore bars. Returns count of new rows inserted."""
        if not bars:
            return 0
        rows = [
            (symbol, timeframe, int(b["timestamp"].timestamp() * 1000) if isinstance(b["timestamp"], datetime)
             else int(b["timestamp"]),
             b["open"], b["high"], b["low"], b["close"], b["volume"])
            for b in bars
        ]
        await self._db.executemany(
            "INSERT OR IGNORE INTO ohlcv(symbol,timeframe,timestamp,open,high,low,close,volume) "
            "VALUES(?,?,?,?,?,?,?,?)",
            rows,
        )
        await self._db.commit()
        return len(rows)

    async def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        conditions = ["symbol=?", "timeframe=?"]
        params: list[Any] = [symbol, timeframe]
        if since:
            conditions.append("timestamp>=?")
            params.append(int(since.timestamp() * 1000))
        where = " AND ".join(conditions)
        order = "ORDER BY timestamp ASC"
        lim = f"LIMIT {limit}" if limit else ""
        query = f"SELECT timestamp,open,high,low,close,volume FROM ohlcv WHERE {where} {order} {lim}"
        async with self._db.execute(query, params) as cur:
            rows = await cur.fetchall()
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        return df.reset_index(drop=True)

    async def save_trade(self, trade: Any) -> None:
        from privateye.core.types import TradeRecord
        t: TradeRecord = trade
        await self._db.execute(
            "INSERT INTO trades(symbol,side,entry_price,exit_price,quantity,entry_time,exit_time,"
            "pnl,pnl_pct,fees,strategy_id,exit_reason,bars_held) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (t.symbol, t.side.value, t.entry_price, t.exit_price, t.quantity,
             int(t.entry_time.timestamp() * 1000), int(t.exit_time.timestamp() * 1000),
             t.pnl, t.pnl_pct, t.fees, t.strategy_id, t.exit_reason, t.bars_held),
        )
        await self._db.commit()

    async def load_trades(self, symbol: str | None = None) -> pd.DataFrame:
        query = "SELECT * FROM trades"
        params: list[Any] = []
        if symbol:
            query += " WHERE symbol=?"
            params.append(symbol)
        query += " ORDER BY entry_time ASC"
        async with self._db.execute(query, params) as cur:
            rows = await cur.fetchall()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([dict(r) for r in rows])
        df["entry_time"] = pd.to_datetime(df["entry_time"], unit="ms", utc=True)
        df["exit_time"] = pd.to_datetime(df["exit_time"], unit="ms", utc=True)
        return df

    async def export_trades_csv(self, path: str | Path) -> None:
        df = await self.load_trades()
        df.to_csv(path, index=False)
        log.info(f"Trades exported to {path}")
