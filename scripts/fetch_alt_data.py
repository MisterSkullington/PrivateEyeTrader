#!/usr/bin/env python3
"""
Fetch alternative data (funding rates, Fear & Greed Index) and store in SQLite.

Usage:
    python scripts/fetch_alt_data.py --symbol BTC/USDT --days 365
    python scripts/fetch_alt_data.py --symbol BTC/USDT --days 365 --db data/privateye.db
"""
import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from privateye.utils.logging import setup_logging, get_logger

setup_logging()
log = get_logger()

CREATE_FUNDING = """
CREATE TABLE IF NOT EXISTS funding_rates (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT    NOT NULL,
    timestamp     INTEGER NOT NULL,
    funding_rate  REAL    NOT NULL,
    open_interest REAL    NOT NULL DEFAULT 0,
    UNIQUE(symbol, timestamp)
)
"""

CREATE_FEAR_GREED = """
CREATE TABLE IF NOT EXISTS fear_greed (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp      INTEGER NOT NULL UNIQUE,
    fear_greed     REAL    NOT NULL,
    classification TEXT    NOT NULL DEFAULT ''
)
"""


async def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch alt data into SQLite")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--days", type=int, default=365, help="Days of history to fetch")
    parser.add_argument("--db", default="data/privateye.db", help="SQLite database path")
    parser.add_argument("--skip-funding", action="store_true")
    parser.add_argument("--skip-sentiment", action="store_true")
    args = parser.parse_args()

    import aiosqlite

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(db_path) as db:
        await db.execute(CREATE_FUNDING)
        await db.execute(CREATE_FEAR_GREED)
        await db.commit()

        if not args.skip_funding:
            await _fetch_funding(db, args.symbol, args.days)

        if not args.skip_sentiment:
            await _fetch_sentiment(db, args.days)

    log.info("Alt data fetch complete.")


async def _fetch_funding(db, symbol: str, days: int) -> None:
    from privateye.data.providers.funding_rates import FundingRateProvider

    provider = FundingRateProvider(symbol=symbol)
    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

    log.info(f"Fetching funding rate history for {symbol} ({days} days)...")
    df = await provider.fetch_history(since_ms=since_ms, limit=1000)

    if df.empty:
        log.warning("No funding rate data returned.")
        return

    rows = [
        (
            symbol,
            int(row["timestamp"].timestamp() * 1000),
            float(row["funding_rate"]),
            float(row.get("open_interest", 0.0)),
        )
        for _, row in df.iterrows()
    ]
    await db.executemany(
        "INSERT OR IGNORE INTO funding_rates(symbol, timestamp, funding_rate, open_interest) "
        "VALUES(?,?,?,?)",
        rows,
    )
    await db.commit()
    log.info(f"Stored {len(rows)} funding rate records for {symbol}")


async def _fetch_sentiment(db, days: int) -> None:
    from privateye.data.providers.sentiment import FearGreedProvider

    provider = FearGreedProvider()
    log.info(f"Fetching Fear & Greed history ({days} days)...")
    df = await provider.fetch_history(limit=days)

    if df.empty:
        log.warning("No Fear & Greed data returned.")
        return

    rows = [
        (
            int(row["timestamp"].timestamp() * 1000),
            float(row["fear_greed"]),
            str(row.get("classification", "")),
        )
        for _, row in df.iterrows()
    ]
    await db.executemany(
        "INSERT OR IGNORE INTO fear_greed(timestamp, fear_greed, classification) VALUES(?,?,?)",
        rows,
    )
    await db.commit()
    log.info(f"Stored {len(rows)} Fear & Greed records")


if __name__ == "__main__":
    asyncio.run(main())
