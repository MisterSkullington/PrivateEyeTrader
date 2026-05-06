#!/usr/bin/env python3
"""
Download historical OHLCV data from Binance and save to CSV + SQLite.

Usage:
    python scripts/fetch_data.py --symbol BTC/USDT --timeframe 1h --days 365
    python scripts/fetch_data.py --symbol BTC/USDT --timeframe 1h --days 365 --sandbox
"""
import argparse
import asyncio
import sys
from pathlib import Path

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

import ccxt
import pandas as pd

from privateye.utils.logging import setup_logging, get_logger
from privateye.utils.time import now_utc, tf_to_ms

setup_logging()
log = get_logger()


def fetch_ohlcv_ccxt(
    symbol: str,
    timeframe: str,
    days: int,
    sandbox: bool = False,
) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    if sandbox:
        exchange.set_sandbox_mode(True)

    since_ms = int((now_utc().timestamp() - days * 86400) * 1000)
    tf_ms = tf_to_ms(timeframe)
    all_bars: list[list] = []

    log.info(f"Fetching {symbol} {timeframe} ({days}d) from Binance…")
    while True:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=1000)
        if not bars:
            break
        all_bars.extend(bars)
        if len(bars) < 1000:
            break
        since_ms = bars[-1][0] + tf_ms

    if not all_bars:
        log.error("No data returned")
        return pd.DataFrame()

    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    log.info(f"Fetched {len(df)} bars from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    return df


def save_csv(df: pd.DataFrame, symbol: str, timeframe: str, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
    path = data_dir / filename
    df.to_csv(path, index=False)
    log.info(f"Saved CSV: {path}")
    return path


async def save_sqlite(df: pd.DataFrame, symbol: str, timeframe: str, db_path: str) -> None:
    from privateye.data.storage.sqlite_store import SQLiteStore
    store = SQLiteStore(db_path)
    await store.open()
    bars = df.to_dict("records")
    n = await store.upsert_ohlcv(symbol, timeframe, bars)
    await store.close()
    log.info(f"Saved {n} bars to SQLite: {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical OHLCV data")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair, e.g. BTC/USDT")
    parser.add_argument("--timeframe", default="1h", help="Bar timeframe, e.g. 1h")
    parser.add_argument("--days", type=int, default=365, help="Number of days of history")
    parser.add_argument("--sandbox", action="store_true", help="Use Binance sandbox")
    parser.add_argument("--data-dir", default="data/historical", help="Directory for CSV files")
    parser.add_argument("--db", default="data/privateye.db", help="SQLite database path")
    parser.add_argument("--csv-only", action="store_true", help="Save CSV only, skip SQLite")
    args = parser.parse_args()

    df = fetch_ohlcv_ccxt(args.symbol, args.timeframe, args.days, args.sandbox)
    if df.empty:
        sys.exit(1)

    save_csv(df, args.symbol, args.timeframe, Path(args.data_dir))

    if not args.csv_only:
        asyncio.run(save_sqlite(df, args.symbol, args.timeframe, args.db))


if __name__ == "__main__":
    main()
