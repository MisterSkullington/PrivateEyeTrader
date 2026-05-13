#!/usr/bin/env python3
"""
Download historical OHLCV data from Binance and save to CSV + Parquet + SQLite.

Single symbol::

    python scripts/fetch_data.py --symbol BTC/USDT --timeframe 1h --days 730
    python scripts/fetch_data.py --symbol ETH/USDT --timeframe 4h --days 730 --format parquet

Multiple symbols::

    python scripts/fetch_data.py --symbols BTC/USDT,ETH/USDT --timeframes 1h,4h --days 730

Batch mode (all primary symbols × timeframes)::

    python scripts/fetch_data.py --all --days 730 --validate

Data quality report::

    python scripts/fetch_data.py --symbol BTC/USDT --timeframe 1h --days 30 --validate
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

# ── Primary universe ──────────────────────────────────────────────────────────
PRIMARY_SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "ADA/USDT",
    "DOGE/USDT",
]
# 1m/5m excluded from primary batch — too large for strategy-level backtest
PRIMARY_TIMEFRAMES = ["1h", "4h", "1d"]


# ── Fetch ─────────────────────────────────────────────────────────────────────

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


# ── Savers ────────────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame, symbol: str, timeframe: str, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
    path = data_dir / filename
    df.to_csv(path, index=False)
    log.info(f"Saved CSV: {path}")
    return path


def save_parquet(df: pd.DataFrame, symbol: str, timeframe: str, data_dir: Path) -> Path:
    """Save to Parquet (requires pyarrow)."""
    try:
        from privateye.data.providers.parquet_provider import ParquetProvider
        path = ParquetProvider(data_dir).save(df, symbol, timeframe)
        return path
    except ImportError:
        log.warning("[fetch_data] pyarrow not installed — skipping Parquet output. "
                    "Install with: pip install pyarrow")
        return data_dir / f"{symbol.replace('/', '_')}_{timeframe}.parquet"


async def save_sqlite(df: pd.DataFrame, symbol: str, timeframe: str, db_path: str) -> None:
    from privateye.data.storage.sqlite_store import SQLiteStore
    store = SQLiteStore(db_path)
    await store.open()
    bars = df.to_dict("records")
    n = await store.upsert_ohlcv(symbol, timeframe, bars)
    await store.close()
    log.info(f"Saved {n} bars to SQLite: {db_path}")


# ── Validation ────────────────────────────────────────────────────────────────

def run_validation(df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
    """Run DataQualityReport and log summary. Returns is_clean."""
    from privateye.data.quality import validate_bars
    report = validate_bars(df, symbol, timeframe)
    log.info(report.summary())
    if not report.is_clean:
        log.warning(f"[fetch_data] Data quality issues found for {symbol} {timeframe}:")
        for issue in report.issues[:10]:   # cap at 10 to avoid flooding logs
            log.warning(f"  {issue}")
        if len(report.issues) > 10:
            log.warning(f"  … and {len(report.issues) - 10} more")
    return report.is_clean


# ── Per-symbol save+validate ──────────────────────────────────────────────────

def process_one(
    symbol: str,
    timeframe: str,
    days: int,
    sandbox: bool,
    data_dir: Path,
    db: str,
    fmt: str,
    csv_only: bool,
    validate: bool,
) -> bool:
    """Fetch, save, and optionally validate one symbol/timeframe. Returns success."""
    df = fetch_ohlcv_ccxt(symbol, timeframe, days, sandbox)
    if df.empty:
        log.error(f"No data for {symbol} {timeframe} — skipping")
        return False

    if fmt in ("csv", "both"):
        save_csv(df, symbol, timeframe, data_dir)
    if fmt in ("parquet", "both"):
        save_parquet(df, symbol, timeframe, data_dir)

    if not csv_only and fmt not in ("parquet",):
        asyncio.run(save_sqlite(df, symbol, timeframe, db))

    if validate:
        run_validation(df, symbol, timeframe)

    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch historical OHLCV data")
    # Single-symbol args (original)
    parser.add_argument("--symbol",    default="BTC/USDT",
                        help="Trading pair, e.g. BTC/USDT")
    parser.add_argument("--timeframe", default="1h",
                        help="Bar timeframe, e.g. 1h")
    parser.add_argument("--days",      type=int, default=365,
                        help="Number of days of history")
    parser.add_argument("--sandbox",   action="store_true",
                        help="Use Binance sandbox")
    parser.add_argument("--data-dir",  default="data/historical",
                        help="Directory for output files")
    parser.add_argument("--db",        default="data/privateye.db",
                        help="SQLite database path")
    parser.add_argument("--csv-only",  action="store_true",
                        help="Save CSV only, skip SQLite")
    # New Phase 0 args
    parser.add_argument("--all",       action="store_true",
                        help="Batch: fetch PRIMARY_SYMBOLS × PRIMARY_TIMEFRAMES")
    parser.add_argument("--symbols",   default=None,
                        help="Comma-separated symbols, e.g. ETH/USDT,SOL/USDT")
    parser.add_argument("--timeframes", default=None,
                        help="Comma-separated timeframes, e.g. 1h,4h,1d")
    parser.add_argument("--format",    choices=["csv", "parquet", "both"],
                        default="both",
                        help="Output format (default: both)")
    parser.add_argument("--validate",  action="store_true",
                        help="Run DataQualityReport after fetch")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # ── Build fetch list ──────────────────────────────────────────────────────
    if args.all:
        symbols    = PRIMARY_SYMBOLS
        timeframes = PRIMARY_TIMEFRAMES
    elif args.symbols or args.timeframes:
        symbols    = [s.strip() for s in (args.symbols or args.symbol).split(",")]
        timeframes = [t.strip() for t in (args.timeframes or args.timeframe).split(",")]
    else:
        symbols    = [args.symbol]
        timeframes = [args.timeframe]

    # ── Execute ───────────────────────────────────────────────────────────────
    total  = len(symbols) * len(timeframes)
    ok     = 0
    failed = []

    for sym in symbols:
        for tf in timeframes:
            log.info(f"{'─'*60}")
            success = process_one(
                symbol=sym,
                timeframe=tf,
                days=args.days,
                sandbox=args.sandbox,
                data_dir=data_dir,
                db=args.db,
                fmt=args.format,
                csv_only=args.csv_only,
                validate=args.validate,
            )
            if success:
                ok += 1
            else:
                failed.append(f"{sym} {tf}")

    log.info(f"{'='*60}")
    log.info(f"Fetch complete: {ok}/{total} succeeded")
    if failed:
        log.warning(f"Failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
