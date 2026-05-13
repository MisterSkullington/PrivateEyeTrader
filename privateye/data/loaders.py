"""
Unified bar-loading utility.

Prefers Parquet over CSV (faster, smaller on disk) and falls back
gracefully when only one format is available.

Usage::

    from privateye.data.loaders import load_bars

    bars = load_bars("BTC/USDT", "1h")
    bars = load_bars("ETH/USDT", "4h", data_dir="data/historical")
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from privateye.utils.logging import get_logger

log = get_logger()


def load_bars(
    symbol: str,
    timeframe: str,
    data_dir: str | Path = "data/historical",
    limit: int | None = None,
) -> pd.DataFrame:
    """Load OHLCV bars, preferring Parquet over CSV.

    Tries in order:
    1. ``ParquetProvider`` — ``{symbol_safe}_{timeframe}.parquet``
    2. ``CSVProvider``     — ``{symbol_safe}_{timeframe}.csv``

    Returns an empty DataFrame when neither file exists.
    Returned DataFrame is sorted ascending by timestamp with no duplicates.
    """
    data_dir = Path(data_dir)

    # ── 1. Try Parquet ────────────────────────────────────────────────────────
    try:
        from privateye.data.providers.parquet_provider import ParquetProvider
        df = ParquetProvider(data_dir).load(symbol, timeframe, limit=limit)
        if not df.empty:
            log.debug(f"[loaders] Loaded {len(df)} bars from parquet ({symbol} {timeframe})")
            return df
    except ImportError:
        pass  # pyarrow not installed — fall through to CSV
    except Exception as exc:
        log.debug(f"[loaders] Parquet load failed ({symbol} {timeframe}): {exc}")

    # ── 2. Fall back to CSV ───────────────────────────────────────────────────
    from privateye.data.providers.csv_provider import CSVProvider
    df = CSVProvider(data_dir).load(symbol, timeframe, limit=limit)
    if not df.empty:
        log.debug(f"[loaders] Loaded {len(df)} bars from CSV ({symbol} {timeframe})")
    else:
        log.warning(
            f"[loaders] No data found for {symbol} {timeframe} in {data_dir}. "
            "Run: python scripts/fetch_data.py "
            f"--symbol {symbol} --timeframe {timeframe}"
        )
    return df
