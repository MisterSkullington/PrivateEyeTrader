"""CSV data provider for backtesting. Reads OHLCV data from local CSV files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from privateye.core.exceptions import DataError
from privateye.utils.logging import get_logger

log = get_logger()

EXPECTED_COLS = {"timestamp", "open", "high", "low", "close", "volume"}


class CSVProvider:
    """
    Reads OHLCV bars from CSV files. Expected filename format:
        {data_dir}/{symbol_safe}_{timeframe}.csv
    e.g.: data/historical/BTC_USDT_1h.csv

    CSV format (with or without header):
        timestamp, open, high, low, close, volume
    timestamp can be: ISO-8601 string, Unix seconds, or Unix milliseconds.
    """

    def __init__(self, data_dir: str | Path = "data/historical") -> None:
        self.data_dir = Path(data_dir)

    def _symbol_safe(self, symbol: str) -> str:
        return symbol.replace("/", "_")

    def _find_file(self, symbol: str, timeframe: str) -> Path:
        name = f"{self._symbol_safe(symbol)}_{timeframe}.csv"
        path = self.data_dir / name
        if not path.exists():
            raise DataError(
                f"CSV not found: {path}. "
                f"Run: python scripts/fetch_data.py --symbol {symbol} --timeframe {timeframe}"
            )
        return path

    def load(self, symbol: str, timeframe: str, limit: int | None = None) -> pd.DataFrame:
        path = self._find_file(symbol, timeframe)
        df = pd.read_csv(path)

        # Normalise column names
        df.columns = [c.lower().strip() for c in df.columns]
        if not EXPECTED_COLS.issubset(df.columns):
            # Try positional (no header)
            df = pd.read_csv(path, header=None,
                             names=["timestamp", "open", "high", "low", "close", "volume"])

        # Parse timestamp
        ts = df["timestamp"]
        if pd.api.types.is_numeric_dtype(ts):
            # Distinguish ms vs seconds
            if ts.iloc[0] > 1e12:
                df["timestamp"] = pd.to_datetime(ts, unit="ms", utc=True)
            else:
                df["timestamp"] = pd.to_datetime(ts, unit="s", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(ts, utc=True)

        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.drop_duplicates("timestamp")

        if limit:
            df = df.tail(limit).reset_index(drop=True)

        log.debug(f"Loaded {len(df)} bars from {path}")
        return df

    def available(self) -> list[tuple[str, str]]:
        """Return list of (symbol, timeframe) tuples available in data_dir."""
        result = []
        for f in self.data_dir.glob("*.csv"):
            parts = f.stem.rsplit("_", 1)
            if len(parts) == 2:
                symbol = parts[0].replace("_", "/")
                timeframe = parts[1]
                result.append((symbol, timeframe))
        return result
