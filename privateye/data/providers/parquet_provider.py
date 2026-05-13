"""
Parquet-backed OHLCV data provider.

Same interface as CSVProvider — drop-in replacement with ~5× faster load times
on large bar sets thanks to columnar compression.

File naming convention: ``{symbol_safe}_{timeframe}.parquet``
where ``symbol_safe = symbol.replace('/', '_')``.

Requires ``pyarrow`` (``pip install pyarrow`` or ``pip install privateye[phase0]``).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from privateye.utils.logging import get_logger

log = get_logger()

_REQUIRED_COLS = {"timestamp", "open", "high", "low", "close", "volume"}


class ParquetProvider:
    """Load and save OHLCV data in Parquet format.

    Args:
        data_dir: Directory containing ``*.parquet`` files.
    """

    def __init__(self, data_dir: str | Path = "data/historical") -> None:
        self.data_dir = Path(data_dir)

    # ── Public API ────────────────────────────────────────────────────────────

    def load(
        self,
        symbol: str,
        timeframe: str,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Load bars for ``symbol`` / ``timeframe``.

        Returns an empty DataFrame if the file does not exist.
        Returned DataFrame is guaranteed sorted ascending by timestamp,
        duplicates removed, and column names lower-cased.
        """
        path = self._path(symbol, timeframe)
        if not path.exists():
            return pd.DataFrame()

        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            log.warning(f"[ParquetProvider] Failed to read {path}: {exc}")
            return pd.DataFrame()

        df = self._normalise(df)
        if df.empty:
            return df

        if limit is not None:
            df = df.tail(limit).reset_index(drop=True)

        return df

    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> Path:
        """Save ``df`` to ``{data_dir}/{symbol_safe}_{timeframe}.parquet``.

        Normalises column names and sorts by timestamp before writing.
        Returns the path written.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        df = self._normalise(df.copy())
        path = self._path(symbol, timeframe)
        df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
        log.info(f"[ParquetProvider] Saved {len(df)} bars → {path}")
        return path

    def available(self) -> list[tuple[str, str]]:
        """Return all ``(symbol, timeframe)`` pairs available in data_dir."""
        results: list[tuple[str, str]] = []
        if not self.data_dir.exists():
            return results
        for p in sorted(self.data_dir.glob("*.parquet")):
            parts = p.stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            sym_safe, tf = parts
            results.append((sym_safe.replace("_", "/"), tf))
        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _path(self, symbol: str, timeframe: str) -> Path:
        safe = symbol.replace("/", "_")
        return self.data_dir / f"{safe}_{timeframe}.parquet"

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase columns, ensure timestamp is tz-aware, deduplicate, sort."""
        df.columns = [c.strip().lower() for c in df.columns]
        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            log.warning(f"[ParquetProvider] Missing columns: {missing}")
            return pd.DataFrame()

        # Ensure timezone-aware datetime
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        elif df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

        df = df.drop_duplicates("timestamp")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
