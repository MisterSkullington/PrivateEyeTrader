"""Multi-timeframe bar aggregation and alignment utilities."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from privateye.utils.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger()

RESAMPLE_MAP: dict[str, str] = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1D",
    "1w": "1W",
}


def resample_ohlcv(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    """
    Aggregate a lower-timeframe OHLCV DataFrame into a higher timeframe.
    df must have a UTC-aware 'timestamp' column.
    """
    rule = RESAMPLE_MAP.get(target_tf)
    if rule is None:
        raise ValueError(f"Unknown timeframe for resampling: {target_tf}")

    ts = df.set_index("timestamp")
    ts.index = pd.DatetimeIndex(ts.index)

    resampled = ts.resample(rule, label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open"])

    resampled = resampled.reset_index().rename(columns={"index": "timestamp"})
    # Ensure timestamp column name is correct after reset
    if "timestamp" not in resampled.columns:
        resampled = resampled.rename(columns={resampled.columns[0]: "timestamp"})
    return resampled


class MultiTimeframeAggregator:
    """
    Maintains rolling OHLCV bars for multiple timeframes derived from a
    base (lowest) timeframe feed. Used during live/paper mode to avoid
    requesting multiple REST endpoints for each TF.

    During backtesting, each timeframe is loaded separately from CSV/SQLite
    and aligned bar-by-bar to prevent look-ahead.
    """

    def __init__(self, base_tf: str, target_timeframes: list[str]) -> None:
        self.base_tf = base_tf
        self.target_timeframes = target_timeframes
        self._base_bars: pd.DataFrame = pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        self._cache: dict[str, pd.DataFrame] = {}

    def push(self, bar: dict) -> None:
        """Append a new base-timeframe bar and update all aggregated caches."""
        new_row = pd.DataFrame([bar])
        if not isinstance(new_row["timestamp"].iloc[0], pd.Timestamp):
            new_row["timestamp"] = pd.to_datetime(new_row["timestamp"], utc=True)
        self._base_bars = pd.concat([self._base_bars, new_row], ignore_index=True)
        # Keep only last 5000 base bars to bound memory
        if len(self._base_bars) > 5000:
            self._base_bars = self._base_bars.tail(5000).reset_index(drop=True)
        self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        for tf in self.target_timeframes:
            if tf == self.base_tf:
                self._cache[tf] = self._base_bars.copy()
            else:
                try:
                    self._cache[tf] = resample_ohlcv(self._base_bars, tf)
                except Exception as e:
                    log.debug(f"MTF resample {self.base_tf}→{tf}: {e}")

    def get(self, timeframe: str, window: int = 500) -> pd.DataFrame:
        bars = self._cache.get(timeframe, pd.DataFrame())
        if bars.empty:
            return bars
        return bars.tail(window).reset_index(drop=True)

    def get_all(self, window: int = 500) -> dict[str, pd.DataFrame]:
        return {tf: self.get(tf, window) for tf in self.target_timeframes}
