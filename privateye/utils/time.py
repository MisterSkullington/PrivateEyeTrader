"""Timestamp helpers, bar alignment, and timeframe utilities."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta


TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "12h": 43200,
    "1d": 86400,
    "1w": 604800,
}


def tf_to_seconds(timeframe: str) -> int:
    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    return TIMEFRAME_SECONDS[timeframe]


def tf_to_ms(timeframe: str) -> int:
    return tf_to_seconds(timeframe) * 1000


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def ts_to_datetime(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)


def datetime_to_ts(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def floor_to_bar(dt: datetime, timeframe: str) -> datetime:
    """Round datetime down to the start of the containing bar."""
    secs = tf_to_seconds(timeframe)
    epoch = dt.timestamp()
    floored = int(epoch // secs) * secs
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def bar_range(start: datetime, end: datetime, timeframe: str) -> list[datetime]:
    """Generate bar open timestamps from start to end (inclusive)."""
    secs = tf_to_seconds(timeframe)
    result = []
    current = floor_to_bar(start, timeframe)
    while current <= end:
        result.append(current)
        current += timedelta(seconds=secs)
    return result


def days_ago(n: int) -> datetime:
    return now_utc() - timedelta(days=n)
