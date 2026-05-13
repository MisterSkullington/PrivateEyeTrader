"""
Data quality validation and manifest generation for OHLCV bar data.

Usage::
    from privateye.data.quality import validate_bars, generate_manifest

    report = validate_bars(df, "BTC/USDT", "1h")
    if not report.is_clean:
        for issue in report.issues:
            print(issue)

    manifest = generate_manifest("data/historical", "data_manifest.yaml")
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from privateye.utils.logging import get_logger

log = get_logger()

# Expected bar-interval in seconds per timeframe (used for gap detection)
_TF_SECONDS: dict[str, int] = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "12h": 43200,
    "1d": 86400, "1w": 604800,
}


@dataclass
class BarQualityIssue:
    bar_index: int
    timestamp: Any
    issue_type: str   # "duplicate" | "gap" | "zero_volume" | "price_jump" | "negative_price"
    details: str

    def __str__(self) -> str:
        return f"[{self.issue_type}] bar {self.bar_index} @ {self.timestamp}: {self.details}"


@dataclass
class DataQualityReport:
    symbol: str
    timeframe: str
    n_bars: int
    date_from: str
    date_to: str
    checksum: str                       # SHA-256 hex of the close price array bytes
    issues: list[BarQualityIssue] = field(default_factory=list)
    n_duplicates: int = 0
    n_gaps: int = 0                     # gaps > max_gap_bars consecutive bars
    n_zero_volume: int = 0
    n_price_jumps: int = 0              # single-bar close change > max_price_jump_pct
    n_negative: int = 0                 # negative OHLCV values (critical)
    is_clean: bool = True               # True when n_duplicates == 0 and n_negative == 0

    def summary(self) -> str:
        status = "CLEAN" if self.is_clean else "ISSUES FOUND"
        lines = [
            f"[DataQuality] {self.symbol} {self.timeframe} — {status}",
            f"  Bars     : {self.n_bars} ({self.date_from} → {self.date_to})",
            f"  Checksum : {self.checksum[:16]}…",
            f"  Issues   : duplicates={self.n_duplicates} gaps={self.n_gaps} "
            f"zero_vol={self.n_zero_volume} price_jumps={self.n_price_jumps} "
            f"negative={self.n_negative}",
        ]
        return "\n".join(lines)


def validate_bars(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    max_gap_bars: int = 3,
    max_price_jump_pct: float = 0.15,
) -> DataQualityReport:
    """Validate a bars DataFrame.

    Checks performed:
    * Duplicate timestamps (critical → is_clean=False)
    * Timestamp gaps > ``max_gap_bars`` consecutive missing bars
    * Zero or negative volume
    * Single-bar close-to-close price change > ``max_price_jump_pct``
    * Negative OHLCV values (critical → is_clean=False)

    The checksum is ``sha256(close_array.astype(float64).tobytes()).hexdigest()``.
    Running the same DataFrame through this function twice always produces the
    same checksum; it does NOT depend on row order because the DataFrame is
    assumed to be pre-sorted by the caller.
    """
    if df.empty:
        return DataQualityReport(
            symbol=symbol, timeframe=timeframe, n_bars=0,
            date_from="", date_to="", checksum="",
            is_clean=False,
        )

    issues: list[BarQualityIssue] = []

    # ── Checksum ─────────────────────────────────────────────────────────────
    close_arr = df["close"].to_numpy(dtype=np.float64)
    checksum = hashlib.sha256(close_arr.tobytes()).hexdigest()

    date_from = str(df["timestamp"].iloc[0])
    date_to   = str(df["timestamp"].iloc[-1])
    n_bars    = len(df)

    # ── Duplicate timestamps ─────────────────────────────────────────────────
    dup_mask = df["timestamp"].duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    for idx in df[dup_mask].index:
        issues.append(BarQualityIssue(
            bar_index=int(idx),
            timestamp=df["timestamp"].iloc[idx] if idx < n_bars else "?",
            issue_type="duplicate",
            details=f"timestamp duplicated: {df.loc[idx, 'timestamp']}",
        ))

    # ── Negative OHLCV ───────────────────────────────────────────────────────
    n_neg = 0
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            continue
        neg = (df[col] < 0)
        n_neg += int(neg.sum())
        for i in df.index[neg]:
            issues.append(BarQualityIssue(
                bar_index=int(i),
                timestamp=df["timestamp"].iloc[i] if i < n_bars else "?",
                issue_type="negative_price",
                details=f"{col}={df.loc[i, col]:.6f}",
            ))

    # ── Timestamp gaps ───────────────────────────────────────────────────────
    tf_sec = _TF_SECONDS.get(timeframe)
    n_gaps = 0
    if tf_sec is not None and len(df) > 1:
        ts = pd.to_datetime(df["timestamp"])
        diffs_sec = ts.diff().dt.total_seconds().dropna()
        expected  = float(tf_sec)
        for i, diff in enumerate(diffs_sec, start=1):
            if diff <= 0:
                continue  # already caught as duplicate
            missing = round((diff - expected) / expected)
            if missing > max_gap_bars:
                n_gaps += 1
                issues.append(BarQualityIssue(
                    bar_index=i,
                    timestamp=df["timestamp"].iloc[i],
                    issue_type="gap",
                    details=f"{int(missing)} bars missing before this bar",
                ))

    # ── Zero / negative volume ───────────────────────────────────────────────
    n_zero_vol = 0
    if "volume" in df.columns:
        zero_mask = df["volume"] <= 0
        n_zero_vol = int(zero_mask.sum())
        for i in df.index[zero_mask]:
            issues.append(BarQualityIssue(
                bar_index=int(i),
                timestamp=df["timestamp"].iloc[i] if i < n_bars else "?",
                issue_type="zero_volume",
                details=f"volume={df.loc[i, 'volume']}",
            ))

    # ── Price jumps ──────────────────────────────────────────────────────────
    n_jumps = 0
    if "close" in df.columns and len(df) > 1:
        pct_changes = df["close"].pct_change().abs()
        jump_mask   = pct_changes > max_price_jump_pct
        n_jumps     = int(jump_mask.sum())
        for i in df.index[jump_mask]:
            issues.append(BarQualityIssue(
                bar_index=int(i),
                timestamp=df["timestamp"].iloc[i] if i < n_bars else "?",
                issue_type="price_jump",
                details=f"close change {pct_changes.loc[i]:.1%} > {max_price_jump_pct:.0%} threshold",
            ))

    is_clean = (n_dup == 0) and (n_neg == 0)

    return DataQualityReport(
        symbol=symbol,
        timeframe=timeframe,
        n_bars=n_bars,
        date_from=date_from,
        date_to=date_to,
        checksum=checksum,
        issues=issues,
        n_duplicates=n_dup,
        n_gaps=n_gaps,
        n_zero_volume=n_zero_vol,
        n_price_jumps=n_jumps,
        n_negative=n_neg,
        is_clean=is_clean,
    )


def generate_manifest(
    data_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict:
    """Scan *data_dir* for all ``.csv`` and ``.parquet`` files.

    For each file found, loads it, calls :func:`validate_bars`, and records:
    * date_from / date_to
    * n_bars
    * checksum
    * is_clean flag
    * any issues (summarised as counts)

    Writes the manifest to ``output_path`` as YAML when provided.
    Returns the manifest dict regardless.

    Example manifest entry::

        BTC/USDT:
          1h:
            n_bars: 17520
            date_from: "2024-05-07T15:00:00+00:00"
            date_to:   "2026-05-07T14:00:00+00:00"
            checksum:  "a3f9..."
            is_clean:  true
            n_duplicates: 0
            n_gaps: 0
    """
    data_dir = Path(data_dir)
    manifest: dict = {}

    # Collect files — prefer parquet over csv for same symbol/timeframe
    seen: dict[tuple[str, str], Path] = {}
    for ext in (".parquet", ".csv"):
        for p in sorted(data_dir.glob(f"*{ext}")):
            stem = p.stem  # e.g. BTC_USDT_1h
            parts = stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            sym_safe, tf = parts
            symbol = sym_safe.replace("_", "/")
            key = (symbol, tf)
            if key not in seen:          # parquet takes priority (listed first)
                seen[key] = p

    for (symbol, tf), path in sorted(seen.items()):
        try:
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.sort_values("timestamp").reset_index(drop=True)

            report = validate_bars(df, symbol, tf)
            entry: dict = {
                "n_bars":       report.n_bars,
                "date_from":    report.date_from,
                "date_to":      report.date_to,
                "checksum":     report.checksum,
                "is_clean":     report.is_clean,
                "n_duplicates": report.n_duplicates,
                "n_gaps":       report.n_gaps,
                "n_zero_volume": report.n_zero_volume,
                "n_price_jumps": report.n_price_jumps,
            }
            manifest.setdefault(symbol, {})[tf] = entry
            log.info(report.summary())
        except Exception as exc:
            log.warning(f"[DataQuality] Failed to validate {path}: {exc}")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            yaml.safe_dump(manifest, fh, default_flow_style=False, sort_keys=True)
        log.info(f"[DataQuality] Manifest written to {output_path}")

    return manifest
