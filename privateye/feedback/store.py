"""
FeedbackStore — records signal→outcome pairs and human thumbs-up/down overrides.

Signal key formula:
    hashlib.md5(f"{symbol}|{direction}|{entry_price:.4f}|{strategy_id}".encode()).hexdigest()[:12]

Each FeedbackRecord is stored in an in-memory deque (maxlen configurable, default 1000).
When persist_path is given, records are also appended to a JSONL file so they survive
process restarts.
"""
from __future__ import annotations

import hashlib
import json
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from privateye.utils.logging import get_logger

if TYPE_CHECKING:
    from privateye.core.types import TradingSignal

log = get_logger()


@dataclass
class FeedbackRecord:
    signal_key:           str
    model_version_id:     str
    symbol:               str
    direction:            str
    confidence:           float
    entry_price:          float
    stacking_attribution: dict | None
    shap_snapshot:        dict | None
    pnl:                  float | None   # filled on FILL event
    outcome:              str | None     # "win" | "loss" | None
    thumbs_up:            bool | None    # None = no human override
    timestamp:            datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _make_signal_key(
    symbol: str,
    direction: str,
    entry_price: float,
    strategy_id: str,
) -> str:
    """Return a 12-character hex key for a given signal."""
    raw = f"{symbol}|{direction}|{entry_price:.4f}|{strategy_id}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


class FeedbackStore:
    """In-memory (+ optional JSONL) signal-outcome recorder.

    Parameters
    ----------
    maxlen:
        Maximum number of ``FeedbackRecord`` entries kept in memory.
    persist_path:
        Optional path to a JSONL file for durable storage. When provided,
        every new record is appended and the store is pre-loaded on init.
    """

    def __init__(
        self,
        maxlen: int = 1000,
        persist_path: str | Path | None = None,
    ) -> None:
        self._maxlen = int(maxlen)
        self._records: deque[FeedbackRecord] = deque(maxlen=self._maxlen)
        self._index: dict[str, FeedbackRecord] = {}  # signal_key → record
        self._persist_path: Path | None = Path(persist_path) if persist_path else None

        if self._persist_path is not None:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    # ── Public API ─────────────────────────────────────────────────────────────

    def record_signal(
        self,
        signal: "TradingSignal",
        model_version_id: str = "",
    ) -> str:
        """Create and store a FeedbackRecord for *signal*.

        Parameters
        ----------
        signal:
            The TradingSignal to record. Must have attributes:
            ``symbol``, ``direction`` (or ``.value``), ``confidence``,
            ``entry_price``, ``strategy_id``, ``metadata``.
        model_version_id:
            Checkpoint version string (e.g. "GBMClassifier_20260513_120000").

        Returns
        -------
        str
            The 12-character signal key.
        """
        direction = getattr(signal.direction, "value", str(signal.direction))
        key = _make_signal_key(
            signal.symbol, direction, float(signal.entry_price), signal.strategy_id
        )

        metadata = getattr(signal, "metadata", {}) or {}
        record = FeedbackRecord(
            signal_key=key,
            model_version_id=model_version_id,
            symbol=signal.symbol,
            direction=direction,
            confidence=float(signal.confidence),
            entry_price=float(signal.entry_price),
            stacking_attribution=metadata.get("stacking_attribution"),
            shap_snapshot=metadata.get("top_features"),
            pnl=None,
            outcome=None,
            thumbs_up=None,
        )

        # Evict the oldest entry from index if deque is full
        if len(self._records) == self._maxlen and self._records:
            oldest = self._records[0]
            self._index.pop(oldest.signal_key, None)

        self._records.append(record)
        self._index[key] = record

        if self._persist_path is not None:
            self._append_to_disk(record)

        return key

    def record_outcome(self, signal_key: str, pnl: float) -> None:
        """Update the P&L and outcome for a previously recorded signal.

        Parameters
        ----------
        signal_key:
            Key returned by :meth:`record_signal`.
        pnl:
            Realised P&L for this trade.
        """
        record = self._index.get(signal_key)
        if record is None:
            return

        record.pnl = float(pnl)
        record.outcome = "win" if pnl > 0 else "loss"

    def submit_feedback(self, signal_key: str, thumbs_up: bool) -> bool:
        """Set a human thumbs-up/down override on an existing record.

        Parameters
        ----------
        signal_key:
            Key returned by :meth:`record_signal`.
        thumbs_up:
            True = positive feedback; False = negative feedback.

        Returns
        -------
        bool
            True if the record was found and updated; False otherwise.
        """
        record = self._index.get(signal_key)
        if record is None:
            return False
        record.thumbs_up = bool(thumbs_up)
        return True

    def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent *limit* records as dicts, newest first.

        Parameters
        ----------
        limit:
            Maximum number of records to return.
        """
        items = list(self._records)
        items.reverse()
        return [self._to_dict(r) for r in items[:limit]]

    def get_attribution_trail(self, limit: int = 100) -> list[dict[str, Any]]:
        """Return records that have stacking_attribution data, newest first.

        Parameters
        ----------
        limit:
            Maximum number of records to return.
        """
        items = [r for r in reversed(self._records) if r.stacking_attribution]
        return [self._to_dict(r) for r in items[:limit]]

    def get_win_rate_by_version(self) -> dict[str, float]:
        """Return win rate per model_version_id for records with outcomes.

        Returns
        -------
        dict[str, float]
            ``{"version_id": win_rate_fraction, ...}``
        """
        wins: dict[str, int] = defaultdict(int)
        totals: dict[str, int] = defaultdict(int)

        for record in self._records:
            if record.outcome is None or not record.model_version_id:
                continue
            totals[record.model_version_id] += 1
            if record.outcome == "win":
                wins[record.model_version_id] += 1

        return {
            version: wins[version] / total
            for version, total in totals.items()
            if total > 0
        }

    # ── Internal helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _to_dict(record: FeedbackRecord) -> dict[str, Any]:
        d = asdict(record)
        # Serialise datetime to ISO string
        if isinstance(d.get("timestamp"), datetime):
            d["timestamp"] = d["timestamp"].isoformat()
        return d

    def _append_to_disk(self, record: FeedbackRecord) -> None:
        assert self._persist_path is not None
        try:
            line = json.dumps(self._to_dict(record), default=str)
            with self._persist_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            log.warning(f"[FeedbackStore] Could not persist record: {exc}")

    def _load_from_disk(self) -> None:
        assert self._persist_path is not None
        if not self._persist_path.exists():
            return
        try:
            with self._persist_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        ts = data.get("timestamp")
                        if isinstance(ts, str):
                            try:
                                data["timestamp"] = datetime.fromisoformat(ts)
                            except ValueError:
                                data["timestamp"] = datetime.now(timezone.utc)
                        record = FeedbackRecord(**data)
                        if len(self._records) == self._maxlen and self._records:
                            oldest = self._records[0]
                            self._index.pop(oldest.signal_key, None)
                        self._records.append(record)
                        self._index[record.signal_key] = record
                    except (json.JSONDecodeError, TypeError, KeyError):
                        pass  # skip malformed lines
        except OSError as exc:
            log.warning(f"[FeedbackStore] Could not load persisted records: {exc}")
