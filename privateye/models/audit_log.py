"""
Append-only JSONL audit trail for every model update cycle.

Each entry is one JSON line::

    {
        "timestamp":       "2026-05-13T12:00:00+00:00",
        "trigger":         "scheduled" | "drift" | "manual",
        "models_updated":  ["gbm", "lstm"],
        "models_skipped":  ["regime"],
        "bars_seen":       2000,
        "drift_detected":  false,
        "drift_fraction":  null,
        "psi_score":       null,
        "live_perf_gate":  "passed" | "rollback" | "not_evaluated",
        "ewc_active":      false
    }
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from privateye.utils.logging import get_logger

log = get_logger()


class ModelUpdateAuditLog:
    """Thread-safe append-only JSONL model-update audit log.

    Each :meth:`record` call appends one JSON line to the log file.
    :meth:`tail` reads the most recent ``n`` entries from disk (newest first).
    :meth:`clear` truncates the file — intended for testing only.

    Parameters
    ----------
    log_path:
        Path to the JSONL file. Parent directories are created if missing.
    """

    def __init__(self, log_path: str | Path = "data/model_audit.jsonl") -> None:
        self._path = Path(log_path)
        self._lock = threading.Lock()
        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def record(self, entry: dict[str, Any]) -> None:
        """Append *entry* as a single JSON line.

        A ``"timestamp"`` key (ISO-8601 string) is automatically added when
        absent.  The write is protected by a threading lock so concurrent
        online-learner calls from a thread-pool executor are safe.

        Parameters
        ----------
        entry:
            Dict containing any subset of the standard audit fields.
            Non-JSON-serialisable values are stringified via ``default=str``.
        """
        if "timestamp" not in entry:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **entry,
            }

        line = json.dumps(entry, default=str)
        with self._lock:
            try:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError as exc:
                log.error(f"[ModelUpdateAuditLog] Failed to write entry: {exc}")

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        """Return the most recent *n* audit entries, newest first.

        Reads the entire file and returns the last ``n`` lines parsed as
        dicts.  Malformed lines are silently skipped.

        Parameters
        ----------
        n:
            Maximum number of entries to return (default 50).
        """
        if not self._path.exists():
            return []

        try:
            with self._path.open("r", encoding="utf-8") as fh:
                raw_lines = [line.strip() for line in fh if line.strip()]
        except OSError as exc:
            log.warning(f"[ModelUpdateAuditLog] Could not read log: {exc}")
            return []

        results: list[dict[str, Any]] = []
        for line in reversed(raw_lines):
            if len(results) >= n:
                break
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # skip corrupted lines

        return results

    def clear(self) -> None:
        """Truncate the log file.  For testing purposes only."""
        with self._lock:
            try:
                self._path.write_text("", encoding="utf-8")
            except OSError as exc:
                log.warning(f"[ModelUpdateAuditLog] Could not clear log: {exc}")
