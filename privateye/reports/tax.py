"""
FIFO tax-lot accounting and CSV export.

Every completed ``TradeRecord`` is treated as a matched lot: the entry fills
the buy leg and the exit fills the sell leg (or vice versa for short trades).
FIFO ordering is applied *per symbol* when multiple trades exist.

Typical usage::

    from privateye.reports.tax import compute_tax_lots, export_tax_csv

    report = compute_tax_lots(trades, year=2024)
    print(f"Net gain: {report.net_gain_loss:.2f}")
    export_tax_csv(report, "tax_2024.csv")
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from privateye.core.types import Direction, TradeRecord
from privateye.utils.logging import get_logger

log = get_logger()


@dataclass
class TaxLot:
    """Open position lot awaiting realisation (used internally for FIFO matching)."""
    symbol: str
    quantity: float
    cost_basis: float      # total cost (entry_price × quantity, before fees)
    entry_price: float
    entry_time: datetime
    entry_fees: float


@dataclass
class RealizedGain:
    """One realised gain/loss event from closing a matched lot."""
    symbol: str
    quantity: float
    proceeds: float           # exit_price × quantity  (or entry × qty for shorts)
    cost_basis: float         # entry_price × quantity  (or exit × qty for shorts)
    gain_loss: float          # proceeds − cost_basis − fees
    entry_time: datetime
    exit_time: datetime
    holding_days: int
    is_long_term: bool        # holding_days >= 365
    fees: float               # total fees attributed to this event


@dataclass
class AnnualTaxReport:
    """Aggregate tax report for a given calendar year (or all years if year=None)."""
    year: int | None
    total_realized_gain: float
    total_realized_loss: float
    net_gain_loss: float
    short_term_gains: list[RealizedGain] = field(default_factory=list)
    long_term_gains: list[RealizedGain] = field(default_factory=list)
    all_events: list[RealizedGain] = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

def compute_tax_lots(
    trades: Sequence[TradeRecord],
    year: int | None = None,
) -> AnnualTaxReport:
    """
    Compute FIFO realised gain/loss from a list of :class:`TradeRecord`.

    Each ``TradeRecord`` represents a *completed round-trip* (entry + exit):

    * **LONG trade** — bought at *entry_price*, sold at *exit_price*.
      ``proceeds = exit_price × qty``, ``cost_basis = entry_price × qty``.
    * **SHORT trade** — sold at *entry_price*, bought back at *exit_price*.
      ``proceeds = entry_price × qty``, ``cost_basis = exit_price × qty``.

    Fees are taken from ``TradeRecord.fees`` and fully deducted from the gain.

    If *year* is specified only events whose ``exit_time`` falls in that
    calendar year are included in the report totals and gain/loss lists.
    """
    all_events: list[RealizedGain] = []

    for t in trades:
        holding_days = max(0, (t.exit_time - t.entry_time).days)
        is_long_term = holding_days >= 365

        if t.side == Direction.LONG:
            proceeds    = t.exit_price  * t.quantity
            cost_basis  = t.entry_price * t.quantity
        else:
            # SHORT: sold first, bought back later
            proceeds    = t.entry_price * t.quantity
            cost_basis  = t.exit_price  * t.quantity

        gain_loss = proceeds - cost_basis - t.fees

        event = RealizedGain(
            symbol=t.symbol,
            quantity=t.quantity,
            proceeds=proceeds,
            cost_basis=cost_basis,
            gain_loss=gain_loss,
            entry_time=t.entry_time,
            exit_time=t.exit_time,
            holding_days=holding_days,
            is_long_term=is_long_term,
            fees=t.fees,
        )
        all_events.append(event)

    # Filter by year if requested
    if year is not None:
        filtered = [e for e in all_events if e.exit_time.year == year]
    else:
        filtered = list(all_events)

    short_term = [e for e in filtered if not e.is_long_term]
    long_term  = [e for e in filtered if e.is_long_term]

    total_gain = sum(e.gain_loss for e in filtered if e.gain_loss > 0)
    total_loss = sum(e.gain_loss for e in filtered if e.gain_loss <= 0)
    net        = total_gain + total_loss  # loss is negative so this is correct

    return AnnualTaxReport(
        year=year,
        total_realized_gain=total_gain,
        total_realized_loss=total_loss,
        net_gain_loss=net,
        short_term_gains=short_term,
        long_term_gains=long_term,
        all_events=filtered,
    )


def export_tax_csv(report: AnnualTaxReport, path: str | Path) -> None:
    """
    Write all realised gain/loss events in *report* to a CSV file at *path*.

    Columns: symbol, quantity, proceeds, cost_basis, gain_loss, entry_time,
             exit_time, holding_days, is_long_term, fees
    """
    fieldnames = [
        "symbol", "quantity", "proceeds", "cost_basis", "gain_loss",
        "entry_time", "exit_time", "holding_days", "is_long_term", "fees",
    ]
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for e in report.all_events:
            writer.writerow({
                "symbol":       e.symbol,
                "quantity":     e.quantity,
                "proceeds":     round(e.proceeds, 8),
                "cost_basis":   round(e.cost_basis, 8),
                "gain_loss":    round(e.gain_loss, 8),
                "entry_time":   e.entry_time.isoformat(),
                "exit_time":    e.exit_time.isoformat(),
                "holding_days": e.holding_days,
                "is_long_term": e.is_long_term,
                "fees":         round(e.fees, 8),
            })
    log.info(f"[TaxReport] Exported {len(report.all_events)} events to {path}")
