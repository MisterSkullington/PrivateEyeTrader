"""
Wash-sale pattern detector.

A wash sale (simplified) occurs when a position is closed at a loss and
the same asset is re-entered within the wash-sale window (default 30 days)
after the closing exit.

Regulatory note: As of 2024 the IRS treats cryptocurrency as property
(not a security), so IRC §1091 wash-sale rules do not technically apply.
This detector provides *advisory* flags — useful if regulations change or
for conservative tax compliance in other jurisdictions.

The disallowed-loss field mirrors the amount that would be non-deductible
under a strict wash-sale interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from privateye.core.types import TradeRecord


@dataclass
class WashSaleFlag:
    """A detected wash-sale pattern between two trades on the same symbol."""

    symbol:          str
    loss_trade:      TradeRecord  # trade closed at a loss
    reentry_trade:   TradeRecord  # subsequent re-entry within the window
    days_between:    int          # calendar days from loss exit → reentry entry
    disallowed_loss: float        # abs(loss_trade.pnl) — the flagged loss


class WashSaleDetector:
    """
    Detect wash-sale patterns in a completed trade list.

    Algorithm
    ---------
    1. Group trades by symbol.
    2. Within each symbol group, sort chronologically by *exit_time*.
    3. For each **loss** trade (``pnl < 0``), scan forward for the first
       re-entry trade whose ``entry_time`` falls within *window_days* after
       the loss trade's ``exit_time``.
    4. Each matched pair is emitted as one :class:`WashSaleFlag`.
    """

    def __init__(self, window_days: int = 30) -> None:
        self._window = timedelta(days=window_days)

    def detect(self, trades: list[TradeRecord]) -> list[WashSaleFlag]:
        """
        Scan *trades* and return all detected wash-sale flags.

        Returns an empty list when *trades* is empty or no patterns are found.
        """
        if not trades:
            return []

        # Group by symbol; sort within group by exit_time
        by_symbol: dict[str, list[TradeRecord]] = {}
        for trade in trades:
            by_symbol.setdefault(trade.symbol, []).append(trade)

        flags: list[WashSaleFlag] = []

        for symbol, sym_trades in by_symbol.items():
            sorted_trades = sorted(sym_trades, key=lambda t: t.exit_time)

            for i, loss_trade in enumerate(sorted_trades):
                if loss_trade.pnl >= 0:
                    continue  # only loss trades can trigger a wash sale

                # Search forward for the first re-entry within the window
                for reentry in sorted_trades[i + 1:]:
                    gap = reentry.entry_time - loss_trade.exit_time
                    if gap < timedelta(0):
                        # Entry started before the loss exit (overlapping trades);
                        # skip — not a classic wash-sale re-entry
                        continue
                    if gap <= self._window:
                        flags.append(WashSaleFlag(
                            symbol=symbol,
                            loss_trade=loss_trade,
                            reentry_trade=reentry,
                            days_between=gap.days,
                            disallowed_loss=abs(loss_trade.pnl),
                        ))
                        break  # only flag the first re-entry per loss trade
                    else:
                        break  # trades are sorted; no closer re-entry exists

        return flags
