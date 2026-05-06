"""
ComplianceEngine — central compliance gate for PrivateEyeTrader.

Combines :class:`JurisdictionFilter`, :class:`SanctionsChecker`, and
:class:`WashSaleDetector` into a single facade used by the trading engine.

**Pre-trade check** (called inside ``on_signal`` handlers)::

    allowed, reason = compliance_engine.check_symbol(signal.symbol)
    if not allowed:
        log.warning(f"[Compliance] Blocked: {reason}")
        return

**Post-trade analysis** (called after :meth:`BacktestEngine.run`)::

    report = compliance_engine.run_post_analysis(backtest_report.trades)
    print(report)
    compliance_engine.export_tax_report(backtest_report.trades, symbol)

**Dashboard status**::

    get_compliance=compliance_engine.get_status   # passed to build_router()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from privateye.compliance.jurisdiction import JurisdictionFilter
from privateye.compliance.sanctions import SanctionsChecker
from privateye.compliance.wash_sale import WashSaleDetector, WashSaleFlag
from privateye.utils.logging import get_logger

if TYPE_CHECKING:
    from privateye.core.types import TradeRecord

log = get_logger()


@dataclass
class ComplianceReport:
    """Summary of a post-trade compliance analysis run."""

    jurisdiction:          str
    sanctions_enabled:     bool
    blocked_symbols:       list[str]          = field(default_factory=list)
    wash_sale_flags:       list[WashSaleFlag] = field(default_factory=list)
    total_disallowed_loss: float              = 0.0

    def __str__(self) -> str:  # noqa: D105
        sep  = "=" * 58
        thin = "-" * 58
        lines = [
            sep,
            "  COMPLIANCE REPORT",
            sep,
            f"  Jurisdiction      : {self.jurisdiction or 'None (disabled)'}",
            f"  Sanctions enabled : {self.sanctions_enabled}",
            f"  Blocked symbols   : {len(self.blocked_symbols)}",
            f"  Wash-sale flags   : {len(self.wash_sale_flags)}",
            f"  Total disallowed  : {self.total_disallowed_loss:+.2f}",
        ]
        if self.wash_sale_flags:
            lines += [thin, "  WASH-SALE FLAGS"]
            for flag in self.wash_sale_flags:
                lines.append(
                    f"  {flag.symbol:<14}  "
                    f"loss={flag.disallowed_loss:+.2f}  "
                    f"gap={flag.days_between}d"
                )
        lines.append(sep)
        return "\n".join(lines)


class ComplianceEngine:
    """
    Central compliance gate: jurisdiction + sanctions pre-trade, wash-sale
    detection + tax export post-trade.
    """

    def __init__(
        self,
        jurisdiction_filter: JurisdictionFilter,
        sanctions_checker:   SanctionsChecker,
        wash_sale_detector:  WashSaleDetector,
        tax_enabled:         bool = False,
        tax_output_dir:      str  = "data/reports",
        tax_method:          str  = "fifo",
    ) -> None:
        self._jf             = jurisdiction_filter
        self._sc             = sanctions_checker
        self._wsd            = wash_sale_detector
        self._tax_enabled    = tax_enabled
        self._tax_output_dir = Path(tax_output_dir)
        self._tax_method     = tax_method
        self._blocked:  list[str] = []   # in-session audit log of blocked symbols

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ComplianceEngine":
        """Build from the ``compliance:`` section of ``settings.yaml``."""
        tax_cfg = cfg.get("tax_reporting", {})
        return cls(
            jurisdiction_filter=JurisdictionFilter.from_config(cfg),
            sanctions_checker=SanctionsChecker.from_config(
                cfg.get("sanctions", {})
            ),
            wash_sale_detector=WashSaleDetector(
                window_days=int(tax_cfg.get("wash_sale_window_days", 30))
            ),
            tax_enabled=bool(tax_cfg.get("enabled", False)),
            tax_output_dir=str(tax_cfg.get("output_dir", "data/reports")),
            tax_method=str(tax_cfg.get("method", "fifo")),
        )

    # ── Pre-trade ─────────────────────────────────────────────────────────────

    def check_symbol(self, symbol: str) -> tuple[bool, str]:
        """
        Gate a prospective trade on *symbol*.

        Sanctions are checked first (highest priority), then jurisdiction.
        Returns ``(allowed: bool, reason: str)``.
        When ``allowed=False`` the caller must suppress the signal.
        """
        allowed, reason = self._sc.check(symbol)
        if not allowed:
            self._blocked.append(symbol)
            log.warning(f"[Compliance] Blocked — {reason}")
            return False, reason

        allowed, reason = self._jf.check(symbol)
        if not allowed:
            self._blocked.append(symbol)
            log.warning(f"[Compliance] Blocked — {reason}")
            return False, reason

        return True, ""

    # ── Post-trade ────────────────────────────────────────────────────────────

    def run_post_analysis(
        self, trades: "list[TradeRecord]"
    ) -> ComplianceReport:
        """
        Run wash-sale detection on a completed trade list and return a
        :class:`ComplianceReport`.
        """
        flags           = self._wsd.detect(trades)
        total_disallowed = sum(f.disallowed_loss for f in flags)
        return ComplianceReport(
            jurisdiction=self._jf.jurisdiction,
            sanctions_enabled=self._sc.enabled,
            blocked_symbols=list(self._blocked),
            wash_sale_flags=flags,
            total_disallowed_loss=total_disallowed,
        )

    def export_tax_report(
        self,
        trades: "list[TradeRecord]",
        symbol: str,
        year:   int | None = None,
    ) -> Path | None:
        """
        Export a tax CSV for *trades* when ``tax_reporting.enabled=true``.

        Creates ``<tax_output_dir>/tax_<symbol>_<year>.csv``.
        Returns the CSV :class:`~pathlib.Path`, or ``None`` when disabled or
        when *trades* is empty.
        """
        if not self._tax_enabled or not trades:
            return None

        from privateye.reports.tax import compute_tax_lots, export_tax_csv

        tax_report = compute_tax_lots(trades, year=year)
        self._tax_output_dir.mkdir(parents=True, exist_ok=True)
        safe_symbol = symbol.replace("/", "_")
        suffix      = f"_{year}" if year else ""
        path        = self._tax_output_dir / f"tax_{safe_symbol}{suffix}.csv"
        export_tax_csv(tax_report, path)
        log.info(f"[Compliance] Tax CSV exported → {path}")
        return path

    # ── Dashboard status ──────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """
        Return a JSON-serialisable status dict suitable for the dashboard
        ``/api/compliance`` endpoint.
        """
        return {
            "jurisdiction":      self._jf.jurisdiction or None,
            "sanctions_enabled": self._sc.enabled,
            "blocked_symbols":   list(self._blocked),
            "wash_sale_flags":   [],   # populated via run_post_analysis()
            "total_disallowed_loss": 0.0,
        }
