"""
Phase 12 test suite — Legal & Ethical Guardrails.

30 tests across:
  TestJurisdictionFilter  (5)  — US/CN/empty/unknown/from_config
  TestSanctionsChecker    (5)  — sanctioned/clean/extra_banned/disabled/case
  TestWashSaleDetector    (8)  — flagged, window, profit, symbol, multi, empty,
                                 disallowed_loss, flag fields
  TestComplianceEngine    (6)  — sanctions block, jurisdiction block, clean,
                                 post_analysis, wash_sale_in_report, disabled
  TestTaxWiring           (3)  — export called, export skipped, engine built
  TestDashboardCompliance (3)  — /api/compliance 200, wash_sale key, no engine
"""
from __future__ import annotations

import csv
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from privateye.compliance.engine import ComplianceEngine, ComplianceReport
from privateye.compliance.jurisdiction import JurisdictionFilter, JURISDICTION_RULES
from privateye.compliance.sanctions import SanctionsChecker
from privateye.compliance.wash_sale import WashSaleDetector, WashSaleFlag
from privateye.core.types import Direction, TradeRecord
from privateye.dashboard.routes import build_router


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dt(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def _trade(
    symbol: str = "BTC/USDT",
    pnl: float = 100.0,
    entry_offset_days: int = 0,
    exit_offset_days: int = 1,
    base_date: datetime | None = None,
    side: Direction = Direction.LONG,
) -> TradeRecord:
    """Build a minimal TradeRecord for testing."""
    base = base_date or _dt(2024, 1, 1)
    entry = base + timedelta(days=entry_offset_days)
    exit_ = base + timedelta(days=exit_offset_days)
    return TradeRecord(
        symbol=symbol,
        side=side,
        entry_price=40_000.0,
        exit_price=40_000.0 + (pnl / 0.1),   # quantity=0.1
        quantity=0.1,
        entry_time=entry,
        exit_time=exit_,
        pnl=pnl,
        pnl_pct=pnl / 4000.0,
        fees=1.0,
        strategy_id="test",
        exit_reason="target",
        bars_held=24,
    )


def _engine(jurisdiction: str = "", sanctions_enabled: bool = True,
            extra_banned: list | None = None,
            tax_enabled: bool = False) -> ComplianceEngine:
    return ComplianceEngine.from_config({
        "jurisdiction": jurisdiction,
        "sanctions": {
            "enabled": sanctions_enabled,
            "extra_banned": extra_banned or [],
        },
        "tax_reporting": {"enabled": tax_enabled},
    })


# ── TestJurisdictionFilter ────────────────────────────────────────────────────

class TestJurisdictionFilter:

    def test_us_blocks_xrp(self):
        """XRP/USDT is a US-restricted symbol (SEC enforcement)."""
        jf = JurisdictionFilter(jurisdiction="US")
        allowed, reason = jf.check("XRP/USDT")
        assert allowed is False
        assert "XRP" in reason or "SEC" in reason

    def test_cn_blocks_all_symbols(self):
        """China jurisdiction blocks every symbol via block_all=True."""
        jf = JurisdictionFilter(jurisdiction="CN")
        for sym in ["BTC/USDT", "ETH/BTC", "SOL/USDT"]:
            allowed, _ = jf.check(sym)
            assert allowed is False, f"CN should block {sym}"

    def test_empty_jurisdiction_allows_everything(self):
        """Disabled (empty) jurisdiction never blocks."""
        jf = JurisdictionFilter(jurisdiction="")
        assert jf.enabled is False
        for sym in ["BTC/USDT", "XRP/USDT", "LUNA/USDT"]:
            allowed, _ = jf.check(sym)
            assert allowed is True

    def test_unknown_jurisdiction_fails_open(self):
        """Unrecognised country codes must not block anything (fail-open)."""
        jf = JurisdictionFilter(jurisdiction="ZZ")
        allowed, reason = jf.check("BTC/USDT")
        assert allowed is True
        assert reason == ""

    def test_from_config_reads_jurisdiction(self):
        jf = JurisdictionFilter.from_config({"jurisdiction": "us"})
        assert jf.jurisdiction == "US"   # upper-cased
        assert jf.enabled is True


# ── TestSanctionsChecker ──────────────────────────────────────────────────────

class TestSanctionsChecker:

    def test_luna_usdt_is_sanctioned(self):
        sc = SanctionsChecker()
        allowed, reason = sc.check("LUNA/USDT")
        assert allowed is False
        assert "sanction" in reason.lower() or "list" in reason.lower()

    def test_btc_usdt_is_not_sanctioned(self):
        sc = SanctionsChecker()
        allowed, reason = sc.check("BTC/USDT")
        assert allowed is True
        assert reason == ""

    def test_extra_banned_symbols_blocked(self):
        """User-supplied extra_banned list is checked alongside built-ins."""
        sc = SanctionsChecker(extra_banned=["EVIL/USDT"])
        allowed, _ = sc.check("EVIL/USDT")
        assert allowed is False

    def test_disabled_checker_allows_everything(self):
        sc = SanctionsChecker(enabled=False)
        for sym in ["LUNA/USDT", "FTT/USDT", "TORN/USDT"]:
            allowed, _ = sc.check(sym)
            assert allowed is True, f"Disabled checker should allow {sym}"

    def test_case_insensitive_match(self):
        """Sanctions check must be case-insensitive."""
        sc = SanctionsChecker()
        for variant in ["luna/usdt", "LUNA/USDT", "Luna/Usdt"]:
            allowed, _ = sc.check(variant)
            assert allowed is False, f"Case variant {variant} should be blocked"


# ── TestWashSaleDetector ──────────────────────────────────────────────────────

class TestWashSaleDetector:

    def test_loss_then_reentry_within_window_is_flagged(self):
        """Loss trade + same-symbol re-entry 10 days later → wash-sale flag."""
        base = _dt(2024, 1, 1)
        loss   = _trade("BTC/USDT", pnl=-200.0, entry_offset_days=0,
                        exit_offset_days=5, base_date=base)
        reentry = _trade("BTC/USDT", pnl=100.0, entry_offset_days=10,
                         exit_offset_days=20, base_date=base)
        flags = WashSaleDetector(window_days=30).detect([loss, reentry])
        assert len(flags) == 1
        assert flags[0].symbol == "BTC/USDT"

    def test_loss_then_reentry_after_window_not_flagged(self):
        """Re-entry 40 days after loss exit is outside the 30-day window."""
        base = _dt(2024, 1, 1)
        loss    = _trade("BTC/USDT", pnl=-200.0, entry_offset_days=0,
                         exit_offset_days=5, base_date=base)
        reentry = _trade("BTC/USDT", pnl=100.0, entry_offset_days=46,
                         exit_offset_days=60, base_date=base)
        flags = WashSaleDetector(window_days=30).detect([loss, reentry])
        assert flags == []

    def test_profit_trade_never_flagged(self):
        """Only loss trades (pnl < 0) can trigger wash-sale flags."""
        base = _dt(2024, 1, 1)
        profit  = _trade("BTC/USDT", pnl=500.0, exit_offset_days=5, base_date=base)
        reentry = _trade("BTC/USDT", pnl=100.0, entry_offset_days=10,
                         exit_offset_days=20, base_date=base)
        flags = WashSaleDetector().detect([profit, reentry])
        assert flags == []

    def test_different_symbol_not_flagged(self):
        """Loss on BTC/USDT + ETH/USDT re-entry → no flag (different symbols)."""
        base = _dt(2024, 1, 1)
        loss    = _trade("BTC/USDT", pnl=-200.0, exit_offset_days=5, base_date=base)
        reentry = _trade("ETH/USDT", pnl=100.0, entry_offset_days=10,
                         exit_offset_days=20, base_date=base)
        flags = WashSaleDetector().detect([loss, reentry])
        assert flags == []

    def test_multiple_flags_detected(self):
        """Multiple independent wash-sale patterns on different symbols."""
        base = _dt(2024, 1, 1)
        trades = [
            _trade("BTC/USDT", pnl=-100.0, exit_offset_days=5,  base_date=base),
            _trade("BTC/USDT", pnl=50.0,   entry_offset_days=10, exit_offset_days=20, base_date=base),
            _trade("ETH/USDT", pnl=-80.0,  exit_offset_days=3,  base_date=base),
            _trade("ETH/USDT", pnl=40.0,   entry_offset_days=8,  exit_offset_days=15, base_date=base),
        ]
        flags = WashSaleDetector(window_days=30).detect(trades)
        assert len(flags) == 2

    def test_empty_trades_returns_no_flags(self):
        flags = WashSaleDetector().detect([])
        assert flags == []

    def test_flag_contains_correct_disallowed_loss(self):
        """disallowed_loss must equal |loss_trade.pnl|."""
        base = _dt(2024, 1, 1)
        loss    = _trade("BTC/USDT", pnl=-350.0, exit_offset_days=5, base_date=base)
        reentry = _trade("BTC/USDT", pnl=100.0,  entry_offset_days=10,
                         exit_offset_days=20, base_date=base)
        flags = WashSaleDetector().detect([loss, reentry])
        assert len(flags) == 1
        assert flags[0].disallowed_loss == pytest.approx(350.0)

    def test_flag_contains_both_trades(self):
        """WashSaleFlag must reference both the loss trade and the re-entry trade."""
        base = _dt(2024, 1, 1)
        loss    = _trade("BTC/USDT", pnl=-100.0, exit_offset_days=5, base_date=base)
        reentry = _trade("BTC/USDT", pnl=60.0,   entry_offset_days=10,
                         exit_offset_days=20, base_date=base)
        flags = WashSaleDetector().detect([loss, reentry])
        assert flags[0].loss_trade is loss
        assert flags[0].reentry_trade is reentry


# ── TestComplianceEngine ──────────────────────────────────────────────────────

class TestComplianceEngine:

    def test_check_symbol_blocks_sanctioned(self):
        eng = _engine(sanctions_enabled=True)
        allowed, reason = eng.check_symbol("LUNA/USDT")
        assert allowed is False
        assert reason != ""

    def test_check_symbol_blocks_jurisdiction_restricted(self):
        eng = _engine(jurisdiction="US", sanctions_enabled=False)
        allowed, reason = eng.check_symbol("XRP/USDT")
        assert allowed is False
        assert "XRP" in reason or "US" in reason

    def test_check_symbol_allows_clean_symbol(self):
        eng = _engine(jurisdiction="US", sanctions_enabled=True)
        allowed, reason = eng.check_symbol("BTC/USDT")
        assert allowed is True
        assert reason == ""

    def test_run_post_analysis_returns_compliance_report(self):
        eng = _engine()
        report = eng.run_post_analysis([])
        assert isinstance(report, ComplianceReport)

    def test_run_post_analysis_includes_wash_sale_flags(self):
        """Wash-sale patterns in trades must appear in post-analysis report."""
        eng = _engine()
        base = _dt(2024, 1, 1)
        loss    = _trade("BTC/USDT", pnl=-200.0, exit_offset_days=5, base_date=base)
        reentry = _trade("BTC/USDT", pnl=100.0,  entry_offset_days=10,
                         exit_offset_days=20, base_date=base)
        report = eng.run_post_analysis([loss, reentry])
        assert len(report.wash_sale_flags) == 1
        assert report.total_disallowed_loss == pytest.approx(200.0)

    def test_disabled_sanctions_allows_everything(self):
        eng = _engine(jurisdiction="", sanctions_enabled=False)
        for sym in ["LUNA/USDT", "FTT/USDT", "XRP/USDT"]:
            allowed, _ = eng.check_symbol(sym)
            assert allowed is True, f"Engine with all disabled should allow {sym}"


# ── TestTaxWiring ─────────────────────────────────────────────────────────────

class TestTaxWiring:

    def test_export_tax_report_writes_csv_when_enabled(self):
        """export_tax_report must create a CSV file when tax_reporting.enabled=true."""
        with tempfile.TemporaryDirectory() as tmpdir:
            eng = ComplianceEngine.from_config({
                "tax_reporting": {
                    "enabled": True,
                    "output_dir": tmpdir,
                },
            })
            trades = [_trade("BTC/USDT", pnl=500.0)]
            path = eng.export_tax_report(trades, "BTC/USDT")
            assert path is not None
            assert Path(path).exists()
            with open(path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 1
            assert "symbol" in rows[0]

    def test_export_tax_report_skipped_when_disabled(self):
        """export_tax_report returns None when tax_reporting.enabled=false."""
        eng = _engine(tax_enabled=False)
        result = eng.export_tax_report([_trade()], "BTC/USDT")
        assert result is None

    def test_compliance_engine_built_from_cfg_compliance_block(self):
        """_build_compliance_engine reads cfg['compliance'] and returns engine."""
        from privateye.main import _build_compliance_engine
        cfg = {
            "compliance": {
                "jurisdiction": "US",
                "sanctions": {"enabled": True, "extra_banned": []},
                "tax_reporting": {"enabled": False},
            }
        }
        engine = _build_compliance_engine(cfg)
        assert isinstance(engine, ComplianceEngine)
        # US jurisdiction should block XRP
        allowed, _ = engine.check_symbol("XRP/USDT")
        assert allowed is False


# ── TestDashboardCompliance ───────────────────────────────────────────────────

def _build_test_client(get_compliance=None) -> TestClient:
    app = FastAPI()
    mock_portfolio = MagicMock()
    mock_portfolio.equity = 10000.0
    mock_portfolio.cash = 10000.0
    mock_portfolio.invested = 0.0
    mock_portfolio.daily_pnl = 0.0
    mock_portfolio.daily_drawdown_pct = 0.0
    mock_portfolio.drawdown_pct = 0.0
    mock_portfolio.peak_equity = 10000.0
    mock_portfolio.total_trades = 0
    mock_portfolio.positions = {}
    mock_rm = MagicMock()
    mock_rm.is_halted.return_value = False
    mock_rm._halt_reason = None
    router = build_router(
        get_portfolio=lambda: mock_portfolio,
        get_trades=lambda: [],
        get_fills=lambda: [],
        exec_engine=MagicMock(),
        risk_manager=mock_rm,
        get_compliance=get_compliance,
    )
    app.include_router(router)
    return TestClient(app)


class TestDashboardCompliance:

    def test_get_compliance_returns_200(self):
        eng = _engine(jurisdiction="US")
        client = _build_test_client(get_compliance=eng.get_status)
        resp = client.get("/api/compliance")
        assert resp.status_code == 200
        data = resp.json()
        assert data["jurisdiction"] == "US"
        assert data["sanctions_enabled"] is True

    def test_get_compliance_contains_wash_sale_flags_key(self):
        eng = _engine()
        client = _build_test_client(get_compliance=eng.get_status)
        resp = client.get("/api/compliance")
        data = resp.json()
        assert "wash_sale_flags" in data
        assert isinstance(data["wash_sale_flags"], list)

    def test_no_compliance_engine_returns_empty_structure(self):
        """When get_compliance is None, endpoint returns safe empty defaults."""
        client = _build_test_client(get_compliance=None)
        resp = client.get("/api/compliance")
        assert resp.status_code == 200
        data = resp.json()
        assert data["jurisdiction"] is None
        assert data["sanctions_enabled"] is False
        assert data["blocked_symbols"] == []
        assert data["wash_sale_flags"] == []
