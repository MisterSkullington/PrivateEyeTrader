"""
ExposureMonitor — real-time portfolio exposure and concentration tracker.

Computes per-bar:
  - Concentration: each position's notional as % of equity
  - Portfolio BTC-beta: weighted beta of all open positions
  - Correlation matrix: log-return correlation between held symbols (when ≥ 2)

The check_signal() method gates proposed new entries against:
  1. Single-symbol concentration ceiling
  2. Total notional exposure ceiling
  3. Portfolio beta ceiling

All checks are disabled when enabled=False (default).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from privateye.core.types import PortfolioState, TradingSignal
from privateye.utils.logging import get_logger

log = get_logger()


@dataclass
class ExposureReport:
    timestamp: datetime
    concentration_by_symbol: dict[str, float]    # symbol → notional/equity
    btc_beta_exposure: float                     # weighted portfolio beta
    correlation_matrix: pd.DataFrame | None      # None when < 2 positions
    max_concentration_pct: float                 # highest single symbol %
    is_over_concentrated: bool
    breach_reason: str = ""                      # empty if compliant


class ExposureMonitor:
    def __init__(
        self,
        enabled: bool = False,
        max_concentration_pct: float = 0.40,
        max_total_notional_pct: float = 0.80,
        max_portfolio_beta: float = 2.0,
        correlation_lookback: int = 60,
        btc_beta_symbols: dict[str, float] | None = None,
    ) -> None:
        self.enabled = enabled
        self.max_concentration_pct = max_concentration_pct
        self.max_total_notional_pct = max_total_notional_pct
        self.max_portfolio_beta = max_portfolio_beta
        self.correlation_lookback = correlation_lookback
        # symbol → beta vs BTC; BTC/USDT always = 1.0; unknowns default to 0.0
        self._btc_beta: dict[str, float] = {"BTC/USDT": 1.0}
        if btc_beta_symbols:
            self._btc_beta.update(btc_beta_symbols)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ExposureMonitor":
        return cls(
            enabled=config.get("enabled", False),
            max_concentration_pct=float(config.get("max_concentration_pct", 0.40)),
            max_total_notional_pct=float(config.get("max_total_notional_pct", 0.80)),
            max_portfolio_beta=float(config.get("max_portfolio_beta", 2.0)),
            correlation_lookback=int(config.get("correlation_lookback", 60)),
            btc_beta_symbols=config.get("btc_beta_symbols", {}),
        )

    def check_signal(
        self,
        signal: TradingSignal,
        portfolio: PortfolioState,
        bars_by_symbol: dict[str, pd.DataFrame],
        proposed_notional: float,
    ) -> tuple[bool, str]:
        """
        Returns (approved: bool, reason: str).
        Always returns (True, "") when enabled=False or portfolio is empty.
        """
        if not self.enabled:
            return True, ""

        equity = portfolio.equity
        if equity <= 0:
            return True, ""

        symbol = signal.symbol

        # Gate 1: Per-symbol concentration after adding proposed position
        existing_notional = 0.0
        if symbol in portfolio.positions:
            pos = portfolio.positions[symbol]
            price = (
                float(bars_by_symbol[symbol]["close"].iloc[-1])
                if symbol in bars_by_symbol and len(bars_by_symbol[symbol]) > 0
                else pos.entry_price
            )
            existing_notional = pos.quantity * price

        new_symbol_notional = existing_notional + proposed_notional
        symbol_concentration = new_symbol_notional / equity
        if symbol_concentration > self.max_concentration_pct:
            return False, (
                f"Concentration breach: {symbol} would reach "
                f"{symbol_concentration:.1%} > {self.max_concentration_pct:.1%} max"
            )

        # Gate 2: Total portfolio notional
        total_notional = self._total_notional(portfolio, bars_by_symbol)
        projected_total = total_notional + proposed_notional
        if projected_total / equity > self.max_total_notional_pct:
            return False, (
                f"Total notional breach: {projected_total / equity:.1%} "
                f"> {self.max_total_notional_pct:.1%} max"
            )

        # Gate 3: Portfolio beta
        if self._btc_beta:
            projected_beta = self._projected_beta(
                signal, proposed_notional, portfolio, bars_by_symbol
            )
            if projected_beta > self.max_portfolio_beta:
                return False, (
                    f"Beta breach: projected portfolio beta {projected_beta:.2f} "
                    f"> {self.max_portfolio_beta:.2f} max"
                )

        return True, ""

    def compute_report(
        self,
        portfolio: PortfolioState,
        bars_by_symbol: dict[str, pd.DataFrame],
    ) -> ExposureReport:
        """Compute a full exposure snapshot for logging / dashboard display."""
        ts = datetime.now(timezone.utc)
        equity = portfolio.equity
        if equity <= 0 or not portfolio.positions:
            return ExposureReport(
                timestamp=ts,
                concentration_by_symbol={},
                btc_beta_exposure=0.0,
                correlation_matrix=None,
                max_concentration_pct=0.0,
                is_over_concentrated=False,
            )

        conc: dict[str, float] = {}
        beta_exposure = 0.0

        for sym, pos in portfolio.positions.items():
            price = (
                float(bars_by_symbol[sym]["close"].iloc[-1])
                if sym in bars_by_symbol and len(bars_by_symbol[sym]) > 0
                else pos.entry_price
            )
            notional = pos.quantity * price
            c = notional / equity
            conc[sym] = c
            beta = self._btc_beta.get(sym, 0.0)
            beta_exposure += c * beta

        max_conc = max(conc.values()) if conc else 0.0
        over_conc = max_conc > self.max_concentration_pct
        breach = (
            f"Symbol concentration {max_conc:.1%} > {self.max_concentration_pct:.1%}"
            if over_conc else ""
        )

        corr_matrix = self._correlation_matrix(portfolio, bars_by_symbol)

        return ExposureReport(
            timestamp=ts,
            concentration_by_symbol=conc,
            btc_beta_exposure=beta_exposure,
            correlation_matrix=corr_matrix,
            max_concentration_pct=max_conc,
            is_over_concentrated=over_conc,
            breach_reason=breach,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _total_notional(
        self,
        portfolio: PortfolioState,
        bars_by_symbol: dict[str, pd.DataFrame],
    ) -> float:
        total = 0.0
        for sym, pos in portfolio.positions.items():
            price = (
                float(bars_by_symbol[sym]["close"].iloc[-1])
                if sym in bars_by_symbol and len(bars_by_symbol[sym]) > 0
                else pos.entry_price
            )
            total += pos.quantity * price
        return total

    def _projected_beta(
        self,
        signal: TradingSignal,
        proposed_notional: float,
        portfolio: PortfolioState,
        bars_by_symbol: dict[str, pd.DataFrame],
    ) -> float:
        equity = portfolio.equity
        if equity <= 0:
            return 0.0
        existing_total = self._total_notional(portfolio, bars_by_symbol)
        new_total = existing_total + proposed_notional
        if new_total <= 0:
            return 0.0

        # Weighted beta of existing positions
        weighted_beta = 0.0
        for sym, pos in portfolio.positions.items():
            price = (
                float(bars_by_symbol[sym]["close"].iloc[-1])
                if sym in bars_by_symbol and len(bars_by_symbol[sym]) > 0
                else pos.entry_price
            )
            notional = pos.quantity * price
            weighted_beta += (notional / new_total) * self._btc_beta.get(sym, 0.0)

        # Add proposed position's contribution
        new_beta = self._btc_beta.get(signal.symbol, 0.0)
        weighted_beta += (proposed_notional / new_total) * new_beta
        return weighted_beta

    def _correlation_matrix(
        self,
        portfolio: PortfolioState,
        bars_by_symbol: dict[str, pd.DataFrame],
    ) -> pd.DataFrame | None:
        symbols = [s for s in portfolio.positions if s in bars_by_symbol]
        if len(symbols) < 2:
            return None

        returns_data: dict[str, pd.Series] = {}
        for sym in symbols:
            bars = bars_by_symbol[sym]
            if len(bars) < self.correlation_lookback + 1:
                continue
            log_ret = np.log(bars["close"] / bars["close"].shift(1)).dropna()
            returns_data[sym] = log_ret.tail(self.correlation_lookback)

        if len(returns_data) < 2:
            return None

        df = pd.DataFrame(returns_data)
        return df.corr()
