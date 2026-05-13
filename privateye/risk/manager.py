"""
Risk manager — the last gate before an order reaches the exchange.

Gate order (entry signals):
 12. portfolio circuit breaker → reject when portfolio DD ≥ threshold (Phase 5)
  1. Direction.FLAT      → always approved (exit)
  2. restricted_assets   → reject
  3. _halted             → reject
  4. daily drawdown      → halt + reject
  5. confidence          → reject
  6. AssetFilter         → reject (skipped when filter=None or bars=None)
  7. _compute_size()     → qty via SizingRouter (uses PortfolioOptimizer cap if set)
  8. ExposureMonitor     → reject (skipped when monitor=None)
  9. cash availability   → reject
 10. existing same-side  → reject
 11. PreTradeCostAnalyzer → reject when net_edge < min_net_edge_pct (Phase 3)
  → build Order (respects preferred_order_type from signal.metadata)

Returns (approved: bool, reason: str, order_if_approved: Order | None).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from privateye.core.exceptions import DailyDrawdownBreachedError, RiskLimitBreachedError
from privateye.core.types import (
    Direction, Order, OrderSide, OrderStatus, OrderType,
    PortfolioState, TradingSignal, TradeRecord,
)
from privateye.risk.sizing import cap_to_max_notional
from privateye.risk.sizing_router import SizingMethod, route_sizing
from privateye.utils.logging import get_logger

log = get_logger()


class RiskManager:
    def __init__(
        self,
        config: dict[str, Any],
        exposure_monitor: Any = None,       # ExposureMonitor | None
        asset_filter: Any = None,           # AssetFilter | None
        pre_trade_analyzer: Any = None,     # PreTradeCostAnalyzer | None  (Phase 3 Gate 11)
        portfolio_optimizer: Any = None,    # PortfolioOptimizer | None  (Phase 5 Gate 12)
    ) -> None:
        self._config = config
        self.max_risk_pct: float     = config.get("max_risk_per_trade_pct", 0.01)
        self.max_daily_dd_pct: float = config.get("max_daily_drawdown_pct", 0.05)
        self.max_notional_pct: float = config.get("max_position_notional_pct", 0.20)
        self.min_confidence: float   = config.get("min_confidence", 0.55)
        # Phase 13 (M-5): cash buffer is configurable. Default 1% (was hardcoded).
        self.cash_buffer_pct: float  = config.get("cash_buffer_pct", 0.01)
        # Phase 13 (H-2): SHORT signals on spot exchanges are infeasible by default.
        # Set ``allow_spot_short: true`` in risk config to override (e.g. with margin).
        self.allow_spot_short: bool  = config.get("allow_spot_short", False)
        self.exchange_default_type: str = config.get("exchange_default_type", "spot")
        self._restricted: set[str]   = set(config.get("restricted_assets", []))
        self._halted: bool           = False
        self._halt_reason: str       = ""

        # Phase 5 components (all optional / disabled by default)
        self._exposure_monitor = exposure_monitor
        self._asset_filter = asset_filter
        self.sizing_method = SizingMethod(config.get("sizing_method", "fixed_risk"))

        # Phase 3 Gate 11: Pre-trade cost analysis (disabled by default — pass analyzer to enable)
        self._pre_trade_analyzer = pre_trade_analyzer

        # Phase 5 Gate 12: Portfolio-level daily drawdown circuit breaker
        self._portfolio_dd_halt_pct: float = float(config.get("portfolio_dd_halt_pct", 0.0))
        self._portfolio_daily_hwm: float   = float("-inf")
        self._portfolio_circuit_open: bool = False
        self._portfolio_optimizer = portfolio_optimizer

    # ── Public interface ────────────────────────────────────────────────────

    def evaluate_signal(
        self,
        signal: TradingSignal,
        portfolio: PortfolioState,
        bars: pd.DataFrame | None = None,
        bars_by_symbol: dict[str, pd.DataFrame] | None = None,
        trade_history: list[TradeRecord] | None = None,
    ) -> tuple[bool, str, Order | None]:
        """
        Evaluate a TradingSignal.
        Returns (approved, reason, order_if_approved).

        New kwargs (all backward-compatible — default None skips Phase 5 gates):
          bars           — current symbol's OHLCV window (for AssetFilter + sizing)
          bars_by_symbol — dict[symbol → bars] (for ExposureMonitor)
          trade_history  — list[TradeRecord] (for Kelly sizing)
        """
        # Gate 12 (Phase 5): Portfolio-level daily drawdown circuit breaker.
        # Tracks a portfolio-wide HWM; if equity drops ≥ portfolio_dd_halt_pct from
        # that HWM, ALL new entries are blocked (not just the triggering symbol).
        # Set portfolio_dd_halt_pct = 0.0 to disable (default).
        if self._portfolio_dd_halt_pct > 0:
            equity = portfolio.equity
            if equity > self._portfolio_daily_hwm:
                self._portfolio_daily_hwm = equity
            dd = 0.0
            if self._portfolio_daily_hwm > 0:
                dd = (self._portfolio_daily_hwm - equity) / self._portfolio_daily_hwm
                if dd >= self._portfolio_dd_halt_pct:
                    self._portfolio_circuit_open = True
            if self._portfolio_circuit_open:
                return False, (
                    f"Portfolio circuit breaker: DD={dd:.1%} ≥ {self._portfolio_dd_halt_pct:.1%}"
                ), None

        # Gate 1: FLAT signals are always approved (exits)
        if signal.direction == Direction.FLAT:
            order = self._build_exit_order(signal, portfolio)
            return True, "exit", order

        # Gate 2: Restricted assets — block all entry signals
        if signal.symbol in self._restricted:
            return False, f"Asset {signal.symbol} is in restricted_assets list", None

        # Gate 2.5 (H-2): SHORT feasibility — spot exchanges don't support shorts
        # without a margin account. Reject SHORT signals unless explicitly allowed.
        if signal.direction == Direction.SHORT:
            if self.exchange_default_type == "spot" and not self.allow_spot_short:
                return False, (
                    "SHORT signal on spot exchange — set risk.allow_spot_short=true "
                    "or risk.exchange_default_type='margin' to enable"
                ), None

        # Gate 3: Check halt
        if self._halted:
            return False, f"Trading halted: {self._halt_reason}", None

        # Gate 4: Check daily drawdown
        if portfolio.daily_drawdown_pct >= self.max_daily_dd_pct:
            self._halt("daily drawdown limit reached")
            return False, "Daily drawdown circuit breaker triggered", None

        # Gate 5: Confidence filter
        if signal.confidence < self.min_confidence:
            return False, f"Confidence {signal.confidence:.2f} < min {self.min_confidence}", None

        # Gate 6 (Phase 5): Asset filter
        if self._asset_filter is not None and bars is not None:
            tradeable, filter_reason = self._asset_filter.is_tradeable(signal.symbol, bars)
            if not tradeable:
                return False, f"AssetFilter: {filter_reason}", None

        # Gate 7: Compute position size (Phase 5: passes bars_by_symbol for optimizer cap)
        qty = self._compute_size(signal, portfolio, bars, trade_history, bars_by_symbol)

        if qty <= 0:
            return False, "Computed quantity is zero (stop too close or equity too low)", None

        # Gate 8 (Phase 5): Exposure concentration check
        if self._exposure_monitor is not None:
            proposed_notional = qty * signal.entry_price
            bbs = bars_by_symbol or (
                {signal.symbol: bars} if bars is not None else {}
            )
            ok, exposure_reason = self._exposure_monitor.check_signal(
                signal, portfolio, bbs, proposed_notional
            )
            if not ok:
                return False, f"ExposureMonitor: {exposure_reason}", None

        # Gate 9: Check cash availability — M-5: configurable buffer (default 1%)
        required_cash = qty * signal.entry_price
        max_spend = portfolio.cash * (1.0 - self.cash_buffer_pct)
        if required_cash > max_spend:
            return False, (
                f"Insufficient cash: need {required_cash:.2f}, have {portfolio.cash:.2f} "
                f"(buffer={self.cash_buffer_pct * 100:.1f}%)"
            ), None

        # Gate 10: Check existing position in this symbol
        if signal.symbol in portfolio.positions:
            existing = portfolio.positions[signal.symbol]
            if existing.side == signal.direction:
                return False, f"Already in {signal.direction.value} position for {signal.symbol}", None

        # Gate 11 (Phase 3): Pre-trade cost analysis — reject when net edge is too thin
        if self._pre_trade_analyzer is not None:
            result = self._pre_trade_analyzer.analyze(signal, qty, portfolio, bars)
            if not result.approved:
                log.info(
                    f"[RiskManager] Gate 11 REJECTED {signal.symbol}: {result.reason}"
                )
                return False, f"Pre-trade cost screen: {result.reason}", None

        order = self._build_entry_order(signal, qty)
        log.info(
            f"[RiskManager] APPROVED {signal.direction.value} {signal.symbol} "
            f"qty={qty:.6f} entry={signal.entry_price:.2f} stop={signal.stop_price:.2f} "
            f"risk={qty * abs(signal.entry_price - signal.stop_price):.2f} "
            f"confidence={signal.confidence:.2f}"
        )
        return True, "approved", order

    def update_portfolio(self, portfolio: PortfolioState) -> None:
        """Called after each bar/fill to check ongoing risk limits."""
        if portfolio.daily_drawdown_pct >= self.max_daily_dd_pct and not self._halted:
            self._halt("daily drawdown limit reached during session")

    def halt(self, reason: str) -> None:
        self._halt(reason)

    def resume(self) -> None:
        self._halted = False
        self._halt_reason = ""
        log.info("[RiskManager] Trading resumed")

    def is_halted(self) -> bool:
        return self._halted

    def reset_portfolio_daily_hwm(self) -> None:
        """Reset the portfolio-level daily high-water mark and clear the circuit breaker.

        Call at midnight (same pattern as the per-symbol daily DD reset in SimulatedExchange)
        so the circuit resets each trading day.
        """
        self._portfolio_daily_hwm = float("-inf")
        self._portfolio_circuit_open = False
        log.info("[RiskManager] Portfolio daily HWM reset")

    # ── Internal ────────────────────────────────────────────────────────────

    def _halt(self, reason: str) -> None:
        if not self._halted:
            self._halted = True
            self._halt_reason = reason
            log.critical(f"[RiskManager] HALT — {reason}")

    def _compute_size(
        self,
        signal: TradingSignal,
        portfolio: PortfolioState,
        bars: pd.DataFrame | None,
        trade_history: list[TradeRecord] | None,
        bars_by_symbol: dict[str, pd.DataFrame] | None = None,
    ) -> float:
        """Route to the configured sizing algorithm and apply notional cap.

        Phase 5: when a PortfolioOptimizer is set and bars_by_symbol is provided,
        the effective notional cap is min(config_max_pct, optimizer_weight[symbol]).
        This is a soft cap — the absolute config limit always wins.
        """
        qty = route_sizing(
            method=self.sizing_method,
            equity=portfolio.equity,
            config=self._config,
            signal=signal,
            bars=bars,
            trade_history=trade_history,
        )

        # Phase 5: optimizer-aware notional cap
        if self._portfolio_optimizer is not None and bars_by_symbol:
            weights = self._portfolio_optimizer.compute_weights(
                list(bars_by_symbol.keys()), bars_by_symbol
            )
            optimizer_cap = weights.get(signal.symbol, 1.0)
            effective_max_pct = min(self.max_notional_pct, optimizer_cap)
        else:
            effective_max_pct = self.max_notional_pct

        return cap_to_max_notional(qty, signal.entry_price, portfolio.equity, effective_max_pct)

    def _build_entry_order(self, signal: TradingSignal, qty: float) -> Order:
        side = OrderSide.BUY if signal.direction == Direction.LONG else OrderSide.SELL
        # Honor preferred order type set by FeeOptimizer (stored in signal metadata)
        order_type_str = signal.metadata.get("preferred_order_type", "market")
        try:
            order_type = OrderType(order_type_str)
        except ValueError:
            order_type = OrderType.MARKET
        return Order(
            symbol=signal.symbol,
            side=side,
            order_type=order_type,
            quantity=qty,
            price=signal.entry_price,
            stop_price=signal.stop_price,
            strategy_id=signal.strategy_id,
        )

    def _build_exit_order(self, signal: TradingSignal, portfolio: PortfolioState) -> Order | None:
        pos = portfolio.positions.get(signal.symbol)
        if not pos:
            return None
        exit_side = OrderSide.SELL if pos.side == Direction.LONG else OrderSide.BUY
        # M-4: market exits don't have a meaningful limit price — set to 0.0 so
        # ccxt doesn't validate it and the simulator uses bar.close + slippage.
        return Order(
            symbol=signal.symbol,
            side=exit_side,
            order_type=OrderType.MARKET,
            quantity=pos.quantity,
            price=0.0,
            stop_price=0.0,
            strategy_id=signal.strategy_id,
        )
