"""
Risk manager — the last gate before an order reaches the exchange.

Gate order (entry signals):
  1. Direction.FLAT      → always approved (exit)
  2. restricted_assets   → reject
  3. _halted             → reject
  4. daily drawdown      → halt + reject
  5. confidence          → reject
  6. AssetFilter         → reject (skipped when filter=None or bars=None)
  7. _compute_size()     → qty via SizingRouter
  8. ExposureMonitor     → reject (skipped when monitor=None)
  9. cash availability   → reject
 10. existing same-side  → reject
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
        exposure_monitor: Any = None,   # ExposureMonitor | None
        asset_filter: Any = None,       # AssetFilter | None
    ) -> None:
        self._config = config
        self.max_risk_pct: float     = config.get("max_risk_per_trade_pct", 0.01)
        self.max_daily_dd_pct: float = config.get("max_daily_drawdown_pct", 0.05)
        self.max_notional_pct: float = config.get("max_position_notional_pct", 0.20)
        self.min_confidence: float   = config.get("min_confidence", 0.55)
        self._restricted: set[str]   = set(config.get("restricted_assets", []))
        self._halted: bool           = False
        self._halt_reason: str       = ""

        # Phase 5 components (all optional / disabled by default)
        self._exposure_monitor = exposure_monitor
        self._asset_filter = asset_filter
        self.sizing_method = SizingMethod(config.get("sizing_method", "fixed_risk"))

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
        # Gate 1: FLAT signals are always approved (exits)
        if signal.direction == Direction.FLAT:
            order = self._build_exit_order(signal, portfolio)
            return True, "exit", order

        # Gate 2: Restricted assets — block all entry signals
        if signal.symbol in self._restricted:
            return False, f"Asset {signal.symbol} is in restricted_assets list", None

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

        # Gate 7: Compute position size
        qty = self._compute_size(signal, portfolio, bars, trade_history)

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

        # Gate 9: Check cash availability
        required_cash = qty * signal.entry_price
        if required_cash > portfolio.cash * 0.99:  # 1% buffer
            return False, f"Insufficient cash: need {required_cash:.2f}, have {portfolio.cash:.2f}", None

        # Gate 10: Check existing position in this symbol
        if signal.symbol in portfolio.positions:
            existing = portfolio.positions[signal.symbol]
            if existing.side == signal.direction:
                return False, f"Already in {signal.direction.value} position for {signal.symbol}", None

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
    ) -> float:
        """Route to the configured sizing algorithm and apply notional cap."""
        qty = route_sizing(
            method=self.sizing_method,
            equity=portfolio.equity,
            config=self._config,
            signal=signal,
            bars=bars,
            trade_history=trade_history,
        )
        return cap_to_max_notional(qty, signal.entry_price, portfolio.equity, self.max_notional_pct)

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
        return Order(
            symbol=signal.symbol,
            side=exit_side,
            order_type=OrderType.MARKET,
            quantity=pos.quantity,
            price=signal.entry_price,  # current close
            stop_price=0.0,
            strategy_id=signal.strategy_id,
        )
