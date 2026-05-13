"""
PortfolioBacktestEngine — time-synchronized multi-symbol portfolio backtest.

All symbols share a single SimulatedExchange so cash depletion from one
symbol constrains entries in all others, and the portfolio equity curve
reflects true cross-symbol interaction.

Usage:
    from privateye.backtesting.portfolio_engine import PortfolioBacktestEngine

    engine = PortfolioBacktestEngine(cfg)
    report = engine.run(
        bars_by_symbol={"BTC/USDT": btc_df, "ETH/USDT": eth_df},
        timeframe="1h",
    )
    print(report)

The engine uses the same deferred-import pattern as WalkForwardEngine._run_fold()
to avoid circular dependencies and ensure fresh strategy/risk instances.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from privateye.backtesting.metrics import (
    BacktestReport,
    compute_metrics,
    timeframe_bars_per_year,
)
from privateye.core.types import DataSnapshot, Direction, TradingSignal
from privateye.utils.logging import get_logger
from privateye.utils.time import now_utc

log = get_logger()


@dataclass
class PortfolioBacktestReport:
    """Aggregate results from a multi-symbol portfolio backtest."""
    symbols: list[str] = field(default_factory=list)
    timeframe: str = "1h"
    n_bars_total: int = 0               # total bar-symbol observations replayed
    per_symbol_reports: dict[str, BacktestReport] = field(default_factory=dict)
    aggregate_equity_curve: list[float] = field(default_factory=list)
    aggregate_pnl: float = 0.0          # sum of per-symbol total_pnl
    aggregate_sharpe: float = 0.0       # Sharpe on aggregate equity curve
    aggregate_max_dd: float = 0.0       # worst peak-to-trough on aggregate curve
    total_trades: int = 0               # sum across all symbols

    def __str__(self) -> str:
        sep = "=" * 58
        lines = [
            sep,
            "  PORTFOLIO BACKTEST REPORT",
            sep,
            f"  Symbols         : {', '.join(self.symbols)}",
            f"  Timeframe       : {self.timeframe}",
            f"  Total Trades    : {self.total_trades}",
            f"  Aggregate PnL   : {self.aggregate_pnl:+.2f}",
            f"  Aggregate Sharpe: {self.aggregate_sharpe:.2f}",
            f"  Aggregate Max DD: {self.aggregate_max_dd * 100:.1f}%",
            sep,
            "  PER-SYMBOL SUMMARY",
            sep,
        ]
        for sym, r in self.per_symbol_reports.items():
            lines.append(
                f"  {sym:12s}  trades={r.total_trades:4d}  "
                f"pnl={r.total_pnl:+9.2f}  "
                f"sharpe={r.sharpe_ratio:5.2f}  "
                f"maxDD={r.max_drawdown_pct * 100:5.1f}%"
            )
        lines.append(sep)
        return "\n".join(lines)


class PortfolioBacktestEngine:
    """
    Multi-symbol portfolio backtest with time-synchronized bar replay.

    All symbols share one SimulatedExchange — entering BTC reduces available
    cash for ETH entries. Per-symbol BacktestReports are derived by filtering
    the shared exchange's trade_records after the joint replay loop.
    """

    def __init__(self, cfg: dict[str, Any], black_swan_guard: Any = None) -> None:
        self._cfg = cfg
        self._black_swan_guard = black_swan_guard

    def run(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        timeframe: str,
    ) -> PortfolioBacktestReport:
        """
        Replay all symbols in time-synchronized lock-step.

        Args:
            bars_by_symbol: dict mapping symbol → OHLCV DataFrame.
                            Each DataFrame must have a 'timestamp' column.
            timeframe: bar interval string (e.g. "1h").

        Returns:
            PortfolioBacktestReport with per-symbol and aggregate metrics.
        """
        symbols = list(bars_by_symbol.keys())
        if not symbols:
            return PortfolioBacktestReport(symbols=[], timeframe=timeframe)

        # Deferred imports to avoid circular dependencies
        from privateye.backtesting.simulator import SimulatedExchange
        from privateye.risk.manager import RiskManager

        bt_cfg = self._cfg.get("backtesting", {})
        initial_capital: float = float(bt_cfg.get("initial_capital", 10000.0))
        bar_window: int = int(self._cfg.get("data", {}).get("bar_window", 500))
        min_warmup: int = max(200, bar_window // 2)

        # Single shared exchange for all symbols
        exchange = SimulatedExchange(
            initial_capital=initial_capital,
            fee_maker=float(bt_cfg.get("fee_maker", 0.001)),
            fee_taker=float(bt_cfg.get("fee_taker", 0.001)),
            slippage_pct=float(bt_cfg.get("slippage_pct", 0.0005)),
            max_fill_pct_of_volume=float(bt_cfg.get("max_fill_pct_of_volume", 0.30)),
        )

        # Build strategies and risk manager
        strategies = self._build_strategies()
        risk_manager = self._build_risk_manager()

        # Build the sorted union of all timestamps across all symbols
        all_timestamps: list[Any] = sorted(
            set(
                ts
                for bars in bars_by_symbol.values()
                for ts in bars["timestamp"].tolist()
            )
        )

        # For each symbol, build a position-indexed lookup: timestamp → row index
        symbol_ts_index: dict[str, dict[Any, int]] = {}
        for sym, bars in bars_by_symbol.items():
            symbol_ts_index[sym] = {
                ts: i for i, ts in enumerate(bars["timestamp"].tolist())
            }

        aggregate_equity_curve: list[float] = []
        # Per-symbol bar counter for warmup tracking
        symbol_bar_counts: dict[str, int] = {s: 0 for s in symbols}
        n_bars_total: int = 0

        for ts in all_timestamps:
            # Process every symbol that has a bar at this timestamp
            for sym in symbols:
                ts_map = symbol_ts_index[sym]
                if ts not in ts_map:
                    continue  # this symbol has no bar at this timestamp

                bar_idx = ts_map[ts]
                bars = bars_by_symbol[sym]
                bar = bars.iloc[bar_idx].to_dict()
                symbol_bar_counts[sym] += 1
                n_bars_total += 1

                # Warmup: just process the bar without strategy evaluation
                if symbol_bar_counts[sym] <= min_warmup:
                    exchange.process_bar(sym, bar)
                    continue

                # Build DataSnapshot with rolling window
                window_start = max(0, bar_idx + 1 - bar_window)
                snapshot_bars = bars.iloc[window_start: bar_idx + 1].reset_index(drop=True)
                snapshot = DataSnapshot(
                    symbol=sym,
                    timeframe=timeframe,
                    bars=snapshot_bars,
                    timestamp=now_utc(),
                )

                portfolio = exchange.get_portfolio_state()
                risk_manager.update_portfolio(portfolio)

                # Black-swan guard (optional)
                if self._black_swan_guard is not None and self._black_swan_guard.enabled:
                    funding = float(bar.get("funding_rate", 0.0))
                    event = self._black_swan_guard.check(sym, snapshot_bars, funding_rate=funding)
                    if event is not None:
                        log.warning(
                            f"[PortfolioBacktest] BlackSwan {event.trigger.value} on {sym}"
                        )
                        risk_manager.halt(f"black_swan:{event.trigger.value}")
                        portfolio = exchange.get_portfolio_state()
                        close_price = float(bar["close"])
                        for pos_sym, pos in list(portfolio.positions.items()):
                            flat_sig = TradingSignal(
                                symbol=pos_sym,
                                direction=Direction.FLAT,
                                confidence=1.0,
                                entry_price=close_price,
                                stop_price=0.0,
                                target_price=0.0,
                                strategy_id="black_swan_guard",
                                timeframe=timeframe,
                            )
                            _, _, flat_order = risk_manager.evaluate_signal(
                                flat_sig, portfolio
                            )
                            if flat_order:
                                exchange.submit_order(flat_order)
                        exchange.process_bar(sym, bar)
                        continue

                # Entry signals
                trade_history = exchange.trade_records
                bars_by_sym_snapshot = {sym: snapshot_bars}
                for strategy in strategies:
                    signals = strategy.on_data(snapshot)
                    for signal in signals:
                        approved, reason, order = risk_manager.evaluate_signal(
                            signal,
                            portfolio,
                            bars=snapshot_bars,
                            bars_by_symbol=bars_by_sym_snapshot,
                            trade_history=trade_history,
                        )
                        if approved and order:
                            exchange.submit_order(order)
                        elif not approved:
                            log.debug(f"[PortfolioBacktest] {sym} signal rejected: {reason}")

                # Fill pending orders against current bar
                fills = exchange.process_bar(sym, bar)
                for fill in fills:
                    for strategy in strategies:
                        if strategy.strategy_id == fill.strategy_id:
                            strategy.on_fill(fill, portfolio)

                # Exit / trailing-stop signals
                portfolio = exchange.get_portfolio_state()
                for strategy in strategies:
                    exit_signals = strategy.on_bar_end(snapshot, portfolio)
                    for signal in exit_signals:
                        if signal.direction == Direction.FLAT:
                            approved, _, order = risk_manager.evaluate_signal(
                                signal, portfolio,
                                bars=snapshot_bars,
                                bars_by_symbol=bars_by_sym_snapshot,
                            )
                            if approved and order:
                                exchange.submit_order(order)
                                exit_fills = exchange.process_bar(sym, bar)
                                for fill in exit_fills:
                                    for s in strategies:
                                        if s.strategy_id == fill.strategy_id:
                                            s.on_fill(fill, portfolio)

            # Record portfolio equity after processing all symbols at this timestamp
            aggregate_equity_curve.append(exchange.equity)

        # ── Compute per-symbol and aggregate reports ─────────────────────────
        per_symbol_reports: dict[str, BacktestReport] = {}
        bpy = timeframe_bars_per_year(timeframe)

        for sym in symbols:
            sym_trades = [t for t in exchange.trade_records if t.symbol == sym]
            # Build a synthetic equity curve for this symbol:
            # start at initial_capital/n, add each trade's PnL in order
            per_sym_initial = initial_capital / len(symbols)
            sym_equity: list[float] = [per_sym_initial]
            running = per_sym_initial
            for t in sorted(sym_trades, key=lambda x: x.entry_time):
                running += t.pnl
                sym_equity.append(running)

            per_symbol_reports[sym] = compute_metrics(
                trades=sym_trades,
                equity_curve=sym_equity,
                initial_capital=per_sym_initial,
                bars_per_year=bpy,
            )

        # Aggregate Sharpe on the combined equity curve
        aggregate_sharpe = _compute_sharpe(aggregate_equity_curve, bpy)
        aggregate_max_dd = _compute_max_dd(aggregate_equity_curve)

        return PortfolioBacktestReport(
            symbols=symbols,
            timeframe=timeframe,
            n_bars_total=n_bars_total,
            per_symbol_reports=per_symbol_reports,
            aggregate_equity_curve=aggregate_equity_curve,
            aggregate_pnl=sum(r.total_pnl for r in per_symbol_reports.values()),
            aggregate_sharpe=aggregate_sharpe,
            aggregate_max_dd=aggregate_max_dd,
            total_trades=sum(r.total_trades for r in per_symbol_reports.values()),
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _build_strategies(self) -> list[Any]:
        """Build a fresh strategy list from config. Deferred imports."""
        from privateye.main import _build_strategies  # type: ignore[import]
        return _build_strategies(self._cfg)

    def _build_risk_manager(self) -> Any:
        """Build a fresh RiskManager from config. Deferred imports."""
        from privateye.risk.manager import RiskManager
        from privateye.main import _build_advanced_risk  # type: ignore[import]
        asset_filter, exposure_monitor, _ = _build_advanced_risk(self._cfg)
        return RiskManager(
            self._cfg.get("risk", {}),
            exposure_monitor=exposure_monitor,
            asset_filter=asset_filter,
        )


# ── Module-level helpers ──────────────────────────────────────────────────────

def _compute_sharpe(equity_curve: list[float], bars_per_year: float) -> float:
    """Annualised Sharpe ratio from an equity curve. Returns 0.0 on edge cases."""
    if len(equity_curve) < 2:
        return 0.0
    try:
        arr = np.array(equity_curve, dtype=float)
        returns = np.diff(arr) / np.where(arr[:-1] != 0, arr[:-1], 1.0)
        if len(returns) < 2:
            return 0.0
        std = float(np.std(returns))
        if std < 1e-12:
            return 0.0
        mean_r = float(np.mean(returns))
        return float(mean_r / std * np.sqrt(bars_per_year))
    except Exception:
        return 0.0


def _compute_max_dd(equity_curve: list[float]) -> float:
    """Maximum peak-to-trough drawdown fraction. Returns 0.0 on edge cases."""
    if len(equity_curve) < 2:
        return 0.0
    try:
        arr = np.array(equity_curve, dtype=float)
        peak = arr[0]
        max_dd = 0.0
        for e in arr:
            if e > peak:
                peak = e
            if peak > 0:
                dd = (peak - e) / peak
                if dd > max_dd:
                    max_dd = dd
        return float(max_dd)
    except Exception:
        return 0.0
