"""
BacktestEngine — strict temporal bar replay with zero look-ahead bias.

For each bar i:
  1. Build DataSnapshot from bars[0:i+1] (only past data)
  2. [Phase 5] BlackSwanGuard.check() — flatten + continue if triggered
  3. Call strategy.on_data(snapshot) → signals
  4. Pass signals through RiskManager (with bars + trade_history kwargs)
  5. Submit approved orders to SimulatedExchange
  6. SimulatedExchange.process_bar(bar[i]) — fills against bar i's OHLCV
  7. Call strategy.on_bar_end(snapshot, portfolio) → exit signals
  8. Process exit signals
  9. Update equity curve
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from privateye.backtesting.metrics import BacktestReport, compute_metrics, timeframe_bars_per_year
from privateye.backtesting.simulator import SimulatedExchange
from privateye.core.types import DataSnapshot, Direction, PortfolioState, TradingSignal
from privateye.risk.manager import RiskManager
from privateye.strategies.base import AbstractStrategy
from privateye.utils.logging import get_logger
from privateye.utils.time import now_utc

log = get_logger()


class BacktestEngine:
    def __init__(
        self,
        strategies: list[AbstractStrategy],
        risk_manager: RiskManager,
        exchange: SimulatedExchange,
        config: dict[str, Any],
        black_swan_guard: Any = None,   # BlackSwanGuard | None
    ) -> None:
        self.strategies = strategies
        self.risk_manager = risk_manager
        self.exchange = exchange
        self.config = config
        self.bar_window: int = config.get("data", {}).get("bar_window", 500)
        self.black_swan_guard = black_swan_guard

    def run(self, bars: pd.DataFrame, symbol: str, timeframe: str) -> BacktestReport:
        """
        Replay `bars` (sorted chronologically) for one symbol/timeframe.
        Returns a BacktestReport with full performance metrics.
        """
        log.info(f"Backtest start: {symbol} {timeframe} — {len(bars)} bars")
        equity_curve: list[float] = []
        min_warmup = max(200, self.bar_window // 2)

        # Periodic progress logging — every ~10% of bars (or every 1000, whichever
        # is more frequent). Without this, ML-heavy backtests look hung for the
        # 10–30 min the bar loop is silently grinding through 8000+ bars.
        import time as _time
        _progress_every = max(100, min(1000, len(bars) // 10))
        _t_start = _time.perf_counter()
        _last_log = _t_start

        for i in range(len(bars)):
            bar = bars.iloc[i].to_dict()

            if i > 0 and i % _progress_every == 0:
                _now = _time.perf_counter()
                _bars_per_sec = i / max(1e-6, _now - _t_start)
                _eta_sec = (len(bars) - i) / max(1e-6, _bars_per_sec)
                _trades_so_far = len(self.exchange.trade_records)
                log.info(
                    f"[Backtest] {i}/{len(bars)} bars "
                    f"({100*i/len(bars):.0f}%, {_bars_per_sec:.1f} bars/s, "
                    f"ETA {_eta_sec/60:.1f}m, trades={_trades_so_far})"
                )
                _last_log = _now

            # Strict: only data up to and including bar i (no future)
            window_start = max(0, i + 1 - self.bar_window)
            snapshot_bars = bars.iloc[window_start : i + 1].reset_index(drop=True)

            if i < min_warmup:
                self.exchange.process_bar(symbol, bar)
                equity_curve.append(self.exchange.equity)
                continue

            snapshot = DataSnapshot(
                symbol=symbol,
                timeframe=timeframe,
                bars=snapshot_bars,
                timestamp=now_utc(),
            )

            portfolio = self.exchange.get_portfolio_state()
            self.risk_manager.update_portfolio(portfolio)

            # ── Phase 5: Black-swan guard ─────────────────────────────────────
            if self.black_swan_guard is not None and self.black_swan_guard.enabled:
                funding = float(bar.get("funding_rate", 0.0))
                event = self.black_swan_guard.check(symbol, snapshot_bars, funding_rate=funding)
                if event is not None:
                    log.critical(
                        f"[BlackSwanGuard] {event.trigger.value} triggered: {event.detail}"
                    )
                    self.risk_manager.halt(f"black_swan:{event.trigger.value}")
                    # Flatten all open positions via the existing FLAT-signal path
                    portfolio = self.exchange.get_portfolio_state()
                    close_price = float(bar["close"])
                    for pos_symbol, pos in list(portfolio.positions.items()):
                        flat_sig = TradingSignal(
                            symbol=pos_symbol,
                            direction=Direction.FLAT,
                            confidence=1.0,
                            entry_price=close_price,
                            stop_price=0.0,
                            target_price=0.0,
                            strategy_id="black_swan_guard",
                            timeframe=timeframe,
                        )
                        _, _, flat_order = self.risk_manager.evaluate_signal(flat_sig, portfolio)
                        if flat_order:
                            self.exchange.submit_order(flat_order)
                    self.exchange.process_bar(symbol, bar)
                    equity_curve.append(self.exchange.equity)
                    continue   # skip strategy signal processing this bar
            # ─────────────────────────────────────────────────────────────────

            # Entry signals
            trade_history = self.exchange.trade_records  # for Kelly sizing
            for strategy in self.strategies:
                signals = strategy.on_data(snapshot)
                for signal in signals:
                    approved, reason, order = self.risk_manager.evaluate_signal(
                        signal,
                        portfolio,
                        bars=snapshot_bars,
                        bars_by_symbol={symbol: snapshot_bars},
                        trade_history=trade_history,
                    )
                    if approved and order:
                        self.exchange.submit_order(order)
                    elif not approved:
                        log.debug(f"[Backtest] Signal rejected: {reason}")

            # Fill pending orders against current bar
            fills = self.exchange.process_bar(symbol, bar)
            for fill in fills:
                for strategy in self.strategies:
                    if strategy.strategy_id == fill.strategy_id:
                        strategy.on_fill(fill, portfolio)

            # Exit / trailing-stop signals
            portfolio = self.exchange.get_portfolio_state()
            for strategy in self.strategies:
                exit_signals = strategy.on_bar_end(snapshot, portfolio)
                for signal in exit_signals:
                    if signal.direction == Direction.FLAT:
                        approved, _, order = self.risk_manager.evaluate_signal(
                            signal, portfolio,
                            bars=snapshot_bars,
                            bars_by_symbol={symbol: snapshot_bars},
                        )
                        if approved and order:
                            self.exchange.submit_order(order)
                            # Hotfix 2026-05-07: capture exit fills and notify
                            # the originating strategy. Without this on_fill
                            # call, the strategy's internal position tracker
                            # never closes — every subsequent bar's on_bar_end
                            # re-emits a stale exit signal (rejected as
                            # "Signal rejected: exit"), and bars_held keeps
                            # growing. The line-128 process_bar already does
                            # this for entry fills; this mirrors it for exits.
                            exit_fills = self.exchange.process_bar(symbol, bar)
                            for fill in exit_fills:
                                for s in self.strategies:
                                    if s.strategy_id == fill.strategy_id:
                                        s.on_fill(fill, portfolio)

            equity_curve.append(self.exchange.equity)

        final_portfolio = self.exchange.get_portfolio_state()
        bpy = timeframe_bars_per_year(timeframe)
        report = compute_metrics(
            trades=self.exchange.trade_records,
            equity_curve=equity_curve,
            initial_capital=self.config.get("backtesting", {}).get("initial_capital", 10000.0),
            bars_per_year=bpy,
            cost_stats=self.exchange.get_cost_stats(),
        )
        log.info(f"Backtest complete: {len(self.exchange.trade_records)} trades, "
                 f"final equity={final_portfolio.equity:.2f}")
        return report
