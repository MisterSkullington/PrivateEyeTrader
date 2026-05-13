"""Backtest performance metrics: Sharpe, Sortino, max drawdown, Calmar, win rate, profit factor."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from privateye.core.types import TradeRecord


@dataclass
class BacktestReport:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_abs: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    avg_bars_held: float = 0.0
    annualised_return_pct: float = 0.0
    initial_capital: float = 0.0
    final_equity: float = 0.0
    # Trade cost analysis
    total_fees_paid: float = 0.0
    fee_drag_pct: float = 0.0       # fees as % of gross profit
    slippage_cost_pct: float = 0.0  # slippage as % of initial capital
    limit_fill_rate: float = 0.0    # fraction of limit orders that filled
    limit_orders_placed: int = 0
    limit_orders_filled: int = 0
    # Advanced analytics (Phase 7)
    ulcer_index: float = 0.0            # RMS of drawdown depth over equity curve
    var_95_pct: float = 0.0             # 5th-percentile per-trade return (as %)
    cvar_95_pct: float = 0.0            # mean return below the VaR threshold
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    omega_ratio: float = 0.0            # gains above zero / |losses below zero|
    # Phase 0 — per-direction and edge metrics
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0          # win rate among LONG-side trades only
    short_win_rate: float = 0.0         # win rate among SHORT-side trades only
    expected_value: float = 0.0         # (win_rate × avg_win) + ((1−win_rate) × avg_loss)
    equity_curve: list[float] = field(default_factory=list)
    trades: list[TradeRecord] = field(default_factory=list)

    def __str__(self) -> str:
        sep = "-" * 50
        lines = [
            sep,
            "  BACKTEST REPORT",
            sep,
            f"  Trades          : {self.total_trades}  (W:{self.winning_trades} L:{self.losing_trades})",
            f"  Long / Short    : {self.long_trades} / {self.short_trades}",
            f"  Long Win Rate   : {self.long_win_rate * 100:.1f}%",
            f"  Short Win Rate  : {self.short_win_rate * 100:.1f}%",
            f"  Expected Value  : {self.expected_value:+.4f} per trade",
            f"  Win Rate        : {self.win_rate * 100:.1f}%",
            f"  Profit Factor   : {self.profit_factor:.2f}",
            f"  Total PnL       : {self.total_pnl:+.2f} ({self.total_pnl_pct:+.1f}%)",
            f"  Ann. Return     : {self.annualised_return_pct:+.1f}%",
            f"  Initial Capital : {self.initial_capital:.2f}",
            f"  Final Equity    : {self.final_equity:.2f}",
            f"  Max Drawdown    : {self.max_drawdown_pct * 100:.1f}%  ({self.max_drawdown_abs:.2f})",
            f"  Sharpe Ratio    : {self.sharpe_ratio:.2f}",
            f"  Sortino Ratio   : {self.sortino_ratio:.2f}",
            f"  Calmar Ratio    : {self.calmar_ratio:.2f}",
            f"  Avg Win         : {self.avg_win:+.2f}",
            f"  Avg Loss        : {self.avg_loss:+.2f}",
            f"  Avg Bars Held   : {self.avg_bars_held:.1f}",
            sep,
            "  TRADE COSTS",
            sep,
            f"  Total Fees Paid : {self.total_fees_paid:.2f} "
            f"({self.total_fees_paid / self.initial_capital * 100:.2f}% of capital)"
            if self.initial_capital > 0 else f"  Total Fees Paid : {self.total_fees_paid:.2f}",
            f"  Fee Drag        : {self.fee_drag_pct * 100:.1f}% of gross PnL",
            f"  Slippage Cost   : {self.slippage_cost_pct:.2f}% of capital",
            f"  Limit Fill Rate : {self.limit_fill_rate * 100:.1f}%"
            f"  ({self.limit_orders_filled}/{self.limit_orders_placed} limit orders filled)"
            if self.limit_orders_placed > 0 else "  Limit Fill Rate : N/A (no limit orders placed)",
            sep,
            "  ADVANCED ANALYTICS",
            sep,
            f"  Ulcer Index     : {self.ulcer_index:.4f}",
            f"  VaR 95%         : {self.var_95_pct:+.2f}%",
            f"  CVaR 95%        : {self.cvar_95_pct:+.2f}%",
            f"  Max Consec Wins : {self.max_consecutive_wins}",
            f"  Max Consec Loss : {self.max_consecutive_losses}",
            f"  Omega Ratio     : {self.omega_ratio:.2f}",
            sep,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialise all scalar fields to JSON-safe primitives.

        Converts ``equity_curve`` to a flat list of floats.
        Excludes the raw ``trades`` list (too large for API responses).
        """
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "total_pnl_pct": self.total_pnl_pct,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "profit_factor": self.profit_factor,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_abs": self.max_drawdown_abs,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "avg_bars_held": self.avg_bars_held,
            "annualised_return_pct": self.annualised_return_pct,
            "initial_capital": self.initial_capital,
            "final_equity": self.final_equity,
            "total_fees_paid": self.total_fees_paid,
            "fee_drag_pct": self.fee_drag_pct,
            "slippage_cost_pct": self.slippage_cost_pct,
            "limit_fill_rate": self.limit_fill_rate,
            "limit_orders_placed": self.limit_orders_placed,
            "limit_orders_filled": self.limit_orders_filled,
            "ulcer_index": self.ulcer_index,
            "var_95_pct": self.var_95_pct,
            "cvar_95_pct": self.cvar_95_pct,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "omega_ratio": self.omega_ratio,
            "long_trades": self.long_trades,
            "short_trades": self.short_trades,
            "long_win_rate": self.long_win_rate,
            "short_win_rate": self.short_win_rate,
            "expected_value": self.expected_value,
            "equity_curve": [float(v) for v in self.equity_curve],
        }


def compute_metrics(
    trades: list[TradeRecord],
    equity_curve: list[float],
    initial_capital: float,
    bars_per_year: int = 8760,  # hourly bars default; adjust per timeframe
    cost_stats: dict | None = None,
) -> BacktestReport:
    report = BacktestReport(initial_capital=initial_capital)
    report.trades = trades
    report.equity_curve = equity_curve

    if not trades:
        report.final_equity = equity_curve[-1] if equity_curve else initial_capital
        return report

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    report.total_trades = len(trades)
    report.winning_trades = len(wins)
    report.losing_trades = len(losses)
    report.win_rate = len(wins) / len(trades) if trades else 0.0
    report.total_pnl = sum(pnls)
    report.avg_win = float(np.mean(wins)) if wins else 0.0
    report.avg_loss = float(np.mean(losses)) if losses else 0.0
    report.avg_bars_held = float(np.mean([t.bars_held for t in trades]))

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    report.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    final_equity = equity_curve[-1] if equity_curve else initial_capital + report.total_pnl
    report.final_equity = final_equity
    report.total_pnl_pct = (final_equity - initial_capital) / initial_capital * 100

    # Drawdown from equity curve
    eq = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd_abs = peak - eq
    dd_pct = dd_abs / np.where(peak > 0, peak, 1)
    report.max_drawdown_abs = float(dd_abs.max())
    report.max_drawdown_pct = float(dd_pct.max())

    # Returns per bar
    if len(eq) > 1:
        bar_returns = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1)
        mean_ret = float(np.mean(bar_returns))
        std_ret = float(np.std(bar_returns))
        downside_ret = bar_returns[bar_returns < 0]
        downside_std = float(np.std(downside_ret)) if len(downside_ret) > 0 else std_ret

        sqrt_bpy = math.sqrt(bars_per_year)
        report.sharpe_ratio = (mean_ret / std_ret * sqrt_bpy) if std_ret > 0 else 0.0
        report.sortino_ratio = (mean_ret / downside_std * sqrt_bpy) if downside_std > 0 else 0.0

        n_bars = len(eq)
        years = n_bars / bars_per_year
        if years > 0 and initial_capital > 0:
            cagr = (final_equity / initial_capital) ** (1 / years) - 1
            report.annualised_return_pct = cagr * 100
        report.calmar_ratio = (
            report.annualised_return_pct / 100 / report.max_drawdown_pct
            if report.max_drawdown_pct > 0 else 0.0
        )

    if cost_stats:
        report.total_fees_paid    = float(cost_stats.get("total_fees_paid", 0.0))
        lp = int(cost_stats.get("limit_orders_placed", 0))
        lf = int(cost_stats.get("limit_orders_filled", 0))
        report.limit_orders_placed = lp
        report.limit_orders_filled = lf
        report.limit_fill_rate     = lf / lp if lp > 0 else 0.0
        gross_pnl = sum(t.pnl + t.fees for t in trades) if trades else 0.0
        report.fee_drag_pct = report.total_fees_paid / gross_pnl if gross_pnl > 0 else 0.0
        slippage_abs = float(cost_stats.get("total_slippage_cost", 0.0))
        report.slippage_cost_pct = (
            slippage_abs / initial_capital * 100 if initial_capital > 0 else 0.0
        )

    # ── Advanced analytics ────────────────────────────────────────────────────

    # Ulcer Index: RMS of drawdown percentage over equity curve
    if len(eq) > 1:
        peak_ui = np.maximum.accumulate(eq)
        dd_pct_ui = np.where(peak_ui > 0, (peak_ui - eq) / peak_ui, 0.0)
        report.ulcer_index = float(np.sqrt(np.mean(dd_pct_ui ** 2)))

    # VaR / CVaR at 95% confidence on per-trade returns (%)
    if trades:
        trade_rets = np.array([t.pnl_pct for t in trades])
        report.var_95_pct = float(np.percentile(trade_rets, 5))
        below_var = trade_rets[trade_rets <= report.var_95_pct]
        report.cvar_95_pct = float(np.mean(below_var)) if len(below_var) > 0 else report.var_95_pct

    # Consecutive win/loss streaks
    if trades:
        max_wins = max_losses = cur_wins = cur_losses = 0
        for t in trades:
            if t.pnl > 0:
                cur_wins += 1
                cur_losses = 0
            else:
                cur_losses += 1
                cur_wins = 0
            max_wins   = max(max_wins, cur_wins)
            max_losses = max(max_losses, cur_losses)
        report.max_consecutive_wins   = max_wins
        report.max_consecutive_losses = max_losses

    # Omega ratio: E[max(r, 0)] / E[max(-r, 0)], threshold = 0
    if trades:
        trade_rets_arr = np.array([t.pnl_pct for t in trades])
        gains  = float(np.sum(np.maximum(trade_rets_arr, 0.0)))
        losses = float(np.sum(np.maximum(-trade_rets_arr, 0.0)))
        report.omega_ratio = gains / losses if losses > 0 else float("inf")

    # ── Phase 0: per-direction metrics + expected value ───────────────────────
    if trades:
        from privateye.core.types import Direction
        long_t  = [t for t in trades if getattr(t, "side", None) == Direction.LONG]
        short_t = [t for t in trades if getattr(t, "side", None) == Direction.SHORT]
        report.long_trades  = len(long_t)
        report.short_trades = len(short_t)
        report.long_win_rate  = (
            sum(1 for t in long_t  if t.pnl > 0) / len(long_t)  if long_t  else 0.0
        )
        report.short_win_rate = (
            sum(1 for t in short_t if t.pnl > 0) / len(short_t) if short_t else 0.0
        )
        report.expected_value = (
            report.win_rate * report.avg_win
            + (1.0 - report.win_rate) * report.avg_loss
        )

    return report


def timeframe_bars_per_year(timeframe: str) -> int:
    mapping = {
        "1m": 525600, "5m": 105120, "15m": 35040, "30m": 17520,
        "1h": 8760, "4h": 2190, "1d": 365,
    }
    return mapping.get(timeframe, 8760)
