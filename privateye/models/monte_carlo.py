"""
Monte Carlo simulation for backtesting robustness.

Two modes:
  1. Trade permutation  — shuffles realised PnL sequences 1000×,
     recomputes equity curves → distribution of Sharpe / max DD.
  2. Block bootstrap    — resamples overlapping price blocks,
     reconstructs synthetic price paths → full backtest on each.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from privateye.utils.logging import get_logger

log = get_logger()


@dataclass
class MonteCarloReport:
    n_simulations: int
    mode: str

    p5_sharpe:  float = 0.0
    p50_sharpe: float = 0.0
    p95_sharpe: float = 0.0

    p5_max_dd:  float = 0.0
    p50_max_dd: float = 0.0
    p95_max_dd: float = 0.0

    p5_total_return:  float = 0.0
    p50_total_return: float = 0.0
    p95_total_return: float = 0.0

    ruin_probability: float = 0.0  # fraction of runs with equity drop > 50%

    sharpe_distribution: list[float] = field(default_factory=list, repr=False)
    dd_distribution:     list[float] = field(default_factory=list, repr=False)

    def __str__(self) -> str:
        lines = [
            f"-- Monte Carlo ({self.mode}, n={self.n_simulations}) -------------",
            f"  Sharpe  P5={self.p5_sharpe:+.2f}  P50={self.p50_sharpe:+.2f}  P95={self.p95_sharpe:+.2f}",
            f"  Max DD  P5={self.p5_max_dd:.1%}  P50={self.p50_max_dd:.1%}  P95={self.p95_max_dd:.1%}",
            f"  Return  P5={self.p5_total_return:.1%}  P50={self.p50_total_return:.1%}  P95={self.p95_total_return:.1%}",
            f"  Ruin probability (equity drop >50%): {self.ruin_probability:.1%}",
        ]
        return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sharpe(returns: np.ndarray, bars_per_year: int = 8760) -> float:
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(bars_per_year))


def _max_drawdown(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    dd   = (peak - equity_curve) / np.where(peak > 0, peak, 1)
    return float(dd.max())


def _equity_from_pnl(pnl_sequence: np.ndarray, initial: float = 10_000.0) -> np.ndarray:
    return initial + np.concatenate([[0.0], np.cumsum(pnl_sequence)])


# ── Public API ────────────────────────────────────────────────────────────────

def run_trade_permutation(
    pnl_sequence: list[float] | np.ndarray,
    initial_capital: float = 10_000.0,
    n_simulations: int = 1000,
    bars_per_year: int = 8760,
    rng_seed: int = 42,
) -> MonteCarloReport:
    """
    Shuffle the realised PnL sequence `n_simulations` times.
    Tests whether the strategy's edge is robust to trade ordering.
    """
    pnl = np.asarray(pnl_sequence, dtype=float)
    rng = np.random.default_rng(rng_seed)

    sharpes: list[float] = []
    dds:     list[float] = []
    returns: list[float] = []

    for _ in range(n_simulations):
        shuffled = rng.permutation(pnl)
        eq       = _equity_from_pnl(shuffled, initial_capital)
        ret      = eq[1:] / eq[:-1] - 1
        sharpes.append(_sharpe(ret, bars_per_year))
        dds.append(_max_drawdown(eq))
        returns.append(float((eq[-1] - initial_capital) / initial_capital))

    sh = np.array(sharpes)
    dd = np.array(dds)
    rt = np.array(returns)
    ruin = float(np.mean(rt < -0.5))

    report = MonteCarloReport(
        n_simulations=n_simulations,
        mode="trade_permutation",
        p5_sharpe=float(np.percentile(sh, 5)),
        p50_sharpe=float(np.percentile(sh, 50)),
        p95_sharpe=float(np.percentile(sh, 95)),
        p5_max_dd=float(np.percentile(dd, 5)),
        p50_max_dd=float(np.percentile(dd, 50)),
        p95_max_dd=float(np.percentile(dd, 95)),
        p5_total_return=float(np.percentile(rt, 5)),
        p50_total_return=float(np.percentile(rt, 50)),
        p95_total_return=float(np.percentile(rt, 95)),
        ruin_probability=ruin,
        sharpe_distribution=sh.tolist(),
        dd_distribution=dd.tolist(),
    )
    log.info(f"[MonteCarlo] Trade permutation complete: "
             f"median Sharpe={report.p50_sharpe:.2f}, ruin={ruin:.1%}")
    return report


def run_block_bootstrap(
    close_prices: np.ndarray,
    strategy_fn,
    initial_capital: float = 10_000.0,
    n_simulations: int = 500,
    block_size: int = 20,
    bars_per_year: int = 8760,
    rng_seed: int = 42,
) -> MonteCarloReport:
    """
    Resample overlapping price blocks to build synthetic price paths,
    then run `strategy_fn(synthetic_closes) -> pnl_sequence` on each.

    Args:
        close_prices: 1-D array of historical close prices
        strategy_fn:  callable(np.ndarray) → np.ndarray of per-bar PnL
        n_simulations, block_size: bootstrap parameters
    """
    prices = np.asarray(close_prices, dtype=float)
    n      = len(prices)
    rng    = np.random.default_rng(rng_seed)

    sharpes: list[float] = []
    dds:     list[float] = []
    returns: list[float] = []

    for sim in range(n_simulations):
        # Build a synthetic price path of the same length via block resampling
        synth = [prices[0]]
        while len(synth) < n:
            start    = rng.integers(0, max(1, n - block_size))
            block    = prices[start : start + block_size]
            rets     = block[1:] / block[:-1]  # relative returns within block
            for r in rets:
                synth.append(synth[-1] * r)
                if len(synth) >= n:
                    break
        synth = np.array(synth[:n])

        try:
            pnl = strategy_fn(synth)
            eq  = _equity_from_pnl(pnl, initial_capital)
        except Exception:
            continue

        ret = eq[1:] / eq[:-1] - 1
        sharpes.append(_sharpe(ret, bars_per_year))
        dds.append(_max_drawdown(eq))
        returns.append(float((eq[-1] - initial_capital) / initial_capital))

        if (sim + 1) % 100 == 0:
            log.debug(f"[MonteCarlo] Block bootstrap: {sim+1}/{n_simulations}")

    if not sharpes:
        return MonteCarloReport(n_simulations=0, mode="block_bootstrap")

    sh = np.array(sharpes)
    dd = np.array(dds)
    rt = np.array(returns)
    ruin = float(np.mean(rt < -0.5))

    report = MonteCarloReport(
        n_simulations=len(sharpes),
        mode="block_bootstrap",
        p5_sharpe=float(np.percentile(sh, 5)),
        p50_sharpe=float(np.percentile(sh, 50)),
        p95_sharpe=float(np.percentile(sh, 95)),
        p5_max_dd=float(np.percentile(dd, 5)),
        p50_max_dd=float(np.percentile(dd, 50)),
        p95_max_dd=float(np.percentile(dd, 95)),
        p5_total_return=float(np.percentile(rt, 5)),
        p50_total_return=float(np.percentile(rt, 50)),
        p95_total_return=float(np.percentile(rt, 95)),
        ruin_probability=ruin,
        sharpe_distribution=sh.tolist(),
        dd_distribution=dd.tolist(),
    )
    log.info(f"[MonteCarlo] Block bootstrap complete: "
             f"median Sharpe={report.p50_sharpe:.2f}, ruin={ruin:.1%}")
    return report
