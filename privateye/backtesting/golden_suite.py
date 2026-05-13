"""
Golden Backtest Suite — v1.0 Immutable Baseline.

Once ``artifacts/golden_results_v1.0.json`` is committed and tagged
``v1.0-baseline``, this suite must produce **identical** output on every
subsequent run (same seeds, same data slice, same engine version).

Usage::

    from privateye.backtesting.golden_suite import run_golden_backtest
    result = run_golden_backtest()           # uses default cfg from settings.yaml
    result = run_golden_backtest(cfg=my_cfg) # override config
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from privateye.utils.logging import get_logger
from privateye.utils.time import now_utc

log = get_logger()

# ── Immutable baseline constants ──────────────────────────────────────────────
GOLDEN_SYMBOL    = "BTC/USDT"
GOLDEN_TIMEFRAME = "1h"
GOLDEN_DATE_FROM = "2023-01-01"
GOLDEN_DATE_TO   = "2026-01-01"
GOLDEN_SEED      = 42
GOLDEN_CAPITAL   = 10_000.0
OUTPUT_PATH      = Path("artifacts/golden_results_v1.0.json")


def seed_all(seed: int = GOLDEN_SEED) -> None:
    """Lock Python ``random``, NumPy, and PyTorch seeds for reproducibility.

    Must be called before any random operations in the golden suite run.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass   # torch optional — determinism still enforced for NumPy/random


def _load_data(cfg: dict, data_dir: str) -> pd.DataFrame:
    """Load BTC/USDT 1h bars and slice to the golden date range."""
    from privateye.data.loaders import load_bars
    bars = load_bars(GOLDEN_SYMBOL, GOLDEN_TIMEFRAME, data_dir=data_dir)
    if bars.empty:
        raise FileNotFoundError(
            f"No data found for {GOLDEN_SYMBOL} {GOLDEN_TIMEFRAME} in {data_dir}. "
            "Run: python scripts/fetch_data.py --symbol BTC/USDT --timeframe 1h --days 1100"
        )

    # Slice to golden date range (inclusive)
    ts = pd.to_datetime(bars["timestamp"])
    mask = (ts >= GOLDEN_DATE_FROM) & (ts <= GOLDEN_DATE_TO)
    sliced = bars[mask].reset_index(drop=True)
    log.info(
        f"[GoldenSuite] Data slice: {len(sliced)} bars "
        f"({GOLDEN_DATE_FROM} → {GOLDEN_DATE_TO})"
    )
    return sliced


def _run_backtest(bars: pd.DataFrame, cfg: dict) -> dict:
    """Run BacktestEngine and return to_dict() result."""
    from privateye.backtesting.engine import BacktestEngine
    from privateye.backtesting.simulator import SimulatedExchange
    from privateye.risk.manager import RiskManager

    bt_cfg = cfg.get("backtesting", {})
    risk_cfg = cfg.get("risk", {})
    strategy_cfg = cfg.get("strategies", {}).get("directional", {})
    strategy_cfg.setdefault("enabled", True)
    strategy_cfg.setdefault("timeframe", GOLDEN_TIMEFRAME)

    from privateye.strategies.directional import DirectionalStrategy
    strat = DirectionalStrategy(strategy_cfg)

    sim = SimulatedExchange(
        initial_capital=GOLDEN_CAPITAL,
        fee_taker=bt_cfg.get("fee_taker", 0.001),
        slippage_pct=bt_cfg.get("slippage_pct", 0.0005),
    )
    rm = RiskManager(risk_cfg)
    engine = BacktestEngine([strat], rm, sim, cfg)
    report = engine.run(bars, GOLDEN_SYMBOL, GOLDEN_TIMEFRAME)
    return report.to_dict()


def _run_walk_forward(bars: pd.DataFrame, cfg: dict) -> dict:
    """Run WalkForwardEngine and return key aggregate metrics."""
    from privateye.backtesting.walk_forward import WalkForwardConfig, WalkForwardEngine
    wf_raw = cfg.get("walk_forward", {})
    wf_cfg = WalkForwardConfig(
        train_bars=wf_raw.get("train_bars", 2000),
        test_bars=wf_raw.get("test_bars", 500),
        step_bars=wf_raw.get("step_bars", 500),
        min_train_bars=wf_raw.get("min_train_bars", 500),
        min_test_trades=wf_raw.get("min_test_trades", 3),
    )
    engine = WalkForwardEngine(cfg, wf_cfg)
    report = engine.run(bars, GOLDEN_SYMBOL, GOLDEN_TIMEFRAME)
    return {
        "n_folds":              report.n_folds,
        "mean_oos_sharpe":      round(report.mean_oos_sharpe, 4),
        "std_oos_sharpe":       round(report.std_oos_sharpe, 4),
        "stability_score":      round(report.stability_score, 4),
        "mean_efficiency_ratio": round(report.mean_efficiency_ratio, 4),
        "mean_oos_max_dd":      round(report.mean_oos_max_dd, 4),
        "mean_oos_win_rate":    round(report.mean_oos_win_rate, 4),
        "total_oos_trades":     report.total_oos_trades,
        "total_oos_pnl":        round(report.total_oos_pnl, 4),
    }


def _run_kfold(bars: pd.DataFrame, cfg: dict) -> dict:
    """Run PurgedKFoldEngine and return key aggregate metrics."""
    from privateye.backtesting.kfold import PurgedKFoldConfig, PurgedKFoldEngine
    kf_raw = cfg.get("kfold", {})
    kf_cfg = PurgedKFoldConfig(
        n_folds=kf_raw.get("n_folds", 5),
        purge_bars=kf_raw.get("purge_bars", 50),
        embargo_bars=kf_raw.get("embargo_bars", 10),
        min_train_bars=kf_raw.get("min_train_bars", 500),
        min_test_trades=kf_raw.get("min_test_trades", 3),
    )
    engine = PurgedKFoldEngine(cfg, kf_cfg)
    report = engine.run(bars, GOLDEN_SYMBOL, GOLDEN_TIMEFRAME)
    return {
        "n_folds":         report.n_folds,
        "cv_mean_score":   round(report.cv_mean_score, 4),
        "cv_std_score":    round(report.cv_std_score, 4),
        "stability_score": round(report.stability_score, 4),
        "mean_is_score":   round(report.mean_is_score, 4),
        "overfitting_gap": round(report.overfitting_gap, 4),
        "total_cv_trades": report.total_cv_trades,
        "total_cv_pnl":    round(report.total_cv_pnl, 4),
    }


def _run_monte_carlo(trades: list, seed: int) -> dict:
    """Run trade-permutation Monte Carlo and return key percentile stats."""
    if not trades:
        return {"n_simulations": 0, "ruin_probability": 0.0,
                "p5_sharpe": 0.0, "p50_sharpe": 0.0, "p95_sharpe": 0.0}
    from privateye.backtesting.monte_carlo import run_trade_permutation
    results = run_trade_permutation(trades, n=1000, seed=seed)
    sharpes = [r["sharpe"] for r in results]
    ruins   = sum(1 for r in results if r.get("max_drawdown_pct", 0) >= 1.0)
    return {
        "n_simulations":    len(results),
        "ruin_probability": round(ruins / len(results), 4),
        "p5_sharpe":  round(float(np.percentile(sharpes,  5)), 4),
        "p50_sharpe": round(float(np.percentile(sharpes, 50)), 4),
        "p95_sharpe": round(float(np.percentile(sharpes, 95)), 4),
    }


# ── Public entry point ────────────────────────────────────────────────────────

def run_golden_backtest(cfg: dict[str, Any] | None = None) -> dict:
    """Run the full golden suite and save results to ``OUTPUT_PATH``.

    Steps:
    1. Load BTC/USDT 1h data and slice to ``GOLDEN_DATE_FROM`` → ``GOLDEN_DATE_TO``
    2. Lock all random seeds (``seed_all(GOLDEN_SEED)``)
    3. BacktestEngine → BacktestReport
    4. WalkForwardEngine → WalkForwardReport
    5. PurgedKFoldEngine → KFoldReport
    6. Monte Carlo (1 000 permutations)
    7. Assemble golden dict and write to ``OUTPUT_PATH``

    Returns the golden dict.
    """
    if cfg is None:
        from privateye.config.loader import load_config
        cfg = load_config("privateye/config/settings.yaml")

    p0 = cfg.get("phase0", {})
    seed      = int(p0.get("golden_seed", GOLDEN_SEED))
    date_from = p0.get("golden_date_from", GOLDEN_DATE_FROM)
    date_to   = p0.get("golden_date_to",   GOLDEN_DATE_TO)
    output    = Path(p0.get("golden_output", str(OUTPUT_PATH)))
    data_dir  = cfg.get("backtesting", {}).get("data_dir",
                cfg.get("data", {}).get("data_dir", "data/historical"))

    log.info(f"[GoldenSuite] Starting golden backtest (seed={seed})")
    log.info(f"[GoldenSuite] Date range: {date_from} → {date_to}")

    seed_all(seed)

    bars = _load_data(cfg, data_dir)

    # ── 1. Backtest ───────────────────────────────────────────────────────────
    log.info("[GoldenSuite] Step 1/4: BacktestEngine …")
    seed_all(seed)   # re-seed before each sub-run for strict reproducibility
    bt_result = _run_backtest(bars, cfg)

    # Reconstruct TradeRecord list for Monte Carlo
    # (BacktestEngine stores them in the simulator — fetch via to_dict trades list)
    # We pass the raw trades from the already-run sim instead of re-running.
    # Re-derive via direct call (sim is fresh each time due to _run_backtest).
    # For MC we use the pnl list from to_dict — no raw objects needed.

    # ── 2. Walk-forward ───────────────────────────────────────────────────────
    log.info("[GoldenSuite] Step 2/4: WalkForwardEngine …")
    seed_all(seed)
    wf_result = _run_walk_forward(bars, cfg)

    # ── 3. Purged K-fold ──────────────────────────────────────────────────────
    log.info("[GoldenSuite] Step 3/4: PurgedKFoldEngine …")
    seed_all(seed)
    kf_result = _run_kfold(bars, cfg)

    # ── 4. Monte Carlo (uses pnl list from backtest) ──────────────────────────
    log.info("[GoldenSuite] Step 4/4: Monte Carlo (1 000 permutations) …")
    seed_all(seed)
    # Reconstruct minimal trade objects for monte_carlo from bt_result equity_curve
    # The existing run_trade_permutation expects a list of objects with .pnl / .pnl_pct
    # We need a fresh BacktestEngine run to get the actual TradeRecord objects.
    # Use a lightweight re-run just for MC data:
    from privateye.backtesting.engine import BacktestEngine
    from privateye.backtesting.simulator import SimulatedExchange
    from privateye.risk.manager import RiskManager
    from privateye.strategies.directional import DirectionalStrategy

    seed_all(seed)
    bt_cfg = cfg.get("backtesting", {})
    risk_cfg = cfg.get("risk", {})
    strategy_cfg = dict(cfg.get("strategies", {}).get("directional", {}))
    strategy_cfg.setdefault("enabled", True)
    strategy_cfg.setdefault("timeframe", GOLDEN_TIMEFRAME)
    strat2 = DirectionalStrategy(strategy_cfg)
    sim2 = SimulatedExchange(
        initial_capital=GOLDEN_CAPITAL,
        fee_taker=bt_cfg.get("fee_taker", 0.001),
        slippage_pct=bt_cfg.get("slippage_pct", 0.0005),
    )
    rm2 = RiskManager(risk_cfg)
    rep2 = BacktestEngine([strat2], rm2, sim2, cfg).run(bars, GOLDEN_SYMBOL, GOLDEN_TIMEFRAME)
    mc_result = _run_monte_carlo(rep2.trades, seed)

    # ── Assemble ──────────────────────────────────────────────────────────────
    golden: dict = {
        "generated_at":  now_utc().isoformat(),
        "symbol":        GOLDEN_SYMBOL,
        "timeframe":     GOLDEN_TIMEFRAME,
        "date_from":     date_from,
        "date_to":       date_to,
        "seed":          seed,
        "n_bars":        len(bars),
        "backtest":      bt_result,
        "walk_forward":  wf_result,
        "kfold":         kf_result,
        "monte_carlo":   mc_result,
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as fh:
        json.dump(golden, fh, indent=2, default=str)

    log.info(f"[GoldenSuite] ✓ Results saved to {output}")
    log.info(
        f"[GoldenSuite] Sharpe={bt_result.get('sharpe_ratio', 0):.3f}  "
        f"OOS Sharpe={wf_result.get('mean_oos_sharpe', 0):.3f}  "
        f"MC ruin={mc_result.get('ruin_probability', 0):.1%}"
    )
    return golden
