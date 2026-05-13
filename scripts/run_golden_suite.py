#!/usr/bin/env python3
"""
Run the golden backtest suite and save the locked v1.0 baseline.

Usage::

    python scripts/run_golden_suite.py
    python scripts/run_golden_suite.py --config privateye/config/settings.yaml

After running, commit ``artifacts/golden_results_v1.0.json`` and tag ``v1.0-baseline``.
Every subsequent run must produce an identical JSON to prove reproducibility.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from privateye.utils.logging import setup_logging, get_logger
from privateye.backtesting.golden_suite import run_golden_backtest, OUTPUT_PATH

setup_logging()
log = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run golden backtest suite")
    parser.add_argument(
        "--config",
        default="privateye/config/settings.yaml",
        help="Path to settings.yaml",
    )
    args = parser.parse_args()

    from privateye.config.loader import load_config
    cfg = load_config(args.config)

    log.info("=" * 60)
    log.info("  GOLDEN BACKTEST SUITE — v1.0 Baseline")
    log.info("=" * 60)

    result = run_golden_backtest(cfg)

    bt  = result["backtest"]
    wf  = result["walk_forward"]
    kf  = result["kfold"]
    mc  = result["monte_carlo"]

    print()
    print("=" * 60)
    print("  GOLDEN SUITE RESULTS")
    print("=" * 60)
    print(f"  Symbol / TF    : {result['symbol']} {result['timeframe']}")
    print(f"  Date range     : {result['date_from']} → {result['date_to']}")
    print(f"  Bars           : {result['n_bars']}")
    print(f"  Seed           : {result['seed']}")
    print("-" * 60)
    print("  BACKTEST")
    print(f"    Trades       : {bt.get('total_trades', 0)}")
    print(f"    Win Rate     : {bt.get('win_rate', 0):.1%}")
    print(f"    Total PnL    : {bt.get('total_pnl', 0):+.2f}")
    print(f"    Sharpe       : {bt.get('sharpe_ratio', 0):.3f}")
    print(f"    Max DD       : {bt.get('max_drawdown_pct', 0):.1%}")
    print(f"    Expected Val : {bt.get('expected_value', 0):+.4f}")
    print("-" * 60)
    print("  WALK-FORWARD (OOS)")
    print(f"    Folds        : {wf.get('n_folds', 0)}")
    print(f"    OOS Sharpe   : {wf.get('mean_oos_sharpe', 0):.3f} ± {wf.get('std_oos_sharpe', 0):.3f}")
    print(f"    Stability    : {wf.get('stability_score', 0):.1%} of folds profitable")
    print(f"    Efficiency   : {wf.get('mean_efficiency_ratio', 0):.3f} (OOS/IS Sharpe)")
    print("-" * 60)
    print("  PURGED K-FOLD CV")
    print(f"    Folds        : {kf.get('n_folds', 0)}")
    print(f"    CV Score     : {kf.get('cv_mean_score', 0):.3f} ± {kf.get('cv_std_score', 0):.3f}")
    print(f"    Overfitting  : {kf.get('overfitting_gap', 0):+.3f} (IS − CV gap)")
    print("-" * 60)
    print("  MONTE CARLO (1 000 permutations)")
    print(f"    Ruin Prob    : {mc.get('ruin_probability', 0):.1%}")
    print(f"    p5  Sharpe   : {mc.get('p5_sharpe', 0):.3f}")
    print(f"    p50 Sharpe   : {mc.get('p50_sharpe', 0):.3f}")
    print(f"    p95 Sharpe   : {mc.get('p95_sharpe', 0):.3f}")
    print("=" * 60)
    print(f"  Saved to: {OUTPUT_PATH}")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Review results above")
    print("  2. git add artifacts/golden_results_v1.0.json")
    print("  3. git commit -m 'chore: lock v1.0 golden baseline'")
    print("  4. git tag v1.0-baseline")


if __name__ == "__main__":
    main()
