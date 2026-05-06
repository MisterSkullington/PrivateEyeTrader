#!/usr/bin/env python3
"""
CLI backtest runner with Rich output.

Usage:
    python scripts/backtest.py
    python scripts/backtest.py --config privateye/config/settings.yaml --symbol BTC/USDT --tf 1h
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from privateye.utils.logging import setup_logging, get_logger

setup_logging()
log = get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest and print report")
    parser.add_argument("--config", default="privateye/config/settings.yaml")
    parser.add_argument("--symbol", default=None, help="Override symbol from config")
    parser.add_argument("--tf", default=None, help="Override timeframe from config")
    parser.add_argument("--strategy", default=None, choices=["directional", "fusion"],
                        help="Strategy to run; 'fusion' enables ML (overrides config)")
    parser.add_argument("--export-trades", default=None, help="Export trades to CSV path")
    args = parser.parse_args()

    from privateye.config.loader import load_config
    cfg = load_config(args.config)
    if args.symbol:
        cfg["symbols"] = [args.symbol]
    if args.tf:
        cfg["primary_timeframe"] = args.tf
    if args.strategy == "fusion":
        cfg.setdefault("ml", {})["enabled"] = True
        cfg["ml"]["strategy"] = "fusion"
    elif args.strategy == "directional":
        cfg.setdefault("ml", {})["enabled"] = False

    from privateye.main import run_backtest
    report = run_backtest(cfg)

    # Monte Carlo robustness check (trade permutation) when we have trade history
    if report is not None and report.trades:
        try:
            from privateye.models.monte_carlo import run_trade_permutation
            mc_cfg = cfg.get("ml", {}).get("monte_carlo", {})
            pnl_seq = [t.pnl for t in report.trades]
            mc = run_trade_permutation(
                pnl_seq,
                initial_capital=report.initial_capital,
                n_simulations=mc_cfg.get("n_simulations", 500),
            )
            print(mc)
        except Exception as e:
            log.warning(f"Monte Carlo skipped: {e}")

    if args.export_trades:
        import asyncio
        from privateye.data.storage.sqlite_store import SQLiteStore

        async def _export():
            store = SQLiteStore(cfg.get("data", {}).get("sqlite_path", "data/privateye.db"))
            await store.open()
            await store.export_trades_csv(args.export_trades)
            await store.close()
        asyncio.run(_export())
        print(f"\nTrades exported to: {args.export_trades}")


if __name__ == "__main__":
    main()
