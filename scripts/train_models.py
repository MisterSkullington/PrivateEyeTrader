"""
CLI: train all Phase-2 ML models and save artifacts.

Usage:
    python scripts/train_models.py --symbol BTC/USDT --timeframe 1h

Prerequisites:
    - Historical OHLCV data in the SQLite store (run scripts/fetch_data.py first)
    - Phase-2 dependencies installed: torch, xgboost, hmmlearn (or sklearn),
      stable-baselines3, gymnasium
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from privateye.config.loader import load_config
from privateye.data.storage.sqlite_store import SQLiteStore
from privateye.utils.logging import get_logger

log = get_logger()
console = Console(highlight=False)

ARTIFACTS_DIR = Path("privateye/models/artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_bars(store: SQLiteStore, symbol: str, timeframe: str) -> pd.DataFrame:
    import asyncio

    async def _fetch():
        await store.open()
        try:
            return await store.load_ohlcv(symbol, timeframe)
        finally:
            await store.close()

    bars = asyncio.run(_fetch())
    if bars.empty:
        console.print(f"[red]No data for {symbol} {timeframe}. Run scripts/fetch_data.py first.[/red]")
        sys.exit(1)
    console.print(f"[green]Loaded {len(bars)} bars for {symbol} {timeframe}.[/green]")
    return bars


def train_regime(bars: pd.DataFrame) -> None:
    console.print("[cyan]Training Regime Detector...[/cyan]")
    try:
        from privateye.models.regime_detector import RegimeDetector
        m = RegimeDetector(artifacts_dir=ARTIFACTS_DIR)
        m.fit(bars)
        m.save()
        console.print("[green][OK] Regime Detector saved.[/green]")
    except Exception as e:
        console.print(f"[red][FAIL] Regime Detector failed: {e}[/red]")


def train_lstm(bars: pd.DataFrame, cfg: dict) -> None:
    console.print("[cyan]Training LSTM Forecaster (this may take 1-2h on CPU)...[/cyan]")
    try:
        from privateye.models.lstm_forecaster import LSTMForecaster
        ml_cfg = cfg.get("ml", {}).get("lstm", {})
        m = LSTMForecaster(
            lookback=ml_cfg.get("lookback_bars", 60),
            label_horizon=ml_cfg.get("label_horizon_bars", 5),
            label_threshold=ml_cfg.get("label_threshold", 0.002),
            hidden_size=ml_cfg.get("hidden_size", 256),
            num_layers=ml_cfg.get("num_layers", 2),
            dropout=ml_cfg.get("dropout", 0.2),
            artifacts_dir=ARTIFACTS_DIR,
        )
        m.fit(bars)
        m.save()
        console.print("[green][OK] LSTM Forecaster saved.[/green]")
    except Exception as e:
        console.print(f"[red][FAIL] LSTM Forecaster failed: {e}[/red]")


def train_gbm(bars: pd.DataFrame, cfg: dict) -> None:
    console.print("[cyan]Training GBM Classifier...[/cyan]")
    try:
        from privateye.models.gbm_classifier import GBMClassifier
        ml_cfg = cfg.get("ml", {}).get("gbm", {})
        m = GBMClassifier(
            n_estimators=ml_cfg.get("n_estimators", 500),
            max_depth=ml_cfg.get("max_depth", 6),
            gate_threshold=ml_cfg.get("gate_threshold", 0.45),
            artifacts_dir=ARTIFACTS_DIR,
        )
        m.fit(bars)
        m.save()
        console.print("[green][OK] GBM Classifier saved.[/green]")
        # Print top features
        top = m.get_top_features(10)
        console.print("  Top features by importance:")
        for name, score in top:
            console.print(f"    {name:30s} {score:.4f}")
    except Exception as e:
        console.print(f"[red][FAIL] GBM Classifier failed: {e}[/red]")


def train_rl(bars: pd.DataFrame, cfg: dict) -> None:
    console.print("[cyan]Training RL Policy (PPO — this may take 2-6h on CPU)...[/cyan]")
    try:
        from privateye.models.rl_policy import RLPolicy
        ml_cfg = cfg.get("ml", {}).get("rl", {})
        m = RLPolicy(
            total_timesteps=ml_cfg.get("total_timesteps", 500_000),
            artifacts_dir=ARTIFACTS_DIR,
        )
        m.fit(bars)
        m.save()
        console.print("[green][OK] RL Policy saved.[/green]")
    except Exception as e:
        console.print(f"[red][FAIL] RL Policy failed: {e}[/red]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase-2 ML models")
    parser.add_argument("--symbol",    default="BTC/USDT",  help="Trading pair")
    parser.add_argument("--timeframe", default="1h",         help="Bar timeframe")
    parser.add_argument("--config",    default="privateye/config/settings.yaml")
    parser.add_argument("--skip",      nargs="*", default=[],
                        choices=["regime", "lstm", "gbm", "rl"],
                        help="Models to skip")
    args = parser.parse_args()

    cfg = load_config(args.config)
    db_path = cfg.get("data", {}).get("sqlite_path", "data/privateye.db")
    store = SQLiteStore(db_path)

    bars = _load_bars(store, args.symbol, args.timeframe)

    console.print(f"\n[bold]Training Phase-2 ML models on {len(bars)} bars[/bold]")
    console.print(f"Symbol: {args.symbol}  Timeframe: {args.timeframe}")
    console.print(f"Artifacts: {ARTIFACTS_DIR.resolve()}\n")

    if "regime" not in args.skip:
        train_regime(bars)

    if "lstm" not in args.skip:
        train_lstm(bars, cfg)

    if "gbm" not in args.skip:
        train_gbm(bars, cfg)

    if "rl" not in args.skip:
        train_rl(bars, cfg)

    console.print("\n[bold green]All models trained![/bold green]")
    console.print("Next: run scripts/backtest.py --strategy fusion to validate.")


if __name__ == "__main__":
    main()
