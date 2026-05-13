"""
PrivateEyeTrader entry point.

Usage:
    python -m privateye.main --mode backtest --config privateye/config/settings.yaml
    python -m privateye.main --mode paper
    python -m privateye.main --mode live
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from privateye.core.exceptions import ConfigError
from privateye.utils.logging import setup_logging, get_logger

setup_logging()
log = get_logger()


# ── Phase 13 audit-fix safety flag ────────────────────────────────────────────
# Set to True only after all Phase 13 audit fixes are merged AND the full test
# suite is green. Live trading is blocked while this is False.
# Search the plan file for "Phase 13" for the full punch-list.
#
# Flipped True on 2026-05-06 after all 11 Phase 13 audit fixes landed and
# the full suite reached 457 passing (409 baseline + 48 new tests).
PHASE13_COMPLETE: bool = True

# Flip True after golden_results_v1.0.json is committed and tagged v1.0-baseline.
# Run: python scripts/run_golden_suite.py
PHASE0_COMPLETE: bool = False

# Flipped True 2026-05-13 after MacroProvider, FeatureEngineer, FeatureDriftDetector,
# live alt-data enrichment, and all 564 tests passed.
PHASE1_COMPLETE: bool = True

PHASE2_COMPLETE: bool = True

# Flip True after all Phase 3 tests pass and execution-stats endpoint is green.
PHASE3_COMPLETE: bool = True

# Flip True after all Phase 4 tests pass (654 total).
PHASE4_COMPLETE: bool = True

# Flip True after all Phase 5 tests pass (684 total).
PHASE5_COMPLETE: bool = True


# ── Phase 3 builder helpers ───────────────────────────────────────────────────

def _build_slippage_predictor(cfg: dict[str, Any]) -> Any:
    """Build a SlippagePredictor from config.execution.predictive_slippage."""
    from privateye.execution.slippage_predictor import SlippagePredictor
    ps_cfg = cfg.get("execution", {}).get("predictive_slippage", {})
    return SlippagePredictor(
        alpha=float(ps_cfg.get("alpha", 0.1)),
        avg_daily_volume_usd=float(ps_cfg.get("avg_daily_volume_usd", 5e8)),
    )


def _build_smart_order_router(cfg: dict[str, Any], predictor: Any) -> Any | None:
    """Build SmartOrderRouter when config.execution.smart_routing.enabled is true."""
    from privateye.execution.smart_order_router import SmartOrderRouter
    sr_cfg = cfg.get("execution", {}).get("smart_routing", {})
    exchanges = cfg.get("exchanges", {})
    enabled_venues = [
        name for name, ex in exchanges.items()
        if isinstance(ex, dict) and ex.get("enabled", False)
    ]
    fee_rates = {
        name: float(ex.get("fee_taker", 0.001))
        for name, ex in exchanges.items()
        if isinstance(ex, dict)
    }
    return SmartOrderRouter(
        slippage_predictor=predictor,
        enabled_venues=enabled_venues or ["binance"],
        fee_rates=fee_rates or {"binance": 0.001},
        enabled=bool(sr_cfg.get("enabled", False)),
        urgency_threshold=float(sr_cfg.get("urgency_threshold", 0.85)),
        min_score_diff_bps=float(sr_cfg.get("min_score_diff_bps", 0.5)),
    )


def _build_pre_trade_analyzer(cfg: dict[str, Any], predictor: Any) -> Any | None:
    """Build PreTradeCostAnalyzer when config.execution.pre_trade_analysis.enabled is true."""
    pt_cfg = cfg.get("execution", {}).get("pre_trade_analysis", {})
    if not pt_cfg.get("enabled", False):
        return None
    from privateye.execution.pre_trade_analyzer import PreTradeCostAnalyzer
    return PreTradeCostAnalyzer(
        slippage_predictor=predictor,
        min_net_edge_pct=float(pt_cfg.get("min_net_edge_pct", 0.0025)),
        fee_taker=float(cfg.get("backtesting", {}).get("fee_taker", 0.001)),
    )


def _maybe_fit_vwap(exec_engine: Any, cfg: dict[str, Any]) -> None:
    """Fit VWAPExecutor on recent historical bars if data is available."""
    if not cfg.get("execution", {}).get("use_twap", False):
        return  # TWAP/VWAP slicing is disabled — nothing to fit
    from privateye.data.loaders import load_bars
    from privateye.config.loader import get_backtest_config
    bt_cfg = get_backtest_config(cfg)
    symbol = cfg.get("symbols", ["BTC/USDT"])[0]
    timeframe = cfg.get("primary_timeframe", "1h")
    try:
        bars = load_bars(symbol, timeframe, data_dir=bt_cfg.get("data_dir", "data/historical"))
        if not bars.empty:
            exec_engine.fit_volume_profile(bars)
    except Exception as exc:
        log.debug(f"[main] VWAP fit skipped: {exc}")


def _active_exchange_name(cfg: dict[str, Any]) -> str:
    """Return the single exchange enabled for trading.

    Phase 13 (C-5): the sandbox safeguard previously inspected hardcoded
    ``cfg["exchanges"]["binance"]``, which silently misbehaved when a user
    enabled a different exchange. Live mode requires exactly one enabled
    exchange — multiple is ambiguous, zero is a misconfiguration.
    """
    enabled = [
        name for name, ex in cfg.get("exchanges", {}).items()
        if isinstance(ex, dict) and ex.get("enabled", False)
    ]
    if len(enabled) == 0:
        raise ConfigError(
            "No exchange enabled. Set exchanges.<name>.enabled=true for at least one."
        )
    if len(enabled) > 1:
        raise ConfigError(
            f"Live mode requires exactly one enabled exchange, got: {enabled}. "
            "Disable all but one to proceed."
        )
    return enabled[0]


def _build_strategies(cfg: dict[str, Any]) -> list:
    from privateye.config.loader import get_strategy_config, get_risk_config

    risk_cfg = get_risk_config(cfg)
    ml_cfg   = cfg.get("ml", {})

    if ml_cfg.get("enabled", False) and ml_cfg.get("strategy", "") == "fusion":
        from privateye.strategies.fusion_strategy import FusionStrategy
        dir_cfg = get_strategy_config(cfg, "directional")
        fusion_cfg = {
            **dir_cfg,
            "artifacts_dir":       ml_cfg.get("artifacts_dir", "privateye/models/artifacts"),
            "gbm_gate":            ml_cfg.get("fusion", {}).get("gbm_gate", True),
            "gbm_gate_threshold":  ml_cfg.get("gbm", {}).get("gate_threshold", 0.45),
            "regime_gate":         ml_cfg.get("fusion", {}).get("regime_gate", True),
            "sharpe_window_trades": ml_cfg.get("fusion", {}).get("sharpe_window_trades", 30),
            "atr_stop_multiplier": risk_cfg.get("atr_stop_multiplier", 2.0),
            "max_bars_in_trade":   risk_cfg.get("max_bars_in_trade", 48),
            "enabled":             True,
        }
        strategy = FusionStrategy(fusion_cfg)
        log.info("Loaded FusionStrategy (ML enabled)")
        return [strategy]

    from privateye.strategies.directional import DirectionalStrategy
    from privateye.strategies.mean_reversion import MeanReversionStrategy

    strategies = []

    dir_cfg = get_strategy_config(cfg, "directional")
    dir_cfg["atr_stop_multiplier"] = risk_cfg.get("atr_stop_multiplier", 2.0)
    dir_cfg["max_bars_in_trade"] = risk_cfg.get("max_bars_in_trade", 48)
    if dir_cfg.get("enabled", True):
        strategies.append(DirectionalStrategy(dir_cfg))

    mr_cfg = get_strategy_config(cfg, "mean_reversion")
    mr_cfg["atr_stop_multiplier"] = risk_cfg.get("atr_stop_multiplier", 1.0)
    mr_cfg["max_bars_in_trade"] = risk_cfg.get("max_bars_in_trade", 24)
    if mr_cfg.get("enabled", False):
        strategies.append(MeanReversionStrategy(mr_cfg))

    log.info(f"Loaded {len(strategies)} strategies: {[s.strategy_id for s in strategies]}")
    return strategies


def _merge_alt_data(bars, symbol: str, cfg: dict[str, Any]):
    """
    Load funding rates and Fear & Greed from SQLite and merge onto OHLCV bars.
    Uses pd.merge_asof (forward-fill from left) so every bar gets the most
    recent alt-data reading without look-ahead.
    Returns bars with optional extra columns: funding_rate, open_interest, fear_greed.
    """
    import asyncio as _asyncio
    import pandas as _pd

    db_path = cfg.get("data", {}).get("sqlite_path", "data/privateye.db")
    alt_cfg = cfg.get("alt_data", {})

    async def _load():
        import aiosqlite
        from pathlib import Path
        p = Path(db_path)
        if not p.exists():
            return None, None

        async with aiosqlite.connect(p) as db:
            fr_df = None
            fg_df = None

            if alt_cfg.get("funding_rates", {}).get("enabled", True):
                try:
                    async with db.execute(
                        "SELECT timestamp, funding_rate, open_interest FROM funding_rates "
                        "WHERE symbol=? ORDER BY timestamp ASC",
                        (symbol,),
                    ) as cur:
                        rows = await cur.fetchall()
                    if rows:
                        fr_df = _pd.DataFrame(rows, columns=["timestamp", "funding_rate", "open_interest"])
                        fr_df["timestamp"] = _pd.to_datetime(fr_df["timestamp"], unit="ms", utc=True)
                except Exception:
                    pass

            if alt_cfg.get("fear_greed", {}).get("enabled", True):
                try:
                    async with db.execute(
                        "SELECT timestamp, fear_greed FROM fear_greed ORDER BY timestamp ASC"
                    ) as cur:
                        rows = await cur.fetchall()
                    if rows:
                        fg_df = _pd.DataFrame(rows, columns=["timestamp", "fear_greed"])
                        fg_df["timestamp"] = _pd.to_datetime(fg_df["timestamp"], unit="ms", utc=True)
                except Exception:
                    pass

        return fr_df, fg_df

    fr_df, fg_df = _asyncio.run(_load())

    result = bars.copy()
    ts_col = _pd.to_datetime(result["timestamp"], utc=True)

    if fr_df is not None and not fr_df.empty:
        fr_df = fr_df.sort_values("timestamp")
        result = _pd.merge_asof(
            result.sort_values("timestamp"),
            fr_df,
            on="timestamp",
            direction="backward",
        ).sort_index()
        log.info(f"[AltData] Merged {len(fr_df)} funding rate records onto {len(result)} bars")
    else:
        result["funding_rate"] = 0.0
        result["open_interest"] = 0.0

    if fg_df is not None and not fg_df.empty:
        fg_df = fg_df.sort_values("timestamp")
        result = _pd.merge_asof(
            result.sort_values("timestamp"),
            fg_df,
            on="timestamp",
            direction="backward",
        ).sort_index()
        log.info(f"[AltData] Merged {len(fg_df)} Fear & Greed records onto {len(result)} bars")
    else:
        result["fear_greed"] = 50.0  # neutral default

    # Fill any remaining NaN from merge gaps with neutral/zero values
    result["funding_rate"]  = result["funding_rate"].fillna(0.0)
    result["open_interest"] = result["open_interest"].fillna(0.0)
    result["fear_greed"]    = result["fear_greed"].fillna(50.0)
    return result.reset_index(drop=True)


def _build_advanced_risk(cfg: dict[str, Any]):
    """Build Phase 5 advanced risk components from config. All disabled by default."""
    from privateye.risk.asset_filter import AssetFilter
    from privateye.risk.black_swan_guard import BlackSwanGuard
    from privateye.risk.exposure_monitor import ExposureMonitor

    adv_cfg = cfg.get("risk", {}).get("advanced", {})
    asset_filter     = AssetFilter.from_config(adv_cfg.get("asset_filter", {}))
    exposure_monitor = ExposureMonitor.from_config(adv_cfg.get("exposure_monitor", {}))
    black_swan_guard = BlackSwanGuard.from_config(adv_cfg.get("black_swan", {}))
    return (
        asset_filter     if asset_filter.enabled     else None,
        exposure_monitor if exposure_monitor.enabled else None,
        black_swan_guard if black_swan_guard.enabled else None,
    )


def _build_compliance_engine(cfg: dict[str, Any]) -> Any:
    """
    Build a :class:`~privateye.compliance.engine.ComplianceEngine` from the
    ``compliance:`` section of settings.  Always succeeds — if the section is
    absent a default engine (sanctions enabled, no jurisdiction) is returned.
    """
    from privateye.compliance.engine import ComplianceEngine
    return ComplianceEngine.from_config(cfg.get("compliance", {}))


def _build_alert_manager(bus: Any, cfg: dict[str, Any]) -> Any:
    """
    Build and subscribe an AlertManager.  Always succeeds even when
    Telegram/email credentials are absent — Notifier degrades gracefully.
    Returns the AlertManager so callers can pass ``get_recent_alerts`` to the
    dashboard.
    """
    from privateye.alerts.alert_manager import AlertManager, _build_notifier_config
    from privateye.alerts.notifier import Notifier

    alerts_cfg = cfg.get("alerts", {})
    notifier   = Notifier(_build_notifier_config(alerts_cfg))
    manager    = AlertManager(notifier, bus, alerts_cfg)
    manager.subscribe_all()
    return manager


def _build_feedback_store(cfg: dict[str, Any]) -> Any:
    """Build a FeedbackStore with optional JSONL persistence.

    Returns a FeedbackStore instance.  Always succeeds.
    """
    from privateye.feedback.store import FeedbackStore
    feedback_cfg = cfg.get("phase4", {}).get("feedback", {})
    maxlen = int(feedback_cfg.get("maxlen", 1000))
    persist_path = feedback_cfg.get("persist_path", None)
    return FeedbackStore(maxlen=maxlen, persist_path=persist_path)


def _build_model_versions_callback(artifacts_dir: str) -> Any:
    """Return a callable that lists checkpoint history for all 5 model classes.

    Used to populate GET /api/model-versions on the dashboard.
    """
    from pathlib import Path
    from privateye.models.checkpoint import list_checkpoints

    checkpoint_dir = Path(artifacts_dir) / "checkpoints"
    model_names = [
        "GBMClassifier", "LGBMClassifier", "LSTMForecaster",
        "AttentionLSTM", "RegimeDetector",
    ]

    def _get_versions() -> list[dict]:
        results = []
        for name in model_names:
            try:
                ckpts = list_checkpoints(name, checkpoint_dir)
                for ck in ckpts:
                    results.append({**ck, "model_name": name})
            except Exception:
                pass
        # Sort newest first
        results.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
        return results

    return _get_versions


def _maybe_wire_online_learner(
    bus: Any, strategies: list, cfg: dict[str, Any]
) -> None:
    """Wire OnlineLearner to MARKET_DATA if online_learning is enabled and a FusionStrategy exists."""
    ol_cfg = cfg.get("robustness", {}).get("online_learning", {})
    if not ol_cfg.get("enabled", False):
        return

    try:
        from privateye.strategies.fusion_strategy import FusionStrategy
    except ImportError:
        return

    fusion = next((s for s in strategies if isinstance(s, FusionStrategy)), None)
    if fusion is None:
        log.warning(
            "[main] robustness.online_learning.enabled=true but no FusionStrategy "
            "found — OnlineLearner not wired"
        )
        return

    from privateye.core.types import EventType
    from privateye.models.online_learning import OnlineLearner

    artifacts_dir = cfg.get("ml", {}).get("artifacts_dir", "privateye/models/artifacts")
    drift_cfg = cfg.get("phase1", {}).get("drift_detection", {})
    # Phase 4: pass audit_log config from robustness block into online_learning config
    robustness_cfg = cfg.get("robustness", {})
    audit_log_cfg = robustness_cfg.get("audit_log", {})
    ol_cfg_with_audit = {**ol_cfg, "audit_log": audit_log_cfg}
    learner = OnlineLearner(
        bus, fusion._ensemble, ol_cfg_with_audit,
        artifacts_dir=artifacts_dir,
        drift_config=drift_cfg,
    )
    bus.subscribe(EventType.MARKET_DATA, learner.on_market_data)

    # Phase 4B: wire live performance gate to FILL events
    if ol_cfg.get("live_performance_gate", {}).get("enabled", False):
        bus.subscribe(EventType.FILL, learner.on_fill)
        log.info("[main] OnlineLearner live performance gate wired to FILL events")

    log.info(
        "[main] OnlineLearner wired — retrain every "
        f"{ol_cfg.get('retrain_every_bars', 200)} bars on "
        f"{ol_cfg.get('models', ['gbm', 'lstm', 'regime'])}"
        + (" (drift gate active)" if drift_cfg.get("enabled", False) else "")
    )


async def _enrich_bars_with_alt_data(
    bars: "Any",
    symbol: str,
    fr_provider: "Any | None",
    fg_provider: "Any | None",
    cfg: dict[str, Any],
) -> "Any":
    """Enrich a bars DataFrame with live alt-data columns (funding_rate, fear_greed).

    Phase 1: called from run_paper() / run_live() on_market_data when
    alt_data.auto_fetch_live is True.  Any provider error is caught; bars are
    returned unchanged so the strategy loop is never blocked.
    """
    import pandas as _pd

    result = bars

    if fr_provider is not None:
        try:
            fr_data = await fr_provider.fetch_latest()
            result = result.copy()
            result["funding_rate"] = float(fr_data.get("funding_rate", 0.0))
            result["open_interest"] = float(fr_data.get("open_interest", 0.0))
        except Exception as exc:
            log.warning(f"[main] FundingRateProvider failed: {exc!r}")

    if fg_provider is not None:
        try:
            fg_data = await fg_provider.fetch_latest()
            if "fear_greed" not in result.columns:
                result = result.copy()
            result["fear_greed"] = float(fg_data.get("fear_greed", 50.0))
        except Exception as exc:
            log.warning(f"[main] FearGreedProvider failed: {exc!r}")

    return result


def run_backtest(cfg: dict[str, Any]):
    from privateye.backtesting.engine import BacktestEngine
    from privateye.backtesting.simulator import SimulatedExchange
    from privateye.config.loader import get_backtest_config
    from privateye.data.loaders import load_bars
    from privateye.risk.manager import RiskManager

    bt_cfg = get_backtest_config(cfg)
    initial_capital = bt_cfg.get("initial_capital", 10000.0)
    data_dir = bt_cfg.get("data_dir", "data/historical")

    asset_filter, exposure_monitor, black_swan_guard = _build_advanced_risk(cfg)

    strategies = _build_strategies(cfg)
    risk_manager = RiskManager(
        cfg.get("risk", {}),
        exposure_monitor=exposure_monitor,
        asset_filter=asset_filter,
    )
    exchange = SimulatedExchange(
        initial_capital=initial_capital,
        fee_maker=bt_cfg.get("fee_maker", 0.001),
        fee_taker=bt_cfg.get("fee_taker", 0.001),
        slippage_pct=bt_cfg.get("slippage_pct", 0.0005),
        max_fill_pct_of_volume=bt_cfg.get("max_fill_pct_of_volume", 0.30),
    )
    engine = BacktestEngine(strategies, risk_manager, exchange, cfg,
                            black_swan_guard=black_swan_guard)

    # Phase 0: optional MLflow tracking
    from privateye.tracking.mlflow_tracker import MLflowTracker
    tracker = MLflowTracker(cfg.get("phase0", {}).get("mlflow", {}))

    symbols = cfg.get("symbols", ["BTC/USDT"])
    timeframe = cfg.get("primary_timeframe", "1h")

    last_report = None
    for symbol in symbols:
        bars = load_bars(symbol, timeframe, data_dir=data_dir)
        if bars.empty:
            log.error(f"No data for {symbol} {timeframe}. Run fetch_data.py first.")
            continue

        if cfg.get("alt_data", {}).get("enabled", False):
            bars = _merge_alt_data(bars, symbol, cfg)

        report = engine.run(bars, symbol, timeframe)
        print(report)
        last_report = report

        # Monte Carlo stress-test (trade permutation) — optional
        mc_cfg = cfg.get("ml", {}).get("monte_carlo", {})
        if mc_cfg.get("enabled", False) and report.trades:
            from privateye.models.monte_carlo import run_trade_permutation
            pnl_seq = [t.pnl for t in report.trades]
            n_sims  = int(mc_cfg.get("n_simulations", 1000))
            seed    = int(mc_cfg.get("seed", 42))
            mc_report = run_trade_permutation(
                pnl_sequence=pnl_seq,
                initial_capital=initial_capital,
                n_simulations=n_sims,
                rng_seed=seed,
            )
            print(mc_report)

        # Save trades to SQLite
        async def _save() -> None:
            from privateye.data.storage.sqlite_store import SQLiteStore
            store = SQLiteStore(cfg.get("data", {}).get("sqlite_path", "data/privateye.db"))
            await store.open()
            for trade in report.trades:
                await store.save_trade(trade)
            await store.close()

        asyncio.run(_save())

        # Compliance post-analysis: wash-sale flags + optional tax CSV export
        compliance_engine = _build_compliance_engine(cfg)
        if report.trades:
            c_report = compliance_engine.run_post_analysis(report.trades)
            if c_report.wash_sale_flags:
                log.warning(
                    f"[Compliance] {len(c_report.wash_sale_flags)} wash-sale "
                    f"flag(s) detected — total disallowed: "
                    f"{c_report.total_disallowed_loss:.2f}"
                )
            compliance_engine.export_tax_report(report.trades, symbol)

        # Phase 0: log to MLflow
        tracker.log_backtest(
            report,
            run_name=f"backtest_{symbol.replace('/', '_')}_{timeframe}",
            params={"symbol": symbol, "timeframe": timeframe,
                    "initial_capital": initial_capital},
        )

    return last_report


async def run_paper(cfg: dict[str, Any]) -> None:
    from privateye.backtesting.simulator import SimulatedExchange
    from privateye.config.loader import get_backtest_config, get_exchange_config
    from privateye.core.event_bus import AsyncEventBus
    from privateye.core.types import EventType
    from privateye.data.pipeline import DataPipeline
    from privateye.data.providers.binance import BinanceProvider
    from privateye.execution.engine import ExecutionEngine
    from privateye.execution.paper import PaperTrader
    from privateye.risk.manager import RiskManager

    bt_cfg = get_backtest_config(cfg)
    ex_cfg = get_exchange_config(cfg)
    initial_capital = bt_cfg.get("initial_capital", 10000.0)
    symbols = cfg.get("symbols", ["BTC/USDT"])
    timeframes = cfg.get("timeframes", ["1h"])
    primary_tf = cfg.get("primary_timeframe", "1h")
    bar_window = cfg.get("data", {}).get("bar_window", 500)
    poll_interval = cfg.get("data", {}).get("poll_interval_seconds", 60)

    bus = AsyncEventBus()
    alert_manager = _build_alert_manager(bus, cfg)
    compliance_engine = _build_compliance_engine(cfg)
    pipeline = DataPipeline(bus, bar_window=bar_window)
    strategies = _build_strategies(cfg)
    asset_filter, exposure_monitor, _ = _build_advanced_risk(cfg)

    # Phase 4: feedback store + model versions
    _p4_feedback_store = _build_feedback_store(cfg)
    _p4_artifacts_dir = cfg.get("ml", {}).get("artifacts_dir", "privateye/models/artifacts")
    _p4_model_versions_cb = _build_model_versions_callback(_p4_artifacts_dir)

    # Phase 1: live alt-data providers (optional, only when auto_fetch_live=true)
    _alt_fr_provider: Any = None
    _alt_fg_provider: Any = None
    alt_cfg = cfg.get("alt_data", {})
    if alt_cfg.get("auto_fetch_live", False) and alt_cfg.get("enabled", False):
        try:
            from privateye.data.providers.funding_rates import FundingRateProvider
            from privateye.data.providers.sentiment import FearGreedProvider
            _alt_fr_provider = FundingRateProvider(
                symbol=alt_cfg.get("funding_rates", {}).get("symbol", "BTC/USDT"),
            )
            _alt_fg_provider = FearGreedProvider()
            log.info("[main] Live alt-data enrichment enabled (FundingRate + FearGreed)")
        except Exception as exc:
            log.warning(f"[main] Live alt-data providers unavailable: {exc!r}")

    # Phase 3: slippage predictor, smart router, pre-trade analyzer
    slip_predictor = _build_slippage_predictor(cfg)
    smart_router   = _build_smart_order_router(cfg, slip_predictor)
    pre_trade_analyzer = _build_pre_trade_analyzer(cfg, slip_predictor)

    risk_manager = RiskManager(
        cfg.get("risk", {}),
        exposure_monitor=exposure_monitor,
        asset_filter=asset_filter,
        pre_trade_analyzer=pre_trade_analyzer,
    )

    sim_exchange = SimulatedExchange(
        initial_capital=initial_capital,
        fee_maker=bt_cfg.get("fee_maker", 0.001),
        fee_taker=bt_cfg.get("fee_taker", 0.001),
        slippage_pct=bt_cfg.get("slippage_pct", 0.0005),
    )
    exec_engine = ExecutionEngine(bus, simulator=sim_exchange, smart_router=smart_router)
    paper_trader = PaperTrader(bus, sim_exchange)
    _maybe_fit_vwap(exec_engine, cfg)

    # Wire MARKET_DATA → strategies → risk → execution
    async def on_signal(signal: Any) -> None:
        allowed, c_reason = compliance_engine.check_symbol(signal.symbol)
        if not allowed:
            log.debug(f"[Compliance] Signal suppressed: {c_reason}")
            return
        portfolio = sim_exchange.get_portfolio_state()
        approved, reason, order = risk_manager.evaluate_signal(signal, portfolio)
        if approved and order:
            await exec_engine.submit_order(order)
        else:
            log.debug(f"Signal rejected: {reason}")

    async def on_market_data(snapshot: Any) -> None:
        # Phase 1: optionally enrich bars with live alt-data before strategy dispatch
        enriched_snapshot = snapshot
        if _alt_fr_provider is not None or _alt_fg_provider is not None:
            import dataclasses as _dc
            enriched_bars = await _enrich_bars_with_alt_data(
                snapshot.bars, snapshot.symbol, _alt_fr_provider, _alt_fg_provider, cfg
            )
            enriched_snapshot = _dc.replace(snapshot, bars=enriched_bars)
        for strategy in strategies:
            entry_signals = strategy.on_data(enriched_snapshot)
            for s in entry_signals:
                await on_signal(s)
            portfolio = sim_exchange.get_portfolio_state()
            exit_signals = strategy.on_bar_end(enriched_snapshot, portfolio)
            for s in exit_signals:
                await on_signal(s)

    bus.subscribe(EventType.MARKET_DATA, on_market_data)

    async def on_fill(fill: Any) -> None:
        portfolio = sim_exchange.get_portfolio_state()
        for strategy in strategies:
            if strategy.strategy_id == fill.strategy_id:
                strategy.on_fill(fill, portfolio)
        log.info(f"Fill: {fill.side.value} {fill.quantity:.6f} {fill.symbol} @ {fill.price:.2f}")

    bus.subscribe(EventType.FILL, on_fill)

    # Phase 6: wire OnlineLearner if enabled
    _maybe_wire_online_learner(bus, strategies, cfg)

    # Start dashboard
    from privateye.dashboard.server import start_dashboard
    dashboard_cfg = cfg.get("dashboard", {})
    dashboard_task = asyncio.create_task(
        start_dashboard(
            host=dashboard_cfg.get("host", "0.0.0.0"),
            port=dashboard_cfg.get("port", 8081),
            get_portfolio=sim_exchange.get_portfolio_state,
            get_trades=lambda: sim_exchange.trade_records,
            get_fills=lambda: sim_exchange.fills,
            exec_engine=exec_engine,
            risk_manager=risk_manager,
            initial_capital=initial_capital,
            get_alerts=alert_manager.get_recent_alerts,
            get_compliance=compliance_engine.get_status,
            get_execution_stats=None,   # Phase 3: ShadowTracker only in run_shadow
            get_model_versions=_p4_model_versions_cb,        # Phase 4
            get_feedback=_p4_feedback_store.get_recent,      # Phase 4
            post_feedback=_p4_feedback_store.submit_feedback, # Phase 4
        )
    )

    provider = BinanceProvider(
        api_key=ex_cfg.get("api_key", ""),
        api_secret=ex_cfg.get("api_secret", ""),
        sandbox=ex_cfg.get("sandbox", True),
        poll_interval=poll_interval,
    )

    async def on_bar(symbol: str, tf: str, bars) -> None:
        await pipeline.push_bars(symbol, tf, bars)

    bus_task = asyncio.create_task(bus.run())
    log.info(f"Paper trading started. Dashboard: http://localhost:{dashboard_cfg.get('port', 8081)}")

    try:
        await provider.start(symbols, timeframes, on_bar)
    except asyncio.CancelledError:
        pass
    finally:
        provider.stop()
        bus.stop()
        dashboard_task.cancel()
        bus_task.cancel()


async def run_live(cfg: dict[str, Any]) -> None:
    """Live trading entry point.

    Phase 13 changes:
      • C-5: Active-exchange sandbox safeguard (was Binance-only)
      • C-3: Cached portfolio snapshot for dashboard (was async-from-sync crash)
      • H-6: Removed dead `live_portfolio` variable
      • PHASE13_COMPLETE flag blocks live trading until full audit fix-set lands
    """
    if not PHASE13_COMPLETE:
        log.critical(
            "Live mode is disabled until Phase 13 audit fixes are complete. "
            "Run `pytest tests/ -q` and ensure all 449 tests pass, then flip "
            "PHASE13_COMPLETE=True in privateye/main.py."
        )
        return

    # C-5: Sandbox safeguard inspects the ACTIVE exchange, not hardcoded Binance
    active_name = _active_exchange_name(cfg)
    ex_cfg = cfg["exchanges"][active_name]

    if ex_cfg.get("sandbox", True):
        log.warning(
            f"LIVE mode but {active_name}.sandbox=true — forcing paper mode for safety"
        )
        await run_paper(cfg)
        return

    # Confirm live intent
    if not ex_cfg.get("api_key"):
        log.error(f"{active_name.upper()}_API_KEY not set. Cannot run live.")
        return

    # C-4: Live mode requires dashboard auth unless explicitly disabled
    from privateye.dashboard.auth import auth_enabled
    require_auth = cfg.get("dashboard", {}).get("require_auth", True)
    if require_auth and not auth_enabled():
        log.error(
            "Live mode requires DASHBOARD_API_KEY to be set when "
            "dashboard.require_auth=true. Generate a key with: "
            "python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )
        return

    log.critical(f"LIVE MODE — real orders will be placed on {active_name}")
    from privateye.execution.adapter import ExchangeAdapter
    from privateye.core.event_bus import AsyncEventBus
    from privateye.core.types import EventType, PortfolioState
    from privateye.data.pipeline import DataPipeline
    from privateye.data.providers.binance import BinanceProvider
    from privateye.execution.engine import ExecutionEngine
    from privateye.risk.manager import RiskManager

    bus = AsyncEventBus()
    alert_manager = _build_alert_manager(bus, cfg)
    compliance_engine = _build_compliance_engine(cfg)
    pipeline = DataPipeline(bus, bar_window=cfg.get("data", {}).get("bar_window", 500))
    strategies = _build_strategies(cfg)
    asset_filter, exposure_monitor, _ = _build_advanced_risk(cfg)

    # Phase 4: feedback store + model versions
    _p4_feedback_store = _build_feedback_store(cfg)
    _p4_artifacts_dir = cfg.get("ml", {}).get("artifacts_dir", "privateye/models/artifacts")
    _p4_model_versions_cb = _build_model_versions_callback(_p4_artifacts_dir)

    # Phase 3: slippage predictor, smart router, pre-trade analyzer
    slip_predictor_live = _build_slippage_predictor(cfg)
    smart_router_live   = _build_smart_order_router(cfg, slip_predictor_live)
    pre_trade_live      = _build_pre_trade_analyzer(cfg, slip_predictor_live)

    risk_manager = RiskManager(
        cfg.get("risk", {}),
        exposure_monitor=exposure_monitor,
        asset_filter=asset_filter,
        pre_trade_analyzer=pre_trade_live,
    )
    adapter = ExchangeAdapter(ex_cfg)
    exec_engine = ExecutionEngine(bus, adapter=adapter, smart_router=smart_router_live)
    _maybe_fit_vwap(exec_engine, cfg)
    symbols = cfg.get("symbols", ["BTC/USDT"])
    timeframes = cfg.get("timeframes", ["1h"])
    primary_tf = cfg.get("primary_timeframe", "1h")
    poll_interval = cfg.get("data", {}).get("poll_interval_seconds", 60)
    initial_capital = float(cfg.get("backtesting", {}).get("initial_capital", 10000.0))

    # C-3: Cached portfolio snapshot. The dashboard's get_portfolio callback runs
    # SYNCHRONOUSLY from inside the FastAPI event loop, so it cannot await.
    # A background task refreshes this dict; the callback returns the latest snapshot.
    _portfolio_cache: dict[str, PortfolioState] = {
        "state": PortfolioState(equity=initial_capital, cash=initial_capital)
    }

    async def _refresh_live_portfolio() -> None:
        while True:
            try:
                _portfolio_cache["state"] = await exec_engine.get_portfolio_state(symbols)
            except Exception as exc:   # noqa: BLE001 — keep loop alive on any error
                log.warning(f"[main] Live portfolio refresh failed: {exc}")
            await asyncio.sleep(5)

    async def on_signal(signal: Any) -> None:
        allowed, c_reason = compliance_engine.check_symbol(signal.symbol)
        if not allowed:
            log.debug(f"[Compliance] Signal suppressed: {c_reason}")
            return
        portfolio = (await exec_engine.get_portfolio_state(symbols))
        approved, reason, order = risk_manager.evaluate_signal(signal, portfolio)
        if approved and order:
            await exec_engine.submit_order(order)
        else:
            log.debug(f"Signal rejected: {reason}")

    async def on_market_data(snapshot: Any) -> None:
        for strategy in strategies:
            portfolio = await exec_engine.get_portfolio_state(symbols)
            entry_signals = strategy.on_data(snapshot)
            for s in entry_signals:
                await on_signal(s)
            exit_signals = strategy.on_bar_end(snapshot, portfolio)
            for s in exit_signals:
                await on_signal(s)

    async def on_fill(fill: Any) -> None:
        portfolio = await exec_engine.get_portfolio_state(symbols)
        for strategy in strategies:
            strategy.on_fill(fill, portfolio)

    bus.subscribe(EventType.MARKET_DATA, on_market_data)
    bus.subscribe(EventType.FILL, on_fill)

    # Phase 6: wire OnlineLearner if enabled
    _maybe_wire_online_learner(bus, strategies, cfg)

    provider = BinanceProvider(
        api_key=ex_cfg.get("api_key", ""),
        api_secret=ex_cfg.get("api_secret", ""),
        sandbox=False,
        poll_interval=poll_interval,
    )

    async def on_bar(symbol: str, tf: str, bars) -> None:
        await pipeline.push_bars(symbol, tf, bars)

    from privateye.dashboard.server import start_dashboard
    dashboard_cfg = cfg.get("dashboard", {})
    refresh_task = asyncio.create_task(_refresh_live_portfolio())
    dashboard_task = asyncio.create_task(
        start_dashboard(
            host=dashboard_cfg.get("host", "127.0.0.1"),
            port=dashboard_cfg.get("port", 8081),
            get_portfolio=lambda: _portfolio_cache["state"],   # C-3: cached snapshot
            get_trades=lambda: [],
            get_fills=lambda: [],
            exec_engine=exec_engine,
            risk_manager=risk_manager,
            initial_capital=initial_capital,
            get_alerts=alert_manager.get_recent_alerts,
            get_compliance=compliance_engine.get_status,
            get_execution_stats=None,   # Phase 3: ShadowTracker only in run_shadow
            get_model_versions=_p4_model_versions_cb,        # Phase 4
            get_feedback=_p4_feedback_store.get_recent,      # Phase 4
            post_feedback=_p4_feedback_store.submit_feedback, # Phase 4
        )
    )
    bus_task = asyncio.create_task(bus.run())

    try:
        await provider.start(symbols, timeframes, on_bar)
    except asyncio.CancelledError:
        pass
    finally:
        provider.stop()
        bus.stop()
        refresh_task.cancel()
        dashboard_task.cancel()
        bus_task.cancel()


async def run_shadow(cfg: dict[str, Any]) -> None:
    """
    Paper simulation + live price tracking.

    Identical to run_paper() but wires a ShadowTracker to FILL events so every
    simulated fill is compared against the real exchange mid-price at fill time.
    No real orders are placed.  Use this before going live to quantify slippage.
    """
    from privateye.backtesting.simulator import SimulatedExchange
    from privateye.config.loader import get_backtest_config, get_exchange_config
    from privateye.core.event_bus import AsyncEventBus
    from privateye.core.types import EventType
    from privateye.data.pipeline import DataPipeline
    from privateye.data.providers.binance import BinanceProvider
    from privateye.execution.adapter import ExchangeAdapter
    from privateye.execution.engine import ExecutionEngine
    from privateye.execution.paper import PaperTrader
    from privateye.execution.shadow_tracker import ShadowTracker
    from privateye.risk.manager import RiskManager

    bt_cfg = get_backtest_config(cfg)
    ex_cfg = get_exchange_config(cfg)
    initial_capital = bt_cfg.get("initial_capital", 10000.0)
    symbols = cfg.get("symbols", ["BTC/USDT"])
    timeframes = cfg.get("timeframes", ["1h"])
    bar_window = cfg.get("data", {}).get("bar_window", 500)
    poll_interval = cfg.get("data", {}).get("poll_interval_seconds", 60)

    bus = AsyncEventBus()
    alert_manager = _build_alert_manager(bus, cfg)
    compliance_engine = _build_compliance_engine(cfg)
    pipeline = DataPipeline(bus, bar_window=bar_window)
    strategies = _build_strategies(cfg)
    asset_filter, exposure_monitor, _ = _build_advanced_risk(cfg)

    # Phase 4: feedback store + model versions
    _p4_feedback_store = _build_feedback_store(cfg)
    _p4_artifacts_dir = cfg.get("ml", {}).get("artifacts_dir", "privateye/models/artifacts")
    _p4_model_versions_cb = _build_model_versions_callback(_p4_artifacts_dir)

    # Phase 3: slippage predictor, smart router, pre-trade analyzer
    slip_predictor_shadow = _build_slippage_predictor(cfg)
    smart_router_shadow   = _build_smart_order_router(cfg, slip_predictor_shadow)
    pre_trade_shadow      = _build_pre_trade_analyzer(cfg, slip_predictor_shadow)

    risk_manager = RiskManager(
        cfg.get("risk", {}),
        exposure_monitor=exposure_monitor,
        asset_filter=asset_filter,
        pre_trade_analyzer=pre_trade_shadow,
    )

    sim_exchange = SimulatedExchange(
        initial_capital=initial_capital,
        fee_maker=bt_cfg.get("fee_maker", 0.001),
        fee_taker=bt_cfg.get("fee_taker", 0.001),
        slippage_pct=bt_cfg.get("slippage_pct", 0.0005),
    )
    exec_engine = ExecutionEngine(bus, simulator=sim_exchange, smart_router=smart_router_shadow)
    paper_trader = PaperTrader(bus, sim_exchange)
    _maybe_fit_vwap(exec_engine, cfg)

    async def on_signal(signal: Any) -> None:
        allowed, c_reason = compliance_engine.check_symbol(signal.symbol)
        if not allowed:
            log.debug(f"[Compliance] Signal suppressed: {c_reason}")
            return
        portfolio = sim_exchange.get_portfolio_state()
        approved, reason, order = risk_manager.evaluate_signal(signal, portfolio)
        if approved and order:
            await exec_engine.submit_order(order)
        else:
            log.debug(f"Signal rejected: {reason}")

    async def on_market_data(snapshot: Any) -> None:
        for strategy in strategies:
            entry_signals = strategy.on_data(snapshot)
            for s in entry_signals:
                await on_signal(s)
            portfolio = sim_exchange.get_portfolio_state()
            exit_signals = strategy.on_bar_end(snapshot, portfolio)
            for s in exit_signals:
                await on_signal(s)

    async def on_fill(fill: Any) -> None:
        portfolio = sim_exchange.get_portfolio_state()
        for strategy in strategies:
            if strategy.strategy_id == fill.strategy_id:
                strategy.on_fill(fill, portfolio)
        log.info(f"[Shadow] Fill: {fill.side.value} {fill.quantity:.6f} {fill.symbol} @ {fill.price:.2f}")

    bus.subscribe(EventType.MARKET_DATA, on_market_data)
    bus.subscribe(EventType.FILL, on_fill)

    # Phase 6 + Phase 3: wire ShadowTracker with Reality Score — price queries only, no real orders
    shadow_cfg = cfg.get("robustness", {}).get("shadow_trading", {})
    shadow_mode2_cfg = cfg.get("execution", {}).get("shadow_mode_2", {})
    # Merge shadow_mode_2 keys into the config dict passed to ShadowTracker
    merged_shadow_cfg = {**shadow_cfg, **shadow_mode2_cfg}
    baseline_slippage = float(bt_cfg.get("slippage_pct", 0.0005))
    adapter = ExchangeAdapter({
        **ex_cfg,
        "sandbox": ex_cfg.get("sandbox", True),  # read-only, keep sandbox
    })
    tracker = ShadowTracker(bus, adapter, merged_shadow_cfg, baseline_slippage=baseline_slippage)
    bus.subscribe(EventType.FILL, tracker.on_fill)

    # Phase 6: wire OnlineLearner if enabled
    _maybe_wire_online_learner(bus, strategies, cfg)

    # Start dashboard — Phase 3: pass get_execution_stats=tracker.get_reality_stats
    from privateye.dashboard.server import start_dashboard
    dashboard_cfg = cfg.get("dashboard", {})
    dashboard_task = asyncio.create_task(
        start_dashboard(
            host=dashboard_cfg.get("host", "0.0.0.0"),
            port=dashboard_cfg.get("port", 8081),
            get_portfolio=sim_exchange.get_portfolio_state,
            get_trades=lambda: sim_exchange.trade_records,
            get_fills=lambda: sim_exchange.fills,
            exec_engine=exec_engine,
            risk_manager=risk_manager,
            initial_capital=initial_capital,
            get_alerts=alert_manager.get_recent_alerts,
            get_compliance=compliance_engine.get_status,
            get_execution_stats=tracker.get_reality_stats,
            get_model_versions=_p4_model_versions_cb,        # Phase 4
            get_feedback=_p4_feedback_store.get_recent,      # Phase 4
            post_feedback=_p4_feedback_store.submit_feedback, # Phase 4
        )
    )

    provider = BinanceProvider(
        api_key=ex_cfg.get("api_key", ""),
        api_secret=ex_cfg.get("api_secret", ""),
        sandbox=ex_cfg.get("sandbox", True),
        poll_interval=poll_interval,
    )

    async def on_bar(symbol: str, tf: str, bars) -> None:
        await pipeline.push_bars(symbol, tf, bars)

    bus_task = asyncio.create_task(bus.run())
    log.info(
        f"Shadow trading started. Dashboard: http://localhost:{dashboard_cfg.get('port', 8081)}\n"
        f"ShadowTracker enabled={tracker.enabled}, "
        f"threshold={shadow_cfg.get('divergence_alert_threshold_pct', 2.0):.1f}%"
    )

    try:
        await provider.start(symbols, timeframes, on_bar)
    except asyncio.CancelledError:
        pass
    finally:
        report = tracker.get_report()
        if report.n_fills > 0:
            log.info(
                f"[ShadowTracker] Session report: {report.n_fills} fills, "
                f"mean_slippage={report.mean_slippage_pct:.4%}, "
                f"max={report.max_slippage_pct:.4%}"
            )
        provider.stop()
        bus.stop()
        dashboard_task.cancel()
        bus_task.cancel()


def run_walk_forward(cfg: dict[str, Any]):
    """
    Walk-forward backtesting: rolling IS/OOS windows, per-fold BacktestEngine,
    aggregate OOS Sharpe, stability score, and efficiency ratio.
    """
    from privateye.backtesting.walk_forward import WalkForwardConfig, WalkForwardEngine
    from privateye.config.loader import get_backtest_config
    from privateye.data.loaders import load_bars

    wf_cfg_raw = cfg.get("walk_forward", {})
    wf_config  = WalkForwardConfig.from_config(wf_cfg_raw)
    bt_cfg     = get_backtest_config(cfg)
    data_dir   = bt_cfg.get("data_dir", "data/historical")
    symbols    = cfg.get("symbols", ["BTC/USDT"])
    timeframe  = cfg.get("primary_timeframe", "1h")

    engine = WalkForwardEngine(cfg, wf_config)

    last_report = None
    for symbol in symbols:
        bars = load_bars(symbol, timeframe, data_dir=data_dir)
        if bars.empty:
            log.error(f"No data for {symbol} {timeframe}. Run fetch_data.py first.")
            continue
        report = engine.run(bars, symbol, timeframe)
        print(report)
        last_report = report
    return last_report


def run_kfold(cfg: dict[str, Any]):
    """
    Purged K-fold cross-validation: rotate held-out test folds across the full
    dataset, purge look-back boundary bars, report CV mean/std/stability and
    overfitting gap (mean IS score – CV mean score).
    """
    from privateye.backtesting.kfold import PurgedKFoldConfig, PurgedKFoldEngine
    from privateye.config.loader import get_backtest_config
    from privateye.data.loaders import load_bars

    kf_cfg_raw = cfg.get("kfold", {})
    kf_config  = PurgedKFoldConfig.from_config(kf_cfg_raw)
    metric     = kf_cfg_raw.get("metric", "sharpe_ratio")
    bt_cfg     = get_backtest_config(cfg)
    data_dir   = bt_cfg.get("data_dir", "data/historical")
    symbols    = cfg.get("symbols", ["BTC/USDT"])
    timeframe  = cfg.get("primary_timeframe", "1h")

    engine = PurgedKFoldEngine(cfg, kf_config, metric=metric)

    last_report = None
    for symbol in symbols:
        bars = load_bars(symbol, timeframe, data_dir=data_dir)
        if bars.empty:
            log.error(f"No data for {symbol} {timeframe}. Run fetch_data.py first.")
            continue
        report = engine.run(bars, symbol, timeframe)
        print(report)
        last_report = report
    return last_report


def run_optimize(cfg: dict[str, Any]):
    """
    Grid-search parameter optimization over a strategy's parameter space.

    Reads ``cfg["optimization"]``, builds an ``OptimizationConfig``, loads bars
    from CSV, runs ``GridSearchOptimizer``, and prints the ranked
    ``OptimizationReport``.  Returns the report (or ``None`` if no data).
    """
    from privateye.backtesting.optimizer import GridSearchOptimizer, OptimizationConfig
    from privateye.config.loader import get_backtest_config
    from privateye.data.loaders import load_bars

    opt_raw       = cfg.get("optimization", {})
    strategy_name = opt_raw.get("strategy_name", "directional")
    raw_grid      = opt_raw.get("param_grid", {}).get(strategy_name, {})

    opt_config = OptimizationConfig(
        strategy_name=strategy_name,
        param_grid=raw_grid,
        metric=opt_raw.get("metric", "sharpe_ratio"),
        min_trades=int(opt_raw.get("min_trades", 5)),
        symbol=cfg.get("symbols", ["BTC/USDT"])[0],
        timeframe=cfg.get("primary_timeframe", "1h"),
    )

    data_dir = get_backtest_config(cfg).get("data_dir", "data/historical")
    bars     = load_bars(opt_config.symbol, opt_config.timeframe, data_dir=data_dir)

    if bars.empty:
        log.error(
            f"No data for {opt_config.symbol} {opt_config.timeframe}. "
            "Run fetch_data.py first."
        )
        return None

    optimizer = GridSearchOptimizer(cfg, opt_config)
    report    = optimizer.run(bars)
    print(report)
    return report


def run_golden_suite(cfg: dict[str, Any]):
    """Run the golden backtest suite and save artifacts/golden_results_v1.0.json."""
    from privateye.backtesting.golden_suite import run_golden_backtest
    return run_golden_backtest(cfg)


def run_portfolio_backtest(cfg: dict[str, Any]):
    """
    Time-synchronized multi-symbol portfolio backtest.

    All symbols share a single SimulatedExchange so cash depletion from
    one symbol constrains entries in all others.  Reads bars from
    ``cfg["backtesting"]["data_dir"]``, replays them in lock-step, and
    prints a ``PortfolioBacktestReport`` with per-symbol and aggregate metrics.
    Returns the report (or ``None`` when no data is available).
    """
    from privateye.backtesting.portfolio_engine import PortfolioBacktestEngine
    from privateye.config.loader import get_backtest_config
    from privateye.data.loaders import load_bars

    bt_cfg   = get_backtest_config(cfg)
    data_dir = bt_cfg.get("data_dir", "data/historical")
    symbols  = cfg.get("symbols", ["BTC/USDT"])
    timeframe = cfg.get("primary_timeframe", "1h")

    bars_by_symbol: dict = {}
    for symbol in symbols:
        bars = load_bars(symbol, timeframe, data_dir=data_dir)
        if bars.empty:
            log.warning(f"[portfolio_backtest] No data for {symbol} — skipping")
            continue
        bars_by_symbol[symbol] = bars

    if not bars_by_symbol:
        log.error(
            "No bars loaded for any symbol. Run fetch_data.py first."
        )
        return None

    engine = PortfolioBacktestEngine(cfg)
    report = engine.run(bars_by_symbol, timeframe)
    print(report)
    return report


def cli() -> None:
    parser = argparse.ArgumentParser(description="PrivateEyeTrader")
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper", "live", "shadow",
                 "walk_forward", "optimize", "kfold", "golden_suite",
                 "portfolio_backtest"],
        help="Override mode from config",
    )
    parser.add_argument("--config", default="privateye/config/settings.yaml",
                        help="Path to settings.yaml")
    args = parser.parse_args()

    from privateye.config.loader import load_config
    cfg = load_config(args.config)

    mode = args.mode or cfg.get("mode", "paper")
    log.info(f"Starting PrivateEyeTrader — mode={mode}")

    if mode == "backtest":
        run_backtest(cfg)
    elif mode == "paper":
        asyncio.run(run_paper(cfg))
    elif mode == "live":
        asyncio.run(run_live(cfg))
    elif mode == "shadow":
        asyncio.run(run_shadow(cfg))
    elif mode == "walk_forward":
        run_walk_forward(cfg)
    elif mode == "optimize":
        run_optimize(cfg)
    elif mode == "kfold":
        run_kfold(cfg)
    elif mode == "golden_suite":
        run_golden_suite(cfg)
    elif mode == "portfolio_backtest":
        run_portfolio_backtest(cfg)
    else:
        log.error(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
