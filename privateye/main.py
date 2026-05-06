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

from privateye.utils.logging import setup_logging, get_logger

setup_logging()
log = get_logger()


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
    learner = OnlineLearner(bus, fusion._ensemble, ol_cfg, artifacts_dir=artifacts_dir)
    bus.subscribe(EventType.MARKET_DATA, learner.on_market_data)
    log.info(
        "[main] OnlineLearner wired — retrain every "
        f"{ol_cfg.get('retrain_every_bars', 200)} bars on "
        f"{ol_cfg.get('models', ['gbm', 'lstm', 'regime'])}"
    )


def run_backtest(cfg: dict[str, Any]):
    from privateye.backtesting.engine import BacktestEngine
    from privateye.backtesting.simulator import SimulatedExchange
    from privateye.config.loader import get_backtest_config
    from privateye.data.providers.csv_provider import CSVProvider
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
    provider = CSVProvider(data_dir)

    symbols = cfg.get("symbols", ["BTC/USDT"])
    timeframe = cfg.get("primary_timeframe", "1h")

    last_report = None
    for symbol in symbols:
        bars = provider.load(symbol, timeframe)
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
    risk_manager = RiskManager(
        cfg.get("risk", {}),
        exposure_monitor=exposure_monitor,
        asset_filter=asset_filter,
    )

    sim_exchange = SimulatedExchange(
        initial_capital=initial_capital,
        fee_maker=bt_cfg.get("fee_maker", 0.001),
        fee_taker=bt_cfg.get("fee_taker", 0.001),
        slippage_pct=bt_cfg.get("slippage_pct", 0.0005),
    )
    exec_engine = ExecutionEngine(bus, simulator=sim_exchange)
    paper_trader = PaperTrader(bus, sim_exchange)

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
        for strategy in strategies:
            entry_signals = strategy.on_data(snapshot)
            for s in entry_signals:
                await on_signal(s)
            portfolio = sim_exchange.get_portfolio_state()
            exit_signals = strategy.on_bar_end(snapshot, portfolio)
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
    from privateye.config.loader import get_exchange_config
    ex_cfg = get_exchange_config(cfg)

    if ex_cfg.get("sandbox", True):
        log.warning("LIVE mode but sandbox=true — forcing paper mode for safety")
        await run_paper(cfg)
        return

    # Confirm live intent
    if not ex_cfg.get("api_key"):
        log.error("BINANCE_API_KEY not set. Cannot run live.")
        return

    log.critical("LIVE MODE — real orders will be placed on Binance")
    from privateye.execution.adapter import ExchangeAdapter
    from privateye.core.event_bus import AsyncEventBus
    from privateye.data.pipeline import DataPipeline
    from privateye.data.providers.binance import BinanceProvider
    from privateye.execution.engine import ExecutionEngine
    from privateye.risk.manager import RiskManager
    from privateye.core.types import EventType

    bus = AsyncEventBus()
    alert_manager = _build_alert_manager(bus, cfg)
    compliance_engine = _build_compliance_engine(cfg)
    pipeline = DataPipeline(bus, bar_window=cfg.get("data", {}).get("bar_window", 500))
    strategies = _build_strategies(cfg)
    asset_filter, exposure_monitor, _ = _build_advanced_risk(cfg)
    risk_manager = RiskManager(
        cfg.get("risk", {}),
        exposure_monitor=exposure_monitor,
        asset_filter=asset_filter,
    )
    adapter = ExchangeAdapter(ex_cfg)
    exec_engine = ExecutionEngine(bus, adapter=adapter)
    symbols = cfg.get("symbols", ["BTC/USDT"])
    timeframes = cfg.get("timeframes", ["1h"])
    primary_tf = cfg.get("primary_timeframe", "1h")
    poll_interval = cfg.get("data", {}).get("poll_interval_seconds", 60)
    live_portfolio: list = []  # mutable container for current state

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
    dashboard_task = asyncio.create_task(
        start_dashboard(
            host=dashboard_cfg.get("host", "0.0.0.0"),
            port=dashboard_cfg.get("port", 8081),
            get_portfolio=lambda: asyncio.run(exec_engine.get_portfolio_state(symbols)),
            get_trades=lambda: [],
            get_fills=lambda: [],
            exec_engine=exec_engine,
            risk_manager=risk_manager,
            initial_capital=float(
                cfg.get("backtesting", {}).get("initial_capital", 10000.0)
            ),
            get_alerts=alert_manager.get_recent_alerts,
            get_compliance=compliance_engine.get_status,
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
    risk_manager = RiskManager(
        cfg.get("risk", {}),
        exposure_monitor=exposure_monitor,
        asset_filter=asset_filter,
    )

    sim_exchange = SimulatedExchange(
        initial_capital=initial_capital,
        fee_maker=bt_cfg.get("fee_maker", 0.001),
        fee_taker=bt_cfg.get("fee_taker", 0.001),
        slippage_pct=bt_cfg.get("slippage_pct", 0.0005),
    )
    exec_engine = ExecutionEngine(bus, simulator=sim_exchange)
    paper_trader = PaperTrader(bus, sim_exchange)

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

    # Phase 6: wire ShadowTracker — price queries only, no real orders
    shadow_cfg = cfg.get("robustness", {}).get("shadow_trading", {})
    adapter = ExchangeAdapter({
        **ex_cfg,
        "sandbox": ex_cfg.get("sandbox", True),  # read-only, keep sandbox
    })
    tracker = ShadowTracker(bus, adapter, shadow_cfg)
    bus.subscribe(EventType.FILL, tracker.on_fill)

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
    from privateye.data.providers.csv_provider import CSVProvider

    wf_cfg_raw = cfg.get("walk_forward", {})
    wf_config  = WalkForwardConfig.from_config(wf_cfg_raw)
    bt_cfg     = get_backtest_config(cfg)
    data_dir   = bt_cfg.get("data_dir", "data/historical")
    symbols    = cfg.get("symbols", ["BTC/USDT"])
    timeframe  = cfg.get("primary_timeframe", "1h")

    engine   = WalkForwardEngine(cfg, wf_config)
    provider = CSVProvider(data_dir)

    last_report = None
    for symbol in symbols:
        bars = provider.load(symbol, timeframe)
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
    from privateye.data.providers.csv_provider import CSVProvider

    kf_cfg_raw = cfg.get("kfold", {})
    kf_config  = PurgedKFoldConfig.from_config(kf_cfg_raw)
    metric     = kf_cfg_raw.get("metric", "sharpe_ratio")
    bt_cfg     = get_backtest_config(cfg)
    data_dir   = bt_cfg.get("data_dir", "data/historical")
    symbols    = cfg.get("symbols", ["BTC/USDT"])
    timeframe  = cfg.get("primary_timeframe", "1h")

    engine   = PurgedKFoldEngine(cfg, kf_config, metric=metric)
    provider = CSVProvider(data_dir)

    last_report = None
    for symbol in symbols:
        bars = provider.load(symbol, timeframe)
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
    from privateye.data.providers.csv_provider import CSVProvider

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
    bars     = CSVProvider(data_dir).load(opt_config.symbol, opt_config.timeframe)

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


def cli() -> None:
    parser = argparse.ArgumentParser(description="PrivateEyeTrader")
    parser.add_argument("--mode", choices=["backtest", "paper", "live", "shadow", "walk_forward", "optimize", "kfold"],
                        help="Override mode from config")
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
    else:
        log.error(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
