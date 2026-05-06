# PrivateEyeTrader — Architecture Reference

This document describes the high-level design of PrivateEyeTrader: how components fit together, why they were designed that way, and where to look when something needs changing.

---

## Table of Contents

- [System Overview](#system-overview)
- [Event Bus — The Backbone](#event-bus--the-backbone)
- [Data Layer](#data-layer)
- [Strategy Pipeline](#strategy-pipeline)
- [Risk Pipeline](#risk-pipeline)
- [Execution Layer](#execution-layer)
- [ML Subsystem](#ml-subsystem)
- [Backtesting & Validation](#backtesting--validation)
- [Compliance Layer](#compliance-layer)
- [Dashboard & Alerts](#dashboard--alerts)
- [Configuration System](#configuration-system)
- [Runtime Modes](#runtime-modes)
- [Key Design Decisions](#key-design-decisions)
- [Adding a New Strategy](#adding-a-new-strategy)
- [Adding a New Risk Gate](#adding-a-new-risk-gate)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        LIVE / PAPER / SHADOW                     │
│                                                                   │
│  ┌──────────────┐    MARKET_DATA     ┌────────────────────────┐  │
│  │ DataPipeline │ ────────────────→  │  AsyncEventBus         │  │
│  │  (Binance /  │                    │                        │  │
│  │   Bybit /    │  ←── subscribe ──  │  SIGNAL, FILL,         │  │
│  │   CSV)       │                    │  RISK_BREACH, KILL,    │  │
│  └──────────────┘                    │  MODEL_UPDATED, …      │  │
│                                      └───────────┬────────────┘  │
│                                                  │               │
│              ┌───────────────────────────────────┼─────────────┐ │
│              │              subscribers           │             │ │
│              ▼                    ▼               ▼             │ │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────┐   │ │
│  │  Strategies     │  │  AlertManager    │  │  OnlineLearner│   │ │
│  │  (Directional,  │  │  (Telegram/email)│  │  (retrain on  │   │ │
│  │   MeanRev,      │  └──────────────────┘  │   live data)  │   │ │
│  │   Fusion/ML)    │                        └──────────────┘   │ │
│  └────────┬────────┘                                           │ │
│           │ SIGNAL                                             │ │
│           ▼                                                    │ │
│  ┌─────────────────┐    compliance gate                       │ │
│  │  ComplianceEngine│ ─────────────────┐                      │ │
│  └────────┬────────┘                   │ blocked → drop        │ │
│           │ allowed                    ▼                       │ │
│           ▼                                                    │ │
│  ┌─────────────────┐                                           │ │
│  │  RiskManager    │ ─── 10-gate pipeline ─┐                  │ │
│  │  (10 gates)     │                       │ rejected → drop   │ │
│  └────────┬────────┘                       ▼                  │ │
│           │ approved Order                                     │ │
│           ▼                                                    │ │
│  ┌─────────────────┐                                           │ │
│  │ ExecutionEngine │ → SimulatedExchange (paper)              │ │
│  │                 │ → ExchangeAdapter/ccxt (live)            │ │
│  └────────┬────────┘                                          │ │
│           │ FILL event                                        │ │
│           ▼                                                   │ │
│  ┌─────────────────┐  ┌──────────────────┐                   │ │
│  │  PaperTrader /  │  │  ShadowTracker   │                   │ │
│  │  SQLiteStore    │  │  (slippage diff) │                   │ │
│  └─────────────────┘  └──────────────────┘                   │ │
│                                                               │ │
│  ┌──────────────────────────────────────────────────────────┐ │ │
│  │  FastAPI Dashboard  (REST + WebSocket)                   │ │ │
│  │  http://localhost:8081                                   │ │ │
│  └──────────────────────────────────────────────────────────┘ │ │
└───────────────────────────────────────────────────────────────┘ │
```

---

## Event Bus — The Backbone

**File:** `privateye/core/event_bus.py`

All inter-component communication goes through `AsyncEventBus`. No component holds a direct reference to another; they communicate exclusively by publishing and subscribing to named events.

### Why this design?

- **Decoupling** — strategies don't know about execution; execution doesn't know about alerts. Each component can be tested in isolation by mocking the bus.
- **Testability** — publishing a fake `MARKET_DATA` event is enough to drive a strategy through its full logic in unit tests.
- **Extensibility** — adding a new subscriber (e.g. a database logger) doesn't touch any existing code.

### EventType catalogue

| Event | Publisher | Subscribers |
|-------|-----------|-------------|
| `MARKET_DATA` | DataPipeline | Strategies, OnlineLearner |
| `SIGNAL` | Strategies | (consumed internally via callback, not bus) |
| `ORDER` | ExecutionEngine | SimulatedExchange / ccxt adapter |
| `FILL` | SimulatedExchange / ccxt adapter | PaperTrader, ShadowTracker, AlertManager, SQLiteStore |
| `RISK_BREACH` | RiskManager | AlertManager |
| `KILL` | ExecutionEngine (`/api/kill`) | AlertManager, RiskManager |
| `PROVIDER_HEALTH` | BinanceProvider / BybitProvider | AlertManager, Dashboard |
| `MODEL_UPDATED` | OnlineLearner | AlertManager |
| `SHADOW_DIVERGENCE` | ShadowTracker | AlertManager |

### Publish vs dispatch

- **`bus.publish(event, payload)`** — queues the event; processed on the next iteration of the event loop. Use this from async contexts.
- **`bus.dispatch(event, payload)`** — fires handlers immediately (synchronously within the async context). Use when ordering matters.
- **`bus.publish_sync(event, payload)`** — fire-and-forget from synchronous code (spawns a task).

---

## Data Layer

```
BinanceProvider ──┐
BybitProvider  ──┤─→ DataPipeline ──→ MARKET_DATA event ──→ bus
CSVProvider    ──┘         │
                           └─→ FeatureExtractor (50 features)
                           └─→ SQLiteStore (persistence)
```

**Files:** `privateye/data/`

### DataPipeline

`DataPipeline` aggregates multi-timeframe bars from one or more providers, maintains a rolling window of `bar_window` bars per symbol/timeframe, and publishes `DataSnapshot` objects as `MARKET_DATA` events.

A `DataSnapshot` bundles:
- The full rolling bar DataFrame
- Convenience properties: `latest`, `close`, `high`, `low`, `volume`
- Symbol and timeframe metadata

Strategies receive `DataSnapshot` objects, not raw bars — they never touch the provider directly.

### Providers

| Provider | Use case | Key behaviour |
|----------|----------|--------------|
| `BinanceProvider` | Live / paper | WebSocket OHLCV stream; circuit breaker with 30-min auto-recovery |
| `BybitProvider` | Fallback | Identical pattern to Binance |
| `CSVProvider` | Backtest | Loads `data/historical/{SYMBOL}_{timeframe}.csv`; no network I/O |

Both live providers implement a circuit breaker: after 5 consecutive fetch failures, the circuit opens and no further fetches are attempted until the auto-recovery probe succeeds or `reset_circuit_breaker()` is called manually.

### Feature Extractor

`FeatureExtractor` transforms a raw OHLCV DataFrame into a 50-column ML feature matrix:
- Momentum: RSI (14, 21), ROC, MFI
- Trend: EMA ratios (9/21/50/200), ADX
- Volatility: ATR, Bollinger Band width, Keltner Channel
- Volume: OBV, VWAP ratio, volume Z-score
- Pattern: candle body ratio, wick ratios
- Alt-data: funding rate, open interest, Fear & Greed index (when enabled)

---

## Strategy Pipeline

```
DataSnapshot
     │
     ▼
AbstractStrategy.on_data(snapshot)
     │
     ├─→ indicator calculation (via indicators/library.py)
     ├─→ entry condition check
     ├─→ if conditions met: build TradingSignal
     └─→ publish SIGNAL or return signal directly

AbstractStrategy.on_bar_end(snapshot, portfolio)
     └─→ check exit conditions (trailing stop, timeout) for open positions
         → build exit TradingSignal if needed
```

**Files:** `privateye/strategies/`

### Signal structure

```python
@dataclass
class TradingSignal:
    symbol:      str
    direction:   Direction        # LONG | SHORT | FLAT (FLAT = close position)
    confidence:  float            # 0.0–1.0
    entry_price: float | None     # None = market order
    stop_price:  float | None
    target_price: float | None
    strategy_id: str
    metadata:    dict             # reason, indicators, top_features (SHAP), etc.
```

### DirectionalStrategy logic

1. Require minimum `bar_window // 4` bars of warmup
2. Compute EMA(200), MACD(12,26,9), RSI(14), ATR(14)
3. **Long entry:** MACD histogram turning positive + close > EMA(200) + RSI < overbought threshold
4. **Exit:** trailing stop (2×ATR below peak close) or `max_bars_in_trade` timeout
5. No pyramiding — one position per symbol at a time

### FusionStrategy logic

1. Run `DirectionalStrategy` logic to get a raw signal
2. **Regime gate:** if `RegimeDetector` says bear or high-vol → suppress signal
3. **GBM gate:** if `GBMClassifier` probability < threshold → suppress signal
4. **LSTM weight:** adjust confidence using `LSTMForecaster` directional probability
5. **SHAP annotation:** attach `top_features` list from `GBMClassifier.get_shap_explanation()`
6. Emit final weighted signal

---

## Risk Pipeline

`RiskManager` sits between strategies and execution. Every `TradingSignal` passes through 10 sequential gates. The first failure short-circuits the pipeline.

**File:** `privateye/risk/manager.py`

```
TradingSignal
     │
     ├─ Gate 1: halt check           — is the system halted?
     ├─ Gate 2: direction check      — already in same-side position?
     ├─ Gate 3: restricted assets    — symbol in cfg.risk.restricted_assets?
     ├─ Gate 4: confidence threshold — signal.confidence >= min_confidence?
     ├─ Gate 5: asset filter         — volume/volatility floors (Phase 5)
     ├─ Gate 6: black-swan guard     — flash crash / funding extreme? (Phase 5)
     ├─ Gate 7: position sizing      — calculate quantity (fixed-risk/Kelly/vol-target)
     ├─ Gate 8: exposure monitor     — concentration / beta limits (Phase 5)
     ├─ Gate 9: cash check           — sufficient free cash for the sized order?
     └─ Gate 10: daily drawdown      — breach 5% limit?
          │
          ▼
     approved Order  →  ExecutionEngine
```

### Position sizing methods

| Method | Formula | Config key |
|--------|---------|-----------|
| `fixed_risk` | `(equity × risk_pct) / (entry - stop)` | `sizing_method: "fixed_risk"` |
| `kelly` | Half-Kelly: `f* = (edge/odds) × 0.5` | `sizing_method: "kelly"` |
| `volatility_targeted` | `target_vol / current_vol × equity / price` | `sizing_method: "volatility_targeted"` |

All methods are capped by `max_position_notional_pct` (default 20% of equity).

---

## Execution Layer

**Files:** `privateye/execution/`

### Paper mode

`ExecutionEngine` routes approved `Order` objects to `SimulatedExchange`:

```
Order → SimulatedExchange.submit_order()
             │
             └─→ process_bar(bar):
                   • market orders fill at open of next bar
                   • limit orders fill if bar low/high crosses price
                   • partial fills capped at 30% of bar volume
                   • slippage applied (0.05% default)
                   • fee applied (0.1% default)
                   → emit FILL event
```

### Live mode

`ExecutionEngine` routes to `ExchangeAdapter` (ccxt):

```
Order → ExchangeAdapter.submit_order()
             │
             └─→ ccxt.create_order() on Binance
                 → poll for fill → emit FILL event
```

### TWAP / VWAP

For large orders (`notional > twap_threshold_usd`), `ExecutionEngine` slices the order into equal time-chunks via `TWAPExecutor` before submitting. `VWAPExecutor` slices proportionally to recent volume.

### Fee optimiser

`FeeOptimizer` estimates round-trip cost for a signal before routing. If `expected_return < fee_adjusted_min_return`, the signal is dropped — preventing trades where fees exceed potential profit.

---

## ML Subsystem

**Files:** `privateye/models/`

### Training pipeline

```
scripts/train_models.py
    │
    ├─→ CSVProvider → load bars
    ├─→ FeatureExtractor → 50-column feature matrix
    ├─→ LSTMForecaster.fit(bars)
    ├─→ GBMClassifier.fit(X, y)
    ├─→ RegimeDetector.fit(bars)
    └─→ RLPolicy.learn(env)
         │
         └─→ save artifacts to privateye/models/artifacts/
```

### Inference pipeline (live/paper)

```
DataSnapshot
    │
    ▼
ModelEnsemble.predict(bars)
    ├─→ RegimeDetector.predict()  → regime label (0-3)
    ├─→ LSTMForecaster.predict()  → directional probability
    ├─→ GBMClassifier.predict()   → quality probability
    ├─→ GBMClassifier.get_shap_explanation() → feature importances
    └─→ returns dict: {direction, confidence, regime, gbm_prob, shap}
```

### Online learning

`OnlineLearner` subscribes to `MARKET_DATA`:

```
on_market_data(snapshot):
    │
    ├─→ append latest bar to rolling buffer (max 2,000 bars)
    ├─→ increment bars_since_retrain
    └─→ if bars_since_retrain >= retrain_every_bars (200):
             acquire _retrain_lock (skip if already retraining)
             for each model in ["gbm", "lstm", "regime"]:
                 1. save_checkpoint(model)         — versioned backup
                 2. _retrain_model(buffer_bars)    — warm-start continuation
                 3. validation gate: new_loss <= old_loss × 1.10?
                    ✓ promote: replace artifacts, reload ensemble
                    ✗ reject: restore from checkpoint, log warning
             publish MODEL_UPDATED event
             reset bars_since_retrain = 0
```

### Model artefacts

All trained models are saved under `privateye/models/artifacts/`:

```
artifacts/
├── lstm_forecaster.pt          # PyTorch state_dict
├── gbm_classifier.pkl          # XGBoost booster
├── regime_detector.pkl         # HMM model (hmmlearn)
├── rl_policy.zip               # SB3 PPO zip
└── checkpoints/
    └── gbm_classifier_20240615_143022/
        ├── gbm_classifier.pkl
        └── meta.json           # bars_seen, saved_at, artifact_files
```

---

## Backtesting & Validation

**Files:** `privateye/backtesting/`

### BacktestEngine replay loop

```python
for i in range(len(bars)):
    snapshot = DataSnapshot(bars.iloc[:i+1], symbol, timeframe)

    # Phase 5: black-swan check on bar i
    if black_swan_guard and black_swan_guard.check(bar, portfolio):
        exchange.flatten_all()
        continue

    signals = strategy.on_data(snapshot)
    for signal in signals:
        order = risk_manager.evaluate_signal(signal, portfolio, bars, trades)
        if order:
            exchange.submit_order(order)

    exchange.process_bar(bar)     # fills run against bar i's OHLCV

    exit_signals = strategy.on_bar_end(snapshot, portfolio)
    for signal in exit_signals:
        ...

    equity_curve.append(exchange.portfolio.equity)
```

Critical invariant: `bars.iloc[:i+1]` — only bars up to and including bar `i` are ever visible to the strategy. No future data leaks.

### Walk-forward

```
Total bars (n)
│
├── Fold 1: IS [0 … train_bars]   OOS [train_bars … train_bars+test_bars]
├── Fold 2: IS [step … step+train_bars]   OOS [...]
└── ...

For each fold:
    is_report  = BacktestEngine.run(is_bars)
    oos_report = BacktestEngine.run(oos_bars)   ← fresh SimulatedExchange
    efficiency_ratio = oos_sharpe / is_sharpe
```

Each fold creates completely independent instances of `SimulatedExchange`, `RiskManager`, and strategies — no state bleeds between folds.

### Purged K-fold

```
Fold k (of K):
    test window:  [test_start … test_end]
    purge window: [test_start - purge_bars … test_start]   ← excluded from training
    embargo:      [test_end … test_end + embargo_bars]     ← excluded from training
    train bars:   everything outside the above windows

    is_report  = BacktestEngine.run(train_bars)
    cv_report  = BacktestEngine.run(test_bars)
    cv_score   = getattr(cv_report, metric)
```

The purge prevents information leakage through autocorrelated features (e.g., a 200-bar EMA carries information about bars that are "in the future" relative to the test window). The embargo prevents leakage through strategies that take time to react to a trend.

---

## Compliance Layer

**Files:** `privateye/compliance/`

```
TradingSignal
     │
     ▼
ComplianceEngine.check_symbol(symbol)
     ├─→ SanctionsChecker.check(symbol)    — OFAC + known fraud list
     └─→ JurisdictionFilter.check(symbol)  — country-specific restrictions
          │
          ├─ blocked → log warning, drop signal
          └─ allowed → continue to RiskManager


After backtest:
ComplianceEngine.run_post_analysis(trades)
     └─→ WashSaleDetector.detect(trades)
          └─→ emit WashSaleFlag for each loss + re-entry within window

ComplianceEngine.export_tax_report(trades, symbol)
     └─→ reports/tax.py: compute_tax_lots() → FIFO matching
         export_tax_csv() → CSV to data/reports/
```

Compliance runs as a synchronous layer — no async required, no event bus involvement. The `get_status()` method is a simple callable passed to the dashboard router.

---

## Dashboard & Alerts

**Files:** `privateye/dashboard/`, `privateye/alerts/`

```
FastAPI app (port 8081)
    │
    ├─ GET /api/* endpoints    ← query portfolio, trades, metrics on demand
    ├─ POST /api/kill          ← flatten all + halt
    └─ WS /ws                  ← push updates every ws_push_interval_seconds


AlertManager (subscribes to bus)
    ├─ on FILL          → notify_fill()        → Telegram + email
    ├─ on RISK_BREACH   → notify_risk_breach() → Telegram + email
    ├─ on KILL          → notify_kill_switch() → Telegram + email
    ├─ on PROVIDER_HEALTH (circuit_open=True) → CRITICAL alert
    ├─ on MODEL_UPDATED → INFO alert
    └─ on SHADOW_DIVERGENCE → INFO alert if |slippage| > threshold

    Rate limiter: same event type suppressed for rate_limit_seconds (60s)
    History:      deque(maxlen=history_size) — viewable at GET /api/alerts
```

The dashboard is read-only for all endpoints except `/api/kill` and `/api/resume`. It holds references to `get_portfolio`, `get_trades`, `get_fills`, `get_alerts`, and `get_compliance` callables — it never mutates state directly.

---

## Configuration System

**File:** `privateye/config/loader.py`, `settings.yaml`

The entire system is driven by a single YAML file. `main.py` loads it once and passes the resulting dict down the call stack. No global config singletons.

```python
# In main.py
cfg = yaml.safe_load(open(config_path))

# Helper accessors (config/loader.py)
get_risk_config(cfg)       → cfg["risk"]
get_strategy_config(cfg, "directional") → cfg["strategies"]["directional"]
get_backtest_config(cfg)   → cfg["backtesting"]
```

All components accept `cfg: dict` — they are unaware of the YAML file. This makes them trivially testable: pass a minimal dict in tests, no files required.

### Config override via CLI

```bash
# Override the mode without editing settings.yaml
privateye --mode paper --config /path/to/custom_settings.yaml
```

---

## Runtime Modes

All modes are implemented as top-level functions in `privateye/main.py` and dispatched from `cli()`.

| Mode | Function | What runs | Orders placed |
|------|----------|-----------|--------------|
| `backtest` | `run_backtest(cfg)` | BacktestEngine on CSV bars | None (simulated) |
| `paper` | `run_paper(cfg)` | Full live stack; SimulatedExchange | None (simulated) |
| `live` | `run_live(cfg)` | Full live stack; ccxt ExchangeAdapter | **Real orders** |
| `shadow` | `run_shadow(cfg)` | Paper + ShadowTracker slippage analysis | None (simulated) |
| `walk_forward` | `run_walk_forward(cfg)` | Rolling IS/OOS BacktestEngine | None |
| `optimize` | `run_optimize(cfg)` | Grid-search BacktestEngine | None |
| `kfold` | `run_kfold(cfg)` | Purged K-fold BacktestEngine | None |

The live/paper/shadow modes are `async` functions run under `asyncio.run()`. The validation modes (walk_forward, optimize, kfold) are synchronous and CPU-bound.

### Startup sequence (paper / live / shadow)

```
1. Build AsyncEventBus
2. Build AlertManager → subscribe to bus
3. Build ComplianceEngine
4. Build DataPipeline (providers)
5. Build FeatureExtractor
6. Build strategies (_build_strategies)
7. Build RiskManager + advanced risk (_build_advanced_risk)
8. Build ExecutionEngine + PaperTrader / ExchangeAdapter
9. Maybe wire OnlineLearner (if robustness.online_learning.enabled)
10. Maybe wire ShadowTracker (shadow mode only)
11. Subscribe on_signal handler to MARKET_DATA events
12. Start dashboard (asyncio.create_task)
13. Start DataPipeline (asyncio.create_task)
14. Run event bus loop (bus.run())
```

---

## Key Design Decisions

### 1. No global state

Every component receives its dependencies via constructor or function arguments. This makes the codebase easy to test (no import-time side effects) and eliminates a whole class of bugs from shared mutable state.

### 2. Deferred imports in hot paths

Functions like `_run_fold()` (WalkForwardEngine), `_run_combination()` (GridSearchOptimizer), and `export_tax_report()` (ComplianceEngine) defer their imports to inside the function body. This prevents circular import issues when the module is loaded and avoids paying the import cost when the feature isn't used.

### 3. Fresh instances per backtest fold

Walk-forward and K-fold folds each create entirely new `SimulatedExchange`, `RiskManager`, and strategy instances. This guarantees no state bleeds between folds — a common source of bugs in backtesting frameworks.

### 4. Compliance as a pre-trade gate, not post-filter

`check_symbol()` runs synchronously in the `on_signal` callback before the order even reaches the risk manager. Blocked signals are dropped with a log line; they never enter the order pipeline.

### 5. Fail-open for unknown jurisdictions

`JurisdictionFilter` returns `(True, "")` for any country code not in `JURISDICTION_RULES`. This is intentional — it's better to allow trading in an unrecognised jurisdiction than to silently block a legitimate user who set the wrong code.

### 6. Alert rate-limiting is per event type

The rate limiter in `AlertManager` throttles by `EventType`, not by message content. Two different fills happening within 60 seconds produce one Telegram message. This prevents alert spam in volatile markets without losing the first notification.

---

## Adding a New Strategy

1. Create `privateye/strategies/my_strategy.py` extending `AbstractStrategy`:

```python
from privateye.strategies.base import AbstractStrategy
from privateye.core.types import DataSnapshot, PortfolioState, TradingSignal, Direction

class MyStrategy(AbstractStrategy):
    strategy_id = "my_strategy"

    def __init__(self, cfg: dict) -> None:
        self.my_param = cfg.get("my_param", 14)

    def on_data(self, snapshot: DataSnapshot) -> list[TradingSignal]:
        if len(snapshot.bars) < self.my_param + 1:
            return []   # not enough warmup bars
        # ... compute indicators, check conditions
        return [TradingSignal(
            symbol=snapshot.symbol,
            direction=Direction.LONG,
            confidence=0.75,
            entry_price=snapshot.close,
            stop_price=snapshot.close * 0.98,
            target_price=snapshot.close * 1.04,
            strategy_id=self.strategy_id,
            metadata={"reason": "my condition met"},
        )]

    def on_bar_end(self, snapshot: DataSnapshot, portfolio: PortfolioState) -> list[TradingSignal]:
        return []   # implement exit logic if needed
```

2. Add a config section in `settings.yaml`:

```yaml
strategies:
  my_strategy:
    enabled: true
    my_param: 14
```

3. Register in `_build_strategies()` in `privateye/main.py`:

```python
my_cfg = get_strategy_config(cfg, "my_strategy")
if my_cfg.get("enabled", False):
    strategies.append(MyStrategy(my_cfg))
```

4. Write tests in `tests/test_my_strategy.py` using a mock `DataSnapshot`.

---

## Adding a New Risk Gate

Risk gates are implemented as methods on `RiskManager` and called sequentially in `evaluate_signal()`.

1. Add a method to `privateye/risk/manager.py`:

```python
def _check_my_gate(
    self,
    signal: TradingSignal,
    portfolio: PortfolioState,
) -> tuple[bool, str]:
    """Return (allowed, reason). reason='' when allowed."""
    if some_condition:
        return False, "My gate: reason for rejection"
    return True, ""
```

2. Insert the call in `evaluate_signal()` in the correct position:

```python
allowed, reason = self._check_my_gate(signal, portfolio)
if not allowed:
    log.debug(f"[RiskManager] {reason}")
    return None
```

3. Add any required config keys to `settings.yaml` and read them in `RiskManager.__init__()`.

4. Write tests covering the allow and reject paths.
