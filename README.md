# PrivateEyeTrader

An AI-powered cryptocurrency trading system built for personal, solo use. Combines classical technical analysis with machine learning (LSTM, GBM, regime detection, RL), a 10-gate risk manager, walk-forward validation, and a live web dashboard — all configurable from a single YAML file.

> **Risk Disclaimer:** Trading cryptocurrencies carries substantial risk of loss. This software is provided for educational and research purposes. Past backtest performance is not indicative of future results. Never trade with money you cannot afford to lose. See the [full disclaimer](#risk-disclaimer) at the bottom of this document.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Backtesting](#backtesting)
  - [Parameter Optimisation](#parameter-optimisation)
  - [Walk-Forward Validation](#walk-forward-validation)
  - [Purged K-Fold Cross-Validation](#purged-k-fold-cross-validation)
  - [Paper Trading](#paper-trading)
  - [Shadow Mode](#shadow-mode)
  - [Live Trading](#live-trading)
- [Sample Backtest Results](#sample-backtest-results)
- [Dashboard](#dashboard)
- [ML Models](#ml-models)
- [Compliance & Tax Reporting](#compliance--tax-reporting)
- [Alerts](#alerts)
- [Docker Deployment](#docker-deployment)
- [Development](#development)
- [Risk Disclaimer](#risk-disclaimer)

---

## Features

### Strategy Engine
- **DirectionalStrategy** — MACD crossover + EMA(200) trend filter + RSI guard
- **MeanReversionStrategy** — Bollinger Band touch + RSI extreme reversal
- **FusionStrategy** — ML-weighted ensemble of both strategies, gated by regime detector and GBM confidence scorer

### Machine Learning (optional, `pip install privateye-trader[ml]`)
- **LSTM Forecaster** — 2-layer LSTM(256) trained on 60-bar lookback, 50 technical features
- **GBM Classifier** — XGBoost signal quality gate with SHAP explainability
- **Regime Detector** — 4-state Hidden Markov Model (bull / bear / sideways / high-vol)
- **RL Policy** — PPO agent (Stable-Baselines3) for position sizing
- **Online Learning** — incremental model retraining every N bars on a rolling buffer, with validation gate preventing regressions

### Risk Management
10-gate pre-trade approval pipeline:

| Gate | Purpose |
|------|---------|
| Halt check | Block if circuit breaker is tripped |
| Direction check | Skip if already in same-side position |
| Confidence threshold | Require ≥55% signal confidence |
| Asset filter | Volume and volatility floor checks |
| Position sizing | Fixed-risk / Kelly / Volatility-targeted |
| Exposure monitor | Single position ≤40% equity, total ≤80%, beta ≤2.0 |
| Cash check | Ensure sufficient free capital |
| Max daily drawdown | 5% circuit breaker |
| Black-Swan guard | Flash-crash, volume spike, funding extremes, stablecoin depeg |
| Compliance gate | Sanctions list and jurisdiction filter |

### Backtesting & Validation
- **Zero look-ahead bias** — strict bar-by-bar replay
- **Realistic simulation** — 0.1% maker/taker fees, 0.05% slippage, partial fills capped at 30% of bar volume
- **Walk-forward validation** — rolling IS/OOS windows (252-day train, 63-day test)
- **Purged K-fold CV** — 5-fold with 100-bar purge + 10-bar embargo to prevent feature leakage
- **Monte Carlo stress testing** — 1,000 trade permutations for ruin probability estimate
- **Parameter optimisation** — grid search ranked by Sharpe / PnL / Calmar

### Infrastructure
- **Live dashboard** — FastAPI + WebSocket, equity curve, drawdown, positions, alerts
- **Dual exchange support** — Binance (primary) + Bybit (fallback)
- **Auto-recovery** — provider circuit breaker with 30-min auto-reset probe
- **Shadow trading** — paper sim + live price comparison before going live
- **Tax reporting** — FIFO cost-basis CSV, short/long-term gain classification
- **Compliance** — jurisdiction filter (US, CN, EG, MA), OFAC sanctions list, wash-sale detector
- **Alerts** — Telegram + email, rate-limited, history stored in dashboard

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-username/PrivateEyeTrader.git
cd PrivateEyeTrader
pip install -e .

# 2. Download historical data (requires free Binance API key in .env)
python scripts/fetch_data.py --symbol BTC/USDT --timeframe 1h --days 730

# 3. Run a backtest
privateye --mode backtest

# 4. Start paper trading with live dashboard
privateye --mode paper
# → open http://localhost:8081
```

---

## Installation

### Requirements

- Python ≥ 3.11
- A free Binance account (even sandbox mode requires API keys)

### Base install

```bash
pip install -e .
```

Installs all core dependencies: CCXT, FastAPI, pandas, numpy, pandas-ta, aiosqlite, loguru, etc.

### With ML support

```bash
pip install -e ".[ml]"
```

Adds: PyTorch, XGBoost, LightGBM, SHAP, scikit-learn, hmmlearn, Stable-Baselines3.

> **Note:** ML dependencies are large (~3 GB with PyTorch). Only needed when `ml.enabled: true` in `settings.yaml`.

### Development extras

```bash
pip install -e ".[dev]"
```

Adds: pytest, pytest-asyncio, pytest-cov.

### Environment variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

```dotenv
# Required for live/paper trading
BINANCE_API_KEY=your_key
BINANCE_SECRET=your_secret

# Optional: Bybit fallback
BYBIT_API_KEY=
BYBIT_SECRET=

# Optional: Telegram alerts
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Optional: Email alerts
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASS=app_password
ALERT_EMAIL_TO=you@gmail.com
```

---

## Configuration

All behaviour is controlled by `privateye/config/settings.yaml`. The most important sections:

```yaml
mode: paper          # backtest | paper | live | shadow | walk_forward | optimize | kfold

symbols:
  - "BTC/USDT"

primary_timeframe: "1h"

exchanges:
  binance:
    enabled: true
    sandbox: true    # ← set false for live orders

risk:
  max_risk_per_trade_pct: 0.01     # 1% of equity per trade
  max_daily_drawdown_pct: 0.05     # 5% circuit breaker
  sizing_method: "fixed_risk"      # fixed_risk | kelly | volatility_targeted

strategies:
  directional:
    enabled: true
    macd_fast: 12
    macd_slow: 26
    rsi_overbought: 70

ml:
  enabled: false      # set true after running scripts/train_models.py
  strategy: "fusion"

compliance:
  jurisdiction: ""    # "US" | "CN" | "" (disabled)
  sanctions:
    enabled: true
  tax_reporting:
    enabled: false    # set true to auto-export CSV after each backtest
```

See [`privateye/config/settings.yaml`](privateye/config/settings.yaml) for the full reference with all options and inline comments.

---

## Usage

All modes are accessible via the `privateye` CLI (or `python -m privateye.main`):

```
privateye --mode <mode> [--config path/to/settings.yaml]
```

### Backtesting

Requires historical CSV data in `data/historical/`.

```bash
# Fetch 2 years of 1h BTC/USDT bars
python scripts/fetch_data.py --symbol BTC/USDT --timeframe 1h --days 730

# Run backtest
privateye --mode backtest
```

Output:
```
--------------------------------------------------
  BACKTEST REPORT
--------------------------------------------------
  Trades          : 87  (W:47 L:40)
  Win Rate        : 54.0%
  Profit Factor   : 1.62
  Total PnL       : +4,231.40 (+42.3%)
  Ann. Return     : +19.8%
  Initial Capital : 10000.00
  Final Equity    : 14231.40
  Max Drawdown    : 14.2%  (1,420.00)
  Sharpe Ratio    : 1.31
  Sortino Ratio   : 1.87
  Calmar Ratio    : 0.89
  ...
```

### Parameter Optimisation

Grid-search strategy parameters and rank by Sharpe (or `total_pnl`, `calmar_ratio`):

```yaml
# settings.yaml
optimization:
  strategy_name: "directional"
  metric: "sharpe_ratio"
  min_trades: 10
  param_grid:
    directional:
      macd_fast: [8, 10, 12]
      macd_slow: [22, 26, 30]
      macd_signal: [7, 9]
```

```bash
privateye --mode optimize
```

Output:
```
============================================================
  OPTIMIZATION REPORT
============================================================
  Strategy      : directional
  Metric        : sharpe_ratio
  Combinations  : 18 total, 14 passed (min_trades=10)
------------------------------------------------------------
  Best params   : macd_fast=10, macd_slow=26, macd_signal=9
  Best score    : 1.43
------------------------------------------------------------
  RANK   SCORE   MACD_FAST  MACD_SLOW  MACD_SIG  TRADES
     1   1.430          10         26         9      94
     2   1.312          12         26         9      87
  ...
```

### Walk-Forward Validation

Tests out-of-sample performance across rolling windows to detect overfitting:

```bash
privateye --mode walk_forward
```

Key outputs:
- **Mean OOS Sharpe** across folds
- **Stability score** — fraction of folds with positive OOS Sharpe
- **Efficiency ratio** — OOS Sharpe ÷ IS Sharpe (near 1.0 = no overfitting)

### Purged K-Fold Cross-Validation

5-fold CV with purge + embargo windows that prevent autocorrelation leakage:

```bash
privateye --mode kfold
```

Key output: **overfitting gap** = mean IS score − mean CV score. Values near 0 indicate the strategy generalises well.

### Paper Trading

Simulated trading against live Binance price feeds, with full dashboard:

```bash
privateye --mode paper
# Dashboard → http://localhost:8081
```

Paper trading uses the same strategy and risk logic as live mode but routes all orders to a `SimulatedExchange` — no real money moves.

### Shadow Mode

Paper simulation with real-time slippage measurement against live exchange prices:

```bash
privateye --mode shadow
```

Each fill is compared to the live mid-price at fill time. A divergence report is printed on exit, showing whether your simulated fills are realistic before committing real capital.

### Live Trading

```bash
# First: set sandbox: false in settings.yaml and confirm .env credentials
privateye --mode live
```

> ⚠️ **Live mode places real orders on Binance.** Ensure `sandbox: false` is intentional, start with minimal capital, and verify paper/shadow results first.

### Train ML Models

```bash
# Fetch data first
python scripts/fetch_data.py --symbol BTC/USDT --timeframe 1h --days 730

# Train all models (LSTM, GBM, Regime, RL)
python scripts/train_models.py

# Enable in settings.yaml
# ml.enabled: true
# ml.strategy: "fusion"
```

---

## Sample Backtest Results

> ⚠️ **These are illustrative results on synthetic data for documentation purposes only. They do not represent actual trading performance and should not be used to make investment decisions.**

The following illustrates what the backtest report format looks like for a `DirectionalStrategy` (MACD 12/26/9 + EMA 200 + RSI 14) on BTC/USDT 1h bars, $10,000 initial capital, 0.1% fees, 0.05% slippage:

```
--------------------------------------------------
  BACKTEST REPORT
--------------------------------------------------
  Trades          : 87  (W:47 L:40)
  Win Rate        : 54.0%
  Profit Factor   : 1.62
  Total PnL       : +4,231.40 (+42.3%)
  Ann. Return     : +19.8%
  Initial Capital : 10000.00
  Final Equity    : 14231.40
  Max Drawdown    : 14.2%  (-1,420.00)
  Sharpe Ratio    : 1.31
  Sortino Ratio   : 1.87
  Calmar Ratio    : 0.89
  Avg Win         : +$126.40
  Avg Loss        : -$78.20
  Avg Bars Held   : 18.4
--------------------------------------------------
  TRADE COSTS
--------------------------------------------------
  Total Fees Paid : 184.20 (1.84% of capital)
  Fee Drag        : 4.2% of gross PnL
  Slippage Cost   : 0.41% of capital
  Limit Fill Rate : N/A
--------------------------------------------------
  ADVANCED ANALYTICS
--------------------------------------------------
  Ulcer Index     : 0.0421
  VaR 95%         : -3.12%
  CVaR 95%        : -4.87%
  Max Consec Wins : 6
  Max Consec Loss : 4
  Omega Ratio     : 2.14
--------------------------------------------------
```

**Walk-forward summary (5 folds, 63-day OOS each):**

| Fold | IS Sharpe | OOS Sharpe | Efficiency |
|------|-----------|------------|------------|
| 1    | 1.42      | 1.18       | 0.83       |
| 2    | 1.31      | 0.97       | 0.74       |
| 3    | 1.55      | 1.24       | 0.80       |
| 4    | 1.28      | 0.88       | 0.69       |
| 5    | 1.38      | 1.11       | 0.80       |
| **Mean** | **1.39** | **1.08** | **0.77** |

Stability score: 100% (all folds OOS Sharpe > 0). Overfitting gap: 0.31.

---

## Dashboard

Start any live mode and open `http://localhost:8081`.

| Endpoint | Description |
|----------|-------------|
| `GET /api/portfolio` | Equity, cash, daily PnL, drawdown |
| `GET /api/positions` | Open positions with entry/stop/target |
| `GET /api/trades` | Closed trade history |
| `GET /api/metrics` | Sharpe, win rate, profit factor, etc. |
| `GET /api/equity-curve` | Time-series equity for charting |
| `GET /api/drawdown-curve` | Time-series drawdown % |
| `GET /api/alerts` | Recent alert history |
| `GET /api/compliance` | Jurisdiction, sanctions, wash-sale flags |
| `GET /api/status` | Halted status and halt reason |
| `POST /api/kill` | Emergency flatten all positions |
| `WS /ws` | Real-time push (equity, PnL, positions) every 5 s |

---

## ML Models

When `ml.enabled: true` and `ml.strategy: "fusion"`, the system uses `FusionStrategy`:

```
Signal flow:
  DirectionalStrategy → raw signal
         ↓
  RegimeDetector → suppress if bear/high-vol regime
         ↓
  LSTMForecaster → directional probability
         ↓
  GBMClassifier  → quality gate (suppress if prob < 0.30)
         ↓
  RLPolicy       → position size adjustment (optional)
         ↓
  FusionStrategy → emit weighted signal with SHAP explanation
```

Each emitted signal carries a `top_features` list (from SHAP) explaining which indicators drove the decision:

```python
signal.metadata["top_features"] = [
    {"feature": "rsi_14",        "importance": 0.31},
    {"feature": "macd_hist",     "importance": 0.24},
    {"feature": "funding_rate",  "importance": 0.18},
    ...
]
```

### Online Learning

When `robustness.online_learning.enabled: true`, the `OnlineLearner` subscribes to `MARKET_DATA` events and retrains models every 200 bars on a rolling 2,000-bar buffer. A validation gate prevents promotion of a newly-trained model if its validation loss regresses by more than 10%.

---

## Compliance & Tax Reporting

```yaml
compliance:
  jurisdiction: "US"      # blocks XRP, LBRY (SEC enforcement)
  sanctions:
    enabled: true         # always blocks LUNA, FTT, TORN, SQUID
    extra_banned: []      # add your own symbols
  tax_reporting:
    enabled: true
    output_dir: "data/reports"
    wash_sale_window_days: 30
```

After each backtest, PrivateEyeTrader will:
1. Log any wash-sale patterns (loss + same-symbol re-entry within 30 days)
2. Export `data/reports/tax_BTC_USDT.csv` with FIFO cost-basis details

CSV columns: `symbol, quantity, proceeds, cost_basis, gain_loss, entry_time, exit_time, holding_days, is_long_term, fees`

This CSV is compatible with manual import into Koinly, TokenTax, or accountant spreadsheets.

> **Note:** Wash-sale rules (IRC §1091) currently apply to securities, not cryptocurrency, under US law. PrivateEyeTrader's wash-sale flags are advisory only and may become relevant if regulations change or in other jurisdictions.

---

## Alerts

Configure once in `.env`; PrivateEyeTrader handles the rest.

**Events that trigger alerts:**

| Event | Default level |
|-------|--------------|
| Trade fill | INFO |
| Daily drawdown breach | CRITICAL |
| Kill switch activated | CRITICAL |
| Exchange circuit tripped | CRITICAL |
| Shadow slippage > threshold | INFO |
| ML model updated | INFO |

Rate limiting (default 60 s per event type) prevents spam. Alert history is viewable at `GET /api/alerts`.

---

## Docker Deployment

```bash
# Start paper trading in Docker
docker compose up -d

# View logs
docker compose logs -f trader

# Change to live mode
# 1. Edit privateye/config/settings.yaml → mode: live, sandbox: false
# 2. docker compose restart trader

# Stop
docker compose down
```

The container mounts `./data` and `./logs` as volumes so your trade history and model artefacts persist across restarts.

---

## Development

```bash
# Install with dev extras
pip install -e ".[dev]"

# Run the full test suite (409 tests)
pytest tests/ -v

# Run a specific phase
pytest tests/test_compliance.py -v

# Coverage report
pytest tests/ --cov=privateye --cov-report=html
```

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/fetch_data.py` | Download historical OHLCV bars from Binance |
| `scripts/fetch_alt_data.py` | Fetch funding rates and Fear & Greed index |
| `scripts/train_models.py` | Full offline training of LSTM, GBM, Regime, RL |
| `scripts/backtest.py` | Quick backtest runner (CLI wrapper) |
| `scripts/retrain_weekly.bat` | Windows Task Scheduler weekly retrain |
| `scripts/register_task.bat` | Register the weekly retrain as a Windows scheduled task |

### Project structure

```
privateye/
├── alerts/           AlertManager, Notifier (Telegram + email)
├── backtesting/      BacktestEngine, SimulatedExchange, metrics, walk-forward, K-fold, optimizer
├── compliance/       ComplianceEngine, JurisdictionFilter, SanctionsChecker, WashSaleDetector
├── config/           YAML loader, settings.yaml
├── core/             AsyncEventBus, types (TradeRecord, Signal, Fill, Order, Position, …)
├── dashboard/        FastAPI app, REST + WebSocket routes, HTML templates
├── data/             DataPipeline, BinanceProvider, BybitProvider, CSVProvider, FeatureExtractor, SQLite store
├── execution/        ExecutionEngine, PaperTrader, ExchangeAdapter, TWAP, VWAP, FeeOptimizer, ShadowTracker
├── indicators/       Technical indicator library (EMA, RSI, MACD, BB, ATR, …)
├── models/           LSTM, GBM, Regime, RL, OnlineLearner, ModelCheckpoint
├── reports/          Tax CSV export
├── risk/             RiskManager, SizingRouter, AssetFilter, ExposureMonitor, BlackSwanGuard
├── strategies/       DirectionalStrategy, MeanReversionStrategy, FusionStrategy
└── utils/            Logging (loguru), time helpers
```

---

## Risk Disclaimer

**READ THIS BEFORE USING THIS SOFTWARE.**

1. **Cryptocurrency trading is highly speculative and risky.** The value of cryptocurrencies can fall to zero. You can lose your entire investment.

2. **Past performance does not guarantee future results.** Backtest results — including those shown in this README — are based on historical data and simulated conditions. They do not reflect real trading performance and are not a reliable predictor of future returns.

3. **Backtest results are inherently optimistic.** Simulations cannot perfectly replicate real-world conditions including: exchange downtime, liquidity gaps, API rate limits, regulatory changes, flash crashes, and the psychological impact of real losses.

4. **This software is not financial advice.** PrivateEyeTrader is a research and educational tool. Nothing in this codebase, README, or associated documentation constitutes investment advice, financial advice, trading advice, or any other type of advice.

5. **You are solely responsible for your trading decisions.** The authors and contributors of PrivateEyeTrader accept no liability for financial losses incurred through the use of this software.

6. **Start with paper trading.** Validate the system thoroughly in paper mode and shadow mode before committing any real capital. Consider the shadow mode divergence report — significant slippage between simulated and real fills is a warning sign.

7. **Risk only what you can afford to lose.** The default configuration risks 1% of equity per trade and halts trading at a 5% daily drawdown. These are conservative defaults; do not increase them without fully understanding the implications.

8. **Regulatory compliance is your responsibility.** Cryptocurrency regulations vary by jurisdiction and change frequently. Ensure your trading activity complies with local laws, tax obligations, and any applicable financial regulations. The compliance features in this software are advisory aids, not legal guarantees.

9. **API key security.** Never share your exchange API keys. Use keys with trading permissions only — never withdrawal permissions. IP-whitelist your keys where possible.

10. **This software is provided "as is"**, without warranty of any kind. The authors make no representations about its fitness for any particular purpose.

---

*PrivateEyeTrader v1.0.0 — Built for personal, educational use.*
