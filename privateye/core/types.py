"""Core domain types shared across all modules."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

import pandas as pd


class EventType(str, Enum):
    MARKET_DATA       = "market_data"
    SIGNAL            = "signal"
    ORDER             = "order"
    FILL              = "fill"
    RISK_BREACH       = "risk_breach"
    SYSTEM            = "system"
    KILL              = "kill"
    # Phase 6 — Robustness
    PROVIDER_HEALTH   = "provider_health"    # payload: dict from get_health()
    MODEL_UPDATED     = "model_updated"      # payload: {"models_updated": [...], "bars_seen": N}
    SHADOW_DIVERGENCE = "shadow_divergence"  # payload: ShadowFillRecord


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class AlertLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    CRITICAL = "CRITICAL"


@dataclass
class OHLCV:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    timeframe: str = ""


@dataclass
class DataSnapshot:
    """Rolling window of OHLCV bars for one symbol/timeframe, published as MARKET_DATA events."""
    symbol: str
    timeframe: str
    bars: pd.DataFrame  # columns: timestamp, open, high, low, close, volume; latest bar = last row
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def latest(self) -> pd.Series:
        return self.bars.iloc[-1]

    @property
    def close(self) -> float:
        return float(self.bars["close"].iloc[-1])

    @property
    def high(self) -> float:
        return float(self.bars["high"].iloc[-1])

    @property
    def low(self) -> float:
        return float(self.bars["low"].iloc[-1])

    @property
    def volume(self) -> float:
        return float(self.bars["volume"].iloc[-1])


@dataclass
class TradingSignal:
    """Signal emitted by a strategy. Contains full XAI trail in metadata."""
    symbol: str
    direction: Direction
    confidence: float          # 0.0–1.0
    entry_price: float
    stop_price: float
    target_price: float
    strategy_id: str
    timeframe: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = field(default_factory=dict)
    # metadata keys: reason (str), indicators (dict), features (dict)

    @property
    def risk_reward(self) -> float:
        if self.direction == Direction.LONG:
            reward = self.target_price - self.entry_price
            risk = self.entry_price - self.stop_price
        else:
            reward = self.entry_price - self.target_price
            risk = self.stop_price - self.entry_price
        return reward / risk if risk > 0 else 0.0


@dataclass
class Position:
    symbol: str
    side: Direction
    quantity: float
    entry_price: float
    stop_price: float
    target_price: float
    strategy_id: str
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    bars_held: int = 0
    unrealised_pnl: float = 0.0
    trailing_stop: float = 0.0
    entry_fees: float = 0.0       # Phase 13: fee paid to open this position; combined into TradeRecord.fees on close

    @property
    def notional(self) -> float:
        return abs(self.quantity * self.entry_price)

    def current_pnl(self, current_price: float) -> float:
        if self.side == Direction.LONG:
            return (current_price - self.entry_price) * self.quantity
        return (self.entry_price - current_price) * self.quantity


@dataclass
class Order:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float               # limit price or last known price for market
    stop_price: float
    strategy_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    post_only: bool = False


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    strategy_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    realised_pnl: float = 0.0  # populated on closing fills


@dataclass
class PortfolioState:
    equity: float
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    daily_pnl: float = 0.0
    daily_drawdown_pct: float = 0.0
    peak_equity: float = 0.0
    total_trades: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def invested(self) -> float:
        return sum(p.notional for p in self.positions.values())

    @property
    def drawdown_pct(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity


@dataclass
class TradeRecord:
    """Immutable record of a completed round-trip trade (entry + exit)."""
    symbol: str
    side: Direction
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    fees: float
    strategy_id: str
    exit_reason: str  # "stop", "target", "timeout", "signal_reversal", "kill"
    bars_held: int
