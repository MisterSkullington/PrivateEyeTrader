"""Abstract strategy base class. All strategies implement this interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from privateye.core.types import DataSnapshot, Direction, Position, TradingSignal


class AbstractStrategy(ABC):
    """
    All strategies receive DataSnapshot events and emit TradingSignal objects.

    Lifecycle:
        on_data(snapshot) → list[TradingSignal]   (called every bar)
        on_fill(fill, portfolio) → None           (position tracking)
        on_bar_end(portfolio) → list[TradingSignal]  (exit / trailing stop updates)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.strategy_id: str = config.get("strategy_id", self.__class__.__name__)
        self.enabled: bool = config.get("enabled", True)
        # Tracked open positions, keyed by symbol
        self._positions: dict[str, Position] = {}

    @abstractmethod
    def on_data(self, snapshot: DataSnapshot) -> list[TradingSignal]:
        """Called on each new bar for a symbol. Return signals (may be empty list)."""
        ...

    def on_fill(self, fill: Any, portfolio: Any) -> None:
        """Called when an order is filled. Update internal position state."""

    def on_bar_end(self, snapshot: DataSnapshot, portfolio: Any) -> list[TradingSignal]:
        """Called after on_data for the same bar. Used for trailing stop / exit updates."""
        return []

    def _has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def _open_position(self, symbol: str, position: Position) -> None:
        self._positions[symbol] = position

    def _close_position(self, symbol: str) -> Position | None:
        return self._positions.pop(symbol, None)

    def _get_position(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    def _flat_signal(self, snapshot: DataSnapshot, reason: str) -> TradingSignal:
        close = float(snapshot.bars["close"].iloc[-1])
        return TradingSignal(
            symbol=snapshot.symbol,
            direction=Direction.FLAT,
            confidence=1.0,
            entry_price=close,
            stop_price=close,
            target_price=close,
            strategy_id=self.strategy_id,
            timeframe=snapshot.timeframe,
            metadata={"reason": reason},
        )

    @property
    def open_positions(self) -> dict[str, Position]:
        return dict(self._positions)
