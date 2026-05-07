"""
Mean-reversion strategy using Bollinger Bands + RSI.

Entry logic:
  Long:  close crosses below lower BB AND RSI < oversold threshold
  Short: close crosses above upper BB AND RSI > overbought threshold

Exit logic:
  Target: return to BB midline (20-period SMA)
  Stop:   1×ATR from entry
  Timeout: max_bars_in_trade
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from privateye.core.types import DataSnapshot, Direction, Position, TradingSignal
from privateye.indicators.library import compute_all
from privateye.strategies.base import AbstractStrategy
from privateye.utils.logging import get_logger

log = get_logger()


class MeanReversionStrategy(AbstractStrategy):
    STRATEGY_ID = "mean_reversion"

    def __init__(self, config: dict[str, Any]) -> None:
        config.setdefault("strategy_id", self.STRATEGY_ID)
        super().__init__(config)
        self.timeframe: str = config.get("timeframe", "1h")
        self.bb_period: int = config.get("bb_period", 20)
        self.bb_std: float = config.get("bb_std", 2.0)
        self.rsi_period: int = config.get("rsi_period", 14)
        self.rsi_overbought: float = config.get("rsi_overbought", 70)
        self.rsi_oversold: float = config.get("rsi_oversold", 30)
        self.atr_stop_mult: float = config.get("atr_stop_multiplier", 1.0)
        self.max_bars: int = config.get("max_bars_in_trade", 24)

    def on_data(self, snapshot: DataSnapshot) -> list[TradingSignal]:
        if not self.enabled:
            return []
        if snapshot.timeframe != self.timeframe:
            return []
        bars = snapshot.bars
        if len(bars) < self.bb_period + 5:
            return []
        if self._has_position(snapshot.symbol):
            return []

        ind = compute_all(bars, self.config)
        signal = self._check_entry(snapshot, ind)
        return [signal] if signal else []

    def on_bar_end(self, snapshot: DataSnapshot, portfolio: Any) -> list[TradingSignal]:
        # Hotfix 2026-05-07: timeframe guard — on_bar_end must not run on
        # non-target timeframes, otherwise bars_held inflates 5× per poll and
        # exit conditions are checked against stale closes from other timeframes
        # (e.g. yesterday's 1d close incorrectly tripping the stop).
        if snapshot.timeframe != self.timeframe:
            return []
        if not self._has_position(snapshot.symbol):
            return []
        pos = self._get_position(snapshot.symbol)
        assert pos is not None
        close = snapshot.close
        pos.bars_held += 1

        # Hotfix 2026-05-07 (Option B): skip exit checks on the entry bar
        # (bars_held == 1). The bar's close already has the entry baked in,
        # but stop/target levels were computed from the same close — a bar
        # whose intra-bar move trips them would mis-fire if checked now.
        # See directional.py:on_bar_end for the full rationale.
        exit_reason = None
        if pos.bars_held > 1:
            if pos.side == Direction.LONG:
                if close >= pos.target_price:
                    exit_reason = "target"
                elif close <= pos.stop_price:
                    exit_reason = "stop"
                elif pos.bars_held >= self.max_bars:
                    exit_reason = "timeout"
            else:  # SHORT
                if close <= pos.target_price:
                    exit_reason = "target"
                elif close >= pos.stop_price:
                    exit_reason = "stop"
                elif pos.bars_held >= self.max_bars:
                    exit_reason = "timeout"

        if exit_reason:
            return [self._flat_signal(snapshot, exit_reason)]
        return []

    def _check_entry(self, snapshot: DataSnapshot, ind: pd.DataFrame) -> TradingSignal | None:
        last = ind.iloc[-1]
        prev = ind.iloc[-2] if len(ind) > 1 else last

        close = float(last["close"])
        bb_upper = float(last.get("bb_upper", close))
        bb_lower = float(last.get("bb_lower", close))
        bb_mid = float(last.get("bb_mid", close))
        rsi = float(last.get("rsi_14", 50))
        atr14 = float(last.get("atr_14", close * 0.01))

        prev_close = float(prev["close"])
        prev_bb_lower = float(prev.get("bb_lower", bb_lower))
        prev_bb_upper = float(prev.get("bb_upper", bb_upper))

        # Long: close crosses below lower BB (was above, now below)
        crossed_below_lower = prev_close >= prev_bb_lower and close < bb_lower
        if crossed_below_lower and rsi < self.rsi_oversold:
            stop = close - self.atr_stop_mult * atr14
            target = bb_mid  # mean-revert to midline
            confidence = self._score_long(rsi, bb_mid, close, bb_lower, atr14)
            reason = (
                f"Long {snapshot.symbol} mean-reversion: "
                f"close {close:.2f} crossed below BB_lower {bb_lower:.2f}, "
                f"RSI={rsi:.1f} (oversold)"
            )
            return TradingSignal(
                symbol=snapshot.symbol,
                direction=Direction.LONG,
                confidence=confidence,
                entry_price=close,
                stop_price=stop,
                target_price=target,
                strategy_id=self.strategy_id,
                timeframe=self.timeframe,
                metadata={"reason": reason, "indicators": {
                    "rsi": rsi, "bb_lower": bb_lower, "bb_mid": bb_mid,
                    "bb_upper": bb_upper, "atr": atr14,
                }},
            )

        # Short: close crosses above upper BB
        crossed_above_upper = prev_close <= prev_bb_upper and close > bb_upper
        if crossed_above_upper and rsi > self.rsi_overbought:
            stop = close + self.atr_stop_mult * atr14
            target = bb_mid
            confidence = self._score_short(rsi, bb_mid, close, bb_upper, atr14)
            reason = (
                f"Short {snapshot.symbol} mean-reversion: "
                f"close {close:.2f} crossed above BB_upper {bb_upper:.2f}, "
                f"RSI={rsi:.1f} (overbought)"
            )
            return TradingSignal(
                symbol=snapshot.symbol,
                direction=Direction.SHORT,
                confidence=confidence,
                entry_price=close,
                stop_price=stop,
                target_price=target,
                strategy_id=self.strategy_id,
                timeframe=self.timeframe,
                metadata={"reason": reason, "indicators": {
                    "rsi": rsi, "bb_lower": bb_lower, "bb_mid": bb_mid,
                    "bb_upper": bb_upper, "atr": atr14,
                }},
            )

        return None

    def _score_long(self, rsi, bb_mid, close, bb_lower, atr) -> float:
        score = 0.55
        if rsi < 25:
            score += 0.15
        elif rsi < 30:
            score += 0.1
        distance = (bb_mid - close) / atr if atr > 0 else 1
        if distance > 2:
            score += 0.1
        return min(score, 1.0)

    def _score_short(self, rsi, bb_mid, close, bb_upper, atr) -> float:
        score = 0.55
        if rsi > 75:
            score += 0.15
        elif rsi > 70:
            score += 0.1
        distance = (close - bb_mid) / atr if atr > 0 else 1
        if distance > 2:
            score += 0.1
        return min(score, 1.0)

    def on_fill(self, fill: Any, portfolio: Any) -> None:
        from privateye.core.types import Direction, OrderSide
        symbol = fill.symbol
        if fill.side == OrderSide.BUY and not self._has_position(symbol):
            atr_est = fill.price * 0.01
            pos = Position(
                symbol=symbol, side=Direction.LONG, quantity=fill.quantity,
                entry_price=fill.price,
                stop_price=fill.price - self.atr_stop_mult * atr_est,
                target_price=fill.price * 1.02,  # placeholder, updated by on_data
                strategy_id=self.strategy_id,
            )
            self._open_position(symbol, pos)
        elif fill.side == OrderSide.SELL and self._has_position(symbol):
            self._close_position(symbol)
