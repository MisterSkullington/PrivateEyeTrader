"""
Directional trend-following strategy.

Entry logic:
  Long:  MACD histogram crosses above zero AND close > EMA(200) AND RSI not overbought
  Short: MACD histogram crosses below zero AND close < EMA(200) AND RSI not oversold

Exit logic:
  Trailing stop (2×ATR from highest close since entry for longs, lowest for shorts)
  OR max_bars_in_trade timeout
  OR opposing MACD signal (optional)

Phase 13 audit fixes:
  • H-3: ``on_fill`` mirrors the simulator's Position (with the real stop_price
    from the originating order) instead of fabricating a 1% ATR estimate.
    Also adds SHORT tracking which was previously absent.
  • H-8: ``on_bar_end`` uses an explicit ``if pos is None`` guard instead of
    ``assert``, which is stripped under ``python -O``.
  • H-9: Trailing stop fires when the bar's ``low`` (for LONG) or ``high``
    (for SHORT) crosses the stop level — not the close. Previously, a bar
    that pierced the stop intra-bar but recovered to close above it would
    not exit in backtest, systematically over-reporting performance.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import pandas as pd

from privateye.core.types import DataSnapshot, Direction, Position, TradingSignal
from privateye.indicators.library import compute_all
from privateye.strategies.base import AbstractStrategy
from privateye.utils.logging import get_logger
from privateye.utils.time import now_utc

log = get_logger()


class DirectionalStrategy(AbstractStrategy):
    STRATEGY_ID = "directional"

    def __init__(self, config: dict[str, Any]) -> None:
        config.setdefault("strategy_id", self.STRATEGY_ID)
        super().__init__(config)
        self.timeframe: str = config.get("timeframe", "1h")
        self.macd_fast: int = config.get("macd_fast", 12)
        self.macd_slow: int = config.get("macd_slow", 26)
        self.macd_signal: int = config.get("macd_signal", 9)
        self.ema_trend: int = config.get("ema_trend", 200)
        self.rsi_period: int = config.get("rsi_period", 14)
        self.rsi_overbought: float = config.get("rsi_overbought", 70)
        self.rsi_oversold: float = config.get("rsi_oversold", 30)
        self.atr_stop_mult: float = config.get("atr_stop_multiplier", 2.0)
        self.atr_target_mult: float = config.get("atr_target_multiplier", 4.0)
        self.max_bars: int = config.get("max_bars_in_trade", 48)
        self._extremes: dict[str, float] = {}  # symbol → best close since entry

    def on_data(self, snapshot: DataSnapshot) -> list[TradingSignal]:
        if not self.enabled:
            return []
        if snapshot.timeframe != self.timeframe:
            return []
        bars = snapshot.bars
        if len(bars) < self.ema_trend + 10:
            return []

        ind = compute_all(bars, self.config)
        signals: list[TradingSignal] = []

        if self._has_position(snapshot.symbol):
            # Manage existing position via on_bar_end
            return []

        # Entry signal detection
        signal = self._check_entry(snapshot, ind)
        if signal:
            signals.append(signal)

        return signals

    def on_bar_end(self, snapshot: DataSnapshot, portfolio: Any) -> list[TradingSignal]:
        # Hotfix 2026-05-07: on_bar_end must filter by timeframe like on_data does.
        # Without this filter, the bus dispatches MARKET_DATA for every configured
        # timeframe (1m/5m/1h/4h/1d) per poll cycle, so each poll runs the exit
        # logic 5 times — inflating bars_held, recomputing trailing_stop with the
        # wrong ATR, and (worst) checking stops against stale closes from non-1h
        # timeframes (the 1d close is yesterday's daily, sometimes below stop).
        if snapshot.timeframe != self.timeframe:
            return []
        if not self._has_position(snapshot.symbol):
            return []
        pos = self._get_position(snapshot.symbol)
        if pos is None:                                     # H-8: explicit guard
            return []

        last_bar = snapshot.bars.iloc[-1]
        close = float(last_bar["close"])
        bar_low = float(last_bar["low"])
        bar_high = float(last_bar["high"])
        signals: list[TradingSignal] = []

        # Update trailing stop extreme + check exit
        if pos.side == Direction.LONG:
            self._extremes[snapshot.symbol] = max(
                self._extremes.get(snapshot.symbol, close), close
            )
            trail_ref = self._extremes[snapshot.symbol]
            ind = compute_all(snapshot.bars, self.config)
            a = float(ind["atr_14"].iloc[-1])
            new_stop = trail_ref - self.atr_stop_mult * a
            pos.trailing_stop = max(pos.trailing_stop, new_stop)
            pos.bars_held += 1

            # H-9: stop fires when bar.low pierces the level (not just close).
            # Hotfix 2026-05-07 (Option B): skip exit checks on the entry bar
            # (bars_held == 1). The trailing_stop is still armed/updated above —
            # we only suppress the comparison against bar_low/bar_high/target on
            # the bar where the position was just opened, because those values
            # pre-date the fill and would otherwise trigger an immediate exit
            # on any bar whose range exceeds the ATR-stop distance.
            exit_reason = None
            if pos.bars_held > 1:
                if bar_low <= pos.trailing_stop:
                    exit_reason = "trailing_stop"
                elif pos.bars_held >= self.max_bars:
                    exit_reason = "timeout"
                elif bar_high >= pos.target_price > 0:
                    exit_reason = "target"
        else:  # SHORT
            self._extremes[snapshot.symbol] = min(
                self._extremes.get(snapshot.symbol, close), close
            )
            trail_ref = self._extremes[snapshot.symbol]
            ind = compute_all(snapshot.bars, self.config)
            a = float(ind["atr_14"].iloc[-1])
            new_stop = trail_ref + self.atr_stop_mult * a
            pos.trailing_stop = min(
                pos.trailing_stop if pos.trailing_stop > 0 else new_stop, new_stop
            )
            pos.bars_held += 1

            # H-9: short stop fires when bar.high crosses up through the level.
            # See LONG branch above for the bars_held>1 rationale.
            exit_reason = None
            if pos.bars_held > 1:
                if bar_high >= pos.trailing_stop > 0:
                    exit_reason = "trailing_stop"
                elif pos.bars_held >= self.max_bars:
                    exit_reason = "timeout"
                elif bar_low <= pos.target_price and pos.target_price > 0:
                    exit_reason = "target"

        if exit_reason:
            signals.append(self._flat_signal(snapshot, exit_reason))

        return signals

    def _check_entry(self, snapshot: DataSnapshot, ind: pd.DataFrame) -> TradingSignal | None:
        last = ind.iloc[-1]
        prev = ind.iloc[-2] if len(ind) > 1 else last

        close = float(last["close"])
        ema200 = float(last.get(f"ema_{self.ema_trend}", close))
        rsi = float(last.get("rsi_14", 50))
        macd_hist = float(last.get("macd_hist", 0))
        prev_hist = float(prev.get("macd_hist", 0))
        atr14 = float(last.get("atr_14", close * 0.01))
        adx_val = float(last.get("adx", 0))

        # Require some trend strength (ADX > 20 or relax if unavailable)
        trend_strong = adx_val > 20 or adx_val == 0

        # Long entry
        macd_cross_up = prev_hist <= 0 < macd_hist
        above_trend = close > ema200 * 1.001  # 0.1% buffer
        rsi_ok_long = rsi < self.rsi_overbought

        if macd_cross_up and above_trend and rsi_ok_long and trend_strong:
            stop = close - self.atr_stop_mult * atr14
            target = close + self.atr_target_mult * atr14
            confidence = self._score_long(last, rsi, adx_val)
            reason = (
                f"Long {snapshot.symbol}: MACD hist crossed positive "
                f"({prev_hist:.4f}→{macd_hist:.4f}), "
                f"close {close:.2f} > EMA{self.ema_trend} {ema200:.2f}, "
                f"RSI={rsi:.1f}, ADX={adx_val:.1f}"
            )
            log.debug(reason)
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
                    "macd_hist": macd_hist, "ema200": ema200,
                    "rsi": rsi, "atr": atr14, "adx": adx_val,
                }},
            )

        # Short entry
        macd_cross_down = prev_hist >= 0 > macd_hist
        below_trend = close < ema200 * 0.999
        rsi_ok_short = rsi > self.rsi_oversold

        if macd_cross_down and below_trend and rsi_ok_short and trend_strong:
            stop = close + self.atr_stop_mult * atr14
            target = close - self.atr_target_mult * atr14
            confidence = self._score_short(last, rsi, adx_val)
            reason = (
                f"Short {snapshot.symbol}: MACD hist crossed negative "
                f"({prev_hist:.4f}→{macd_hist:.4f}), "
                f"close {close:.2f} < EMA{self.ema_trend} {ema200:.2f}, "
                f"RSI={rsi:.1f}, ADX={adx_val:.1f}"
            )
            log.debug(reason)
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
                    "macd_hist": macd_hist, "ema200": ema200,
                    "rsi": rsi, "atr": atr14, "adx": adx_val,
                }},
            )

        return None

    def _score_long(self, row: pd.Series, rsi: float, adx: float) -> float:
        score = 0.5
        if rsi < 50:
            score += 0.1
        if rsi < 40:
            score += 0.1
        if adx > 25:
            score += 0.1
        if adx > 35:
            score += 0.1
        bb_pct = float(row.get("bb_pct", 0.5))
        if bb_pct < 0.4:
            score += 0.1
        return min(score, 1.0)

    def _score_short(self, row: pd.Series, rsi: float, adx: float) -> float:
        score = 0.5
        if rsi > 50:
            score += 0.1
        if rsi > 60:
            score += 0.1
        if adx > 25:
            score += 0.1
        if adx > 35:
            score += 0.1
        bb_pct = float(row.get("bb_pct", 0.5))
        if bb_pct > 0.6:
            score += 0.1
        return min(score, 1.0)

    def on_fill(self, fill: Any, portfolio: Any) -> None:
        """Mirror the simulator's authoritative Position into the strategy tracker.

        Phase 13 (H-3): we used to fabricate stop/target from a 1% ATR estimate
        because the original signal's ATR wasn't available at fill time. Now we
        copy the Position the simulator already built (which carries the real
        ``stop_price`` and ``target_price`` from the originating Order).

        Also adds SHORT tracking — previously a SELL fill with no existing
        long was a no-op, leaving the strategy unaware of the open short.
        """
        from privateye.core.types import Direction, OrderSide

        symbol = fill.symbol
        sim_pos = (
            portfolio.positions.get(symbol)
            if portfolio is not None and hasattr(portfolio, "positions")
            else None
        )

        if fill.side == OrderSide.BUY:
            # BUY closes an existing SHORT or opens a LONG
            existing = self._get_position(symbol)
            if existing is not None and existing.side == Direction.SHORT:
                self._close_position(symbol)
                self._extremes.pop(symbol, None)
                return

            if sim_pos is not None and sim_pos.side == Direction.LONG and not self._has_position(symbol):
                pos = dataclasses.replace(sim_pos)            # mirror simulator
                pos.trailing_stop = pos.stop_price             # initialise trail from real stop
                self._open_position(symbol, pos)
                self._extremes[symbol] = fill.price
            elif not self._has_position(symbol):
                # Fallback: simulator didn't expose the position (e.g. live mode);
                # fabricate a conservative stop using configured multiplier on a 1% ATR proxy.
                atr_est = fill.price * 0.01
                pos = Position(
                    symbol=symbol, side=Direction.LONG, quantity=fill.quantity,
                    entry_price=fill.price,
                    stop_price=fill.price - self.atr_stop_mult * atr_est,
                    target_price=fill.price + self.atr_target_mult * atr_est,
                    strategy_id=self.strategy_id,
                )
                pos.trailing_stop = pos.stop_price
                self._open_position(symbol, pos)
                self._extremes[symbol] = fill.price

        else:  # SELL
            # SELL closes an existing LONG or opens a SHORT
            existing = self._get_position(symbol)
            if existing is not None and existing.side == Direction.LONG:
                self._close_position(symbol)
                self._extremes.pop(symbol, None)
                return

            if sim_pos is not None and sim_pos.side == Direction.SHORT and not self._has_position(symbol):
                pos = dataclasses.replace(sim_pos)
                pos.trailing_stop = pos.stop_price
                self._open_position(symbol, pos)
                self._extremes[symbol] = fill.price
