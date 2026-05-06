"""
RL Trading Environment: gymnasium.Env wrapping historical OHLCV data.

Observation: (4090,) = 68-feature × 60-bar window flattened + 10 portfolio features
Actions: 0=FLAT, 1=LONG, 2=SHORT (mapped to FLAT until shorts are supported), 3=HOLD
Reward: log-return − churn_penalty − drawdown_penalty

SHORT positions are simulated internally for training purposes. When used in
FusionStrategy, action=SHORT maps to FLAT (exchange doesn't support shorts yet).

Requires: gymnasium>=0.29
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from privateye.data.feature_extractor import extract_features
from privateye.utils.logging import get_logger

log = get_logger()

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False

ACTIONS = {0: "flat", 1: "long", 2: "short", 3: "hold"}
N_ACTIONS = 4
OBS_LOOKBACK = 60
N_FEATURES   = 68
N_PORTFOLIO  = 10
OBS_DIM      = OBS_LOOKBACK * N_FEATURES + N_PORTFOLIO  # 68*60+10 = 4090


def _require_gym() -> None:
    if not _GYM_AVAILABLE:
        raise ImportError("gymnasium is required. Install with: pip install gymnasium>=0.29")


class TradingEnv(gym.Env if _GYM_AVAILABLE else object):
    """
    Single-symbol trading environment for RL training.

    Episode = entire bars DataFrame (or a walk-forward window thereof).
    Position is fully closed at episode end.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        bars: pd.DataFrame,
        initial_capital: float = 10_000.0,
        fee_taker: float = 0.001,
        slippage_pct: float = 0.0005,
        max_position_pct: float = 0.95,
        max_bars_held: int = 48,
        churn_penalty: float = 1e-4,
        drawdown_penalty: float = 1e-3,
        drawdown_threshold: float = 0.02,
    ) -> None:
        _require_gym()
        super().__init__()
        self.bars             = bars.reset_index(drop=True)
        self.initial_capital  = initial_capital
        self.fee_taker        = fee_taker
        self.slippage_pct     = slippage_pct
        self.max_position_pct = max_position_pct
        self.max_bars_held    = max_bars_held
        self.churn_penalty    = churn_penalty
        self.drawdown_penalty = drawdown_penalty
        self.drawdown_threshold = drawdown_threshold

        # Pre-compute feature matrix for the whole episode
        self._features: np.ndarray = extract_features(bars)  # (N, 50)

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # State variables (reset each episode)
        self._step: int = 0
        self._cash: float = initial_capital
        self._equity: float = initial_capital
        self._peak_equity: float = initial_capital
        self._position: float = 0.0   # qty (>0 long, <0 short)
        self._entry_price: float = 0.0
        self._bars_held: int = 0
        self._prev_action: int = 0
        self._daily_start_equity: float = initial_capital

    # ── gym interface ─────────────────────────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step          = OBS_LOOKBACK   # start after warmup
        self._cash          = self.initial_capital
        self._equity        = self.initial_capital
        self._peak_equity   = self.initial_capital
        self._position      = 0.0
        self._entry_price   = 0.0
        self._bars_held     = 0
        self._prev_action   = 0
        self._daily_start_equity = self.initial_capital
        return self._obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._step >= len(self.bars):
            return self._obs(), 0.0, True, False, {}

        price = float(self.bars["close"].iloc[self._step])
        prev_equity = self._equity

        self._execute_action(action, price)
        self._equity = self._cash + self._position_value(price)
        self._peak_equity = max(self._peak_equity, self._equity)

        log_ret = np.log(self._equity / prev_equity) if prev_equity > 0 else 0.0
        dd = (self._peak_equity - self._equity) / self._peak_equity if self._peak_equity > 0 else 0.0
        churn = self.churn_penalty * abs(action - self._prev_action)
        dd_pen = self.drawdown_penalty * max(0.0, dd - self.drawdown_threshold)
        reward = float(log_ret - churn - dd_pen)

        self._prev_action = action
        self._step += 1
        terminated = self._step >= len(self.bars)

        # Liquidation: terminate on catastrophic loss (>90% equity loss)
        if self._equity < self.initial_capital * 0.10:
            if self._position != 0.0:
                self._flat_out(price)
            terminated = True
            reward -= 1.0  # large liquidation penalty

        if terminated and self._position != 0.0:
            self._flat_out(price)

        return self._obs(), reward, terminated, False, {
            "equity": self._equity, "position": self._position,
            "drawdown": dd, "action": ACTIONS[action],
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _execute_action(self, action: int, price: float) -> None:
        if action == 3:  # HOLD
            self._bars_held += 1
            if self._position != 0.0 and self._bars_held >= self.max_bars_held:
                self._flat_out(price)  # timeout
            return

        if action == 0:  # FLAT
            if self._position != 0.0:
                self._flat_out(price)
            self._bars_held = 0
            return

        # LONG or SHORT: flip if needed
        target_side = 1.0 if action == 1 else -1.0
        current_side = np.sign(self._position)
        if current_side == target_side:
            self._bars_held += 1
            return

        if self._position != 0.0:
            self._flat_out(price)

        # Size position from current equity to ensure non-negative equity
        safe_notional = max(0.0, self._equity) * self.max_position_pct
        if safe_notional < 1.0:
            return  # insufficient equity

        fill_price = price * (1 + self.slippage_pct * target_side)
        if fill_price <= 0:
            return
        qty = (safe_notional / fill_price) * target_side
        fee = safe_notional * self.fee_taker

        if target_side > 0:  # LONG: spend cash
            self._cash -= safe_notional + fee
        else:  # SHORT: receive proceeds
            self._cash += safe_notional - fee

        self._position    = qty
        self._entry_price = fill_price
        self._bars_held   = 1

    def _flat_out(self, price: float) -> None:
        if self._position == 0.0:
            return
        fill_price = price * (1 - self.slippage_pct * np.sign(self._position))
        notional = abs(self._position * fill_price)
        fee = notional * self.fee_taker

        if self._position > 0:  # closing LONG: receive proceeds
            self._cash += notional - fee
        else:  # closing SHORT: pay to buy back
            self._cash -= notional + fee

        self._position    = 0.0
        self._entry_price = 0.0
        self._bars_held   = 0

    def _position_value(self, price: float) -> float:
        if self._position == 0.0:
            return 0.0
        return self._position * price

    def _obs(self) -> np.ndarray:
        idx = max(0, self._step - OBS_LOOKBACK)
        window = self._features[idx : self._step]
        if len(window) < OBS_LOOKBACK:
            pad = np.zeros((OBS_LOOKBACK - len(window), N_FEATURES), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)
        flat = window.flatten()  # (3000,)

        price = float(self.bars["close"].iloc[min(self._step, len(self.bars) - 1)])
        ep = self._entry_price if self._entry_price else price
        dd = (self._peak_equity - self._equity) / self._peak_equity if self._peak_equity > 0 else 0.0
        portfolio = np.array([
            self._cash / (self._equity + 1e-9),              # cash_pct
            float(np.sign(self._position)),                   # position_side
            abs(self._position * ep) / (self._equity + 1e-9), # notional_pct
            (price - ep) / ep * np.sign(self._position) if ep > 0 else 0.0,  # unrealised_pnl_pct
            self._bars_held / self.max_bars_held,             # bars_held_norm
            dd,                                               # drawdown
            price / ep - 1.0 if ep > 0 else 0.0,             # entry_deviation
            float(self._features[min(self._step, len(self._features)-1), 18]),  # atr_norm
            self._equity / self.initial_capital - 1.0,        # equity_growth
            float(np.sign(self._position)) * dd,              # signed_dd
        ], dtype=np.float32)

        obs = np.concatenate([flat, portfolio]).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)
