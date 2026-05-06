"""Tests for the RL trading environment (requires gymnasium)."""
import numpy as np
import pandas as pd
import pytest

gym = pytest.importorskip("gymnasium", reason="gymnasium not installed")

from privateye.models.rl_env import (
    N_ACTIONS, OBS_DIM, OBS_LOOKBACK, TradingEnv,
)


def _make_bars(n: int = 200, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 30_000 + np.cumsum(rng.normal(0, 100, n))
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open":   closes - 20,
        "high":   closes + 100,
        "low":    closes - 100,
        "close":  closes,
        "volume": rng.uniform(200, 1000, n),
    })


class TestTradingEnv:
    def setup_method(self):
        self.bars = _make_bars(200)
        self.env  = TradingEnv(self.bars)

    def test_spaces(self):
        assert self.env.observation_space.shape == (OBS_DIM,)
        assert self.env.action_space.n == N_ACTIONS

    def test_reset(self):
        obs, info = self.env.reset()
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_obs_range(self):
        obs, _ = self.env.reset()
        assert (obs >= -10.0 - 1e-6).all()
        assert (obs <=  10.0 + 1e-6).all()

    def test_step_flat(self):
        self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(0)  # FLAT
        assert obs.shape == (OBS_DIM,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "equity" in info

    def test_step_long(self):
        self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(1)  # LONG
        assert obs.shape == (OBS_DIM,)

    def test_episode_terminates(self):
        obs, _ = self.env.reset()
        done = False
        steps = 0
        while not done and steps < 1000:
            obs, reward, terminated, truncated, info = self.env.step(3)  # HOLD
            done = terminated or truncated
            steps += 1
        assert done, "Episode did not terminate"

    def test_equity_non_negative(self):
        """Equity should not go deeply negative; liquidation terminates at <10% of initial."""
        obs, _ = self.env.reset()
        done = False
        min_equity = float("inf")
        while not done:
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            min_equity = min(min_equity, info["equity"])
        # Equity should not drop below -50% of initial capital (liquidation should kick in earlier)
        assert min_equity > -self.env.initial_capital * 0.5, (
            f"Equity dropped too low: {min_equity:.2f}"
        )

    def test_reset_restores_state(self):
        obs1, _ = self.env.reset()
        for _ in range(10):
            self.env.step(1)
        obs2, _ = self.env.reset()
        assert np.allclose(obs1, obs2), "Reset does not restore initial state"

    def test_deterministic_with_seed(self):
        bars = _make_bars(200)
        env1 = TradingEnv(bars)
        env2 = TradingEnv(bars)
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        assert np.allclose(obs1, obs2)

    def test_short_action_affects_position(self):
        self.env.reset()
        # Open a long, then open a short (should close long and open short)
        self.env.step(1)  # LONG
        assert self.env._position > 0.0, "Expected long position"
        self.env.step(2)  # SHORT
        assert self.env._position <= 0.0, "Expected position closed or short"
