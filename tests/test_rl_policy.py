"""
RLPolicy HOLD-action mapping tests.

Discovered in production (paper-mode log, 2026-05-06): when PPO emits action 3
(HOLD), the string "hold" leaks through ModelEnsemble → FusionStrategy →
``Direction("hold")``, raising ``ValueError`` and crashing the bus handler.

Both ``predict()`` and ``act()`` must map ``"hold"`` → ``("flat", 0.0)`` so
HOLD becomes a no-op vote, not a rogue 4th category in ``_weighted_vote``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("stable_baselines3")

from privateye.models.rl_policy import RLPolicy


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def stub_policy():
    """RLPolicy with a mocked PPO model — avoids loading the real artifact."""
    policy = RLPolicy()
    policy._model = MagicMock()
    policy.is_fitted = True
    # Deterministic confidence so we can assert on it
    policy._action_confidence = lambda obs: 0.85
    return policy


# ── act(): direct obs path used by callers that already have an obs vector ───

class TestRLActMapping:

    @pytest.mark.parametrize("action_int, expected_direction, expected_conf", [
        (0, "flat", 0.85),   # FLAT passes through, confidence preserved
        (1, "long", 0.85),   # LONG passes through
        (2, "flat", 0.85),   # SHORT → FLAT, confidence preserved (spot has no shorts)
        (3, "flat", 0.0),    # HOLD → FLAT, confidence ZEROED (no-op vote)
    ])
    def test_action_to_direction_mapping(
        self, stub_policy, action_int, expected_direction, expected_conf,
    ):
        stub_policy._model.predict.return_value = (np.array(action_int), None)
        obs = np.zeros((4090,), dtype=np.float32)

        direction, confidence = stub_policy.act(obs)
        assert direction == expected_direction
        assert confidence == expected_conf

    def test_act_never_returns_hold_string(self, stub_policy):
        """Defensive: even with action=3, the literal 'hold' string must not escape."""
        stub_policy._model.predict.return_value = (np.array(3), None)
        direction, _ = stub_policy.act(np.zeros((4090,), dtype=np.float32))
        assert direction != "hold"


# ── predict(): bars path used by ModelEnsemble in live/paper inference ───────

class TestRLPredictMapping:
    """``predict()`` builds a TradingEnv internally and steps to the latest bar.

    To unit-test the HOLD mapping without a real PPO + a real env, we monkeypatch
    ``TradingEnv`` to a stub that immediately reports ``done`` so the env-step
    loop body never runs."""

    def test_predict_maps_hold_to_flat_with_zero_confidence(
        self, stub_policy, monkeypatch,
    ):
        # Stub env: reset returns a zero obs; step returns done=True immediately
        stub_env = MagicMock()
        stub_env.reset.return_value = (np.zeros(4090, dtype=np.float32), {})
        stub_env.step.return_value = (
            np.zeros(4090, dtype=np.float32), 0.0, True, False, {},
        )
        monkeypatch.setattr(
            "privateye.models.rl_policy.TradingEnv",
            lambda *_a, **_kw: stub_env,
        )
        # Force PPO action 3 (HOLD)
        stub_policy._model.predict.return_value = (np.array(3), None)

        bars = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC"),
            "open":  [100.0, 101.0],
            "high":  [101.0, 102.0],
            "low":   [ 99.0, 100.0],
            "close": [100.0, 101.0],
            "volume":[1000.0, 1000.0],
        })

        direction, confidence = stub_policy.predict(bars)
        assert direction == "flat"
        assert confidence == 0.0

    def test_predict_maps_long_through(
        self, stub_policy, monkeypatch,
    ):
        stub_env = MagicMock()
        stub_env.reset.return_value = (np.zeros(4090, dtype=np.float32), {})
        stub_env.step.return_value = (
            np.zeros(4090, dtype=np.float32), 0.0, True, False, {},
        )
        monkeypatch.setattr(
            "privateye.models.rl_policy.TradingEnv",
            lambda *_a, **_kw: stub_env,
        )
        stub_policy._model.predict.return_value = (np.array(1), None)

        bars = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC"),
            "open":  [100.0, 101.0], "high": [101.0, 102.0],
            "low":   [ 99.0, 100.0], "close": [100.0, 101.0],
            "volume":[1000.0, 1000.0],
        })
        direction, confidence = stub_policy.predict(bars)
        assert direction == "long"
        assert confidence == pytest.approx(0.85)
