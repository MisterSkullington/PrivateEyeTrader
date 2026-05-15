"""RLPolicy.predict() latency + obs-shape regression guard.

Discovered 2026-05-13: the previous predict() implementation built a
TradingEnv on every call and ran ``len(bars) - 1`` inner PPO predict()
calls to "step" the env to the latest bar. With ``bar_window=500`` and
8760 backtest bars, that was ~4.4M inner SB3 forwards per backtest —
which hung the process inside SB3's ``policy.predict() → obs_to_tensor
→ self.device`` lookup.

The fix replaces the env-replay loop with a direct ``_build_obs(bars)``
that constructs the observation in O(1) PPO forwards per call. This
file ensures the regression cannot return silently.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("stable_baselines3")

from privateye.models.rl_policy import RLPolicy


def _synthetic_bars(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Random-walk OHLCV bars suitable as RL inference input."""
    rng = np.random.default_rng(seed)
    closes = 30000.0 + np.cumsum(rng.normal(0, 100, n))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
        "open":   closes,
        "high":   closes + 50,
        "low":    closes - 50,
        "close":  closes,
        "volume": rng.uniform(500, 5000, n),
    })


class TestRLPredictPerf:
    """Regression tests for RLPolicy.predict() performance and obs shape.

    All tests require the real ``rl_policy.zip`` artifact. Skip cleanly
    when the artifact is missing so CI on a fresh checkout still passes.
    """

    @pytest.fixture
    def real_policy(self):
        policy = RLPolicy(artifacts_dir="privateye/models/artifacts")
        try:
            policy.load()
        except FileNotFoundError:
            pytest.skip(
                "rl_policy.zip artifact missing; run scripts/train_models.py"
            )
        return policy

    def test_predict_on_500_bars_under_200ms(self, real_policy):
        """Hard upper bound on per-call latency.

        Pre-fix: ~minutes per call (env-replay over 500 bars × ~1ms/inner-call).
        Post-fix: ~25-55 ms per call on a typical CPU (one PPO forward +
        feature extraction). 200ms threshold gives ~4× headroom for slow CI.
        """
        bars = _synthetic_bars(500)
        real_policy.predict(bars)   # warmup (first call may JIT / cache)

        t0 = time.perf_counter()
        for _ in range(10):
            real_policy.predict(bars)
        avg_ms = (time.perf_counter() - t0) * 1000 / 10

        assert avg_ms < 200.0, (
            f"predict() took {avg_ms:.1f}ms/call (>200ms threshold). "
            "Env-replay regression may have returned — check RLPolicy.predict()."
        )

    def test_predict_returns_canonical_direction(self, real_policy):
        """direction ∈ {long, flat}; confidence ∈ [0, 1].

        SHORT must be mapped to FLAT (no spot shorts).
        HOLD must be mapped to FLAT with confidence=0 (Phase 13 hotfix).
        """
        bars = _synthetic_bars(500)
        direction, confidence = real_policy.predict(bars)
        assert direction in ("long", "flat"), f"got direction={direction!r}"
        assert 0.0 <= confidence <= 1.0, f"got confidence={confidence}"

    def test_build_obs_matches_env_obs_dim(self, real_policy):
        """The constructed obs must match the env's OBS_DIM exactly."""
        from privateye.models.rl_env import OBS_DIM

        obs = real_policy._build_obs(_synthetic_bars(500))
        assert obs.shape == (OBS_DIM,), f"got shape={obs.shape}"
        assert obs.dtype == np.float32, f"got dtype={obs.dtype}"

    def test_build_obs_short_bars_pads(self, real_policy):
        """Fewer than OBS_LOOKBACK=60 bars must still produce a valid obs.

        Front-padding with zeros mirrors env._obs() behaviour during the
        first OBS_LOOKBACK bars of a training episode.
        """
        from privateye.models.rl_env import OBS_DIM

        obs = real_policy._build_obs(_synthetic_bars(30))
        assert obs.shape == (OBS_DIM,)
        assert np.all(np.isfinite(obs)), "obs contains NaN/inf"
