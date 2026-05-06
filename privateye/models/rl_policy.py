"""
RL Policy: stable-baselines3 PPO wrapper for the TradingEnv.

Training is offline on historical data. Inference maps the 3010-dim observation
to an action {0=FLAT, 1=LONG, 2=SHORT, 3=HOLD} with a confidence derived from
policy entropy.

Requires: stable-baselines3>=2.3, gymnasium>=0.29
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from privateye.models.base import BaseModel
from privateye.models.rl_env import ACTIONS, N_ACTIONS, TradingEnv
from privateye.utils.logging import get_logger

log = get_logger()

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False


def _require_sb3() -> None:
    if not _SB3_AVAILABLE:
        raise ImportError(
            "stable-baselines3 is required. Install with: pip install stable-baselines3>=2.3"
        )


class RLPolicy(BaseModel):
    """
    PPO policy trained on a TradingEnv episode.

    Confidence is derived from policy entropy:
        confidence = 1 − H(π) / log(n_actions)
    A deterministic (low-entropy) policy → high confidence; random → low.
    """

    def __init__(
        self,
        total_timesteps: int = 500_000,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        lr: float = 3e-4,
        clip_range: float = 0.2,
        artifacts_dir: str | Path = "privateye/models/artifacts",
    ) -> None:
        _require_sb3()
        super().__init__(artifacts_dir)
        self.total_timesteps = total_timesteps
        self.n_steps         = n_steps
        self.batch_size      = batch_size
        self.n_epochs        = n_epochs
        self.lr              = lr
        self.clip_range      = clip_range
        self._model: PPO | None = None

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, bars: pd.DataFrame, **kwargs) -> None:
        _require_sb3()
        log.info(f"[RLPolicy] Training PPO on {len(bars)} bars "
                 f"({self.total_timesteps:,} timesteps).")

        env = TradingEnv(bars)

        self._model = PPO(
            "MlpPolicy",
            env,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            learning_rate=self.lr,
            clip_range=self.clip_range,
            verbose=0,
        )
        self._model.learn(total_timesteps=self.total_timesteps)
        self.is_fitted = True
        log.info("[RLPolicy] Training complete.")

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, bars: pd.DataFrame) -> tuple[str, float]:
        """Returns (direction, confidence) for the current bar."""
        self._require_fitted()
        _require_sb3()

        # Build obs from a temporary env stepped to the last bar
        tmp_env = TradingEnv(bars)
        obs, _ = tmp_env.reset()
        # Step through all bars up to the last one to get the latest observation
        for i in range(len(bars) - 1):
            action, _ = self._model.predict(obs, deterministic=True)
            obs, _, done, _, _ = tmp_env.step(int(action))
            if done:
                break

        action_int, _ = self._model.predict(obs, deterministic=False)
        action_int    = int(action_int)
        direction     = ACTIONS[action_int]

        # Estimate confidence from action probability distribution
        confidence = self._action_confidence(obs)

        # Map SHORT to FLAT (exchange doesn't support shorts yet)
        if direction == "short":
            direction = "flat"

        return direction, float(confidence)

    def act(self, obs: np.ndarray) -> tuple[str, float]:
        """Direct inference on a pre-built observation vector."""
        self._require_fitted()
        _require_sb3()
        action_int, _ = self._model.predict(obs, deterministic=False)
        direction = ACTIONS[int(action_int)]
        if direction == "short":
            direction = "flat"
        return direction, self._action_confidence(obs)

    def _action_confidence(self, obs: np.ndarray) -> float:
        """Confidence = 1 − normalised_entropy of the action distribution."""
        try:
            import torch
            obs_t = torch.from_numpy(obs[np.newaxis].astype(np.float32))
            dist  = self._model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs[0].detach().numpy()
            # Shannon entropy normalised by log(n_actions)
            h = -np.sum(probs * np.log(probs + 1e-9))
            return float(max(0.0, 1.0 - h / math.log(N_ACTIONS)))
        except Exception:
            return 0.5

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        _require_sb3()
        self._require_fitted()
        path = self.artifacts_dir / "rl_policy.zip"
        self._model.save(str(path))
        log.info(f"[RLPolicy] Saved to {path}")

    def load(self) -> None:
        _require_sb3()
        path = self.artifacts_dir / "rl_policy.zip"
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        self._model = PPO.load(str(path))
        self.is_fitted = True
        log.info(f"[RLPolicy] Loaded from {path}")
