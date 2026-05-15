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
import torch

from privateye.data.feature_extractor import extract_features
from privateye.models.base import BaseModel
from privateye.models.rl_env import (
    ACTIONS,
    N_ACTIONS,
    N_FEATURES,
    N_PORTFOLIO,
    OBS_LOOKBACK,
    TradingEnv,
)
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
        """Returns (direction, confidence) for the most recent bar.

        Builds the observation directly from the last OBS_LOOKBACK bars without
        stepping a TradingEnv — ~500× faster than the previous env-replay loop
        and matches the env reset-state distribution that PPO was trained on.

        The previous implementation re-built a TradingEnv on every call and ran
        len(bars)-1 inner PPO predict() calls to "step" to the latest bar. That
        was O(window_size) PPO forwards per outer call (~500× too many) AND it
        fed the policy a fictional portfolio state from the policy's own
        deterministic self-replay — not the real backtest portfolio.
        """
        self._require_fitted()
        _require_sb3()

        obs = self._build_obs(bars)
        action_int, _ = self._model.predict(obs, deterministic=False)
        direction = ACTIONS[int(action_int)]
        confidence = self._action_confidence(obs)

        # Map SHORT to FLAT (exchange doesn't support shorts yet)
        if direction == "short":
            direction = "flat"
        # Map HOLD to FLAT with zero confidence — RL is abstaining this bar,
        # not voting to flatten existing positions. Without this, "hold" leaks
        # downstream into Direction("hold") which raises ValueError and crashes
        # the FusionStrategy MARKET_DATA handler.
        elif direction == "hold":
            direction = "flat"
            confidence = 0.0

        return direction, float(confidence)

    def _build_obs(self, bars: pd.DataFrame) -> np.ndarray:
        """Construct an (OBS_DIM,) observation matching TradingEnv._obs() at reset.

        Market half (4080 floats): last OBS_LOOKBACK rows of extract_features(bars),
                                   flattened. Front-padded with zeros if fewer
                                   than OBS_LOOKBACK bars are available.
        Portfolio half (10 floats): zeros (flat reset state) with atr_norm at slot
                                    7 set from the latest bar's real value
                                    (feature index 18) — mirrors env._obs() field
                                    for the only portfolio slot that depends on
                                    market data rather than agent state.
        """
        features = extract_features(bars)  # (N, N_FEATURES)
        window = features[-OBS_LOOKBACK:]
        if len(window) < OBS_LOOKBACK:
            pad = np.zeros((OBS_LOOKBACK - len(window), N_FEATURES), dtype=np.float32)
            window = np.concatenate([pad, window], axis=0)
        flat = window.flatten().astype(np.float32)

        portfolio = np.zeros(N_PORTFOLIO, dtype=np.float32)
        if len(features) > 0:
            portfolio[7] = float(features[-1, 18])  # atr_norm

        obs = np.concatenate([flat, portfolio]).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    def act(self, obs: np.ndarray) -> tuple[str, float]:
        """Direct inference on a pre-built observation vector."""
        self._require_fitted()
        _require_sb3()
        action_int, _ = self._model.predict(obs, deterministic=False)
        direction = ACTIONS[int(action_int)]
        confidence = self._action_confidence(obs)
        if direction == "short":
            direction = "flat"
        elif direction == "hold":
            direction = "flat"
            confidence = 0.0
        return direction, confidence

    def _action_confidence(self, obs: np.ndarray) -> float:
        """Confidence = 1 − normalised_entropy of the action distribution.

        Wrapped in torch.no_grad() — inference only; avoids building the
        backprop graph on every call.
        """
        try:
            with torch.no_grad():
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
        # Disable dropout / batch-norm during inference. SB3 also handles this
        # per-call internally, but setting it once at load is more efficient
        # and matches conventional PyTorch inference practice.
        self._model.policy.eval()
        self.is_fitted = True
        log.info(f"[RLPolicy] Loaded from {path}")
