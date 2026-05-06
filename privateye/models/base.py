"""Abstract base class for all ML models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Common interface for all Phase-2 models.

    Subclasses implement fit() for offline training and predict() for inference.
    predict() always returns (direction: str, confidence: float) where direction
    is one of "long", "short", "flat" and confidence is in [0.0, 1.0].
    """

    def __init__(self, artifacts_dir: str | Path = "privateye/models/artifacts") -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, bars: pd.DataFrame, **kwargs) -> None:
        """Train the model on historical OHLCV bars."""

    @abstractmethod
    def predict(self, bars: pd.DataFrame) -> tuple[str, float]:
        """
        Return (direction, confidence) for the current bar (last row of bars).
        direction ∈ {"long", "short", "flat"}
        confidence ∈ [0.0, 1.0]
        """

    @abstractmethod
    def save(self) -> None:
        """Persist model artifact(s) to artifacts_dir."""

    @abstractmethod
    def load(self) -> None:
        """Load model artifact(s) from artifacts_dir."""

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} is not fitted. Call fit() or load() first.")
