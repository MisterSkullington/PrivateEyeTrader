"""
Walk-forward cross-validation and purged k-fold splitter.

Walk-forward CV produces rolling train/test index pairs suitable for time-series
model validation without look-ahead bias. Purged k-fold adds an embargo gap
between training and validation windows to prevent label leakage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass
class WalkForwardFold:
    train_idx: np.ndarray   # integer row indices for training
    test_idx:  np.ndarray   # integer row indices for testing
    fold_num:  int


def walk_forward_splits(
    n: int,
    train_window: int = 6048,   # ~252 trading days × 24h
    test_window:  int = 1512,   # ~63 days
    step:         int = 1512,   # step forward by one test window
    min_train:    int = 100,
) -> list[WalkForwardFold]:
    """
    Generate walk-forward (anchored or sliding) train/test splits.

    Args:
        n:            total number of samples
        train_window: number of bars in each training window
        test_window:  number of bars in each test window
        step:         how far to advance the window each fold
        min_train:    skip folds with fewer training samples than this

    Returns:
        List of WalkForwardFold objects.
    """
    folds: list[WalkForwardFold] = []
    fold_num = 0
    test_start = train_window

    while test_start + test_window <= n:
        train_start = max(0, test_start - train_window)
        train_idx = np.arange(train_start, test_start)
        test_idx  = np.arange(test_start, test_start + test_window)

        if len(train_idx) >= min_train:
            folds.append(WalkForwardFold(
                train_idx=train_idx,
                test_idx=test_idx,
                fold_num=fold_num,
            ))
            fold_num += 1

        test_start += step

    return folds


class PurgedKFold:
    """
    K-fold cross-validator that removes training samples whose label horizon
    overlaps with the validation fold (to prevent label leakage).

    Args:
        n_splits:       number of folds
        label_horizon:  number of forward bars used to construct labels
        embargo_pct:    additional fraction of training samples to remove after
                        the purge gap (handles autocorrelation spillover)
    """

    def __init__(
        self,
        n_splits: int = 5,
        label_horizon: int = 5,
        embargo_pct: float = 0.01,
    ) -> None:
        self.n_splits = n_splits
        self.label_horizon = label_horizon
        self.embargo_pct = embargo_pct

    def split(self, X: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        indices = np.arange(n)
        fold_size = n // self.n_splits
        embargo_n = max(1, int(n * self.embargo_pct))

        for k in range(self.n_splits):
            val_start = k * fold_size
            val_end   = val_start + fold_size if k < self.n_splits - 1 else n

            val_idx = indices[val_start:val_end]

            # Purge: remove training samples whose label window touches the val fold
            purge_end = val_end + self.label_horizon + embargo_n

            train_idx = np.concatenate([
                indices[:max(0, val_start - self.label_horizon)],
                indices[min(n, purge_end):],
            ])

            if len(train_idx) == 0:
                continue

            yield train_idx, val_idx

    def get_n_splits(self) -> int:
        return self.n_splits


def build_lstm_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    lookback: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) sequences for LSTM training.

    Args:
        features: (N, 50) float32 feature matrix
        labels:   (N,) int array of labels (0 or 1) aligned to each bar
        lookback: number of bars in each input sequence

    Returns:
        X: (M, lookback, 50), y: (M,) where M = N - lookback
    """
    n = len(features)
    xs, ys = [], []
    for i in range(lookback, n):
        xs.append(features[i - lookback : i])
        ys.append(labels[i])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64)


def make_direction_labels(
    bars: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.002,
) -> np.ndarray:
    """
    Generate binary direction labels: 1 if close[t+horizon] > close[t] * (1 + threshold).

    Caution: these labels USE FUTURE DATA and must only be used for offline training
    where the future is known. Never call this during live/backtest inference.

    Returns:
        np.ndarray of shape (N,), dtype int8. Last `horizon` rows are 0 (unknown).
    """
    close = bars["close"].values.astype(float)
    n = len(close)
    labels = np.zeros(n, dtype=np.int8)
    for i in range(n - horizon):
        if close[i + horizon] > close[i] * (1 + threshold):
            labels[i] = 1
    return labels
