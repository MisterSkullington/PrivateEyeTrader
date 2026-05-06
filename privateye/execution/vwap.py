"""
VWAPExecutor — volume-weighted order slicer.

Splits a single order into n_slices sub-orders, weighting each slice by
that hour's share of average intraday volume. Falls back to uniform
(TWAP) splitting if no volume profile has been fitted.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from privateye.core.types import Order
from privateye.execution.twap import TWAPExecutor
from privateye.utils.logging import get_logger

log = get_logger()


class VWAPExecutor:
    def __init__(self, n_slices: int = 4) -> None:
        if n_slices < 1:
            raise ValueError(f"n_slices must be >= 1, got {n_slices}")
        self.n_slices = n_slices
        self._volume_profile: dict[int, float] | None = None  # hour-of-day → weight

    def fit_volume_profile(self, bars: pd.DataFrame) -> None:
        """
        Compute average hourly volume share from historical OHLCV bars.

        bars must have a 'timestamp' column (datetime-like) and a 'volume' column.
        Normalises weights so they sum to 1.0.
        """
        df = bars.copy()
        df["_hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        hourly_mean = df.groupby("_hour")["volume"].mean()
        total = hourly_mean.sum()
        if total <= 0:
            log.warning("[VWAPExecutor] Zero total volume — falling back to uniform profile")
            return
        self._volume_profile = (hourly_mean / total).to_dict()
        log.info(f"[VWAPExecutor] Volume profile fitted on {len(df)} bars ({len(self._volume_profile)} hours)")

    def split(self, order: Order, current_hour: int | None = None) -> list[Order]:
        """
        Split `order` into n_slices sub-orders weighted by the volume profile.

        Falls back to uniform (TWAP) splitting when:
          - No profile has been fitted
          - current_hour is not provided
        """
        if order.quantity <= 0:
            return [order]

        if self._volume_profile is None or current_hour is None:
            return TWAPExecutor(self.n_slices).split(order)

        # Pick weights for the next n_slices hours starting from current_hour
        hours = [(current_hour + i) % 24 for i in range(self.n_slices)]
        raw_weights = np.array([self._volume_profile.get(h, 1.0 / 24) for h in hours], dtype=float)
        weights = raw_weights / raw_weights.sum()  # normalise selected slice weights

        slices: list[Order] = []
        qty_remaining = order.quantity
        for i, w in enumerate(weights):
            if i == self.n_slices - 1:
                slice_qty = qty_remaining  # absorb rounding into last slice
            else:
                slice_qty = round(order.quantity * float(w), 8)
                qty_remaining -= slice_qty

            if slice_qty <= 0:
                continue

            slices.append(dataclasses.replace(
                order,
                quantity=slice_qty,
                id=f"{order.id}_v{i}",
                filled_quantity=0.0,
                average_fill_price=0.0,
            ))

        return slices if slices else [order]
