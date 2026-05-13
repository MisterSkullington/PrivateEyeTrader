"""
Feature drift detector using the KS test + Population Stability Index (PSI).

Used by ``OnlineLearner._run_retrain()`` as a safety gate: if the incoming
bar buffer's feature distribution has drifted significantly from the reference
distribution the model was trained on, the retrain is skipped.

Rationale: models trained on calm-market distributions often regress badly
when promoted during high-volatility regime shifts.  Two independent drift
signals are used:

  KS test   — ``drift_fraction >= drift_threshold``
  PSI       — ``mean_psi >= psi_threshold`` (PSI > 0.20 = significant shift)

Final gate uses OR logic so either signal alone can block a dangerous retrain.

Config keys (under phase1.drift_detection):
  enabled               (bool,  default False)
  p_threshold           (float, default 0.01)   — KS p-value below which a feature is "drifted"
  drift_threshold       (float, default 0.20)   — fraction of KS-drifted features to flag
  psi_threshold         (float, default 0.20)   — mean PSI above which data is "drifted"
  min_reference_rows    (int,   default 100)    — fit() is a no-op below this count
  reference_window_bars (int,   default 500)    — bars fed to fit() on initialization
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from privateye.utils.logging import get_logger

log = get_logger()


# ── PSI helper ─────────────────────────────────────────────────────────────────

def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Compute the Population Stability Index between two 1-D arrays.

    Uses equal-quantile binning derived from the *reference* distribution so
    that each bucket holds roughly the same proportion of reference mass.

    PSI = Σ (expected_i − actual_i) × ln(expected_i / actual_i)

    Rules of thumb:
      PSI < 0.10  — no significant change
      PSI 0.10–0.20 — moderate change; investigate
      PSI > 0.20  — significant distribution shift

    Parameters
    ----------
    reference:
        1-D array of reference (training) values. NaN/Inf are filtered.
    current:
        1-D array of current (incoming) values. NaN/Inf are filtered.
    n_bins:
        Number of equal-quantile bins (default 10).

    Returns
    -------
    float
        PSI value.  Returns 0.0 if either array has fewer than 5 finite values.
    """
    ref = reference[np.isfinite(reference)]
    cur = current[np.isfinite(current)]

    if len(ref) < 5 or len(cur) < 5:
        return 0.0

    # Build bin edges from reference quantiles (n_bins + 1 edges)
    quantile_pts = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(ref, quantile_pts)
    # Make edges unique to avoid empty buckets from flat distributions
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    epsilon = 1e-4

    # Count proportions per bucket
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = np.clip(ref_counts / ref_counts.sum(), epsilon, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), epsilon, None)

    psi_values = (ref_pct - cur_pct) * np.log(ref_pct / cur_pct)
    return float(np.sum(psi_values))


@dataclass
class DriftReport:
    """Result of a KS-test + PSI drift check against the reference distribution."""
    n_features_total:      int
    n_features_drifted:    int
    drift_fraction:        float          # n_features_drifted / n_features_total
    is_drifted:            bool           # (drift_fraction >= ks_threshold) OR psi_drifted
    drifted_feature_indices: list[int]    # indices of features that failed the KS test
    min_p_value:           float          # smallest p-value seen (most drifted feature)
    # Phase 4 — PSI fields (default 0/False for backward compat)
    psi_score:    float = 0.0             # mean PSI across features
    psi_drifted:  bool  = False           # True when psi_score > psi_threshold
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class FeatureDriftDetector:
    """Two-sample Kolmogorov-Smirnov test on per-feature column distributions.

    Workflow::

        detector = FeatureDriftDetector(p_threshold=0.01, drift_threshold=0.20)

        # 1. Fit on the training distribution
        detector.fit(reference_features)   # shape (N, F)

        # 2. Check incoming features against reference
        report = detector.check(new_features)  # shape (M, F)
        if report.is_drifted:
            log.warning("Distribution shifted — skipping retrain")

    When fewer than ``min_reference_rows`` rows are supplied to ``fit()``,
    the call is silently skipped and ``is_fitted`` remains False.  When
    ``check()`` is called before ``fit()``, it returns a neutral
    ``DriftReport`` with ``is_drifted=False`` rather than raising.
    """

    def __init__(
        self,
        p_threshold:        float = 0.01,
        drift_threshold:    float = 0.20,
        min_reference_rows: int   = 100,
        psi_threshold:      float = 0.20,   # Phase 4 — PSI threshold
    ) -> None:
        self._p_threshold    = float(p_threshold)
        self._drift_threshold = float(drift_threshold)
        self._min_rows       = int(min_reference_rows)
        self._psi_threshold  = float(psi_threshold)   # Phase 4
        self._reference: np.ndarray | None = None   # shape (N, F)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, features: np.ndarray) -> None:
        """Store the reference distribution.

        Args:
            features: 2-D array of shape (N, F). Silently ignored when N < min_reference_rows.
        """
        if features.ndim != 2 or features.shape[0] < self._min_rows:
            log.debug(
                f"[FeatureDriftDetector] fit() skipped — "
                f"need ≥ {self._min_rows} rows, got {features.shape[0] if features.ndim == 2 else '?'}"
            )
            return
        self._reference = features.astype(np.float64)
        log.debug(
            f"[FeatureDriftDetector] Reference fitted: "
            f"{self._reference.shape[0]} rows × {self._reference.shape[1]} features"
        )

    def check(self, features: np.ndarray) -> DriftReport:
        """Run per-column KS test against the reference distribution.

        Args:
            features: 2-D array of shape (M, F).

        Returns:
            DriftReport with ``is_drifted=False`` when not yet fitted.
        """
        n_total = features.shape[1] if features.ndim == 2 else 0

        if self._reference is None or n_total == 0:
            return DriftReport(
                n_features_total=n_total,
                n_features_drifted=0,
                drift_fraction=0.0,
                is_drifted=False,
                drifted_feature_indices=[],
                min_p_value=1.0,
            )

        try:
            from scipy.stats import ks_2samp  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "scipy is required for FeatureDriftDetector. "
                "Install with: pip install scipy  or  pip install privateye[ml]"
            ) from exc

        n_cols = min(n_total, self._reference.shape[1])
        new_arr = features.astype(np.float64)
        drifted_indices: list[int] = []
        min_p = 1.0
        psi_values: list[float] = []

        for col_idx in range(n_cols):
            ref_col = self._reference[:, col_idx]
            new_col = new_arr[:, col_idx]
            # Filter out NaN / Inf before the test
            ref_col = ref_col[np.isfinite(ref_col)]
            new_col = new_col[np.isfinite(new_col)]
            if len(ref_col) < 5 or len(new_col) < 5:
                continue  # not enough data to test

            _, p_value = ks_2samp(ref_col, new_col)
            if p_value < min_p:
                min_p = p_value
            if p_value < self._p_threshold:
                drifted_indices.append(col_idx)

            # Phase 4 — PSI per feature
            col_psi = _compute_psi(ref_col, new_col)
            psi_values.append(col_psi)

        n_drifted  = len(drifted_indices)
        drift_frac = n_drifted / n_cols if n_cols > 0 else 0.0

        # Phase 4 — mean PSI across features
        mean_psi   = float(np.mean(psi_values)) if psi_values else 0.0
        psi_drifted = mean_psi >= self._psi_threshold

        # OR logic: either KS or PSI can flag drift
        is_drifted = (drift_frac >= self._drift_threshold) or psi_drifted

        if is_drifted:
            log.warning(
                f"[FeatureDriftDetector] Drift detected: "
                f"{n_drifted}/{n_cols} KS-features (threshold={self._drift_threshold:.0%}), "
                f"min_p={min_p:.4f}, PSI={mean_psi:.3f} (psi_drifted={psi_drifted})"
            )

        return DriftReport(
            n_features_total=n_cols,
            n_features_drifted=n_drifted,
            drift_fraction=drift_frac,
            is_drifted=is_drifted,
            drifted_feature_indices=drifted_indices,
            min_p_value=min_p,
            psi_score=mean_psi,
            psi_drifted=psi_drifted,
        )

    @property
    def is_fitted(self) -> bool:
        """True after a successful ``fit()`` call."""
        return self._reference is not None
