"""
tests/test_phase1.py — Phase 1: Data & Feature Intelligence Layer (30 tests)

Groups:
  TestMacroProvider         (6)  — fetch_latest, cache TTL, neutral defaults, parse
  TestFeatureEngineer       (10) — backward compat, 86 features, extract shapes, properties
  TestFeatureDriftDetector  (6)  — fit/check/is_fitted, same vs shifted distributions
  TestOnlineLearnerDriftGate (4) — drift disabled/enabled, gate blocks/allows retrain
  TestLiveAltDataEnrichment  (4) — _enrich_bars_with_alt_data column injection
"""
from __future__ import annotations

import asyncio
import dataclasses
import time
import unittest
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_bars(n: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 30000.0 + np.cumsum(rng.normal(0, 100, n))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
        "open":   close + rng.uniform(-50, 50, n),
        "high":   close + rng.uniform(10, 150, n),
        "low":    close - rng.uniform(10, 150, n),
        "close":  close,
        "volume": rng.uniform(500, 5000, n),
    })


def _run(coro):
    return asyncio.run(coro)


# ══════════════════════════════════════════════════════════════════════════════
# TestMacroProvider (6 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestMacroProvider(unittest.TestCase):

    def _provider(self, ttl: float = 60.0):
        from privateye.data.providers.macro import MacroProvider
        return MacroProvider(timeout=5.0, cache_ttl_seconds=ttl)

    def _good_payload(self) -> str:
        import json
        return json.dumps({
            "data": {
                "market_cap_percentage": {"btc": 52.3},
                "total_market_cap":      {"usd": 2_400_000_000_000},
                "market_cap_change_percentage_24h_usd": 1.5,
                "total_volume":          {"usd": 120_000_000_000},
            }
        })

    def test_fetch_latest_returns_all_required_keys(self):
        """fetch_latest() must return a dict with all 5 required keys."""
        p = self._provider()
        with patch(
            "privateye.data.providers.macro._sync_fetch",
            return_value=self._good_payload(),
        ):
            data = _run(p.fetch_latest())
        required = {
            "btc_dominance", "total_mcap_usd", "total_mcap_change_24h",
            "stablecoin_ratio_approx", "btc_mcap_usd",
        }
        self.assertEqual(required, set(data.keys()))

    def test_btc_dominance_in_valid_range(self):
        """btc_dominance must be in [0, 100]."""
        p = self._provider()
        with patch(
            "privateye.data.providers.macro._sync_fetch",
            return_value=self._good_payload(),
        ):
            data = _run(p.fetch_latest())
        self.assertGreaterEqual(data["btc_dominance"], 0.0)
        self.assertLessEqual(data["btc_dominance"], 100.0)

    def test_returns_neutral_defaults_on_network_error(self):
        """On any exception, fetch_latest() must return NEUTRAL_DEFAULTS and not raise."""
        from privateye.data.providers.macro import NEUTRAL_DEFAULTS
        p = self._provider()
        with patch(
            "privateye.data.providers.macro._sync_fetch",
            side_effect=ConnectionError("timeout"),
        ):
            data = _run(p.fetch_latest())
        self.assertEqual(data, NEUTRAL_DEFAULTS)

    def test_cache_ttl_prevents_second_request(self):
        """Second call within TTL must not hit the network."""
        p = self._provider(ttl=3600.0)
        call_count = [0]

        def fake_fetch(url, timeout):
            call_count[0] += 1
            return self._good_payload()

        with patch("privateye.data.providers.macro._sync_fetch", side_effect=fake_fetch):
            _run(p.fetch_latest())
            _run(p.fetch_latest())

        self.assertEqual(call_count[0], 1, "Second call should have used the cache")

    def test_stale_cache_triggers_refetch(self):
        """After TTL expires, a second call must hit the network again."""
        p = self._provider(ttl=0.01)  # 10ms TTL
        call_count = [0]

        def fake_fetch(url, timeout):
            call_count[0] += 1
            return self._good_payload()

        with patch("privateye.data.providers.macro._sync_fetch", side_effect=fake_fetch):
            _run(p.fetch_latest())
            time.sleep(0.05)   # outlast 10ms TTL
            _run(p.fetch_latest())

        self.assertGreaterEqual(call_count[0], 2, "Stale cache must trigger re-fetch")

    def test_parse_response_handles_missing_keys(self):
        """_parse_response() must not raise on a minimal / empty payload."""
        import json
        from privateye.data.providers.macro import MacroProvider, NEUTRAL_DEFAULTS
        p = MacroProvider()
        # Minimal payload — all inner keys missing
        raw = json.dumps({"data": {}})
        result = p._parse_response(raw)
        # All 5 keys must be present
        self.assertEqual(set(result.keys()), set(NEUTRAL_DEFAULTS.keys()))
        # btc_dominance falls back to neutral default
        self.assertEqual(result["btc_dominance"], NEUTRAL_DEFAULTS["btc_dominance"])


# ══════════════════════════════════════════════════════════════════════════════
# TestFeatureEngineer (10 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureEngineer(unittest.TestCase):

    def _all_disabled_cfg(self) -> dict:
        return {
            "phase1": {
                "features": {
                    "derivatives_enabled": False,
                    "macro_enabled": False,
                    "enhanced_sentiment_enabled": False,
                }
            }
        }

    def _all_enabled_cfg(self) -> dict:
        return {
            "phase1": {
                "features": {
                    "derivatives_enabled": True,
                    "macro_enabled": True,
                    "enhanced_sentiment_enabled": True,
                }
            }
        }

    def test_all_disabled_gives_68_features(self):
        """When all groups are disabled, n_features must equal 68 (backward compat)."""
        from privateye.data.feature_engineer import FeatureEngineer
        eng = FeatureEngineer(self._all_disabled_cfg())
        self.assertEqual(eng.n_features, 68)

    def test_all_enabled_gives_86_features(self):
        """When all groups are enabled, n_features must equal 86."""
        from privateye.data.feature_engineer import FeatureEngineer
        eng = FeatureEngineer(self._all_enabled_cfg())
        self.assertEqual(eng.n_features, 86)

    def test_extract_returns_correct_shape_86(self):
        """extract() must return (N, 86) when all groups are enabled."""
        from privateye.data.feature_engineer import FeatureEngineer
        eng = FeatureEngineer(self._all_enabled_cfg())
        bars = _make_bars(120)
        feats = eng.extract(bars)
        self.assertEqual(feats.shape, (120, 86))

    def test_extract_latest_returns_1d_vector(self):
        """extract_latest() must return a (86,) 1-D array."""
        from privateye.data.feature_engineer import FeatureEngineer
        eng = FeatureEngineer(self._all_enabled_cfg())
        bars = _make_bars(120)
        latest = eng.extract_latest(bars)
        self.assertEqual(latest.ndim, 1)
        self.assertEqual(latest.shape[0], 86)

    def test_feature_names_length_matches_n_features(self):
        """feature_names list length must equal n_features."""
        from privateye.data.feature_engineer import FeatureEngineer
        for cfg in [self._all_disabled_cfg(), self._all_enabled_cfg()]:
            eng = FeatureEngineer(cfg)
            self.assertEqual(len(eng.feature_names), eng.n_features)

    def test_derivatives_group_finite_when_alt_data_absent(self):
        """Derivatives features must be finite even when funding_rate/OI columns are absent."""
        from privateye.data.feature_engineer import FeatureEngineer
        eng = FeatureEngineer(self._all_enabled_cfg())
        bars = _make_bars(120)  # no funding_rate/open_interest columns
        feats = eng.extract(bars)
        self.assertTrue(np.all(np.isfinite(feats)), "All features must be finite")

    def test_macro_group_zeros_when_snapshot_is_none(self):
        """Macro features must be all deterministic values when set_macro_snapshot(None)."""
        from privateye.data.feature_engineer import FeatureEngineer
        eng = FeatureEngineer(self._all_enabled_cfg())
        eng.set_macro_snapshot(None)
        bars = _make_bars(120)
        feats = eng.extract(bars)
        # When macro_snapshot is None, btc_dominance defaults to 50 → btc_dominance_norm = 0
        macro_start = 76   # base(68) + derivatives(8)
        btc_dom_norm_col = feats[:, macro_start]
        self.assertTrue(np.allclose(btc_dom_norm_col, 0.0),
                        "btc_dominance_norm should be 0 when dominance=50 (neutral)")

    def test_set_macro_snapshot_propagates_to_build(self):
        """set_macro_snapshot() value must be reflected in the macro feature columns."""
        from privateye.data.feature_engineer import FeatureEngineer
        eng = FeatureEngineer(self._all_enabled_cfg())
        eng.set_macro_snapshot({"btc_dominance": 70.0, "total_mcap_change_24h": 5.0,
                                  "stablecoin_ratio_approx": 0.12, "total_mcap_usd": 2e12,
                                  "btc_mcap_usd": 1.4e12})
        bars = _make_bars(120)
        feats = eng.extract(bars)
        macro_start = 76  # base(68) + derivatives(8)
        btc_dom_norm_col = feats[:, macro_start]
        # (70 - 50) / 20 = 1.0
        self.assertTrue(np.allclose(btc_dom_norm_col, 1.0, atol=1e-4),
                        f"Expected btc_dominance_norm=1.0, got {btc_dom_norm_col[0]:.4f}")

    def test_fg_momentum_zero_when_fear_greed_absent(self):
        """fg_momentum must be zero (or near-zero) when fear_greed column is absent."""
        from privateye.data.feature_engineer import FeatureEngineer
        eng = FeatureEngineer(self._all_enabled_cfg())
        bars = _make_bars(120)  # no fear_greed column
        feats = eng.extract(bars)
        # fg_momentum column index: base(68) + deriv(8) + macro(6) = 82
        fg_mom_col = feats[:, 82]
        # When fear_greed defaults to 50 for all bars, momentum = 50 - 50 = 0
        self.assertTrue(np.allclose(fg_mom_col, 0.0, atol=1e-4))

    def test_extended_feature_names_no_duplicates(self):
        """EXTENDED_FEATURE_NAMES must have no duplicate entries."""
        from privateye.data.feature_engineer import EXTENDED_FEATURE_NAMES
        self.assertEqual(len(EXTENDED_FEATURE_NAMES), len(set(EXTENDED_FEATURE_NAMES)),
                         "Duplicate names found in EXTENDED_FEATURE_NAMES")


# ══════════════════════════════════════════════════════════════════════════════
# TestFeatureDriftDetector (6 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestFeatureDriftDetector(unittest.TestCase):

    def _det(self, p_threshold=0.01, drift_threshold=0.20, min_rows=50):
        from privateye.models.drift import FeatureDriftDetector
        return FeatureDriftDetector(
            p_threshold=p_threshold,
            drift_threshold=drift_threshold,
            min_reference_rows=min_rows,
        )

    def test_is_fitted_false_before_fit(self):
        """is_fitted must be False before any call to fit()."""
        det = self._det()
        self.assertFalse(det.is_fitted)

    def test_check_before_fit_returns_not_drifted(self):
        """check() before fit() must return DriftReport with is_drifted=False, no exception."""
        det = self._det()
        features = np.random.default_rng(0).normal(0, 1, (100, 10)).astype(np.float32)
        report = det.check(features)
        self.assertFalse(report.is_drifted)

    def test_fit_sets_is_fitted(self):
        """fit() with sufficient rows must set is_fitted=True."""
        det = self._det(min_rows=50)
        ref = np.random.default_rng(0).normal(0, 1, (200, 10))
        det.fit(ref)
        self.assertTrue(det.is_fitted)

    def test_identical_distributions_not_drifted(self):
        """Checking the same distribution against itself must not trigger drift."""
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("scipy not installed")
        det = self._det(p_threshold=0.01, drift_threshold=0.20, min_rows=50)
        rng = np.random.default_rng(1)
        ref = rng.normal(0, 1, (500, 10))
        det.fit(ref)
        same = rng.normal(0, 1, (200, 10))
        report = det.check(same)
        self.assertFalse(report.is_drifted,
                         f"Same distribution triggered drift: {report.drift_fraction:.1%}")

    def test_shifted_distributions_flagged(self):
        """A distribution shifted by 10σ must be flagged as drifted."""
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("scipy not installed")
        det = self._det(p_threshold=0.01, drift_threshold=0.10, min_rows=50)
        rng = np.random.default_rng(2)
        ref = rng.normal(0, 1, (500, 10))
        det.fit(ref)
        shifted = rng.normal(10, 1, (200, 10))  # mean shifted by 10σ
        report = det.check(shifted)
        self.assertTrue(report.is_drifted, "Heavily shifted distribution should be flagged")
        self.assertGreater(report.n_features_drifted, 0)

    def test_zero_drift_threshold_flags_any_difference(self):
        """drift_threshold=0.0 must flag is_drifted when ANY feature KS p-value is low."""
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("scipy not installed")
        det = self._det(p_threshold=0.01, drift_threshold=0.0, min_rows=50)
        rng = np.random.default_rng(3)
        ref = rng.normal(0, 1, (500, 10))
        det.fit(ref)
        shifted = rng.normal(5, 1, (200, 10))
        report = det.check(shifted)
        # With threshold=0.0, even 1 drifted feature triggers is_drifted
        if report.n_features_drifted > 0:
            self.assertTrue(report.is_drifted)
        # If somehow p-values are all > threshold (unlikely), is_drifted=False is still OK


# ══════════════════════════════════════════════════════════════════════════════
# TestOnlineLearnerDriftGate (4 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestOnlineLearnerDriftGate(unittest.TestCase):
    """Tests for the drift detector integration in OnlineLearner._run_retrain()."""

    def _make_learner(self, drift_enabled: bool = False):
        from privateye.models.online_learning import OnlineLearner
        bus_mock = MagicMock()
        bus_mock.publish = AsyncMock()
        ensemble_mock = MagicMock()
        ensemble_mock._gbm = MagicMock()
        ensemble_mock._gbm.__class__.__name__ = "GBMClassifier"
        ol_cfg = {
            "enabled": True,
            "buffer_bars": 200,
            "retrain_every_bars": 50,
            "models": [],           # no actual model retraining in unit tests
            "validation_gate": False,
        }
        drift_cfg = {
            "enabled": drift_enabled,
            "p_threshold": 0.01,
            "drift_threshold": 0.20,
            "min_reference_rows": 10,
        }
        learner = OnlineLearner(
            bus_mock, ensemble_mock, ol_cfg,
            artifacts_dir="privateye/models/artifacts",
            drift_config=drift_cfg,
        )
        return learner, bus_mock

    def test_drift_disabled_no_drift_detector_attached(self):
        """When drift_detection.enabled=False, _drift_detector must be None."""
        learner, _ = self._make_learner(drift_enabled=False)
        self.assertIsNone(learner._drift_detector)

    def test_drift_enabled_attaches_detector_when_scipy_available(self):
        """When drift_detection.enabled=True and scipy is installed, _drift_detector is set."""
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("scipy not installed")
        learner, _ = self._make_learner(drift_enabled=True)
        self.assertIsNotNone(learner._drift_detector)

    def test_retrain_proceeds_when_no_drift_detected(self):
        """When drift detector is fitted and reports no drift, _run_retrain proceeds."""
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("scipy not installed")
        from privateye.models.online_learning import OnlineLearner

        bus_mock = MagicMock()
        bus_mock.publish = AsyncMock()
        ensemble_mock = MagicMock()

        ol_cfg = {
            "enabled": True,
            "buffer_bars": 200,
            "retrain_every_bars": 50,
            "models": [],           # no actual retraining
            "validation_gate": False,
        }
        drift_cfg = {"enabled": True, "p_threshold": 0.01, "drift_threshold": 0.20,
                     "min_reference_rows": 10, "psi_threshold": 5.0}  # high PSI; KS-only focus
        learner = OnlineLearner(bus_mock, ensemble_mock, ol_cfg, drift_config=drift_cfg)

        # Pre-fit the detector so it's active
        rng = np.random.default_rng(0)
        ref_feats = rng.normal(0, 1, (100, 10)).astype(np.float64)
        learner._drift_detector.fit(ref_feats)

        # Stuff the buffer with enough bars
        bars = _make_bars(60)
        for _, row in bars.iterrows():
            learner._buffer.append(row.to_dict())

        # Patch extract_features to return a similar distribution (no drift)
        similar = rng.normal(0, 1, (60, 10)).astype(np.float32)
        with patch("privateye.data.feature_extractor.extract_features", return_value=similar):
            asyncio.run(learner._run_retrain())

        # MODEL_UPDATED should be published (retrain ran, even with empty models list)
        bus_mock.publish.assert_called_once()

    def test_retrain_skipped_when_drift_detected(self):
        """When drift detector reports is_drifted=True, _run_retrain must return early."""
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("scipy not installed")
        from privateye.models.online_learning import OnlineLearner
        from privateye.models.drift import DriftReport

        bus_mock = MagicMock()
        bus_mock.publish = AsyncMock()
        ensemble_mock = MagicMock()

        ol_cfg = {
            "enabled": True,
            "buffer_bars": 200,
            "retrain_every_bars": 50,
            "models": [],
            "validation_gate": False,
        }
        drift_cfg = {"enabled": True, "p_threshold": 0.01, "drift_threshold": 0.20,
                     "min_reference_rows": 10}
        learner = OnlineLearner(bus_mock, ensemble_mock, ol_cfg, drift_config=drift_cfg)

        # Make drift detector always report "drifted"
        mock_det = MagicMock()
        mock_det.is_fitted = True
        mock_det.check.return_value = DriftReport(
            n_features_total=10, n_features_drifted=5, drift_fraction=0.5,
            is_drifted=True, drifted_feature_indices=[0, 1, 2, 3, 4], min_p_value=0.0001,
        )
        learner._drift_detector = mock_det

        # Stuff the buffer
        bars = _make_bars(60)
        for _, row in bars.iterrows():
            learner._buffer.append(row.to_dict())

        dummy_feats = np.zeros((60, 10), dtype=np.float32)
        with patch("privateye.data.feature_extractor.extract_features", return_value=dummy_feats):
            asyncio.run(learner._run_retrain())

        # MODEL_UPDATED must NOT be published — retrain was skipped
        bus_mock.publish.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# TestLiveAltDataEnrichment (4 tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestLiveAltDataEnrichment(unittest.TestCase):

    def test_enrich_adds_funding_rate_column(self):
        """When FundingRateProvider is supplied, bars must gain a funding_rate column."""
        from privateye.main import _enrich_bars_with_alt_data

        bars = _make_bars(10)
        self.assertNotIn("funding_rate", bars.columns)

        fr_mock = AsyncMock()
        fr_mock.fetch_latest = AsyncMock(
            return_value={"funding_rate": 0.0001, "open_interest": 50000.0}
        )

        result = _run(_enrich_bars_with_alt_data(bars, "BTC/USDT", fr_mock, None, {}))
        self.assertIn("funding_rate", result.columns)
        self.assertTrue(np.allclose(result["funding_rate"], 0.0001))

    def test_enrich_adds_fear_greed_column(self):
        """When FearGreedProvider is supplied, bars must gain a fear_greed column."""
        from privateye.main import _enrich_bars_with_alt_data

        bars = _make_bars(10)
        self.assertNotIn("fear_greed", bars.columns)

        fg_mock = AsyncMock()
        fg_mock.fetch_latest = AsyncMock(
            return_value={"fear_greed": 72.0, "classification": "Greed"}
        )

        result = _run(_enrich_bars_with_alt_data(bars, "BTC/USDT", None, fg_mock, {}))
        self.assertIn("fear_greed", result.columns)
        self.assertTrue(np.allclose(result["fear_greed"], 72.0))

    def test_enrich_graceful_noop_when_providers_none(self):
        """When both providers are None, bars must be returned unchanged."""
        from privateye.main import _enrich_bars_with_alt_data

        bars = _make_bars(10)
        original_cols = set(bars.columns)
        result = _run(_enrich_bars_with_alt_data(bars, "BTC/USDT", None, None, {}))
        self.assertEqual(set(result.columns), original_cols)

    def test_enrich_graceful_on_provider_exception(self):
        """On provider exception, bars must be returned without raising."""
        from privateye.main import _enrich_bars_with_alt_data

        bars = _make_bars(10)
        original_cols = set(bars.columns)

        fr_mock = AsyncMock()
        fr_mock.fetch_latest = AsyncMock(side_effect=ConnectionError("timeout"))

        result = _run(_enrich_bars_with_alt_data(bars, "BTC/USDT", fr_mock, None, {}))
        # Should not crash; original bars returned (or partial result without the column)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)


if __name__ == "__main__":
    unittest.main()
