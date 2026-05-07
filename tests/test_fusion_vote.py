"""
FusionStrategy._weighted_vote — defense-in-depth tests.

Discovered in production (paper-mode log, 2026-05-06): non-canonical direction
strings (e.g. RL action 3 = "hold") propagated into the weighted vote, became
the winning ``best_dir``, and then crashed at ``Direction(best_dir)``.

The fix in ``_weighted_vote`` filters the source set down to the canonical
``{"long", "short", "flat"}`` triplet so any future model emitting a new label
becomes a no-op contributor instead of a crash.
"""
from __future__ import annotations

from typing import Any

import pytest

from privateye.strategies.fusion_strategy import FusionStrategy


def _minimal_cfg(**overrides: Any) -> dict:
    cfg = {
        "strategy_id":         "fusion",
        "timeframe":           "1h",
        "enabled":             True,
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "ema_trend": 200, "rsi_period": 14,
        "rsi_overbought": 70, "rsi_oversold": 30,
        "bb_period": 20, "bb_std": 2.0,
        "atr_stop_multiplier": 2.0, "max_bars_in_trade": 48,
        "gbm_gate":           True,
        "gbm_gate_threshold": 0.45,
        "regime_gate":        True,
        "sharpe_window_trades": 30,
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def strat():
    """Constructs a FusionStrategy. ModelEnsemble is lazy-loaded so this is cheap."""
    return FusionStrategy(_minimal_cfg())


# ── Defense in depth: rogue direction strings must not crash the vote ────────

class TestNonCanonicalDirection:

    def test_hold_rogue_label_is_filtered(self, strat):
        """RL emitting 'hold' (action 3) must not crash _weighted_vote.
        It must contribute nothing — the lstm 'long' must win."""
        sources = {
            "rl":   ("hold", 0.9),    # rogue label, large weight
            "lstm": ("long", 0.6),
        }
        direction, confidence = strat._weighted_vote(sources, gbm_pass=True)
        assert direction == "long"
        assert confidence > 0

    def test_arbitrary_rogue_label_is_filtered(self, strat):
        """Any future label outside {long, short, flat} is treated as no-op."""
        sources = {
            "experimental": ("buy_strong", 0.95),
            "directional":  ("short", 0.5),
        }
        direction, confidence = strat._weighted_vote(sources, gbm_pass=True)
        assert direction == "short"
        assert confidence > 0

    def test_all_rogue_labels_returns_flat(self, strat):
        """If every source emits a rogue label, the vote falls to FLAT (no contribution)."""
        sources = {
            "rl":           ("hold", 0.9),
            "experimental": ("buy_strong", 0.7),
        }
        direction, confidence = strat._weighted_vote(sources, gbm_pass=True)
        assert direction == "flat"
        assert confidence == 0.0


# ── Sanity: canonical labels still vote correctly ────────────────────────────

class TestCanonicalVote:

    def test_long_wins_with_two_long_sources(self, strat):
        sources = {
            "directional": ("long", 0.7),
            "lstm":        ("long", 0.5),
            "rl":          ("flat", 0.0),
        }
        direction, _ = strat._weighted_vote(sources, gbm_pass=True)
        assert direction == "long"

    def test_gbm_gate_blocks_non_flat(self, strat):
        """When gbm_pass=False, even a strong long vote downgrades to flat."""
        sources = {
            "directional": ("long", 0.9),
            "lstm":        ("long", 0.7),
        }
        direction, confidence = strat._weighted_vote(sources, gbm_pass=False)
        assert direction == "flat"
        assert confidence == 0.0

    def test_short_wins_when_majority_short(self, strat):
        sources = {
            "directional": ("short", 0.8),
            "mean_reversion": ("short", 0.6),
            "lstm": ("long", 0.3),
        }
        direction, _ = strat._weighted_vote(sources, gbm_pass=True)
        assert direction == "short"


# ── Regression: the exact bug from production logs ───────────────────────────

class TestProductionBugRegression:

    def test_hold_with_other_voters_no_longer_crashes(self, strat):
        """The exact failure mode from logs/privateye_2026-05-06.log @ 23:06:58:
        RL emits 'hold' alongside other sources. Previously crashed at
        Direction('hold'); now falls through cleanly."""
        sources = {
            "rl":             ("hold", 0.95),    # would have won the vote pre-fix
            "directional":    ("long", 0.4),
            "lstm":           ("long", 0.3),
            "mean_reversion": ("flat", 0.0),
        }
        # Must not raise
        direction, confidence = strat._weighted_vote(sources, gbm_pass=True)
        # And must produce a valid Direction-string
        assert direction in ("long", "short", "flat")
