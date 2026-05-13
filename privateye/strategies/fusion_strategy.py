"""
FusionStrategy: aggregates rule-based strategies + ML model predictions
into a single weighted-vote signal.

Fusion algorithm:
  1. Collect direction + confidence from all sources (DirectionalStrategy,
     MeanReversionStrategy, LSTMForecaster, RLPolicy).
  2. Normalise each source's weight by its trailing 30-trade Sharpe (so
     recently profitable sources get more weight).
  3. Weighted vote: score_long = Σ conf_i × weight_i for direction=LONG.
  4. GBM gate: if GBM P(profitable) < gate_threshold, suppress non-FLAT signals.
  5. Regime gate: suppress directional strategies in ranging regimes; suppress
     MeanReversion in trending regimes.
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from privateye.core.types import DataSnapshot, Direction, TradingSignal
from privateye.indicators.library import atr
from privateye.models.inference import ModelEnsemble
from privateye.strategies.base import AbstractStrategy
from privateye.strategies.directional import DirectionalStrategy
from privateye.strategies.mean_reversion import MeanReversionStrategy
from privateye.utils.logging import get_logger
from privateye.utils.time import now_utc

log = get_logger()

# Regime integers from RegimeDetector (convention):
# 0=trending-bull, 1=trending-bear, 2=ranging-low-vol, 3=ranging-high-vol
_TRENDING_REGIMES = {0, 1}
_RANGING_REGIMES  = {2, 3}


class FusionStrategy(AbstractStrategy):
    """
    Meta-strategy that combines DirectionalStrategy, MeanReversionStrategy,
    LSTMForecaster, and RLPolicy via confidence-weighted voting.
    """

    STRATEGY_ID = "fusion"

    def __init__(self, config: dict[str, Any]) -> None:
        config.setdefault("strategy_id", self.STRATEGY_ID)
        super().__init__(config)

        self.timeframe: str = config.get("timeframe", "1h")

        # Sub-strategies (rule-based)
        dir_cfg = dict(config)
        dir_cfg.update({"strategy_id": "fusion_directional", "enabled": True})
        self._directional = DirectionalStrategy(dir_cfg)

        mr_cfg = dict(config)
        mr_cfg.update({"strategy_id": "fusion_mean_reversion", "enabled": True})
        self._mean_reversion = MeanReversionStrategy(mr_cfg)

        # ML ensemble
        artifacts_dir = config.get("artifacts_dir", "privateye/models/artifacts")
        self._ensemble = ModelEnsemble(artifacts_dir=artifacts_dir)

        # Gate settings
        self.gbm_gate: bool  = config.get("gbm_gate", True)
        self.gbm_threshold: float = config.get("gbm_gate_threshold", 0.45)
        self.regime_gate: bool = config.get("regime_gate", True)
        self.sharpe_window: int = config.get("sharpe_window_trades", 30)

        # Trailing Sharpe tracking per source
        self._source_pnls: dict[str, deque] = {
            "directional":    deque(maxlen=self.sharpe_window),
            "mean_reversion": deque(maxlen=self.sharpe_window),
            "lstm":           deque(maxlen=self.sharpe_window),
            "rl":             deque(maxlen=self.sharpe_window),
            "lgbm":           deque(maxlen=self.sharpe_window),
            "attn_lstm":      deque(maxlen=self.sharpe_window),
        }

        self._current_regime: int = 0

        # Phase 5 — Multi-timeframe confirmation gate (disabled by default)
        mtf_cfg = config.get("mtf_confirmation", {})
        self._mtf_enabled: bool = bool(mtf_cfg.get("enabled", False))
        self._mtf_higher_tf: str = str(mtf_cfg.get("higher_timeframe", "4h"))
        self._higher_tf_bars: dict = {}   # symbol → latest higher-TF bars DataFrame

        # Phase 4 — cached model version id (updated on init + MODEL_UPDATED events)
        self._cached_model_version: str = self._load_model_version_id(
            config.get("artifacts_dir", "privateye/models/artifacts")
        )

        # Optional FeeOptimizer — screens marginal trades and chooses order type
        self._fee_optimizer = None
        if config.get("use_fee_optimizer", False):
            from privateye.execution.fee_optimizer import FeeOptimizer
            self._fee_optimizer = FeeOptimizer(
                fee_maker=config.get("fee_maker", 0.001),
                fee_taker=config.get("fee_taker", 0.001),
                slippage_pct=config.get("slippage_pct", 0.0005),
                min_net_return=config.get("fee_adjusted_min_return", 0.003),
            )

    # ── Entry point ──────────────────────────────────────────────────────────

    def on_data(self, snapshot: DataSnapshot) -> list[TradingSignal]:
        if not self.enabled or snapshot.timeframe != self.timeframe:
            return []
        bars = snapshot.bars
        if len(bars) < 250:
            return []

        # Update regime — prefer neural_regime when available
        ml_result = self._ensemble.predict(bars)
        regime_key = "neural_regime" if "neural_regime" in ml_result else "regime"
        if regime_key in ml_result:
            self._current_regime = int(ml_result[regime_key][0])

        # ── Phase 2: use stacking meta-learner when fitted ─────────────────
        stacking = ml_result.get("stacking")
        if stacking is not None:
            # Phase 4: predict() now returns 3-tuple (direction, confidence, attribution)
            if len(stacking) == 3:
                direction, confidence, stacking_attribution = stacking
            else:
                direction, confidence = stacking
                stacking_attribution = {}
            if direction == "flat":
                return []
            # Phase 5: MTF confirmation gate (fail-open when HTF bars unavailable)
            if self._mtf_suppresses(snapshot.symbol, direction):
                return []
            # Build the signal using the stacking output
            base = self._build_base_signal(
                snapshot, direction, confidence,
                {"stacking": (direction, confidence)}, ml_result,
                stacking_attribution=stacking_attribution,
            )
            if self._fee_optimizer is not None:
                preferred = self._fee_optimizer.choose_order_type(base)
                if preferred is None:
                    log.debug(f"[FusionStrategy] Signal fee-screened: {base.symbol}")
                    return []
                base.metadata["preferred_order_type"] = preferred.value
            return [base]

        # ── Fallback: weighted vote (original path) ────────────────────────
        # Collect all source signals
        sources: dict[str, tuple[str, float]] = {}

        # Rule-based — apply regime gate
        if not (self.regime_gate and self._current_regime in _RANGING_REGIMES):
            dir_sigs = self._directional.on_data(snapshot)
            if dir_sigs:
                s = dir_sigs[0]
                sources["directional"] = (s.direction.value, s.confidence)

        if not (self.regime_gate and self._current_regime in _TRENDING_REGIMES):
            mr_sigs = self._mean_reversion.on_data(snapshot)
            if mr_sigs:
                s = mr_sigs[0]
                sources["mean_reversion"] = (s.direction.value, s.confidence)

        # ML models (Phase 1 original)
        if "lstm" in ml_result:
            sources["lstm"] = ml_result["lstm"]

        if "rl" in ml_result:
            sources["rl"] = ml_result["rl"]

        # Phase 2 additional sources (fallback path only)
        lgbm_prob = ml_result.get("lgbm", None)
        if lgbm_prob is not None:
            lgbm_threshold = self.gbm_threshold  # reuse same threshold
            lgbm_dir = "long" if float(lgbm_prob) >= lgbm_threshold else "flat"
            sources["lgbm"] = (lgbm_dir, float(lgbm_prob))

        if "attn_lstm" in ml_result:
            sources["attn_lstm"] = ml_result["attn_lstm"]

        # GBM gate
        gbm_prob = ml_result.get("gbm", 0.5)
        gbm_pass = (not self.gbm_gate) or float(gbm_prob) >= self.gbm_threshold

        if not sources:
            return []

        # Compute weighted vote
        direction, confidence = self._weighted_vote(sources, gbm_pass)

        if direction == "flat":
            return []

        # Phase 5: MTF confirmation gate (fail-open when HTF bars unavailable)
        if self._mtf_suppresses(snapshot.symbol, direction):
            return []

        # Build signal from best rule-based sub-signal (for stop/target prices)
        base = self._build_base_signal(snapshot, direction, confidence,
                                       sources, ml_result)

        # Apply FeeOptimizer: screen marginal trades and annotate order type
        if self._fee_optimizer is not None:
            preferred = self._fee_optimizer.choose_order_type(base)
            if preferred is None:
                log.debug(f"[FusionStrategy] Signal fee-screened: {base.symbol}")
                return []
            base.metadata["preferred_order_type"] = preferred.value

        return [base]

    def on_bar_end(self, snapshot: DataSnapshot, portfolio: Any) -> list[TradingSignal]:
        # Hotfix 2026-05-07: only delegate exit management to ``_directional``.
        # MeanReversionStrategy's exit logic uses BB-based stops/targets that
        # don't reflect the FusionStrategy-computed entry levels, and on every
        # BUY fill it would open a phantom internal position with placeholder
        # stop=fill*0.98 / target=fill*1.02 that didn't match the real order.
        # Since ``_directional.on_fill`` mirrors the simulator's authoritative
        # Position via ``dataclasses.replace(sim_pos)``, only directional has
        # the correct stop/target/trailing context to manage the exit.
        exits = self._directional.on_bar_end(snapshot, portfolio)
        # Hotfix 2026-05-07 (round-trip): rewrite strategy_id from the
        # sub-strategy ID ("fusion_directional") to the parent ID ("fusion")
        # so the resulting fill flows back through FusionStrategy.on_fill.
        # Without this, the on_fill filter in main.run_paper / backtest engine
        # (``strategy.strategy_id == fill.strategy_id``) silently drops the
        # exit fill — directional's internal tracker never closes, bars_held
        # keeps incrementing, and on_bar_end emits a stale exit signal every
        # subsequent bar (rejected as "Signal rejected: exit"). The same
        # mismatch then prevents on_fill from opening the next BUY's tracker,
        # leading to permanent strategy/simulator state divergence.
        for s in exits:
            s.strategy_id = self.STRATEGY_ID
        return exits

    def on_fill(self, fill: Any, portfolio: Any) -> None:
        # Only directional tracks the position — see on_bar_end comment.
        self._directional.on_fill(fill, portfolio)
        # Record P&L for trailing Sharpe (use realised_pnl from the fill)
        pnl = getattr(fill, "realised_pnl", 0.0)
        if pnl != 0.0:
            for q in self._source_pnls.values():
                q.append(pnl)

    # ── Voting ────────────────────────────────────────────────────────────────

    _VALID_DIRECTIONS = ("long", "short", "flat")

    def _weighted_vote(
        self,
        sources: dict[str, tuple[str, float]],
        gbm_pass: bool,
    ) -> tuple[str, float]:
        weights = self._compute_weights(list(sources.keys()))

        # Defense-in-depth: any source returning a non-canonical label (e.g. RL
        # action 3 = "hold") is treated as a no-op contributor. Without this,
        # an unknown direction would propagate to Direction(direction) and
        # crash the bus handler. The RLPolicy maps "hold" → ("flat", 0.0) at
        # source so this filter rarely fires, but keeps any future model
        # introducing a new label from breaking signal evaluation.
        score: dict[str, float] = {"long": 0.0, "short": 0.0, "flat": 0.0}
        for name, (direction, conf) in sources.items():
            if direction not in self._VALID_DIRECTIONS:
                log.debug(
                    f"[FusionStrategy] {name} returned non-canonical direction "
                    f"'{direction}' — treating as no-op"
                )
                continue
            w = weights.get(name, 1.0)
            score[direction] = score[direction] + conf * w

        total = sum(score.values())
        if total == 0:
            # No canonical source contributed — fall to flat with no opinion.
            # Without this, max() on tied zeros picks "long" (first key) and a
            # zero-confidence long signal propagates pointlessly downstream.
            return "flat", 0.0

        best_dir = max(score, key=lambda d: score[d])
        confidence = score[best_dir] / total

        if best_dir != "flat" and not gbm_pass:
            log.debug(f"[FusionStrategy] GBM gate blocked {best_dir} signal.")
            return "flat", 0.0

        return best_dir, float(confidence)

    # ── Phase 4: model version caching ────────────────────────────────────────

    @staticmethod
    def _load_model_version_id(artifacts_dir: str) -> str:
        """Return the latest GBMClassifier checkpoint version string, or ''."""
        try:
            from pathlib import Path
            from privateye.models.checkpoint import list_checkpoints
            ckpts = list_checkpoints("GBMClassifier", Path(artifacts_dir) / "checkpoints")
            if ckpts:
                return ckpts[0].get("saved_at", "")
        except Exception:
            pass
        return ""

    def refresh_model_version_id(self) -> None:
        """Refresh the cached model version id (call on MODEL_UPDATED events)."""
        artifacts_dir = getattr(self._ensemble, "_artifacts_dir",
                                "privateye/models/artifacts")
        self._cached_model_version = self._load_model_version_id(str(artifacts_dir))

    # ── Phase 5: Multi-timeframe confirmation ────────────────────────────────

    def update_higher_tf_bars(self, symbol: str, bars: Any) -> None:
        """Update the cached higher-timeframe bars for a symbol.

        Called by main.py / paper loop when a higher-TF bar closes.
        Used by the MTF confirmation gate in on_data() when mtf_confirmation.enabled=true.
        """
        self._higher_tf_bars[symbol] = bars

    def _mtf_suppresses(self, symbol: str, direction: str) -> bool:
        """Return True when the MTF gate should suppress the signal.

        Fail-open contract: returns False (signal passes) whenever:
          - MTF confirmation is disabled
          - direction is "flat" (exits always allowed)
          - higher-TF bars are not yet available for the symbol
          - fewer than 50 higher-TF bars are available

        Only suppresses when HTF EMA(20) disagrees with the signal direction
        AND at least 50 higher-TF bars are available.
        """
        if not self._mtf_enabled or direction == "flat":
            return False
        htf_bars = self._higher_tf_bars.get(symbol)
        if htf_bars is None or len(htf_bars) < 50:
            return False   # fail-open — no higher-TF data yet
        try:
            htf_ema = float(htf_bars["close"].ewm(span=20).mean().iloc[-1])
            htf_price = float(htf_bars["close"].iloc[-1])
            htf_direction = "long" if htf_price > htf_ema else "flat"
            if htf_direction != direction:
                log.debug(
                    f"[FusionStrategy] MTF gate: {symbol} lower-TF={direction} "
                    f"but {self._mtf_higher_tf} EMA(20) says {htf_direction} — suppressing"
                )
                return True
        except Exception:
            return False   # fail-open on any error
        return False

    def _compute_weights(self, source_names: list[str]) -> dict[str, float]:
        weights: dict[str, float] = {}
        for name in source_names:
            pnl_hist = list(self._source_pnls.get(name, []))
            if len(pnl_hist) >= 5:
                arr  = np.array(pnl_hist)
                std  = arr.std()
                mean = arr.mean()
                sharpe = (mean / std) if std > 0 else 0.0
                weights[name] = max(0.1, 1.0 + sharpe)  # floor at 0.1
            else:
                weights[name] = 1.0
        return weights

    def _build_base_signal(
        self,
        snapshot: DataSnapshot,
        direction: str,
        confidence: float,
        sources: dict[str, tuple[str, float]],
        ml_result: dict,
        stacking_attribution: dict | None = None,
    ) -> TradingSignal:
        bars    = snapshot.bars
        close   = float(bars["close"].iloc[-1])
        atr_val = float(atr(bars, 14).iloc[-1]) if len(bars) >= 14 else close * 0.01

        if direction == "long":
            stop   = close - 2.0 * atr_val
            target = close + 4.0 * atr_val
        else:
            stop   = close + 2.0 * atr_val
            target = close - 4.0 * atr_val

        regime_probs = ml_result.get("regime", (0, []))[1]
        regime_probs_list = list(regime_probs) if hasattr(regime_probs, "__iter__") else []

        reason = (
            f"Fusion {direction}: sources={list(sources.keys())} "
            f"conf={confidence:.2f} regime={self._current_regime} "
            f"gbm={ml_result.get('gbm', 'n/a')}"
        )

        # Top SHAP feature importances for XAI decision trail
        shap = ml_result.get("shap")
        if shap:
            sorted_features = sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)
            top_features = [
                {"feature": k, "importance": float(v)}
                for k, v in sorted_features[:5]
            ]
        else:
            top_features = []

        # Phase 4 — stacking attribution and model version
        attribution = stacking_attribution if stacking_attribution is not None else {}
        # For the fallback (weighted-vote) path, get attribution from ml_result if present
        if not attribution:
            stk = ml_result.get("stacking")
            if stk is not None and len(stk) == 3:
                attribution = stk[2] or {}

        return TradingSignal(
            symbol=snapshot.symbol,
            direction=Direction(direction),
            confidence=confidence,
            entry_price=close,
            stop_price=stop,
            target_price=target,
            strategy_id=self.STRATEGY_ID,
            timeframe=snapshot.timeframe,
            timestamp=now_utc(),
            metadata={
                "reason": reason,
                "sources": {k: v for k, v in sources.items()},
                "regime": self._current_regime,
                "regime_probs": regime_probs_list,
                "gbm_gate": float(ml_result.get("gbm", 0.5)),
                "top_features": top_features,
                "stacking_attribution": attribution,          # Phase 4
                "model_version_id": self._cached_model_version,  # Phase 4
            },
        )
