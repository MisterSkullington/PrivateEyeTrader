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
        }

        self._current_regime: int = 0

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

        # Update regime
        ml_result = self._ensemble.predict(bars)
        if "regime" in ml_result:
            self._current_regime = int(ml_result["regime"][0])

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

        # ML models
        if "lstm" in ml_result:
            sources["lstm"] = ml_result["lstm"]

        if "rl" in ml_result:
            sources["rl"] = ml_result["rl"]

        # GBM gate
        gbm_prob = ml_result.get("gbm", 0.5)
        gbm_pass = (not self.gbm_gate) or float(gbm_prob) >= self.gbm_threshold

        if not sources:
            return []

        # Compute weighted vote
        direction, confidence = self._weighted_vote(sources, gbm_pass)

        if direction == "flat":
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
            },
        )
