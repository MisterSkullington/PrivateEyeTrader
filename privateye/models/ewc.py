"""
Elastic Weight Consolidation (EWC) — Kirkpatrick et al. 2017.

Prevents catastrophic forgetting during online LSTM incremental retraining by
penalising changes to weights that were important for the previous task.

Usage in _retrain_lstm():
    # After full initial training (model saved):
    ewc.consolidate(net, val_loader, device)

    # During next incremental retrain — add penalty to batch loss:
    loss = criterion(output, labels) + ewc.penalty(new_net)

Fisher approximation: for each parameter p,
    F_p ≈ mean(grad(log p(y|x))²) over n_samples forward passes. Diagonal only.

Penalty formula:
    lambda/2 * Σ F_i * (θ_i − θ_old_i)²  summed over all named parameters.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from privateye.utils.logging import get_logger

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader
    import torch

log = get_logger()


class ElasticWeightConsolidation:
    """Diagonal Fisher-information EWC (Kirkpatrick et al. 2017).

    Attributes
    ----------
    is_consolidated : bool
        True after a successful ``consolidate()`` call.
    """

    def __init__(self, lambda_: float = 400.0) -> None:
        self._lambda: float = float(lambda_)
        self._fisher: dict[str, "torch.Tensor"] | None = None
        self._theta_old: dict[str, "torch.Tensor"] | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def consolidate(
        self,
        model: "nn.Module",
        dataloader: "DataLoader",
        device: "torch.device",
        n_samples: int = 200,
    ) -> None:
        """Compute diagonal Fisher information and store reference parameters.

        Parameters
        ----------
        model:
            The trained network whose weights we want to protect.
        dataloader:
            Validation DataLoader yielding (x, y) batches.
        device:
            Device that *model* is currently on.
        n_samples:
            Maximum number of samples to use for Fisher estimation.
            When the DataLoader yields fewer total samples, all are used.
        """
        import torch

        if n_samples <= 0:
            log.warning("[EWC] n_samples must be > 0 — consolidation skipped")
            return

        model.eval()

        # Accumulate squared gradients per parameter
        fisher: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, device=device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        samples_seen = 0
        for xb, yb in dataloader:
            if samples_seen >= n_samples:
                break

            xb = xb.to(device)
            yb = yb.to(device)

            # Forward pass
            model.zero_grad()
            try:
                output = model(xb)
            except Exception as exc:
                log.warning(f"[EWC] Forward pass failed during consolidation: {exc}")
                continue

            # Use cross-entropy log-likelihood (two-class case)
            log_probs = torch.nn.functional.log_softmax(output, dim=-1)
            # Sample class from current predictions (empirical Fisher)
            with torch.no_grad():
                probs = torch.exp(log_probs)
                # clamp for safety
                probs = torch.clamp(probs, min=1e-8, max=1.0)
                sampled_classes = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Compute gradient of log p(sampled_class | x)
            selected_log_probs = log_probs.gather(1, sampled_classes.unsqueeze(1)).squeeze()
            loss = selected_log_probs.mean()
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2

            samples_seen += xb.size(0)

        if samples_seen == 0:
            log.warning("[EWC] No samples processed — consolidation skipped")
            return

        # Normalise by number of samples
        for name in fisher:
            fisher[name] /= float(samples_seen)

        self._fisher = fisher
        # Store a detached copy of the current parameters
        self._theta_old = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        log.info(
            f"[EWC] Consolidated {len(self._fisher)} parameter tensors "
            f"over {samples_seen} samples"
        )

    def penalty(self, model: "nn.Module") -> "torch.Tensor":
        """EWC regularisation term.

        Returns a scalar 0-tensor when ``consolidate()`` has not been called.

        Parameters
        ----------
        model:
            The network currently being trained (may differ from the one
            passed to ``consolidate()`` but must have the same architecture).
        """
        import torch

        if self._fisher is None or self._theta_old is None:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0)
        params = dict(model.named_parameters())
        for name, f in self._fisher.items():
            if name not in params:
                continue
            theta = params[name]
            theta_old = self._theta_old[name].to(theta.device)
            f = f.to(theta.device)
            penalty = penalty + (f * (theta - theta_old) ** 2).sum()

        return (self._lambda / 2.0) * penalty

    # ── Properties ──────────────────────────────────────────────────────────────

    @property
    def is_consolidated(self) -> bool:
        """True after a successful ``consolidate()`` call."""
        return self._fisher is not None and self._theta_old is not None
