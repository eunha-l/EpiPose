"""Training losses.

  pose_bce    main loss, fits the pose quality q = |C and y| / |C|
  delta_rank  auxiliary loss, ranks poses by the structural change of the
              residues they cover

The auxiliary target is the bound/unbound change of the antigen surface, which
exists for training units only and never enters inference.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pose_bce(logit: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Binary cross entropy on the pose quality. logit (P,), q (P,) in [0, 1]."""
    return F.binary_cross_entropy_with_logits(logit, q)


def _pairwise_rank(logit: torch.Tensor, target: torch.Tensor,
                   margin: float = 0.0) -> torch.Tensor:
    """Orders logits over the pairs whose targets differ by more than margin."""
    keep = (target[:, None] - target[None, :]) > margin
    if not bool(keep.any()):
        return logit.sum() * 0.0
    return -F.logsigmoid(logit[:, None] - logit[None, :])[keep].mean()


def _delta_mean(contact: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Mean target over the residues a pose covers.
    contact (P, N), delta (N, D) -> (P, D)."""
    c = contact.float()
    return (c @ delta) / c.sum(1, keepdim=True).clamp(min=1)


def delta_rank(logit: torch.Tensor, contact: torch.Tensor, delta: torch.Tensor,
               margin: float = 0.0) -> torch.Tensor:
    """Averages the ranking loss over the target columns."""
    dm = _delta_mean(contact, delta)
    terms = [_pairwise_rank(logit, dm[:, j], margin) for j in range(dm.shape[1])]
    return torch.stack(terms).mean()
