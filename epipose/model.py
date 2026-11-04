"""EpiPose model.

Scores the docking poses of one antigen and turns those scores into a
per-residue epitope map:

    pose tokens (P, Fp) -- self attention -- cross attention -- FFN -- Linear(d, 1)
    residue tokens (N, Fr) -- Linear(d) ----------'                       |
                                                                         v
                                        map = softmax(z(logit) / tau) @ contact

Self attention lets the poses of one antigen compare against each other, and
cross attention brings the residue properties into the pose score.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Block(nn.Module):
    """Pre-norm self attention over poses, cross attention onto residues, FFN."""

    def __init__(self, dim: int, heads: int, drop: float):
        super().__init__()
        self.n1, self.n2, self.n3 = (nn.LayerNorm(dim), nn.LayerNorm(dim),
                                     nn.LayerNorm(dim))
        self.at = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.ca = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(drop),
                                nn.Linear(2 * dim, dim))
        self.dp = nn.Dropout(drop)

    def forward(self, x, r, mask):
        h = self.n1(x)
        x = x + self.dp(self.at(h, h, h, need_weights=False)[0])
        h = self.n3(x)
        x = x + self.dp(self.ca(h, r, r, key_padding_mask=mask,
                                need_weights=False)[0])
        return x + self.dp(self.ff(self.n2(x)))


class EpiPose(nn.Module):
    """Pose features and residue features to one score per pose.

    The output layer starts at zero, so training begins from the point where
    every pose carries the same weight.
    """

    def __init__(self, n_pose_feat: int, n_res_feat: int, dim: int = 48,
                 heads: int = 4, blocks: int = 2, drop: float = 0.1,
                 res_drop: float = 0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_pose_feat, dim), nn.LayerNorm(dim))
        self.rinp = nn.Sequential(nn.Linear(n_res_feat, dim), nn.LayerNorm(dim),
                                  nn.Dropout(res_drop))
        self.body = nn.ModuleList([Block(dim, heads, drop) for _ in range(blocks)])
        self.out = nn.Linear(dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, r, res_mask):
        """x (B, P, Fp), r (B, N, Fr), res_mask (B, N) with True for padding."""
        g = self.rinp(r)
        h = self.inp(x)
        for blk in self.body:
            h = blk(h, g, res_mask)
        return self.out(h).squeeze(-1)


def to_map(logit: torch.Tensor, contact: torch.Tensor,
           tau: float = 1.0) -> torch.Tensor:
    """Pose scores to a residue map. logit (P,), contact (P, N) -> (N,).

    Scores are standardised within the unit before the softmax, so a constant
    score gives every pose the same weight.
    """
    z = (logit - logit.mean()) / (logit.std() + 1e-9)
    return torch.softmax(z / tau, 0) @ contact.float()
