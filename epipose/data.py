"""Feature loading and batching.

Reads four files from data/features/:
  pose_rep.pkl        pose features of the representative apo structure
  pose_multi_apo.pkl  pose features of further apo structures of the same antigen
  residue.pkl         residue properties of the antigen
  delta_target.pkl    bound/unbound structural change, training units only
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

SPLITS = ("train", "val", "test")
NORM_UNITS = 200          # training units used for the residue feature statistics


@dataclass
class Unit:
    """One antigen-antibody pair, with one pose view per apo structure."""
    gid: str
    split: str
    n_res: int
    y: torch.Tensor                      # (nres,) epitope labels
    res: torch.Tensor                    # (nres, Fr) residue features
    views: list = field(default_factory=list)   # [{x, contact, q}]
    delta: torch.Tensor | None = None    # (nres, D) auxiliary target


def _unpack(bits: np.ndarray, n: int) -> np.ndarray:
    return np.unpackbits(bits, axis=1)[:, :n].astype(bool)


class FeatureStore:
    """Loads the pickles into Unit objects held on the given device."""

    def __init__(self, root: str | Path, device: torch.device, *,
                 top_poses: int = 500, use_multi_apo: bool = True,
                 use_delta: bool = True, score_only_test: bool = True):
        root = Path(root)
        with open(root / "pose_rep.pkl", "rb") as f:
            rep = pickle.load(f)
        self.pose_names: list[str] = list(rep["name"])
        feat = rep["feat"]
        with open(root / "residue.pkl", "rb") as f:
            res = pickle.load(f)
        self.res_names: list[str] = list(res["name"])
        res = res["feat"]

        multi: dict[str, list] = {}
        if use_multi_apo and (root / "pose_multi_apo.pkl").exists():
            with open(root / "pose_multi_apo.pkl", "rb") as f:
                for (gid, k), r in pickle.load(f)["feat"].items():
                    multi.setdefault(gid, []).append(r)

        delta: dict[str, np.ndarray] = {}
        self.delta_names: list[str] = []
        if use_delta and (root / "delta_target.pkl").exists():
            with open(root / "delta_target.pkl", "rb") as f:
                _d = pickle.load(f)
            delta = _d["feat"]
            self.delta_names = list(_d["name"])

        self.s0_col = self.pose_names.index("s0") if "s0" in self.pose_names else None
        self.top_poses = top_poses
        self.device = device

        self.units: dict[str, Unit] = {}
        self.by_split: dict[str, list[str]] = {s: [] for s in SPLITS}
        for gid, r in feat.items():
            if gid not in res:
                continue
            if r["split"] == "test" and score_only_test and not r.get("score", False):
                continue
            n = int(r["nres_d"])
            u = Unit(gid=gid, split=r["split"], n_res=n,
                     y=torch.tensor(r["y"][:n], dtype=torch.float32, device=device),
                     res=torch.tensor(np.asarray(res[gid][:n], np.float32),
                                      device=device))
            u.views.append(self._view(r, n))
            if r["split"] == "train":
                for r2 in multi.get(gid, []):
                    if int(r2["nres_d"]) == n:
                        u.views.append(self._view(r2, n))
                if gid in delta:
                    dv = np.asarray(delta[gid], np.float32)[:n]
                    u.delta = torch.tensor(dv, device=device)
            self.units[gid] = u
            self.by_split[r["split"]].append(gid)
        for k in self.by_split:
            self.by_split[k].sort()

        xs = np.concatenate([feat[g]["F"] for g in self.by_split["train"]]
                            ).astype(np.float64)
        self.x_mu = torch.tensor(xs.mean(0), dtype=torch.float32, device=device)
        self.x_sd = torch.tensor(xs.std(0) + 1e-6, dtype=torch.float32, device=device)
        del xs
        src = self.by_split["train"][:NORM_UNITS]
        rs = np.concatenate([np.asarray(res[g], np.float64) for g in src])
        self.r_mu = torch.tensor(rs.mean(0), dtype=torch.float32, device=device)
        self.r_sd = torch.tensor(rs.std(0) + 1e-6, dtype=torch.float32, device=device)
        del rs

    def _view(self, r, n: int) -> dict:
        F = r["F"]
        q = np.asarray(r["q"], np.float32)
        C = _unpack(r["C"], n)
        if self.top_poses and self.s0_col is not None and self.top_poses < len(F):
            order = np.argsort(-F[:, self.s0_col])[:self.top_poses]
            F, q, C = F[order], q[order], C[order]
        return dict(x=torch.tensor(F.astype(np.float32), device=self.device),
                    contact=torch.tensor(C, device=self.device),
                    q=torch.tensor(q, device=self.device))

    @property
    def n_pose_feat(self) -> int:
        return len(self.pose_names)

    @property
    def n_res_feat(self) -> int:
        return len(self.res_names)

    def pick_view(self, gid: str, train: bool) -> dict:
        """Sample an apo structure during training; always use the first one
        for evaluation."""
        views = self.units[gid].views
        if train and len(views) > 1:
            return views[int(np.random.randint(len(views)))]
        return views[0]

    def batch(self, gids: list[str], train: bool):
        """Returns (pose features, residue features, residue mask, views).
        The residue axis is padded to the longest antigen in the batch."""
        us = [self.units[g] for g in gids]
        views = [self.pick_view(g, train) for g in gids]
        nmax = max(u.n_res for u in us)
        x = torch.stack([(v["x"] - self.x_mu) / self.x_sd for v in views])
        r = torch.zeros(len(us), nmax, self.n_res_feat, device=self.device)
        mask = torch.ones(len(us), nmax, dtype=torch.bool, device=self.device)
        for j, u in enumerate(us):
            r[j, :u.n_res] = (u.res - self.r_mu) / self.r_sd
            mask[j, :u.n_res] = False
        return x, r, mask, views

    def summary(self) -> str:
        n = {s: len(v) for s, v in self.by_split.items()}
        nv = np.mean([len(self.units[g].views) for g in self.by_split["train"]])
        nd = sum(1 for g in self.by_split["train"] if self.units[g].delta is not None)
        return (f"units: train {n['train']}  val {n['val']}  test {n['test']}\n"
                f"  {self.n_pose_feat} pose features, "
                f"{self.n_res_feat} residue features\n"
                f"  {nv:.2f} apo structures per training unit, "
                f"{nd} units with an auxiliary target")
