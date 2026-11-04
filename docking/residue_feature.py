#!/usr/bin/env python
"""Builds the antigen residue features straight from the source files.

  prop, kl, id, rsa          sequence and conservation values mapped onto the
                             docking residue axis
  kd                         Kyte-Doolittle hydrophobicity
  protrusion                 distance from the residue centre to the antigen centre
  dens6, dens10              neighbouring atoms within 6 and 10 A
  q_abs                      absolute partial charge of the residue

Usage:
  python docking/residue_feature.py --out data/features/residue.pkl
  SHARD=0/4 python docking/residue_feature.py --out part0.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "_vendor"))
import paths  # noqa: E402

NAME = ["prop", "kl", "id", "rsa", "kd", "protrusion", "dens6", "dens10", "q_abs"]
RESN = ["prop_r", "cons_kl", "cons_id", "rsa"]          # keys inside res_feat_fix
KD = {"ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5, "MET": 1.9,
      "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8, "TRP": -0.9,
      "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5, "GLN": -3.5,
      "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5}


def residue_props(tensor_file: Path, n_res: int) -> dict[str, np.ndarray]:
    """The five structure based columns, from atom coordinates and charges."""
    d = np.load(tensor_file, allow_pickle=True)
    xyz = np.asarray(d["ag"], np.float64)
    rid = np.asarray(d["ag_res"], np.int64)
    chg = np.asarray(d["ag_q"], np.float64)
    meta = np.asarray(d["ag_meta"], object)

    cnt = np.bincount(rid, minlength=n_res).astype(np.float64)
    out = {"q_abs": np.bincount(rid, weights=np.abs(chg), minlength=n_res)}

    first = np.full(n_res, -1, np.int64)
    for a in range(len(rid) - 1, -1, -1):
        first[rid[a]] = a
    rname = ["" if i < 0 else str(meta[i, 1]) for i in first]
    out["kd"] = np.array([KD.get(r, 0.0) for r in rname])

    centre = np.zeros((n_res, 3))
    for k in range(3):
        centre[:, k] = (np.bincount(rid, weights=xyz[:, k], minlength=n_res)
                        / np.maximum(cnt, 1))
    out["protrusion"] = np.linalg.norm(centre - xyz.mean(0), axis=1)

    dist = np.linalg.norm(centre[:, None, :] - xyz[None, :, :], axis=2)
    same = rid[None, :] == np.arange(n_res)[:, None]
    for radius in (6.0, 10.0):
        out[f"dens{int(radius)}"] = ((dist < radius) & ~same).sum(1).astype(float)
    return out


def sequence_props(rf: dict, c2d: np.ndarray | None, split: str, gid: str,
                   n_res: int) -> np.ndarray:
    """Maps the canonical axis of res_feat_fix onto the docking residue axis."""
    out = np.zeros((n_res, 4), np.float32)
    if c2d is None:
        return out
    for j, key in enumerate(RESN):
        full = f"{split}|{gid}|{key}"
        if full not in rf:
            continue
        v = np.nan_to_num(np.asarray(rf[full], np.float32), nan=0.0)
        ok = (c2d >= 0) & (c2d < len(v))
        m = min(n_res, len(c2d))
        out[:m, j][ok[:m]] = v[c2d[:m][ok[:m]]]
    return out


def find_tensor(gid: str, tag: str) -> Path | None:
    """The representative tensor, skipping the further apo structures."""
    suffix = f".{tag}.npz" if tag else ".npz"
    for f in sorted(paths.DOCK_OUT.glob(f"{gid}__*{suffix}")):
        stem = f.name[:-len(suffix)]
        if len(stem.split("__")) == 2:
            return f
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/features/residue.pkl")
    ap.add_argument("--tag", default="mdf00", help="antigen tensor tag")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    rf = dict(np.load(paths.require(paths.RES_FEAT, "res_feat_fix.npz"),
                      allow_pickle=True))
    c2d = {k: v for k, v in np.load(paths.require(paths.DOCK2CANON,
                                                  "dock2canon.npz")).items()}
    groups = pd.read_csv(paths.require(paths.GROUPS_CSV, "epipose_split.csv"))
    groups = groups[groups.use == True]          # noqa: E712
    split_of = dict(zip(groups.gid, groups.split))

    gids = sorted(g for g in split_of if g in c2d)
    if a.limit:
        gids = gids[:a.limit]
    shard = os.environ.get("SHARD")
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        gids = gids[i::n]
        print(f"  shard {i}/{n}: {len(gids)} units", flush=True)

    out, bad, t0 = {}, 0, time.time()
    for i, gid in enumerate(gids, 1):
        tf = find_tensor(gid, a.tag)
        if tf is None:
            bad += 1
            continue
        n_res = int(np.load(tf, allow_pickle=True)["is_core"].shape[0])
        try:
            geo = residue_props(tf, n_res)
        except Exception as e:                                    # noqa: BLE001
            print(f"  failed on {gid}: {type(e).__name__} {e}", flush=True)
            bad += 1
            continue
        seq = sequence_props(rf, c2d.get(gid), split_of[gid], gid, n_res)
        cols = [seq[:, j] for j in range(4)] + [
            np.asarray(geo[k], np.float32) for k in NAME[4:]]
        out[gid] = np.stack(cols, 1).astype(np.float32)
        if i % 100 == 0:
            print(f"  {i}/{len(gids)}  {(time.time() - t0) / 60:.1f} min  {bad} failed",
                  flush=True)

    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        pickle.dump({"name": NAME, "feat": out}, f, protocol=4)
    print(f"wrote {dest} for {len(out)} units ({bad} failed), columns {NAME}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
