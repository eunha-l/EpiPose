#!/usr/bin/env python
"""Builds the auxiliary training target from the bound/unbound change.

    delta[i] = (bound value - unbound value) / unbound standard deviation

for every antigen residue i, over four surface descriptors: relative solvent
accessibility, protrusion, and neighbour density within 6 and 10 A. The two
density columns are negated so that all four point the same way.

Accessibility is measured on the antigen alone. Including the antibody would
bury the interface and hand the epitope labels to the model.

The bound coordinates come from the complex cif in EPIPOSE_RAW and the unbound
ones from the apo tensor. Only training units are built, since using the bound
structure of a validation or test unit would leak the answer.

Usage:
  python docking/delta_target.py --out data/features/delta_target.pkl
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "_vendor"))
import paths  # noqa: E402
from _sasa_fast import sasa  # noqa: E402
from dock_fft_prep_apo import chain_residues  # noqa: E402

NAME = ["d_rsa", "d_protrusion", "d_dens6", "d_dens10"]
SIGN = np.array([1.0, 1.0, -1.0, -1.0], np.float32)
MAXASA = {"ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167,
          "GLN": 225, "GLU": 223, "GLY": 104, "HIS": 224, "ILE": 197,
          "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159,
          "SER": 155, "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174}
RADIUS = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "SE": 1.90, "P": 1.80}


def residue_names(tensor_file: Path, n_res: int) -> list[str]:
    """Residue names from a tensor, used as the accessibility denominator."""
    d = np.load(tensor_file, allow_pickle=True)
    if "ag_meta" not in d.files:
        return [""] * n_res
    rid = np.asarray(d["ag_res"], np.int64)
    meta = np.asarray(d["ag_meta"], object)
    first = np.full(n_res, -1, np.int64)
    for a in range(len(rid) - 1, -1, -1):
        first[rid[a]] = a
    return ["" if i < 0 else str(meta[i, 1]) for i in first]


def residue_axis(tensor_file: Path) -> list[tuple[int, str]]:
    """The (residue number, insertion code) axis the model works on."""
    d = np.load(tensor_file, allow_pickle=True)
    seq = np.asarray(d["res_resseq"]).tolist()
    ic = np.asarray(d["res_icode"]).tolist()
    return [(int(a), str(b)) for a, b in zip(seq, ic)]


def from_cif(gid: str, chain: str, keys: list[tuple[int, str]]):
    """Reads the antigen chain of a complex cif onto the given residue axis."""
    if paths.RAW is None:
        raise RuntimeError("EPIPOSE_RAW must point at the complex cif files")
    hr = chain_residues(str(paths.RAW / f"{gid}.cif"), chain).get(chain, [])
    pos = dict(hr)
    xyz, rid, elem = [], [], []
    rname = [""] * len(keys)
    for i, k in enumerate(keys):
        v = pos.get(k)
        if v is None:
            continue
        rname[i] = str(v["comp"])
        for at in v["atoms"]:          # (atom name, element, x, y, z)
            xyz.append((at[2], at[3], at[4]))
            rid.append(i)
            elem.append(str(at[1]))
    if not xyz:
        return None
    return (np.asarray(xyz, np.float64), np.asarray(rid, np.int64),
            np.array([RADIUS.get(e, 1.70) for e in elem], np.float64), rname)


def geometry_xyz(xyz: np.ndarray, rid: np.ndarray, rad: np.ndarray,
                 n_res: int, rname: list[str]) -> np.ndarray:
    """Four surface descriptors per residue, on the antigen alone."""
    cnt = np.bincount(rid, minlength=n_res).astype(np.float64)
    out = np.zeros((n_res, 4), np.float64)

    area = np.bincount(rid, weights=sasa(xyz, rad), minlength=n_res)
    maxa = np.array([MAXASA.get(r, 200) for r in rname], np.float64)
    out[:, 0] = area / np.maximum(maxa, 1)

    centre = np.zeros((n_res, 3))
    for k in range(3):
        centre[:, k] = (np.bincount(rid, weights=xyz[:, k], minlength=n_res)
                        / np.maximum(cnt, 1))
    out[:, 1] = np.linalg.norm(centre - xyz.mean(0), axis=1)

    d = np.linalg.norm(centre[:, None, :] - xyz[None, :, :], axis=2)
    same = rid[None, :] == np.arange(n_res)[:, None]
    out[:, 2] = ((d < 6.0) & ~same).sum(1)
    out[:, 3] = ((d < 10.0) & ~same).sum(1)
    return out


def geometry(tensor_file: Path, n_res: int, rname: list[str]) -> np.ndarray:
    """The same descriptors, from a tensor."""
    d = np.load(tensor_file, allow_pickle=True)
    xyz = np.asarray(d["ag"], np.float64)
    rid = np.asarray(d["ag_res"], np.int64)
    rad = (np.asarray(d["ag_rad"], np.float64) if "ag_rad" in d.files
           else np.full(len(xyz), 1.70))
    return geometry_xyz(xyz, rid, rad, n_res, rname)


def apo_file(gid: str) -> Path | None:
    """The representative apo tensor, which also carries the residue axis."""
    for f in sorted(paths.DOCK_OUT.glob(f"{gid}__*.apoN.npz")):
        if len(f.name[:-len(".apoN.npz")].split("__")) == 2:
            return f
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/features/delta_target.pkl")
    ap.add_argument("--split", default="train",
                    help="split to build, train by default")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    groups = pd.read_csv(paths.require(paths.GROUPS_CSV, "epipose_split.csv"))
    groups = groups[groups.use == True]          # noqa: E712
    gids = sorted(g for g, s in zip(groups.gid, groups.split) if s == a.split)
    if a.limit:
        gids = gids[:a.limit]
    shard = os.environ.get("SHARD")
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        gids = gids[i::n]

    out, missing, bad, t0 = {}, 0, 0, time.time()
    for i, gid in enumerate(gids, 1):
        apo_f = apo_file(gid)
        if apo_f is None:
            missing += 1
            continue
        n_res = int(np.load(apo_f, allow_pickle=True)["is_core"].shape[0])
        chain = apo_f.name[:-len(".apoN.npz")].split("__")[1]
        try:
            cif = from_cif(gid, chain, residue_axis(apo_f))
            if cif is None:
                missing += 1
                continue
            hxyz, hrid, hrad, rname = cif
            h = geometry_xyz(hxyz, hrid, hrad, n_res, rname)
            p = geometry(apo_f, n_res, residue_names(apo_f, n_res))
        except Exception as e:                                   # noqa: BLE001
            print(f"  failed on {gid}: {type(e).__name__} {e}", flush=True)
            bad += 1
            continue
        d = ((h - p) / (p.std(0) + 1e-9)).astype(np.float32) * SIGN
        out[gid] = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        if i % 100 == 0:
            print(f"  {i}/{len(gids)}  {(time.time() - t0) / 60:.1f} min  "
                  f"{missing} without a tensor, {bad} failed", flush=True)

    dest = Path(a.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        pickle.dump({"name": NAME, "feat": out}, f, protocol=4)
    print(f"wrote {dest} for {len(out)} {a.split} units "
          f"({missing} without a tensor, {bad} failed)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
