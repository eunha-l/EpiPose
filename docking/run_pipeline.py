#!/usr/bin/env python
"""Docking and feature extraction pipeline.

Running EpiPose on a new antigen-antibody pair takes four steps. Each step
writes into $EPIPOSE_WORK and the next one reads from there.

  1 tensor    complex cif to atom tensors
  2 dock      FFT docking, keeps the top poses by docking score
  3 pose      one feature vector per pose      -> data/features/pose_rep.pkl
  4 residue   antigen residue properties       -> data/features/residue.pkl

The auxiliary target used during training is built separately, since it needs
the bound structure.

  python docking/run_pipeline.py tensor   --units units.txt
  python docking/run_pipeline.py dock     --units units.txt --gpu 0
  python docking/run_pipeline.py pose     --gpu 0 --out data/features/pose_rep.pkl
  python docking/run_pipeline.py residue  --out data/features/residue.pkl
  python docking/run_pipeline.py delta    --out data/features/delta_target.pkl

The files under _vendor/ are the original research code with the paths moved
into environment variables (_vendor/paths.py).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENDOR = HERE / "_vendor"
sys.path.insert(0, str(VENDOR))
import paths  # noqa: E402


def _run(script: str, args: list[str], env: dict | None = None) -> int:
    cmd = [sys.executable, str(VENDOR / script), *args]
    e = dict(os.environ)
    e.setdefault("EPIPOSE_ROOT", str(paths.ROOT))
    e.setdefault("EPIPOSE_WORK", str(paths.WORK))
    if env:
        e.update(env)
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, env=e, cwd=str(paths.ROOT))


def cmd_tensor(a) -> int:
    """Complex cif to atom tensors. Needs cif files in EPIPOSE_RAW."""
    paths.require(paths.RAW, "complex cif directory (EPIPOSE_RAW)")
    paths.ensure_work()
    return _run("prep_core.py", [str(a.nproc)])


def cmd_dock(a) -> int:
    """FFT docking over the rotation set, keeping the best scoring poses."""
    paths.ensure_work()
    env = dict(LIST=a.units, TAG=a.tag, SPLIT="0",
               W_CLASH=str(a.w_clash), KEEP=str(a.keep),
               WEIGHTS="dock_weights_raw.npz")
    if a.frame:
        env["FRAME"] = a.frame
    return _run("dock_runner.py",
                [str(a.gpu), str(a.shard), str(a.n_shard), a.mode], env)


POSE_DROP = ("n_res", "contact_rg")


def cmd_pose(a) -> int:
    """One feature vector per pose, written to a pickle."""
    env = {"RESTAG": a.tag}
    if a.c_from_res:
        env["C_FROM_RES"] = "1"
    if a.shard:
        env["SHARD"] = a.shard
    rc = _run("pose_feature_core.py", [str(a.gpu), a.out, a.split], env)
    if rc == 0:
        _drop_pose_columns(Path(paths.ROOT) / a.out if not
                           Path(a.out).is_absolute() else Path(a.out))
    return rc


def _drop_pose_columns(path: Path) -> None:
    import pickle

    import numpy as np
    with open(path, "rb") as f:
        d = pickle.load(f)
    name = list(d["name"])
    keep = [i for i, nm in enumerate(name) if nm not in POSE_DROP]
    if len(keep) == len(name):
        return
    d["name"] = [name[i] for i in keep]
    for r in d["feat"].values():
        r["F"] = np.ascontiguousarray(np.asarray(r["F"])[:, keep])
    with open(path, "wb") as f:
        pickle.dump(d, f, protocol=4)
    print(f"kept {len(keep)} of {len(name)} pose features", flush=True)


def _run_local(script: str, args: list[str], env: dict | None = None) -> int:
    cmd = [sys.executable, str(HERE / script), *args]
    e = dict(os.environ)
    e.setdefault("EPIPOSE_ROOT", str(paths.ROOT))
    e.setdefault("EPIPOSE_WORK", str(paths.WORK))
    if env:
        e.update(env)
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, env=e, cwd=str(paths.ROOT))


def cmd_residue(a) -> int:
    """Antigen residue properties."""
    env = {"SHARD": a.shard} if a.shard else None
    return _run_local("residue_feature.py", ["--out", a.out, "--tag", a.apo_tag], env)


def cmd_delta(a) -> int:
    """Auxiliary target from the bound/unbound structural change."""
    env = {"SHARD": a.shard} if a.shard else None
    return _run_local("delta_target.py", ["--out", a.out, "--split", a.split], env)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("tensor", help="cif to atom tensors")
    p.add_argument("--nproc", type=int, default=16)
    p.set_defaults(fn=cmd_tensor)

    p = sub.add_parser("dock", help="FFT docking")
    p.add_argument("--units", required=True, help="file listing unit names, relative to WORK")
    p.add_argument("--tag", default="res_k12")
    p.add_argument("--mode", default="md", choices=("run", "md", "apo", "unbound"))
    p.add_argument("--frame", default="mdf00")
    p.add_argument("--keep", type=int, default=500, help="number of poses to keep")
    p.add_argument("--w_clash", type=float, default=1.5)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n_shard", type=int, default=1)
    p.set_defaults(fn=cmd_dock)

    p = sub.add_parser("pose", help="pose features")
    p.add_argument("--out", default="data/features/pose_rep.pkl")
    p.add_argument("--tag", default="res_k12")
    p.add_argument("--split", default="", help="train, val, test, or empty for all")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--shard", default="", help='shard as "i/n"')
    p.add_argument("--c_from_res", action="store_true",
                   help="read the contact matrix from the docking result file")
    p.set_defaults(fn=cmd_pose)

    p = sub.add_parser("residue", help="residue properties")
    p.add_argument("--out", default="data/features/residue.pkl")
    p.add_argument("--apo_tag", default="mdf00")
    p.add_argument("--shard", default="")
    p.set_defaults(fn=cmd_residue)

    p = sub.add_parser("delta", help="auxiliary training target")
    p.add_argument("--out", default="data/features/delta_target.pkl")
    p.add_argument("--split", default="train")
    p.add_argument("--shard", default="")
    p.set_defaults(fn=cmd_delta)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
