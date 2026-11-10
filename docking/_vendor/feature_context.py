"""Shared context for pose feature extraction.

Builds the atom grids and residue metadata that pose_feature_core.py reads.
"""
import os as _os, sys as _sys
_VD = _os.environ.get('EPIPOSE_VENDOR') or _os.path.dirname(
    _os.path.abspath(globals().get('__file__', 'paths.py')))
_sys.path.insert(0, _VD)
from paths import (ROOT as _ROOT, ASSETS as _ASSETS, WORK as _WORK,
                   DOCK_OUT as _DOCK_OUT, GROUPS_CSV as _GROUPS,
                   DOCK2CANON as _D2C, RES_FEAT as _RESFEAT,
                   ZEPII as _ZEPII, RAW as _RAW, APO as _APO,
                   APOC as _APOC)

import argparse
import glob
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
HERE = str(_ROOT)
os.environ.setdefault("WEIGHTS", "dock_weights_raw.npz")
sys.path.insert(0, _VD)
sys.path.insert(0, HERE); os.chdir(HERE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import dock_fft_gpu2 as D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ag_ch", default="man", choices=["man", "full", "new5"])
ap.add_argument("--inter", type=int, default=0)
A = ap.parse_args()
CKDIR = os.environ.get("DOVE_OUT", str(_WORK))
os.makedirs(CKDIR, exist_ok=True)
torch.manual_seed(A.seed); np.random.seed(A.seed)
dev = torch.device(f"cuda:{A.gpu}")

sys.argv = ["pose_builder.py", "--gpu", str(A.gpu), "--seen", "1",
            "--keep_q", "1"]
G_ = {}
exec(compile(open(_os.path.join(_VD, "pose_builder.py")).read()
             .split('\nif __name__ == "__main__":')[0], "pose_builder.py", "exec"), G_)
G = pd.read_csv(str(_GROUPS)); G = G[G.use == True]
G_["SPL"] = dict(zip(G.gid, G.split)); G_["APO"] = dict(zip(G.gid, G.apo_code))
G_["CIDX"] = {k: v for k, v in np.load(str(_D2C)).items()}
G_["ROT"] = D.rot_matrices(D.quaternions(D.NROT)).to(dev)
SC = dict(zip(G.gid, G.scored.astype(bool)))
AG_CH = {"man": 7, "new5": 5, "full": 16}[A.ag_ch]
NEW5X = [10, 12, 13, 14, 15]
RBF_MU = torch.tensor([2.8, 3.2, 3.6, 4.0, 4.4, 5.0, 5.6])
RBF_SG = 0.4
INTER_CH = (len(RBF_MU) + 2) if A.inter else 0
SIDE_CH = 5 if (A.inter and A.ag_ch == "man") else 0
ZEP = torch.tensor(np.load(str(_ZEPII)), dtype=torch.float32)
AA20 = "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
AIX20 = {a: i for i, a in enumerate(AA20)}
NEW5 = [10, 12, 13, 14, 15]
NCH_ALL = AG_CH + 6 + INTER_CH + SIDE_CH
CHSEL = None
NCH = NCH_ALL
H = D.H


_META = {}


def res_meta(u):
    """Atom to (residue index, amino acid index) for antigen and antibody.

    The antigen atom count differs between apo structures, so the result is
    cached per (unit, apo index). Antibody metadata always comes from the
    representative structure.
    """
    gid = u["gid"]; ak = u.get("apo_k")
    key = (gid, ak)
    if key in _META:
        return _META[key]
    rtf = G_["rep_files"](gid)[0]
    if ak is None:
        agf = rtf
    else:
        ch = os.path.basename(rtf)[:-len(".mdf00.npz")].split("__")[1]
        agf = f"{_DOCK_OUT}/{gid}__{ch}__M{ak}.mdf00.npz"
    za = np.load(agf, allow_pickle=True)
    zr = np.load(rtf, allow_pickle=True)
    d = {"ab_meta": zr["ab_meta"]}
    if "ag_meta" in za.files:
        d["ag_meta"] = za["ag_meta"]
    else:
        d["ag_meta"] = None
        d["_ag_res"] = za["ag_res"]; d["_rep_meta"] = zr["ag_meta"]

    def f(m):
        key = [f"{r[2]}|{r[3]}|{r[4]}" for r in m]
        u_ = {}; ri = np.empty(len(key), np.int64)
        for i, k in enumerate(key):
            if k not in u_: u_[k] = len(u_)
            ri[i] = u_[k]
        aa = np.full(len(u_), -1, np.int64)
        for i, k in enumerate(key): aa[ri[i]] = AIX20.get(str(m[i][1]), -1)
        return (torch.tensor(ri, device=dev), torch.tensor(aa, device=dev))
    if d["ag_meta"] is not None:
        agp = f(d["ag_meta"])
    else:
        ri = torch.tensor(d["_ag_res"].astype(np.int64), device=dev)
        aa = f(d["_rep_meta"])[1]
        nr = int(ri.max()) + 1
        if len(aa) < nr:
            aa = torch.cat([aa, torch.full((nr - len(aa),), -1,
                                           dtype=aa.dtype, device=dev)])
        agp = (ri, aa[:nr])
    _META[key] = (agp, f(d["ab_meta"]))
    return _META[key]


