"""Pose assembly and docking score terms.

Places each stored pose, collects the antigen residues it contacts, and
recomputes the docking score terms that make up the pose features.
"""
import os as _os, sys as _sys
_VD = _os.environ.get('EPIPOSE_VENDOR') or _os.path.dirname(
    _os.path.abspath(globals().get('__file__', 'paths.py')))
_sys.path.insert(0, _VD)
from paths import (ROOT as _ROOT, ASSETS as _ASSETS, WORK as _WORK,
                   DOCK_OUT as _DOCK_OUT, GROUPS_CSV as _GROUPS,
                   DOCK2CANON as _D2C, RES_FEAT as _RESFEAT,
                   ZEPII as _ZEPII, RAW as _RAW, APO as _APO,
                   APOC as _APOC,
                   CONSASA as _CONSASA)

import os, sys, glob, time, argparse, warnings
warnings.filterwarnings("ignore")
HERE = str(_ROOT)
os.environ.setdefault("WEIGHTS", "dock_weights_raw.npz")
sys.path.insert(0, _VD)
sys.path.insert(0, HERE); os.chdir(HERE)
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
import dock_fft as D
from sklearn.metrics import roc_auc_score, average_precision_score

DK = str(_DOCK_OUT)
CS = str(_CONSASA)
_RFP = os.environ.get("RES_FEAT", f"{HERE}/rerank/res_feat.npz")
if not os.path.isabs(_RFP):
    _RFP = f"{HERE}/{_RFP}"
RF = dict(np.load(_RFP, allow_pickle=True))
print(f"[res_feat] {_RFP}", flush=True)
_MPF = f"{HERE}/rerank/model_p.npz"
if os.path.exists(_MPF):
    RF.update(dict(np.load(_MPF, allow_pickle=True)))
W = dict(contact=0.5, pot=3.0, elec=-300.0, hard=1.5, soft=1.5)
NCH_IN, NCH_OUT = 16, 7

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mp", type=int, default=0)
ap.add_argument("--feat", default="full", choices=["full", "man", "new5"])
ap.add_argument("--out_mode", default="field", choices=["field", "bonus", "mul"])
ap.add_argument("--direct", type=int, default=0)
ap.add_argument("--dock_feat", type=int, default=0)
ap.add_argument("--keep_q", type=int, default=0)
ap.add_argument("--seen", type=int, default=1)
A = ap.parse_args()
torch.manual_seed(A.seed); np.random.seed(A.seed)
dev = torch.device(f"cuda:{A.gpu}")
SEEN_MASK = bool(A.seen)
BONUS = A.out_mode == "bonus"
MUL = A.out_mode == "mul"
DIRECT = bool(A.direct)
DOCKF = bool(A.dock_feat)
if DOCKF:
    NCH_IN += 2
MP = bool(A.mp)
if MP:
    NCH_IN = 17
if A.feat == "man":
    NCH_IN = 7
elif A.feat == "new5":
    NCH_IN = 5                       # rsa, cons_kl, cons_id, has_cons, prop_r
if A.out_mode == "bonus":
    NCH_OUT = 1


RESTAG = os.environ.get("RESTAG", "res_mdf00")
C_FROM_RES = os.environ.get("C_FROM_RES", "0") == "1"


def rep_files(gid):
    for f in sorted(glob.glob(f"{DK}/{gid}__*.mdf00.npz")):
        b = os.path.basename(f)
        if ".res_" in b: continue
        if len(b[:-len(".mdf00.npz")].split("__")) == 2:
            return f, f[:-len(".mdf00.npz")] + f".{RESTAG}.npz"
    return None, None


def gauss_density(P, n, dev, vals=None, sigma=1.2, cut=3.0):
    """Splats atoms onto the grid as gaussians, optionally weighted."""
    off, dist = D.stencil(cut, dev)
    w = torch.exp(-dist ** 2 / (2 * sigma ** 2))
    out = torch.zeros(n * n * n, device=dev)
    gi = torch.round(P / D.H).long() % n
    v = torch.ones(len(gi), device=dev) if vals is None else vals
    for c in torch.split(torch.arange(len(gi), device=dev), 2048):
        p = (gi[c][:, None, :] + off[None]) % n
        flat = ((p[..., 0] * n + p[..., 1]) * n + p[..., 2]).reshape(-1)
        out.scatter_add_(0, flat, (v[c][:, None] * w[None]).reshape(-1))
    return out.view(n, n, n)


def build(gid, dev, Raug=None, apo_k=None, shift=None):
    """Antigen grids, pose coordinates, contacts and labels for one unit.

    With apo_k set, the antigen side comes from that apo structure while the
    antibody coordinates and the labels stay with the representative one.
    """
    rtf, rrf = rep_files(gid)
    if rtf is None: return None
    if apo_k is None:
        tf, rf, cf = rtf, rrf, f"{CS}/{gid}__{APO[gid]}.npz"
    else:
        ch = os.path.basename(rtf)[:-len(".mdf00.npz")].split("__")[1]
        tf = f"{DK}/{gid}__{ch}__M{apo_k}.mdf00.npz"
        rf = f"{DK}/{gid}__{ch}__M{apo_k}.{RESTAG}.npz"
        cf = f"{_APOC}/{gid}__{ch}__M{apo_k}.npz"
        if not (os.path.exists(tf) and os.path.exists(rf) and os.path.exists(cf)):
            return None
    d = dict(np.load(tf, allow_pickle=True))
    d["ab"] = np.load(rtf, allow_pickle=True)["ab"]
    res = np.load(rf, allow_pickle=True)
    t = lambda k, ty=torch.float32: torch.tensor(d[k], device=dev, dtype=ty)
    ag, ab = t("ag"), t("ab")
    _c0 = ag.mean(0)
    ag_o = ag
    if Raug is not None:
        ag = (ag - _c0) @ Raug.T + _c0
    if shift is not None:
        ag = ag + shift; ag_o = ag_o + shift; ab = ab + shift
    acl = t("ag_cls", torch.long); abb = t("ag_bb", torch.long).bool()
    E = torch.tensor(np.load(_ASSETS / os.environ.get("WEIGHTS", "dock_weights_raw.npz"))["E"], device=dev,
                     dtype=torch.float32)
    NCLS = int(E.shape[-1])
    ag_c, ab_c = ag.mean(0), ab.mean(0)
    need = (float((ag - ag_c).norm(dim=1).max())
            + float((ab - ab_c).norm(dim=1).max())) * 2 + 8.0
    n = next((s for s in D.SIZES if s >= need / D.H), D.SIZES[-1])
    origin = ag_c - (n / 2) * D.H
    P = ag - origin
    dmins = torch.stack([D.nearest_field(P[acl == c], n, D.SHELLS[-1][1], dev)
                         if int((acl == c).sum()) else
                         torch.full((n, n, n), D.SHELLS[-1][1] + 1.0, device=dev)
                         for c in range(NCLS)])
    dmin, ncls = dmins.min(dim=0)
    d_bb = D.nearest_field(P[abb], n, D.CLASH + 1.0, dev)
    d_sc = (D.nearest_field(P[~abb], n, D.CLASH + 1.0, dev) if bool((~abb).any())
            else torch.full((n, n, n), D.CLASH + 2.0, device=dev))
    core_bb = (d_bb < D.CLASH).float()
    core_sc = ((d_sc < D.CLASH) & (d_bb >= D.CLASH)).float()
    bands = [[(D.ramp(dmin, lo, D.SMOOTH, True) * D.ramp(dmin, hi, D.SMOOTH, False)
               * (ncls == c).float()) for c in range(NCLS)] for lo, hi in D.SHELLS]
    shell = sum(sum(b) for b in bands)
    R_pot = torch.stack([sum(E[s, c2, c] * bands[s][c2]
                             for s in range(len(D.SHELLS)) for c2 in range(NCLS))
                         for c in range(NCLS)])
    phi = D.coulomb(P, t("ag_q"), n, dev)
    Aman = torch.stack([shell, R_pot[0], R_pot[1], R_pot[2], core_bb, core_sc, phi])

    dens = torch.stack([gauss_density(P[acl == c], n, dev) if int((acl == c).sum())
                        else torch.zeros(n, n, n, device=dev) for c in range(3)])
    dbb = gauss_density(P[abb], n, dev)
    dsc = gauss_density(P[~abb], n, dev) if bool((~abb).any()) else torch.zeros_like(dbb)
    dq = gauss_density(P, n, dev, vals=t("ag_q"))
    nb10 = gauss_density(P, n, dev, sigma=4.0, cut=10.0)
    ag_res = t("ag_res", torch.long)
    sp = SPL.get(gid, "train")
    d2c = CIDX.get(gid)
    dens_all = gauss_density(P, n, dev).clamp(min=1e-6)
    def paint(name):
        """Paints residue scalars onto voxels, weighted by the atom gaussians."""
        key = f"{sp}|{gid}|{name}"
        if key not in RF or d2c is None: return torch.zeros(n, n, n, device=dev)
        v = np.nan_to_num(np.asarray(RF[key], np.float32), nan=0.0)
        nres_d = len(d2c)
        pr = np.zeros(nres_d, np.float32)
        m = (d2c >= 0) & (d2c < len(v))
        pr[m] = v[d2c[m]]
        ar = ag_res.clamp(0, nres_d - 1)
        av = torch.tensor(pr, device=dev)[ar]
        return gauss_density(P, n, dev, vals=av) / dens_all
    cons_kl, cons_id = paint("cons_kl"), paint("cons_id")
    _ck = f"{sp}|{gid}|cons_id"
    _cok = _ck in RF and bool(np.isfinite(np.asarray(RF[_ck], np.float64)).any())
    has_c = torch.full((n, n, n), 1.0 if _cok else 0.0, device=dev)
    prop_r = paint("prop_r")
    rsa = paint("rsa")
    Xch = [dens[0], dens[1], dens[2], dbb, dsc, dq,
           dmin.clamp(max=8.0), d_bb.clamp(max=8.0),
           (dmin < 2.0).float(), nb10, rsa, shell,
           cons_kl, cons_id, has_c, prop_r]
    if MP:
        Xch.append(paint("model_p"))
    X = torch.stack(Xch)
    mask = (dmin < 8.0).float()
    PAD = 10
    lo = (P.min(0).values / D.H).floor().long().clamp(min=0) - PAD
    hi = (P.max(0).values / D.H).ceil().long() + PAD + 1
    lo = lo.clamp(min=0); hi = torch.minimum(hi, torch.tensor([n, n, n], device=dev))
    sl = tuple(slice(int(a_), int(b_)) for a_, b_ in zip(lo.tolist(), hi.tolist()))
    Xc = X[(slice(None),) + sl].contiguous()
    if A.feat == "man":
        Xc = (mask[None] * Aman)[(slice(None),) + sl].contiguous()
    elif A.feat == "new5":
        Xc = X[[10, 12, 13, 14, 15]][(slice(None),) + sl].contiguous()

    rot = torch.tensor(np.asarray(res["pose_rot"]), device=dev).long()
    tr = torch.tensor(np.asarray(res["pose_trans"]), device=dev).long()
    ti = torch.stack([tr // (n * n), (tr // n) % n, tr % n], -1).float() * D.H
    v = ab - ab_c
    Q = []
    for c in torch.split(torch.arange(len(rot), device=dev), 50):
        q = torch.einsum("bij,aj->bai", ROT[rot[c]], v) + ti[c][:, None, :] + origin
        q = q - torch.round((q.mean(1, keepdim=True) - ag_o.mean(0)) / (n * D.H)) * (n * D.H)
        if Raug is not None:
            q = torch.einsum("ij,baj->bai", Raug, q - _c0) + _c0
        Q.append(q)
    Q = torch.cat(Q)

    if not os.path.exists(cf): return None
    z = np.load(cf, allow_pickle=True)
    if "pose_c" not in z.files: return None
    nres = int(z["n_res"])
    if C_FROM_RES:
        if int(np.asarray(res["contacts"]).shape[1]) != nres: return None
        _agr = t("ag_res", torch.long)
        C = torch.zeros(len(Q), nres, dtype=torch.float32, device=dev)
        for _c in torch.split(torch.arange(len(Q), device=dev), 25):
            _h = (torch.cdist(Q[_c], ag_o) <= 4.5).any(1)      # (b, n_ag_atom)
            for _j in range(_h.shape[0]):
                C[_c[_j], torch.unique(_agr[_h[_j]])] = 1.0
    else:
        C = torch.tensor(np.unpackbits(z["pose_c"], axis=1, count=nres).astype(np.float32),
                         device=dev)
    y = torch.tensor(np.asarray(z["is_core"], np.float32), device=dev)
    if y.sum() < 3 or C.shape[0] != Q.shape[0]: return None
    vlab = vmsk = None
    if DIRECT:
        _lo, _hi = D.SHELLS[0][0], D.SHELLS[-1][1]
        _ea = (y[ag_res.clamp(0, len(y) - 1)] > 0.5)
        if not bool(_ea.any()): return None
        _de = D.nearest_field(P[_ea], n, _hi, dev)
        _vl = ((_de >= _lo) & (_de < _hi)).float()
        _vm = ((dmin >= _lo) & (dmin < _hi)).float()
        vlab = _vl[sl].contiguous(); vmsk = _vm[sl].contiguous()
        if float(vmsk.sum()) < 100 or float((vlab * vmsk).sum()) < 10: return None
    cdr = t("ab_cdr", torch.long).bool(); bb = t("ab_bb", torch.long).bool()
    bcl = t("ab_cls", torch.long); abq = t("ab_q")
    Amc = (mask[None] * Aman)[(slice(None),) + sl].contiguous()
    shp = tuple(x.stop - x.start for x in sl)
    NV = shp[0] * shp[1] * shp[2]
    def flat(Qs):
        """Flat index in the crop frame; atoms outside it get a sentinel."""
        i = torch.round((Qs - origin) / D.H).long() % n
        rel = torch.stack([i[..., k] - sl[k].start for k in range(3)], -1)
        ok = ((rel >= 0) & (rel < torch.tensor(shp, device=dev))).all(-1)
        f = (rel[..., 0] * shp[1] + rel[..., 1]) * shp[2] + rel[..., 2]
        return torch.where(ok, f, torch.full_like(f, NV)).int()
    IDX = dict(cdr=flat(Q[:, cdr]), all=flat(Q), bb=flat(Q[:, bb]), sc=flat(Q[:, ~bb]))
    IDX["pot"] = [flat(Q[:, cdr][:, bcl[cdr] == c]) if bool((bcl[cdr] == c).any())
                  else None for c in range(NCLS)]
    seen = torch.zeros(NV + 1, dtype=torch.bool, device=dev)
    _keys = ("cdr",) if BONUS else ("cdr", "all", "bb", "sc")
    for _k in _keys:
        seen[IDX[_k].long().reshape(-1)] = True
    if not BONUS:
        for _c in range(NCLS):
            if IDX["pot"][_c] is not None:
                seen[IDX["pot"][_c].long().reshape(-1)] = True
    seen = seen[:NV].view(*shp).float()
    if DOCKF:
        _n = float(C.shape[0])
        _pd = torch.zeros(NV + 1, device=dev)
        _fl = IDX["cdr"].long().reshape(-1)
        _pd.scatter_add_(0, _fl, torch.ones_like(_fl, dtype=torch.float32))
        _pose = torch.log1p(_pd[:NV].view(*shp) / _n)
        _dm = C.mean(0)
        _av = _dm[ag_res.clamp(0, len(_dm) - 1)]
        _dmv = (gauss_density(P, n, dev, vals=_av) / dens_all)[sl]
        Xc = torch.cat([Xc, _pose[None], _dmv[None].contiguous()], 0)
    u = dict(gid=gid, X=Xc, Aman=Amc, shp=shp, NV=NV, IDX=IDX, C=C, y=y,
             seen=seen, abq=abq, NCLS=NCLS, vlab=vlab, vmsk=vmsk, apo_k=apo_k)
    if A.keep_q:
        u.update(Q=Q, origin=origin, lo=lo, ag=ag, ag_res=ag_res,
                 ab_cdr=cdr, ab_bb=bb, ab_cls=bcl, n_grid=n)
    with torch.no_grad():
        u["s0"] = pose_scores(u, Amc)
    return u



def gat(fld, idx):
    """Gathers field values by flat index, with a trailing zero for the sentinel."""
    f = torch.cat([fld.reshape(-1), fld.new_zeros(1)])
    return f[idx.long()]


def pose_scores(u, Afield):
    """Pose scores from the field channels."""
    I = u["IDX"]
    contact = gat(Afield[0], I["cdr"]).sum(1)
    pot = 0
    for c in range(u["NCLS"]):
        if I["pot"][c] is not None:
            pot = pot + gat(Afield[1 + c], I["pot"][c]).sum(1)
    elec = (gat(Afield[6], I["all"]) * u["abq"][None]).sum(1)
    hardc = gat(Afield[4], I["all"]).sum(1) + gat(Afield[5], I["bb"]).sum(1)
    softc = gat(Afield[5], I["sc"]).sum(1)
    return (W["contact"] * contact + W["pot"] * pot + W["elec"] * elec
            - W["hard"] * hardc - W["soft"] * softc)


