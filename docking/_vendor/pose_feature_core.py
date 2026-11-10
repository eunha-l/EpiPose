"""Pose feature extraction.

Produces one feature vector per docking pose, summarising the interface the
pose forms: the antigen residues it covers, the antibody atoms facing them,
and the docking score terms.

Interfaces are defined symmetrically at 4.5 A between heavy atoms.

Usage:
  python docking/run_pipeline.py pose --gpu 0 --out data/features/pose_rep.pkl
"""
import os as _os, sys as _sys
# Under exec there is no __file__, so the vendor directory comes from an
# environment variable that run_pipeline.py sets.
_VD = _os.environ.get('EPIPOSE_VENDOR') or _os.path.dirname(
    _os.path.abspath(globals().get('__file__', 'paths.py')))
_sys.path.insert(0, _VD)
from paths import (ROOT as _ROOT, ASSETS as _ASSETS, WORK as _WORK,
                   DOCK_OUT as _DOCK_OUT, GROUPS_CSV as _GROUPS,
                   DOCK2CANON as _D2C, RES_FEAT as _RESFEAT,
                   ZEPII as _ZEPII, RAW as _RAW, APO as _APO,
                   APOC as _APOC)

import glob
import os
import pickle
import sys
import time

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

HERE = str(_ROOT)
GPU = int(sys.argv[1])
OUT = sys.argv[2]
SEL = sys.argv[3] if len(sys.argv) > 3 else ""

os.environ["RES_FEAT"] = str(_RESFEAT)
sys.argv = ["feature_context.py", "--ag_ch", "man", "--inter", "1",
            "--gpu", str(GPU)]
GD = {"__name__": "_lib", "__file__": _os.path.join(_VD, "feature_context.py")}
exec(compile(open(_os.path.join(_VD, "feature_context.py")).read(), "feature_context.py", "exec"), GD)

dev = GD["dev"]
G_ = GD["G_"]
build, res_meta = G_["build"], GD["res_meta"]
RF, CIDX, SPL = G_["RF"], G_["CIDX"], G_["SPL"]
ZEP = GD["ZEP"].to(dev)
SC = GD["SC"]
G = GD["G"]

NAME = ["n_res", "prop_sum", "prop_mean", "kl_sum", "kl_mean", "id_sum", "id_mean",
        "rsa_sum", "rsa_mean", "zep_sum", "s0",
        "n_ab_atom", "cdr_frac", "h3_n", "cls0_frac", "cls1_frac", "cls2_frac",
        "chg_compl", "hyd_frac",
        # The docking score split into its terms, so the model can weight them
        # for this task instead of reusing the docking weights.
        "t_contact", "t_pot0", "t_pot1", "t_pot2", "t_elec", "t_hardc", "t_softc",
        # Contact split over the six CDR loops, as a fraction of the interface
        # antibody atoms.
        "h1_frac", "h2_frac", "h3_frac", "l1_frac", "l2_frac", "l3_frac",
        # Shape of the contacted patch: residues within 8 A of each other are
        # neighbours, and the components come from the induced subgraph.
        "contact_rg", "contact_components", "largest_component_frac"]


def cdr_of(n):
    """IMGT CDR ranges, as in dock_fft_prep.py."""
    return 1 if 27 <= n <= 38 else 2 if 56 <= n <= 65 else 3 if 105 <= n <= 117 else 0


def res_vec(gid, sp, name, nres_d):
    """Maps residue scalars onto the docking residue axis."""
    key = f"{sp}|{gid}|{name}"
    d2c = CIDX.get(gid)
    out = np.zeros(nres_d, np.float32)
    if key not in RF or d2c is None:
        return torch.tensor(out, device=dev)
    v = np.nan_to_num(np.asarray(RF[key], np.float32), nan=0.0)
    m = (d2c >= 0) & (d2c < len(v))
    nn = min(nres_d, len(d2c))
    mm = m[:nn]
    out[:nn][mm] = v[d2c[:nn][mm]]
    return torch.tensor(out, device=dev)


def unit_feat(gid, apo_k=None):
    u = build(gid, dev, apo_k=apo_k)
    if u is None:
        return None
    C = u["C"]                                   # (P, nres_d) 0/1
    P, nres_d = C.shape
    sp = SPL.get(gid, "train")

    # antigen side of the interface
    nres = C.sum(1)                              # contacted antigen residues
    den = nres.clamp(min=1)
    cols = {}
    for nm, key in (("prop", "prop_r"), ("kl", "cons_kl"),
                    ("id", "cons_id"), ("rsa", "rsa")):
        v = res_vec(gid, sp, key, nres_d)
        s = C @ v
        cols[f"{nm}_sum"] = s
        cols[f"{nm}_mean"] = s / den

    # Antibody metadata comes from the representative structure, and so do the
    # antigen charges and atom classes, so that the atom counts line up.
    rtf = G_["rep_files"](gid)[0]
    zr = np.load(rtf, allow_pickle=True)
    if apo_k is None:
        zag = zr
    else:
        _ch = os.path.basename(rtf)[:-len(".mdf00.npz")].split("__")[1]
        zag = np.load(f"{_DOCK_OUT}/"
                      f"{gid}__{_ch}__M{apo_k}.mdf00.npz", allow_pickle=True)
    abm = zr["ab_meta"]
    part = gid.split("_")
    hch = part[2] if len(part) > 3 else ""
    abnum = np.array([int(x) for x in abm[:, 3]], np.int64)
    abch = abm[:, 2].astype(str)
    h3 = torch.tensor((abch == hch) & (abnum >= 105) & (abnum <= 117), device=dev)
    # CDR labels: 1-3 on the heavy chain, 4-6 on the light chain, 0 elsewhere.
    _c6 = np.array([(cdr_of(n) + (0 if c == hch else 3)) if cdr_of(n) > 0 else 0
                    for n, c in zip(abnum, abch)], np.int64)
    CDR6 = torch.tensor(_c6, device=dev)
    cdr = u["ab_cdr"].bool()
    cls = u["ab_cls"].long()
    abq = u["abq"].float()
    ag = u["ag"].float()
    agq = torch.tensor(np.asarray(zag["ag_q"], np.float32), device=dev)
    agc = torch.tensor(np.asarray(zag["ag_cls"], np.int64), device=dev)
    if len(agq) != len(ag):                      # a different apo has other atoms
        return None
    (ag_ri, ag_aa), (ab_ri, ab_aa) = res_meta(u)
    NAG_R = int(ag_ri.max()) + 1

    Q = u["Q"].float()
    CDRN = ["h1_frac", "h2_frac", "h3_frac", "l1_frac", "l2_frac", "l3_frac"]
    acc = {k: [] for k in ("n_ab_atom", "cdr_frac", "h3_n", "cls0_frac",
                           "cls1_frac", "cls2_frac", "chg_compl", "hyd_frac",
                           "zep_sum", *CDRN)}
    CH = 25
    for i0 in range(0, P, CH):
        Qc = Q[i0:i0 + CH]
        d = torch.cdist(Qc, ag)                  # (B, nab, nag)
        h = d <= 4.5
        hit = h.any(2)                           # (B, nab) interface antibody atoms
        nab_c = hit.sum(1)
        dn = nab_c.clamp(min=1).float()
        acc["n_ab_atom"].append(nab_c.float())
        acc["cdr_frac"].append((hit & cdr).sum(1).float() / dn)
        acc["h3_n"].append((hit & h3).sum(1).float())
        for c in range(3):
            acc[f"cls{c}_frac"].append((hit & (cls == c)).sum(1).float() / dn)
        for _k in range(1, 7):
            acc[CDRN[_k - 1]].append((hit & (CDR6 == _k)).sum(1).float() / dn)
        # charge complementarity, negative when the pair is complementary
        acc["chg_compl"].append(
            (h.float() * (abq[None, :, None] * agq[None, None, :])).sum((1, 2)))
        # fraction of pairs that are hydrophobic on both sides
        npair = h.sum((1, 2)).clamp(min=1).float()
        both0 = (h & (cls == 0)[None, :, None] & (agc == 0)[None, None, :])
        acc["hyd_frac"].append(both0.sum((1, 2)).float() / npair)
        # coevolution score, summed once per residue pair
        bb, ii, jj = h.nonzero(as_tuple=True)
        zs = torch.zeros(len(Qc), device=dev)
        if len(bb):
            key = bb * 10 ** 7 + ab_ri[ii] * NAG_R + ag_ri[jj]
            uq, inv = torch.unique(key, return_inverse=True)
            first = torch.zeros(len(uq), dtype=torch.long, device=dev)
            first.scatter_(0, inv, torch.arange(len(inv), device=dev))
            ai, aj = ab_aa[ab_ri[ii[first]]], ag_aa[ag_ri[jj[first]]]
            vm = (ai >= 0) & (aj >= 0)
            if bool(vm.any()):
                zs.scatter_add_(0, bb[first][vm], ZEP[ai[vm], aj[vm]])
        acc["zep_sum"].append(zs)
    acc = {k: torch.cat(v) for k, v in acc.items()}

    cols["n_res"] = nres.float()
    cols["zep_sum"] = acc["zep_sum"]
    cols["s0"] = u["s0"].float()
    # docking score terms, before they are combined
    gat, I, Af = G_["gat"], u["IDX"], u["Aman"]
    cols["t_contact"] = gat(Af[0], I["cdr"]).sum(1).float()
    for c in range(3):
        cols[f"t_pot{c}"] = (gat(Af[1 + c], I["pot"][c]).sum(1).float()
                             if c < u["NCLS"] and I["pot"][c] is not None
                             else torch.zeros(P, device=dev))
    cols["t_elec"] = (gat(Af[6], I["all"]) * u["abq"][None]).sum(1).float()
    cols["t_hardc"] = (gat(Af[4], I["all"]).sum(1)
                       + gat(Af[5], I["bb"]).sum(1)).float()
    cols["t_softc"] = gat(Af[5], I["sc"]).sum(1).float()
    for k in ("n_ab_atom", "cdr_frac", "h3_n", "cls0_frac", "cls1_frac",
              "cls2_frac", "chg_compl", "hyd_frac", *CDRN):
        cols[k] = acc[k]
    # patch shape from the residue neighbour graph
    _ri = ag_ri.long().clamp(0, nres_d - 1)
    _cen = torch.zeros(nres_d, 3, device=dev)
    _cnt = torch.zeros(nres_d, device=dev)
    _cen.index_add_(0, _ri, ag)
    _cnt.index_add_(0, _ri, torch.ones(len(_ri), device=dev))
    _has = _cnt > 0
    _cen = _cen / _cnt.clamp(min=1)[:, None]
    _A = (torch.cdist(_cen, _cen) <= 8.0) & _has[:, None] & _has[None, :]
    _An = _A.cpu().numpy()
    _cn = _cen.cpu().numpy()
    _Cb = C.bool().cpu().numpy()
    _rg = np.zeros(P, np.float32)
    _nc = np.zeros(P, np.float32)
    _lf = np.zeros(P, np.float32)
    for _k in range(P):
        _ix = np.flatnonzero(_Cb[_k])
        if len(_ix) == 0:
            continue
        _pt = _cn[_ix]
        _rg[_k] = np.sqrt(((_pt - _pt.mean(0)) ** 2).sum(1).mean())
        _nn, _lab = connected_components(
            csr_matrix(_An[np.ix_(_ix, _ix)]), directed=False)
        _nc[_k] = _nn
        _lf[_k] = np.bincount(_lab).max() / len(_ix)
    cols["contact_rg"] = torch.tensor(_rg, device=dev)
    cols["contact_components"] = torch.tensor(_nc, device=dev)
    cols["largest_component_frac"] = torch.tensor(_lf, device=dev)
    F = torch.stack([cols[k] for k in NAME], 1).float().cpu().numpy()
    return dict(split=sp, F=F, apo_k=apo_k,
                y=u["y"].cpu().numpy().astype(np.int8),
                C=np.packbits(C.cpu().numpy().astype(bool), axis=1),
                nres_d=nres_d, score=bool(SC.get(gid, False)))


def apo_ks(g):
    """None for the representative structure, 1-4 for the further ones."""
    return [k for k in (1, 2, 3, 4)
            if glob.glob(f"{_APOC}/{g}__*__M{k}.npz")]


def main():
    gids = [g for g in G.gid if g in CIDX and SPL.get(g) in ("train", "val", "test")]
    jobs = None
    if SEL == "apo":
        # Further apo structures are used for training units only.
        jobs = [(g, k) for g in gids if SPL.get(g) == "train" for k in apo_ks(g)]
    elif SEL in ("train", "val", "test"):
        # the test split keeps the flagged units only
        gids = [g for g in gids if SPL.get(g) == SEL
                and (SEL != "test" or SC.get(g, False))]
    elif SEL.isdigit():
        gids = gids[:int(SEL)]
    if jobs is None:
        jobs = [(g, None) for g in gids]
    # SHARD="i/n" splits the work across several processes.
    _sh = os.environ.get("SHARD")
    if _sh:
        _i, _n = (int(x) for x in _sh.split("/"))
        jobs = jobs[_i::_n]
        print(f"  shard {_i}/{_n}: {len(jobs)} jobs", flush=True)
    print(f"{len(jobs)} jobs, {len(NAME)} features", flush=True)
    print("  " + " ".join(NAME), flush=True)
    res, t0, bad = {}, time.time(), 0
    for i, (g, ak) in enumerate(jobs, 1):
        try:
            r = unit_feat(g, ak)
        except Exception as e:                                   # noqa: BLE001
            print(f"  failed on {g} apo{ak}: {type(e).__name__} {e}", flush=True)
            bad += 1
            continue
        if r is None:
            bad += 1
            continue
        Cb = np.unpackbits(r["C"], axis=1, count=r["nres_d"]).astype(bool)
        yb = r["y"].astype(bool)
        inter = (Cb & yb).sum(1)
        # q is the fraction of contacted residues that are epitope residues
        r["q"] = inter / np.maximum(Cb.sum(1), 1)
        r["f1"] = 2 * inter / np.maximum(Cb.sum(1) + yb.sum(), 1)
        res[(g, ak) if SEL == "apo" else g] = r
        if i % 100 == 0:
            print(f"  {i}/{len(jobs)}  {(time.time()-t0)/60:.1f} min  {bad} failed",
                  flush=True)
    with open(OUT, "wb") as f:
        pickle.dump(dict(feat=res, name=NAME), f)
    print(f"done: {len(res)} ok, {bad} failed, {(time.time()-t0)/60:.1f} min -> {OUT}",
          flush=True)


main()
