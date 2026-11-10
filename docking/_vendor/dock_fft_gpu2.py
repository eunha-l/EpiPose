import os as _os, sys as _sys
# Under exec there is no __file__, so the vendor directory comes from an
# environment variable that run_pipeline.py sets.
_VD = _os.environ.get('EPIPOSE_VENDOR') or _os.path.dirname(
    _os.path.abspath(globals().get('__file__', 'paths.py')))
_sys.path.insert(0, _VD)
from paths import (ROOT as _ROOT, ASSETS as _ASSETS, WORK as _WORK,
                   DOCK_OUT as _DOCK_OUT, GROUPS_CSV as _GROUPS,
                   DOCK2CANON as _D2C, RES_FEAT as _RESFEAT,
                   ZEPII as _ZEPII, RAW as _RAW, APO as _APO)

#!/usr/bin/env python
"""FFT rigid-body docking of an antibody against an unbound antigen.

For every sampled rotation all translations are scored at once by FFT
cross-correlation. The score combines four terms:

  contact     CDR atoms in the antigen contact shell
  pairing     class-dependent pairwise potential over two distance shells
  charge      electrostatics with a distance-dependent dielectric
  overlap     steric clash penalty over all antibody atoms

The terms are kept separate down to the translation grids, so their weights can
be applied afterwards without repeating the transforms.

Output: <EPIPOSE_DOCK_OUT>/<unit>.res_<tag>.npz
"""
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

FFT = _DOCK_OUT
H = 1.0
# overlap radius, loosened for unbound antigens
CLASH = float(os.environ.get("R_CLASH", 2.6))
SHELLS = [(2.6, 3.6), (3.6, 6.0)]
# Width of the ramps that replace the hard distance-band boundaries. Zero, the
# default used throughout, keeps the hard bands.
SMOOTH = float(os.environ.get("SMOOTH", 0.0))


def ramp(x, edge, width, rising):
    """A smooth step at edge, or a hard one when width is zero."""
    if width <= 0:
        return (x >= edge).float() if rising else (x < edge).float()
    z = (x - edge) / width
    return torch.sigmoid(z) if rising else torch.sigmoid(-z)
NROT = 12000
# how many rotations are transformed at once. It sets the peak memory, which is
# this many copies of the grid, and nothing else: the total work is the same
# whatever the batch. Lower it to share a card with something that has already
# reserved most of the memory.
BATCH = int(os.environ.get("BATCH", 16))
# poses kept per unit, the pool the reranker scores
KEEP = int(os.environ.get("KEEP", 500))
SIZES = [96, 128, 144, 160, 192, 224, 256]
EPS_SLOPE = 4.0      # dielectric taken as 4r, so the potential falls as 1/r squared
ELEC_CUT, ELEC_MIN = 12.0, 2.0
# An unbound antigen carries side chains in different rotamers, so a hard overlap
# penalty charges the true pose far more than the decoys. The penalty is bounded.
GRID = json.loads(os.environ["GRID"]) if os.environ.get("GRID") else {
    "pot": [3.0], "elec": [-300.0], "contact": [0.5],
    "clash": [0.0, 0.25, 1.0, 3.0], "cap": [0.0]}
CLASH_CAP = float(os.environ.get("CLASH_CAP", 0.0))
SPLIT = os.environ.get("SPLIT", "1") != "0"   # separate backbone from side chain
REDUCED = os.environ.get("REDUCED", "0") != "0"   # backbone and CB only


def quaternions(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(n, 4, generator=g, dtype=torch.float64)
    return (q / q.norm(dim=1, keepdim=True)).float()


def rot_matrices(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], dim=1)


def stencil(radius, dev):
    r = int(np.ceil(radius / H))
    a = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    g = torch.stack(torch.meshgrid(a, a, a, indexing="ij"), -1) * H
    d = g.norm(dim=-1)
    keep = d <= radius
    return (torch.nonzero(keep) - r), d[keep]


def nearest_field(pts, n, radius, dev):
    """Distance from every cell to the nearest of these atoms, capped at radius."""
    off, dist = stencil(radius, dev)
    out = torch.full((n * n * n,), float(radius) + 1.0, device=dev)
    gi = torch.round(pts / H).long() % n
    for chunk in torch.split(torch.arange(len(gi), device=dev), 2048):
        p = (gi[chunk][:, None, :] + off[None]) % n
        flat = ((p[..., 0] * n + p[..., 1]) * n + p[..., 2]).reshape(-1)
        out.scatter_reduce_(0, flat, dist.expand(len(chunk), -1).reshape(-1),
                            reduce="amin")
    return out.view(n, n, n)


def coulomb(pts, q, n, dev):
    """The antigen's potential on the grid, with the dielectric rising as 4r."""
    off, dist = stencil(ELEC_CUT, dev)
    w = 1.0 / (EPS_SLOPE * torch.clamp(dist, min=ELEC_MIN) ** 2)
    out = torch.zeros(n * n * n, device=dev)
    keep = q.abs() > 1e-6
    gi = torch.round(pts[keep] / H).long() % n
    qq = q[keep]
    for chunk in torch.split(torch.arange(len(gi), device=dev), 512):
        p = (gi[chunk][:, None, :] + off[None]) % n
        flat = ((p[..., 0] * n + p[..., 1]) * n + p[..., 2]).reshape(-1)
        out.scatter_add_(0, flat, (qq[chunk][:, None] * w[None]).reshape(-1))
    return out.view(n, n, n)


def rasterize(pts, n, w=None):
    b = pts.shape[0]
    i = torch.round(pts / H).long() % n
    flat = ((i[..., 0] * n + i[..., 1]) * n + i[..., 2])
    out = torch.zeros(b, n * n * n, device=pts.device)
    val = torch.ones_like(flat, dtype=torch.float32) if w is None else w.expand(b, -1).float()
    out.scatter_add_(1, flat, val)
    return out.view(b, n, n, n)


def prepare(f, dev):
    d = np.load(f, allow_pickle=True)
    d = dict(d)
    if REDUCED and "ag_red" in d and "ab_red" in d:
        # side chains are what differ between an apo structure and a bound one, so
        # dropping them leaves a description that transfers, at the cost of detail
        ka = d["ag_red"].astype(bool)
        kb = d["ab_red"].astype(bool)
        for k in ("ag", "ag_res", "ag_rad", "ag_cls", "ag_q", "ag_bb"):
            if k in d:
                d[k] = d[k][ka]
        for k in ("ab", "ab_cdr", "ab_rad", "ab_cls", "ab_q", "ab_bb"):
            if k in d:
                d[k] = d[k][kb]
    t = lambda k, ty=torch.float32: torch.tensor(d[k], device=dev, dtype=ty)
    ag, ab = t("ag"), t("ab")
    acl, bcl = t("ag_cls", torch.long), t("ab_cls", torch.long)
    cdr = t("ab_cdr", torch.long).bool()
    # pairwise class potential, fitted on training interfaces
    E = torch.tensor(np.load(_ASSETS / os.environ.get("WEIGHTS",
                     "dock_weights_raw.npz"))["E"], device=dev, dtype=torch.float32)
    # the number of atom classes is whatever the potential was fitted with, so a
    # finer potential needs no edit here and a tensor with fewer classes still works
    NCLS = int(E.shape[-1])
    ag_c, ab_c = ag.mean(0), ab.mean(0)
    need = (float((ag - ag_c).norm(dim=1).max()) + float((ab - ab_c).norm(dim=1).max())) \
        * 2 + 8.0
    n = next((s for s in SIZES if s >= need / H), SIZES[-1])
    origin = ag_c - (n / 2) * H
    P = ag - origin

    dmins = torch.stack([nearest_field(P[acl == c], n, SHELLS[-1][1], dev)
                         if int((acl == c).sum()) else
                         torch.full((n, n, n), SHELLS[-1][1] + 1.0, device=dev)
                         for c in range(NCLS)])
    dmin, ncls = dmins.min(dim=0)
    core = ramp(dmin, CLASH, SMOOTH, False)
    abb = t("ag_bb", torch.long).bool() if "ag_bb" in d else None
    if abb is not None and bool(abb.any()):
        d_bb = nearest_field(P[abb], n, CLASH + 1.0, dev)
        d_sc = (nearest_field(P[~abb], n, CLASH + 1.0, dev) if bool((~abb).any())
                else torch.full((n, n, n), CLASH + 2.0, device=dev))
        core_bb = (d_bb < CLASH).float()
        core_sc = ((d_sc < CLASH) & (d_bb >= CLASH)).float()
    else:
        core_bb, core_sc = core, torch.zeros_like(core)
    bands = [[(ramp(dmin, lo, SMOOTH, True) * ramp(dmin, hi, SMOOTH, False)
               * (ncls == c).float()) for c in range(NCLS)] for lo, hi in SHELLS]
    shell_any = sum(sum(b) for b in bands)
    R_pot = [sum(E[s, c2, c] * bands[s][c2] for s in range(len(SHELLS))
                 for c2 in range(NCLS)) for c in range(NCLS)]
    phi = coulomb(P, t("ag_q"), n, dev)
    F = {"contact": torch.fft.rfftn(shell_any), "core": torch.fft.rfftn(core),
         "core_bb": torch.fft.rfftn(core_bb), "core_sc": torch.fft.rfftn(core_sc),
         "phi": torch.fft.rfftn(phi),
         "pot": [torch.fft.rfftn(R_pot[c]) for c in range(NCLS)]}
    bb = t("ab_bb", torch.long).bool() if "ab_bb" in d else torch.zeros(
        len(ab), dtype=torch.bool, device=dev)
    return d, F, n, origin, ab, ab_c, bcl, cdr, t("ab_q"), bb, NCLS


def components(v, n, F, bcl, cdr, q, dev, ncls=3):
    """The four translation grids for one batch of rotations."""
    L_cdr = rasterize(v, n)
    L_all = rasterize(v, n) if False else None
    out = {}
    fc = torch.fft.rfftn(L_cdr, dim=(1, 2, 3))
    out["contact"] = torch.fft.irfftn(F["contact"] * torch.conj(fc), s=(n,) * 3,
                                      dim=(1, 2, 3))
    acc = 0
    for c in range(ncls):
        m = bcl[cdr] == c
        if not bool(m.any()):
            continue
        acc = acc + F["pot"][c] * torch.conj(torch.fft.rfftn(rasterize(v[:, m], n),
                                                             dim=(1, 2, 3)))
    out["pot"] = torch.fft.irfftn(acc, s=(n,) * 3, dim=(1, 2, 3)) if \
        torch.is_tensor(acc) else torch.zeros_like(out["contact"])
    return out


def run(f, dev, weights, calibrate):
    d, F, n, origin, ab, ab_c, bcl, cdr, abq, abb, NCLS = prepare(f, dev)
    nres = int(d["is_core"].shape[0])
    ag = torch.tensor(d["ag"], device=dev)
    ag_res = torch.tensor(d["ag_res"], device=dev).long()
    v_all = ab - ab_c
    v_cdr = v_all[cdr]
    cls_cdr = bcl[cdr]
    R = rot_matrices(quaternions(NROT)).to(dev)
    names = (["contact", "pot", "elec", "hardc", "softc"] if SPLIT
             else ["contact", "pot", "elec", "clash"])
    sets = [dict(zip(GRID, c)) for c in itertools.product(*GRID.values())] if calibrate \
        else [weights]
    for w in sets:
        w.setdefault("cap", CLASH_CAP)
        if SPLIT:
            # the backbone cannot rotate out of the way and a side chain can, so
            # the two overlaps are worth separating; with the antigen unbound the
            # side-chain term is where the true pose is unfairly charged
            w.setdefault("hard", float(os.environ.get("W_HARD",
                                                      w.get("clash", 3.0))))
            w.setdefault("soft", float(os.environ.get("W_SOFT", 1.0)))
    # a bounded penalty needs the terms kept apart, so the fast single-weight path
    # is only taken when no bound is asked for
    single = (sets[0] if len(sets) == 1 and not SPLIT and not sets[0]["cap"]
              else None)
    caps = GRID.get("cap", [0.0])
    hards = GRID.get("hard", GRID.get("clash", [3.0]))
    softs = GRID.get("soft", [1.0])
    corners = [dict(pot=pp, elec=ee, contact=cc, cap=qq, hard=hh, soft=ss,
                    clash=hh)
               for pp in (min(GRID["pot"]), max(GRID["pot"]))
               for ee in (min(GRID["elec"]), max(GRID["elec"]))
               for cc in (min(GRID["contact"]), max(GRID["contact"]))
               for qq in (min(caps), max(caps))
               for hh in (min(hards), max(hards))
               for ss in (min(softs), max(softs))]
    best_s = torch.full((len(sets), NROT), -1e30, device=dev)
    best_t = torch.zeros(len(sets), NROT, dtype=torch.long, device=dev)

    def grids(Rb):
        Pc = torch.einsum("bij,aj->bai", Rb, v_cdr)
        Pa = torch.einsum("bij,aj->bai", Rb, v_all)
        f_cdr = torch.fft.rfftn(rasterize(Pc, n), dim=(1, 2, 3))
        f_all = torch.fft.rfftn(rasterize(Pa, n), dim=(1, 2, 3))
        f_q = torch.fft.rfftn(rasterize(Pa, n, abq), dim=(1, 2, 3))
        if SPLIT:
            f_bb = torch.fft.rfftn(rasterize(Pa[:, abb], n), dim=(1, 2, 3)) \
                if bool(abb.any()) else None
            f_sc = torch.fft.rfftn(rasterize(Pa[:, ~abb], n), dim=(1, 2, 3)) \
                if bool((~abb).any()) else None
        acc = None
        for c in range(NCLS):
            m = cls_cdr == c
            if not bool(m.any()):
                continue
            term = F["pot"][c] * torch.conj(torch.fft.rfftn(rasterize(Pc[:, m], n),
                                                            dim=(1, 2, 3)))
            acc = term if acc is None else acc + term
        inv = lambda z: torch.fft.irfftn(z, s=(n,) * 3, dim=(1, 2, 3))
        if single is not None:
            z = (single["contact"] * F["contact"] * torch.conj(f_cdr)
                 + single["elec"] * F["phi"] * torch.conj(f_q)
                 - single["clash"] * F["core"] * torch.conj(f_all))
            if acc is not None:
                z = z + single["pot"] * acc
            return {"one": inv(z)}
        out = {"contact": inv(F["contact"] * torch.conj(f_cdr)),
               "elec": inv(F["phi"] * torch.conj(f_q))}
        out["pot"] = (inv(acc) if acc is not None
                      else torch.zeros_like(out["contact"]))
        if SPLIT:
            # any antibody atom inside the antigen backbone, plus antibody backbone
            # inside an antigen side chain: neither can be repacked away
            hz = F["core_bb"] * torch.conj(f_all)
            if f_bb is not None:
                hz = hz + F["core_sc"] * torch.conj(f_bb)
            out["hardc"] = inv(hz)
            out["softc"] = (inv(F["core_sc"] * torch.conj(f_sc))
                            if f_sc is not None else torch.zeros_like(out["contact"]))
        else:
            out["clash"] = inv(F["core"] * torch.conj(f_all))
        return out

    for s0 in range(0, NROT, BATCH):
        Rb = R[s0:s0 + BATCH]
        G = grids(Rb)
        b = Rb.shape[0]
        flat = {} if single is not None else {k: G[k].reshape(b, -1) for k in names}
        if single is not None:
            m, a = G["one"].reshape(b, -1).max(dim=1)
            best_s[0, s0:s0 + b] = m
            best_t[0, s0:s0 + b] = a
            continue
        def mix(w, t):
            base = (w["contact"] * t["contact"] + w["pot"] * t["pot"]
                    + w["elec"] * t["elec"])
            if SPLIT:
                soft = torch.clamp(t["softc"], max=w["cap"]) if w["cap"] else t["softc"]
                return base - w["hard"] * t["hardc"] - w["soft"] * soft
            cl = t["clash"] if not w["cap"] else torch.clamp(t["clash"], max=w["cap"])
            return base - w["clash"] * cl
        cand = []
        for wc in corners:
            cand.append(mix(wc, flat).topk(64, dim=1).indices)
        cand = torch.cat(cand, dim=1)
        vals = {k: torch.gather(flat[k], 1, cand) for k in names}
        for si, w in enumerate(sets):
            m, a = mix(w, vals).max(dim=1)
            best_s[si, s0:s0 + b] = m
            best_t[si, s0:s0 + b] = torch.gather(cand, 1, a[:, None])[:, 0]

    keep_single, single = single, None
    Gn = grids(torch.eye(3, device=dev)[None])
    single = keep_single
    tn = torch.round((ab_c - origin) / H).long() % n
    nat = {k: float(Gn[k][0, tn[0], tn[1], tn[2]]) for k in names}
    out = {"native_terms": np.array([nat[k] for k in names]),
           "is_core": d["is_core"], "res_resseq": d["res_resseq"],
           "res_icode": d["res_icode"]}
    if calibrate:
        def nmix(w):
            base = (w["contact"] * nat["contact"] + w["pot"] * nat["pot"]
                    + w["elec"] * nat["elec"])
            if SPLIT:
                sc = min(nat["softc"], w["cap"]) if w["cap"] else nat["softc"]
                return base - w["hard"] * nat["hardc"] - w["soft"] * sc
            cl = min(nat["clash"], w["cap"]) if w["cap"] else nat["clash"]
            return base - w["clash"] * cl
        nv = np.array([nmix(w) for w in sets])
        out["weights"] = np.array([[w[k] for k in GRID] for w in sets])
        out["native"] = nv
        out["native_rank"] = (best_s.cpu().numpy() >
                              nv[:, None]).sum(axis=1).astype(np.int32)
        out["best"] = best_s.max(dim=1).values.cpu().numpy()
        return out

    w = sets[0]
    if SPLIT:
        sc = min(nat["softc"], w["cap"]) if w["cap"] else nat["softc"]
        nv = (w["contact"] * nat["contact"] + w["pot"] * nat["pot"]
              + w["elec"] * nat["elec"] - w["hard"] * nat["hardc"] - w["soft"] * sc)
    else:
        nv = (w["contact"] * nat["contact"] + w["pot"] * nat["pot"]
              + w["elec"] * nat["elec"]
              - w["clash"] * (min(nat["clash"], w["cap"]) if w["cap"] else nat["clash"]))
    order = torch.argsort(best_s[0], descending=True)[:KEEP]
    contacts = torch.zeros(len(order), nres, dtype=torch.bool, device=dev)
    centres = torch.zeros(len(order), 3, device=dev)
    done = 0
    for c in torch.split(order, 25):
        Pc = torch.einsum("bij,aj->bai", R[c], v_cdr)
        t = best_t[0, c]
        ti = torch.stack([t // (n * n), (t // n) % n, t % n], -1).float() * H
        Q = Pc + ti[:, None, :] + origin
        Q = Q - torch.round((Q.mean(1, keepdim=True) - ag.mean(0)) / (n * H)) * (n * H)
        hit = (torch.cdist(Q, ag[None].expand(Q.shape[0], -1, -1)) <= 4.5).any(dim=1)
        for k in range(hit.shape[0]):
            contacts[done + k, torch.unique(ag_res[hit[k]])] = True
        centres[done:done + hit.shape[0]] = Q.mean(1)
        done += hit.shape[0]
    ti_all = best_t[0, order]
    out.update({"contacts": contacts.cpu().numpy(),
                "pose_rot": order.cpu().numpy().astype(np.int32),
                "pose_trans": ti_all.cpu().numpy().astype(np.int64),
                "grid_n": np.int32(n), "grid_h": np.float32(H),
                "origin": origin.cpu().numpy(),
                "ab_centre": ab_c.cpu().numpy(),
                "pose_score": best_s[0, order].cpu().numpy(),
                "pose_centre": centres.cpu().numpy(), "native": nv,
                "best": float(best_s[0].max()), "n_grid": n,
                "native_rank": int((best_s[0] > nv).sum().item())})
    return out


def main():
    global NROT
    gpu, shard, nsh = (int(x) for x in sys.argv[1:4])
    mode = sys.argv[4] if len(sys.argv) > 4 else "run"
    if mode in ("calib", "ucalib", "acalib"):
        NROT = 3000
    dev = torch.device(f"cuda:{gpu}")
    listing = {"probe": "probe_list.txt", "unbound": "unbound_list.txt",
               "calib": "train_list.txt", "ucalib": "train_list.txt",
               "common": "common90_list.txt", "apo": "apo_list.txt",
               "atune": "apo_tune_list.txt", "md": "md_dock_list.txt",
               "acalib": "apo_train_list.txt"}.get(mode, "list.txt")
    listing = os.environ.get("LIST", listing)
    units = (FFT / listing).read_text().split()
    units = units[shard::nsh]
    # fixed docking weights, tuned for unbound input (see the paper)
    w = {"contact": float(os.environ.get("W_CONTACT", 0.5)),
         "pot": float(os.environ.get("W_POT", 3.0)),
         "elec": float(os.environ.get("W_ELEC", -300.0)),
         "clash": float(os.environ.get("W_CLASH", 3.0))}
    print("weights", {k: round(v, 4) for k, v in w.items()}, flush=True)
    for i, u in enumerate(units, 1):
        t0 = time.time()
        src = FFT / ({"unbound": f"{u}.unbound.npz", "ucalib": f"{u}.unbound.npz",
                      "apo": f"{u}.apo.npz", "acalib": f"{u}.apo.npz",
                      "atune": f"{u}.apo.npz",
                      "md": f"{u}.{os.environ.get('FRAME', 'mdbest')}.npz"
                      }.get(mode, f"{u}.npz"))
        tag = os.environ.get("TAG") or {"calib": "calib", "ucalib": "ucalib",
                                        "acalib": "acalib", "unbound": "result3",
                                        "apo": "result_apo"}.get(mode, "result2")
        r = run(src, dev, w, mode in ("calib", "ucalib", "acalib"))
        np.savez_compressed(FFT / f"{u}.{tag}.npz", **r)
        rk = r["native_rank"]
        print(f"[{gpu}] {i}/{len(units)} {u}  native rank "
              f"{rk.min() if hasattr(rk,'min') else rk}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
