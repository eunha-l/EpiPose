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
"""Prepare the unbound antigen, solved on its own rather than in the complex.

The unbound chain is matched to the canonical antigen by sequence and superposed
onto the bound antigen by least squares on the matched CA atoms, which keeps the
antibody position in the complex meaningful as the native pose and keeps the
residue labels on the right residues. Only the residues present in both are
carried over, and the superposition error is recorded, since that is the change
the docking has to absorb.

Output: <EPIPOSE_DOCK_OUT>/<unit>.apo.npz
"""
import os
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

from dock_fft_prep import AA3, BACKBONE, REDUCED_SET, VDW, charge, classify
from dock_fft_prep_unbound import kabsch

warnings.filterwarnings("ignore")
RAW = _RAW
APO = Path(os.environ.get(
    "APODIR", str(_APO) if _APO else ""))
RES = Path(__file__).resolve().parent / "results"
PKFILE = os.environ.get("PKFILE", "ab_cond_predictions.parquet")
FFT = _DOCK_OUT
ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
aligner = Align.PairwiseAligner(mode="global", match_score=1, mismatch_score=0,
                                open_gap_score=-2, extend_gap_score=-1)


def chain_residues(path, want=None):
    """auth chain -> ordered residues, each with its atoms."""
    d = MMCIF2Dict(str(path))
    n = len(d["_atom_site.id"])
    mdl = d.get("_atom_site.pdbx_PDB_model_num", ["1"] * n)
    # An atom modelled in alternate conformations appears once per altloc. Without
    # this filter a side chain is rasterised several times and counted that many
    # times in the contact and overlap terms.
    alt = d.get("_atom_site.label_alt_id", ["."] * n)
    out, seen = {}, set()
    for grp, el, nm, comp, ch, seq, ic, x, y, z, mm, al in zip(
        d["_atom_site.group_PDB"], d["_atom_site.type_symbol"],
        d["_atom_site.label_atom_id"], d["_atom_site.label_comp_id"],
        d["_atom_site.auth_asym_id"], d["_atom_site.auth_seq_id"],
        d["_atom_site.pdbx_PDB_ins_code"], d["_atom_site.Cartn_x"],
        d["_atom_site.Cartn_y"], d["_atom_site.Cartn_z"], mdl, alt):
        if grp != "ATOM" or comp not in AA3 or el in ("H", "D") or mm != "1":
            continue
        if want is not None and ch != want:
            continue
        k = (int(seq), "" if ic in ("?", ".") else ic)
        # keep the first conformation offered for each (chain, residue, atom name)
        tag = (ch, k, nm)
        if tag in seen:
            continue
        seen.add(tag)
        r = out.setdefault(ch, {}).setdefault(k, {"comp": comp, "atoms": []})
        r["atoms"].append((nm, el, float(x), float(y), float(z)))
    for ch in out:
        out[ch] = [(k, v) for k, v in out[ch].items()
                   if any(a[0] == "CA" for a in v["atoms"])]
    return out


def one(job):
    # a third element names the variant, so several apo crystals of one antigen
    # can be prepared side by side instead of overwriting each other
    unit, apo_entry = job[0], job[1]
    tag = job[2] if len(job) > 2 else "apo"
    inst, chain = unit.split("__")
    p = APO / f"{apo_entry}.cif"
    if not p.is_file():
        return (unit, "no apo file", None)
    old = np.load(FFT / f"{unit}.npz", allow_pickle=True)
    keys = list(zip(old["res_resseq"].astype(int), [str(x) for x in old["res_icode"]]))
    kidx = {k: i for i, k in enumerate(keys)}
    try:
        holo = chain_residues(RAW / f"{inst}.cif", chain).get(chain, [])
        apo = chain_residues(p)
    except Exception as e:
        return (unit, f"parse: {e}", None)
    if len(holo) < 20:
        return (unit, "short holo chain", None)
    hseq = "".join(ONE[v["comp"]] for _k, v in holo)
    best = None
    for cid, rs in apo.items():
        aseq = "".join(ONE[v["comp"]] for _k, v in rs)
        if len(aseq) < 20:
            continue
        aln = aligner.align(hseq, aseq)[0]
        pr = [(i0 + o, j0 + o) for (i0, i1), (j0, _j1) in zip(*aln.aligned)
              for o in range(i1 - i0)]
        if len(pr) < 30:
            continue
        # sequence identity cutoff for accepting an apo chain
        if sum(hseq[i] == aseq[j] for i, j in pr) / len(pr) < 0.92:
            continue
        if best is None or len(pr) > len(best[2]):
            best = (cid, rs, pr)
    if best is None:
        return (unit, "no matching apo chain", None)
    _cid, ars, pr = best

    ca = lambda v: [a[2:5] for a in v["atoms"] if a[0] == "CA"][0]
    P = np.asarray([ca(ars[j][1]) for _i, j in pr], dtype=np.float64)
    Q = np.asarray([ca(holo[i][1]) for i, _j in pr], dtype=np.float64)
    R, t = kabsch(P, Q)
    rmsd = float(np.sqrt((((P @ R.T) + t - Q) ** 2).sum(axis=1).mean()))

    ag, agr, agc, agq, agrad, agbb, agred = [], [], [], [], [], [], []

    agmeta = []
    for i, j in pr:
        tgt = kidx.get(holo[i][0], -1)
        if tgt < 0:
            continue
        v = ars[j][1]
        for nm, el, x, y, z in v["atoms"]:
            xyz = R @ np.array([x, y, z]) + t
            ag.append(xyz); agr.append(tgt)
            # the atom's identity, in the same six columns the other preps use.
            # Without it `feat_build` falls back to the full structure's atom list,
            # which is longer than the mapped set, and indexes past the end.
            agmeta.append((nm, v["comp"], "A", int(keys[tgt][0]),
                           str(keys[tgt][1]), el))
            agc.append(classify(v["comp"], nm, el)); agq.append(charge(v["comp"], nm))
            agrad.append(VDW.get(el, 1.70))
            agbb.append(1 if nm in BACKBONE else 0)
            agred.append(1 if nm in REDUCED_SET else 0)
    if len(ag) < 200:
        return (unit, "too few mapped atoms", None)
    np.savez_compressed(
        FFT / f"{unit}.{tag}.npz",
        ag=np.asarray(ag, np.float32), ag_res=np.asarray(agr, np.int32),
        ag_rad=np.asarray(agrad, np.float32), ag_cls=np.asarray(agc, np.int8),
        ag_q=np.asarray(agq, np.float32), ag_bb=np.asarray(agbb, np.int8),
        ag_red=np.asarray(agred, np.int8),
        ag_meta=np.array(agmeta, dtype=object),
        ab_bb=old["ab_bb"], ab_red=old["ab_red"],
        ab=old["ab"], ab_cdr=old["ab_cdr"], ab_rad=old["ab_rad"],
        ab_cls=old["ab_cls"], ab_q=old["ab_q"],
        res_resseq=old["res_resseq"], res_icode=old["res_icode"],
        is_core=old["is_core"], rmsd=np.float32(rmsd),
        source=np.array(apo_entry), n_mapped=np.int32(len(pr)))
    return (unit, "ok", rmsd)


def main():
    amap = os.environ.get("APOMAP")
    if amap:
        # the assignment comes from the split that defines the evaluation; its
        # table is keyed by instance because it names no antigen chain
        units = (FFT / os.environ["APOLIST"]).read_text().split()
        by = dict(zip(pd.read_csv(RES / amap).unit,
                      pd.read_csv(RES / amap).entry))
        jobs = [(u, by[u.split("__")[0]], os.environ.get("APOTAG", "apo"))
                for u in units if u.split("__")[0] in by]
    else:
        have = {p.name[:-len(".result4.npz")] for p in FFT.glob("*.result4.npz")}
        a = pd.read_parquet(RES / "apo_shape_features.parquet")
        a["unit"] = a.instance + "__" + a.chain
        pick = (a[a.unit.isin(have)].groupby("unit").apo_entry.first())
        jobs = list(pick.items())
    print(f"{len(jobs)} chains have an experimental apo structure", flush=True)
    with Pool(24) as p:
        res = p.map(one, jobs)
    ok = [(u, r) for u, s, r in res if s == "ok"]
    why = pd.Series([s for _u, s, _r in res if s != "ok"]).value_counts()
    print(f"prepared {len(ok)}")
    for k, v in why.items():
        print(f"  {k}: {v}")
    r = np.asarray([x for _u, x in ok])
    print(f"\nsuperposition onto the bound antigen, CA RMSD")
    for q in (10, 25, 50, 75, 90):
        print(f"  {q:2d}th percentile {np.percentile(r, q):5.2f} A")
    (FFT / os.environ.get("APOOUT", "apo_list.txt")).write_text("\n".join(u for u, _ in ok) + "\n")
    test = set()
    sp = Path(str(_WORK) + "/"
              "7a3652e9-cb14-4865-925f-1ad360589c94/scratchpad/epi_seed42.parquet")
    if sp.is_file():
        e = pd.read_parquet(sp)
        test = {f"{i}__{c}" for i, c in zip(e.instance, e.chain)}
    print(f"\nof these, {len({u for u, _ in ok} & test)} are in the shared test set")


if __name__ == "__main__":
    main()
