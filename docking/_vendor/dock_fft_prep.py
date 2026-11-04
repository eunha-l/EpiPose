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
"""Pull the coordinates the FFT docking needs out of the complexes.

The docking itself runs under a different interpreter, the one that has a GPU
build of torch, and that interpreter has no structure parser. So the atoms are
extracted here and written as plain arrays.

Antigen and antibody are separated, the antibody's atoms are marked according to
whether they belong to a CDR loop, and each antigen atom is tagged with the
residue it came from so that poses can be turned back into residue labels. The
antigen is taken from the complex, in the conformation it holds there, which is
the easiest case docking will ever face; if the score cannot find the right site
under those conditions it will not find it under harder ones.

Output: results/dock_fft/<instance>__<chain>.npz
"""
import os
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

warnings.filterwarnings("ignore")
RAW = _RAW
RES = Path(__file__).resolve().parent / "results"
PKFILE = os.environ.get("PKFILE", "ab_cond_predictions.parquet")
OUT = _DOCK_OUT
AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
       "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80}

# counting contacts treats every atom alike, which lets a wide flat patch of
# anything outscore a small real epitope. Three classes are enough to tell
# burying oil from pairing charges: carbon and sulfur, neutral nitrogen and
# oxygen, and the atoms that carry a formal charge.
CHARGED = {("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1"), ("GLU", "OE2"),
           ("LYS", "NZ"), ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
           ("HIS", "ND1"), ("HIS", "NE2")}
# partial charges, kept to the groups that dominate: the formal charges and the
# backbone amide, whose dipoles add up over a whole interface
Q = {("ASP", "OD1"): -0.5, ("ASP", "OD2"): -0.5, ("GLU", "OE1"): -0.5,
     ("GLU", "OE2"): -0.5, ("LYS", "NZ"): 1.0, ("ARG", "NE"): 0.3,
     ("ARG", "NH1"): 0.35, ("ARG", "NH2"): 0.35, ("HIS", "ND1"): 0.1,
     ("HIS", "NE2"): 0.1}


BACKBONE = {"N", "CA", "C", "O", "OXT"}
REDUCED_SET = BACKBONE | {"CB"}


def classify(comp, name, el):
    if (comp, name) in CHARGED:
        return 2
    return 0 if el in ("C", "S") else 1 if el in ("N", "O") else 0


def charge(comp, name):
    if (comp, name) in Q:
        return Q[(comp, name)]
    if name == "N":
        return 0.3
    if name == "O":
        return -0.3
    return 0.0


def cdr_of(n):
    return 1 if 27 <= n <= 38 else 2 if 56 <= n <= 65 else 3 if 105 <= n <= 117 else 0


def one(job):
    inst, chain, keys, core = job
    h, l = inst.split("_")[2:4]
    try:
        d = MMCIF2Dict(str(RAW / f"{inst}.cif"))
    except Exception:
        return None
    kidx = {k: i for i, k in enumerate(keys)}
    ag, agr, agrad, ab, abcdr, abrad = [], [], [], [], [], []
    agc, agq, abc, abq, agbb, abbb = [], [], [], [], [], []
    agred, abred = [], []
    for grp, el, nm, comp, ch, seq, ic, x, y, z in zip(
        d["_atom_site.group_PDB"], d["_atom_site.type_symbol"],
        d["_atom_site.label_atom_id"], d["_atom_site.label_comp_id"],
        d["_atom_site.auth_asym_id"], d["_atom_site.auth_seq_id"],
        d["_atom_site.pdbx_PDB_ins_code"], d["_atom_site.Cartn_x"], d["_atom_site.Cartn_y"],
        d["_atom_site.Cartn_z"]):
        if grp != "ATOM" or comp not in AA3 or el in ("H", "D"):
            continue
        p = (float(x), float(y), float(z))
        r = VDW.get(el, 1.70)
        cls, q = classify(comp, nm, el), charge(comp, nm)
        if ch in (h, l):
            ab.append(p); abcdr.append(1 if cdr_of(int(seq)) else 0); abrad.append(r)
            abc.append(cls); abq.append(q); abbb.append(1 if nm in BACKBONE else 0)
            abred.append(1 if nm in REDUCED_SET else 0)
        elif ch == chain:
            k = (int(seq), "" if ic in ("?", ".") else ic)
            if k in kidx:
                ag.append(p); agr.append(kidx[k]); agrad.append(r)
                agc.append(cls); agq.append(q)
                agbb.append(1 if nm in BACKBONE else 0)
                agred.append(1 if nm in REDUCED_SET else 0)
    ag = np.asarray(ag, dtype=np.float32); ab = np.asarray(ab, dtype=np.float32)
    abcdr = np.asarray(abcdr, dtype=np.int8)
    if len(ag) < 200 or len(ab) < 500 or abcdr.sum() < 50:
        return None
    np.savez_compressed(
        OUT / f"{inst}__{chain}.npz",
        ag=ag, ag_res=np.asarray(agr, dtype=np.int32), ag_rad=np.asarray(agrad, np.float32),
        ab=ab, ab_cdr=abcdr, ab_rad=np.asarray(abrad, np.float32),
        ag_cls=np.asarray(agc, dtype=np.int8), ag_q=np.asarray(agq, dtype=np.float32),
        ab_cls=np.asarray(abc, dtype=np.int8), ab_q=np.asarray(abq, dtype=np.float32),
        ag_bb=np.asarray(agbb, dtype=np.int8), ab_bb=np.asarray(abbb, dtype=np.int8),
        ag_red=np.asarray(agred, dtype=np.int8), ab_red=np.asarray(abred, dtype=np.int8),
        res_resseq=np.asarray([k[0] for k in keys], dtype=np.int32),
        res_icode=np.asarray([k[1] for k in keys], dtype="U2"),
        is_core=np.asarray(core, dtype=np.int8))
    return f"{inst}__{chain}"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    OUT.mkdir(parents=True, exist_ok=True)
    pk = pd.read_parquet(RES / PKFILE)
    pk["unit"] = pk.instance + "|" + pk.chain
    units = sorted(pk.unit.unique())
    want = os.environ.get("ONLY")
    if want:
        keep = set((OUT / want).read_text().split())
        pick = [u for u in units if u.split("|")[0] in keep]
    else:
        rng = np.random.default_rng(7)
        pick = units if n >= len(units) else \
            list(rng.choice(units, size=n, replace=False))
    jobs = []
    for u, g in pk[pk.unit.isin(set(pick))].groupby("unit"):
        jobs.append((g.instance.iloc[0], g.chain.iloc[0],
                     list(zip(g.resseq.astype(int), g.icode.fillna("").astype(str))),
                     g.is_core.values))
    with Pool(16) as p:
        ok = [r for r in p.map(one, jobs) if r]
    print(f"wrote {len(ok)} of {len(jobs)} chains to {OUT}")
    (OUT / os.environ.get("OUTLIST", "list.txt")).write_text("\n".join(sorted(ok)) + "\n")


if __name__ == "__main__":
    main()
