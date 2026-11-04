"""Paths used by the vendored scripts.

Every location is read from an environment variable and defaults to a folder
inside this repository. WORK and RAW are the two that usually need setting.

  EPIPOSE_ROOT   repository root (default: two levels above this file)
  EPIPOSE_WORK   docking output and intermediates (default: $ROOT/work)
  EPIPOSE_RAW    complex cif files, needed to rebuild tensors or the target
  EPIPOSE_APO    apo cif files, needed to rebuild the apo tensors
"""
from __future__ import annotations

import os
from pathlib import Path
from pathlib import Path as pathlib_Path

ROOT = Path(os.environ.get("EPIPOSE_ROOT",
                           Path(__file__).resolve().parents[2]))
ASSETS = ROOT / "data" / "assets"
FEATURES = ROOT / "data" / "features"
WORK = Path(os.environ.get("EPIPOSE_WORK", ROOT / "work"))
# Docking output, by default the structures shipped with the repository.
DOCK_OUT = pathlib_Path(os.environ.get(
    "EPIPOSE_DOCK_OUT", ROOT / "data" / "structures" / "dock_fft"))
# Tensors of the further apo structures, used for training augmentation.
APOC = pathlib_Path(os.environ.get("EPIPOSE_APOC",
                                   ROOT / "data" / "structures" / "apoc"))
# Pose feature extraction reads the residue count and surface flags here.
CONSASA = pathlib_Path(os.environ.get(
    "EPIPOSE_CONSASA", ROOT / "data" / "structures" / "consasa"))
# Complex cif files, read only when rebuilding the auxiliary target.
RAW = pathlib_Path(os.environ.get("EPIPOSE_RAW",
                                  ROOT / "data" / "structures" / "complex_cif"))
APO = Path(os.environ.get("EPIPOSE_APO", "")) if os.environ.get("EPIPOSE_APO") else None

GROUPS_CSV = ASSETS / "epipose_split.csv"
DOCK2CANON = ASSETS / "dock2canon.npz"
RES_FEAT = ASSETS / "res_feat_fix.npz"
ZEPII = ASSETS / "zepii_res.npy"
DOCK_WEIGHTS = ASSETS / "dock_weights_raw.npz"
ROT12000 = ASSETS / "rot12000.npz"


def ensure_work() -> None:
    DOCK_OUT.mkdir(parents=True, exist_ok=True)


def require(path: Path, what: str) -> Path:
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            f"{what} not found: {path}\n"
            f"Set the matching environment variable, see paths.py.")
    return Path(path)
