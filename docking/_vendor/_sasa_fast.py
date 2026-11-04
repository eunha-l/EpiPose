"""Solvent accessible surface area for the first `only` atoms.

The remaining atoms still occlude, so restricting the loop to the antigen gives
the same values as computing the whole complex and discarding the rest.
"""
import numpy as np
from scipy.spatial import cKDTree


def sasa(xyz, rad, probe=1.4, n=100, only=None):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    unit = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi),
                     np.cos(phi)], axis=1)
    R = rad + probe
    tree = cKDTree(xyz)
    m = len(xyz) if only is None else int(only)
    nb = tree.query_ball_point(xyz[:m], 2 * R.max())
    out = np.zeros(m)
    for a in range(m):
        pts = xyz[a] + unit * R[a]
        js = [j for j in nb[a] if j != a]
        if js:
            d = np.linalg.norm(pts[:, None, :] - xyz[js][None, :, :], axis=2)
            free = (d >= R[js][None, :]).all(axis=1)
        else:
            free = np.ones(n, bool)
        out[a] = 4 * np.pi * R[a] ** 2 * free.mean()
    return out
