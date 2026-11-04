"""Least squares superposition, used to place an apo antigen on the bound one.

Returns (R, t) with Q ~= R @ P.T + t, where callers pass P as the apo alpha
carbons and Q as the bound ones.
"""
import numpy as np


def kabsch(P, Q):
    """Rotation R (3,3) and translation t (3,) fitting P (n,3) onto Q (n,3)."""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    cp, cq = P.mean(0), Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cq - R @ cp
    return R, t
