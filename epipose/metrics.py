"""Evaluation metrics.

The global threshold is the value that maximises MCC on the validation split
and is then applied to the test split.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score,
                             matthews_corrcoef, roc_auc_score)

GRID = np.arange(0.002, 0.90, 0.002)


def _mcc_grid(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """MCC at every grid point, from one sort and a cumulative sum."""
    order = np.argsort(-p)
    ys = y[order].astype(np.int64)
    cum = np.concatenate([[0], np.cumsum(ys)])
    n_pos = int(ys.sum())
    n_neg = len(ys) - n_pos
    k = np.searchsorted(-p[order], -GRID, side="left")
    tp = cum[k].astype(np.float64)
    fp = k - tp
    fn = n_pos - tp
    tn = n_neg - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.divide(tp * tn - fp * fn, den, out=np.zeros_like(den), where=den > 0)


def best_threshold(pairs: list[tuple[np.ndarray, np.ndarray]]) -> float:
    p = np.concatenate([a for a, _ in pairs])
    y = np.concatenate([b for _, b in pairs])
    return float(GRID[int(np.argmax(_mcc_grid(p, y)))])


def evaluate(pairs: list[tuple[np.ndarray, np.ndarray]],
             threshold: float | None = None, top_k: int = 30) -> dict:
    """pairs holds one (predicted map, labels) tuple per unit."""
    p = np.concatenate([a for a, _ in pairs])
    y = np.concatenate([b for _, b in pairs])
    if threshold is None:
        threshold = float(GRID[int(np.argmax(_mcc_grid(p, y)))])
    tp = fp = fn = 0
    for pm, yl in pairs:
        yb = yl.astype(bool)
        k = min(top_k, len(pm))
        sel = np.zeros(len(pm), bool)
        sel[np.argsort(-pm)[:k]] = True
        tp += int((sel & yb).sum())
        fp += int((sel & ~yb).sum())
        fn += int((~sel & yb).sum())
    binary = (p > threshold).astype(int)
    per_unit = [roc_auc_score(yl, pm) for pm, yl in pairs if 0 < yl.sum() < len(yl)]
    return dict(threshold=threshold,
                auroc=float(roc_auc_score(y, p)),
                auprc=float(average_precision_score(y, p)),
                mcc=float(matthews_corrcoef(y, binary)),
                f1=float(f1_score(y, binary, zero_division=0)),
                f1_at_k=2 * tp / max(2 * tp + fp + fn, 1),
                auroc_macro=float(np.mean(per_unit)) if per_unit else float("nan"))


def pose_auroc(logits: list[np.ndarray], qs: list[np.ndarray],
               cut: float = 0.5, min_pos: int = 5) -> tuple[float, int]:
    """Pose level AUROC averaged over units, with the number of units
    skipped for having too few positive poses."""
    vals = []
    skipped = 0
    for lg, q in zip(logits, qs):
        lab = q > cut
        if lab.sum() < min_pos or lab.all():
            skipped += 1
            continue
        vals.append(roc_auc_score(lab, lg))
    return (float(np.mean(vals)) if vals else float("nan")), skipped

