from __future__ import annotations
import numpy as np


def brier_score(y_true, p):
    """Brier score for binary classification."""
    p = np.clip(p, 1e-8, 1 - 1e-8)
    y = np.asarray(y_true, dtype=float)
    return np.mean((p - y) ** 2)


def ece_score(y_true, p, n_bins: int = 15):
    """Expected Calibration Error (ECE) for binary classification."""
    p = np.clip(p, 1e-8, 1 - 1e-8)
    y = np.asarray(y_true, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        w = mask.mean()
        ece += w * abs(acc - conf)
    return ece


def competition_score(y_true, p):
    """
    Calculate competition metric: 0.5*(1-AUC) + 0.25*Brier + 0.25*ECE
    Returns: (score, auc, brier, ece)
    """
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y_true, p)
    brier = brier_score(y_true, p)
    ece = ece_score(y_true, p)

    score = 0.5 * (1 - auc) + 0.25 * brier + 0.25 * ece

    return score, auc, brier, ece
