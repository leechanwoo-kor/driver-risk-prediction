from __future__ import annotations
import numpy as np


def brier_score(y_true, p):
    """Brier score for binary classification."""
    p = np.clip(p, 1e-8, 1 - 1e-8)
    y = np.asarray(y_true, dtype=float)
    return np.mean((p - y) ** 2)


def ece_score(y_true, p, n_bins: int = 10):
    """Expected Calibration Error (ECE) - Official competition implementation."""
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy='uniform')
    bin_totals = np.histogram(p, bins=np.linspace(0, 1, n_bins + 1), density=False)[0]
    non_empty_bins = bin_totals > 0
    bin_weights = bin_totals / len(p)
    bin_weights = bin_weights[non_empty_bins]
    prob_true = prob_true[:len(bin_weights)]
    prob_pred = prob_pred[:len(bin_weights)]
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
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
