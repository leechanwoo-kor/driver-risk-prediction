"""
Hyperparameter tuning for XGBoost models.
Uses grid search with cross-validation to find optimal parameters.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from src.metrics import competition_score
from src.utils.logging import get_logger

log = get_logger("tune")

# Grid search space (focused on most impactful parameters)
param_grid = {
    'n_estimators': [200, 300, 500],
    'max_depth': [5, 6, 7],
    'learning_rate': [0.03, 0.05, 0.07],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'gamma': [0, 0.1],
}

def evaluate_params(X, y, params, n_splits=3):
    """Evaluate a set of hyperparameters using CV."""
    # Clean data first
    X_clean = X.fillna(X.median()).replace([np.inf, -np.inf], 0)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X_clean))

    for tr, va in skf.split(X_clean, y):
        model = xgb.XGBClassifier(
            **params,
            objective='binary:logistic',
            random_state=42,
            n_jobs=1,  # Avoid multiprocessing issues
            verbosity=0
        )
        model.fit(
            X_clean.iloc[tr], y.iloc[tr],
            eval_set=[(X_clean.iloc[va], y.iloc[va])],
            verbose=False
        )
        oof[va] = model.predict_proba(X_clean.iloc[va])[:, 1]

    score, auc, brier, ece = competition_score(y, oof)
    return score, auc, brier, ece

def grid_search(X, y, test_name, n_splits=3):
    """Perform grid search for best hyperparameters."""
    log.info(f"\n{'='*70}")
    log.info(f"Tuning hyperparameters for {test_name}")
    log.info(f"{'='*70}")
    log.info(f"Samples: {len(X)}, Features: {X.shape[1]}")
    log.info(f"Positive ratio: {y.mean():.2%}")

    # Create all combinations
    from itertools import product
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    all_combinations = list(product(*values))
    total = len(all_combinations)

    log.info(f"\nTesting {total} parameter combinations...")

    best_score = float('inf')
    best_params = None
    best_metrics = None

    results = []

    for i, combo in enumerate(all_combinations, 1):
        params = dict(zip(keys, combo))

        try:
            score, auc, brier, ece = evaluate_params(X, y, params, n_splits)

            results.append({
                'params': params.copy(),
                'score': score,
                'auc': auc,
                'brier': brier,
                'ece': ece
            })

            if score < best_score:
                best_score = score
                best_params = params.copy()
                best_metrics = (auc, brier, ece)
                log.info(f"[{i}/{total}] ✅ NEW BEST! Score={score:.5f} (AUC={auc:.5f}, Brier={brier:.5f}, ECE={ece:.5f})")
            else:
                if i % 10 == 0:
                    log.info(f"[{i}/{total}] Score={score:.5f}")

        except Exception as e:
            log.error(f"[{i}/{total}] Error with params {params}: {e}")
            continue

    # Print results
    log.info(f"\n{'='*70}")
    log.info(f"BEST PARAMETERS for {test_name}")
    log.info(f"{'='*70}")
    log.info(f"Score: {best_score:.5f}")
    log.info(f"  AUC: {best_metrics[0]:.5f}")
    log.info(f"  Brier: {best_metrics[1]:.5f}")
    log.info(f"  ECE: {best_metrics[2]:.5f}")
    log.info(f"\nParameters:")
    for k, v in best_params.items():
        log.info(f"  {k}: {v}")

    # Show top 5
    results_sorted = sorted(results, key=lambda x: x['score'])[:5]
    log.info(f"\nTop 5 configurations:")
    for i, r in enumerate(results_sorted, 1):
        log.info(f"{i}. Score={r['score']:.5f}: {r['params']}")

    return best_params, best_score

def main():
    log.info("Loading prepared data...")

    prepared_dir = Path("data/prepared")

    # Load Test A data
    X_A = pd.read_pickle(prepared_dir / "X_A.pkl")
    y_A = pd.read_pickle(prepared_dir / "y_A.pkl")

    # Load Test B data
    X_B = pd.read_pickle(prepared_dir / "X_B.pkl")
    y_B = pd.read_pickle(prepared_dir / "y_B.pkl")

    # Tune Test A
    best_params_A, best_score_A = grid_search(X_A, y_A, "Test A", n_splits=3)

    # Tune Test B
    best_params_B, best_score_B = grid_search(X_B, y_B, "Test B", n_splits=3)

    # Overall score
    overall_score = (best_score_A + best_score_B) / 2

    log.info(f"\n{'='*70}")
    log.info(f"FINAL RESULTS")
    log.info(f"{'='*70}")
    log.info(f"Test A Score: {best_score_A:.5f}")
    log.info(f"Test B Score: {best_score_B:.5f}")
    log.info(f"Overall Score: {overall_score:.5f}")
    log.info(f"\nUpdate configs/default.yaml with these parameters:")
    log.info(f"\nTest A:")
    for k, v in best_params_A.items():
        log.info(f"  {k}: {v}")
    log.info(f"\nTest B:")
    for k, v in best_params_B.items():
        log.info(f"  {k}: {v}")

if __name__ == "__main__":
    main()
