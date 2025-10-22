from __future__ import annotations
import argparse, json
from pathlib import Path
from datetime import datetime
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from .config import load_config
from .utils.logging import get_logger
from .utils.seed import set_seed
from .data import load_frames, merge_train, prepare_data_for_test, load_prepared_data
from .models.xgb import XGBWrapper
from .models.catboost_model import CatBoostWrapper
from .models.lgbm import LGBMWrapper
from .metrics import brier_score, ece_score


def _logit(p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def train_one_flag(
    X: pd.DataFrame,
    y: pd.Series,
    feats: list[str],
    cfg,
    flag: str,
    n_splits: int = 5,
    use_time_split: bool = False,
    test_ratio: float = 0.2
):
    log = get_logger(f"train:{flag}")
    log.info(f"Starting training for Test {flag}")
    log.info(f"Test {flag} data: {len(X)} samples, {len(feats)} features")

    model_dir = cfg.model_dir
    y = y.values  # Convert to numpy array for sklearn

    # Choose validation strategy
    if use_time_split:
        # Time-based split: use most recent data as validation
        log.info(f"Using time-based split (last {test_ratio*100:.0f}% as validation)")
        # Assume index is sorted by TestDate
        split_idx = int(len(X) * (1 - test_ratio))
        train_idx = np.arange(split_idx)
        val_idx = np.arange(split_idx, len(X))
        splits = [(train_idx, val_idx)]
        oof = np.zeros(len(X), dtype=float)
    else:
        # StratifiedKFold
        log.info(f"Using StratifiedKFold (n_splits={n_splits})")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_seed)
        splits = skf.split(X, y)
        oof = np.zeros(len(X), dtype=float)

    model_dir.mkdir(parents=True, exist_ok=True)

    # Select model type
    model_type = cfg.model.get("type", "xgb")
    if model_type == "catboost":
        ModelClass = CatBoostWrapper
    elif model_type == "lightgbm":
        ModelClass = LGBMWrapper
    else:
        ModelClass = XGBWrapper

    for i, (tr, va) in enumerate(splits):
        m = ModelClass(params=cfg.model["params"]).fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[va])
        if p.ndim == 2:
            p = p[:, 1]
        oof[va] = p
        joblib.dump(m.model, model_dir / f"model_{flag}_{i}.pkl")
        log.info(f"[{flag}] fold {i} AUC={roc_auc_score(y[va], p):.5f}")

    # 보정기(Isotonic): OOF 확률 → y 로 단조 회귀 (Platt보다 robust)
    cal = IsotonicRegression(out_of_bounds='clip')
    cal.fit(oof, y)
    joblib.dump(cal, model_dir / f"cal_{flag}.pkl")

    # 리포트용 점수
    auc = roc_auc_score(y, oof)
    bri = brier_score(y, oof)
    ece = ece_score(y, oof)
    score_estimate = 0.5 * (1 - auc) + 0.25 * bri + 0.25 * ece
    log.info(
        f"[{flag}] OOF AUC={auc:.5f} Brier={bri:.5f} ECE={ece:.5f}  -> Score≈{score_estimate:.5f}"
    )

    # feature 리스트도 보관
    (model_dir / f"feature_cols_{flag}.json").write_text(
        json.dumps(feats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Save imputation statistics (median of training data)
    impute_stats = X[feats].median().to_dict()
    (model_dir / f"impute_{flag}.json").write_text(
        json.dumps(impute_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(f"Saved imputation statistics for {flag}")
    
    # Save test-specific results
    results = {
        "test_type": flag,
        "n_samples": len(X),
        "n_features": len(feats),
        "n_folds": n_splits,
        "oof_auc": float(auc),
        "oof_brier": float(bri),
        "oof_ece": float(ece),
        "estimated_score": float(score_estimate),
        "fold_aucs": []
    }
    
    # Store individual fold results (re-calculate from saved predictions)
    for i in range(n_splits):
        fold_mask = np.zeros(len(X), dtype=bool)
        # This is approximate - we don't save exact fold splits
        # But we can add it if needed
    
    (model_dir / f"results_{flag}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--version", type=str, default=None, help="Version name (default: timestamp)")
    ap.add_argument("--prepared-data", type=str, default=None,
                    help="Path to prepared data directory (skip feature engineering)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.random_seed)
    log = get_logger("train")

    # Create versioned model directory
    if args.version:
        version_name = args.version
    else:
        # Auto-generate version name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = cfg.model.get("name", "model")
        version_name = f"{model_name}_{timestamp}"

    # Create versioned directory
    base_model_dir = cfg.model_dir
    versioned_model_dir = base_model_dir / version_name
    versioned_model_dir.mkdir(parents=True, exist_ok=True)

    # Override config model_dir with versioned path
    cfg.paths["model_dir"] = str(versioned_model_dir)

    log.info(f"Model version: {version_name}")
    log.info(f"Saving models to: {versioned_model_dir}")

    # Save training metadata
    metadata = {
        "version": version_name,
        "timestamp": datetime.now().isoformat(),
        "model_name": cfg.model.get("name", "unknown"),
        "n_folds": args.folds,
        "random_seed": cfg.random_seed,
        "config_file": args.config,
        "used_prepared_data": args.prepared_data is not None,
    }
    with open(versioned_model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Store results for summary
    all_results = {}

    # Read validation strategy from config
    use_time_split = cfg.validation.get('strategy') == 'time_split'
    test_ratio = cfg.validation.get('test_ratio', 0.2)

    if use_time_split:
        log.info(f"Using time-based validation (last {test_ratio*100:.0f}% as validation)")
    else:
        log.info(f"Using StratifiedKFold with {args.folds} folds")

    # Load or prepare data
    if args.prepared_data:
        # Use pre-prepared data (fast path)
        prepared_dir = Path(args.prepared_data)
        log.info(f"Using prepared data from: {prepared_dir}")

        for flag in ("A", "B"):
            X, feats, y = load_prepared_data(prepared_dir, flag)
            result = train_one_flag(X, y, feats, cfg, flag,
                                   n_splits=args.folds,
                                   use_time_split=use_time_split,
                                   test_ratio=test_ratio)
            all_results[flag] = result

            # Copy imputation statistics to model directory
            impute_src = prepared_dir / f"impute_stats_{flag}.json"
            if impute_src.exists():
                import shutil
                impute_dst = versioned_model_dir / f"impute_stats_{flag}.json"
                shutil.copy(impute_src, impute_dst)
                log.info(f"Copied imputation statistics to {impute_dst}")
    else:
        # Prepare data on-the-fly (original behavior)
        log.info("Loading and preparing data from scratch...")
        train_idx, test_idx, A, B, _ = load_frames(cfg.data_dir)
        df = merge_train(train_idx, A, B)
        log.info(f"Data loaded: {len(df)} rows")

        for flag in ("A", "B"):
            fe_out, feats, y = prepare_data_for_test(
                df.assign(Label=df[cfg.columns["target"]]),
                cfg,
                flag,
                output_dir=None  # Don't save intermediate data
            )
            X = fe_out[feats]
            result = train_one_flag(X, y, feats, cfg, flag,
                                   n_splits=args.folds,
                                   use_time_split=use_time_split,
                                   test_ratio=test_ratio)
            all_results[flag] = result

    # Save overall summary
    total_samples = sum(r.get("n_samples", 0) for r in all_results.values())
    summary = {
        "version": version_name,
        "timestamp": metadata["timestamp"],
        "model_name": cfg.model.get("name", "unknown"),
        "total_samples": total_samples,
        "test_A": all_results.get("A", {}),
        "test_B": all_results.get("B", {}),
        "overall_score": (
            all_results.get("A", {}).get("estimated_score", 0) * 0.5 +
            all_results.get("B", {}).get("estimated_score", 0) * 0.5
        ) if all_results else 0,
    }
    
    with open(versioned_model_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    log.info("Training (CV + Calibration) done.")
    log.info(f"Overall Score (A+B): {summary['overall_score']:.5f}")
    log.info(f"✅ Models saved to: {versioned_model_dir}")
    
    # Create a symlink or copy to 'latest' for convenience
    latest_link = base_model_dir / "latest"
    if latest_link.exists():
        if latest_link.is_symlink() or latest_link.is_dir():
            import shutil
            if latest_link.is_symlink():
                latest_link.unlink()
            else:
                shutil.rmtree(latest_link)
    
    # Create symlink (Windows may require admin rights, fallback to file marker)
    try:
        import os
        os.symlink(versioned_model_dir, latest_link, target_is_directory=True)
        log.info(f"Created symlink: {latest_link} -> {version_name}")
    except (OSError, NotImplementedError):
        # Fallback: create a text file pointing to latest version
        with open(base_model_dir / "LATEST_VERSION.txt", "w") as f:
            f.write(version_name)
        log.info(f"Latest version marker: {version_name}")


if __name__ == "__main__":
    main()
