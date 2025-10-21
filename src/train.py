from __future__ import annotations
import argparse, json
from pathlib import Path
from datetime import datetime
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from .config import load_config
from .utils.logging import get_logger
from .utils.seed import set_seed
from .data import load_frames, merge_train
from .features import build_features
from .models.xgb import XGBWrapper
from .metrics import brier_score, ece_score


def _logit(p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def train_one_flag(df: pd.DataFrame, cfg, flag: str, n_splits: int = 5):
    log = get_logger(f"train:{flag}")
    log.info(f"Starting training for Test {flag}")
    
    col = cfg.columns
    data = df[df[col["test"]] == flag].copy()
    log.info(f"Test {flag} data: {len(data)} samples")
    
    y = data[col["target"]].astype(int).values
    data.drop(columns=[col["target"]], inplace=True)

    model_dir = cfg.model_dir
    use_cols_json = model_dir / f"feature_cols_{flag}.json"
    
    log.info(f"Building features for Test {flag}...")
    fe_out, feats = build_features(data, use_cols_json=use_cols_json)
    log.info(f"Features built: {len(feats)} features")
    
    # Use DataFrame directly to preserve feature names
    X = fe_out[feats]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_seed)
    oof = np.zeros(len(X), dtype=float)

    model_dir.mkdir(parents=True, exist_ok=True)

    for i, (tr, va) in enumerate(skf.split(X, y)):
        m = XGBWrapper(params=cfg.model["params"]).fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[va])
        if p.ndim == 2:
            p = p[:, 1]
        oof[va] = p
        joblib.dump(m.model, model_dir / f"model_{flag}_{i}.pkl")
        log.info(f"[{flag}] fold {i} AUC={roc_auc_score(y[va], p):.5f}")

    # 보정기(Platt): OOF logit → y 로 로지스틱 회귀
    cal = LogisticRegression(max_iter=500)
    cal.fit(_logit(oof).reshape(-1, 1), y)
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
    }
    with open(versioned_model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    log.info("Loading data frames...")
    train_idx, _test_idx, A, B, _ = load_frames(cfg.data_dir)
    log.info("Merging train data...")
    df = merge_train(train_idx, A, B)
    log.info(f"Data loaded: {len(df)} rows")

    # Store results for summary
    all_results = {}
    
    for flag in ("A", "B"):
        result = train_one_flag(
            df.assign(Label=df[cfg.columns["target"]]), cfg, flag, n_splits=args.folds
        )
        all_results[flag] = result

    # Save overall summary
    summary = {
        "version": version_name,
        "timestamp": metadata["timestamp"],
        "model_name": cfg.model.get("name", "unknown"),
        "total_samples": len(df),
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
