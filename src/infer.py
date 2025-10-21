from __future__ import annotations
import argparse, json, glob
from pathlib import Path
import joblib, numpy as np, pandas as pd
from .config import load_config
from .utils.logging import get_logger
from .data import load_frames
from .features import build_features


def _logit(p):
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return np.log(p / (1 - p))


def predict_flag(
    part: pd.DataFrame, model_dir: Path, flag: str, feats_json: Path
) -> pd.Series:
    fe_out, feats = build_features(part, use_cols_json=feats_json)
    # Use DataFrame directly to preserve feature names
    X = fe_out[feats]

    # fold models 평균
    paths = sorted(glob.glob(str(model_dir / f"model_{flag}_*.pkl")))
    preds = []
    for mp in paths:
        m = joblib.load(mp)
        p = m.predict_proba(X)
        if p.ndim == 2:
            p = p[:, 1]
        preds.append(p)
    avg = np.mean(preds, axis=0) if preds else np.zeros(len(X))

    # Platt 보정
    cal = joblib.load(model_dir / f"cal_{flag}.pkl")
    p_final = cal.predict_proba(_logit(avg).reshape(-1, 1))[:, 1]
    return pd.Series(np.clip(p_final, 1e-8, 1 - 1e-8), index=part.index, dtype=float)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", type=str, default=None, help="Model version to use (default: latest)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    log = get_logger("infer")

    # Determine model version to use
    base_model_dir = cfg.model_dir
    if args.version:
        version_name = args.version
    else:
        # Read latest version from LATEST_VERSION.txt
        latest_file = base_model_dir / "LATEST_VERSION.txt"
        if latest_file.exists():
            version_name = latest_file.read_text().strip()
            log.info(f"Using latest model version: {version_name}")
        else:
            raise FileNotFoundError(
                f"No version specified and {latest_file} not found. "
                "Please train a model first or specify --version"
            )

    # Override config model_dir with versioned path
    versioned_model_dir = base_model_dir / version_name
    if not versioned_model_dir.exists():
        raise FileNotFoundError(f"Model version directory not found: {versioned_model_dir}")

    cfg.paths["model_dir"] = str(versioned_model_dir)
    log.info(f"Loading models from: {versioned_model_dir}")

    _train_idx, test_idx, _A, _B, test_ab = load_frames(cfg.data_dir)
    test_full = test_idx.merge(test_ab, on=["Test_id", "Test"], how="left")

    outs = []
    for flag in ("A", "B"):
        part = test_full[test_full[cfg.columns["test"]] == flag].copy()
        if len(part) == 0:
            continue
        proba = predict_flag(
            part,
            cfg.model_dir,
            flag,
            cfg.model_dir / f"feature_cols_{flag}.json",
        )
        outs.append(
            pd.DataFrame(
                {"Test_id": part[cfg.columns["id"]].values, "Label": proba.values}
            )
        )

    sub = pd.concat(outs, ignore_index=True).sort_values("Test_id")
    sub.to_csv(args.out, index=False, encoding="utf-8")
    log.info(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
