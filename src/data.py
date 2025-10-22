from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from .features import build_features
from .utils.logging import get_logger

log = get_logger("data")


def load_frames(data_dir: Path):
    train_idx = pd.read_csv(data_dir / "train.csv")
    test_idx  = pd.read_csv(data_dir / "test.csv")
    A = pd.read_csv(data_dir / "train" / "A.csv")
    B = pd.read_csv(data_dir / "train" / "B.csv")
    # 평가 시 교체되는 테스트 세부
    test_A = pd.read_csv(data_dir / "test" / "A.csv") if (data_dir/"test"/"A.csv").exists() else pd.DataFrame()
    test_B = pd.read_csv(data_dir / "test" / "B.csv") if (data_dir/"test"/"B.csv").exists() else pd.DataFrame()
    test_ab = pd.concat([test_A, test_B], ignore_index=True, sort=False)
    return train_idx, test_idx, A, B, test_ab


def merge_train(train_idx: pd.DataFrame, A: pd.DataFrame, B: pd.DataFrame) -> pd.DataFrame:
    base = pd.concat([A, B], ignore_index=True, sort=False)
    return train_idx.merge(base, on=["Test_id", "Test"], how="left")


def prepare_data_for_test(
    df: pd.DataFrame,
    cfg,
    flag: str,
    output_dir: Path | None = None,
    use_cols_json: Path | None = None
) -> tuple[pd.DataFrame, list[str], pd.Series]:
    """
    Prepare data for a specific test (A or B) with feature engineering.

    Args:
        df: Merged training dataframe
        cfg: Configuration object
        flag: Test type ('A' or 'B')
        output_dir: Directory to save preprocessed data (optional)
        use_cols_json: Path to existing feature columns JSON (optional)

    Returns:
        fe_out: Dataframe with engineered features
        feats: List of feature column names
        y: Target variable (Label)
    """
    log.info(f"Preparing data for Test {flag}")

    col = cfg.columns
    data = df[df[col["test"]] == flag].copy()
    log.info(f"Test {flag} data: {len(data)} samples")

    y = data[col["target"]].astype(int)
    data.drop(columns=[col["target"]], inplace=True)

    log.info(f"Building features for Test {flag}...")
    fe_out, feats, impute_stats = build_features(data, use_cols_json=use_cols_json)
    log.info(f"Features built: {len(feats)} features")

    # Save preprocessed data if output_dir is specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save feature matrix
        fe_out[feats].to_pickle(output_dir / f"X_{flag}.pkl")
        log.info(f"Saved feature matrix to {output_dir / f'X_{flag}.pkl'}")

        # Save target
        y.to_pickle(output_dir / f"y_{flag}.pkl")
        log.info(f"Saved target to {output_dir / f'y_{flag}.pkl'}")

        # Save feature names
        feature_cols_path = output_dir / f"feature_cols_{flag}.json"
        feature_cols_path.write_text(
            json.dumps(feats, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        log.info(f"Saved feature columns to {feature_cols_path}")

        # Save imputation statistics
        if impute_stats:
            impute_path = output_dir / f"impute_stats_{flag}.json"
            impute_path.write_text(
                json.dumps(impute_stats, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            log.info(f"Saved imputation statistics to {impute_path}")

    return fe_out, feats, y


def load_prepared_data(data_dir: Path, flag: str) -> tuple[pd.DataFrame, list[str], pd.Series]:
    """
    Load preprocessed data for a specific test.

    Args:
        data_dir: Directory containing preprocessed data
        flag: Test type ('A' or 'B')

    Returns:
        X: Feature dataframe
        feats: List of feature column names
        y: Target variable
    """
    log.info(f"Loading prepared data for Test {flag} from {data_dir}")

    X = pd.read_pickle(data_dir / f"X_{flag}.pkl")
    y = pd.read_pickle(data_dir / f"y_{flag}.pkl")

    feature_cols_path = data_dir / f"feature_cols_{flag}.json"
    feats = json.loads(feature_cols_path.read_text(encoding="utf-8"))

    log.info(f"Loaded {len(X)} samples with {len(feats)} features")

    return X, feats, y


def prepare_all_data(cfg, output_dir: Path):
    """
    Prepare all training data with feature engineering and save to disk.

    Args:
        cfg: Configuration object
        output_dir: Directory to save preprocessed data
    """
    log.info("Starting full data preparation pipeline")

    log.info("Loading raw data frames...")
    train_idx, _test_idx, A, B, _ = load_frames(cfg.data_dir)

    log.info("Merging train data...")
    df = merge_train(train_idx, A, B)
    log.info(f"Total data loaded: {len(df)} rows")

    # Prepare data for each test type
    for flag in ("A", "B"):
        prepare_data_for_test(
            df.assign(Label=df[cfg.columns["target"]]),
            cfg,
            flag,
            output_dir=output_dir
        )

    log.info(f"Data preparation complete. Files saved to {output_dir}")
