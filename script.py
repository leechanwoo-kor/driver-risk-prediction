"""
Inference script for competition submission.
Reads from data/ and writes to output/submission.csv
"""
import json
import glob
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def seq_mean(series: pd.Series) -> pd.Series:
    """Calculate mean of comma-separated sequences."""
    def fast_mean(val):
        if pd.isna(val) or val == "":
            return np.nan
        try:
            arr = np.fromstring(val, sep=',', dtype=float)
            return arr.mean() if len(arr) > 0 else np.nan
        except:
            return np.nan
    return series.apply(fast_mean)


def seq_std(series: pd.Series) -> pd.Series:
    """Calculate std of comma-separated sequences."""
    def fast_std(val):
        if pd.isna(val) or val == "":
            return np.nan
        try:
            arr = np.fromstring(val, sep=',', dtype=float)
            return arr.std(ddof=1) if len(arr) > 1 else np.nan
        except:
            return np.nan
    return series.apply(fast_std)


def parse_age_bucket(v):
    """Parse age bucket (e.g., '20a' -> 22, '20b' -> 27)."""
    if not isinstance(v, str):
        return np.nan
    s = v.strip()
    if len(s) < 2 or not s[:-1].isdigit():
        return np.nan
    base, suf = int(s[:-1]), s[-1].lower()
    return base + (2 if suf == "a" else 7)


def engineer_features(df: pd.DataFrame, test_type: str) -> pd.DataFrame:
    """Engineer features for given test type."""
    feats = pd.DataFrame(index=df.index)

    # Define sequence columns by test type
    if test_type == "A":
        sequence_cols = [
            'A1-1', 'A1-2', 'A1-3', 'A1-4',
            'A2-1', 'A2-2', 'A2-3', 'A2-4',
            'A3-1', 'A3-2', 'A3-3', 'A3-4', 'A3-5', 'A3-6', 'A3-7',
            'A4-1', 'A4-2', 'A4-3', 'A4-4', 'A4-5',
            'A5-1', 'A5-2', 'A5-3',
        ]
    else:  # test_type == "B"
        sequence_cols = [
            'B1-1', 'B1-2', 'B1-3',
            'B2-1', 'B2-2', 'B2-3',
            'B3-1', 'B3-2',
            'B4-1', 'B4-2',
            'B5-1', 'B5-2',
            'B6', 'B7', 'B8',
        ]

    # Extract mean and std from sequences
    for col in sequence_cols:
        if col in df.columns:
            feats[f'{col}_mean'] = seq_mean(df[col])
            feats[f'{col}_std'] = seq_std(df[col])

    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats


def build_features(df: pd.DataFrame, test_type: str, impute_values: dict) -> pd.DataFrame:
    """Build complete feature set."""
    # Basic features
    if 'Age' in df.columns:
        df['Age_mid'] = df['Age'].apply(parse_age_bucket)

    # Sequence features
    seq_feats = engineer_features(df, test_type)

    # Get numeric columns (excluding sequences) - A6-A9, B9-B10 etc
    numeric_cols = []
    if test_type == "A":
        # A6, A7, A8, A9 are non-sequence columns
        for col in ['A6-1', 'A7-1', 'A8-1', 'A8-2', 'A9-1', 'A9-2', 'A9-3', 'A9-4', 'A9-5']:
            if col in df.columns:
                numeric_cols.append(col)
    else:  # test_type == "B"
        # B9, B10 are non-sequence columns
        for col in ['B9-1', 'B9-2', 'B9-3', 'B9-4', 'B9-5',
                    'B10-1', 'B10-2', 'B10-3', 'B10-4', 'B10-5', 'B10-6']:
            if col in df.columns:
                numeric_cols.append(col)

    # Combine all features
    basic_cols = ['Age_mid'] if 'Age_mid' in df.columns else []
    result = pd.concat([df[basic_cols + numeric_cols], seq_feats], axis=1)

    # Impute missing values
    for col, val in impute_values.items():
        if col in result.columns:
            result[col] = result[col].fillna(val)

    return result


def predict_test(test_df: pd.DataFrame, test_type: str, model_dir: Path) -> pd.Series:
    """Predict for one test type."""
    # Load feature columns and imputation values
    feature_cols = json.loads((model_dir / f"feature_cols_{test_type}.json").read_text())
    impute_values = json.loads((model_dir / f"impute_{test_type}.json").read_text())

    # Build features
    X = build_features(test_df, test_type, impute_values)
    X = X[feature_cols]

    # Load models and predict
    model_paths = sorted(glob.glob(str(model_dir / f"model_{test_type}_*.pkl")))
    preds = []
    for mp in model_paths:
        model = joblib.load(mp)
        p = model.predict_proba(X)
        if p.ndim == 2:
            p = p[:, 1]
        preds.append(p)

    avg = np.mean(preds, axis=0) if preds else np.zeros(len(X))

    # Apply calibration
    cal = joblib.load(model_dir / f"cal_{test_type}.pkl")
    p_final = cal.predict(avg)

    return pd.Series(np.clip(p_final, 1e-8, 1 - 1e-8), index=test_df.index)


def main():
    """Main inference function."""
    # Paths
    data_dir = Path("data")
    model_dir = Path("model")
    output_dir = Path("output")

    # Load test data
    test_idx = pd.read_csv(data_dir / "test.csv")
    test_A = pd.read_csv(data_dir / "test" / "A.csv")
    test_B = pd.read_csv(data_dir / "test" / "B.csv")

    # Merge test data
    test_ab = pd.concat([test_A, test_B], ignore_index=True)
    test_full = test_idx.merge(test_ab, on=["Test_id", "Test"], how="left")

    # Predict for each test type
    results = []
    for test_type in ["A", "B"]:
        part = test_full[test_full["Test"] == test_type].copy()
        if len(part) > 0:
            proba = predict_test(part, test_type, model_dir)
            results.append(pd.DataFrame({
                "Test_id": part["Test_id"].values,
                "Label": proba.values
            }))

    # Combine and save
    submission = pd.concat(results, ignore_index=True).sort_values("Test_id")
    output_dir.mkdir(exist_ok=True)
    submission.to_csv(output_dir / "submission.csv", index=False, encoding="utf-8")
    print(f"Submission saved: {len(submission)} predictions")


if __name__ == "__main__":
    main()
