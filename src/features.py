from __future__ import annotations
from pathlib import Path
import json, numpy as np, pandas as pd
from .feature_engineering import engineer_A_features, engineer_B_features
from .utils.logging import get_logger

log = get_logger("features")

ID_COLS = {"Test_id", "Test", "PrimaryKey"}
META_COLS = {"Age", "TestDate"}
# Sequence columns to drop after feature engineering
SEQ_COLS_A = [
    "A1-1", "A1-2", "A1-3", "A1-4",
    "A2-1", "A2-2", "A2-3", "A2-4",
    "A3-1", "A3-2", "A3-3", "A3-4", "A3-5", "A3-6", "A3-7",
    "A4-1", "A4-2", "A4-3", "A4-4", "A4-5",
    "A5-1", "A5-2", "A5-3",
]
SEQ_COLS_B = [
    "B1-1", "B1-2", "B1-3",
    "B2-1", "B2-2", "B2-3",
    "B3-1", "B3-2",
    "B4-1", "B4-2",
    "B5-1", "B5-2",
    "B6", "B7", "B8",
]


def parse_age_bucket(v):
    if not isinstance(v, str):
        return np.nan
    s = v.strip()
    if len(s) < 2 or not s[:-1].isdigit():
        return np.nan
    base, suf = int(s[:-1]), s[-1].lower()
    return base + (2 if suf == "a" else 7)


def expand_date(df: pd.DataFrame, col: str) -> pd.DataFrame:
    # TEMPORAL FEATURES REMOVED: TestDate_year/month cause distribution shift
    # Baseline (Public LB 0.19067) doesn't use temporal features
    # Train: 2018-2022, Test: 2023 (out-of-distribution)
    return df


def build_features(
    df: pd.DataFrame, use_cols_json: Path | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build features with advanced sequence-based engineering.
    Detects Test type from 'Test' column or columns present.
    """
    out = df.copy()

    # Basic derived features
    log.info("Creating basic derived features...")
    if "Age" in out.columns:
        out["Age_mid"] = out["Age"].map(parse_age_bucket)
    out = expand_date(out, "TestDate")
    # Note: TestDate_q removed (importance = 0.0105 < 0.01)

    # Advanced sequence-based features
    # Determine test type: prefer 'Test' column, fallback to column detection
    log.info("Detecting test type...")
    test_type = None
    if "Test" in out.columns:
        # If Test column exists, check unique values
        unique_tests = out["Test"].unique()
        if len(unique_tests) == 1:
            test_type = unique_tests[0]

    # Fallback: detect by column presence
    if test_type is None:
        if any(col in out.columns for col in SEQ_COLS_A) and not any(col in out.columns for col in SEQ_COLS_B):
            test_type = "A"
        elif any(col in out.columns for col in SEQ_COLS_B) and not any(col in out.columns for col in SEQ_COLS_A):
            test_type = "B"

    # Apply test-specific feature engineering
    if test_type == "A":
        log.info("Engineering Test A sequence features...")
        advanced_feats = engineer_A_features(out)
        log.info(f"Created {len(advanced_feats.columns)} advanced features")
        out = pd.concat([out, advanced_feats], axis=1)
        out = out.drop(columns=[c for c in SEQ_COLS_A if c in out.columns])
    elif test_type == "B":
        log.info("Engineering Test B sequence features...")
        advanced_feats = engineer_B_features(out)
        log.info(f"Created {len(advanced_feats.columns)} advanced features")
        out = pd.concat([out, advanced_feats], axis=1)
        out = out.drop(columns=[c for c in SEQ_COLS_B if c in out.columns])

    # Temporal features (REMOVED: causes distribution shift)
    # log.info("Creating temporal features...")
    # if "TestDate_year" in out.columns and "TestDate_month" in out.columns:
    #     out["YearMonthIndex"] = out["TestDate_year"] * 12 + out["TestDate_month"]

    # Feature selection
    log.info("Selecting and finalizing features...")
    if use_cols_json and use_cols_json.exists():
        feats = json.loads(use_cols_json.read_text(encoding="utf-8"))
        feats = [c for c in feats if c in out.columns]
    else:
        excl = ID_COLS | META_COLS | {"Label"}
        feats = [
            c
            for c in out.columns
            if c not in excl and pd.api.types.is_numeric_dtype(out[c])
        ]
        
        # Remove cross-contamination: exclude opposite test's raw columns
        if test_type == "A":
            # Remove B raw columns from Test A features
            b_raw_cols = [f"B{i}-{j}" for i in range(1, 11) for j in range(1, 10)]
            b_raw_cols.extend(["B6", "B7", "B8"])
            feats = [f for f in feats if f not in b_raw_cols]
        elif test_type == "B":
            # Remove A raw columns from Test B features
            a_raw_cols = [f"A{i}-{j}" for i in range(1, 10) for j in range(1, 10)]
            feats = [f for f in feats if f not in a_raw_cols]

    # Missing value imputation (median)
    if feats:
        out[feats] = out[feats].fillna(out[feats].median())

    return out, feats
