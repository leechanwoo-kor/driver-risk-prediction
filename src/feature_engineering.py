"""
Simple feature engineering - Extract mean and std from all sequence columns.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .utils.logging import get_logger

log = get_logger("feature_eng")


def seq_mean(series: pd.Series) -> pd.Series:
    """Calculate mean of comma-separated numeric sequences."""
    return series.fillna("").apply(
        lambda x: np.fromstring(x, sep=",").mean() if x else np.nan
    )


def seq_std(series: pd.Series) -> pd.Series:
    """Calculate std of comma-separated numeric sequences."""
    return series.fillna("").apply(
        lambda x: np.fromstring(x, sep=",").std() if x else np.nan
    )


def engineer_A_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract mean and std from important Test A sequence columns only."""
    log.info("Engineering Test A features (mean + std, filtered by importance)...")
    feats = pd.DataFrame(index=df.index)

    # Only use Response and ResponseTime columns (Condition columns have importance=0)
    sequence_cols = [
        # A1: Response and ResponseTime only (skip A1-1, A1-2 conditions)
        'A1-3', 'A1-4',
        # A2: Response and ResponseTime only (skip A2-1, A2-2 conditions)
        'A2-3', 'A2-4',
        # A3: Skip A3-1~4 conditions (importance=0), keep A3-5, A3-6, A3-7
        'A3-5', 'A3-6', 'A3-7',
        # A4: Skip A4-1, A4-2 conditions (importance=0), keep A4-3, A4-4, A4-5
        'A4-3', 'A4-4', 'A4-5',
        # A5: All have importance > 0
        'A5-1', 'A5-2', 'A5-3',
    ]

    for col in sequence_cols:
        if col in df.columns:
            feats[f'{col}_mean'] = seq_mean(df[col])
            feats[f'{col}_std'] = seq_std(df[col])

    log.info(f"Test A features: {len(feats.columns)} features")
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats


def engineer_B_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract mean and std from all Test B sequence columns."""
    log.info("Engineering Test B features (mean + std only)...")
    feats = pd.DataFrame(index=df.index)

    # Define all sequence columns for Test B
    sequence_cols = [
        # B1, B2, B3, B4, B5
        'B1-1', 'B1-2', 'B1-3',
        'B2-1', 'B2-2', 'B2-3',
        'B3-1', 'B3-2',
        'B4-1', 'B4-2',
        'B5-1', 'B5-2',
        # B6, B7, B8 are simple accuracy columns (comma-separated)
        'B6', 'B7', 'B8',
    ]

    for col in sequence_cols:
        if col in df.columns:
            feats[f'{col}_mean'] = seq_mean(df[col])
            feats[f'{col}_std'] = seq_std(df[col])

    log.info(f"Test B features: {len(feats.columns)} features")
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats
