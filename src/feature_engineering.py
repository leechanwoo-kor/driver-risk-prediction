"""
Optimized feature engineering - Fast vectorized sequence processing with parallel execution.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
from .utils.logging import get_logger

log = get_logger("feature_eng")


def seq_mean(series: pd.Series) -> pd.Series:
    """Calculate mean of comma-separated sequences (highly optimized)."""
    # Fast numpy-based approach
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
    """Calculate std of comma-separated sequences (highly optimized)."""
    # Fast numpy-based approach
    def fast_std(val):
        if pd.isna(val) or val == "":
            return np.nan
        try:
            arr = np.fromstring(val, sep=',', dtype=float)
            return arr.std(ddof=1) if len(arr) > 1 else np.nan
        except:
            return np.nan

    return series.apply(fast_std)


def _process_column_stats(col_name: str, series: pd.Series) -> tuple[str, pd.Series, pd.Series]:
    """Process a single column to extract mean and std (for parallel execution)."""
    mean_vals = seq_mean(series)
    std_vals = seq_std(series)
    return col_name, mean_vals, std_vals


def engineer_A_features(df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """Extract mean and std from ALL Test A sequence columns (parallelized)."""
    log.info("Engineering Test A features (optimized, parallel processing)...")

    # Use ALL sequence columns including Conditions
    sequence_cols = [
        # A1: All columns (Conditions + Response + ResponseTime)
        'A1-1', 'A1-2', 'A1-3', 'A1-4',
        # A2: All columns
        'A2-1', 'A2-2', 'A2-3', 'A2-4',
        # A3: All columns
        'A3-1', 'A3-2', 'A3-3', 'A3-4', 'A3-5', 'A3-6', 'A3-7',
        # A4: All columns
        'A4-1', 'A4-2', 'A4-3', 'A4-4', 'A4-5',
        # A5: All columns
        'A5-1', 'A5-2', 'A5-3',
    ]

    # Filter to existing columns
    existing_cols = [col for col in sequence_cols if col in df.columns]

    if not existing_cols:
        return pd.DataFrame(index=df.index)

    # Determine number of jobs
    if n_jobs == -1:
        n_jobs = min(cpu_count(), len(existing_cols))

    log.info(f"Processing {len(existing_cols)} columns with {n_jobs} workers...")

    # Parallel processing
    feats = pd.DataFrame(index=df.index)

    with Pool(processes=n_jobs) as pool:
        # Create tasks
        tasks = [(col, df[col]) for col in existing_cols]

        # Process in parallel
        results = pool.starmap(_process_column_stats, tasks)

        # Collect results
        for col_name, mean_vals, std_vals in results:
            feats[f'{col_name}_mean'] = mean_vals
            feats[f'{col_name}_std'] = std_vals

    log.info(f"Test A features: {len(feats.columns)} features")
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats


def engineer_B_features(df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """Extract mean and std from all Test B sequence columns (parallelized)."""
    log.info("Engineering Test B features (optimized, parallel processing)...")

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

    # Filter to existing columns
    existing_cols = [col for col in sequence_cols if col in df.columns]

    if not existing_cols:
        return pd.DataFrame(index=df.index)

    # Determine number of jobs
    if n_jobs == -1:
        n_jobs = min(cpu_count(), len(existing_cols))

    log.info(f"Processing {len(existing_cols)} columns with {n_jobs} workers...")

    # Parallel processing
    feats = pd.DataFrame(index=df.index)

    with Pool(processes=n_jobs) as pool:
        # Create tasks
        tasks = [(col, df[col]) for col in existing_cols]

        # Process in parallel
        results = pool.starmap(_process_column_stats, tasks)

        # Collect results
        for col_name, mean_vals, std_vals in results:
            feats[f'{col_name}_mean'] = mean_vals
            feats[f'{col_name}_std'] = std_vals

    log.info(f"Test B features: {len(feats.columns)} features")
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats
