"""
Advanced feature engineering for sequence-based test data.
Based on baseline approach with sequence parsing and aggregation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .utils.logging import get_logger

log = get_logger("feature_eng")


# ============================================================================
# Sequence Processing Utilities
# ============================================================================


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


def seq_rate(series: pd.Series, target: str = "1") -> pd.Series:
    """Calculate rate of specific value in comma-separated sequences."""
    return series.fillna("").apply(
        lambda x: (
            str(x).split(",").count(target) / len(x.split(",")) if x else np.nan
        )
    )


def masked_mean_from_csv_series(
    cond_series: pd.Series, val_series: pd.Series, mask_val
) -> pd.Series:
    """Calculate mean of values where condition equals mask_val."""
    cond_df = cond_series.fillna("").str.split(",", expand=True)
    val_df = val_series.fillna("").str.split(",", expand=True)
    # Convert to numeric directly (avoids FutureWarning)
    # Use future_stack=True for new implementation (dropna is not needed)
    cond_arr = pd.to_numeric(cond_df.stack(future_stack=True), errors="coerce").values.reshape(cond_df.shape)
    val_arr = pd.to_numeric(val_df.stack(future_stack=True), errors="coerce").values.reshape(val_df.shape)
    mask = cond_arr == mask_val
    with np.errstate(invalid="ignore"):
        sums = np.nansum(np.where(mask, val_arr, np.nan), axis=1)
        counts = np.sum(mask, axis=1)
        out = sums / np.where(counts == 0, np.nan, counts)
    return pd.Series(out, index=cond_series.index)


def masked_mean_in_set_series(
    cond_series: pd.Series, val_series: pd.Series, mask_set
) -> pd.Series:
    """Calculate mean of values where condition is in mask_set."""
    cond_df = cond_series.fillna("").str.split(",", expand=True)
    val_df = val_series.fillna("").str.split(",", expand=True)
    # Convert to numeric directly (avoids FutureWarning)
    # Use future_stack=True for new implementation (dropna is not needed)
    cond_arr = pd.to_numeric(cond_df.stack(future_stack=True), errors="coerce").values.reshape(cond_df.shape)
    val_arr = pd.to_numeric(val_df.stack(future_stack=True), errors="coerce").values.reshape(val_df.shape)
    mask = np.isin(cond_arr, list(mask_set))
    with np.errstate(invalid="ignore"):
        sums = np.nansum(np.where(mask, val_arr, np.nan), axis=1)
        counts = np.sum(mask, axis=1)
        out = sums / np.where(counts == 0, np.nan, counts)
    return pd.Series(out, index=cond_series.index)


# ============================================================================
# Test A Feature Engineering
# ============================================================================


def engineer_A_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate advanced features for Test A."""
    log.info("Starting A1 features...")
    feats = pd.DataFrame(index=df.index)

    # A1 features (반응 속도/정확도)
    if "A1-3" in df.columns and "A1-4" in df.columns:
        # Removed: A1_resp_rate (importance = 0.0106 < 0.01)
        feats["A1_rt_mean"] = seq_mean(df["A1-4"])
        feats["A1_rt_std"] = seq_std(df["A1-4"])

    if "A1-1" in df.columns and "A1-4" in df.columns:
        feats["A1_rt_left"] = masked_mean_from_csv_series(df["A1-1"], df["A1-4"], 1)
        feats["A1_rt_right"] = masked_mean_from_csv_series(df["A1-1"], df["A1-4"], 2)
        feats["A1_rt_side_diff"] = feats["A1_rt_left"] - feats["A1_rt_right"]

    if "A1-2" in df.columns and "A1-4" in df.columns:
        feats["A1_rt_slow"] = masked_mean_from_csv_series(df["A1-2"], df["A1-4"], 1)
        feats["A1_rt_fast"] = masked_mean_from_csv_series(df["A1-2"], df["A1-4"], 3)
        feats["A1_rt_speed_diff"] = feats["A1_rt_slow"] - feats["A1_rt_fast"]

    # A2 features
    log.info("Starting A2 features...")
    if "A2-3" in df.columns and "A2-4" in df.columns:
        feats["A2_resp_rate"] = seq_rate(df["A2-3"], "1")
        feats["A2_rt_mean"] = seq_mean(df["A2-4"])
        feats["A2_rt_std"] = seq_std(df["A2-4"])

    if "A2-1" in df.columns and "A2-4" in df.columns:
        feats["A2_rt_cond1_diff"] = masked_mean_from_csv_series(
            df["A2-1"], df["A2-4"], 1
        ) - masked_mean_from_csv_series(df["A2-1"], df["A2-4"], 3)

    if "A2-2" in df.columns and "A2-4" in df.columns:
        feats["A2_rt_cond2_diff"] = masked_mean_from_csv_series(
            df["A2-2"], df["A2-4"], 1
        ) - masked_mean_from_csv_series(df["A2-2"], df["A2-4"], 3)

    # A3 features (복잡한 시각 판단)
    log.info("Starting A3 features...")
    if "A3-5" in df.columns:
        s = df["A3-5"].fillna("")
        total = s.apply(lambda x: len(x.split(",")) if x else 0)
        correct = s.apply(
            lambda x: sum(v in {"1", "3"} for v in x.split(",")) if x else 0
        )
        feats["A3_correct_ratio"] = (correct / total).replace([np.inf, -np.inf], np.nan)
        # Removed: A3_valid_ratio, A3_invalid_ratio (importance = 0)

    if "A3-6" in df.columns and "A3-7" in df.columns:
        feats["A3_resp2_rate"] = seq_rate(df["A3-6"], "1")
        feats["A3_rt_mean"] = seq_mean(df["A3-7"])
        feats["A3_rt_std"] = seq_std(df["A3-7"])

    if "A3-1" in df.columns and "A3-7" in df.columns:
        feats["A3_rt_size_diff"] = masked_mean_from_csv_series(
            df["A3-1"], df["A3-7"], 1
        ) - masked_mean_from_csv_series(df["A3-1"], df["A3-7"], 2)

    if "A3-3" in df.columns and "A3-7" in df.columns:
        feats["A3_rt_side_diff"] = masked_mean_from_csv_series(
            df["A3-3"], df["A3-7"], 1
        ) - masked_mean_from_csv_series(df["A3-3"], df["A3-7"], 2)

    # A4 features (Stroop 효과)
    log.info("Starting A4 features...")
    if "A4-3" in df.columns and "A4-5" in df.columns:
        feats["A4_acc_rate"] = seq_rate(df["A4-3"], "1")
        # Removed: A4_resp2_rate (importance = 0)
        feats["A4_rt_mean"] = seq_mean(df["A4-5"])
        feats["A4_rt_std"] = seq_std(df["A4-5"])

    if "A4-1" in df.columns and "A4-5" in df.columns:
        feats["A4_stroop_diff"] = masked_mean_from_csv_series(
            df["A4-1"], df["A4-5"], 2
        ) - masked_mean_from_csv_series(df["A4-1"], df["A4-5"], 1)

    if "A4-2" in df.columns and "A4-5" in df.columns:
        feats["A4_rt_color_diff"] = masked_mean_from_csv_series(
            df["A4-2"], df["A4-5"], 1
        ) - masked_mean_from_csv_series(df["A4-2"], df["A4-5"], 2)

    # A5 features (변화 탐지)
    log.info("Starting A5 features...")
    if "A5-2" in df.columns:
        feats["A5_acc_rate"] = seq_rate(df["A5-2"], "1")
        # Removed: A5_resp2_rate (importance = 0.0105 < 0.01)

    if "A5-1" in df.columns and "A5-2" in df.columns:
        feats["A5_acc_nonchange"] = masked_mean_from_csv_series(
            df["A5-1"], df["A5-2"], 1
        )
        feats["A5_acc_change"] = masked_mean_in_set_series(
            df["A5-1"], df["A5-2"], {2, 3, 4}
        )

    # Higher-order features
    log.info("Creating higher-order features...")
    eps = 1e-6
    # Removed: A1_speed_acc_tradeoff (depends on removed A1_resp_rate)

    if "A2_rt_mean" in feats.columns and "A2_resp_rate" in feats.columns:
        feats["A2_speed_acc_tradeoff"] = feats["A2_rt_mean"] / (
            feats["A2_resp_rate"] + eps
        )

    if "A4_rt_mean" in feats.columns and "A4_acc_rate" in feats.columns:
        feats["A4_speed_acc_tradeoff"] = feats["A4_rt_mean"] / (
            feats["A4_acc_rate"] + eps
        )

    # RT coefficient of variation
    for k in ["A1", "A2", "A3", "A4"]:
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        if m in feats.columns and s in feats.columns:
            feats[f"{k}_rt_cv"] = feats[s] / (feats[m] + eps)

    # Absolute differences
    for name, base in [
        ("A1_rt_side_gap_abs", "A1_rt_side_diff"),
        ("A1_rt_speed_gap_abs", "A1_rt_speed_diff"),
        ("A2_rt_cond1_gap_abs", "A2_rt_cond1_diff"),
        ("A2_rt_cond2_gap_abs", "A2_rt_cond2_diff"),
        ("A4_stroop_gap_abs", "A4_stroop_diff"),
        ("A4_color_gap_abs", "A4_rt_color_diff"),
    ]:
        if base in feats.columns:
            feats[name] = feats[base].abs()

    # Removed: A3_valid_invalid_gap, A3_correct_invalid_gap (based on removed features)

    if "A5_acc_change" in feats.columns and "A5_acc_nonchange" in feats.columns:
        feats["A5_change_nonchange_gap"] = (
            feats["A5_acc_change"] - feats["A5_acc_nonchange"]
        )

    log.info(f"Test A feature engineering complete: {len(feats.columns)} features")
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats


# ============================================================================
# Test B Feature Engineering
# ============================================================================


def engineer_B_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate advanced features for Test B."""
    feats = pd.DataFrame(index=df.index)

    # B1 features
    if "B1-1" in df.columns and "B1-2" in df.columns and "B1-3" in df.columns:
        feats["B1_acc_task1"] = seq_rate(df["B1-1"], "1")
        feats["B1_rt_mean"] = seq_mean(df["B1-2"])
        feats["B1_rt_std"] = seq_std(df["B1-2"])
        feats["B1_acc_task2"] = seq_rate(df["B1-3"], "1")

    # B2 features
    if "B2-1" in df.columns and "B2-2" in df.columns and "B2-3" in df.columns:
        feats["B2_acc_task1"] = seq_rate(df["B2-1"], "1")
        feats["B2_rt_mean"] = seq_mean(df["B2-2"])
        feats["B2_rt_std"] = seq_std(df["B2-2"])
        feats["B2_acc_task2"] = seq_rate(df["B2-3"], "1")

    # B3 features
    if "B3-1" in df.columns and "B3-2" in df.columns:
        feats["B3_acc_rate"] = seq_rate(df["B3-1"], "1")
        feats["B3_rt_mean"] = seq_mean(df["B3-2"])
        feats["B3_rt_std"] = seq_std(df["B3-2"])

    # B4 features
    if "B4-1" in df.columns and "B4-2" in df.columns:
        feats["B4_acc_rate"] = seq_rate(df["B4-1"], "1")
        feats["B4_rt_mean"] = seq_mean(df["B4-2"])
        feats["B4_rt_std"] = seq_std(df["B4-2"])

    # B5 features
    if "B5-1" in df.columns and "B5-2" in df.columns:
        feats["B5_acc_rate"] = seq_rate(df["B5-1"], "1")
        feats["B5_rt_mean"] = seq_mean(df["B5-2"])
        feats["B5_rt_std"] = seq_std(df["B5-2"])

    # B6~B8 features (simple accuracy)
    if "B6" in df.columns:
        feats["B6_acc_rate"] = seq_rate(df["B6"], "1")
    if "B7" in df.columns:
        feats["B7_acc_rate"] = seq_rate(df["B7"], "1")
    if "B8" in df.columns:
        feats["B8_acc_rate"] = seq_rate(df["B8"], "1")

    # B9 features (Signal Detection Theory + Visual Error)
    if all(f"B9-{i}" in df.columns for i in range(1, 6)):
        # Auditory task metrics
        hit = df["B9-1"]
        miss = df["B9-2"]
        fa = df["B9-3"]
        cr = df["B9-4"]
        
        # Signal Detection Theory metrics
        total_signal = hit + miss
        total_noise = fa + cr
        # Removed individual rates (importance = 0), keeping only derived metrics
        
        # Calculate rates for dprime
        hit_rate = hit / (total_signal + 1e-8)
        fa_rate = fa / (total_noise + 1e-8)
        
        # d-prime (sensitivity index): Z(hit_rate) - Z(fa_rate)
        # Approximation: higher is better detection
        feats["B9_dprime_approx"] = hit_rate - fa_rate
        
        # Criterion (bias): -0.5 * (Z(hit_rate) + Z(fa_rate))
        # Response bias: negative = liberal, positive = conservative
        feats["B9_criterion_approx"] = -0.5 * (hit_rate + fa_rate)
        
        # Overall auditory accuracy
        feats["B9_aud_acc"] = (hit + cr) / (total_signal + total_noise + 1e-8)
        
        # Visual task error
        feats["B9_vis_err"] = df["B9-5"]
        feats["B9_vis_err_rate"] = df["B9-5"] / 32.0  # 32 trials per spec

    # B10 features (Signal Detection Theory + Dual Visual Tasks)
    if all(f"B10-{i}" in df.columns for i in range(1, 7)):
        # Auditory task metrics
        hit = df["B10-1"]
        miss = df["B10-2"]
        fa = df["B10-3"]
        cr = df["B10-4"]
        
        # Signal Detection Theory metrics
        total_signal = hit + miss
        total_noise = fa + cr
        # Removed individual rates (importance = 0), keeping only derived metrics
        
        # Calculate rates for dprime
        hit_rate = hit / (total_signal + 1e-8)
        fa_rate = fa / (total_noise + 1e-8)
        
        # d-prime and criterion
        feats["B10_dprime_approx"] = hit_rate - fa_rate
        feats["B10_criterion_approx"] = -0.5 * (hit_rate + fa_rate)
        feats["B10_fa_rate"] = fa_rate  # This one is used (0.018243 importance)
        
        # Overall auditory accuracy
        feats["B10_aud_acc"] = (hit + cr) / (total_signal + total_noise + 1e-8)
        
        # Visual task 1 (obstacle avoidance) error
        feats["B10_vis1_err"] = df["B10-5"]
        feats["B10_vis1_err_rate"] = df["B10-5"] / 52.0  # 52 trials
        
        # Visual task 2 (color selection) - raw count only
        feats["B10_vis2_right"] = df["B10-6"]
        # Removed: B10_vis2_acc (importance = 0)
        # Removed: B10_multitask_score (based on removed feature)

    # Cross-test comparisons (B9 vs B10)
    if "B9_dprime_approx" in feats.columns and "B10_dprime_approx" in feats.columns:
        feats["B9_B10_dprime_diff"] = feats["B9_dprime_approx"] - feats["B10_dprime_approx"]
        # Removed: B9_B10_hit_diff, B9_B10_fa_diff (based on removed features)

    # Higher-order features
    eps = 1e-6
    for k, acc_col, rt_col in [
        ("B1", "B1_acc_task1", "B1_rt_mean"),
        ("B2", "B2_acc_task1", "B2_rt_mean"),
        ("B3", "B3_acc_rate", "B3_rt_mean"),
        ("B4", "B4_acc_rate", "B4_rt_mean"),
        ("B5", "B5_acc_rate", "B5_rt_mean"),
    ]:
        if rt_col in feats.columns and acc_col in feats.columns:
            feats[f"{k}_speed_acc_tradeoff"] = feats[rt_col] / (feats[acc_col] + eps)

    # RT coefficient of variation
    for k in ["B1", "B2", "B3", "B4", "B5"]:
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        if m in feats.columns and s in feats.columns:
            feats[f"{k}_rt_cv"] = feats[s] / (feats[m] + eps)

    # Risk score (composite metric)
    parts = []
    for k in ["B4", "B5"]:
        cv = f"{k}_rt_cv"
        if cv in feats.columns:
            parts.append(0.25 * feats[cv].fillna(0))

    for k in ["B3", "B4", "B5"]:
        acc = f"{k}_acc_rate"
        if acc in feats.columns:
            parts.append(0.25 * (1 - feats[acc].fillna(0)))

    for k in ["B1", "B2"]:
        tcol = f"{k}_speed_acc_tradeoff"
        if tcol in feats.columns:
            parts.append(0.25 * feats[tcol].fillna(0))

    if parts:
        feats["RiskScore_B"] = sum(parts)

    log.info(f"Test B feature engineering complete: {len(feats.columns)} features")
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats
