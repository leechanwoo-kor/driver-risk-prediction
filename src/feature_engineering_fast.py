"""
Fast vectorized feature engineering - No apply(), pure numpy operations.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .utils.logging import get_logger

log = get_logger("feature_eng_fast")


def parse_sequences_vectorized(series: pd.Series, max_len: int | None = None) -> np.ndarray:
    """
    Parse all comma-separated sequences at once into a 2D numpy array.

    Returns:
        np.ndarray of shape (n_rows, max_len) with NaN for missing/shorter sequences
    """
    def parse_row(val):
        if pd.isna(val) or val == "":
            return None
        try:
            return np.fromstring(val, sep=',', dtype=float)
        except:
            return None

    # Parse all rows
    parsed = [parse_row(val) for val in series]

    # Find max length if not provided
    if max_len is None:
        max_len = max((len(arr) if arr is not None else 0 for arr in parsed), default=0)

    if max_len == 0:
        return np.full((len(series), 1), np.nan)

    # Create 2D array with NaN padding
    result = np.full((len(series), max_len), np.nan, dtype=float)
    for i, arr in enumerate(parsed):
        if arr is not None and len(arr) > 0:
            length = min(len(arr), max_len)
            result[i, :length] = arr[:length]

    return result


def engineer_A_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Fast vectorized feature engineering for Test A."""
    log.info("Engineering Test A features (vectorized, fast)...")

    feats = pd.DataFrame(index=df.index)
    n = len(df)

    # === A1: 속도예측 검사 (18 trials) ===
    if 'A1-3' in df.columns and 'A1-4' in df.columns:
        a1_resp = parse_sequences_vectorized(df['A1-3'], 18)  # (n, 18)
        a1_rt = parse_sequences_vectorized(df['A1-4'], 18)

        feats['A1_response_rate'] = np.nanmean(a1_resp, axis=1)
        feats['A1_mean_rt'] = np.nanmean(a1_rt, axis=1)
        feats['A1_std_rt'] = np.nanstd(a1_rt, axis=1)
        feats['A1_max_rt'] = np.nanmax(a1_rt, axis=1)

    # === A2: 주의전환 검사 (18 trials) ===
    if 'A2-3' in df.columns and 'A2-4' in df.columns:
        a2_resp = parse_sequences_vectorized(df['A2-3'], 18)
        a2_rt = parse_sequences_vectorized(df['A2-4'], 18)

        feats['A2_response_rate'] = np.nanmean(a2_resp, axis=1)
        feats['A2_mean_rt'] = np.nanmean(a2_rt, axis=1)
        feats['A2_std_rt'] = np.nanstd(a2_rt, axis=1)
        feats['A2_max_rt'] = np.nanmax(a2_rt, axis=1)

    # === A3: 주의집중 검사 (32 trials) ===
    if 'A3-5' in df.columns and 'A3-7' in df.columns:
        a3_resp = parse_sequences_vectorized(df['A3-5'], 32)  # (n, 32)
        a3_rt = parse_sequences_vectorized(df['A3-7'], 32)

        # Accuracy features (vectorized)
        # correct if response in [1, 3]
        correct_mask = np.isin(a3_resp, [1, 3])
        feats['A3_correct_rate'] = np.nanmean(correct_mask.astype(float), axis=1)

        # valid_correct: response == 1
        feats['A3_valid_correct_rate'] = np.nanmean((a3_resp == 1).astype(float), axis=1)

        # invalid: response in [3, 4]
        invalid_mask = np.isin(a3_resp, [3, 4])
        feats['A3_invalid_rate'] = np.nanmean(invalid_mask.astype(float), axis=1)

        # Response time features (vectorized)
        feats['A3_mean_rt'] = np.nanmean(a3_rt, axis=1)
        feats['A3_std_rt'] = np.nanstd(a3_rt, axis=1)
        feats['A3_min_rt'] = np.nanmin(a3_rt, axis=1)
        feats['A3_max_rt'] = np.nanmax(a3_rt, axis=1)

        # Slow trial ratio (>1000ms)
        feats['A3_slow_trial_ratio'] = np.nanmean((a3_rt > 1000).astype(float), axis=1)

        # Fatigue: early (0:16) vs late (16:32) RT
        early_rt = np.nanmean(a3_rt[:, :16], axis=1)
        late_rt = np.nanmean(a3_rt[:, 16:], axis=1)
        feats['A3_fatigue_rt'] = late_rt - early_rt

    # === A4: 색상 주의집중 검사 (80 trials) ===
    if 'A4-1' in df.columns and 'A4-3' in df.columns and 'A4-5' in df.columns:
        a4_cond = parse_sequences_vectorized(df['A4-1'], 80)  # (n, 80)
        a4_resp = parse_sequences_vectorized(df['A4-3'], 80)
        a4_rt = parse_sequences_vectorized(df['A4-5'], 80)

        # Overall accuracy (response == 1)
        feats['A4_correct_rate'] = np.nanmean((a4_resp == 1).astype(float), axis=1)

        # Congruent (cond==1) accuracy
        congruent_mask = (a4_cond == 1)
        congruent_correct = np.where(congruent_mask, (a4_resp == 1).astype(float), np.nan)
        feats['A4_congruent_acc'] = np.nanmean(congruent_correct, axis=1)

        # Incongruent (cond==2) accuracy
        incongruent_mask = (a4_cond == 2)
        incongruent_correct = np.where(incongruent_mask, (a4_resp == 1).astype(float), np.nan)
        feats['A4_incongruent_acc'] = np.nanmean(incongruent_correct, axis=1)

        # Interference effect
        feats['A4_interference_effect'] = feats['A4_congruent_acc'] - feats['A4_incongruent_acc']

        # Response time features
        feats['A4_mean_rt'] = np.nanmean(a4_rt, axis=1)
        feats['A4_std_rt'] = np.nanstd(a4_rt, axis=1)
        feats['A4_max_rt'] = np.nanmax(a4_rt, axis=1)

        # Congruent vs Incongruent RT
        congruent_rt = np.where(congruent_mask, a4_rt, np.nan)
        incongruent_rt = np.where(incongruent_mask, a4_rt, np.nan)
        feats['A4_congruent_rt'] = np.nanmean(congruent_rt, axis=1)
        feats['A4_incongruent_rt'] = np.nanmean(incongruent_rt, axis=1)

        # Slow trial ratio
        feats['A4_slow_trial_ratio'] = np.nanmean((a4_rt > 1000).astype(float), axis=1)

    # === A5: 기억/변화탐지 검사 (36 trials) ===
    if 'A5-1' in df.columns and 'A5-2' in df.columns:
        a5_cond = parse_sequences_vectorized(df['A5-1'], 36)  # (n, 36)
        a5_resp = parse_sequences_vectorized(df['A5-2'], 36)

        # Overall accuracy (response == 1)
        feats['A5_correct_rate'] = np.nanmean((a5_resp == 1).astype(float), axis=1)

        # Accuracy by change type
        for change_type, change_name in [(1, 'no_change'), (2, 'pos_change'),
                                          (3, 'color_change'), (4, 'shape_change')]:
            mask = (a5_cond == change_type)
            correct = np.where(mask, (a5_resp == 1).astype(float), np.nan)
            feats[f'A5_{change_name}_acc'] = np.nanmean(correct, axis=1)

    log.info(f"Test A features: {len(feats.columns)} features (vectorized)")
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats


def engineer_B_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Fast vectorized feature engineering for Test B."""
    log.info("Engineering Test B features (vectorized, fast)...")

    feats = pd.DataFrame(index=df.index)

    # === B1 검사 (16 trials) ===
    if 'B1-1' in df.columns and 'B1-2' in df.columns and 'B1-3' in df.columns:
        b1_resp1 = parse_sequences_vectorized(df['B1-1'], 16)
        b1_rt = parse_sequences_vectorized(df['B1-2'], 16)
        b1_resp2 = parse_sequences_vectorized(df['B1-3'], 16)

        # Response1 accuracy (resp1 == 1)
        feats['B1_resp1_acc'] = np.nanmean((b1_resp1 == 1).astype(float), axis=1)

        # Response2 correct (resp2 in [1, 3])
        correct_mask = np.isin(b1_resp2, [1, 3])
        feats['B1_resp2_correct_rate'] = np.nanmean(correct_mask.astype(float), axis=1)

        feats['B1_mean_rt'] = np.nanmean(b1_rt, axis=1)
        feats['B1_std_rt'] = np.nanstd(b1_rt, axis=1)

    # === B2 검사 (16 trials) ===
    if 'B2-1' in df.columns and 'B2-2' in df.columns and 'B2-3' in df.columns:
        b2_resp1 = parse_sequences_vectorized(df['B2-1'], 16)
        b2_rt = parse_sequences_vectorized(df['B2-2'], 16)
        b2_resp2 = parse_sequences_vectorized(df['B2-3'], 16)

        feats['B2_resp1_acc'] = np.nanmean((b2_resp1 == 1).astype(float), axis=1)

        correct_mask = np.isin(b2_resp2, [1, 3])
        feats['B2_resp2_correct_rate'] = np.nanmean(correct_mask.astype(float), axis=1)

        feats['B2_mean_rt'] = np.nanmean(b2_rt, axis=1)
        feats['B2_std_rt'] = np.nanstd(b2_rt, axis=1)

    # === B3 검사 (15 trials) ===
    if 'B3-1' in df.columns and 'B3-2' in df.columns:
        b3_resp = parse_sequences_vectorized(df['B3-1'], 15)
        b3_rt = parse_sequences_vectorized(df['B3-2'], 15)

        feats['B3_correct_rate'] = np.nanmean((b3_resp == 1).astype(float), axis=1)
        feats['B3_mean_rt'] = np.nanmean(b3_rt, axis=1)
        feats['B3_std_rt'] = np.nanstd(b3_rt, axis=1)

    # === B4 검사 (60 trials) ===
    if 'B4-1' in df.columns and 'B4-2' in df.columns:
        b4_resp = parse_sequences_vectorized(df['B4-1'], 60)
        b4_rt = parse_sequences_vectorized(df['B4-2'], 60)

        # Correct if resp in [1, 3, 5]
        correct_mask = np.isin(b4_resp, [1, 3, 5])
        feats['B4_correct_rate'] = np.nanmean(correct_mask.astype(float), axis=1)

        feats['B4_mean_rt'] = np.nanmean(b4_rt, axis=1)
        feats['B4_std_rt'] = np.nanstd(b4_rt, axis=1)
        feats['B4_max_rt'] = np.nanmax(b4_rt, axis=1)

    # === B5 검사 (20 trials) ===
    if 'B5-1' in df.columns and 'B5-2' in df.columns:
        b5_resp = parse_sequences_vectorized(df['B5-1'], 20)
        b5_rt = parse_sequences_vectorized(df['B5-2'], 20)

        feats['B5_correct_rate'] = np.nanmean((b5_resp == 1).astype(float), axis=1)
        feats['B5_mean_rt'] = np.nanmean(b5_rt, axis=1)
        feats['B5_std_rt'] = np.nanstd(b5_rt, axis=1)

    # === B6, B7, B8 검사 (response only) ===
    for test_num, max_len in [(6, 15), (7, 15), (8, 12)]:
        col = f'B{test_num}'
        if col in df.columns:
            b_resp = parse_sequences_vectorized(df[col], max_len)
            feats[f'B{test_num}_correct_rate'] = np.nanmean((b_resp == 1).astype(float), axis=1)

    log.info(f"Test B features: {len(feats.columns)} features (vectorized)")
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats
