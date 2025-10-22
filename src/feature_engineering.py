"""
Advanced feature engineering - Extract meaningful features from aptitude test responses.

Key insight:
- Condition columns (A1-1, A3-1, A4-1, etc.) are test conditions, not responses
- Response columns (A3-5, A4-3, etc.) contain accuracy information
- ResponseTime columns (A3-7, A4-5, etc.) contain timing information

We extract features from RESPONSES and TIMES, not conditions.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from .utils.logging import get_logger

log = get_logger("feature_eng")


def parse_sequence(val) -> np.ndarray | None:
    """Parse comma-separated sequence to numpy array."""
    if pd.isna(val) or val == "":
        return None
    try:
        return np.fromstring(val, sep=',', dtype=float)
    except:
        return None


def engineer_A_features(df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """
    Extract meaningful features from Test A (신규 자격 검사).

    Test structure:
    - A1 (18 trials): 속도예측 - Response(A1-3), ResponseTime(A1-4)
    - A2 (18 trials): 주의전환 - Response(A2-3), ResponseTime(A2-4)
    - A3 (32 trials): 주의집중 - Response1(A3-5), Response2(A3-6), ResponseTime(A3-7)
    - A4 (80 trials): 색상 주의집중 - Response1(A4-3), Response2(A4-4), ResponseTime(A4-5)
    - A5 (36 trials): 기억/변화탐지 - Response1(A5-2), Response2(A5-3)
    - A6, A7: 문제풀이 (단일값)
    - A8, A9: 질문지 (단일값)
    """
    log.info("Engineering Test A features (advanced, response-based)...")

    feats = pd.DataFrame(index=df.index)

    # === A1: 속도예측 검사 ===
    if 'A1-3' in df.columns and 'A1-4' in df.columns:
        # Response (0=N, 1=Y)
        a1_resp = df['A1-3'].apply(parse_sequence)
        a1_rt = df['A1-4'].apply(parse_sequence)

        feats['A1_response_rate'] = a1_resp.apply(lambda x: np.mean(x) if x is not None and len(x) > 0 else np.nan)
        feats['A1_mean_rt'] = a1_rt.apply(lambda x: np.mean(x) if x is not None and len(x) > 0 else np.nan)
        feats['A1_std_rt'] = a1_rt.apply(lambda x: np.std(x) if x is not None and len(x) > 1 else np.nan)
        feats['A1_max_rt'] = a1_rt.apply(lambda x: np.max(x) if x is not None and len(x) > 0 else np.nan)

    # === A2: 주의전환 검사 ===
    if 'A2-3' in df.columns and 'A2-4' in df.columns:
        a2_resp = df['A2-3'].apply(parse_sequence)
        a2_rt = df['A2-4'].apply(parse_sequence)

        feats['A2_response_rate'] = a2_resp.apply(lambda x: np.mean(x) if x is not None and len(x) > 0 else np.nan)
        feats['A2_mean_rt'] = a2_rt.apply(lambda x: np.mean(x) if x is not None and len(x) > 0 else np.nan)
        feats['A2_std_rt'] = a2_rt.apply(lambda x: np.std(x) if x is not None and len(x) > 1 else np.nan)
        feats['A2_max_rt'] = a2_rt.apply(lambda x: np.max(x) if x is not None and len(x) > 0 else np.nan)

    # === A3: 주의집중 검사 (32 trials) ===
    if 'A3-5' in df.columns and 'A3-7' in df.columns:
        # Response1: 1=valid correct, 2=valid incorrect, 3=invalid correct, 4=invalid incorrect
        a3_resp1 = df['A3-5'].apply(parse_sequence)
        a3_rt = df['A3-7'].apply(parse_sequence)

        # Accuracy features
        feats['A3_correct_rate'] = a3_resp1.apply(
            lambda x: np.mean([1 if v in [1, 3] else 0 for v in x]) if x is not None and len(x) > 0 else np.nan
        )
        feats['A3_valid_correct_rate'] = a3_resp1.apply(
            lambda x: (x == 1).sum() / len(x) if x is not None and len(x) > 0 else np.nan
        )
        feats['A3_invalid_rate'] = a3_resp1.apply(
            lambda x: np.mean([1 if v in [3, 4] else 0 for v in x]) if x is not None and len(x) > 0 else np.nan
        )

        # Response time features
        feats['A3_mean_rt'] = a3_rt.apply(lambda x: np.mean(x) if x is not None and len(x) > 0 else np.nan)
        feats['A3_std_rt'] = a3_rt.apply(lambda x: np.std(x) if x is not None and len(x) > 1 else np.nan)
        feats['A3_min_rt'] = a3_rt.apply(lambda x: np.min(x) if x is not None and len(x) > 0 else np.nan)
        feats['A3_max_rt'] = a3_rt.apply(lambda x: np.max(x) if x is not None and len(x) > 0 else np.nan)

        # Slow trial ratio (>1000ms threshold)
        feats['A3_slow_trial_ratio'] = a3_rt.apply(
            lambda x: np.mean(x > 1000) if x is not None and len(x) > 0 else np.nan
        )

        # Fatigue indicator: early vs late performance
        def early_late_rt_diff(arr):
            if arr is None or len(arr) < 16:
                return np.nan
            early = arr[:16].mean()
            late = arr[16:].mean()
            return late - early  # Positive = getting slower (fatigue)

        feats['A3_fatigue_rt'] = a3_rt.apply(early_late_rt_diff)

    # === A4: 색상 주의집중 검사 (80 trials) ===
    if 'A4-1' in df.columns and 'A4-3' in df.columns and 'A4-5' in df.columns:
        # Condition: 1=congruent, 2=incongruent
        a4_cond = df['A4-1'].apply(parse_sequence)
        # Response1: 1=correct, 2=incorrect
        a4_resp1 = df['A4-3'].apply(parse_sequence)
        a4_rt = df['A4-5'].apply(parse_sequence)

        # Overall accuracy
        feats['A4_correct_rate'] = a4_resp1.apply(
            lambda x: np.mean(x == 1) if x is not None and len(x) > 0 else np.nan
        )

        # Congruent vs Incongruent accuracy
        def congruent_acc(cond, resp):
            if cond is None or resp is None or len(cond) != len(resp):
                return np.nan
            mask = cond == 1
            if mask.sum() == 0:
                return np.nan
            return (resp[mask] == 1).mean()

        def incongruent_acc(cond, resp):
            if cond is None or resp is None or len(cond) != len(resp):
                return np.nan
            mask = cond == 2
            if mask.sum() == 0:
                return np.nan
            return (resp[mask] == 1).mean()

        feats['A4_congruent_acc'] = [congruent_acc(c, r) for c, r in zip(a4_cond, a4_resp1)]
        feats['A4_incongruent_acc'] = [incongruent_acc(c, r) for c, r in zip(a4_cond, a4_resp1)]

        # Interference effect (congruent - incongruent)
        feats['A4_interference_effect'] = feats['A4_congruent_acc'] - feats['A4_incongruent_acc']

        # Response time features
        feats['A4_mean_rt'] = a4_rt.apply(lambda x: np.mean(x) if x is not None and len(x) > 0 else np.nan)
        feats['A4_std_rt'] = a4_rt.apply(lambda x: np.std(x) if x is not None and len(x) > 1 else np.nan)
        feats['A4_max_rt'] = a4_rt.apply(lambda x: np.max(x) if x is not None and len(x) > 0 else np.nan)

        # Congruent vs Incongruent RT
        def congruent_rt(cond, rt):
            if cond is None or rt is None or len(cond) != len(rt):
                return np.nan
            mask = cond == 1
            if mask.sum() == 0:
                return np.nan
            return rt[mask].mean()

        def incongruent_rt(cond, rt):
            if cond is None or rt is None or len(cond) != len(rt):
                return np.nan
            mask = cond == 2
            if mask.sum() == 0:
                return np.nan
            return rt[mask].mean()

        feats['A4_congruent_rt'] = [congruent_rt(c, r) for c, r in zip(a4_cond, a4_rt)]
        feats['A4_incongruent_rt'] = [incongruent_rt(c, r) for c, r in zip(a4_cond, a4_rt)]

        # Slow trial ratio
        feats['A4_slow_trial_ratio'] = a4_rt.apply(
            lambda x: np.mean(x > 1000) if x is not None and len(x) > 0 else np.nan
        )

    # === A5: 기억/변화탐지 검사 (36 trials) ===
    if 'A5-1' in df.columns and 'A5-2' in df.columns:
        # Condition: 1=non change, 2=pos change, 3=color change, 4=shape change
        a5_cond = df['A5-1'].apply(parse_sequence)
        # Response1: 1=correct, 2=incorrect
        a5_resp = df['A5-2'].apply(parse_sequence)

        # Overall accuracy
        feats['A5_correct_rate'] = a5_resp.apply(
            lambda x: np.mean(x == 1) if x is not None and len(x) > 0 else np.nan
        )

        # Accuracy by change type
        def change_type_acc(cond, resp, change_type):
            if cond is None or resp is None or len(cond) != len(resp):
                return np.nan
            mask = cond == change_type
            if mask.sum() == 0:
                return np.nan
            return (resp[mask] == 1).mean()

        feats['A5_no_change_acc'] = [change_type_acc(c, r, 1) for c, r in zip(a5_cond, a5_resp)]
        feats['A5_pos_change_acc'] = [change_type_acc(c, r, 2) for c, r in zip(a5_cond, a5_resp)]
        feats['A5_color_change_acc'] = [change_type_acc(c, r, 3) for c, r in zip(a5_cond, a5_resp)]
        feats['A5_shape_change_acc'] = [change_type_acc(c, r, 4) for c, r in zip(a5_cond, a5_resp)]

    log.info(f"Test A features: {len(feats.columns)} features")
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats


def engineer_B_features(df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """
    Extract meaningful features from Test B (자격 유지 검사).

    Test structure:
    - B1, B2 (16 trials each): Response1(B1-1), ResponseTime(B1-2), Response2(B1-3)
    - B3 (15 trials): Response(B3-1), ResponseTime(B3-2)
    - B4 (60 trials): Response(B4-1), ResponseTime(B4-2)
    - B5 (20 trials): Response(B5-1), ResponseTime(B5-2)
    - B6, B7, B8 (15/15/12 trials): Response only
    - B9, B10: 단일값 (hit/miss/fa/cr/err counts)
    """
    log.info("Engineering Test B features (advanced, response-based)...")

    feats = pd.DataFrame(index=df.index)

    # === B1 검사 ===
    if 'B1-1' in df.columns and 'B1-2' in df.columns and 'B1-3' in df.columns:
        b1_resp1 = df['B1-1'].apply(parse_sequence)
        b1_rt = df['B1-2'].apply(parse_sequence)
        b1_resp2 = df['B1-3'].apply(parse_sequence)

        # Response1: 1=correct, 2=incorrect
        feats['B1_resp1_acc'] = b1_resp1.apply(
            lambda x: np.mean(x == 1) if x is not None and len(x) > 0 else np.nan
        )

        # Response2: 1=change-correct, 2=change-incorrect, 3=non change-correct, 4=non change-incorrect
        feats['B1_resp2_correct_rate'] = b1_resp2.apply(
            lambda x: np.mean([1 if v in [1, 3] else 0 for v in x]) if x is not None and len(x) > 0 else np.nan
        )

        feats['B1_mean_rt'] = b1_rt.apply(lambda x: np.mean(x) if x is not None and len(x) > 0 else np.nan)
        feats['B1_std_rt'] = b1_rt.apply(lambda x: np.std(x) if x is not None and len(x) > 1 else np.nan)

    # === B2 검사 ===
    if 'B2-1' in df.columns and 'B2-2' in df.columns and 'B2-3' in df.columns:
        b2_resp1 = df['B2-1'].apply(parse_sequence)
        b2_rt = df['B2-2'].apply(parse_sequence)
        b2_resp2 = df['B2-3'].apply(parse_sequence)

        feats['B2_resp1_acc'] = b2_resp1.apply(
            lambda x: np.mean(x == 1) if x is not None and len(x) > 0 else np.nan
        )
        feats['B2_resp2_correct_rate'] = b2_resp2.apply(
            lambda x: np.mean([1 if v in [1, 3] else 0 for v in x]) if x is not None and len(x) > 0 else np.nan
        )
        feats['B2_mean_rt'] = b2_rt.apply(lambda x: np.mean(x) if x is not None and len(x) > 0 else np.nan)
        feats['B2_std_rt'] = b2_rt.apply(lambda x: np.std(x) if x is not None and len(x) > 1 else np.nan)

    # === B3 검사 ===
    if 'B3-1' in df.columns and 'B3-2' in df.columns:
        b3_resp = df['B3-1'].apply(parse_sequence)
        b3_rt = df['B3-2'].apply(parse_sequence)

        feats['B3_correct_rate'] = b3_resp.apply(
            lambda x: np.mean(x == 1) if x is not None and len(x) > 0 else np.nan
        )
        feats['B3_mean_rt'] = b3_rt.apply(lambda x: np.mean(x) if x is not None and len(x) > 0 else np.nan)
        feats['B3_std_rt'] = b3_rt.apply(lambda x: np.std(x) if x is not None and len(x) > 1 else np.nan)

    # === B4 검사 (60 trials, congruent/incongruent) ===
    if 'B4-1' in df.columns and 'B4-2' in df.columns:
        b4_resp = df['B4-1'].apply(parse_sequence)
        b4_rt = df['B4-2'].apply(parse_sequence)

        # Response: 1,2=correct/incorrect in congruent, 3,4,5,6=correct/incorrect in incongruent
        feats['B4_correct_rate'] = b4_resp.apply(
            lambda x: np.mean([1 if v in [1, 3, 5] else 0 for v in x]) if x is not None and len(x) > 0 else np.nan
        )
        feats['B4_mean_rt'] = b4_rt.apply(lambda x: np.mean(x) if x is not None and len(x) > 0 else np.nan)
        feats['B4_std_rt'] = b4_rt.apply(lambda x: np.std(x) if x is not None and len(x) > 1 else np.nan)
        feats['B4_max_rt'] = b4_rt.apply(lambda x: np.max(x) if x is not None and len(x) > 0 else np.nan)

    # === B5 검사 ===
    if 'B5-1' in df.columns and 'B5-2' in df.columns:
        b5_resp = df['B5-1'].apply(parse_sequence)
        b5_rt = df['B5-2'].apply(parse_sequence)

        feats['B5_correct_rate'] = b5_resp.apply(
            lambda x: np.mean(x == 1) if x is not None and len(x) > 0 else np.nan
        )
        feats['B5_mean_rt'] = b5_rt.apply(lambda x: np.mean(x) if x is not None and len(x) > 0 else np.nan)
        feats['B5_std_rt'] = b5_rt.apply(lambda x: np.std(x) if x is not None and len(x) > 1 else np.nan)

    # === B6, B7, B8 검사 (response only, no RT) ===
    for test_num in [6, 7, 8]:
        col = f'B{test_num}'
        if col in df.columns:
            b_resp = df[col].apply(parse_sequence)
            feats[f'B{test_num}_correct_rate'] = b_resp.apply(
                lambda x: np.mean(x == 1) if x is not None and len(x) > 0 else np.nan
            )

    log.info(f"Test B features: {len(feats.columns)} features")
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats
