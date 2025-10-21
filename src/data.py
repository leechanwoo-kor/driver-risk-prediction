from __future__ import annotations
from pathlib import Path
import pandas as pd

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
