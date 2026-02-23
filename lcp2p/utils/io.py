from __future__ import annotations

import os
import pandas as pd

def save_csv(df: pd.DataFrame, name: str, save_dir: str) -> str:
    """중간 결과를 UTF-8-SIG CSV로 저장하고 경로를 반환한다."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path
