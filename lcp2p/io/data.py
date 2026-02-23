from __future__ import annotations

import pandas as pd

def load_and_filter_data(file_path: str) -> pd.DataFrame:
    """원자료 로드 및 라벨 생성.

    - loan_status에서 "Fully Paid"는 정상상환(0),
      "Charged Off", "Default"는 부도(1)로 정의.
    - "Current", "In Grace Period", "Late" 등 중간 상태는 제외.

    Returns
    -------
    pd.DataFrame
        id(str), target1(int) 포함.
    """
    df = pd.read_csv(file_path, low_memory=False)
    df["id"] = df["id"].astype(str)

    definitive_status = ["Fully Paid", "Charged Off", "Default"]
    df = df.dropna(subset=["loan_status"])
    df = df[df["loan_status"].isin(definitive_status)].copy()

    df["target1"] = df["loan_status"].apply(
        lambda x: 1 if x in ["Charged Off", "Default"] else 0
    )
    return df
