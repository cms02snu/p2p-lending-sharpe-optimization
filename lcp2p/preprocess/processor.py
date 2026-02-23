from __future__ import annotations

import gc
import pandas as pd

from .member2 import LendingClubPipeline
from .member3 import Member3Processor
from .member4 import Member4Processor

class DataProcessor:
    """전처리 통합: Member2 + Member3 + Member4를 병합."""

    def __init__(self, mode: str = "default_model", min_freq: float = 0.01):
        self.mode = mode
        self.min_freq = min_freq
        self.m2_ = None
        self.m3_ = None
        self.m4_ = None
        self._is_fitted = False

    def fit(self, raw_train: pd.DataFrame):
        X = raw_train.drop(columns=["target1"], errors="ignore")
        self.m4_ = Member4Processor(); self.m4_.fit(X)
        self.m3_ = Member3Processor(); self.m3_.fit(X)
        self.m2_ = LendingClubPipeline(min_freq=self.min_freq); self.m2_.fit(X)
        self._is_fitted = True
        return self

    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("DataProcessor is not fitted.")

        X = raw_df.drop(columns=["target1"], errors="ignore")
        y = raw_df.set_index("id")["target1"] if "target1" in raw_df.columns else None

        X_m4 = self.m4_.transform(X)
        X_m3 = self.m3_.transform(X)
        X_m2 = self.m2_.transform(X, mode=self.mode)

        X_final = (
            X_m4.merge(X_m3, on="id", how="inner", suffixes=("", "_m3dup"))
                .merge(X_m2, on="id", how="inner", suffixes=("", "_m2dup"))
        )
        dup_cols = [c for c in X_final.columns if c.endswith(("_m3dup", "_m2dup"))]
        if dup_cols:
            X_final = X_final.drop(columns=dup_cols)

        X_final = X_final.set_index("id")
        if y is not None:
            out = X_final.join(y.rename("target1")).reset_index()
        else:
            out = X_final.reset_index()

        del X_m4, X_m3, X_m2, X_final
        gc.collect()
        return out

    def fit_transform(self, raw_train: pd.DataFrame) -> pd.DataFrame:
        self.fit(raw_train)
        return self.transform(raw_train)
