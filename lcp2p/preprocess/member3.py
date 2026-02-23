from __future__ import annotations

import numpy as np
import pandas as pd

class Member3Processor:
    """연체/부정 기록 및 최근 신용활동 변수 전처리."""

    def __init__(self):
        self.fill0_cols = [
            "acc_now_delinq", "delinq_amnt", "tot_coll_amt", "num_tl_30dpd",
            "chargeoff_within_12_mths", "collections_12_mths_ex_med",
            "num_tl_90g_dpd_24m", "delinq_2yrs", "num_accts_ever_120_pd",
            "num_tl_120dpd_2m", "pub_rec", "pub_rec_bankruptcies", "tax_liens",
            "open_il_12m", "open_il_24m", "open_rv_12m", "open_rv_24m",
            "acc_open_past_24mths", "num_tl_op_past_12m",
            "open_acc_6m", "inq_fi", "inq_last_12m", "inq_last_6mths",
        ]
        self.nanflag_fill0_cols = [
            "mths_since_recent_bc_dlq", "mths_since_recent_revol_delinq",
            "mths_since_recent_inq", "mths_since_last_delinq",
            "mths_since_last_major_derog", "mo_sin_rcnt_rev_tl_op",
            "mo_sin_rcnt_tl", "mths_since_recent_bc",
        ]
        self.median_cols = ["pct_tl_nvr_dlq"]
        self.target_cols = sorted(set(
            self.fill0_cols + self.nanflag_fill0_cols + self.median_cols
        ))
        self.medians_ = {}
        self.fitted_ = False

    def _existing(self, df: pd.DataFrame, cols):
        return [c for c in cols if c in df.columns]

    def fit(self, X: pd.DataFrame):
        df = X.copy()
        cols = ["id"] + self._existing(df, self.target_cols)
        df = df.loc[:, cols].copy()
        if "pct_tl_nvr_dlq" in df.columns:
            s = pd.to_numeric(df["pct_tl_nvr_dlq"], errors="coerce")
            s = s.where((s >= 0) & (s <= 100), np.nan)
            med = s.median()
            self.medians_["pct_tl_nvr_dlq"] = float(med) if pd.notna(med) else 0.0
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Member3Processor is not fitted.")
        df = X.copy()
        cols = ["id"] + self._existing(df, self.target_cols)
        df_out = df.loc[:, cols].copy()

        for c in self._existing(df_out, self.fill0_cols):
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce").fillna(0)

        for c in self._existing(df_out, self.nanflag_fill0_cols):
            df_out[f"{c}_nan_flag"] = df_out[c].isna().astype("int8")
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce").fillna(0)

        if "pct_tl_nvr_dlq" in df_out.columns:
            df_out["pct_tl_nvr_dlq_nan_flag"] = df_out["pct_tl_nvr_dlq"].isna().astype("int8")
            s = pd.to_numeric(df_out["pct_tl_nvr_dlq"], errors="coerce")
            s = s.where((s >= 0) & (s <= 100), np.nan)
            df_out["pct_tl_nvr_dlq"] = s.fillna(self.medians_.get("pct_tl_nvr_dlq", 0.0)).astype("float32")

        for c in df_out.columns:
            if c == "id": 
                continue
            if c.endswith("_nan_flag"):
                df_out[c] = df_out[c].astype("int8")
            else:
                df_out[c] = pd.to_numeric(df_out[c], errors="coerce").astype("float32")
        return df_out

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
