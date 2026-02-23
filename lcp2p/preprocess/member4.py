from __future__ import annotations

import numpy as np
import pandas as pd

class Member4Processor:
    """신용계좌, 잔액, 날짜 파생변수 전처리."""

    MEDIAN_COLS = [
        "total_acc", "open_acc", "num_sats", "bc_util", "il_util", "all_util",
        "open_act_il", "total_bal_il", "total_cu_tl", "max_bal_bc",
        "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mort_acc"
    ]
    FILL0_COLS = [
        "num_actv_bc_tl", "num_rev_tl_bal_gt_0", "num_actv_rev_tl", "num_bc_sats",
        "num_bc_tl", "num_op_rev_tl", "num_il_tl", "num_rev_accts", "percent_bc_gt_75"
    ]
    M_FLAG_COLS = [
        "bc_util", "il_util", "all_util", "open_act_il", "total_bal_il",
        "total_cu_tl", "max_bal_bc"
    ]
    M_SIMPLE_COLS = [
        "total_acc", "open_acc", "num_sats", "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op"
    ]
    COLS_999_CHECK = ["mo_sin_old_il_acct", "mo_sin_old_rev_tl_op"]
    LOG_COLS = [
        "avg_cur_bal", "bc_open_to_buy", "revol_bal", "tot_cur_bal",
        "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
        "total_rev_hi_lim", "total_il_high_credit_limit", "total_bal_il", "max_bal_bc"
    ]
    DROP_COLS = [
        "issue_d", "earliest_cr_line", "revol_util",
        "sec_app_open_acc", "verification_status_joint", "loan_status"
    ]
    MY_COLS = [
        "total_acc", "open_acc", "num_sats", "num_bc_tl", "num_op_rev_tl",
        "num_il_tl", "num_rev_accts", "num_actv_bc_tl", "mort_acc",
        "num_rev_tl_bal_gt_0", "open_act_il", "num_actv_rev_tl", "num_bc_sats",
        "total_cu_tl", "percent_bc_gt_75", "all_util", "il_util", "bc_util",
        "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "avg_cur_bal",
        "bc_open_to_buy", "revol_bal", "tot_cur_bal", "tot_hi_cred_lim",
        "total_bal_ex_mort", "total_bc_limit", "total_rev_hi_lim",
        "total_il_high_credit_limit", "total_bal_il", "max_bal_bc",
        "revol_util_num", "credit_hist_months", "has_secondary_app",
        "mort_acc_zero_flag", "mort_acc_nan_flag",
        "bc_util_nan_flag", "il_util_nan_flag", "all_util_nan_flag",
        "open_act_il_nan_flag", "total_bal_il_nan_flag",
        "total_cu_tl_nan_flag", "max_bal_bc_nan_flag",
        "issue_Y", "issue_M_sin", "issue_M_cos", "joint_flag"
    ]

    def __init__(self):
        self.medians_: dict = {}

    @staticmethod
    def _existing_cols(df: pd.DataFrame, cols):
        return [c for c in cols if c in df.columns]

    @staticmethod
    def _parse_revol_util(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype(str).str.replace("%", "", regex=False), errors="coerce"
        )

    @staticmethod
    def _date_median(series: pd.Series):
        valid = series.dropna().sort_values().reset_index(drop=True)
        if len(valid) == 0: 
            return pd.NaT
        return valid.iloc[len(valid) // 2]

    def fit(self, X: pd.DataFrame):
        if "revol_util" in X.columns:
            self.medians_["revol_util_num"] = self._parse_revol_util(X["revol_util"]).median()
        if "issue_d" in X.columns:
            i_d = pd.to_datetime(X["issue_d"], format='mixed', errors="coerce")
            self.medians_["issue_d"] = self._date_median(i_d)
        if "earliest_cr_line" in X.columns:
            e_c = pd.to_datetime(X["earliest_cr_line"], format='mixed', errors="coerce")
            self.medians_["earliest_cr_line"] = self._date_median(e_c)
        for col in self._existing_cols(X, self.MEDIAN_COLS):
            if col in self.COLS_999_CHECK:
                self.medians_[col] = X.loc[X[col] != 999, col].median()
            else:
                self.medians_[col] = X[col].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        fill0 = self._existing_cols(df, self.FILL0_COLS)
        df[fill0] = df[fill0].fillna(0)

        for c in self._existing_cols(df, self.M_FLAG_COLS):
            df[f"{c}_nan_flag"] = df[c].isnull().astype(int)
            df[c] = df[c].fillna(self.medians_.get(c, 0))

        for c in self._existing_cols(df, self.M_SIMPLE_COLS):
            if c in self.COLS_999_CHECK:
                df[c] = df[c].replace(999, np.nan)
            df[c] = df[c].fillna(self.medians_.get(c, 0))

        if "revol_util" in df.columns:
            df["revol_util_num"] = self._parse_revol_util(df["revol_util"])
            df["revol_util_num"] = df["revol_util_num"].fillna(
                self.medians_.get("revol_util_num", 0))

        if "issue_d" in df.columns and "earliest_cr_line" in df.columns:
            i_d = pd.to_datetime(df["issue_d"], format='mixed', errors="coerce")
            e_c = pd.to_datetime(df["earliest_cr_line"], format='mixed', errors="coerce")
            i_d = i_d.fillna(self.medians_["issue_d"])
            e_c = e_c.fillna(self.medians_["earliest_cr_line"])
            df["issue_Y"]     = i_d.dt.year
            df["issue_M_sin"] = np.sin(2 * np.pi * i_d.dt.month / 12)
            df["issue_M_cos"] = np.cos(2 * np.pi * i_d.dt.month / 12)
            df["credit_hist_months"] = (
                (i_d.dt.year - e_c.dt.year) * 12 + (i_d.dt.month - e_c.dt.month))

        if "sec_app_open_acc" in df.columns:
            df["has_secondary_app"] = df["sec_app_open_acc"].notnull().astype(int)

        if "verification_status_joint" in df.columns:
            df["joint_flag"] = df["verification_status_joint"].notna().astype(int)

        if "mort_acc" in df.columns:
            df["mort_acc_zero_flag"] = (df["mort_acc"] == 0).astype(int)
            df["mort_acc_nan_flag"]  = df["mort_acc"].isnull().astype(int)
            df["mort_acc"] = df["mort_acc"].fillna(self.medians_.get("mort_acc", 0))

        log_existing = self._existing_cols(df, self.LOG_COLS)
        df[log_existing] = np.log1p(df[log_existing].fillna(0))

        df = df.drop(columns=self._existing_cols(df, self.DROP_COLS))
        final_cols = ["id"] + self._existing_cols(df, self.MY_COLS)
        return df[final_cols]
