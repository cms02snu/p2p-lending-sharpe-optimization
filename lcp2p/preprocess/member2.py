from __future__ import annotations

import pandas as pd

class LendingClubPipeline:
    """범주형 변수 전처리 (purpose, home_ownership, sub_grade 등).

    원본 노트북의 로직을 유지하되, 학습 데이터에서 빈도가 낮은 범주는 OTHER로 병합한다.
    """
    def __init__(self, min_freq: float = 0.01):
        self.min_freq = min_freq
        self.freq_merge_cols = ['purpose', 'home_ownership']
        self.no_merge_cols = ['verification_status', 'initial_list_status']
        self.categorical_cols = self.freq_merge_cols + self.no_merge_cols + ['sub_grade']
        self.keep_categories = {col: None for col in self.categorical_cols}

    def fit(self, df: pd.DataFrame):
        missing = [c for c in self.categorical_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required categorical columns in train: {missing}")
        for col in self.categorical_cols:
            s = df[col].fillna("MISSING").astype(str)
            if col in self.freq_merge_cols:
                freq = s.value_counts(normalize=True)
                keep_cats = set(freq[freq >= self.min_freq].index)
            else:
                keep_cats = set(s.unique())
            keep_cats.add("MISSING")
            self.keep_categories[col] = keep_cats
        return self

    def transform(self, df: pd.DataFrame, mode: str = "default_model") -> pd.DataFrame:
        df_out = df.copy().reset_index(drop=True)

        # 정규화/파싱
        if "int_rate" in df_out.columns:
            s = df_out["int_rate"].astype(str).str.replace("%", "", regex=False)
            df_out["int_rate"] = pd.to_numeric(s, errors="coerce")
        if "term" in df_out.columns:
            s_term = df_out["term"].astype(str).str.extract(r"(\d+)", expand=False)
            df_out["term"] = pd.to_numeric(s_term, errors="coerce")
        if "emp_length" in df_out.columns:
            def parse_emp_length(x):
                if pd.isna(x) or str(x).lower() == "n/a": return -1
                x = str(x).lower()
                if "<" in x: return 0
                if "+" in x: return 10
                try: return int(x.split()[0])
                except: return -1
            df_out["emp_length"] = df_out["emp_length"].apply(parse_emp_length)
        if "fico_range_low" in df_out.columns and "fico_range_high" in df_out.columns:
            df_out["fico_mid"] = (
                pd.to_numeric(df_out["fico_range_low"], errors="coerce")
                + pd.to_numeric(df_out["fico_range_high"], errors="coerce")
            ) / 2

        # 범주형 병합/정리
        for col in self.categorical_cols:
            if col in df_out.columns:
                s = df_out[col].fillna("MISSING").astype(str)
                keep_cats = self.keep_categories[col]
                if keep_cats is None:
                    raise RuntimeError(f"Pipeline not fitted for column: {col}")
                df_out[col] = s.apply(lambda x: x if x in keep_cats else "OTHER").astype("category")

        # 불필요 변수 제거
        drop_list = ["fico_range_low", "fico_range_high"]
        df_out = df_out.drop(columns=[c for c in drop_list if c in df_out.columns])

        # 모드별 제거 변수
        if mode == "default_model":
            if "loan_amnt" in df_out.columns:
                df_out = df_out.drop(columns=["loan_amnt"])
        elif mode == "return_model":
            if "installment" in df_out.columns:
                df_out = df_out.drop(columns=["installment"])

        # id 분리 후 수치형 정리
        temp_id = None
        if "id" in df_out.columns:
            temp_id = df_out["id"]
            df_out = df_out.drop(columns=["id"])

        num_cols = df_out.select_dtypes(exclude=["category"]).columns
        for c in num_cols:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce").astype("float32")

        if temp_id is not None:
            df_out.insert(0, "id", temp_id)

        my_cols = [
            "int_rate", "term", "emp_length", "fico_mid", "dti", "annual_inc",
            "sub_grade", "home_ownership", "purpose", "verification_status",
            "initial_list_status", "loan_amnt", "installment"
        ]
        final_cols = ["id"] + [c for c in my_cols if c in df_out.columns]
        return df_out[final_cols]
