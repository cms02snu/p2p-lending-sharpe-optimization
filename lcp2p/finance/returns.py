from __future__ import annotations

import numpy as np
import pandas as pd

def 수익률산출(데이터: pd.DataFrame) -> pd.DataFrame:
    """대출별 연율화 실현수익률 산출 (계약기간 기준).

    r_i = (총수취액_i - 대출원금_i) / 대출원금_i
    연율화는 계약기간(36/60개월) 기준으로 수행.
    """
    df = 데이터.copy()
    df['총회수액'] = df['total_pymnt'].fillna(0) - df['collection_recovery_fee'].fillna(0)
    df['보유기간수익률'] = (df['총회수액'] / df['funded_amnt']) - 1

    df['계약기간개월'] = pd.to_numeric(
        df['term'].astype(str).str.extract(r'(\d+)', expand=False),
        errors='coerce'
    ).fillna(36)

    df['연율화수익률'] = ((1 + df['보유기간수익률']) ** (12 / df['계약기간개월']) - 1) * 100
    return df[['id', '연율화수익률']].reset_index(drop=True)

def load_fred_data():
    """FRED에서 국채수익률(3Y/5Y) 및 연방기금금리를 로드."""
    t3 = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3")
    t5 = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS5")
    ff = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF")
    return t3, t5, ff

def 월별수익률(df: pd.DataFrame, 값컬럼: str, 출력명: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp['observation_date'] = pd.to_datetime(tmp['observation_date'], errors='coerce')
    tmp[값컬럼] = pd.to_numeric(tmp[값컬럼].replace('.', np.nan), errors='coerce')
    tmp = tmp.dropna()
    tmp['발행년월'] = tmp['observation_date'].dt.to_period('M')
    return tmp.groupby('발행년월')[값컬럼].mean().reset_index().rename(columns={값컬럼: 출력명})

def calc_return(df: pd.DataFrame, m3: pd.DataFrame, m5: pd.DataFrame, interest: pd.DataFrame) -> pd.DataFrame:
    """id 기준 룩업 테이블 구성 (r_return, IRR, r_treasury, r_riskfree, funded_amnt)."""
    out_df = df.copy()
    realized_df = 수익률산출(df)
    out_df = out_df.merge(realized_df, on='id', how='left')
    out_df = out_df.rename(columns={'연율화수익률': 'r_return'})

    if out_df['int_rate'].dtype == 'object':
        out_df['IRR'] = pd.to_numeric(out_df['int_rate'].str.replace('%', ''), errors='coerce')
    else:
        out_df['IRR'] = out_df['int_rate']

    out_df['issue_d_dt'] = pd.to_datetime(out_df['issue_d'], format='mixed', errors='coerce')
    out_df['발행년월'] = out_df['issue_d_dt'].dt.to_period('M')
    out_df = out_df.merge(m3, on='발행년월', how='left')
    out_df = out_df.merge(m5, on='발행년월', how='left')
    out_df = out_df.merge(interest, on='발행년월', how='left')

    out_df['term_str'] = out_df['term'].fillna('').astype(str)
    out_df['r_treasury'] = np.where(
        out_df['term_str'].str.contains('36'),
        out_df['TREASURY3Y'], out_df['TREASURY5Y']
    )
    out_df['r_riskfree'] = out_df['CALL']
    out_df['funded_amnt'] = pd.to_numeric(out_df['funded_amnt'], errors='coerce')

    return out_df[['id', 'r_return', 'IRR', 'r_treasury', 'r_riskfree', 'funded_amnt']]
