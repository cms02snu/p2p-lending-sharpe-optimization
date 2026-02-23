from __future__ import annotations

import numpy as np
import pandas as pd

def divide_interval(
    expected_return_df: pd.DataFrame,
    return_df: pd.DataFrame,
    num_intervals: int = 10,
    mode: str = "quantile",
):
    """기대수익률 축으로 구간화하고 구간별 mu_j, sigma_j 산출."""
    merged = expected_return_df.merge(
        return_df[['id', 'r_return']], on='id', how='inner'
    )
    if mode == 'quantile':
        merged['interval'] = pd.qcut(merged['expected_return'], num_intervals, duplicates='drop')
    else:
        merged['interval'] = pd.cut(merged['expected_return'], num_intervals)

    categories = merged['interval'].cat.categories
    boundaries = np.array([-np.inf] + [iv.right for iv in categories])

    summary = merged.groupby('interval', observed=False).agg(
        기대수익률하한=('expected_return', 'min'),
        기대수익률상한=('expected_return', 'max'),
        관측치수=('r_return', 'count'),
        실현수익률평균=('r_return', 'mean'),
        실현수익률표준편차=('r_return', 'std'),
    ).reset_index()
    summary['구간번호'] = range(1, len(summary) + 1)
    return summary, boundaries

def calc_sharpe_ratio_individual(
    expected_return_df: pd.DataFrame,
    boundaries,
    interval_summary: pd.DataFrame,
    return_df: pd.DataFrame,
) -> pd.DataFrame:
    """개별 대출 위험조정 초과수익률: (E[r_i] - r_treasury,i) / sigma_j."""
    merged = expected_return_df.merge(
        return_df[['id', 'r_treasury', 'funded_amnt']], on='id', how='inner'
    )
    vals = merged['expected_return'].values
    sigmas = interval_summary['실현수익률표준편차'].values
    finite_boundaries = boundaries[1:]
    indices = np.searchsorted(finite_boundaries, vals, side='left')
    indices = np.clip(indices, 0, len(sigmas) - 1)
    merged['sigma'] = sigmas[indices]
    merged['sharpe_ratio_individual'] = (
        (merged['expected_return'] - merged['r_treasury']) / merged['sigma']
    )
    return merged
