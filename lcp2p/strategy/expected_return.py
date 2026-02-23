from __future__ import annotations

import numpy as np
import pandas as pd

def calc_expected_return(raw_df: pd.DataFrame, return_df: pd.DataFrame, default_proba: pd.Series, r_default: float | None = None):
    """상수 방식 기대수익률: E[r_i] = (1-p_i)*IRR_i + p_i*r_default."""
    r_df = return_df.set_index('id')
    p_df = default_proba
    main_df = raw_df.set_index('id')

    if r_default is None:
        mask_default = (main_df['target1'] == 1)
        def_ids = main_df.index[mask_default]
        valid_def_ids = def_ids.intersection(r_df.index)
        r_default = r_df.loc[valid_def_ids, 'r_return'].mean()

    common_idx = main_df.index.intersection(r_df.index).intersection(p_df.index)
    irr = r_df.loc[common_idx, 'IRR']
    irr = (
        irr.astype(str)
        .str.replace('%', '', regex=False)
        .str.strip()
    )
    irr = pd.to_numeric(irr, errors='coerce')
    prob = p_df.loc[common_idx]
    exp_ret = (1.0 - prob) * irr + prob * float(r_default)
    result = pd.DataFrame({'expected_return': exp_ret}).reset_index().rename(columns={'index': 'id'})
    return result, float(r_default)

def calc_decile_params(raw_train: pd.DataFrame, return_df: pd.DataFrame, default_proba: pd.Series, n_deciles: int = 10):
    """십분위별 조건부 수익률 파라미터 산출 (v4)."""
    r_df = return_df.set_index('id')
    p_df = default_proba
    main_df = raw_train.set_index('id')

    common_idx = main_df.index.intersection(r_df.index).intersection(p_df.index)
    df = pd.DataFrame({
        'target1': main_df.loc[common_idx, 'target1'],
        'r_return': r_df.loc[common_idx, 'r_return'],
        'default_proba': p_df.loc[common_idx],
    })

    df['decile'], bins = pd.qcut(df['default_proba'], n_deciles,
                                 labels=False, retbins=True, duplicates='drop')

    decile_params = []
    for d in sorted(df['decile'].unique()):
        sub = df[df['decile'] == d]
        default_mask = (sub['target1'] == 1)
        normal_mask  = (sub['target1'] == 0)
        r_default_d = sub.loc[default_mask, 'r_return'].mean() if default_mask.sum() > 0 else df.loc[df['target1']==1, 'r_return'].mean()
        r_normal_d  = sub.loc[normal_mask, 'r_return'].mean() if normal_mask.sum() > 0 else df.loc[df['target1']==0, 'r_return'].mean()
        decile_params.append(dict(
            decile=d,
            r_default_d=r_default_d,
            r_normal_d=r_normal_d,
            n_default=int(default_mask.sum()),
            n_normal=int(normal_mask.sum()),
            p_mean=sub['default_proba'].mean(),
        ))

    params_df = pd.DataFrame(decile_params)
    return params_df, bins

def calc_expected_return_decile(
    raw_df: pd.DataFrame,
    return_df: pd.DataFrame,
    default_proba: pd.Series,
    decile_params: pd.DataFrame,
    decile_bins,
) -> pd.DataFrame:
    """십분위 방식 기대수익률: E[r_i] = (1-p_i)*r_normal_d(i) + p_i*r_default_d(i)."""
    r_df = return_df.set_index('id')
    p_df = default_proba
    main_df = raw_df.set_index('id')

    common_idx = main_df.index.intersection(r_df.index).intersection(p_df.index)
    prob = p_df.loc[common_idx]

    decile_idx = np.searchsorted(decile_bins[1:-1], prob.values, side='right')
    decile_idx = np.clip(decile_idx, 0, len(decile_params) - 1)

    r_normal_arr  = decile_params['r_normal_d'].values[decile_idx]
    r_default_arr = decile_params['r_default_d'].values[decile_idx]

    exp_ret = (1.0 - prob.values) * r_normal_arr + prob.values * r_default_arr
    return pd.DataFrame({'id': common_idx, 'expected_return': exp_ret})
