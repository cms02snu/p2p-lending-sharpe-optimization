from __future__ import annotations

import numpy as np
import pandas as pd

def calc_sharpe_ratio_portfolio(individual_df: pd.DataFrame, return_df: pd.DataFrame, threshold: float):
    """포트폴리오 샤프비율: SR(tau) = (R_p(tau) - r_treasury_bar(tau)) / sigma_p(tau)."""
    merged = individual_df.merge(
        return_df[['id', 'r_return', 'r_treasury']],
        on='id', how='inner', suffixes=('', '_dup')
    )
    accepted_mask = merged['sharpe_ratio_individual'] > threshold
    merged['final_return'] = np.where(accepted_mask, merged['r_return'], merged['r_treasury'])

    weights = merged['funded_amnt'].values.astype(np.float64)
    total_weight = weights.sum()
    if total_weight == 0:
        return dict(sharpe_ratio=np.nan, acceptance_rate=0.0,
                    portfolio_return=np.nan, portfolio_sigma=np.nan,
                    accepted_count=0, total_count=len(merged))

    rp = np.average(merged['final_return'], weights=weights)
    rt_bar = np.average(merged['r_treasury'], weights=weights)

    w_norm = weights / total_weight
    var_p = np.sum(w_norm * (merged['final_return'].values - rp) ** 2)
    v2 = np.sum(w_norm ** 2)
    if (1.0 - v2) > 0:
        var_p = var_p / (1.0 - v2)
    sigma_p = np.sqrt(var_p)

    sharpe_ratio = (rp - rt_bar) / sigma_p if sigma_p > 0 else np.nan
    return dict(
        sharpe_ratio=sharpe_ratio,
        acceptance_rate=float(accepted_mask.mean()),
        portfolio_return=float(rp),
        portfolio_sigma=float(sigma_p),
        accepted_count=int(accepted_mask.sum()),
        total_count=int(len(merged)),
    )

def grid_search_tau(
    individual_df: pd.DataFrame,
    return_df: pd.DataFrame,
    tau_min: float = 0.0,
    tau_max: float = 2.0,
    tau_step: float = 0.05,
):
    """격자 탐색으로 tau* = argmax SR(tau)."""
    tau_grid = np.arange(tau_min, tau_max + tau_step / 2, tau_step)
    results = []
    for tau in tau_grid:
        stats = calc_sharpe_ratio_portfolio(individual_df, return_df, float(tau))
        stats['threshold'] = round(float(tau), 4)
        results.append(stats)
    results_df = pd.DataFrame(results)
    valid = results_df.dropna(subset=['sharpe_ratio'])
    best_tau = valid.loc[valid['sharpe_ratio'].idxmax(), 'threshold'] if len(valid) > 0 else np.nan
    return results_df, float(best_tau) if best_tau == best_tau else np.nan
