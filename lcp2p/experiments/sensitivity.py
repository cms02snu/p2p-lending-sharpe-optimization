from __future__ import annotations

import gc
import os
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from ..preprocess.processor import DataProcessor
from ..models.xgb_default import fit_xgb, predict_xgb
from ..strategy.expected_return import calc_decile_params, calc_expected_return_decile
from ..strategy.risk_proxy import divide_interval, calc_sharpe_ratio_individual
from ..strategy.portfolio import calc_sharpe_ratio_portfolio, grid_search_tau
from ..utils.io import save_csv


def run_J_sensitivity(
    raw_df: pd.DataFrame,
    return_df: pd.DataFrame,
    optimized_params: dict,
    J_values: List[int] = [5, 10, 15, 20],
    J_REPEAT: int = 5,
    division_mode: str = "quantile",
    save_dir: str | None = None,
) -> Dict[str, pd.DataFrame]:
    """구간 수 J 민감도 분석 (원본 코드: J=5,10,15,20 각 5회)."""
    j_results = []
    for J in J_values:
        for rep in range(J_REPEAT):
            rtv, rte = train_test_split(raw_df, test_size=0.2, stratify=raw_df["target1"], random_state=rep)
            rtr, rva = train_test_split(rtv, test_size=0.25, stratify=rtv["target1"], random_state=rep)

            proc = DataProcessor()
            trd = proc.fit_transform(rtr)
            vad = proc.transform(rva)
            ted = proc.transform(rte)

            mdl = fit_xgb(trd, eval_df=vad, xgb_params=optimized_params)

            dp = predict_xgb(mdl, trd)
            dpar, dbin = calc_decile_params(rtr, return_df, dp)
            er = calc_expected_return_decile(rtr, return_df, dp, dpar, dbin)

            isumm, bds = divide_interval(er, return_df, num_intervals=J, mode=division_mode)

            dpv = predict_xgb(mdl, vad)
            erv = calc_expected_return_decile(rva, return_df, dpv, dpar, dbin)
            iv = calc_sharpe_ratio_individual(erv, bds, isumm, return_df)
            _, bt = grid_search_tau(iv, return_df)

            dpt = predict_xgb(mdl, ted)
            ert = calc_expected_return_decile(rte, return_df, dpt, dpar, dbin)
            it = calc_sharpe_ratio_individual(ert, bds, isumm, return_df)
            ts = calc_sharpe_ratio_portfolio(it, return_df, bt)

            j_results.append(dict(
                J=J, rep=rep, best_tau=bt, sharpe_ratio=ts['sharpe_ratio'],
                acceptance_rate=ts['acceptance_rate'],
            ))
            del mdl, trd, vad, ted
            gc.collect()

    j_results_df = pd.DataFrame(j_results)
    j_summary = j_results_df.groupby('J').agg(
        SR평균=('sharpe_ratio','mean'),
        SR표준편차=('sharpe_ratio','std'),
        승인비율=('acceptance_rate','mean'),
        tau평균=('best_tau','mean')
    ).round(4)

    fig_path = None
    if save_dir is not None:
        save_csv(j_results_df, "j_sensitivity_v4", save_dir)
        fig, ax = plt.subplots(figsize=(8, 5))
        jm = j_results_df.groupby('J')['sharpe_ratio'].mean()
        js = j_results_df.groupby('J')['sharpe_ratio'].std()
        ax.errorbar(jm.index, jm.values, yerr=js.values, fmt='o-', capsize=5, lw=1.5, ms=8, color='steelblue')
        ax.set_xlabel('J'); ax.set_ylabel('Sharpe Ratio'); ax.set_title('Sensitivity to J (Decile)')
        ax.set_xticks(J_values); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join(save_dir, 'fig_J_sensitivity_v4.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')

    return {"j_results_df": j_results_df, "j_summary": j_summary, "fig_path": fig_path}


def run_mode_comparison(
    raw_df: pd.DataFrame,
    return_df: pd.DataFrame,
    optimized_params: dict,
    num_intervals: int = 10,
    J_REPEAT: int = 5,
    modes: List[str] = ["quantile", "equal"],
    save_dir: str | None = None,
) -> pd.DataFrame:
    """등분위(quantile) vs 등간격(equal) 비교 (원본 코드: 5회 반복)."""
    mode_results = []
    for mode in modes:
        for rep in range(J_REPEAT):
            rtv, rte = train_test_split(raw_df, test_size=0.2, stratify=raw_df["target1"], random_state=rep)
            rtr, rva = train_test_split(rtv, test_size=0.25, stratify=rtv["target1"], random_state=rep)

            proc = DataProcessor()
            trd = proc.fit_transform(rtr)
            vad = proc.transform(rva)
            ted = proc.transform(rte)

            mdl = fit_xgb(trd, eval_df=vad, xgb_params=optimized_params)

            dp = predict_xgb(mdl, trd)
            dpar, dbin = calc_decile_params(rtr, return_df, dp)
            er = calc_expected_return_decile(rtr, return_df, dp, dpar, dbin)
            isumm, bds = divide_interval(er, return_df, num_intervals=num_intervals, mode=mode)

            dpv = predict_xgb(mdl, vad)
            erv = calc_expected_return_decile(rva, return_df, dpv, dpar, dbin)
            iv = calc_sharpe_ratio_individual(erv, bds, isumm, return_df)
            _, bt = grid_search_tau(iv, return_df)

            dpt = predict_xgb(mdl, ted)
            ert = calc_expected_return_decile(rte, return_df, dpt, dpar, dbin)
            it = calc_sharpe_ratio_individual(ert, bds, isumm, return_df)
            ts = calc_sharpe_ratio_portfolio(it, return_df, bt)

            mode_results.append(dict(mode=mode, rep=rep, best_tau=bt, sharpe_ratio=ts['sharpe_ratio']))
            del mdl, trd, vad, ted
            gc.collect()

    mode_df = pd.DataFrame(mode_results)
    if save_dir is not None:
        save_csv(mode_df, "mode_comparison_v4", save_dir)
    return mode_df
