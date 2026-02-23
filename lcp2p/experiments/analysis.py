from __future__ import annotations

import gc
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from ..config import Settings
from ..utils.repro import set_reproducibility
from ..utils.io import save_csv
from ..io.data import load_and_filter_data
from ..finance.returns import load_fred_data, 월별수익률, calc_return
from ..preprocess.processor import DataProcessor
from ..models.xgb_default import fit_xgb, predict_xgb, eval_auc, optimize_hyperparams
from ..strategy.expected_return import (
    calc_expected_return, calc_decile_params, calc_expected_return_decile,
)
from ..strategy.risk_proxy import divide_interval, calc_sharpe_ratio_individual
from ..strategy.portfolio import calc_sharpe_ratio_portfolio, grid_search_tau


def calc_er_mae(er_df: pd.DataFrame, return_df: pd.DataFrame, label: str) -> Dict[str, Any]:
    """기대수익률과 실현수익률 사이의 MAE 및 편향(bias)."""
    merged = er_df.merge(return_df[['id', 'r_return']], on='id', how='inner')
    residual = merged['expected_return'] - merged['r_return']
    return dict(
        label=label,
        mae=float(residual.abs().mean()),
        bias=float(residual.mean()),
        er_mean=float(merged['expected_return'].mean()),
        r_return_mean=float(merged['r_return'].mean()),
    )


def make_benchmark_df(raw_test: pd.DataFrame, return_df: pd.DataFrame, score_col_value: float) -> pd.DataFrame:
    bm = raw_test[['id']].copy()
    bm['sharpe_ratio_individual'] = score_col_value
    bm = bm.merge(return_df[['id', 'funded_amnt', 'r_treasury']], on='id', how='inner')
    return bm


def run_repeated_cv(settings: Settings) -> Dict[str, Any]:
    """논문 재현을 위한 메인 파이프라인 (K회 반복)."""
    set_reproducibility(settings.seed, settings.omp_num_threads)

    os.makedirs(settings.save_dir, exist_ok=True)

    raw_df_full = load_and_filter_data(settings.data_path)

    raw_df, _ = train_test_split(
        raw_df_full, train_size=settings.sample_frac,
        stratify=raw_df_full["target1"], random_state=settings.sample_seed,
    )
    del raw_df_full
    gc.collect()

    # FRED 기반 금리 테이블 구성
    t3, t5, ff = load_fred_data()
    m3 = 월별수익률(t3, "DGS3", "TREASURY3Y")
    m5 = 월별수익률(t5, "DGS5", "TREASURY5Y")
    interest = 월별수익률(ff, "DFF", "CALL")

    return_df = calc_return(raw_df, m3, m5, interest)
    save_csv(return_df, "return_df_v4", settings.save_dir)

    results_list: List[dict] = []
    all_tau_grid_results: List[pd.DataFrame] = []
    overfit_check_list: List[dict] = []

    first_split_interval_const = None
    first_split_interval_dec = None
    optimized_params = None

    for k in range(settings.k_repeat):
        t_start = time.time()

        raw_trainval, raw_test = train_test_split(
            raw_df, test_size=0.2, stratify=raw_df["target1"], random_state=k)
        raw_train, raw_val = train_test_split(
            raw_trainval, test_size=0.25, stratify=raw_trainval["target1"], random_state=k)

        processor = DataProcessor()
        train_df = processor.fit_transform(raw_train)
        val_df   = processor.transform(raw_val)
        test_df  = processor.transform(raw_test)

        if k == 0:
            optimized_params = optimize_hyperparams(train_df, val_df, n_trials=settings.optuna_trials)

        model = fit_xgb(train_df, eval_df=val_df, xgb_params=optimized_params)

        dp_train = predict_xgb(model, train_df)
        dp_val   = predict_xgb(model, val_df)
        dp_test  = predict_xgb(model, test_df)

        # 2단계A: 상수 방식
        er_train_const, r_default_const = calc_expected_return(raw_train, return_df, dp_train)
        interval_const, boundaries_const = divide_interval(er_train_const, return_df, num_intervals=settings.num_intervals, mode=settings.division_mode)
        if k == 0:
            first_split_interval_const = interval_const.copy()

        er_val_const, _ = calc_expected_return(raw_val, return_df, dp_val, r_default=r_default_const)
        ind_val_const = calc_sharpe_ratio_individual(er_val_const, boundaries_const, interval_const, return_df)
        _, best_tau_const = grid_search_tau(ind_val_const, return_df, settings.tau_min, settings.tau_max, settings.tau_step)

        # 2단계B: 십분위 방식
        decile_params, decile_bins = calc_decile_params(raw_train, return_df, dp_train, n_deciles=settings.n_deciles)
        er_train_dec = calc_expected_return_decile(raw_train, return_df, dp_train, decile_params, decile_bins)
        interval_dec, boundaries_dec = divide_interval(er_train_dec, return_df, num_intervals=settings.num_intervals, mode=settings.division_mode)
        if k == 0:
            first_split_interval_dec = interval_dec.copy()

        er_val_dec = calc_expected_return_decile(raw_val, return_df, dp_val, decile_params, decile_bins)
        ind_val_dec = calc_sharpe_ratio_individual(er_val_dec, boundaries_dec, interval_dec, return_df)
        tau_grid_df, best_tau_dec = grid_search_tau(ind_val_dec, return_df, settings.tau_min, settings.tau_max, settings.tau_step)

        # 과적합 점검: MAE/Bias 및 SR 비교 (train/val)
        mae_train_const = calc_er_mae(er_train_const, return_df, 'train_const')
        mae_train_dec   = calc_er_mae(er_train_dec, return_df, 'train_decile')
        mae_val_const   = calc_er_mae(er_val_const, return_df, 'val_const')
        mae_val_dec     = calc_er_mae(er_val_dec, return_df, 'val_decile')

        sr_val_const_best = calc_sharpe_ratio_portfolio(ind_val_const, return_df, best_tau_const)
        sr_val_dec_best   = calc_sharpe_ratio_portfolio(ind_val_dec, return_df, best_tau_dec)

        overfit_check_list.append(dict(
            k=k,
            mae_train_const=mae_train_const['mae'], bias_train_const=mae_train_const['bias'],
            mae_train_dec=mae_train_dec['mae'],     bias_train_dec=mae_train_dec['bias'],
            mae_val_const=mae_val_const['mae'],     bias_val_const=mae_val_const['bias'],
            mae_val_dec=mae_val_dec['mae'],         bias_val_dec=mae_val_dec['bias'],
            sr_val_const_best=sr_val_const_best['sharpe_ratio'],
            sr_val_dec_best=sr_val_dec_best['sharpe_ratio'],
            tau_const=best_tau_const, tau_dec=best_tau_dec,
            accept_val_const=sr_val_const_best['acceptance_rate'],
            accept_val_dec=sr_val_dec_best['acceptance_rate'],
        ))

        # 4단계: 평가 데이터 성과
        er_test_dec = calc_expected_return_decile(raw_test, return_df, dp_test, decile_params, decile_bins)
        ind_test_dec = calc_sharpe_ratio_individual(er_test_dec, boundaries_dec, interval_dec, return_df)
        test_stats_dec = calc_sharpe_ratio_portfolio(ind_test_dec, return_df, best_tau_dec)

        er_test_const, _ = calc_expected_return(raw_test, return_df, dp_test, r_default=r_default_const)
        ind_test_const = calc_sharpe_ratio_individual(er_test_const, boundaries_const, interval_const, return_df)
        test_stats_const = calc_sharpe_ratio_portfolio(ind_test_const, return_df, best_tau_const)

        test_auc = eval_auc(model, test_df)

        tau_grid_df['k'] = k
        all_tau_grid_results.append(tau_grid_df)

        results_list.append(dict(
            k=k, test_auc=float(test_auc),
            best_tau_dec=best_tau_dec,
            test_sharpe_dec=test_stats_dec['sharpe_ratio'],
            test_return_dec=test_stats_dec['portfolio_return'],
            test_sigma_dec=test_stats_dec['portfolio_sigma'],
            test_accept_dec=test_stats_dec['acceptance_rate'],
            best_tau_const=best_tau_const,
            test_sharpe_const=test_stats_const['sharpe_ratio'],
            test_return_const=test_stats_const['portfolio_return'],
            test_sigma_const=test_stats_const['portfolio_sigma'],
            test_accept_const=test_stats_const['acceptance_rate'],
            r_default_const=r_default_const,
            elapsed_sec=time.time() - t_start,
        ))

        del model, train_df, val_df, test_df
        gc.collect()

    results_df = pd.DataFrame(results_list)
    all_tau_df = pd.concat(all_tau_grid_results, ignore_index=True)
    overfit_df = pd.DataFrame(overfit_check_list)

    save_csv(results_df, "results_summary_v4", settings.save_dir)
    save_csv(all_tau_df, "tau_grid_all_v4", settings.save_dir)
    save_csv(overfit_df, "overfit_check_v4", settings.save_dir)
    if first_split_interval_const is not None:
        save_csv(first_split_interval_const, "interval_summary_k0_const_v4", settings.save_dir)
    if first_split_interval_dec is not None:
        save_csv(first_split_interval_dec, "interval_summary_k0_decile_v4", settings.save_dir)

    # 벤치마크 (k=0 분할 기준, 테스트셋에서 산출)
    rtv_bm, rte_bm = train_test_split(raw_df, test_size=0.2, stratify=raw_df["target1"], random_state=0)

    oracle_df = rte_bm[['id','target1']].copy()
    oracle_df['sharpe_ratio_individual'] = np.where(oracle_df['target1']==0, 1.0, -1.0)
    oracle_df = oracle_df.merge(return_df[['id','funded_amnt','r_treasury']], on='id', how='inner')
    oracle_stats = calc_sharpe_ratio_portfolio(oracle_df, return_df, 0.0)

    all_accept_stats = calc_sharpe_ratio_portfolio(make_benchmark_df(rte_bm, return_df, 1.0), return_df, 0.0)
    all_reject_stats = calc_sharpe_ratio_portfolio(make_benchmark_df(rte_bm, return_df, -1.0), return_df, 0.0)

    benchmark_df = pd.DataFrame([
        {'벤치마크': 'Oracle', **oracle_stats},
        {'벤치마크': '전체 승인', **all_accept_stats},
        {'벤치마크': '전체 거절', **all_reject_stats},
        {'벤치마크': f'십분위 모형 ({settings.k_repeat}회 평균)',
         'sharpe_ratio': results_df['test_sharpe_dec'].mean(),
         'acceptance_rate': results_df['test_accept_dec'].mean(),
         'portfolio_return': results_df['test_return_dec'].mean(),
         'portfolio_sigma': results_df['test_sigma_dec'].mean()},
        {'벤치마크': f'상수 모형 ({settings.k_repeat}회 평균)',
         'sharpe_ratio': results_df['test_sharpe_const'].mean(),
         'acceptance_rate': results_df['test_accept_const'].mean(),
         'portfolio_return': results_df['test_return_const'].mean(),
         'portfolio_sigma': results_df['test_sigma_const'].mean()},
    ])
    save_csv(benchmark_df, "benchmark_comparison_v4", settings.save_dir)

    return dict(
        raw_df=raw_df,
        results_df=results_df,
        all_tau_df=all_tau_df,
        overfit_df=overfit_df,
        interval_k0_const=first_split_interval_const,
        interval_k0_decile=first_split_interval_dec,
        optimized_params=optimized_params,
        return_df=return_df,
        benchmarks=dict(oracle=oracle_stats, all_accept=all_accept_stats, all_reject=all_reject_stats),
    )


def plot_auc_vs_sharpe(results_df: pd.DataFrame, save_dir: str, k_repeat: int):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(results_df['test_auc'], results_df['test_sharpe_dec'],
               alpha=0.6, edgecolors='black', lw=0.5, s=50, c='coral', label='Decile')
    ax.scatter(results_df['test_auc'], results_df['test_sharpe_const'],
               alpha=0.4, edgecolors='black', lw=0.5, s=50, c='steelblue', marker='s', label='Constant')
    ax.set_xlabel('AUC'); ax.set_ylabel('Sharpe Ratio')
    ax.set_title(f'AUC vs Sharpe Ratio (K={k_repeat})'); ax.grid(True, alpha=0.3); ax.legend()
    corr_dec = results_df[['test_auc','test_sharpe_dec']].corr().iloc[0,1]
    ax.annotate(f'corr(decile) = {corr_dec:.3f}', xy=(0.05,0.95), xycoords='axes fraction', fontsize=11, va='top')
    plt.tight_layout()
    path = os.path.join(save_dir, 'fig_auc_vs_sharpe_v4.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    return path
