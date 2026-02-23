from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_tau_curve(all_tau_df: pd.DataFrame, save_dir: str, k_repeat: int) -> str:
    """tau에 따른 SR 평균 및 95% CI (원본 Figure 2 대응)."""
    summary = all_tau_df.groupby('threshold')['sharpe_ratio'].agg(['mean','std','count']).reset_index()
    summary['se'] = summary['std'] / np.sqrt(summary['count'])
    summary['ci95'] = 1.96 * summary['se']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary['threshold'], summary['mean'])
    ax.fill_between(summary['threshold'],
                    summary['mean'] - summary['ci95'],
                    summary['mean'] + summary['ci95'],
                    alpha=0.2)
    ax.set_xlabel('threshold (tau)')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title(f'SR(tau) 평균 및 95% CI (K={k_repeat})')
    ax.grid(True, alpha=0.3)
    path = os.path.join(save_dir, 'fig_tau_curve_v4.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    return path


def plot_tau_hist(results_df: pd.DataFrame, save_dir: str) -> str:
    """반복에서 선택된 tau* 분포 (원본 Figure 3 대응)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(results_df['best_tau_dec'].dropna(), bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('tau* (decile)')
    ax.set_ylabel('count')
    ax.set_title('최적 선별기준 tau* 분포')
    ax.grid(True, alpha=0.3)
    path = os.path.join(save_dir, 'fig_tau_hist_v4.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    return path


def plot_overfit_check(overfit_df: pd.DataFrame, save_dir: str) -> str:
    """기대수익률 추정 과적합 점검 (train vs val bias/MAE 산점도)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(overfit_df['bias_train_const'], overfit_df['bias_val_const'],
                    alpha=0.6, s=40, edgecolors='black', lw=0.5, label='Constant')
    axes[0].scatter(overfit_df['bias_train_dec'], overfit_df['bias_val_dec'],
                    alpha=0.6, s=40, edgecolors='black', lw=0.5, label='Decile')
    lim = float(np.nanmax(np.abs(overfit_df[['bias_train_const','bias_val_const','bias_train_dec','bias_val_dec']].values))) * 1.1
    axes[0].plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3)
    axes[0].set_xlabel('Bias (Train)'); axes[0].set_ylabel('Bias (Validation)')
    axes[0].set_title('Bias: Train vs Validation'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].scatter(overfit_df['mae_train_const'], overfit_df['mae_val_const'],
                    alpha=0.6, s=40, edgecolors='black', lw=0.5, label='Constant')
    axes[1].scatter(overfit_df['mae_train_dec'], overfit_df['mae_val_dec'],
                    alpha=0.6, s=40, edgecolors='black', lw=0.5, label='Decile')
    mmax = float(np.nanmax(overfit_df[['mae_train_const','mae_val_const','mae_train_dec','mae_val_dec']].values)) * 1.1
    axes[1].plot([0, mmax], [0, mmax], 'k--', alpha=0.3)
    axes[1].set_xlabel('MAE (Train)'); axes[1].set_ylabel('MAE (Validation)')
    axes[1].set_title('MAE: Train vs Validation'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig_overfit_check_v4.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    return path
