from __future__ import annotations

import argparse
import os

from .config import Settings
from .experiments.analysis import run_repeated_cv, plot_auc_vs_sharpe
from .experiments.plots import plot_tau_curve, plot_tau_hist, plot_overfit_check
from .experiments.sensitivity import run_J_sensitivity, run_mode_comparison


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LendingClub P2P: 내생적 투자집합 하의 샤프비율 최적화 재현")
    p.add_argument("--data-path", required=True, help="lending_club_2020_train.csv 경로")
    p.add_argument("--save-dir", required=True, help="결과 저장 폴더")
    p.add_argument("--sample-frac", type=float, default=0.10)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--k-repeat", type=int, default=50)
    p.add_argument("--num-intervals", type=int, default=10)
    p.add_argument("--division-mode", choices=["quantile","equal"], default="quantile")
    p.add_argument("--tau-min", type=float, default=0.0)
    p.add_argument("--tau-max", type=float, default=2.0)
    p.add_argument("--tau-step", type=float, default=0.05)
    p.add_argument("--optuna-trials", type=int, default=25)
    p.add_argument("--n-deciles", type=int, default=10)
    p.add_argument("--skip-sensitivity", action="store_true", help="J 민감도/등분위-등간격 비교 생략")
    return p

def main():
    args = build_parser().parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    settings = Settings(
        data_path=args.data_path,
        save_dir=args.save_dir,
        sample_frac=args.sample_frac,
        sample_seed=args.sample_seed,
        k_repeat=args.k_repeat,
        num_intervals=args.num_intervals,
        division_mode=args.division_mode,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        tau_step=args.tau_step,
        optuna_trials=args.optuna_trials,
        n_deciles=args.n_deciles,
    )

    out = run_repeated_cv(settings)

    # 기본 figure 저장
    plot_tau_curve(out["all_tau_df"], settings.save_dir, settings.k_repeat)
    plot_tau_hist(out["results_df"], settings.save_dir)
    plot_overfit_check(out["overfit_df"], settings.save_dir)
    plot_auc_vs_sharpe(out["results_df"], settings.save_dir, settings.k_repeat)

    if not args.skip_sensitivity:
        run_J_sensitivity(
            raw_df=out["raw_df"],
            return_df=out["return_df"],
            optimized_params=out["optimized_params"],
            save_dir=settings.save_dir,
        )
        run_mode_comparison(
            raw_df=out["raw_df"],
            return_df=out["return_df"],
            optimized_params=out["optimized_params"],
            num_intervals=settings.num_intervals,
            save_dir=settings.save_dir,
        )

if __name__ == "__main__":
    main()
