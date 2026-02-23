from __future__ import annotations

from dataclasses import dataclass

@dataclass
class Settings:
    # 데이터/출력 경로
    data_path: str
    save_dir: str

    # 샘플링/반복
    sample_frac: float = 0.10
    sample_seed: int = 42
    k_repeat: int = 50

    # 구간화(J) 및 방식
    num_intervals: int = 10
    division_mode: str = "quantile"  # 'quantile' or 'equal'

    # 선별기준(tau) 탐색
    tau_min: float = 0.0
    tau_max: float = 2.0
    tau_step: float = 0.05

    # Optuna
    optuna_trials: int = 25

    # 십분위 기대수익률
    n_deciles: int = 10

    # 기타
    omp_num_threads: int = 4
    seed: int = 0
