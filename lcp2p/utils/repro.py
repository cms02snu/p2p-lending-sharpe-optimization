from __future__ import annotations

import os
import random
import numpy as np

def set_reproducibility(seed: int = 0, omp_num_threads: int = 4) -> None:
    """실험 재현성 확보를 위한 시드/환경변수 설정."""
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["OMP_NUM_THREADS"] = str(int(omp_num_threads))
    random.seed(seed)
    np.random.seed(seed)
