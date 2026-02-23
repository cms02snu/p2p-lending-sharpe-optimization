# 내생적 투자집합 하의 포트폴리오 수익률 최적화 (LendingClub P2P 대출)

본 연구에서는 LendingClub P2P 대출에서 각 대출의 개별 샤프비율로 정의하고, 승인 임계값 $$\tau$$를 조정해 $$SR_i \ge \tau$$ 인 대출만 승인했을 때의 포트폴리오 샤프비율을 최대화한다. 즉, 먼저 대출별 부도확률 $$p_i$$를 예측한 뒤 대출별 샤프비율 $$SR_i$$를 계산하고, $$\tau$$에 따라 내생적으로 달라지는 승인 집합으로 포트폴리오를 구성하여 검증 데이터에서 최적의 $$\tau$$를 선택한다.

개별 대출의 평균수익률은 기대수익률로 정의하며, 부도확률과 정상/부도 시 실현수익률을 결합해 $$E[r_i] = (1-p_i)r_{\mathrm{normal},d(i)} + p_ir_{\mathrm{default},d(i)}$$ 로 계산한다. 여기서 $$r_{\mathrm{normal},d}$$ 와 $$r_{\mathrm{default},d}$$ 는 예측 부도확률을 십분위로 나눈 구간 $$d(i)$$ 별로 추정한 정상/부도 조건부 평균 실현수익률로, 위험 수준에 따라 회수율이 달라지는 이질성을 반영한다. 변동성은 개별 대출에서 직접 추정하기 어렵기 때문에, 기대수익률을 여러 구간으로 나눈 뒤 각 구간 내 실현수익률의 횡단면 분산을 위험 대리변수로 사용하여 $$SR_i = E[r_i] / \sigma_{d(i)}$$ 를 계산하고, $$\tau$$ 변화에 따른 평균수익–변동성의 동시 변화를 반영한다.

---

## 폴더 구조

```
lcp2p/
  config.py                 # 실행 설정(dataclass)
  cli.py                    # 커맨드라인 엔트리포인트
  io/
    data.py                 # 데이터 로드/필터링 및 라벨 생성
  preprocess/
    member2.py              # 범주형 전처리 파이프라인
    member3.py              # 연체/부정 기록 전처리
    member4.py              # 계좌/잔액/날짜 파생변수 전처리
    processor.py            # 전처리 통합(DataProcessor)
  finance/
    returns.py              # 실현수익률/국채수익률 테이블 구성(FRED)
  models/
    xgb_default.py          # XGBoost 부도확률 모형 + Optuna 튜닝
  strategy/
    expected_return.py      # 상수 방식 / 십분위 방식 기대수익률
    risk_proxy.py           # 기대수익률 구간화 및 위험 대리변수(sigma_j)
    portfolio.py            # 포트폴리오 SR(tau) 및 격자 탐색
  experiments/
    analysis.py             # K회 반복 교차검증 메인 파이프라인
    plots.py                # Figure 생성 유틸
    sensitivity.py          # J 민감도, 등분위 vs 등간격 비교
  utils/
    repro.py                # 재현성 설정
    io.py                   # CSV 저장 유틸
```

---

## 설치

### 1) 가상환경 (권장)

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -U pip
pip install -e .
```

---

## 실행 방법

### 1) 전체 분석 (K회 반복 + Figure 저장)

```bash
lcp2p-run \
  --data-path /path/to/lending_club_2020_train.csv \
  --save-dir  /path/to/output_dir \
  --k-repeat 50 \
  --sample-frac 0.10 \
  --optuna-trials 25
```

- 결과물은 `--save-dir` 아래에 CSV/PNG로 저장됩니다.
- 기본적으로 아래 산출물을 생성합니다.
  - `results_summary_v4.csv` : 반복별 AUC, tau*, SR, 승인비율 등 요약
  - `tau_grid_all_v4.csv` : 검증 데이터에서의 SR(tau) 격자탐색 결과(반복별)
  - `benchmark_comparison_v4.csv` : Oracle/전부승인/전부거절/제안모형 비교
  - `fig_tau_curve_v4.png`, `fig_tau_hist_v4.png`, `fig_overfit_check_v4.png`, `fig_auc_vs_sharpe_v4.png`

### 2) 민감도 분석 (옵션)

- CLI에서 `--skip-sensitivity`를 끄면 J 민감도(`J=5,10,15,20`) 및 등분위 vs 등간격 비교를 추가로 실행합니다.
- 계산량이 크므로 필요시 따로 실행하는 것을 권장합니다.

---

## 주의사항

- FRED 금리 데이터는 실행 시점에 온라인에서 받아옵니다. (네트워크 필요)
- 원자료(`lending_club_2020_train.csv`)는 용량이 크므로 저장소에 포함하지 않습니다.
