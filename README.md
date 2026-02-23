# 내생적 투자집합 하의 포트폴리오 수익률 최적화 (LendingClub P2P 대출)

이 저장소는 LendingClub P2P 대출 데이터로, 승인 임계값($$\tau$$)을 조정해 승인되는 대출 집합이 내생적으로 바뀌는 상황에서
포트폴리오 샤프비율(Sharpe ratio)을 최대화하는 파이프라인을 구현합니다. 대출별 부도확률 $$p$$를 예측하고,
각 $$\tau$$에 대해 승인된 대출로 포트폴리오를 구성한 뒤, 검증 데이터에서 샤프비율이 최대가 되는 $$\tau$$를 선택합니다.

기대수익률은 상수 부도수익률 방식과 위험수준 이질성을 반영한 “부도확률 십분위 조건부 수익률” 방식을 지원합니다.
개별 대출은 실현수익률이 1개뿐이라 위험을 직접 추정하기 어려우므로, 기대수익률 구간 내 횡단면 분산을 위험 대리변수로 사용해
$$\tau$$ 변화에 따른 평균수익–변동성의 동시 변화를 포착합니다.

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

## 구현 메모 (보고서와의 대응)

- **1단계(부도확률 모형)**: `models/xgb_default.py`
  - `scale_pos_weight`를 **설정하지 않음**(기본값 1.0) → 예측확률 평균이 실제 부도율에 수렴하도록 유지.
- **2단계(기대수익률)**: `strategy/expected_return.py`
  - (상수 방식) \(E[r_i]=(1-p_i)IRR_i + p_i\bar r_{default}\)
  - (십분위 방식, v4) \(E[r_i]=(1-p_i)\bar r_{normal,d(i)} + p_i\bar r_{default,d(i)}\)
- **3단계(위험 대리변수)**: `strategy/risk_proxy.py`
  - 기대수익률 축 구간화 후 구간 내 표준편차 \(\sigma_j\)를 위험 대리변수로 사용.
- **4단계(의사결정 규칙 및 tau 최적화)**: `strategy/portfolio.py`
  - 개별 대출 스코어: \((E[r_i]-r_{treasury,i})/\sigma_j\)
  - 포트폴리오 샤프: \(SR(\tau)=(R_p(\tau)-\bar r_{treasury}(\tau))/\sigma_p(\tau)\)

---

## 주의사항

- FRED 금리 데이터는 실행 시점에 온라인에서 받아옵니다. (네트워크 필요)
- 원자료(`lending_club_2020_train.csv`)는 용량이 크므로 저장소에 포함하지 않습니다.

---

## 라이선스

과제/연구 재현 목적의 코드 구조 예시입니다. 데이터 및 결과 사용 시 원자료/보고서의 인용 및 사용조건을 확인하세요.
