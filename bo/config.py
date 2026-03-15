from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class SAMBOConfig:
    """SAM-BO 전체 하이퍼파라미터 설정을 담는 데이터클래스."""
    rho_min: float = 1e-2 # rho 탐색 범위 (log scale)
    rho_max: float = 2.0 # rho 탐색 범위 (log scale) 
    n_init: int = 5  # 초기 탐색을 위한 rho 샘플 개수
    init_strategy: str = "lhs_log" # 초기 탐색 전략: "log_uniform" | "lhs_log" | "normal_clipped"
    seed: int = 42 # 재현성 있는 초기 샘플링을 위한 랜덤 시드
    init_center: float = 0.1 # normal_clipped 전략에서 초기 샘플링 중심값
    init_sigma: float = 0.5 # normal_clipped 전략에서 초기 샘플링 표준편차

    acquisition: str = "ei" # acquisition function: "ei" (Expected Improvement) | "ucb" (Upper Confidence Bound)
    ei_xi: float = 0.01 # EI에서 개선 여유를 조절하는 파라미터
    auto_set_ei_best: bool = True # EI에서 f_best를 자동으로 설정할지 여부 (최소 관측값으로)
    ucb_beta: float = 2.0 # UCB에서 탐색-활용 균형 조절 파라미터

    kernel_type: str = "matern52" # GP 커널 유형: "matern52" | "rbf"
    lengthscale: float = 1 # GP 커널의 lengthscale (거리 척도)
    noise_var: float = 1e-4 # GP 관측 노이즈 분산
    prior_var: float = 1.0 # GP 사전 분산 (함수 변화량 척도)
    jitter: float = 1e-6   # GP 수치 안정성을 위한 jitter 항
    normalize_targets: bool = True # GP 타겟 정규화 여부 (평균 0, 표준편차 1)
    min_y_std: float = 1e-6 # GP 예측 표준편차의 최솟값 (수치 안정성)
    fit_hyperparams: bool = True # GP 하이퍼파라미터 최적화 여부
    fit_every: int = 1 # GP 하이퍼파라미터 최적화 빈도 (에포크 단위)
    ls_grid: Optional[List[float]] = field(default_factory=lambda: [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]) # GP lengthscale 후보 그리드
    noise_grid: Optional[List[float]] = field(default_factory=lambda: [1e-5, 1e-4, 1e-3, 1e-2]) # GP 노이즈 분산 후보 그리드 
    prior_grid: Optional[List[float]] = field(default_factory=lambda: [0.1, 0.5, 1.0, 2.0]) # GP 사전 분산 후보 그리드
    budget: int = 15 # BO 예산 (평가 횟수)
    train_budget_epochs: int = 2 # 각 BO iteration마다 SAM 모델을 학습할 에포크 수
    eval_n_batches: int = 5 # 각 BO iteration마다 평가에 사용할 배치 수 (평균 성능 안정화)
    epsilon_k: int = 5  # SAM-BO에서 rho를 업데이트하지 않고 고정할 초기 iteration 수
    alpha: float = 0.1 # SAM-BO에서 rho 업데이트 시 이전 값과의 지수 이동 평균에 사용할 alpha
    gamma: float = 1e-7 #  SAM-BO에서 rho 업데이트 시 개선이 충분한지 판단하는 임계값
    proxy_mode: str = "normalized"  # "original" | "normalized" | "sharpness_gap"
    n_candidates: int = 500 # BO에서 후보 rho를 샘플링할 때 사용할 점 개수

    include_noise_in_sigma: bool = False # acquisition 함수에서 GP 예측의 표준편차에 관측 노이즈 포함 여부
    avoid_observed: bool = True # acquisition 최적화 시 이미 평가된 rho 후보를 제외할지 여부
    dedup_radius: float = 0.07 # avoid_observed가 True일 때 이미 평가된 log(rho)에서 제외할 반경 (log scale). 0이면 dedup_decimals 사용
    dedup_decimals: int = 3 # dedup_radius=0일 때 log(rho) 반올림 소수점 자릿수
    refine: bool = False # acquisition 최적화 후 후보 rho를 국소적으로 추가 샘플링하여 최적화할지 여부
    top_k: int = 3 # refine가 True일 때 acquisition 점수가 가장 좋은 상위 k 후보를 국소 최적화 대상으로 선택할 개수
    refine_radius: float = 0.5 # refine가 True일 때 후보 rho 주변을 추가 샘플링할 반경 (log scale)
    refine_candidates: int = 100 # refine가 True일 때 각 후보 주변에 샘플링할 점 개수

    # 랜덤성 옵션
    resample_eval_batches: bool = False # True면 매 evaluate()마다 proxy_loader에서 배치 재샘플링
    random_grid_search: bool = False    # True면 GP hyperparameter grid를 랜덤 서브샘플링
    random_grid_frac: float = 0.5       # random_grid_search 시 각 grid에서 탐색할 비율

'''
gamma, epoch, eval_n_batches, epsilon_k, 고정된 배치를 사용할 것인지, 평가함수
평가함수와 고정배치여부가 중요해보임 그후로 gp관련

GP 초기값 (fit 전 처음 몇 포인트에서 중요)

파라미터	현재	추천	이유
lengthscale	0.5	1.0~1.5	Matern5/2에서 l=0.5이면 거리 0.5 떨어진 점 간 상관 0.1로 너무 단절. log 공간 폭(4.6) 대비 1.01.5가 적절
noise_var	1e-5	1e-4	mini-batch stochasticity로 J(ρ)에 노이즈 존재. 1e-5는 거의 결정론 가정 → GP 과적합 위험
prior_var	1.0	1.0	normalize_targets=True이면 y 표준화 후 분산~1, 유지
Hyperparameter grid (fit_hyperparams가 여기서 MLE 탐색)

파라미터	현재	추천
ls_grid	[0.05, 0.1, 0.2, 0.5, 1.0, 2.0]	[0.3, 0.5, 1.0, 1.5, 2.0, 3.0]
noise_grid	[1e-6, 1e-5, 1e-4, 1e-3]	[1e-5, 1e-4, 1e-3, 1e-2]
prior_grid	[0.1, 0.5, 1.0, 2.0]	[0.3, 0.5, 1.0, 2.0, 4.0]
ls_grid: 0.05/0.1은 너무 지역적 → 노이즈 과적합. log 공간 구조 반영해 0.3~3.0으로
noise_grid: 1e-6은 사실상 결정론 가정. proxy stochasticity 감안해 상한을 1e-2까지
prior_grid: 4.0 추가 — loss landscape가 급격히 변하는 경우 대비
'''