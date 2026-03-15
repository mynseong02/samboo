import math
import numpy as np
from typing import Optional, Protocol
from .config import SAMBOConfig as BOConfig


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    """표준 정규 분포 PDF 계산."""
    z = np.asarray(z, dtype=np.float64)
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    """표준 정규 분포 CDF 계산."""
    z = np.asarray(z, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


class Acquisition(Protocol):
    """BO에서 후보 점을 선택하기 위해 GP 예측의 평균과 표준편차를 점수로 변환하는 acquisition function 인터페이스."""
    maximize_score: bool

    def __call__(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray: ...
    def select_best(self, mu: np.ndarray, sigma: np.ndarray) -> int: ...


class ExpectedImprovement:
    """EI (Expected Improvement) acquisition function (minimize 기준)."""

    def __init__(self, cfg: BOConfig):
        self.minimize = True
        self.xi = float(cfg.ei_xi)
        self.f_best: Optional[float] = None
        self.maximize_score = True

    def set_best(self, f_best: float) -> None:
        self.f_best = float(f_best)

    def __call__(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)

        if self.f_best is None:
            return sigma

        improvement = (self.f_best - mu - self.xi) if self.minimize else (mu - self.f_best - self.xi)
        z = improvement / sigma
        ei = improvement * _normal_cdf(z) + sigma * _normal_pdf(z)
        return np.maximum(ei, 0.0)

    def select_best(self, mu: np.ndarray, sigma: np.ndarray) -> int:
        return int(np.argmax(self(mu, sigma)))


class UCBAcquisition:
    """UCB (Upper Confidence Bound) acquisition function (minimize 기준)."""

    def __init__(self, cfg: BOConfig):
        self.beta = float(cfg.ucb_beta)
        self.minimize = True
        self.maximize_score = False

    def __call__(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
        bonus = math.sqrt(self.beta) * sigma
        return mu - bonus if self.minimize else mu + bonus

    def select_best(self, mu: np.ndarray, sigma: np.ndarray) -> int:
        scores = self(mu, sigma)
        return int(np.argmin(scores) if self.minimize else np.argmax(scores))

# UBC x -> EI