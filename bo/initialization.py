import numpy as np
from .config import SAMBOConfig as BOConfig


def init_rho(cfg: BOConfig) -> np.ndarray:
    """초기 탐색 전략(log_uniform / lhs_log / normal_clipped)으로 rho 초기 샘플 생성."""
    low, high, n = float(cfg.rho_min), float(cfg.rho_max), int(cfg.n_init)
    strategy, seed = str(cfg.init_strategy), int(cfg.seed)

    if low <= 0 or high <= low:
        raise ValueError("rho_min/rho_max must satisfy 0 < rho_min < rho_max.")
    if n <= 0:
        return np.empty((0,), dtype=np.float64)

    rng = np.random.default_rng(seed)
    log_low, log_high = np.log(low), np.log(high)

    if strategy == "log_uniform":
        rho = np.exp(rng.uniform(log_low, log_high, size=n))

    elif strategy == "lhs_log":
        bins = np.linspace(log_low, log_high, n + 1)
        log_rho = rng.uniform(bins[:-1], bins[1:])
        rng.shuffle(log_rho)
        rho = np.exp(log_rho)

    elif strategy == "normal_clipped":
        log_center = np.log(float(cfg.init_center))
        log_rho = np.clip(rng.normal(log_center, float(cfg.init_sigma), size=n), log_low, log_high)
        rho = np.exp(log_rho)

    else:
        raise ValueError(f"Unknown init strategy: {strategy}")

    return rho.astype(np.float64)

# normal_clipped x -> lhs