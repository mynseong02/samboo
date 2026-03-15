import math
import time
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from .acquisition import ExpectedImprovement, UCBAcquisition
from .config import SAMBOConfig as BOConfig
from .initialization import init_rho
from .surrogate import GaussianProcessSurrogate


def _make_acquisition(cfg: BOConfig):
    """cfg.acquisition 설정에 따라 acquisition function 객체 생성."""
    name = cfg.acquisition.lower()
    if name == "ucb":
        return UCBAcquisition(cfg)
    if name == "ei":
        return ExpectedImprovement(cfg)
    raise ValueError('acquisition must be "ei" or "ucb".')


def _mask_observed(candidates: np.ndarray, observed: List[float],
                   dedup_decimals: int, dedup_radius: float = 0.0) -> np.ndarray:
    """이미 평가된 포인트를 후보에서 제외하는 불리언 마스크 생성.

    dedup_radius > 0이면 log space 반경 내 후보를 제외.
    dedup_radius == 0이면 dedup_decimals 반올림 방식으로 비교.
    """
    if not observed:
        return np.ones(len(candidates), dtype=bool)
    obs = np.asarray(observed, dtype=np.float64)
    if dedup_radius > 0.0:
        # 각 후보와 모든 관측값의 최소 거리가 radius 초과인 것만 True
        dists = np.abs(candidates[:, None] - obs[None, :])  # (n_cand, n_obs)
        return np.min(dists, axis=1) > dedup_radius
    obs_set = {round(v, dedup_decimals) for v in observed}
    return np.array([round(float(v), dedup_decimals) not in obs_set for v in candidates])


def optimize_acquisition(gp: GaussianProcessSurrogate, acq, cfg: BOConfig, best_log_rho: Optional[float] = None) -> tuple:
    """후보 그리드에서 acquisition score를 최대화하는 log(rho)와 해당 score를 반환."""
    log_min = math.log(cfg.rho_min)
    log_max = math.log(cfg.rho_max)

    candidates = np.linspace(log_min, log_max, cfg.n_candidates, dtype=np.float64)
    if best_log_rho is not None:
        candidates = np.unique(np.append(candidates, float(best_log_rho)))

    if cfg.auto_set_ei_best and hasattr(acq, "f_best") and acq.f_best is None and gp.y_observed:
        acq.set_best(float(np.min(gp.y_observed)))

    mu, sigma = gp.predict(candidates, include_noise=cfg.include_noise_in_sigma)
    scores = acq(mu, sigma)
    maximize = bool(getattr(acq, "maximize_score", False))

    if cfg.avoid_observed and gp.X_observed:
        mask = _mask_observed(candidates, gp.X_observed, cfg.dedup_decimals, cfg.dedup_radius)
        if mask.any():
            scores = np.where(mask, scores, -np.inf if maximize else np.inf)

    if not cfg.refine:
        idx = int(np.argmax(scores) if maximize else np.argmin(scores))
        return float(candidates[idx]), float(scores[idx])

    k = min(cfg.top_k, scores.size)
    valid = np.where(np.isfinite(scores))[0]
    if valid.size > 0:
        s = scores[valid]
        top_idx = valid[np.argsort(s)[-k:] if maximize else np.argsort(s)[:k]]
    else:
        top_idx = np.argsort(scores)[-k:] if maximize else np.argsort(scores)[:k]

    local_grids = [candidates]
    for i in top_idx:
        c = float(candidates[int(i)])
        lo = max(log_min, c - cfg.refine_radius)
        hi = min(log_max, c + cfg.refine_radius)
        local_grids.append(np.linspace(lo, hi, cfg.refine_candidates, dtype=np.float64))

    refined = np.unique(np.concatenate(local_grids))
    mu2, sigma2 = gp.predict(refined, include_noise=cfg.include_noise_in_sigma)
    scores2 = acq(mu2, sigma2)

    if cfg.avoid_observed and gp.X_observed:
        mask2 = _mask_observed(refined, gp.X_observed, cfg.dedup_decimals, cfg.dedup_radius)
        if mask2.any():
            scores2 = np.where(mask2, scores2, -np.inf if maximize else np.inf)

    best_idx = int(np.argmax(scores2) if maximize else np.argmin(scores2))
    return float(refined[best_idx]), float(scores2[best_idx])


def _log_eval(tag: str, rho: float, info: Dict[str, float], elapsed: float,
              best_y: Optional[float] = None, acq_score: Optional[float] = None,
              best_rho: Optional[float] = None, remaining: Optional[float] = None) -> None:
    """BO 평가 결과 한 줄 출력."""
    parts = [
        f"  {tag}",
        f"rho={rho:.5f}",
        f"J={info['J']:.6f}",
    ]
    if best_y is not None:
        parts.append(f"bestJ={best_y:.6f}")
    if best_rho is not None:
        parts.append(f"best_rho={best_rho:.5f}")
    if acq_score is not None:
        parts.append(f"acq={acq_score:.4f}")
    parts += [
        f"sharp={info['sharpness_term']:.4f}",
        f"Ls={info['clean_loss']:.4f}",
        f"Ls+e={info['perturbed_loss']:.4f}",
    ]
    suffix = f"  ({elapsed:.1f}s"
    if remaining is not None:
        suffix += f", remain~{remaining:.0f}s"
    suffix += ")"
    print("  ".join(parts) + suffix)


def run_bayesian_optimization(eval_fn: Callable[[float], Dict[str, float]], cfg: BOConfig) -> Dict[str, Any]:
    """GP 기반 Bayesian Optimization으로 최적 rho 탐색 후 결과 반환."""
    gp = GaussianProcessSurrogate(cfg)
    acq = _make_acquisition(cfg)

    X_log: List[float] = []
    Y: List[float] = []
    acq_scores: List[Optional[float]] = []
    details: List[Dict[str, float]] = []
    eval_times: List[float] = []

    for rho in init_rho(cfg):
        x_log = float(np.log(rho))
        t0 = time.time()
        info = eval_fn(x_log)
        elapsed = time.time() - t0
        eval_times.append(elapsed)
        y = info["J"]
        gp.add_observation(x_log, y)
        X_log.append(x_log)
        Y.append(y)
        details.append(info)
        acq_scores.append(None)
        cur_best_y = float(min(Y))
        _log_eval("[BO init]", rho, info, elapsed, best_y=cur_best_y)

    best_idx = int(np.argmin(Y))
    best_x_log, best_y = X_log[best_idx], Y[best_idx]
    if hasattr(acq, "set_best"):
        acq.set_best(best_y)

    for i in range(cfg.budget):
        x_next, acq_score = optimize_acquisition(gp, acq, cfg, best_log_rho=best_x_log)
        t0 = time.time()
        info = eval_fn(float(x_next))
        elapsed = time.time() - t0
        eval_times.append(elapsed)
        y_next = info["J"]
        gp.add_observation(float(x_next), float(y_next))
        X_log.append(float(x_next))
        Y.append(float(y_next))
        details.append(info)
        acq_scores.append(acq_score)
        remaining = (cfg.budget - i - 1) * (sum(eval_times) / len(eval_times))
        _log_eval(
            f"[BO {i+1:2d}/{cfg.budget}]", math.exp(x_next), info, elapsed,
            best_y=best_y, acq_score=acq_score,
            best_rho=math.exp(best_x_log), remaining=remaining,
        )

        cur_best = int(np.argmin(Y))
        best_x_log, best_y = X_log[cur_best], Y[cur_best]
        if hasattr(acq, "set_best"):
            acq.set_best(best_y)

    best_idx = int(np.argmin(Y))
    rho_list = [float(math.exp(x)) for x in X_log]
    best_so_far = [float(min(Y[:i+1])) for i in range(len(Y))]

    # GP posterior snapshot (최종 상태에서 fine grid 예측)
    log_min = math.log(cfg.rho_min)
    log_max = math.log(cfg.rho_max)
    gp_grid_log = list(np.linspace(log_min, log_max, 300).tolist())
    gp_mu_raw, gp_sigma_raw = gp.predict(np.array(gp_grid_log), include_noise=False)
    gp_grid_rho = [float(math.exp(x)) for x in gp_grid_log]
    gp_mu    = [float(v) for v in gp_mu_raw]
    gp_sigma = [float(v) for v in gp_sigma_raw]

    return {
        "best_rho":     float(math.exp(X_log[best_idx])),
        "best_y":       float(Y[best_idx]),
        "X_log":        X_log,
        "Y":            Y,
        "rho_list":     rho_list,
        "best_so_far":  best_so_far,
        "acq_scores":   acq_scores,
        "details":      details,
        "n_init":       cfg.n_init,
        "gp_grid_rho":  gp_grid_rho,
        "gp_mu":        gp_mu,
        "gp_sigma":     gp_sigma,
    }
# 실행기