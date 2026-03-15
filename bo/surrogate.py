import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from .config import SAMBOConfig as BOConfig


class GaussianProcessSurrogate:
    """Matern5/2 커널 기반 GP Surrogate (Cholesky 분해 + MLE 하이퍼파라미터 최적화)."""
    def __init__(self, cfg: BOConfig):
        self.lengthscale = float(cfg.lengthscale)
        self.noise_var = float(cfg.noise_var)
        self.prior_var = float(cfg.prior_var)
        self.jitter = float(cfg.jitter)
        self.kernel_type = str(cfg.kernel_type)
        self.normalize_targets = bool(cfg.normalize_targets)
        self.fit_hyperparams = bool(cfg.fit_hyperparams)
        self.fit_every = int(cfg.fit_every)
        self.ls_grid = None if cfg.ls_grid is None else np.asarray(cfg.ls_grid, dtype=np.float64)
        self.noise_grid = None if cfg.noise_grid is None else np.asarray(cfg.noise_grid, dtype=np.float64)
        self.prior_grid = None if cfg.prior_grid is None else np.asarray(cfg.prior_grid, dtype=np.float64)
        self.min_y_std = float(cfg.min_y_std)
        self.random_grid_search = bool(cfg.random_grid_search)
        self.random_grid_frac = float(cfg.random_grid_frac)

        self.X_observed: List[float] = []
        self.y_observed: List[float] = []
        self.y_mean: float = 0.0
        self.y_std: float = 1.0
        self._need_update: bool = True
        self._L: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._n_cache_updates: int = 0

    def add_observation(self, x: float, y: float) -> None:
        """새 관측값 (x, y) 추가 후 캐시 무효화."""
        self.X_observed.append(float(x))
        self.y_observed.append(float(y))
        self._need_update = True

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Matern5/2 또는 RBF 커널 행렬 계산."""
        X1 = np.asarray(X1, dtype=np.float64).reshape(-1, 1)
        X2 = np.asarray(X2, dtype=np.float64).reshape(1, -1)
        d = np.abs(X1 - X2)

        if self.kernel_type == "rbf":
            return self.prior_var * np.exp(-(d * d) / (2.0 * self.lengthscale ** 2))

        r = d / self.lengthscale
        sqrt5_r = math.sqrt(5.0) * r
        return self.prior_var * (1.0 + sqrt5_r + (5.0 / 3.0) * (r * r)) * np.exp(-sqrt5_r)

    def _kernel_diag(self, X: np.ndarray) -> np.ndarray:
        """커널 대각 성분 (prior variance) 반환."""
        return self.prior_var * np.ones(np.asarray(X).reshape(-1).shape[0], dtype=np.float64)

    def _compute_cholesky(self, K: np.ndarray) -> np.ndarray:
        """jitter를 점진적으로 늘려가며 안정적인 Cholesky 분해 수행."""
        n = K.shape[0]
        I = np.eye(n, dtype=np.float64)
        for i in range(8):
            try:
                return np.linalg.cholesky(K + max(self.jitter, 1e-12) * (10.0 ** i) * I)
            except np.linalg.LinAlgError:
                pass
        raise np.linalg.LinAlgError("Cholesky failed even after jitter.")

    def _log_marginal_likelihood(self, X: np.ndarray, y_n: np.ndarray) -> float:
        """GP 로그 주변 우도 계산 (하이퍼파라미터 선택 기준)."""
        n = X.shape[0]
        K = self._kernel(X, X) + self.noise_var * np.eye(n, dtype=np.float64)
        L = self._compute_cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_n))
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        return float(-0.5 * (y_n @ alpha) - 0.5 * logdet - 0.5 * n * math.log(2.0 * math.pi))

    def _maybe_fit_hyperparams(self, X: np.ndarray, y_n: np.ndarray) -> None:
        """grid search MLE로 lengthscale/noise_var/prior_var 최적화."""
        if not self.fit_hyperparams or X.shape[0] < 5:
            return
        if (self._n_cache_updates % self.fit_every) != 0:
            return

        ls_grid    = self.ls_grid    if self.ls_grid    is not None else np.array([0.15, 0.25, 0.4, 0.65, 1.0, 1.6])
        noise_grid = self.noise_grid if self.noise_grid is not None else np.array([1e-6, 1e-5, 1e-4, 1e-3])
        prior_grid = self.prior_grid if self.prior_grid is not None else np.array([0.3, 0.6, 1.0, 2.0, 4.0])

        if self.random_grid_search:
            rng = np.random.default_rng()
            def _sub(g):
                k = max(1, int(np.ceil(len(g) * self.random_grid_frac)))
                return rng.choice(g, size=k, replace=False)
            ls_grid, noise_grid, prior_grid = _sub(ls_grid), _sub(noise_grid), _sub(prior_grid)

        best_lml = -1e100
        best = cur = (self.lengthscale, self.noise_var, self.prior_var)

        for ls in ls_grid:
            for nv in noise_grid:
                for pv in prior_grid:
                    self.lengthscale, self.noise_var, self.prior_var = float(ls), float(nv), float(pv)
                    try:
                        lml = self._log_marginal_likelihood(X, y_n)
                    except np.linalg.LinAlgError:
                        continue
                    if lml > best_lml:
                        best_lml, best = lml, (float(ls), float(nv), float(pv))

        self.lengthscale, self.noise_var, self.prior_var = best if np.isfinite(best_lml) else cur

    def _update_cache(self) -> None:
        """Cholesky 분해 및 alpha 캐시 업데이트."""
        if not self._need_update:
            return

        n = len(self.X_observed)
        if n == 0:
            self._L = self._alpha = None
            self._need_update = False
            return

        X = np.asarray(self.X_observed, dtype=np.float64)
        y = np.asarray(self.y_observed, dtype=np.float64)

        if self.normalize_targets:
            self.y_mean = float(np.mean(y))
            self.y_std = float(max(np.std(y), self.min_y_std))
            y_n = (y - self.y_mean) / self.y_std
        else:
            self.y_mean, self.y_std, y_n = 0.0, 1.0, y

        self._maybe_fit_hyperparams(X, y_n)

        K = self._kernel(X, X) + self.noise_var * np.eye(n, dtype=np.float64)
        L = self._compute_cholesky(K)
        self._L = L
        self._alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_n))
        self._need_update = False
        self._n_cache_updates += 1

    def predict(self, X_test: np.ndarray, include_noise: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """테스트 포인트에서 GP 사후 평균과 표준편차 예측."""
        X_test = np.asarray(X_test, dtype=np.float64).reshape(-1)

        if len(self.X_observed) == 0:
            mu = np.zeros(X_test.shape[0], dtype=np.float64)
            var = self._kernel_diag(X_test) + (self.noise_var if include_noise else 0.0)
            return mu, np.sqrt(np.maximum(var, 1e-12))

        self._update_cache()
        X = np.asarray(self.X_observed, dtype=np.float64)
        K_star = self._kernel(X_test, X)
        mu_n = K_star @ self._alpha

        v = np.linalg.solve(self._L, K_star.T)
        var_n = np.maximum(self._kernel_diag(X_test) - np.sum(v * v, axis=0), 1e-12)
        if include_noise:
            var_n = var_n + self.noise_var

        if self.normalize_targets:
            return mu_n * self.y_std + self.y_mean, np.sqrt(var_n) * self.y_std
        return mu_n, np.sqrt(var_n)

    def get_hyperparams(self) -> Dict[str, float]:
        """현재 GP 하이퍼파라미터 딕셔너리 반환."""
        return {"lengthscale": self.lengthscale, "noise_var": self.noise_var, "prior_var": self.prior_var}
    
# RBF x -> Matern5/2