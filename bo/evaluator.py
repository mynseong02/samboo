import copy
import torch
import math
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from .config import SAMBOConfig


def _collect_fixed_batches(loader, n: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """loader에서 첫 n개 배치를 CPU에 고정 수집."""
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for i, (x, y) in enumerate(loader):
        if i >= n:
            break
        batches.append((x.cpu(), y.cpu()))
    return batches


def _compute_epsilon_star(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    fixed_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """고정 배치에서 first_step으로 ε*(rho) 평균 계산 후 파라미터·optimizer 상태 복원."""
    optim_backup = copy.deepcopy(optimizer.state_dict())
    eps_sum: Dict[str, torch.Tensor] = {}
    count = 0

    model.eval()
    for x, y in fixed_batches:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.enable_grad():
            loss = criterion(model(x), y)
            loss.mean().backward()

        optimizer.first_step(zero_grad=False)

        for name, p in model.named_parameters():
            old_p = optimizer.state.get(p, {}).get("old_p")
            if old_p is not None:
                eps = (p.data - old_p).detach()
                eps_sum[name] = eps_sum.get(name, torch.zeros_like(eps)) + eps
                p.data.copy_(old_p)

        optimizer.zero_grad(set_to_none=True)
        count += 1

    optimizer.load_state_dict(optim_backup)
    model.train()

    if count == 0 or not eps_sum:
        return None

    return {name: eps / count for name, eps in eps_sum.items()}


def _eval_perturbed_loss(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    fixed_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    eps: Dict[str, torch.Tensor],
) -> float:
    """ε* 섭동을 적용한 상태에서 고정 배치 손실 평가 후 섭동 복원."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in eps:
                p.data.add_(eps[name])

    model.eval()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for x, y in fixed_batches:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            bsz = x.size(0)
            total_loss += float(loss.mean().item()) * bsz
            total_n += bsz

    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in eps:
                p.data.sub_(eps[name])

    model.train()
    return total_loss / max(total_n, 1)


def _weight_norm_sq(model: torch.nn.Module) -> float:
    """모델 파라미터 전체의 L2 norm 제곱 계산."""
    with torch.no_grad():
        return float(sum(p.data.pow(2).sum().item() for p in model.parameters()))


def _eval_clean_loss(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    fixed_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> float:
    """섭동 없이 고정 배치에서 clean loss 평가."""
    model.eval()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for x, y in fixed_batches:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            bsz = x.size(0)
            total_loss += float(loss.mean().item()) * bsz
            total_n += bsz
    model.train()
    return total_loss / max(total_n, 1)


class RhoEvaluator:
    """주어진 rho로 short training 후 PAC-Bayes proxy J(rho)를 평가하는 클래스."""
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_loader,
        train_step_fn: Callable,
        cfg: SAMBOConfig,
        proxy_loader=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.train_step_fn = train_step_fn
        self.cfg = cfg
        self.device = next(model.parameters()).device

        _proxy_src = proxy_loader if proxy_loader is not None else train_loader
        self.proxy_loader = _proxy_src
        n_fixed = max(cfg.eval_n_batches, cfg.epsilon_k)
        self.fixed_batches = _collect_fixed_batches(_proxy_src, n=n_fixed)
        self.eps_batches  = self.fixed_batches[:cfg.epsilon_k]
        self.eval_batches = self.fixed_batches[:cfg.eval_n_batches]

        self.rho_ref    = math.sqrt(cfg.rho_min * cfg.rho_max)
        self.L_S_init   = _eval_clean_loss(model, criterion, self.eval_batches, self.device)
        self.w0_norm_sq = _weight_norm_sq(model)
        src_name = "proxy_loader" if proxy_loader is not None else "train_loader"
        print(f"  [SAM-BO] L_S_init={self.L_S_init:.4f}  w0_norm_sq={self.w0_norm_sq:.2f}"
        f"  rho_ref={self.rho_ref:.5f}  proxy_src={src_name}")

    def evaluate(self, rho: float) -> Dict[str, float]:
        """rho를 적용해 short train 후 proxy J와 세부 메트릭 딕셔너리 반환."""
        if self.cfg.resample_eval_batches:
            n_fixed = max(self.cfg.eval_n_batches, self.cfg.epsilon_k)
            batches = _collect_fixed_batches(self.proxy_loader, n=n_fixed)
            eps_batches  = batches[:self.cfg.epsilon_k]
            eval_batches = batches[:self.cfg.eval_n_batches]
        else:
            eps_batches  = self.eps_batches
            eval_batches = self.eval_batches

        model_backup = copy.deepcopy(self.model.state_dict())
        optim_backup = copy.deepcopy(self.optimizer.state_dict())

        try:
            self._set_rho(rho)
            self._train_epochs(self.cfg.train_budget_epochs)

            eps = _compute_epsilon_star(
                self.model, self.optimizer, self.criterion,
                eps_batches, self.device
            )
            if eps is None:
                inf_val = float("inf")
                return {"J": inf_val, "sharpness_term": inf_val, "reg": inf_val,
                        "clean_loss": inf_val, "perturbed_loss": inf_val, "w_ratio": inf_val}

            clean_loss = _eval_clean_loss(
                self.model, self.criterion, eval_batches, self.device
            )
            perturbed_loss = _eval_perturbed_loss(
                self.model, self.criterion, eval_batches, self.device, eps
            )
            w_norm_sq = _weight_norm_sq(self.model)
            w_ratio = w_norm_sq / max(self.w0_norm_sq, 1e-12)

            if self.cfg.proxy_mode == "original":
                sharpness_term = perturbed_loss
                reg = self.cfg.gamma * w_norm_sq / max(rho ** 2, 1e-12)

            elif self.cfg.proxy_mode == "normalized":
                sharpness_term = perturbed_loss / max(self.L_S_init, 1e-12)
                reg = self.cfg.alpha * w_ratio * (self.rho_ref / max(rho, 1e-12)) ** 2

            else:  # sharpness_gap
                denom = clean_loss + self.L_S_init
                sharpness_term = (perturbed_loss - clean_loss) / max(denom, 1e-12)
                reg = self.cfg.alpha * w_ratio * (self.rho_ref / max(rho, 1e-12)) ** 2

            J = sharpness_term + reg

        finally:
            self.model.load_state_dict(model_backup)
            self.optimizer.load_state_dict(optim_backup)

        return {
            "J":               float(J),
            "sharpness_term":  float(sharpness_term),
            "reg":             float(reg),
            "clean_loss":      float(clean_loss),
            "perturbed_loss":  float(perturbed_loss),
            "w_ratio":         float(w_ratio),
        }

    def _set_rho(self, rho: float) -> None:
        """optimizer param_groups의 rho 값을 갱신."""
        for group in self.optimizer.param_groups:
            if "rho" in group:
                group["rho"] = float(rho)

    def _train_epochs(self, n_epochs: int) -> None:
        """train_loader 전체를 n_epochs 동안 학습."""
        if n_epochs <= 0:
            return
        self.model.train()
        for _ in range(n_epochs):
            for batch in self.train_loader:
                self.train_step_fn(
                    model=self.model,
                    optimizer=self.optimizer,
                    criterion=self.criterion,
                    batch=batch,
                    device=self.device,
                )

# nomalized, shapness x -> original