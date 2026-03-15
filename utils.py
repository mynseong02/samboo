import os
import sys
import json
import random
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models_imagenet
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
from functools import partial
from timm.models.vision_transformer import VisionTransformer
from models.pyramidnet import PyramidNet
import models


def _worker_init_fn(worker_id, seed=0):
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def set_seed(seed=1):
    """재현 가능한 학습을 위해 모든 난수 시드를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger:
    """stdout을 터미널과 파일에 동시 출력하는 듀얼 로거."""
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ─────────────────────────────── augmentation ─────────────────────────────────

class Cutout:
    """학습 이미지에 무작위 사각형 마스크를 적용하는 Cutout 증강."""
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.half_size = size // 2
        self.p = p

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image
        left   = torch.randint(-self.half_size, image.size(1) - self.half_size, [1]).item()
        top    = torch.randint(-self.half_size, image.size(2) - self.half_size, [1]).item()
        right  = min(image.size(1), left + self.size)
        bottom = min(image.size(2), top  + self.size)
        image[:, max(0, left):right, max(0, top):bottom] = 0
        return image


# ──────────────────────────────── datasets ────────────────────────────────────

def unpickle(file):
    """CIFAR 원본 바이너리 파일을 역직렬화한다."""
    with open(file, "rb") as fo:
        return pickle.load(fo, encoding="latin1")


class cifar_dataset(Dataset):
    """합성 노이즈 라벨을 지원하는 CIFAR-10/100 Dataset (DivideMix 형식).
    Note: 'labeled'/'unlabeled' 모드는 별도 AUCMeter 의존성이 필요하다.
    """
    def __init__(self, dataset='cifar10', r=0.4, noise_mode='sym',
                 root_dir='./datasets/cifar-10-batches-py',
                 transform=None, mode='all', noise_file='cifar10.json',
                 pred=[], probability=[], log=''):
        self.r = r
        self.transform = transform
        if dataset == 'cifar100':
            root_dir = './datasets/cifar-100-python'
        self.mode = mode
        self.transition = {0:0, 2:0, 4:7, 7:7, 1:1, 9:1, 3:5, 5:3, 6:6, 8:8}
        self.noise_file = os.path.join(root_dir, noise_file)

        if self.mode == 'test':
            if dataset == 'cifar10':
                d = unpickle('%s/test_batch' % root_dir)
                self.test_data  = d['data'].reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))
                self.test_label = d['labels']
            elif dataset == 'cifar100':
                d = unpickle('%s/test' % root_dir)
                self.test_data  = d['data'].reshape((10000, 3, 32, 32)).transpose((0, 2, 3, 1))
                self.test_label = d['fine_labels']
        else:
            train_data, train_label = [], []
            if dataset == 'cifar10':
                for n in range(1, 6):
                    d = unpickle('%s/data_batch_%d' % (root_dir, n))
                    train_data.append(d['data'])
                    train_label += d['labels']
                train_data = np.concatenate(train_data)
            elif dataset == 'cifar100':
                d = unpickle('%s/train' % root_dir)
                train_data, train_label = d['data'], d['fine_labels']
            train_data = train_data.reshape((50000, 3, 32, 32)).transpose((0, 2, 3, 1))

            if os.path.exists(self.noise_file):
                noise_label = json.load(open(self.noise_file, "r"))
            else:
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                noise_idx = set(idx[:int(self.r * 50000)])
                max_class = 9 if dataset == 'cifar10' else 99
                for i in range(50000):
                    if i in noise_idx:
                        if noise_mode == 'sym':
                            noise_label.append(random.randint(0, max_class))
                        elif noise_mode == 'asym':
                            noise_label.append(self.transition[train_label[i]])
                    else:
                        noise_label.append(train_label[i])

            if self.mode == 'all':
                self.train_data  = train_data
                self.noise_label = noise_label
            else:
                if self.mode == 'labeled':
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]
                elif self.mode == 'unlabeled':
                    pred_idx = (1 - pred).nonzero()[0]
                self.train_data  = train_data[pred_idx]
                self.noise_label = [noise_label[i] for i in pred_idx]

    def __getitem__(self, index):
        if self.mode == 'labeled':
            img = self.transform(Image.fromarray(self.train_data[index]))
            return img, img, self.noise_label[index], self.probability[index]
        elif self.mode == 'unlabeled':
            img = self.transform(Image.fromarray(self.train_data[index]))
            return img, img
        elif self.mode == 'all':
            img = self.transform(Image.fromarray(self.train_data[index]))
            return img, self.noise_label[index]
        else:
            img = self.transform(Image.fromarray(self.test_data[index]))
            return img, self.test_label[index]

    def __len__(self):
        return len(self.train_data) if self.mode != 'test' else len(self.test_data)


class cifar_dataloader:
    """합성 노이즈 CIFAR DataLoader 팩토리 (DivideMix 형식)."""
    def __init__(self, dataset='cifar10', r=0.2, noise_mode='sym',
                 batch_size=256, num_workers=4, cutout=False,
                 root_dir='', log='', noise_file='cifar10.json'):
        self.dataset, self.r, self.noise_mode = dataset, r, noise_mode
        self.batch_size, self.num_workers     = batch_size, num_workers
        self.root_dir, self.log, self.noise_file = root_dir, log, noise_file

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        aug = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
               transforms.ToTensor(), normalize]
        if cutout:
            aug.append(Cutout())
        self.transform_train = transforms.Compose(aug)
        self.transform_test  = transforms.Compose([transforms.ToTensor(), normalize])

    def get_loader(self, seed=42):
        """학습/검증 DataLoader 튜플을 반환한다."""
        train_set = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode,
                                  r=self.r, root_dir='./datasets/cifar-10-batches-py',
                                  transform=self.transform_train, mode='all',
                                  noise_file=self.noise_file)
        val_set   = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode,
                                  r=self.r, root_dir='./datasets/cifar-10-batches-py',
                                  transform=self.transform_test, mode='test',
                                  noise_file=self.noise_file)
        return (DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                           num_workers=self.num_workers, pin_memory=True,
                           worker_init_fn=partial(_worker_init_fn, seed=seed)),
                DataLoader(val_set,   batch_size=self.batch_size, shuffle=False,
                           num_workers=self.num_workers, pin_memory=True))


CIFAR10N_PATH  = './datasets/CIFAR-N/CIFAR-10_human.pt'
CIFAR100N_PATH = './datasets/CIFAR-N/CIFAR-100_human.pt'


class CIFAR10N(datasets.CIFAR10):
    """실제 인간 노이즈 라벨이 포함된 CIFAR-10N 데이터셋.
    noise_type: clean / aggre / worse / rand1 / rand2 / rand3
    """
    NOISE_TYPES = {
        'clean': 'clean_label',
        'aggre': 'aggre_label',
        'worse': 'worse_label',
        'rand1': 'random_label1',
        'rand2': 'random_label2',
        'rand3': 'random_label3',
    }

    def __init__(self, root, noise_type='worse', noise_path=CIFAR10N_PATH, **kwargs):
        super().__init__(root, **kwargs)
        assert noise_type in self.NOISE_TYPES, \
            f"noise_type must be one of {list(self.NOISE_TYPES.keys())}"
        noise_file = torch.load(noise_path, weights_only=False)
        clean = noise_file['clean_label'].tolist()
        assert clean == self.targets, \
            "CIFAR-10N clean_label order does not match torchvision CIFAR-10!"
        noisy = noise_file[self.NOISE_TYPES[noise_type]].tolist()
        noise_rate = 1 - sum(a == b for a, b in zip(noisy, clean)) / len(clean)
        self.targets = noisy
        print(f'CIFAR-10N  noise_type={noise_type}  noise_rate={noise_rate:.3f}')


class CIFAR100N(datasets.CIFAR100):
    """실제 인간 노이즈 라벨이 포함된 CIFAR-100N 데이터셋.
    noise_type: clean / noisy
    """
    NOISE_TYPES = {
        'clean': 'clean_label',
        'noisy': 'noisy_label',
    }

    def __init__(self, root, noise_type='noisy', noise_path=CIFAR100N_PATH, **kwargs):
        super().__init__(root, **kwargs)
        assert noise_type in self.NOISE_TYPES, \
            f"noise_type must be one of {list(self.NOISE_TYPES.keys())}"
        noise_file = torch.load(noise_path, weights_only=False)
        clean = noise_file['clean_label'].tolist()
        assert clean == self.targets, \
            "CIFAR-100N clean_label order does not match torchvision CIFAR-100!"
        noisy = noise_file[self.NOISE_TYPES[noise_type]].tolist()
        noise_rate = 1 - sum(a == b for a, b in zip(noisy, clean)) / len(clean)
        self.targets = noisy
        print(f'CIFAR-100N  noise_type={noise_type}  noise_rate={noise_rate:.3f}')


_CIFAR_NORMALIZE = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


def _train_transforms(cutout=True):
    """CIFAR 학습용 augmentation 파이프라인을 생성한다."""
    aug = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
           transforms.ToTensor(), _CIFAR_NORMALIZE]
    if cutout:
        aug.append(Cutout())
    return transforms.Compose(aug)


def _val_transforms():
    """CIFAR 검증용 정규화 변환을 반환한다."""
    return transforms.Compose([transforms.ToTensor(), _CIFAR_NORMALIZE])


def get_datasets_cutout(args):
    """args 설정에 따라 학습/검증 DataLoader 쌍을 생성해 반환한다."""
    cutout    = args.cutout
    n_workers = args.workers
    batch     = args.batch_size
    noise_type = getattr(args, 'noise_type', 'worse')

    if args.datasets == 'CIFAR10':
        train_set = datasets.CIFAR10('./datasets/', train=True, download=True,
                                     transform=_train_transforms(cutout))
        val_set   = datasets.CIFAR10('./datasets/', train=False,
                                     transform=_val_transforms())

    elif args.datasets == 'CIFAR10_noise':
        loader = cifar_dataloader(dataset='cifar10', r=args.noise_ratio,
                                  batch_size=batch, num_workers=n_workers, cutout=cutout)
        return loader.get_loader(seed=args.randomseed)

    elif args.datasets == 'CIFAR10N':
        kw = {}
        if getattr(args, 'noise_path', None):
            kw['noise_path'] = args.noise_path
        train_set = CIFAR10N('./datasets/', noise_type=noise_type, train=True,
                             download=True, transform=_train_transforms(cutout), **kw)
        val_set   = datasets.CIFAR10('./datasets/', train=False, transform=_val_transforms())

    elif args.datasets == 'CIFAR100N':
        kw = {}
        if getattr(args, 'noise_path', None):
            kw['noise_path'] = args.noise_path
        train_set = CIFAR100N('./datasets/', noise_type=noise_type, train=True,
                              download=True, transform=_train_transforms(cutout), **kw)
        val_set   = datasets.CIFAR100('./datasets/', train=False, transform=_val_transforms())

    elif args.datasets == 'CIFAR100':
        train_set = datasets.CIFAR100('./datasets/', train=True, download=True,
                                      transform=_train_transforms(cutout))
        val_set   = datasets.CIFAR100('./datasets/', train=False, transform=_val_transforms())

    else:
        raise ValueError(f"Unknown dataset: {args.datasets}")

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch, shuffle=True,
        num_workers=n_workers, pin_memory=True,
        worker_init_fn=partial(_worker_init_fn, seed=args.randomseed))
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=128, shuffle=False,
        num_workers=n_workers, pin_memory=True)
    return train_loader, val_loader


def get_model(args):
    """args.arch와 args.datasets에 맞는 모델 인스턴스를 반환한다."""
    print(f'Model: {args.arch}')

    if args.datasets in ('CIFAR10', 'CIFAR10_noise', 'CIFAR10N'):
        num_classes = 10
    elif args.datasets in ('CIFAR100', 'CIFAR100N'):
        num_classes = 100
    elif args.datasets == 'ImageNet':
        num_classes = 1000
    else:
        raise ValueError(f"Unknown dataset: {args.datasets}")

    if 'deit' in args.arch:
        model = VisionTransformer(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=num_classes,
            drop_rate=args.drop, drop_path_rate=args.drop_path,
        )
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True,
        )
        checkpoint["model"].pop('head.weight', None)
        checkpoint["model"].pop('head.bias', None)
        model.load_state_dict(checkpoint["model"], strict=False)
        return model

    if args.datasets == 'ImageNet':
        return models_imagenet.__dict__[args.arch]()

    if args.arch == 'PyramidNet110':
        return PyramidNet(110, 270, num_classes)

    model_cfg = getattr(models, args.arch)
    return model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)


# ──────────────────────────────── optimizers ──────────────────────────────────

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al., ICLR 2021)."""
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """ε*(ρ) 섭동을 w에 적용해 worst-case 파라미터로 이동한다."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """w+ε*에서 계산한 gradient로 base optimizer step 후 w를 복원한다."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """closure를 사용하는 SAM 단일 step."""
        assert closure is not None, "SAM requires closure."
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        """SAM 섭동 스케일 계산용 gradient norm을 반환한다."""
        shared_device = self.param_groups[0]["params"][0].device
        return torch.norm(torch.stack([
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"] if p.grad is not None
        ]), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class FriendlySAM(torch.optim.Optimizer):
    """FriendlySAM: momentum 방향 성분을 제거한 SAM 변형 (Li et al., NeurIPS 2023)."""
    def __init__(self, params, base_optimizer, rho=0.05, sigma=1.0, lmbda=0.9,
                 adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.sigma = sigma
        self.lmbda = lmbda
        print(f'FriendlySAM sigma: {self.sigma} lambda: {self.lmbda}')

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """momentum 방향을 제거한 gradient로 ε* 섭동을 적용한다."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.clone()
                if "momentum" not in self.state[p]:
                    self.state[p]["momentum"] = grad
                else:
                    p.grad -= self.state[p]["momentum"] * self.sigma
                    self.state[p]["momentum"] = (
                        self.state[p]["momentum"] * self.lmbda + grad * (1 - self.lmbda)
                    )

        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """w+ε*에서 계산한 gradient로 base optimizer step 후 w를 복원한다."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """closure를 사용하는 FriendlySAM 단일 step."""
        assert closure is not None, "FriendlySAM requires closure."
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        """FriendlySAM 섭동 스케일 계산용 gradient norm을 반환한다."""
        shared_device = self.param_groups[0]["params"][0].device
        return torch.norm(torch.stack([
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"] if p.grad is not None
        ]), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class ASAM(torch.optim.Optimizer):
    """Adaptive SAM (Kwon et al., ICML 2021).
    ε*(ρ,w) = ρ · T_w² · ∇f / ‖T_w · ∇f‖   (T_w = diag(|w| + η))
    selective_weight=True: bias 파라미터는 plain SAM 섭동을 사용한다.
    """
    def __init__(self, params, base_optimizer, rho=0.05, eta=0.01,
                 selective_weight=True, model=None, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        defaults = dict(rho=rho, eta=eta, selective_weight=selective_weight, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self._param_name = (
            {id(p): n for n, p in model.named_parameters()} if model is not None else {}
        )

    def _t_w(self, p, group):
        """T_w = |w|+η (weight) 또는 None (bias → plain SAM)을 반환한다."""
        if not group["selective_weight"]:
            return torch.abs(p.data) + group["eta"]
        if self._param_name:
            if "weight" not in self._param_name.get(id(p), ""):
                return None
        elif p.ndim == 1:
            return None
        return torch.abs(p.data) + group["eta"]

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """ASAM ε* = ρ·T_w²·∇f/‖T_w·∇f‖ 섭동을 w에 적용한다."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "old_p" not in state:
                    state["old_p"] = torch.empty_like(p.data)
                state["old_p"].copy_(p.data)
                t_w = self._t_w(p, group)
                e_w = (p.grad if t_w is None else p.grad * t_w * t_w) * scale.to(p)
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """w+ε*에서 계산한 gradient로 base optimizer step 후 w를 복원한다."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "old_p" in self.state[p]:
                    p.data.copy_(self.state[p]["old_p"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """closure를 사용하는 ASAM 단일 step."""
        assert closure is not None, "ASAM requires closure."
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        """‖T_w · ∇f‖를 계산한다 (ASAM 섭동 정규화 분모)."""
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                t_w = self._t_w(p, group)
                g = p.grad if t_w is None else p.grad * t_w
                norms.append(g.norm(p=2).to(shared_device))
        return (torch.norm(torch.stack(norms), p=2) if norms
                else torch.tensor(0.0, device=shared_device))

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

"""
전체 seed 흐름 (BO 내부)

① 진입 시점 — main() 최상단


set_seed(args.randomseed)
# torch.manual_seed, np.random.seed, random.seed, cudnn.deterministic 설정
# 이후 모든 PyTorch/NumPy 연산의 전역 random state가 여기서 결정됨
② 모델 초기화


model = get_model(args)
# set_seed 직후이므로 같은 seed → 같은 weight 초기화
→ PyTorch global state 소비

③ proxy_loader 구성 — 독립적인 NumPy RNG


_rng = np.random.RandomState(args.randomseed)  # 새 RNG 객체, 전역 state와 무관
_proxy_indices = _rng.choice(...)
# proxy_loader shuffle=False → 인덱스 순서 고정
→ 전역 NumPy state와 분리. seed만 같으면 항상 동일한 1000개 인덱스

④ SAMBOConfig seed


sambo_cfg = SAMBOConfig(seed=args.randomseed)
⑤ 초기 rho 샘플링 — 또 다른 독립 RNG


# initialization.py - init_rho()
rng = np.random.default_rng(seed)  # cfg.seed = args.randomseed, 새 RNG 객체
# lhs_log: 구간 균등 분할 후 각 구간 내 uniform 샘플
→ ③과도 분리된 독립 RNG. seed 같으면 항상 동일한 n_init개 초기 rho

⑥ 고정 배치 수집 — deterministic


# evaluator.py __init__
self.fixed_batches = _collect_fixed_batches(proxy_loader, n=n_fixed)
# proxy_loader shuffle=False → 순서 결정론적
→ 새로운 random 소비 없음. ③의 indices 순서 그대로

⑦ BO 루프 — 각 evaluate(rho) 내부


# evaluate() 시작
model_backup = deepcopy(model.state_dict())
optim_backup = deepcopy(optimizer.state_dict())
# ↑ random state는 백업 안 함

# _train_epochs(): train_loader(shuffle=True) 사용
# → PyTorch global state 소비 (set_seed 이후 누적)
# → 매 evaluate() 호출마다 다른 배치 순서로 학습

# _compute_epsilon_star(): fixed_batches 사용 → deterministic
# _eval_clean_loss(): fixed_batches 사용 → deterministic
# _eval_perturbed_loss(): fixed_batches 사용 → deterministic

# finally:
self.model.load_state_dict(model_backup)    # ✓ 복원
self.optimizer.load_state_dict(optim_backup) # ✓ 복원
# torch random state                         # ✗ 복원 안 함
→ evaluate(ρ₁) 후 소비된 PyTorch state 그대로 evaluate(ρ₂) 진입

→ ρ₁ → ρ₂ 순서로 평가하면 ρ₂의 학습 배치가,

　 ρ₂ → ρ₁ 순서로 평가할 때와 달라짐

⑧ GP/Acquisition — random 없음


# surrogate.py: Cholesky, grid search MLE → 완전 결정론적
# runner.py: linspace grid, argmax → 완전 결정론적
요약 도식


set_seed(S)
    │
    ├─ PyTorch global state ──→ 모델 초기화
    │                         → _train_epochs() 배치 셔플 (소비 후 미복원)
    │
    ├─ np.RandomState(S) ─────→ proxy_indices (③, 독립)
    │
    ├─ np.default_rng(S) ─────→ 초기 rho 샘플 (⑤, 독립)
    │
    └─ fixed_batches ─────────→ ε*, clean/perturbed loss (결정론적, random 없음)
핵심: J(ρ) 평가의 loss 계산 자체는 고정 배치로 결정론적이지만, 그 전에 수행되는 short training은 PyTorch global state를 소비하므로 평가 순서가 바뀌면 학습 결과(→ J값)도 미세하게 달라질 수 있다.
재현성은 완전 확보
"""