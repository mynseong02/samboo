import argparse
import os
import time
import numpy as np
import random
import math
import sys

from torch.nn.modules.batchnorm import _BatchNorm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils import *

# Parse arguments
parser = argparse.ArgumentParser(description='SAM / SAM-BO training')
parser.add_argument('--EXP', metavar='EXP', help='experiment name', default='SGD')
parser.add_argument('--arch', '-a', metavar='ARCH',
                    help='model architecture')
parser.add_argument('--datasets', metavar='DATASETS', default='CIFAR10', type=str,
                    help='training dataset')
parser.add_argument('--optimizer', metavar='OPTIMIZER', default='sgd', type=str,
                    help='optimizer for training')
parser.add_argument('--schedule', metavar='SCHEDULE', default='step', type=str,
                    help='lr schedule type (step | cosine)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='SGD momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100 iterations)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--wandb', dest='wandb', action='store_true',
                    help='log statistics to wandb')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision (16-bit)')
parser.add_argument('--save-dir', dest='save_dir',
                    help='directory to save trained models',
                    default='save_temp', type=str)
parser.add_argument('--log-dir', dest='log_dir',
                    help='directory to save logs',
                    default='save_temp', type=str)
parser.add_argument('--log-name', dest='log_name',
                    help='log file name',
                    default='log', type=str)
parser.add_argument('--randomseed',
                    help='random seed for training and initialization',
                    type=int, default=1)
parser.add_argument('--rho', default=0.1, type=float,
                    metavar='RHO', help='rho for SAM perturbation radius')
parser.add_argument('--cutout', dest='cutout', action='store_true',
                    help='use cutout data augmentation')
parser.add_argument('--sigma', default=1, type=float,
                    metavar='S', help='sigma for FriendlySAM')
parser.add_argument('--lmbda', default=0.95, type=float,
                    metavar='L', help='lambda for FriendlySAM')
parser.add_argument('--eta', default=0.01, type=float,
                    metavar='ETA', help='eta for ASAM adaptive scaling')
parser.add_argument('--selective_weight', action='store_true', default=True,
                    help='ASAM: use plain SAM for bias/BN params')

parser.add_argument('--noise_ratio', default=0.5, type=float,
                    metavar='N', help='noise ratio for dataset')
parser.add_argument('--noise_type', default='worse', type=str,
                    choices=['clean', 'aggre', 'worse', 'rand1', 'rand2', 'rand3', 'noisy'],
                    help='CIFAR-N noise type (CIFAR10N: clean/aggre/worse/rand1-3, CIFAR100N: clean/noisy)')
parser.add_argument('--noise_path', default=None, type=str,
                    help='path to CIFAR-N human label file (.pt). overrides default path in utils.py')

parser.add_argument('--img_size', type=int, default=224, help='input image resolution')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='drop path rate (default: 0.1)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='drop block rate (default: None)')
parser.add_argument('--patch', dest='patch', type=int, default=4)
parser.add_argument('--dimhead', dest='dimhead', type=int, default=512)
parser.add_argument('--convkernel', dest='convkernel', type=int, default=8)

# SAM-BO arguments
parser.add_argument('--sambo', action='store_true', help='enable SAM-BO automatic rho search')
parser.add_argument('--sambo_budget', type=int, default=15, help='number of BO iterations')
parser.add_argument('--sambo_n_init', type=int, default=5, help='number of initial random points')
parser.add_argument('--sambo_epochs', type=int, default=2, help='epochs per rho evaluation')
parser.add_argument('--sambo_alpha', type=float, default=0.1, help='regularization strength alpha for normalized proxy')
parser.add_argument('--sambo_gamma', type=float, default=1e-4, help='gamma for original proxy mode')
parser.add_argument('--sambo_proxy_mode', type=str, default='normalized',
                    choices=['original', 'normalized', 'sharpness_gap'])
parser.add_argument('--sambo_rho_min', type=float, default=0.001, help='minimum rho search bound')
parser.add_argument('--sambo_rho_max', type=float, default=2.0, help='maximum rho search bound')
parser.add_argument('--sambo_eval_batches', type=int, default=20, help='number of fixed batches for loss evaluation')
parser.add_argument('--sambo_epsilon_k', type=int, default=5, help='number of fixed batches for epsilon* computation')
parser.add_argument('--sambo_proxy_size', type=int, default=1000, help='proxy loader subset size')

best_prec1 = 0


def disable_running_stats(model):
    """배치 정규화 running stats 업데이트 비활성화 (SAM second step용)."""
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0
    model.apply(_disable)


def enable_running_stats(model):
    """배치 정규화 running stats 업데이트 활성화 (SAM first step용)."""
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
    model.apply(_enable)


# Training statistics
train_loss = []
train_err = []
ori_train_loss = []
ori_train_err = []
test_loss = []
test_err = []
arr_time = []

args = parser.parse_args()

if args.wandb:
    import wandb
    wandb.init(project="TWA", entity="nblt")
    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    wandb.run.name = args.EXP + date


def get_model_param_vec(model):
    """모델 파라미터 전체를 1D 벡터로 반환."""
    vec = []
    for name, param in model.named_parameters():
        vec.append(param.data.detach().reshape(-1))
    return torch.cat(vec, 0)


def _cosine_annealing(step, total_steps, lr_max, lr_min):
    """코사인 어닐링 스케줄 multiplier 계산."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def get_cosine_annealing_scheduler(optimizer, epochs, steps_per_epoch, base_lr):
    """전체 epoch에 걸쳐 코사인 어닐링 LR 스케줄러 생성."""
    lr_min = 0.0
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _cosine_annealing(step, total_steps, 1, lr_min / base_lr),
    )
    return scheduler


def main():
    """메인 학습 루프: 데이터 준비, optimizer 설정, SAM-BO 탐색, 전체 학습 실행."""
    global args, best_prec1
    global train_loss, train_err, ori_train_loss, ori_train_err, test_loss, test_err, arr_time

    set_seed(args.randomseed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(args.log_dir, args.log_name))
    print('save dir:', args.save_dir)
    print('log dir:', args.log_dir)

    model = get_model(args)
    model.cuda()

    if args.resume:
        ckpt_path = os.path.join(args.save_dir, args.resume)
        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(ckpt_path)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    print('lambda:', args.lmbda)
    print('cutout:', args.cutout)
    train_loader, val_loader = get_datasets_cutout(args)
    print(len(train_loader))
    print(len(train_loader.dataset))

    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()

    print('optimizer:', args.optimizer)

    if args.optimizer == 'SAM':
        base_optimizer = torch.optim.SGD
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=0,
                        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                        nesterov=False)
    elif args.optimizer == 'SAM_adamw':
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=0,
                        lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'FriendlySAM':
        base_optimizer = torch.optim.SGD
        optimizer = FriendlySAM(model.parameters(), base_optimizer, rho=args.rho,
                                sigma=args.sigma, lmbda=args.lmbda, adaptive=0,
                                lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=False)
    elif args.optimizer == 'FriendlySAM_adamw':
        base_optimizer = torch.optim.AdamW
        optimizer = FriendlySAM(model.parameters(), base_optimizer, rho=args.rho,
                                sigma=args.sigma, lmbda=args.lmbda, adaptive=0,
                                lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'ASAM':
        base_optimizer = torch.optim.SGD
        optimizer = ASAM(model.parameters(), base_optimizer, rho=args.rho, eta=args.eta,
                         selective_weight=args.selective_weight, model=model,
                         lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                         nesterov=False)
    elif args.optimizer == 'ASAM_adamw':
        base_optimizer = torch.optim.AdamW
        optimizer = ASAM(model.parameters(), base_optimizer, rho=args.rho, eta=args.eta,
                         selective_weight=args.selective_weight, model=model,
                         lr=args.lr, weight_decay=args.weight_decay)

    print(optimizer)

    if args.schedule == 'step':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer.base_optimizer, milestones=[60, 120, 160], gamma=0.2,
            last_epoch=args.start_epoch - 1)
    elif args.schedule == 'cosine':
        lr_scheduler = get_cosine_annealing_scheduler(
            optimizer, args.epochs, len(train_loader), args.lr)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # SAM-BO: automatic rho search
    if args.sambo:
        from bo import SAMBOConfig, RhoEvaluator, run_bayesian_optimization

        print('\n===== SAM-BO: searching rho =====')

        sambo_cfg = SAMBOConfig(
            rho_min=args.sambo_rho_min,
            rho_max=args.sambo_rho_max,
            n_init=args.sambo_n_init,
            budget=args.sambo_budget,
            train_budget_epochs=args.sambo_epochs,
            alpha=args.sambo_alpha,
            gamma=args.sambo_gamma,
            proxy_mode=args.sambo_proxy_mode,
            eval_n_batches=args.sambo_eval_batches,
            epsilon_k=args.sambo_epsilon_k,
            seed=args.randomseed,
        )

        def _train_step_fn(model, optimizer, criterion, batch, device):
            x, y = batch
            x, y = x.to(device), y.to(device)
            enable_running_stats(model)
            pred = model(x)
            loss = criterion(pred, y)
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)
            disable_running_stats(model)
            loss2 = criterion(model(x), y)
            loss2.mean().backward()
            optimizer.second_step(zero_grad=True)

        _rng = np.random.RandomState(args.randomseed)
        _proxy_n = min(args.sambo_proxy_size, len(train_loader.dataset))
        _proxy_indices = _rng.choice(len(train_loader.dataset), size=_proxy_n, replace=False)
        proxy_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_loader.dataset, _proxy_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

        evaluator = RhoEvaluator(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            train_step_fn=_train_step_fn,
            cfg=sambo_cfg,
            proxy_loader=proxy_loader,
        )

        result = run_bayesian_optimization(
            eval_fn=lambda x_log: evaluator.evaluate(math.exp(x_log)),
            cfg=sambo_cfg,
        )

        best_rho = result["best_rho"]
        print(f'SAM-BO done: best rho = {best_rho:.6f}  (J = {result["best_y"]:.6f})')
        print(f'rho {args.rho:.4f} -> SAM-BO rho {best_rho:.6f}')

        # BO 수렴 요약 출력
        n_init = result["n_init"]
        best_so_far = result["best_so_far"]
        rho_list = result["rho_list"]
        best_found_at = int(np.argmin(result["Y"]))
        phase = "init" if best_found_at < n_init else f"BO iter {best_found_at - n_init + 1}"
        print(f'  best found at: {phase}  (eval #{best_found_at + 1}/{len(rho_list)})')
        print(f'  J improvement after init: {best_so_far[n_init-1]:.6f} -> {best_so_far[-1]:.6f}')
        print(f'  rho trajectory: {[f"{r:.4f}" for r in rho_list]}')

        # JSON 저장
        import json
        bo_result_path = os.path.join(args.log_dir, 'bo_result.json')
        with open(bo_result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'  BO result saved: {bo_result_path}')

        print('===== SAM-BO done, starting full training =====\n')

        # apply best rho
        args.rho = best_rho
        for group in optimizer.param_groups:
            if "rho" in group:
                group["rho"] = best_rho

    print('Start training:', args.start_epoch, '->', args.epochs)

    for epoch in range(args.start_epoch, args.epochs):
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, lr_scheduler, epoch)
        prec1 = validate(val_loader, model, criterion)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

    print('train loss:', train_loss)
    print('train err:', train_err)
    print('test loss:', test_loss)
    print('test err:', test_err)
    print('ori train loss:', ori_train_loss)
    print('ori train err:', ori_train_err)
    print('time:', arr_time)
    print(f'\n===== Final Results =====')
    print(f'Best Test Acc : {best_prec1:.2f}%')
    print(f'Final Test Acc: {(1 - test_err[-1]) * 100:.2f}%')
    if args.sambo:
        print(f'SAM-BO rho*  : {args.rho:.6f}')


def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch):
    """한 epoch 학습 수행: SAM two-step forward-backward."""
    global train_loss, train_err, ori_train_loss, ori_train_err, arr_time

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    total_loss, total_err = 0, 0
    ori_total_loss, ori_total_err = 0, 0

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        enable_running_stats(model)

        # first forward-backward step
        predictions = model(input_var)
        loss = criterion(predictions, target_var)
        loss.mean().backward()
        optimizer.first_step(zero_grad=True)

        # second forward-backward step
        disable_running_stats(model)
        output_adv = model(input_var)
        loss_adv = criterion(output_adv, target_var)
        loss_adv.mean().backward()
        optimizer.second_step(zero_grad=True)

        lr_scheduler.step()

        output = predictions.float()
        loss = loss.float()

        total_loss += loss.item() * input_var.shape[0]
        total_err += (output.max(dim=1)[1] != target_var).sum().item()

        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        ori_total_loss += loss_adv.item() * input_var.shape[0]
        ori_total_err += (output_adv.max(dim=1)[1] != target_var).sum().item()

        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

    print('Total time for epoch [{0}] : {1:.3f}'.format(epoch, batch_time.sum))

    train_loss.append(total_loss / len(train_loader.dataset))
    train_err.append(total_err / len(train_loader.dataset))
    ori_train_loss.append(ori_total_loss / len(train_loader.dataset))
    ori_train_err.append(ori_total_err / len(train_loader.dataset))

    if args.wandb:
        wandb.log({"train loss": total_loss / len(train_loader.dataset)})
        wandb.log({"train acc": 1 - total_err / len(train_loader.dataset)})

    arr_time.append(batch_time.sum)


def validate(val_loader, model, criterion):
    """검증 데이터셋에서 모델 성능 평가."""
    global test_err, test_loss

    total_loss = 0
    total_err = 0

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            total_loss += loss.item() * input_var.shape[0]
            total_err += (output.max(dim=1)[1] != target_var).sum().item()

            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    test_loss.append(total_loss / len(val_loader.dataset))
    test_err.append(total_err / len(val_loader.dataset))

    if args.wandb:
        wandb.log({"test loss": total_loss / len(val_loader.dataset)})
        wandb.log({"test acc": 1 - total_err / len(val_loader.dataset)})

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """학습 모델 체크포인트 저장."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


class AverageMeter(object):
    """현재 값과 누적 평균을 추적하는 미터."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """지정된 top-k에 대한 분류 정확도 계산."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

'''
모델 및 데이터셋 확장
데이터 로드 관련해서도
'''