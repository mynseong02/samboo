"""BO 수렴 분석 스크립트 — bo_result.json을 읽어 시각화."""
import json
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
# python plot_bo.py results\SAMBO\SAM\CIFAR10\resnet18\sambo_budget25_init5_ep1_original_resnet18_bz128_wd0.001_CIFAR10_cosine_seed1\bo_result.json --save bo.png --proxy --save-proxy bo_proxy.png

# ── 인수 파싱 ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('result',        type=str,  help='bo_result.json 경로')
parser.add_argument('--save',        type=str,  default=None, help='BO figure 저장 경로')
parser.add_argument('--proxy',       action='store_true',     help='proxy term 분석 figure 표시')
parser.add_argument('--save-proxy',  type=str,  default=None, help='proxy figure 저장 경로')
args = parser.parse_args()

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
with open(args.result) as f:
    res = json.load(f)

rho_list    = res["rho_list"]
Y           = res["Y"]
best_so_far = res["best_so_far"]
n_init      = res["n_init"]
best_rho    = res["best_rho"]
best_y      = res["best_y"]
n_total     = len(Y)
iters       = list(range(1, n_total + 1))

acq_scores  = res.get("acq_scores",  [None] * n_total)
gp_grid_rho = res.get("gp_grid_rho", None)
gp_mu       = res.get("gp_mu",       None)
gp_sigma    = res.get("gp_sigma",    None)
details     = res.get("details",     [])

# inf 필터 (epsilon 계산 실패 평가 제외)
valid_mask = [math.isfinite(y) for y in Y]

bo_acq_vals  = [acq_scores[i] for i in range(n_init, n_total) if acq_scores[i] is not None]
bo_acq_iters = [i + 1          for i in range(n_init, n_total) if acq_scores[i] is not None]

C_INIT = 'steelblue'
C_BO   = 'tomato'
C_BEST = 'green'

# ════════════════════════════════════════════════════════════════════════════
# Figure 1: BO 수렴 분석 (2×3)
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 9))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

# ── (0,0) J per evaluation ──────────────────────────────────────────────────
# 매 평가의 J 값과 best-so-far 곡선.
# 파란 원=init, 빨간 네모=BO iter. 녹색 선이 지금까지의 최솟값을 추적.
ax = fig.add_subplot(gs[0, 0])
ax.axvline(n_init + 0.5, color='gray', linestyle='--', linewidth=1, label='init / BO')
ax.plot(iters[:n_init], Y[:n_init], 'o', color=C_INIT, label='init',    zorder=3)
ax.plot(iters[n_init:], Y[n_init:], 's', color=C_BO,   label='BO iter', zorder=3)
ax.plot(iters, best_so_far, '-', color=C_BEST, linewidth=1.8, label='best so far')
ax.set_xlabel('Evaluation #'); ax.set_ylabel('J(ρ)')
ax.set_title('J(ρ) per evaluation'); ax.legend(fontsize=7)

# ── (0,1) ρ trajectory ──────────────────────────────────────────────────────
# 각 평가에서 선택된 ρ 값의 시계열. log scale.
# BO가 잘 수렴하면 후반부 점들이 best_rho 근처(점선)에 모여야 함.
ax = fig.add_subplot(gs[0, 1])
ax.axvline(n_init + 0.5, color='gray', linestyle='--', linewidth=1)
ax.plot(iters[:n_init], rho_list[:n_init], 'o', color=C_INIT, label='init')
ax.plot(iters[n_init:], rho_list[n_init:], 's', color=C_BO,   label='BO iter')
ax.axhline(best_rho, color=C_BEST, linestyle=':', linewidth=1.5,
           label=f'best ρ={best_rho:.4f}')
ax.set_xlabel('Evaluation #'); ax.set_ylabel('ρ')
ax.set_yscale('log'); ax.set_title('ρ trajectory (log scale)'); ax.legend(fontsize=7)

# ── (0,2) GP posterior (최종 상태) ──────────────────────────────────────────
# BO 종료 시점의 GP 사후 분포. 주황 곡선=GP 평균, 음영=±2σ.
# 음영이 좁을수록 GP가 해당 구간을 충분히 탐색한 것.
# best_rho(점선) 근처에서 GP 평균이 최솟값을 가져야 good.
ax = fig.add_subplot(gs[0, 2])
if gp_grid_rho is not None:
    gp_rho = np.array(gp_grid_rho)
    gp_m   = np.array(gp_mu)
    gp_s   = np.array(gp_sigma)
    ax.plot(gp_rho, gp_m, '-', color='darkorange', linewidth=1.8, label='GP mean')
    ax.fill_between(gp_rho, gp_m - 2*gp_s, gp_m + 2*gp_s,
                    alpha=0.25, color='darkorange', label='±2σ')
    ax.scatter(rho_list[:n_init], Y[:n_init], marker='o', color=C_INIT,
               zorder=4, s=40, label='init obs')
    ax.scatter(rho_list[n_init:], Y[n_init:], marker='s', color=C_BO,
               zorder=4, s=40, label='BO obs')
    ax.axvline(best_rho, color=C_BEST, linestyle=':', linewidth=1.5)
    ax.set_xscale('log'); ax.set_xlabel('ρ'); ax.set_ylabel('J(ρ)')
    ax.set_title('GP posterior (final)'); ax.legend(fontsize=7)
else:
    ax.text(0.5, 0.5, 'No GP data\n(updated code로 재실행 필요)',
            ha='center', va='center', transform=ax.transAxes, color='gray')
    ax.set_title('GP posterior (final)')

# ── (1,0) best-so-far 수렴 곡선 ─────────────────────────────────────────────
# best_so_far가 빠르게 꺾이고 이후 plateau면 BO가 빠르게 수렴한 것.
# 오른쪽 텍스트: 마지막 5 step의 Δbest와 log10(ρ) std로 수렴 자동 판정.
ax = fig.add_subplot(gs[1, 0])
ax.axvline(n_init + 0.5, color='gray', linestyle='--', linewidth=1)
ax.plot(iters, best_so_far, '-o', color=C_BEST, markersize=4, linewidth=1.8)
ax.set_xlabel('Evaluation #'); ax.set_ylabel('Best J so far')
ax.set_title('Best-so-far convergence')

window = 5
converged = False
rho_std = float('nan')
if len(best_so_far) >= window + 1:
    delta_best = best_so_far[-(window+1)] - best_so_far[-1]
    converged  = delta_best < 0.01
    rho_log_recent = [math.log10(r) for r in rho_list[-window:]]
    rho_std = float(np.std(rho_log_recent))
    label = f"last-{window} Δbest={delta_best:.4f}\nlog10(ρ) std={rho_std:.3f}"
    ax.text(0.97, 0.97, label, transform=ax.transAxes, fontsize=7,
            va='top', ha='right',
            color=C_BEST if converged else C_BO,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

# ── (1,1) ρ vs J scatter ────────────────────────────────────────────────────
# J landscape를 ρ 공간에서 직접 시각화. 숫자=eval 순서.
# 점들이 특정 ρ 구간에 밀집되고 그 구간의 J가 낮으면 BO가 잘 된 것.
ax = fig.add_subplot(gs[1, 1])
ax.scatter(rho_list[:n_init], Y[:n_init], marker='o', color=C_INIT,
           label='init', zorder=3, s=40)
ax.scatter(rho_list[n_init:], Y[n_init:], marker='s', color=C_BO,
           label='BO iter', zorder=3, s=40)
ax.axvline(best_rho, color=C_BEST, linestyle=':', linewidth=1.5,
           label=f'best ρ={best_rho:.4f}')
for i, (r, y) in enumerate(zip(rho_list, Y)):
    if math.isfinite(y):
        ax.annotate(str(i+1), (r, y), fontsize=6, ha='center', va='bottom')
ax.set_xlabel('ρ'); ax.set_ylabel('J(ρ)')
ax.set_xscale('log'); ax.set_title('ρ vs J (landscape view)'); ax.legend(fontsize=7)

# ── (1,2) Acquisition score decay ───────────────────────────────────────────
# BO iter마다 선택된 점의 acquisition score(EI/UCB 값).
# 잘 수렴하면 후반부로 갈수록 값이 줄어들어야 함(탐색할 여지가 없어짐).
# 값이 계속 높으면 GP가 landscape를 아직 못 학습한 것.
ax = fig.add_subplot(gs[1, 2])
if bo_acq_vals:
    ax.plot(bo_acq_iters, bo_acq_vals, 's-', color=C_BO, markersize=5, linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Evaluation #'); ax.set_ylabel('Acquisition score')
    ax.set_title('Acquisition score (BO iters)')
    if len(bo_acq_vals) >= 3:
        z = np.polyfit(bo_acq_iters, bo_acq_vals, 1)
        ax.plot(bo_acq_iters, np.poly1d(z)(bo_acq_iters),
                '--', color='gray', linewidth=1, label=f'trend {z[0]:+.4f}/iter')
        ax.legend(fontsize=7)
else:
    ax.text(0.5, 0.5, 'No acq data\n(updated code로 재실행 필요)',
            ha='center', va='center', transform=ax.transAxes, color='gray')
    ax.set_title('Acquisition score (BO iters)')

# ── 수렴 요약 콘솔 출력 ─────────────────────────────────────────────────────
valid_Y   = [y for y in Y if math.isfinite(y)]
best_at   = int(np.argmin(Y))
phase     = "init" if best_at < n_init else f"BO iter {best_at - n_init + 1}"
init_best = min(y for y in Y[:n_init] if math.isfinite(y))
bo_best   = min((y for y in Y[n_init:] if math.isfinite(y)), default=float('inf'))
bo_impr   = init_best - bo_best

print(f"Best rho      : {best_rho:.6f}")
print(f"Best J        : {best_y:.6f}")
print(f"Best found at : {phase} (eval #{best_at+1}/{n_total})")
print(f"Init best J   : {init_best:.6f}")
print(f"BO best J     : {bo_best:.6f}")
print(f"BO improvement: {bo_impr:+.6f}  ({'improved' if bo_impr > 0 else 'no improvement'})")

rho_rounded = [round(r, 4) for r in rho_list]
duplicates  = {r: c for r, c in Counter(rho_rounded).items() if c > 1}
if duplicates:
    print(f"Duplicate rho : {duplicates}  -> avoid_observed=True 권장")

if len(best_so_far) >= window + 1:
    print(f"Conv check    : last-{window} Δbest={delta_best:.4f}, "
          f"log10(ρ) std={rho_std:.3f} "
          f"-> {'CONVERGED' if converged else 'NOT converged'}")

fig.suptitle(f"SAM-BO convergence  |  best_rho={best_rho:.5f}  best_J={best_y:.6f}",
             fontsize=11, y=1.01)

if args.save:
    plt.savefig(args.save, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.save}")
else:
    plt.show()

# ════════════════════════════════════════════════════════════════════════════
# Figure 2: Proxy term 분석 (2×3) — --proxy 플래그 시 표시
# ════════════════════════════════════════════════════════════════════════════
if not (args.proxy or args.save_proxy):
    exit(0)

if not details:
    print("No details in JSON — proxy figure 생략.")
    exit(0)

sharp  = [d["sharpness_term"] for d in details]
reg    = [d["reg"]            for d in details]
Ls     = [d["clean_loss"]     for d in details]
Ls_e   = [d["perturbed_loss"] for d in details]
wratio = [d["w_ratio"]        for d in details]
gap    = [e - c for c, e in zip(Ls, Ls_e)]

fig2 = plt.figure(figsize=(18, 9))
gs2  = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.42, wspace=0.32)

# ── (0,0) J 분해: sharpness + reg ───────────────────────────────────────────
# J = sharpness_term + reg 를 누적 영역으로 표시.
# sharpness(주황)가 J의 대부분을 차지해야 정상.
# reg(파랑)가 크면 gamma가 너무 크거나 w_norm이 급격히 증가한 것.
ax = fig2.add_subplot(gs2[0, 0])
ax.axvline(n_init + 0.5, color='gray', linestyle='--', linewidth=1)
ax.stackplot(iters, sharp, reg,
             labels=['sharpness_term', 'reg'],
             colors=['#f4a261', '#457b9d'], alpha=0.75)
ax.plot(iters, Y, 'k-', linewidth=1.2, label='J (total)')
ax.set_xlabel('Evaluation #'); ax.set_ylabel('Value')
ax.set_title('J decomposition: sharpness + reg'); ax.legend(fontsize=7)

# ── (0,1) sharpness vs reg scatter ──────────────────────────────────────────
# 색(viridis)이 eval 순서. 별=best J.
# 이상적으로는 best J 점이 sharpness가 작고 reg도 작은 좌하단에 위치.
ax = fig2.add_subplot(gs2[0, 1])
sc = ax.scatter(sharp, reg, c=iters, cmap='viridis', zorder=3, s=40)
fig2.colorbar(sc, ax=ax, label='Eval #')
best_d = details[int(np.argmin(Y))]
ax.scatter([best_d["sharpness_term"]], [best_d["reg"]],
           marker='*', color=C_BEST, s=150, zorder=5, label='best J')
ax.set_xlabel('sharpness_term'); ax.set_ylabel('reg')
ax.set_title('sharpness vs reg  (color=eval order)'); ax.legend(fontsize=7)

# ── (0,2) clean_loss vs perturbed_loss ──────────────────────────────────────
# Lₛ(파랑)=clean, Lₛ+ε(빨강)=perturbed. 음영=두 값의 차이(sharpness gap).
# gap이 클수록 해당 ρ에서 perturbation 효과가 큰 것.
# 두 곡선이 안정적으로 비슷하면 proxy 노이즈가 적은 것.
ax = fig2.add_subplot(gs2[0, 2])
ax.axvline(n_init + 0.5, color='gray', linestyle='--', linewidth=1)
ax.plot(iters, Ls,   'o-', color='steelblue', markersize=4, linewidth=1.2, label='Lₛ (clean)')
ax.plot(iters, Ls_e, 's-', color='tomato',    markersize=4, linewidth=1.2, label='Lₛ+ε (perturbed)')
ax.fill_between(iters, Ls, Ls_e, alpha=0.15, color='purple', label='gap')
ax.set_xlabel('Evaluation #'); ax.set_ylabel('Loss')
ax.set_title('Clean vs Perturbed loss'); ax.legend(fontsize=7)

# ── (1,0) sharpness gap per eval ────────────────────────────────────────────
# Lₛ+ε − Lₛ = 실제 손실이 perturbation으로 얼마나 올랐는지.
# ρ가 클수록 gap이 커야 함(더 큰 perturbation). 그래야 J가 ρ에 민감.
# gap이 rho와 무관하게 평탄하면 SAM perturbation이 제대로 안 된 것.
ax = fig2.add_subplot(gs2[1, 0])
ax.axvline(n_init + 0.5, color='gray', linestyle='--', linewidth=1)
ax.plot(iters, gap, 'o-', color='purple', markersize=4, linewidth=1.2)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_xlabel('Evaluation #'); ax.set_ylabel('Lₛ+ε − Lₛ')
ax.set_title('Sharpness gap per eval')

# ── (1,1) w_ratio per eval ──────────────────────────────────────────────────
# ||w||² / ||w₀||² : proxy 학습 중 가중치 노름이 얼마나 변했는지.
# 1에 가까울수록 proxy가 초기 상태에 가까운 것(reg 항 의도대로).
# 급격히 증가하면 proxy epoch 동안 가중치가 너무 많이 변한 것 → reg 항 검토.
ax = fig2.add_subplot(gs2[1, 1])
ax.axvline(n_init + 0.5, color='gray', linestyle='--', linewidth=1)
ax.plot(iters, wratio, 'o-', color='slategray', markersize=4, linewidth=1.2)
ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, label='w_ratio=1')
ax.set_xlabel('Evaluation #'); ax.set_ylabel('||w||² / ||w₀||²')
ax.set_title('Weight norm ratio (w_ratio)'); ax.legend(fontsize=7)

# ── (1,2) proxy terms vs ρ ──────────────────────────────────────────────────
# ρ를 x축으로 각 항을 scatter. eval 번호 표시.
# sharpness(주황)가 ρ에 따라 증가하고, 최솟값 근처에 best_rho가 있어야 함.
# reg(파랑)가 작은 ρ에서 급격히 증가하면 reg 항이 landscape를 왜곡하는 것.
ax = fig2.add_subplot(gs2[1, 2])
ax.scatter(rho_list, sharp, marker='o', color='#f4a261', label='sharpness', zorder=3, s=35)
ax.scatter(rho_list, reg,   marker='s', color='#457b9d', label='reg',       zorder=3, s=35)
ax.scatter(rho_list, Y,     marker='^', color='black',   label='J (total)', zorder=4, s=35)
for i, (r, y) in enumerate(zip(rho_list, Y)):
    if math.isfinite(y):
        ax.annotate(str(i+1), (r, y), fontsize=6, ha='center', va='bottom')
ax.axvline(best_rho, color=C_BEST, linestyle=':', linewidth=1.5,
           label=f'best ρ={best_rho:.4f}')
ax.set_xscale('log'); ax.set_xlabel('ρ'); ax.set_ylabel('Value')
ax.set_title('Proxy terms vs ρ'); ax.legend(fontsize=7)

# ── 콘솔 테이블 출력 ─────────────────────────────────────────────────────────
print("\n── Proxy term summary ──────────────────────────────────────────────────")
print(f"{'':5s} {'rho':>8s}  {'J':>8s}  {'sharp':>8s}  {'reg':>8s}  "
      f"{'Ls':>7s}  {'Ls+e':>7s}  {'w_ratio':>7s}  {'gap':>7s}")
best_idx = int(np.argmin(Y))
for i, (r, d) in enumerate(zip(rho_list, details)):
    tag = "* " if i == best_idx else "  "
    g   = d['perturbed_loss'] - d['clean_loss']
    print(f"[{i+1:2d}]{tag}{r:8.5f}  {d['J']:8.6f}  {d['sharpness_term']:8.6f}"
          f"  {d['reg']:8.6f}  {d['clean_loss']:7.4f}  {d['perturbed_loss']:7.4f}"
          f"  {d['w_ratio']:7.4f}  {g:7.4f}")

fig2.suptitle(f"Proxy term analysis  |  best_rho={best_rho:.5f}  best_J={best_y:.6f}",
              fontsize=11, y=1.01)

if args.save_proxy:
    plt.savefig(args.save_proxy, dpi=150, bbox_inches='tight')
    print(f"Saved proxy figure: {args.save_proxy}")
else:
    plt.show()
