#!/bin/bash
# CIFAR-10N (worse noise) 실험 스크립트
# Base: SAM/ASAM/FriendlySAM × rho 6개 = 18 experiments
# BO  : SAMBO/ASAMBO/FSAMBO             =  3 experiments
# Total: 21 experiments

# ── 공통 설정 ──────────────────────────────────────────────────────────────
GPU=0
SEED=1
DATASET="CIFAR10N"
NOISE_TYPE="worse"
NOISE_PATH="/path/to/CIFAR-10_human.pt"   # <-- 실제 경로로 수정
MODEL="resnet18"
SCHEDULE="cosine"
WD=0.001
EPOCHS=200
BZ=128
LR=0.05
SIGMA=1
LMBDA=0.6

# Base 실험용 rho 목록 (6개, BO 탐색 범위 0.01~2.0과 동일)
RHO_LIST=(0.01 0.05 0.1 0.5 1.0 2.0 5.0 10.0)

# SAM-BO config (run_sambo.ps1과 동일)
BO_BUDGET=25 #15
BO_N_INIT=5 #5
BO_EPOCHS=1
BO_ALPHA=0.1
BO_GAMMA=1e-7
BO_PROXY_MODE="original"
BO_RHO_MIN=0.01
BO_RHO_MAX=2.0
BO_EVAL_BATCHES=5 #10
BO_EPSILON_K=5 # 10
BO_PROXY_SIZE=2600 #1500

# SAM-BO config
# $sambo_budget       = 25
# $sambo_n_init       = 5
# $sambo_epochs       = 1
# $sambo_alpha        = 0.1
# $sambo_gamma        = 1e-7
# $sambo_proxy_mode   = "original"
# $sambo_rho_min      = 0.01
# $sambo_rho_max      = 2.0
# $sambo_eval_batches = 5
# $sambo_epsilon_k    = 5
# # proxy_size >= max(epsilon_k, eval_n_batches) * batch_size = 20*128 = 2560
# $sambo_proxy_size   = 2600

# ── Base 실험 (fixed rho) ──────────────────────────────────────────────────
run_base() {
    local OPT=$1
    local RHO=$2
    local DST="results/${OPT}/CIFAR10N/${MODEL}/${OPT}_rho${RHO}_${MODEL}_bz${BZ}_wd${WD}_${NOISE_TYPE}_${SCHEDULE}_seed${SEED}"

    echo "===== [BASE] opt=${OPT}  rho=${RHO} ====="
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --datasets    ${DATASET}   \
        --arch        ${MODEL}     \
        --epochs      ${EPOCHS}    \
        --weight-decay ${WD}       \
        --randomseed  ${SEED}      \
        --lr          ${LR}        \
        --rho         ${RHO}       \
        --optimizer   ${OPT}       \
        --save-dir    "${DST}/checkpoints" \
        --log-dir     "${DST}"     \
        --log-name    log          \
        -p 200                     \
        --schedule    ${SCHEDULE}  \
        -b            ${BZ}        \
        --cutout                   \
        --sigma       ${SIGMA}     \
        --lmbda       ${LMBDA}     \
        --noise_type  ${NOISE_TYPE} \
        --noise_path  ${NOISE_PATH}
}

# ── BO 실험 ────────────────────────────────────────────────────────────────
run_bo() {
    local OPT=$1
    local DST="results/SAMBO/${OPT}/CIFAR10N/${MODEL}/sambo_budget${BO_BUDGET}_init${BO_N_INIT}_ep${BO_EPOCHS}_${BO_PROXY_MODE}_${MODEL}_bz${BZ}_wd${WD}_${NOISE_TYPE}_${SCHEDULE}_seed${SEED}"

    echo "===== [BO]   opt=${OPT} ====="
    CUDA_VISIBLE_DEVICES=${GPU} python -u train.py \
        --datasets    ${DATASET}   \
        --arch        ${MODEL}     \
        --epochs      ${EPOCHS}    \
        --weight-decay ${WD}       \
        --randomseed  ${SEED}      \
        --lr          ${LR}        \
        --rho         0.1          \
        --optimizer   ${OPT}       \
        --save-dir    "${DST}/checkpoints" \
        --log-dir     "${DST}"     \
        --log-name    log          \
        -p 200                     \
        --schedule    ${SCHEDULE}  \
        -b            ${BZ}        \
        --cutout                   \
        --sigma       ${SIGMA}     \
        --lmbda       ${LMBDA}     \
        --noise_type  ${NOISE_TYPE} \
        --noise_path  ${NOISE_PATH} \
        --sambo                    \
        --sambo_budget       ${BO_BUDGET}       \
        --sambo_n_init       ${BO_N_INIT}       \
        --sambo_epochs       ${BO_EPOCHS}       \
        --sambo_alpha        ${BO_ALPHA}        \
        --sambo_gamma        ${BO_GAMMA}        \
        --sambo_proxy_mode   ${BO_PROXY_MODE}   \
        --sambo_rho_min      ${BO_RHO_MIN}      \
        --sambo_rho_max      ${BO_RHO_MAX}      \
        --sambo_eval_batches ${BO_EVAL_BATCHES} \
        --sambo_epsilon_k    ${BO_EPSILON_K}    \
        --sambo_proxy_size   ${BO_PROXY_SIZE}
}

# ── 실험 루프 ──────────────────────────────────────────────────────────────
OPTS=("SAM" "ASAM" "FriendlySAM")
N_BASE=$(( ${#OPTS[@]} * ${#RHO_LIST[@]} ))   
N_BO=${#OPTS[@]}                               
TOTAL=$(( N_BASE + N_BO ))                     
COUNT=0

echo "Total experiments: ${TOTAL}"
echo ""

# Base experiments
for OPT in "${OPTS[@]}"; do
    for RHO in "${RHO_LIST[@]}"; do
        COUNT=$((COUNT + 1))
        echo "[${COUNT}/${TOTAL}]"
        run_base "${OPT}" "${RHO}"
        echo ""
    done
done

# BO experiments
for OPT in "${OPTS[@]}"; do
    COUNT=$((COUNT + 1))
    echo "[${COUNT}/${TOTAL}]"
    run_bo "${OPT}"
    echo ""
done

echo "All ${TOTAL} experiments done."
