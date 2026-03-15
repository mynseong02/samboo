# CIFAR-10N (worse noise) 실험 스크립트 (Windows PowerShell)
# Base: SAM/ASAM/FriendlySAM × rho 8개 = 24 experiments
# BO  : SAMBO/ASAMBO/FSAMBO             =  3 experiments
# Total: 27 experiments

# ── 공통 설정 ──────────────────────────────────────────────────────────────
$GPU        = 0
$SEED       = 1
$DATASET    = "CIFAR10N"
$NOISE_TYPE = "worse"
$NOISE_PATH = "./datasets/CIFAR-N/CIFAR-10_human.pt"
$MODEL      = "resnet18"
$SCHEDULE   = "cosine"
$WD         = 0.001
$EPOCHS     = 200
$BZ         = 128
$LR         = 0.05
$SIGMA      = 1
$LMBDA      = 0.6

# Base 실험용 rho 목록
$RHO_LIST = @(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)

# SAM-BO config
$BO_BUDGET       = 15
$BO_N_INIT       = 5
$BO_EPOCHS       = 1
$BO_ALPHA        = 0.1
$BO_GAMMA        = 1e-7
$BO_PROXY_MODE   = "original"
$BO_RHO_MIN      = 0.01
$BO_RHO_MAX      = 2.0
$BO_EVAL_BATCHES = 10
$BO_EPSILON_K    = 10
$BO_PROXY_SIZE   = 1500

# ── Base 실험 함수 ──────────────────────────────────────────────────────────
function Run-Base {
    param($OPT, $RHO)
    $DST = "results/${OPT}/CIFAR10N/${MODEL}/${OPT}_rho${RHO}_${MODEL}_bz${BZ}_wd${WD}_${NOISE_TYPE}_${SCHEDULE}_seed${SEED}"
    Write-Host "===== [BASE] opt=${OPT}  rho=${RHO} ====="
    $env:CUDA_VISIBLE_DEVICES = $GPU
    python -u train.py `
        --datasets     $DATASET    `
        --arch         $MODEL      `
        --epochs       $EPOCHS     `
        --weight-decay $WD         `
        --randomseed   $SEED       `
        --lr           $LR         `
        --rho          $RHO        `
        --optimizer    $OPT        `
        --save-dir     "${DST}/checkpoints" `
        --log-dir      $DST        `
        --log-name     log         `
        -p             200         `
        --schedule     $SCHEDULE   `
        -b             $BZ         `
        --cutout                   `
        --sigma        $SIGMA      `
        --lmbda        $LMBDA      `
        --noise_type   $NOISE_TYPE `
        --noise_path   $NOISE_PATH
}

# ── BO 실험 함수 ────────────────────────────────────────────────────────────
function Run-BO {
    param($OPT)
    $DST = "results/SAMBO/${OPT}/CIFAR10N/${MODEL}/sambo_budget${BO_BUDGET}_init${BO_N_INIT}_ep${BO_EPOCHS}_${BO_PROXY_MODE}_${MODEL}_bz${BZ}_wd${WD}_${NOISE_TYPE}_${SCHEDULE}_seed${SEED}"
    Write-Host "===== [BO]   opt=${OPT} ====="
    $env:CUDA_VISIBLE_DEVICES = $GPU
    python -u train.py `
        --datasets     $DATASET    `
        --arch         $MODEL      `
        --epochs       $EPOCHS     `
        --weight-decay $WD         `
        --randomseed   $SEED       `
        --lr           $LR         `
        --rho          0.1         `
        --optimizer    $OPT        `
        --save-dir     "${DST}/checkpoints" `
        --log-dir      $DST        `
        --log-name     log         `
        -p             200         `
        --schedule     $SCHEDULE   `
        -b             $BZ         `
        --cutout                   `
        --sigma        $SIGMA      `
        --lmbda        $LMBDA      `
        --noise_type   $NOISE_TYPE `
        --noise_path   $NOISE_PATH `
        --sambo                    `
        --sambo_budget       $BO_BUDGET       `
        --sambo_n_init       $BO_N_INIT       `
        --sambo_epochs       $BO_EPOCHS       `
        --sambo_alpha        $BO_ALPHA        `
        --sambo_gamma        $BO_GAMMA        `
        --sambo_proxy_mode   $BO_PROXY_MODE   `
        --sambo_rho_min      $BO_RHO_MIN      `
        --sambo_rho_max      $BO_RHO_MAX      `
        --sambo_eval_batches $BO_EVAL_BATCHES `
        --sambo_epsilon_k    $BO_EPSILON_K    `
        --sambo_proxy_size   $BO_PROXY_SIZE
}

# ── 실험 루프 ──────────────────────────────────────────────────────────────
$OPTS  = @("SAM", "ASAM", "FriendlySAM")
$TOTAL = $OPTS.Count * $RHO_LIST.Count + $OPTS.Count
$COUNT = 0

Write-Host "Total experiments: ${TOTAL}"
Write-Host ""

# Base experiments
foreach ($OPT in $OPTS) {
    foreach ($RHO in $RHO_LIST) {
        $COUNT++
        Write-Host "[${COUNT}/${TOTAL}]"
        Run-Base $OPT $RHO
        Write-Host ""
    }
}

# BO experiments
foreach ($OPT in $OPTS) {
    $COUNT++
    Write-Host "[${COUNT}/${TOTAL}]"
    Run-BO $OPT
    Write-Host ""
}

Write-Host "All ${TOTAL} experiments done."
