$env:CUDA_VISIBLE_DEVICES = "0"
$seed = 1
$datasets = "CIFAR10"
$model = "resnet18"
$schedule = "cosine"
$wd = 0.001
$epoch = 200
$bz = 128
$sigma = 1
$lmbda = 0.6
$opt = "FriendlySAM"

# SAM-BO config
$sambo_budget       = 25
$sambo_n_init       = 5
$sambo_epochs       = 1
$sambo_alpha        = 0.1
$sambo_gamma        = 1e-7
$sambo_proxy_mode   = "original"
$sambo_rho_min      = 0.01
$sambo_rho_max      = 2.0
$sambo_eval_batches = 5
$sambo_epsilon_k    = 5
# proxy_size >= max(epsilon_k, eval_n_batches) * batch_size = 20*128 = 2560
$sambo_proxy_size   = 2600

$DST = "results/SAMBO/$opt/$datasets/$model/sambo_budget${sambo_budget}_init${sambo_n_init}_ep${sambo_epochs}_${sambo_proxy_mode}_${model}_bz${bz}_wd${wd}_${datasets}_${schedule}_seed${seed}"

python -u train.py `
    --datasets $datasets `
    --arch $model `
    --epochs $epoch `
    --weight-decay $wd `
    --randomseed $seed `
    --lr 0.05 `
    --rho 0.1 `
    --optimizer $opt `
    --save-dir "$DST/checkpoints" `
    --log-dir $DST `
    --log-name log `
    -p 200 `
    --schedule $schedule `
    -b $bz `
    --cutout `
    --sigma $sigma `
    --lmbda $lmbda `
    --sambo `
    --sambo_budget $sambo_budget `
    --sambo_n_init $sambo_n_init `
    --sambo_epochs $sambo_epochs `
    --sambo_alpha $sambo_alpha `
    --sambo_gamma $sambo_gamma `
    --sambo_proxy_mode $sambo_proxy_mode `
    --sambo_rho_min $sambo_rho_min `
    --sambo_rho_max $sambo_rho_max `
    --sambo_eval_batches $sambo_eval_batches `
    --sambo_epsilon_k $sambo_epsilon_k `
    --sambo_proxy_size $sambo_proxy_size
