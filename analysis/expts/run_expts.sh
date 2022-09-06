set -e

base_dir=${1:-$(dirname $(dirname $(pwd)))}
run="python -m braincode"
props="task-*"
models="code-*"
joint="task-content+task-structure+task-nodes+task-lines"
kwargs="-m FisherCorr --score_only"

cd $base_dir

$run mvpa -t $props $kwargs
$run prda -t $props $kwargs

$run nlea -t $joint $kwargs
$run prea -t $joint $kwargs

$run nlea -t $models $kwargs
$run cnlea $kwargs