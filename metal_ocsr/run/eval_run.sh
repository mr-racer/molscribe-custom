#!/bin/bash
# Evaluate every epoch checkpoint of a run on T1-T4 and log to the run's MLflow entry (step = epoch).
#   bash metal_ocsr/run/eval_run.sh E1
# Checkpoint selection stays with the validation score (best_valid.json); these test curves are diagnostics.
set -euo pipefail
source "$(dirname "$0")/env.sh"
RUN=$1
RUN_DIR=$ROOT/runs/$RUN
RUN_ID=$(cat $RUN_DIR/mlflow_run_id.txt)
cd $CODE
for ck in $(ls $RUN_DIR/*_ep*.pth | sort -V); do
  ep=$(( $(basename $ck .pth | sed 's/.*_ep//') + 1 ))
  out=$ROOT/eval/$RUN/epoch_$(printf %02d $ep)
  [ -f $out/metrics.json ] && { echo "skip epoch $ep"; continue; }
  $PY -m metal_ocsr.eval_metal --root $ROOT --ckpt $ck --out $out --mlflow_run_id $RUN_ID --step $ep --batch_size 32
done
echo "best epoch by validation: $(python3 -c "import json;print(json.load(open('$RUN_DIR/best_valid.json'))['epoch'])")"
