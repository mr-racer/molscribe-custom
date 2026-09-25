#!/bin/bash
# End-to-end smoke test of the E1 pipeline (not a training run): 20 steps, batch 8, tiny validation sets,
# separate MLflow experiment. Checks data loading with exact aux coordinates, the metal augmentation profile,
# loss/backward, multi-set validation with the metal metric, checkpoint saving and MLflow logging.
set -euo pipefail
source "$(dirname "$0")/env.sh"
S=$ROOT/data/smoke
mkdir -p $S $ROOT/runs/smoke
$PY - <<EOF
import pandas as pd
d = '$ROOT/data'
pd.read_csv(f'{d}/organic/train_organic.csv').head(400).to_csv('$S/train_organic.csv', index=False)
pd.read_csv(f'{d}/synth/train_metal.csv').sample(400, random_state=0).to_csv('$S/train_metal.csv', index=False)
pd.read_csv(f'{d}/synth/val_metal.csv').head(48).to_csv('$S/val_metal.csv', index=False)
pd.read_csv(f'{d}/organic/val_uspto_1k.csv').head(48).to_csv('$S/val_uspto.csv', index=False)
EOF
cd $CODE
$TORCHRUN --nproc_per_node=1 --master_port=$(shuf -n 1 -i 20000-40000) train.py \
  $MODEL_ARGS \
  --data_path $ROOT/data \
  --train_file smoke/train_organic.csv \
  --aux_file smoke/train_metal.csv --coords_file aux_file_exact --aux_profile metal \
  --valid_file smoke/val_metal.csv,smoke/val_uspto.csv \
  --dynamic_indigo --augment --mol_augment --include_condensed --save_image \
  --load_path $BASE_CKPT \
  --encoder_lr 5e-5 --decoder_lr 1e-4 --warmup_ratio 0.05 --label_smoothing 0.1 \
  --epochs 1 --train_steps_per_epoch 20 --batch_size ${BATCH:-8} \
  --use_checkpoint --fp16 --backend nccl --num_workers 4 \
  --save_path $ROOT/runs/smoke --save_mode last --print_freq 5 \
  --mlflow_experiment molscribe-metal-ocsr-smoke --mlflow_run_name smoke \
  --do_train 2>&1 | tee $ROOT/runs/smoke/train.log
