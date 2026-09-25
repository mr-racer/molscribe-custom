#!/bin/bash
# E1: stage-1 fine-tune on synthetic organometallics + organic replay (spec section 6).
# Run inside tmux on 10.2.0.181 with the whole GPU free:
#   tmux new -s metal_e1
#   bash /mnt/hard1/ivans_data/metal_ocsr/code/molscribe/metal_ocsr/run/train_E1.sh
# Per epoch: ~206k rendered metal images (aux) + 200k organic SMILES rendered on the fly (train), 6 epochs.
# Checkpoints -> $ROOT/runs/E1/, metrics -> MLflow experiment molscribe-metal-ocsr, run E1-synthetic.
set -euo pipefail
source "$(dirname "$0")/env.sh"
RUN=${RUN:-E1}
BATCH=${BATCH:-64}      # per step; x ACCUM = effective batch 128
# CKPT_FLAG="" disables gradient checkpointing (faster, more GPU memory)
ACCUM=${ACCUM:-2}
EPOCHS=${EPOCHS:-6}
mkdir -p $ROOT/runs/$RUN
cd $CODE
$TORCHRUN --nproc_per_node=1 --master_port=$(shuf -n 1 -i 20000-40000) train.py \
  $MODEL_ARGS \
  --data_path $ROOT/data \
  --train_file organic/train_organic.csv \
  --aux_file synth/train_metal.csv --coords_file aux_file_exact --aux_profile metal \
  --valid_file synth/val_metal.csv,organic/val_uspto_1k.csv \
  --dynamic_indigo --augment --mol_augment --include_condensed \
  --load_path $BASE_CKPT \
  --encoder_lr 5e-5 --decoder_lr 1e-4 --warmup_ratio 0.05 --scheduler cosine \
  --label_smoothing 0.1 --epochs $EPOCHS \
  --batch_size $BATCH --gradient_accumulation_steps $ACCUM \
  ${CKPT_FLAG---use_checkpoint} --fp16 --backend nccl --num_workers 24 \
  --save_path $ROOT/runs/$RUN --save_mode all --print_freq 100 \
  --mlflow_experiment molscribe-metal-ocsr --mlflow_run_name ${RUN}-synthetic \
  --do_train 2>&1 | tee $ROOT/runs/$RUN/train.log
echo "training done; evaluate every epoch with: bash $CODE/metal_ocsr/run/eval_run.sh $RUN"
