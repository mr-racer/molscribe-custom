#!/bin/bash
# E1-control (optional): same schedule as E1, organic replay only, no metal data. Separates "the metal data helped"
# from "any fine-tuning with fresh augmentation moved the metrics". E1 sees ~406k images per epoch for 6 epochs; the
# control sees the 200k organic SMILES for 12 epochs, i.e. the same number of optimizer steps and the same LR curve.
#   tmux new -s metal_ctrl
#   bash /mnt/hard1/ivans_data/metal_ocsr/code/molscribe/metal_ocsr/run/train_E1_control.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
RUN=${RUN:-E1-control}
mkdir -p $ROOT/runs/$RUN
cd $CODE
$TORCHRUN --nproc_per_node=1 --master_port=$(shuf -n 1 -i 20000-40000) train.py \
  $MODEL_ARGS \
  --data_path $ROOT/data \
  --train_file organic/train_organic.csv \
  --valid_file synth/val_metal.csv,organic/val_uspto_1k.csv \
  --dynamic_indigo --augment --mol_augment --include_condensed \
  --load_path $BASE_CKPT \
  --encoder_lr 5e-5 --decoder_lr 1e-4 --warmup_ratio 0.05 --scheduler cosine \
  --label_smoothing 0.1 --epochs ${EPOCHS:-12} \
  --batch_size ${BATCH:-64} --gradient_accumulation_steps ${ACCUM:-2} \
  --use_checkpoint --fp16 --backend nccl --num_workers 24 \
  --save_path $ROOT/runs/$RUN --save_mode all --print_freq 100 \
  --mlflow_experiment molscribe-metal-ocsr --mlflow_run_name $RUN \
  --do_train 2>&1 | tee $ROOT/runs/$RUN/train.log
echo "done; evaluate with: bash $CODE/metal_ocsr/run/eval_run.sh $RUN"
