#!/bin/bash
# E2: dative edge class + R-labelled synthetic images, gentler on organic chemistry than E1.
# vs E1: --edge_classes 8 (padded from the base checkpoint), synth_e2 data (dative lines, 30 % R labels, arrows,
# Co^III / [Fe] metal labels), organic replay 300k : metal ~195k per epoch (60/40), LR 2e-5 / 5e-5 (was 5e-5 / 1e-4),
# 4 epochs (E1 plateaued by epoch 4), checkpoint selection guarded by the organic validation score
# (base model 0.925 post_smiles on val_uspto_1k -> guard 0.915).
#   tmux new -s metal_e2
#   bash /mnt/hard1/ivans_data/metal_ocsr/code/molscribe/metal_ocsr/run/train_E2.sh
# ~2.8 h per epoch on the A6000 (batch 64 x 2, gradient checkpointing, 18.5 GB).
set -euo pipefail
source "$(dirname "$0")/env.sh"
RUN=${RUN:-E2}
BATCH=${BATCH:-64}
ACCUM=${ACCUM:-2}
EPOCHS=${EPOCHS:-4}
GUARD=${GUARD:-0.915}
mkdir -p $ROOT/runs/$RUN
cd $CODE
$TORCHRUN --nproc_per_node=1 --master_port=$(shuf -n 1 -i 20000-40000) train.py \
  $MODEL_ARGS \
  --edge_classes 8 \
  --data_path $ROOT/data \
  --train_file organic/train_organic_300k.csv \
  --aux_file synth_e2/train_metal.csv --coords_file aux_file_exact --aux_profile metal \
  --valid_file synth_e2/val_metal.csv,organic/val_uspto_1k.csv \
  --organic_guard $GUARD \
  --dynamic_indigo --augment --mol_augment --include_condensed \
  --load_path $BASE_CKPT \
  --encoder_lr 2e-5 --decoder_lr 5e-5 --warmup_ratio 0.05 --scheduler cosine \
  --label_smoothing 0.1 --epochs $EPOCHS \
  --batch_size $BATCH --gradient_accumulation_steps $ACCUM \
  ${CKPT_FLAG---use_checkpoint} --fp16 --backend nccl --num_workers 24 \
  --save_path $ROOT/runs/$RUN --save_mode all --print_freq 100 \
  --mlflow_experiment molscribe-metal-ocsr --mlflow_run_name ${RUN}-dative \
  --do_train 2>&1 | tee $ROOT/runs/$RUN/train.log
echo "training done; evaluate: SETS=t1_lebedev_metal,t2_mrbw_metal,t3_general,t4_synth_val,t4e2_synth_val,organic_val bash $CODE/metal_ocsr/run/eval_run.sh $RUN"
