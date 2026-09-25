# Common environment for the metal-OCSR runs on 10.2.0.181. Source it: `source metal_ocsr/run/env.sh`
export ROOT=/mnt/hard1/ivans_data/metal_ocsr
export CODE=$ROOT/code/molscribe
export PY=$ROOT/env/venv/bin/python
export TORCHRUN="$PY -m torch.distributed.run"
export PYTHONPATH=$CODE
export NO_ALBUMENTATIONS_UPDATE=1
export MLFLOW_TRACKING_URI=http://localhost:5000
export BASE_CKPT=$ROOT/ckpts/base/swin_base_char_aux_1m680k.pth
# architecture of the base checkpoint (must not change for fine-tuning)
export MODEL_ARGS="--encoder swin_base --decoder transformer --formats chartok_coords,edges --coord_bins 64 --sep_xy \
  --input_size 384 --vocab_file $CODE/molscribe/vocab/vocab_chars.json"
