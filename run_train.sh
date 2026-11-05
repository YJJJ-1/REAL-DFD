#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
torchrun --standalone --nproc_per_node=8 training/train.py \
  --ddp \
  --detector_path training/config/detector/asym_multi.yaml \
  --no-save_feat
