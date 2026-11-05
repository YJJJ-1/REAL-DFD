#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "Usage: bash run_test.sh /path/to/test/avg/ckpt_best.pth [dataset ...]" >&2
  exit 2
fi
checkpoint_path="$(realpath "$1")"
shift
if [ ! -f "$checkpoint_path" ]; then
  echo "Checkpoint not found: $checkpoint_path" >&2
  exit 1
fi
cd "$(dirname "$0")"
dataset_args=()
if [ "$#" -gt 0 ]; then
  dataset_args=(--test_dataset "$@")
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python training/test.py \
  --detector_path training/config/detector/asym_multi.yaml \
  --weights_path "$checkpoint_path" \
  --config-loader \
  --no-save-results \
  "${dataset_args[@]}"
