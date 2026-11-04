#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "用法: bash run_test.sh /path/to/test/avg/ckpt_best.pth [dataset ...]" >&2
  exit 2
fi
ckpt="$1"
shift
cd "$(dirname "$0")"
if [ ! -f "$ckpt" ]; then
  echo "权重不存在: $ckpt" >&2
  exit 1
fi
args=()
if [ "$#" -gt 0 ]; then
  args=(--test_dataset "$@")
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python training/test.py \
  --detector_path training/config/detector/asym_multi_vggface2_repro.yaml \
  --weights_path "$ckpt" \
  --config-loader \
  --no-save-results \
  "${args[@]}"
