#!/usr/bin/env bash
# Watch gpu40 parity jobs and start the next reference/unified/wave2 step.
set -euo pipefail
SS="${SS:-/data/home/jzlong/codes/SceneShifter}"
CONDA_SH="${CONDA_SH:-/data/home/jzlong/miniconda3/etc/profile.d/conda.sh}"
OUT="${SCENESHIFTER_PARITY_ROOT:-/data/home/jzlong/data/SceneShifter_dataset/results/sceneshifter_unified_parity_0918}"
LOG="$OUT/logs"
mkdir -p "$LOG"

busy_gpus() {
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | wc -l
}

gpu_free() {
  local idx="$1"
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx" | awk '{print int($1)}')"
  [[ "${used:-999}" -lt 800 ]]
}

has_video() {
  local backbone="$1"
  local side="$2"
  find "$OUT/$backbone/$side" -name "*.mp4" ! -name "tmp_*" 2>/dev/null | grep -q .
}

in_flight() {
  local backbone="$1"
  local side="$2"
  pgrep -f "parity_generate.py --side ${side} --backbone ${backbone}" >/dev/null
}

start_side() {
  local backbone="$1"
  local side="$2"
  local gpu="$3"
  local env_name="$4"
  if in_flight "$backbone" "$side"; then
    echo "[watch] $backbone $side already in flight"
    return 0
  fi
  nohup bash -c "
    source '$CONDA_SH'
    conda activate '$env_name'
    export PYTHONPATH='$SS' SCENESHIFTER_PARITY_ROOT='$OUT' CUDA_VISIBLE_DEVICES='$gpu' CUDA_DEVICE_ORDER=PCI_BUS_ID
    python -u '$SS/scripts/parity_generate.py' --side '$side' --backbone '$backbone' --gpu '$gpu'
    python -u '$SS/scripts/compare_parity_videos.py' --backbone '$backbone' --json_out '$OUT/${backbone}_compare.json' || true
  " >>"$LOG/${backbone}.log" 2>&1 &
  echo "[watch] started $backbone $side on GPU$gpu pid=$!"
}

first_free_gpu() {
  local g
  for g in 0 4 5 6 7; do
    if gpu_free "$g"; then
      echo "$g"
      return 0
    fi
  done
  return 1
}

echo "[watch] polling $OUT"
while true; do
  # Wave1 unified after reference videos exist.
  if has_video multitalk reference && ! has_video multitalk unified && ! in_flight multitalk unified && gpu_free 0; then
    start_side multitalk unified 0 multitalk
  fi
  if has_video infinitetalk reference && ! has_video infinitetalk unified && ! in_flight infinitetalk unified && gpu_free 4; then
    start_side infinitetalk unified 4 infinitetalk
  fi
  if has_video fantasytalking reference && ! has_video fantasytalking unified && ! in_flight fantasytalking unified && ! pgrep -f "batch_generate_sceneshifter.py.*fantasytalking" >/dev/null && gpu_free 5; then
    start_side fantasytalking unified 5 multitalk
  fi
  if has_video wan_s2v reference && ! has_video wan_s2v unified && ! in_flight wan_s2v unified && gpu_free 6; then
    start_side wan_s2v unified 6 wans2v
  fi
  if has_video hunyuan_avatar reference && ! has_video hunyuan_avatar unified && ! in_flight hunyuan_avatar unified && gpu_free 7; then
    start_side hunyuan_avatar unified 7 HunyuanVideo-Avatar
  fi

  sleep 45
done
