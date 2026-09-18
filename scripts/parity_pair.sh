#!/usr/bin/env bash
set -euo pipefail
backbone="${1:?backbone}"
gpu="${2:?gpu}"
env_name="${3:?conda env}"
SS="${SS:-/data/home/jzlong/codes/SceneShifter}"
CONDA_SH="${CONDA_SH:-/data/home/jzlong/miniconda3/etc/profile.d/conda.sh}"
OUT="${SCENESHIFTER_PARITY_ROOT:-/data/home/jzlong/data/SceneShifter_dataset/results/sceneshifter_unified_parity_0918}"
LOG="$OUT/logs"
mkdir -p "$LOG"
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$env_name"
export PYTHONPATH="$SS${PYTHONPATH:+:$PYTHONPATH}"
export SCENESHIFTER_PARITY_ROOT="$OUT"
export CUDA_VISIBLE_DEVICES="$gpu"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
python -u "$SS/scripts/parity_generate.py" --side reference --backbone "$backbone" --gpu "$gpu"
python -u "$SS/scripts/parity_generate.py" --side unified --backbone "$backbone" --gpu "$gpu"
python -u "$SS/scripts/compare_parity_videos.py" --backbone "$backbone" --json_out "$OUT/${backbone}_compare.json"
