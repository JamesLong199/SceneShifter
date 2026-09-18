#!/usr/bin/env bash
# Launch SceneShifter reference-then-unified parity jobs on gpu40.
set -euo pipefail

SS="${SS:-/data/home/jzlong/codes/SceneShifter}"
CONDA_SH="${CONDA_SH:-/data/home/jzlong/miniconda3/etc/profile.d/conda.sh}"
OUT="${SCENESHIFTER_PARITY_ROOT:-/data/home/jzlong/data/SceneShifter_dataset/results/sceneshifter_unified_parity_0918}"
LOG="$OUT/logs"
CMP="$SS/scripts/compare_parity_videos.py"

source "$CONDA_SH"
mkdir -p "$LOG" "$OUT/_assets/wan_animate_012"

prepare_animate_assets() {
  local dest="$OUT/_assets/wan_animate_012"
  cp -f "$SS/configs/parity/assets/wan_animate_012_frame_prompt.json" "$dest/012_frame_prompt.json"
  if [[ -d /tmp/wan_animate_mask_demo_self/exp_012/process ]]; then
    rsync -a /tmp/wan_animate_mask_demo_self/exp_012/process/ "$dest/process/"
  elif [[ ! -d "$dest/process" ]]; then
    echo "[launch] missing Animate 012 process assets at /tmp/wan_animate_mask_demo_self/exp_012/process" >&2
    return 1
  fi
}

ensure_animate_worktree() {
  local src="/data/home/jzlong/codes/Wan2.2"
  local dest="/data/home/jzlong/codes/Wan2.2-animate"
  if [[ ! -d "$dest/.git" ]]; then
    git -C "$src" worktree add "$dest" animate_sceneshifter || \
      git -C "$src" worktree add -b animate_sceneshifter "$dest" origin/animate_sceneshifter
  fi
  git -C "$dest" checkout animate_sceneshifter
}

run_pair() {
  local backbone="$1"
  local gpu="$2"
  local env_name="$3"
  SS="$SS" SCENESHIFTER_PARITY_ROOT="$OUT" CONDA_SH="$CONDA_SH" \
    nohup bash "$SS/scripts/parity_pair.sh" "$backbone" "$gpu" "$env_name" \
    >"$LOG/${backbone}.log" 2>&1 &
  echo "started $backbone on GPU$gpu pid=$! log=$LOG/${backbone}.log"
}

usage() {
  cat <<EOF
Usage: $0 wave1|wave2|all|compare
  wave1   MultiTalk/InfiniteTalk/Fantasy/S2V/Hunyuan on GPUs 0/4/5/6/7
  wave2   Animate + Fun Camera on the first two free GPUs
  compare Print the 7-row decoded-frame table
EOF
}

cmd="${1:-wave1}"
case "$cmd" in
  wave1)
    prepare_animate_assets || true
    ensure_animate_worktree || true
    run_pair multitalk 0 multitalk
    run_pair infinitetalk 4 infinitetalk
    run_pair fantasytalking 5 multitalk
    run_pair wan_s2v 6 wans2v
    run_pair hunyuan_avatar 7 HunyuanVideo-Avatar
    echo "wave1 launched. logs in $LOG"
    ;;
  wave2)
    prepare_animate_assets
    ensure_animate_worktree
    # Prefer GPU 0 then 4 once wave1 pair processes have exited.
    run_pair wan_animate 0 wans2v
    run_pair wan_fun_camera 4 multitalk
    echo "wave2 launched. logs in $LOG"
    ;;
  compare)
    source "$CONDA_SH"
    conda activate multitalk
    export PYTHONPATH="$SS:${PYTHONPATH:-}"
    python -u "$CMP" --json_out "$OUT/parity_compare.json"
    ;;
  *)
    usage
    exit 1
    ;;
esac
