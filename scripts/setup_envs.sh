#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<EOF
Create one SceneShifter conda environment.

Usage: $0 --family NAME

  NAME is one of:
    multitalk   sceneshifter_multitalk   (multitalk, infinitetalk)
    fantasy     sceneshifter_fantasy     (fantasytalking)
    wans2v      sceneshifter_wans2v      (wan_s2v, wan_animate)
    videoxfun   sceneshifter_videoxfun   (wan_fun_camera)
    hunyuan     sceneshifter_hunyuan     (hunyuan_avatar)
    all         every environment above
EOF
}

create() {
  local env_name="$1" file="$2"
  echo "==> conda env $env_name from $file"
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Install conda, then rerun this script." >&2
    exit 1
  fi
  conda env create -f "$file" --force
  echo "Created $env_name. Activate it with: conda activate $env_name"
}

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$1" == "--family" ]]; then
  NAME="${2:-}"
else
  NAME="$1"
fi
if [[ -z "$NAME" ]]; then
  usage
  exit 1
fi

case "$NAME" in
  multitalk) create sceneshifter_multitalk "$ROOT/envs/sceneshifter_multitalk.yml" ;;
  fantasy) create sceneshifter_fantasy "$ROOT/envs/sceneshifter_fantasy.yml" ;;
  wans2v) create sceneshifter_wans2v "$ROOT/envs/sceneshifter_wans2v.yml" ;;
  videoxfun) create sceneshifter_videoxfun "$ROOT/envs/sceneshifter_videoxfun.yml" ;;
  hunyuan) create sceneshifter_hunyuan "$ROOT/envs/sceneshifter_hunyuan.yml" ;;
  all)
    create sceneshifter_multitalk "$ROOT/envs/sceneshifter_multitalk.yml"
    create sceneshifter_fantasy "$ROOT/envs/sceneshifter_fantasy.yml"
    create sceneshifter_wans2v "$ROOT/envs/sceneshifter_wans2v.yml"
    create sceneshifter_videoxfun "$ROOT/envs/sceneshifter_videoxfun.yml"
    create sceneshifter_hunyuan "$ROOT/envs/sceneshifter_hunyuan.yml"
    ;;
  *) echo "unknown family $NAME" >&2; usage; exit 1 ;;
esac
