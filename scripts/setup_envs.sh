#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAMILY="${1:---family}"
NAME="all"
if [[ "$FAMILY" == "--family" ]]; then
  NAME="${2:-all}"
elif [[ "$FAMILY" != --* ]]; then
  NAME="$FAMILY"
fi

create() {
  local env_name="$1" file="$2"
  echo "==> conda env $env_name from $file"
  if command -v conda >/dev/null 2>&1; then
    conda env create -f "$file" --force
  else
    echo "conda not found; create the env described in $file manually" >&2
  fi
}

case "$NAME" in
  talking) create ss-talking "$ROOT/envs/ss-talking.yml" ;;
  wan22) create ss-wan22 "$ROOT/envs/ss-wan22.yml" ;;
  hva) create ss-hva "$ROOT/envs/ss-hva.yml" ;;
  fantasytalking) create ss-fantasytalking "$ROOT/envs/ss-fantasytalking.yml" ;;
  videox-fun) create ss-videox-fun "$ROOT/envs/ss-videox-fun.yml" ;;
  all)
    create ss-talking "$ROOT/envs/ss-talking.yml"
    create ss-wan22 "$ROOT/envs/ss-wan22.yml"
    create ss-hva "$ROOT/envs/ss-hva.yml"
    create ss-fantasytalking "$ROOT/envs/ss-fantasytalking.yml"
    create ss-videox-fun "$ROOT/envs/ss-videox-fun.yml"
    ;;
  *) echo "unknown family $NAME" >&2; exit 1 ;;
esac

echo "Activate an env and run: pip install -e $ROOT"
