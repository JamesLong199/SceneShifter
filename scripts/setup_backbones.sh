#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$ROOT/third_party/backbones.lock"
WITH_REFERENCE=0

usage() {
  cat <<EOF
Clone official backbone mains into third_party/ (read-only after clone).

Usage: $0 [--backbone NAME] [--with-reference] [--only NAME]

  --backbone NAME   official code for one demo backbone:
                      multitalk, infinitetalk, fantasytalking,
                      wan_s2v, wan_animate, wan_fun_camera,
                      hunyuan_avatar, or all
                    wan_s2v and wan_animate share the Wan2.2 checkout
  --only NAME       clone one lockfile section (multitalk, wan2.2, ...)
  --with-reference  also clone jameslong199 feature branches into tests/reference/
EOF
}

ONLY=""
BACKBONE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-reference) WITH_REFERENCE=1 ;;
    --only) ONLY="$2"; shift ;;
    --backbone) BACKBONE="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

if [[ -n "$BACKBONE" && -n "$ONLY" ]]; then
  echo "pass only one of --backbone and --only" >&2
  exit 1
fi

if [[ -n "$BACKBONE" ]]; then
  case "$BACKBONE" in
    multitalk) ONLY="multitalk" ;;
    infinitetalk) ONLY="infinitetalk" ;;
    fantasytalking) ONLY="fantasy-talking" ;;
    wan_s2v|wan_animate) ONLY="wan2.2" ;;
    wan_fun_camera) ONLY="videox-fun" ;;
    hunyuan_avatar) ONLY="hunyuanvideo-avatar" ;;
    all) ONLY="" ;;
    *) echo "unknown backbone $BACKBONE" >&2; usage; exit 1 ;;
  esac
fi

clone_locked() {
  local section="$1"
  python3 - "$LOCK" "$section" "$ROOT" <<'PY'
import ast, pathlib, sys
lock, section, root = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
text = pathlib.Path(lock).read_text()
# Minimal TOML-ish parser for this lockfile.
current = None
data = {}
for line in text.splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    if line.startswith("[") and line.endswith("]"):
        current = line[1:-1]
        data[current] = {}
        continue
    if current and "=" in line:
        key, value = [p.strip() for p in line.split("=", 1)]
        data[current][key] = ast.literal_eval(value)
if section not in data:
    raise SystemExit(f"missing section {section}")
cfg = data[section]
path = root / cfg["path"]
print(cfg["url"])
print(cfg["ref"])
print(cfg.get("sha", ""))
print(path)
PY
}

clone_one() {
  local section="$1"
  readarray -t META < <(clone_locked "$section")
  local url="${META[0]}" ref="${META[1]}" sha="${META[2]}" dest="${META[3]}"
  echo "==> $section"
  echo "    $url ($ref) -> $dest"
  if [[ ! -d "$dest/.git" ]]; then
    git clone --branch "$ref" --single-branch "$url" "$dest"
  else
    git -C "$dest" fetch origin "$ref"
    git -C "$dest" checkout "$ref"
  fi
  if [[ -n "$sha" ]]; then
    git -C "$dest" checkout --detach "$sha"
  fi
}

if [[ -n "$ONLY" ]]; then
  clone_one "backbones.$ONLY"
else
  for name in multitalk infinitetalk fantasy-talking hunyuanvideo-avatar wan2.2 videox-fun; do
    clone_one "backbones.$name"
  done
fi

if [[ "$WITH_REFERENCE" -eq 1 ]]; then
  for name in multitalk infinitetalk fantasy-talking hunyuanvideo-avatar wan2.2-s2v wan2.2-animate videox-fun; do
    clone_one "reference.$name"
  done
fi

echo "Done. Official trees must stay unmodified."
