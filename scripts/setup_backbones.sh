#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$ROOT/third_party/backbones.lock"
WITH_REFERENCE=0

usage() {
  cat <<EOF
Clone official backbone mains into third_party/ (read-only after clone).

Usage: $0 [--with-reference] [--only NAME]

  --with-reference  also clone jameslong199 feature branches into tests/reference/
  --only NAME       clone a single official backbone (multitalk, wan2.2, ...)
EOF
}

ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-reference) WITH_REFERENCE=1 ;;
    --only) ONLY="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
  shift
done

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
