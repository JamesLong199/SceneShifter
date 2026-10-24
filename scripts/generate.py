#!/usr/bin/env python3
"""Run one SceneShifter demo on an official backbone checkout."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sceneshifter.common import (  # noqa: E402
    apply_demo_config,
    find_videos,
    git_head,
    load_backbone_config,
    repo_root,
    write_run_args,
)
from sceneshifter.unified import resolve_official_root, run_unified  # noqa: E402

BACKBONES = [
    "multitalk",
    "infinitetalk",
    "fantasytalking",
    "wan_s2v",
    "wan_animate",
    "wan_fun_camera",
    "hunyuan_avatar",
]


def _prepend_python_bin_to_path() -> str:
    env_bin = str(Path(sys.executable).resolve().parent)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if env_bin not in path_parts:
        os.environ["PATH"] = os.pathsep.join([env_bin, *path_parts])
    return env_bin


def _output_dir(backbone: str, root: str | None) -> Path:
    base = Path(root) if root else repo_root() / "outputs" / "demos"
    if not base.is_absolute():
        base = repo_root() / base
    path = base / backbone
    path.mkdir(parents=True, exist_ok=True)
    (path / "videos").mkdir(parents=True, exist_ok=True)
    return path


def _require_official_sha(cfg: Dict[str, Any], repo: str) -> str:
    expected = str((cfg.get("locks") or {}).get("official_sha") or "")
    actual = git_head(repo)
    if not expected:
        raise RuntimeError(f"{cfg['backbone']} config is missing locks.official_sha")
    if actual != expected:
        raise RuntimeError(
            f"{cfg['backbone']} official revision drift: expected {expected}, got {actual} "
            f"at {repo}"
        )
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a SceneShifter demo")
    parser.add_argument("--backbone", required=True, choices=BACKBONES)
    parser.add_argument("--output_root", default=None)
    parser.add_argument(
        "--demo",
        default="examples",
        help="Shared demo directory.",
    )
    parser.add_argument("--inject", action="store_true", default=True)
    parser.add_argument("--no_inject", action="store_false", dest="inject")
    args = parser.parse_args()

    cfg = apply_demo_config(load_backbone_config(args.backbone), args.demo)
    cfg["inject"] = args.inject
    out_dir = _output_dir(args.backbone, args.output_root)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Media helpers are subprocesses. Running an env's Python directly does not
    # activate that env, so make its ffmpeg/ffprobe binaries discoverable.
    _prepend_python_bin_to_path()

    official = resolve_official_root(cfg)
    if not official.is_dir():
        raise FileNotFoundError(
            f"official checkout missing: {official}. Run scripts/setup_backbones.sh first."
        )
    payload: Dict[str, Any] = {
        "backbone": args.backbone,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "sample_id": cfg["sample_id"],
        "sceneshifter_sha": git_head(str(repo_root())),
        "official_root": str(official),
        "official_sha": _require_official_sha(cfg, str(official)),
        "config": cfg,
    }
    write_run_args(out_dir, payload)
    run_unified(cfg, out_dir)
    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    payload["official_sha_after"] = git_head(str(official))
    write_run_args(out_dir, payload)
    videos = find_videos(out_dir, cfg.get("expected_video_glob", "*.mp4"))
    if not videos:
        raise SystemExit(f"no mp4 written under {out_dir}")
    print(f"[generate] wrote {videos[0]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
