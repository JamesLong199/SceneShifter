#!/usr/bin/env python3
"""Launch one SceneShifter parity side: reference feature-branch or unified inject."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sceneshifter.parity.common import (  # noqa: E402
    apply_demo_config,
    conda_python,
    find_videos,
    git_head,
    load_parity_config,
    repo_root,
    side_dir,
    write_run_args,
)
from sceneshifter.parity.unified import resolve_official_root, run_unified  # noqa: E402

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


def _checkout(repo: str, branch: str) -> None:
    current = subprocess.check_output(
        ["git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()
    if current == branch:
        return
    dirty = subprocess.check_output(["git", "-C", repo, "status", "--porcelain"], text=True)
    if dirty.strip():
        raise RuntimeError(
            f"{repo} is dirty on {current}; refuse to checkout {branch}:\n{dirty}"
        )
    subprocess.check_call(["git", "-C", repo, "checkout", branch])


def _require_locked_sha(cfg: Dict[str, Any], side: str, repo: str) -> str:
    key = "reference_feature_sha" if side == "reference" else "official_sha"
    expected = str((cfg.get("locks") or {}).get(key) or "")
    actual = git_head(repo)
    if not expected:
        raise RuntimeError(f"{cfg['backbone']} parity config is missing locks.{key}")
    if actual != expected:
        raise RuntimeError(
            f"{cfg['backbone']} {side} revision drift: expected {expected}, got {actual} "
            f"at {repo}"
        )
    return actual


def _physical_device() -> str:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return "0"
    first = raw.split(",")[0].strip()
    return first or "0"


def _reference_command(cfg: Dict[str, Any], out_dir: Path) -> List[str]:
    ref = cfg["reference"]
    python = conda_python(cfg["conda_env"])
    script = str(Path(ref["repo"]) / ref["script"])
    extra = list(ref.get("extra_args") or [])
    backbone = cfg["backbone"]
    cmd = [python, "-u", script]
    # Most batch workers call torch.cuda.set_device() inside the inherited
    # CUDA_VISIBLE_DEVICES mapping, so they need ordinal 0. Fantasy's worker
    # overwrites CUDA_VISIBLE_DEVICES itself and therefore needs the physical id.
    visible_gpu = _physical_device() if backbone == "fantasytalking" else "0"
    if backbone == "hunyuan_avatar":
        cmd += [
            "--dataset-root", cfg["dataset_root"],
            "--output-root", str(out_dir),
            "--gpus", visible_gpu,
        ]
    elif backbone == "wan_fun_camera":
        cmd += ["--output", str(out_dir)]
    elif backbone == "wan_animate":
        cmd += [
            "--save_file",
            str(out_dir / "videos" / f"{cfg['sample_id']}_sceneshifter.mp4"),
        ]
        (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    else:
        cmd += [
            "--dataset_root", cfg["dataset_root"],
            "--output_root", str(out_dir),
            "--gpus", visible_gpu,
        ]
    cmd += extra
    return cmd


def run_reference(cfg: Dict[str, Any], out_dir: Path) -> None:
    ref = cfg["reference"]
    repo = ref["repo"]
    _checkout(repo, ref["branch"])
    cmd = _reference_command(cfg, out_dir)
    env = os.environ.copy()
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    print("[parity-reference]", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=repo, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description="SceneShifter gpu40 parity generate")
    parser.add_argument(
        "--side",
        default="unified",
        choices=["reference", "unified"],
        help="unified runs SceneShifter. reference is the internal comparison checkout.",
    )
    parser.add_argument("--backbone", required=True, choices=BACKBONES)
    parser.add_argument("--output_root", default=None)
    parser.add_argument(
        "--demo",
        default=None,
        help="Shared demo directory (examples/demos).",
    )
    parser.add_argument("--trace_dir", default=None)
    parser.add_argument("--trace_steps", default="0")
    parser.add_argument("--trace_layers", default="0,37")
    args = parser.parse_args()

    cfg = load_parity_config(args.backbone)
    if args.demo:
        cfg = apply_demo_config(cfg, args.demo)
    out_dir = side_dir(args.backbone, args.side, root=args.output_root)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Media helpers are subprocesses. Running an env's Python directly does not
    # activate that env, so make its ffmpeg/ffprobe binaries discoverable.
    _prepend_python_bin_to_path()
    if args.trace_dir:
        os.environ["SCENESHIFTER_TRACE_DIR"] = str(Path(args.trace_dir).resolve())
        os.environ["SCENESHIFTER_TRACE_STEPS"] = str(args.trace_steps)
        os.environ["SCENESHIFTER_TRACE_LAYERS"] = str(args.trace_layers)

    payload: Dict[str, Any] = {
        "side": args.side,
        "backbone": args.backbone,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "sample_id": cfg["sample_id"],
        "sceneshifter_sha": git_head(str(repo_root())),
        "config": cfg,
        "trace": {
            "directory": os.environ.get("SCENESHIFTER_TRACE_DIR"),
            "steps": os.environ.get("SCENESHIFTER_TRACE_STEPS"),
            "layers": os.environ.get("SCENESHIFTER_TRACE_LAYERS"),
        },
    }
    if args.side == "reference":
        _checkout(cfg["reference"]["repo"], cfg["reference"]["branch"])
        payload["checkout_sha"] = _require_locked_sha(
            cfg, "reference", cfg["reference"]["repo"]
        )
        payload["checkout_branch"] = cfg["reference"]["branch"]
        payload["resolved_argv"] = _reference_command(cfg, out_dir)
        write_run_args(out_dir, payload)
        run_reference(cfg, out_dir)
    else:
        official = resolve_official_root(cfg)
        payload["official_root"] = str(official)
        if not official.is_dir():
            raise FileNotFoundError(f"official checkout missing: {official}")
        payload["official_sha"] = _require_locked_sha(cfg, "unified", str(official))
        write_run_args(out_dir, payload)
        run_unified(cfg, out_dir)
    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    payload["checkout_sha_after"] = (
        git_head(cfg["reference"]["repo"])
        if args.side == "reference"
        else git_head(str(resolve_official_root(cfg)))
    )
    write_run_args(out_dir, payload)
    videos = find_videos(out_dir, cfg.get("expected_video_glob", "*.mp4"))
    if not videos:
        raise SystemExit(f"no mp4 written under {out_dir}")
    print(f"[parity] wrote {videos[0]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
