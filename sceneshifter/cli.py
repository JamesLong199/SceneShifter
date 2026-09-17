"""Unified SceneShifter CLI. Official trees stay unmodified."""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

from .config import load_frame_prompt_config
from .hooks import inject_router
from .paths import official_root, push_official_path
from .router import create_router

BACKENDS = [
    "multitalk",
    "infinitetalk",
    "fantasytalking",
    "hunyuan_avatar",
    "wan_s2v",
    "wan_animate",
    "wan_fun_camera",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SceneShifter unified generate")
    parser.add_argument("--backbone", required=True, choices=BACKENDS)
    parser.add_argument("--frame_prompt_config", required=True)
    parser.add_argument("--third_party_root", default=None)
    parser.add_argument("--ckpt_dir", default=None)
    parser.add_argument("--input_json", default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    routing = load_frame_prompt_config(args.frame_prompt_config, backend=args.backbone)
    router = create_router(routing, backend=args.backbone, validate=False)
    root = official_root(args.backbone, args.third_party_root)
    if args.dry_run:
        print(f"backbone={args.backbone} enabled={routing.get('enabled')} router={router is not None}")
        print(f"official_root={root} exists={os.path.isdir(root)}")
        return 0
    if not os.path.isdir(root):
        raise SystemExit(
            f"Official clone missing at {root}. Run scripts/setup_backbones.sh first."
        )
    push_official_path(root)
    # Adapters wrap after the caller constructs the official pipeline.
    # Keep this entrypoint import-safe so community scripts can do:
    #   router = create_router(...); inject_router(pipe, router, adapter=...)
    print(
        "Construct the official pipeline, then call "
        f"inject_router(model, router, adapter={args.backbone!r}).",
        file=sys.stderr,
    )
    _ = inject_router
    return 0
