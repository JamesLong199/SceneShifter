#!/usr/bin/env python3
"""Decode both parity mp4s and require identical frames (max abs == 0)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sceneshifter.parity.common import find_videos, load_parity_config, output_root, side_dir


def decode_mp4(path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("opencv-python is required to decode parity videos") from exc
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {path}")
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames, axis=0)


def pick_video(directory: Path, glob_pat: str) -> Path:
    hits = find_videos(directory, glob_pat)
    if not hits:
        raise FileNotFoundError(f"no mp4 under {directory} matching {glob_pat}")
    preferred = [p for p in hits if "tmp_" not in p.name and "video-only" not in p.name]
    return (preferred or hits)[0]


def first_diff(a: np.ndarray, b: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if a.shape != b.shape:
        return None
    ne = np.not_equal(a, b)
    if not ne.any():
        return None
    idx = np.argwhere(ne)[0]
    return tuple(int(v) for v in idx)


def compare_pair(backbone: str, root: Optional[str] = None) -> dict:
    cfg = load_parity_config(backbone)
    glob_pat = cfg.get("expected_video_glob", "*.mp4")
    ref_dir = side_dir(backbone, "reference", root=root)
    uni_dir = side_dir(backbone, "unified", root=root)
    ref = pick_video(ref_dir, glob_pat)
    uni = pick_video(uni_dir, glob_pat)
    ref_frames = decode_mp4(ref)
    uni_frames = decode_mp4(uni)
    ref_sha = "unknown"
    uni_sha = "unknown"
    ref_args = ref_dir / "run_args.json"
    uni_args = uni_dir / "run_args.json"
    if ref_args.is_file():
        ref_sha = json.loads(ref_args.read_text()).get("checkout_sha", "unknown")
    if uni_args.is_file():
        uni_sha = json.loads(uni_args.read_text()).get("official_sha", "unknown")
    result = {
        "backbone": backbone,
        "id": cfg["sample_id"],
        "reference": str(ref),
        "unified": str(uni),
        "reference_sha": ref_sha,
        "unified_sha": uni_sha,
        "reference_shape": list(ref_frames.shape),
        "unified_shape": list(uni_frames.shape),
        "max_abs": None,
        "mean_abs": None,
        "first_diff": None,
        "pass": False,
    }
    if ref_frames.shape != uni_frames.shape:
        result["error"] = (
            f"shape mismatch {ref_frames.shape} vs {uni_frames.shape}"
        )
        return result
    diff = np.abs(ref_frames.astype(np.int16) - uni_frames.astype(np.int16))
    result["max_abs"] = int(diff.max())
    result["mean_abs"] = float(diff.mean())
    result["first_diff"] = first_diff(ref_frames, uni_frames)
    result["pass"] = result["max_abs"] == 0
    if not result["pass"] and result["first_diff"] is not None:
        t, y, x, c = result["first_diff"]
        dump = Path(output_root(root)) / backbone / f"first_diff_t{t}.png"
        try:
            import cv2

            cv2.imwrite(str(dump), np.concatenate([ref_frames[t], uni_frames[t]], axis=1))
            result["first_diff_dump"] = str(dump)
        except Exception:
            pass
        print(f"[compare] first differing frame t={t} y={y} x={x} ch={c}", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--json_out", default=None)
    args = parser.parse_args()
    backbones = (
        [args.backbone]
        if args.backbone
        else [
            "multitalk",
            "infinitetalk",
            "fantasytalking",
            "wan_s2v",
            "hunyuan_avatar",
            "wan_animate",
            "wan_fun_camera",
        ]
    )
    rows = []
    failed = False
    print(
        f"{'backbone':18} {'id':6} {'ref_sha':12} {'uni_sha':12} {'max_abs':8} result"
    )
    for backbone in backbones:
        try:
            row = compare_pair(backbone, args.output_root)
        except Exception as exc:  # noqa: BLE001
            row = {
                "backbone": backbone,
                "id": "?",
                "reference_sha": "missing",
                "unified_sha": "missing",
                "max_abs": None,
                "pass": False,
                "error": str(exc),
            }
        rows.append(row)
        status = "PASS" if row.get("pass") else "FAIL"
        if not row.get("pass"):
            failed = True
        print(
            f"{row['backbone']:18} {str(row.get('id', '?')):6} "
            f"{str(row.get('reference_sha', '?'))[:12]:12} "
            f"{str(row.get('unified_sha', '?'))[:12]:12} "
            f"{str(row.get('max_abs')):8} {status}"
            + (f"  {row.get('error', '')}" if row.get("error") else "")
        )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
