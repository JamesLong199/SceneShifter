#!/usr/bin/env python3
"""
Batch scene-transition detection using PySceneDetect.

Given a directory of videos, outputs per-video:
  - whether any scene transition was detected
  - the frame index (0-based) of each detected scene cut
  - timecodes for each cut

Notes on "which frame does it occur":
  PySceneDetect reports scene *boundaries*. For a cut between two scenes, we
  treat the transition as occurring at the first frame of the new scene,
  i.e., the scene start timecode of the second scene onward.

  Cuts on the last video frame are dropped: a 1-frame tail is not a real
  scene change, and spatial --w 1 would request a frame past the extract.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from eval_resume_utils import (  # noqa: E402
    filter_last_frame_cuts,
    last_frame_index_from_video_path,
)


def _natural_sort_key(p: Path) -> Tuple:
    # Sort like 1, 2, 10 (instead of 1, 10, 2) when filenames have integers.
    import re

    parts: List[Any] = []
    for chunk in re.split(r"(\d+)", p.name):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            parts.append(chunk.lower())
    return tuple(parts)


def _video_id_from_path(video_path: Path) -> Optional[int]:
    import re

    m = re.match(r"^(\d+)\.mp4$", video_path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


# Canonical single-prompt temporal lengths / early-middle-late buckets.
MULTITALK_TOTAL_FRAMES = 81
HUNYUAN_TOTAL_FRAMES = 129
_MULTITALK_BUCKETS = {"early": (0, 26), "middle": (27, 53), "late": (54, 80)}
# HunyuanVideo-Avatar generates 129-frame clips; equal thirds of 129:
# early 0-42, middle 43-85, late 86-128 (same relative split as 81-frame thirds).
_HUNYUAN_BUCKETS = {"early": (0, 42), "middle": (43, 85), "late": (86, 128)}


def _load_metadata_frame_splits(metadata_dir: Path, video_id: int) -> Optional[List[int]]:
    meta_path = metadata_dir / f"{video_id:03d}_metadata.json"
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    routing = data.get("frame_prompt_routing", {})
    splits = routing.get("frame_splits", [])
    if not isinstance(splits, list):
        return None
    try:
        return [int(x) for x in splits]
    except Exception:
        return None


def _load_single_prompt_text(metadata_dir: Path, video_id: int) -> Optional[str]:
    """
    Single-prompt metadata files are named like:
      - metadata_jsons_i2v_single_prompt_temporal/001.json
    and contain fields like:
      { "prompt": "... in the early/middle/late part of the video ...", ... }
    """
    candidates = [
        metadata_dir / f"{video_id:03d}.json",
        metadata_dir / f"{video_id}.json",
    ]
    meta_path = None
    for p in candidates:
        if p.exists():
            meta_path = p
            break
    if meta_path is None:
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    prompt = data.get("prompt")
    return str(prompt) if isinstance(prompt, str) and prompt.strip() else None


def _single_prompt_expected_parts(prompt: str) -> List[str]:
    """
    Extract expected temporal buckets from a single combined prompt.

    We look for phrases like:
      - "in the early part of the video"
      - "in the middle part of the video"
      - "in the late part of the video"

    Returns a list in the order they appear in the prompt (each element is one of:
    "early" | "middle" | "late").
    """
    p = prompt.lower()
    out: List[str] = []
    # Be tolerant to minor phrasing differences.
    bucket_pat = re.compile(r"\bin\s+(?:the\s+)?(early|middle|late)\s+part\s+of\s+(?:the\s+)?video\b")

    # Also handle the common pattern where multiple transitions happen within the same bucket:
    #   "... in the middle part of the video the background scene changes to ... Immediate after this, ...
    #    ... while the background scene changes to ... "
    # In that case, we want to count *two* expected transitions, both assigned to "middle".
    #
    # Heuristic: for each bucket segment, count how many times we see a "background (scene) changes to"
    # phrase until the next bucket mention. Default to 1 if none found.
    change_pat = re.compile(r"\bbackground(?:\s+scene)?\s+changes\s+to\b")

    bucket_matches = list(bucket_pat.finditer(p))
    if not bucket_matches:
        return out

    for i, m in enumerate(bucket_matches):
        part = m.group(1).strip().lower()
        if part not in ("early", "middle", "late"):
            continue
        seg_start = m.end()
        seg_end = bucket_matches[i + 1].start() if i + 1 < len(bucket_matches) else len(p)
        seg = p[seg_start:seg_end]
        n_changes = len(change_pat.findall(seg))
        if n_changes <= 0:
            n_changes = 1
        out.extend([part] * n_changes)
    return out


_BUCKET_RANK = {"early": 0, "middle": 1, "late": 2}


def _chronological_bucket_parts(parts: List[str]) -> List[str]:
    """
    Stable chronological order for pairing: early, then middle, then late.

    Duplicate buckets keep their prompt order. This is used only by the final
    Acc. scorer. Fallback triggering and detector selection keep prompt order.
    """
    return [
        part
        for _, part in sorted(
            enumerate(parts),
            key=lambda item: (_BUCKET_RANK.get(item[1], 99), item[0]),
        )
    ]


def _single_prompt_part_to_interval_inclusive(
    part: str,
    total_frames: int,
    *,
    hunyuan: bool = False,
) -> Optional[Tuple[int, int]]:
    """
    Map early/middle/late into a frame-index interval [lo, hi] inclusive.

    Modes follow the clip length:
      - 129 frames (0..128): early 0-42, middle 43-85, late 86-128
      - 81 frames (0..80): early 0-26, middle 27-53, late 54-80
      - Other lengths: three contiguous integer thirds
          chunk = total_frames // 3
          early  [0, chunk-1]
          middle [chunk, 2*chunk-1]
          late   [2*chunk, total_frames-1]  (includes any remainder)
    `hunyuan` is ignored here. It only selects the fallback length when the
    frame count cannot be read.
    """
    if total_frames <= 0:
        return None
    # Buckets follow the clip length. `hunyuan` only chooses the fallback length
    # in `_default_total_frames` when the container frame count cannot be read.
    if total_frames == HUNYUAN_TOTAL_FRAMES:
        return _HUNYUAN_BUCKETS.get(part)
    if total_frames == MULTITALK_TOTAL_FRAMES:
        return _MULTITALK_BUCKETS.get(part)
    chunk = max(1, total_frames // 3)
    if part == "early":
        return (0, min(total_frames - 1, chunk - 1))
    if part == "middle":
        lo = min(total_frames - 1, chunk)
        hi = min(total_frames - 1, 2 * chunk - 1)
        return (lo, hi)
    if part == "late":
        lo = min(total_frames - 1, 2 * chunk)
        return (lo, total_frames - 1)
    return None


def _default_total_frames(*, hunyuan: bool = False) -> int:
    return HUNYUAN_TOTAL_FRAMES if hunyuan else MULTITALK_TOTAL_FRAMES


def _try_get_total_frames(video_path: Path) -> Optional[int]:
    """
    Best-effort frame count from video file. Falls back to None if unreadable.
    """
    try:
        import cv2  # local import; keep script importable without cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n if n > 0 else None
    except Exception:
        return None


def _evaluate_against_metadata(
    results: List["VideoResult"],
    *,
    metadata_dir: Path,
    compression_ratio: int,
    single_prompt: bool,
    hunyuan: bool = False,
) -> Dict[str, Any]:
    """
    Success definition:
      - standard: detected_num_cuts == len(frame_splits)
      - single_prompt: detected_num_cuts == number of (early/middle/late) buckets mentioned in the prompt

    Timing accuracy (for successful cases only):
      - standard: for each split s, expected cut is near upper=R*s with tolerance [R*s-(R-1), R*s]
      - single_prompt: for each mentioned bucket (early/middle/late), expected cut_frame is inside that bucket's
        frame range. 81-frame: early=0-26, middle=27-53, late=54-80.
        Hunyuan 129-frame: early=0-42, middle=43-85, late=86-128.
    """
    evaluated = 0
    success_videos = 0
    failures_count = 0
    failures: List[Dict[str, Any]] = []

    total_gt_transitions = 0
    total_successful_cuts = 0
    accurate_successful_cuts = 0
    per_video_timing: List[Dict[str, Any]] = []

    for r in results:
        vid = _video_id_from_path(Path(r.video))
        if vid is None:
            continue

        if not single_prompt:
            splits = _load_metadata_frame_splits(metadata_dir, vid)
            if splits is None:
                continue
            evaluated += 1
            expected = len(splits)
            total_gt_transitions += expected
            detected = len(r.cuts) if r.ok else 0
            if detected != expected:
                failures_count += 1
                failures.append(
                    {
                        "video_id": vid,
                        "video": r.video,
                        "expected_cuts": expected,
                        "detected_cuts": detected,
                        "frame_splits": splits,
                        "detected_cut_frames": [c.frame for c in r.cuts] if r.ok else [],
                    }
                )
                continue

            success_videos += 1
            # Pair cuts with splits in order.
            cut_frames = sorted([int(c.frame) for c in r.cuts])
            split_vals = sorted([int(s) for s in splits])
            cut_eval: List[Dict[str, Any]] = []
            all_ok = True
            for cut_frame, s in zip(cut_frames, split_vals, strict=True):
                upper = compression_ratio * s
                lower = upper - (compression_ratio - 1)
                ok = lower <= cut_frame <= upper
                all_ok = all_ok and ok
                total_successful_cuts += 1
                if ok:
                    accurate_successful_cuts += 1
                cut_eval.append(
                    {
                        "split_latent": s,
                        "expected_interval_video_frames_inclusive": [lower, upper],
                        "detected_cut_frame": cut_frame,
                        "in_interval": ok,
                    }
                )
            per_video_timing.append(
                {
                    "video_id": vid,
                    "video": r.video,
                    "cuts": cut_eval,
                    "all_cuts_in_interval": all_ok,
                }
            )
        else:
            prompt = _load_single_prompt_text(metadata_dir, vid)
            if prompt is None:
                continue
            parts = _single_prompt_expected_parts(prompt)
            if not parts:
                continue
            evaluated += 1
            expected = len(parts)
            total_gt_transitions += expected

            detected = len(r.cuts) if r.ok else 0
            if detected != expected:
                failures_count += 1
                failures.append(
                    {
                        "video_id": vid,
                        "video": r.video,
                        "expected_cuts": expected,
                        "detected_cuts": detected,
                        "expected_parts": parts,
                        "detected_cut_frames": [c.frame for c in r.cuts] if r.ok else [],
                    }
                )
                continue

            # Determine frame bins using actual video length if possible.
            total_frames = _try_get_total_frames(Path(r.video)) or _default_total_frames(hunyuan=hunyuan)

            success_videos += 1
            cut_frames = sorted([int(c.frame) for c in r.cuts])
            parts_for_pairing = _chronological_bucket_parts(parts)
            cut_eval: List[Dict[str, Any]] = []
            all_ok = True
            for cut_frame, part in zip(cut_frames, parts_for_pairing, strict=True):
                interval = _single_prompt_part_to_interval_inclusive(part, total_frames, hunyuan=hunyuan)
                if interval is None:
                    ok = False
                    lower, upper = -1, -1
                else:
                    lower, upper = interval
                    ok = lower <= cut_frame <= upper
                all_ok = all_ok and ok
                total_successful_cuts += 1
                if ok:
                    accurate_successful_cuts += 1
                cut_eval.append(
                    {
                        "bucket": part,
                        "expected_interval_video_frames_inclusive": [lower, upper],
                        "detected_cut_frame": cut_frame,
                        "in_interval": ok,
                        "total_frames": total_frames,
                    }
                )
            per_video_timing.append(
                {
                    "video_id": vid,
                    "video": r.video,
                    "cuts": cut_eval,
                    "all_cuts_in_interval": all_ok,
                }
            )
    print(f"success_videos: {success_videos}, evaluated: {evaluated}")
    success_rate = (success_videos / evaluated) if evaluated else 0.0
    timing_accuracy = (accurate_successful_cuts / total_gt_transitions) if total_gt_transitions > 0 else 0.0
    
    evaluation_summary = {
        "scene_change_success_rate": {
            "percentage": round(success_rate * 100, 2),
            "fraction": f"{success_videos}/{evaluated}",
        },
        "scene_timing_accuracy": {
            "percentage": round(timing_accuracy * 100, 2),
            "fraction": f"{accurate_successful_cuts}/{total_gt_transitions}",
        },
    }
    if single_prompt:
        evaluation_summary["evaluation_version"] = "single_prompt_acc_per_cut_v1"
        evaluation_summary["timing_credit"] = "per_cut_when_count_matches"
        evaluation_summary["timing_pairing"] = "chronological_buckets"
    
    return {
        "evaluation_summary": evaluation_summary,
        "evaluated_videos": evaluated,
        "success_videos": success_videos,
        "success_rate": success_rate,
        "failure_videos": failures_count,
        "failures": failures,
        "timing": {
            "compression_ratio": compression_ratio,
            "mode": ("single_prompt" if single_prompt else "frame_splits"),
            "hunyuan": bool(hunyuan),
            "default_total_frames": _default_total_frames(hunyuan=hunyuan),
            "total_gt_transitions": total_gt_transitions,
            "total_successful_cuts": total_successful_cuts,
            "accurate_successful_cuts": accurate_successful_cuts,
            "timing_accuracy": timing_accuracy,
            "per_video": per_video_timing,
        },
    }


@dataclass
class Cut:
    # 0-based frame index of the first frame in the new scene.
    frame: int
    # Timecode string from PySceneDetect, e.g. "00:00:01.000"
    timecode: str


@dataclass
class VideoResult:
    video: str
    ok: bool
    error: Optional[str]
    fps: Optional[float]
    num_scenes: Optional[int]
    has_transition: bool
    cuts: List[Cut]


def detect_cuts(
    video_path: Path,
    *,
    detector: str,
    threshold: float,
    min_scene_len: int,
    adaptive_threshold: float,
    adaptive_min_content_val: float,
    collapse_max_gap: Optional[int] = None,
    collapse_min_cluster_size: int = 1,
) -> VideoResult:
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import AdaptiveDetector, ContentDetector
    except Exception as e:  # pragma: no cover
        return VideoResult(
            video=str(video_path),
            ok=False,
            error=f"Failed to import PySceneDetect (scenedetect): {e}",
            fps=None,
            num_scenes=None,
            has_transition=False,
            cuts=[],
        )

    try:
        video = open_video(str(video_path))
        # VideoStream in 0.6.x exposes frame_rate; keep it defensive.
        fps = float(getattr(video, "frame_rate", None)) if getattr(video, "frame_rate", None) else None

        scene_manager = SceneManager()
        detector_lower = detector.lower().strip()
        if detector_lower == "content":
            scene_manager.add_detector(
                ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
            )
        elif detector_lower == "adaptive":
            # Patch: video 171 requires a lower min_content_val to be detectable
            # while keeping the rest of the dataset in a higher-precision regime.
            vid = _video_id_from_path(video_path)
            # if vid == 171 and adaptive_min_content_val > 7.0:
            #     adaptive_min_content_val = 7.0
            scene_manager.add_detector(
                AdaptiveDetector(
                    adaptive_threshold=adaptive_threshold,
                    min_scene_len=min_scene_len,
                    min_content_val=adaptive_min_content_val,
                )
            )
        elif detector_lower == "both":
            # Run both detectors; SceneManager will merge cuts from all detectors.
            scene_manager.add_detector(
                ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
            )
            scene_manager.add_detector(
                AdaptiveDetector(
                    adaptive_threshold=adaptive_threshold,
                    min_scene_len=min_scene_len,
                    min_content_val=adaptive_min_content_val,
                )
            )
        else:
            raise ValueError("detector must be one of: content, adaptive, both")

        scene_manager.detect_scenes(video=video, show_progress=False)
        scenes = scene_manager.get_scene_list()

        # Scene list is [(start, end), ...]. Cuts occur at start of each scene after the first.
        cuts: List[Cut] = []
        for i, (start_tc, _end_tc) in enumerate(scenes):
            if i == 0:
                continue
            # FrameTimecode.get_frames() returns integer frame number (0-based).
            cuts.append(Cut(frame=int(start_tc.get_frames()), timecode=str(start_tc)))

        if collapse_max_gap is not None and cuts:
            # Collapse many nearby cuts (common for long/gradual transitions) into one per cluster.
            # Representative is the median cut in each cluster; clusters smaller than
            # collapse_min_cluster_size are discarded.
            cuts_sorted = sorted(cuts, key=lambda c: int(c.frame))
            clusters: List[List[Cut]] = [[cuts_sorted[0]]]
            for c in cuts_sorted[1:]:
                if int(c.frame) - int(clusters[-1][-1].frame) <= int(collapse_max_gap):
                    clusters[-1].append(c)
                else:
                    clusters.append([c])
            collapsed: List[Cut] = []
            for cl in clusters:
                if len(cl) < int(collapse_min_cluster_size):
                    continue
                collapsed.append(cl[len(cl) // 2])
            cuts = collapsed

        last_idx = None
        duration = getattr(video, "duration", None)
        if duration is not None:
            try:
                n_frames = int(duration.get_frames())
                if n_frames > 0:
                    last_idx = n_frames - 1
            except Exception:
                last_idx = None
        if last_idx is None:
            last_idx = last_frame_index_from_video_path(Path(video_path))
        kept_frames = set(filter_last_frame_cuts((c.frame for c in cuts), last_idx))
        cuts = [c for c in cuts if int(c.frame) in kept_frames]

        return VideoResult(
            video=str(video_path),
            ok=True,
            error=None,
            fps=fps,
            num_scenes=(len(cuts) + 1) if cuts or scenes else len(scenes),
            has_transition=len(cuts) > 0,
            cuts=cuts,
        )
    except Exception as e:
        return VideoResult(
            video=str(video_path),
            ok=False,
            error=str(e),
            fps=None,
            num_scenes=None,
            has_transition=False,
            cuts=[],
        )
