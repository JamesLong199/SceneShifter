#!/usr/bin/env python3
"""
Hybrid scene-transition detection for MultiTalk single-prompt experiments.

User intent:
  - MultiTalk runs can have either instant or gradual transitions.
  - First run an "instant" detector over all videos.
  - Then run a "gradual" detector ONLY on cases that failed (based on single-prompt GT),
    and combine results (per-video choose the better output).

GT source:
  - <input_dir>/tmp_input_jsons/<id>.json, or --config-dir <id>.json / <id>_metadata.json,
    contains a single prompt including:
      "in the early/middle/late part of the video ..."
    Each mentioned bucket counts as one expected transition.

Output:
  - Writes a scene_transitions.json-like payload with:
      - evaluation_summary (success rate + timing accuracy) computed using the same logic as
        detect_scene_transitions.py --single_prompt
      - Acc. gives one point per cut inside its bucket when the cut count matches.
        Count mismatches still contribute their expected cuts to the denominator and
        zero to the numerator. This final score is separate from _score_output_for_video,
        which still decides fallback triggering and which detector output is kept.
      - per-video field "source" in {"instant","gradual"} to indicate which detector was selected.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from eval_resume_utils import (  # noqa: E402
    add_ids_resume_args,
    filter_last_frame_cuts,
    filter_paths_by_ids,
    ids_from_list,
    last_frame_index_from_video_path,
    load_json,
    log_resume_plan,
    merge_list_by_video_id,
    parse_ids,
    resolve_todo_ids,
    write_json,
)


_BENCH_ROOT = Path(__file__).resolve().parents[1]


def _video_id_from_path(video_path: Path) -> Optional[int]:
    m = re.match(r"^(\d+)\.mp4$", video_path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _resolve_video_path(videos_dir: Path, vid: int) -> Optional[Path]:
    p = videos_dir / f"{vid:03d}.mp4"
    return p if p.exists() else None


def _natural_sort_key(p: Path) -> Tuple:
    parts: List[Any] = []
    for chunk in re.split(r"(\d+)", p.name):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            parts.append(chunk.lower())
    return tuple(parts)


def _single_prompt_expected_parts(prompt: str) -> List[str]:
    p = prompt.lower()
    out: List[str] = []
    bucket_pat = re.compile(r"\bin\s+(?:the\s+)?(early|middle|late)\s+part\s+of\s+(?:the\s+)?video\b")
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


def _bucket_interval_inclusive(part: str, total_frames: int, *, hunyuan: bool = False) -> Optional[Tuple[int, int]]:
    from detect_scene_transitions import _single_prompt_part_to_interval_inclusive  # type: ignore

    return _single_prompt_part_to_interval_inclusive(part, total_frames, hunyuan=hunyuan)


def _try_get_total_frames(video_path: Path) -> Optional[int]:
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return n if n > 0 else None
    except Exception:
        return None


def _default_total_frames(*, hunyuan: bool = False) -> int:
    from detect_scene_transitions import _default_total_frames as _default_tf  # type: ignore

    return int(_default_tf(hunyuan=hunyuan))


def _prompt_json_path(config_dir: Path, vid: int) -> Optional[Path]:
    candidates = [
        config_dir / f"{vid:03d}.json",
        config_dir / f"{vid:03d}_metadata.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_prompt_for_video(config_dir: Path, vid: int) -> str:
    meta_path = _prompt_json_path(config_dir, vid)
    if meta_path is None:
        return ""
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    prompt = data.get("prompt", "")
    return prompt if isinstance(prompt, str) else ""


def _evaluate_single_prompt(
    results: List[Dict[str, Any]],
    *,
    config_dir: Path,
    hunyuan: bool = False,
) -> Dict[str, Any]:
    from detect_scene_transitions import _chronological_bucket_parts  # type: ignore

    evaluated = 0
    success_videos = 0
    total_gt_transitions = 0
    accurate_successful_cuts = 0

    for r in results:
        vp = Path(r["video"])
        vid = _video_id_from_path(vp)
        if vid is None:
            continue

        prompt = _load_prompt_for_video(config_dir, vid)
        if not isinstance(prompt, str) or not prompt.strip():
            continue

        parts = _single_prompt_expected_parts(prompt)
        if not parts:
            continue

        evaluated += 1
        expected = len(parts)
        total_gt_transitions += expected

        cuts = r.get("cuts", []) or []
        detected = len(cuts)
        if detected != expected:
            continue
        success_videos += 1

        # Timing accuracy: one point per cut inside its chronologically paired bucket.
        # A miss does not cancel credit for the other cuts. Count mismatches never reach here.
        total_frames = _try_get_total_frames(vp) or _default_total_frames(hunyuan=hunyuan)
        cut_frames = sorted([int(c["frame"]) for c in cuts])
        parts_for_pairing = _chronological_bucket_parts(parts)
        for cut_frame, part in zip(cut_frames, parts_for_pairing, strict=True):
            interval = _bucket_interval_inclusive(part, total_frames, hunyuan=hunyuan)
            if interval is None:
                continue
            lo, hi = interval
            if lo <= cut_frame <= hi:
                accurate_successful_cuts += 1

    success_rate = (success_videos / evaluated) if evaluated else 0.0
    timing_accuracy = (accurate_successful_cuts / total_gt_transitions) if total_gt_transitions else 0.0
    return {
        "evaluation_summary": {
            "scene_change_success_rate": {"percentage": round(success_rate * 100, 2), "fraction": f"{success_videos}/{evaluated}"},
            "scene_timing_accuracy": {
                "percentage": round(timing_accuracy * 100, 2),
                "fraction": f"{accurate_successful_cuts}/{total_gt_transitions}",
            },
            "evaluation_version": "single_prompt_acc_per_cut_v1",
            "timing_credit": "per_cut_when_count_matches",
            "timing_pairing": "chronological_buckets",
        },
        "evaluated_videos": evaluated,
        "success_videos": success_videos,
        "success_rate": success_rate,
        "total_gt_transitions": total_gt_transitions,
        "accurate_successful_cuts": accurate_successful_cuts,
        "timing_accuracy": timing_accuracy,
        "hunyuan": bool(hunyuan),
    }


def _expected_parts_for_video(config_dir: Path, vid: int) -> List[str]:
    prompt = _load_prompt_for_video(config_dir, vid)
    if not prompt.strip():
        return []
    return _single_prompt_expected_parts(prompt)


def _score_output_for_video(
    cut_frames: List[int],
    *,
    expected_parts: List[str],
    total_frames: int,
    hunyuan: bool = False,
) -> Tuple[int, int]:
    """
    Lower is better.
      primary: absolute difference in cut count
      secondary: number of bucket mismatches (only meaningful if counts match)
    """
    expected = len(expected_parts)
    detected = len(cut_frames)
    count_err = abs(detected - expected)
    if count_err != 0 or expected == 0:
        return (count_err, 9999)
    mismatches = 0
    cut_frames_sorted = sorted([int(x) for x in cut_frames])
    for cut_frame, part in zip(cut_frames_sorted, expected_parts, strict=True):
        interval = _bucket_interval_inclusive(part, total_frames, hunyuan=hunyuan)
        if interval is None:
            mismatches += 1
            continue
        lo, hi = interval
        if not (lo <= cut_frame <= hi):
            mismatches += 1
    return (count_err, mismatches)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid MultiTalk scene-transition detection (instant then gradual on failures).")
    parser.add_argument("--input-dir", type=str, default="", help="Run directory containing videos/. The shell always passes this.")
    parser.add_argument("--glob", type=str, default="*.mp4")
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(_BENCH_ROOT / "configs" / "sceneshifter_multitalk"),
        help="Directory containing single-prompt JSON configs (<id>.json or <id>_metadata.json).",
    )
    parser.add_argument(
        "--hunyuan",
        action="store_true",
        help="Fallback total frame count is 129 when the container frame count cannot be read. Known lengths still use their own early/middle/late buckets.",
    )
    parser.add_argument("--output-json", type=str, default=None, help="Default: <input-dir>/scene_transitions.json")
    add_ids_resume_args(parser)
    args = parser.parse_args()
    if not str(args.input_dir).strip():
        raise SystemExit("--input-dir is required")

    hunyuan = bool(args.hunyuan)
    default_tf = _default_total_frames(hunyuan=hunyuan)

    input_dir = Path(args.input_dir).expanduser().resolve()
    videos_dir = input_dir / "videos"
    config_dir = Path(args.config_dir).expanduser().resolve()
    if not videos_dir.exists():
        raise SystemExit(f"Missing videos dir: {videos_dir}")
    if not config_dir.exists():
        raise SystemExit(f"Missing single-prompt config dir: {config_dir}")

    out_json = Path(args.output_json).expanduser().resolve() if args.output_json else (input_dir / "scene_transitions.json")

    from detect_scene_transitions import detect_cuts  # type: ignore

    # Preset params (must match detect_scene_transitions.py presets).
    instant_cfg = dict(
        detector="content",
        # Keep aligned with the tuned instant preset used for SceneShifter robustness:
        # content threshold=17, min_scene_len=5
        threshold=17.0,
        min_scene_len=5,
        adaptive_threshold=1.5,
        adaptive_min_content_val=12.5,
        collapse_max_gap=None,
        collapse_min_cluster_size=1,
    )
    gradual_cfg = dict(
        detector="adaptive",
        threshold=10.0,
        min_scene_len=4,
        adaptive_threshold=1.0,
        adaptive_min_content_val=5.0,
        collapse_max_gap=3,
        collapse_min_cluster_size=3,
    )

    videos = sorted(videos_dir.glob(args.glob), key=_natural_sort_key)
    requested_ids = parse_ids(args.ids)
    videos = filter_paths_by_ids(videos, requested_ids)
    existing_payload = load_json(out_json) if args.resume else None
    existing_results = (
        existing_payload.get("results", [])
        if isinstance(existing_payload, dict) and isinstance(existing_payload.get("results"), list)
        else []
    )
    done_ids = ids_from_list(existing_results, id_fields=("video_id", "video")) if args.resume else set()
    available_ids = [vid for vid in (_video_id_from_path(vp) for vp in videos) if vid is not None]
    todo_ids, skipped_ids = resolve_todo_ids(
        requested_ids=requested_ids,
        available_ids=available_ids,
        done_ids=done_ids,
        resume=bool(args.resume),
    )
    log_resume_plan(label="scene_transitions", todo=todo_ids, skipped=skipped_ids, requested=requested_ids)
    videos = [vp for vp in videos if _video_id_from_path(vp) in todo_ids]

    if not videos:
        if args.resume and existing_payload is not None:
            print("[scene_transitions] nothing to evaluate; keeping existing results")
            write_json(out_json, existing_payload)
            return
        print("[scene_transitions] no videos to evaluate")
        return

    combined_results: List[Dict[str, Any]] = []
    instant_failed: List[int] = []

    # First pass: instant for all.
    instant_cache: Dict[int, Any] = {}
    gradual_cache: Dict[int, Any] = {}

    for vp in videos:
        vid = _video_id_from_path(vp)
        if vid is None:
            continue

        expected_parts = _expected_parts_for_video(config_dir, vid)
        # If no GT parts found, still run instant and accept it (can't judge failure).
        total_frames = _try_get_total_frames(vp) or default_tf

        r_inst = detect_cuts(vp, **instant_cfg)
        inst_cuts = filter_last_frame_cuts(
            ([int(c.frame) for c in r_inst.cuts] if r_inst.ok else []),
            last_frame_index_from_video_path(vp),
        )
        instant_cache[vid] = (r_inst, inst_cuts, expected_parts, total_frames)

        if expected_parts:
            score_inst = _score_output_for_video(
                inst_cuts, expected_parts=expected_parts, total_frames=total_frames, hunyuan=hunyuan
            )
            if score_inst != (0, 0):
                instant_failed.append(vid)

    # Second pass: gradual only on failed (by GT).
    for vid in instant_failed:
        vp = _resolve_video_path(videos_dir, vid)
        if vp is None:
            continue
        r_grad = detect_cuts(vp, **gradual_cfg)
        grad_cuts = filter_last_frame_cuts(
            ([int(c.frame) for c in r_grad.cuts] if r_grad.ok else []),
            last_frame_index_from_video_path(vp),
        )
        gradual_cache[vid] = (r_grad, grad_cuts)

    # Combine: choose best per-video by GT score when available, otherwise instant.
    for vp in videos:
        vid = _video_id_from_path(vp)
        if vid is None:
            continue

        r_inst, inst_cuts, expected_parts, total_frames = instant_cache.get(vid, (None, [], [], default_tf))
        chosen = "instant"
        chosen_r = r_inst
        chosen_cuts = inst_cuts

        if expected_parts and vid in gradual_cache:
            r_grad, grad_cuts = gradual_cache[vid]
            score_inst = _score_output_for_video(
                inst_cuts, expected_parts=expected_parts, total_frames=total_frames, hunyuan=hunyuan
            )
            score_grad = _score_output_for_video(
                grad_cuts, expected_parts=expected_parts, total_frames=total_frames, hunyuan=hunyuan
            )
            if score_grad < score_inst:
                chosen = "gradual"
                chosen_r = r_grad
                chosen_cuts = grad_cuts

        # Build result entry in the same shape as detect_scene_transitions output.
        if chosen_r is None:
            entry = {
                "video": str(vp),
                "ok": False,
                "error": "missing_detection_result",
                "fps": None,
                "num_scenes": None,
                "has_transition": False,
                "cuts": [],
                "source": chosen,
            }
        else:
            entry = {
                **asdict(chosen_r),
                "cuts": [asdict(c) for c in chosen_r.cuts],
                "source": chosen,
            }
            # Ensure cuts match chosen representation (in case chosen_r had more cuts but collapsing differs).
            # detect_scene_transitions.detect_cuts already applies collapsing for gradual_cfg, so this is just defensive.
            if chosen_cuts is not None:
                entry["cuts"] = [{"frame": int(f), "timecode": ""} for f in chosen_cuts]
                entry["has_transition"] = len(chosen_cuts) > 0
        entry["video_id"] = int(vid)
        combined_results.append(entry)

    if args.resume and existing_results:
        combined_results = merge_list_by_video_id(
            existing_results,
            combined_results,
            id_fields=("video_id", "video"),
        )

    payload: Dict[str, Any] = {
        "input_dir": str(input_dir),
        "config_dir": str(config_dir),
        "glob": args.glob,
        "hunyuan": hunyuan,
        "detectors": {"instant": instant_cfg, "gradual": gradual_cfg},
        "hybrid_policy": "instant_for_all_then_gradual_on_gt_failures_choose_best_by(count_err, bucket_mismatches)",
        "results": combined_results,
        "summary": {
            "total_videos": len(combined_results),
            "videos_with_transition": sum(1 for r in combined_results if r.get("ok") and r.get("has_transition")),
            "videos_failed": sum(1 for r in combined_results if not r.get("ok")),
            "instant_failed_by_gt": len(instant_failed),
        },
    }

    # Add evaluation_summary using the combined results (single_prompt GT).
    eval_result = _evaluate_single_prompt(combined_results, config_dir=config_dir, hunyuan=hunyuan)
    payload = {"evaluation_summary": eval_result["evaluation_summary"], **payload, "evaluation": eval_result}

    write_json(out_json, payload)
    print(f"Wrote: {out_json}")
    print(json.dumps(payload["summary"], indent=2))
    print(json.dumps(payload["evaluation_summary"], indent=2))


if __name__ == "__main__":
    main()

