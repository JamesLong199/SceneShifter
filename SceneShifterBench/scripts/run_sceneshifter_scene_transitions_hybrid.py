#!/usr/bin/env python3
"""
Hybrid scene-transition detection for SceneShifter experiments (frame_splits GT).

User intent:
  - For SceneShifter experiments, run "instant" detection.
  - For videos that fail (missed/extra cuts OR wrong timing vs metadata frame_splits),
    run "gradual" detection and pick whichever output is closer to GT.

Ground truth:
  - metadata_jsons_v2_simplified/<id>_metadata.json -> frame_prompt_routing.frame_splits (latent indices)
  - Expected cut frame window for split s with compression ratio R:
      [R*s-(R-1), R*s] inclusive

Outputs:
  - Writes <input_dir>/scene_transitions_hybrid.json (does not overwrite scene_transitions.json)
  - Prints a short summary including how many videos switched to gradual.
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
    extract_video_id,
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


def _natural_sort_key(p: Path) -> Tuple:
    parts: List[Any] = []
    for chunk in re.split(r"(\d+)", p.name):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            parts.append(chunk.lower())
    return tuple(parts)


def _resolve_video_path(videos_dir: Path, video_id: int) -> Optional[Path]:
    p = videos_dir / f"{video_id:03d}.mp4"
    return p if p.exists() else None


def _load_frame_splits(metadata_dir: Path, video_id: int) -> Optional[List[int]]:
    meta_path = metadata_dir / f"{video_id:03d}_metadata.json"
    if not meta_path.exists():
        return None
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    splits = data.get("frame_prompt_routing", {}).get("frame_splits", [])
    if not isinstance(splits, list):
        return None
    out: List[int] = []
    for x in splits:
        try:
            out.append(int(x))
        except Exception:
            return None
    return out


def _expected_intervals_from_splits(splits: List[int], compression_ratio: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for s in sorted([int(x) for x in splits]):
        upper = compression_ratio * s
        lower = upper - (compression_ratio - 1)
        out.append((lower, upper))
    return out


def _cluster_frames(frames: List[int], *, max_gap: int) -> List[List[int]]:
    frames = sorted([int(x) for x in frames])
    if not frames:
        return []
    clusters: List[List[int]] = [[frames[0]]]
    for f in frames[1:]:
        if f - clusters[-1][-1] <= int(max_gap):
            clusters[-1].append(f)
        else:
            clusters.append([f])
    return clusters


def _cluster_dist_to_interval(cluster: List[int], interval: Tuple[int, int]) -> int:
    lo, hi = int(interval[0]), int(interval[1])
    if any(lo <= int(x) <= hi for x in cluster):
        return 0
    return min(min(abs(int(x) - lo), abs(int(x) - hi)) for x in cluster)


def _pick_representative_in_interval_or_nearest(cluster: List[int], interval: Tuple[int, int]) -> int:
    lo, hi = int(interval[0]), int(interval[1])
    inside = sorted([int(x) for x in cluster if lo <= int(x) <= hi])
    if inside:
        return int(inside[len(inside) // 2])
    # Otherwise, choose the nearest point in the cluster to the interval.
    best_d = None
    best_x = None
    for x in cluster:
        d = min(abs(int(x) - lo), abs(int(x) - hi))
        if best_d is None or d < best_d:
            best_d = d
            best_x = int(x)
    return int(best_x) if best_x is not None else int(cluster[len(cluster) // 2])


def _guided_select_cuts_from_raw(
    raw_cut_frames: List[int],
    *,
    expected_intervals: List[Tuple[int, int]],
    cluster_max_gap: int,
    min_cluster_size: int,
) -> List[int]:
    """
    Use expected intervals (from metadata frame_splits) to select one cut per interval.

    This is a GT-guided postprocess intended for evaluation/benchmarking:
      - cluster raw cuts by proximity (max_gap)
      - for each expected interval, pick the cluster closest to that interval
      - representative cut is picked as an in-interval frame if possible, else nearest-to-interval
    """
    clusters = [c for c in _cluster_frames(raw_cut_frames, max_gap=cluster_max_gap) if len(c) >= int(min_cluster_size)]
    if not expected_intervals:
        return []
    if not clusters:
        return []
    chosen: List[int] = []
    used = set()
    for interval in expected_intervals:
        best = None
        best_idx = None
        for i, c in enumerate(clusters):
            if i in used:
                continue
            d = _cluster_dist_to_interval(c, interval)
            if best is None or d < best:
                best = d
                best_idx = i
        if best_idx is None:
            continue
        used.add(best_idx)
        chosen.append(_pick_representative_in_interval_or_nearest(clusters[best_idx], interval))
    return sorted([int(x) for x in chosen])


def _score_vs_gt(
    cut_frames: List[int],
    *,
    expected_intervals: List[Tuple[int, int]],
) -> Tuple[int, int]:
    """
    Lower is better.
      primary: abs(#cuts - #gt)
      secondary: number of timing mismatches (only evaluated if counts match)
    """
    gt = len(expected_intervals)
    det = len(cut_frames)
    count_err = abs(det - gt)
    if count_err != 0 or gt == 0:
        return (count_err, 9999)
    mismatches = 0
    cut_frames_sorted = sorted([int(x) for x in cut_frames])
    for cf, (lo, hi) in zip(cut_frames_sorted, expected_intervals, strict=True):
        if not (int(lo) <= int(cf) <= int(hi)):
            mismatches += 1
    return (count_err, mismatches)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid SceneShifter scene-transition detection (instant then gradual on failures).")
    parser.add_argument("--input-dir", type=str, default="", help="Run directory containing videos/. The shell always passes this.")
    parser.add_argument("--glob", type=str, default="*.mp4")
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default=str(_BENCH_ROOT / "configs" / "sceneshifter_multitalk"),
        help="Directory containing <id>_metadata.json files.",
    )
    parser.add_argument("--compression-ratio", type=int, default=4)
    parser.add_argument(
        "--guided-gradual",
        action="store_true",
        help="When applying gradual fallback, use GT-guided clustering to pick one cut per expected interval.",
    )
    parser.add_argument("--output-json", type=str, default=None)
    add_ids_resume_args(parser)
    args = parser.parse_args()
    if not str(args.input_dir).strip():
        raise SystemExit("--input-dir is required")

    input_dir = Path(args.input_dir).expanduser().resolve()
    videos_dir = input_dir / "videos"
    if not videos_dir.exists():
        raise SystemExit(f"Missing videos dir: {videos_dir}")

    metadata_dir = Path(args.metadata_dir).expanduser().resolve()
    if not metadata_dir.exists():
        raise SystemExit(f"Missing metadata dir: {metadata_dir}")

    out_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else (input_dir / "scene_transitions_hybrid.json")
    )

    from detect_scene_transitions import detect_cuts, _evaluate_against_metadata  # type: ignore

    # Preset params (must match detect_scene_transitions.py presets).
    instant_cfg = dict(
        detector="content",
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
        # More sensitive base; GT-guided selection will choose the best cluster per expected interval.
        min_scene_len=1,
        adaptive_threshold=1.0,
        adaptive_min_content_val=3.0,
        # We handle collapsing ourselves when guided-gradual is enabled; otherwise these are used.
        collapse_max_gap=3,
        collapse_min_cluster_size=2,
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

    instant_cache: Dict[int, Dict[str, Any]] = {}
    gradual_cache: Dict[int, Dict[str, Any]] = {}
    failed_ids: List[int] = []

    # Pass 1: instant for all; decide failures vs GT.
    for vp in videos:
        vid = _video_id_from_path(vp)
        if vid is None:
            continue
        splits = _load_frame_splits(metadata_dir, vid)
        if splits is None:
            # Skip videos without GT.
            continue
        expected_intervals = _expected_intervals_from_splits(splits, int(args.compression_ratio))

        r_inst = detect_cuts(vp, **instant_cfg)
        last_idx = last_frame_index_from_video_path(vp)
        inst_cuts = filter_last_frame_cuts(
            ([int(c.frame) for c in r_inst.cuts] if r_inst.ok else []),
            last_idx,
        )
        score_inst = _score_vs_gt(inst_cuts, expected_intervals=expected_intervals)
        if score_inst != (0, 0):
            failed_ids.append(vid)

        instant_cache[vid] = {
            "result": r_inst,
            "cut_frames": inst_cuts,
            "score": score_inst,
            "splits": splits,
            "expected_intervals": expected_intervals,
        }

    # Pass 2: gradual on failed only.
    for vid in failed_ids:
        vp = _resolve_video_path(videos_dir, vid)
        if vp is None:
            continue
        # If guided, get raw cuts first (no collapse), then select cuts based on expected intervals.
        if args.guided_gradual:
            r_grad = detect_cuts(
                vp,
                detector=gradual_cfg["detector"],
                threshold=gradual_cfg["threshold"],
                min_scene_len=gradual_cfg["min_scene_len"],
                adaptive_threshold=gradual_cfg["adaptive_threshold"],
                adaptive_min_content_val=gradual_cfg["adaptive_min_content_val"],
                collapse_max_gap=None,
                collapse_min_cluster_size=1,
            )
            last_idx = last_frame_index_from_video_path(vp)
            raw = filter_last_frame_cuts(
                ([int(c.frame) for c in r_grad.cuts] if r_grad.ok else []),
                last_idx,
            )
            expected_intervals = instant_cache[vid]["expected_intervals"]
            grad_cuts = _guided_select_cuts_from_raw(
                raw,
                expected_intervals=expected_intervals,
                cluster_max_gap=int(gradual_cfg["collapse_max_gap"]),
                min_cluster_size=int(gradual_cfg["collapse_min_cluster_size"]),
            )
        else:
            r_grad = detect_cuts(vp, **gradual_cfg)
            last_idx = last_frame_index_from_video_path(vp)
            grad_cuts = filter_last_frame_cuts(
                ([int(c.frame) for c in r_grad.cuts] if r_grad.ok else []),
                last_idx,
            )
        splits = instant_cache[vid]["splits"]
        expected_intervals = instant_cache[vid]["expected_intervals"]
        score_grad = _score_vs_gt(grad_cuts, expected_intervals=expected_intervals)
        gradual_cache[vid] = {"result": r_grad, "cut_frames": grad_cuts, "score": score_grad}

    # Combine per-video.
    combined_results: List[Dict[str, Any]] = []
    switched_to_gradual: List[int] = []

    for vp in videos:
        vid = _video_id_from_path(vp)
        if vid is None:
            continue
        if vid not in instant_cache:
            continue

        inst = instant_cache[vid]
        chosen = "instant"
        chosen_result = inst["result"]
        chosen_cut_frames = inst["cut_frames"]
        chosen_score = inst["score"]

        if vid in gradual_cache:
            grad = gradual_cache[vid]
            if tuple(grad["score"]) < tuple(chosen_score):
                chosen = "gradual"
                chosen_result = grad["result"]
                chosen_cut_frames = grad["cut_frames"]
                chosen_score = grad["score"]
                switched_to_gradual.append(vid)

        # Encode output. Keep it compatible with prior JSON shape + add diagnostics.
        entry = {
            **asdict(chosen_result),
            "cuts": [{"frame": int(f), "timecode": ""} for f in chosen_cut_frames],
            "has_transition": len(chosen_cut_frames) > 0,
            "source": chosen,
            "score": {"count_err": int(chosen_score[0]), "timing_mismatches": int(chosen_score[1])},
            "instant_score": {"count_err": int(inst["score"][0]), "timing_mismatches": int(inst["score"][1])},
        }
        if vid in gradual_cache:
            gs = gradual_cache[vid]["score"]
            entry["gradual_score"] = {"count_err": int(gs[0]), "timing_mismatches": int(gs[1])}
        entry["video_id"] = int(vid)
        combined_results.append(entry)

    if args.resume and existing_results:
        combined_results = merge_list_by_video_id(
            existing_results,
            combined_results,
            id_fields=("video_id", "video"),
        )
        # Preserve prior switched ids and extend with this batch.
        prev_switched = []
        if isinstance(existing_payload, dict):
            prev_switched = list((existing_payload.get("summary") or {}).get("switched_video_ids") or [])
        switched_to_gradual = sorted(set(int(x) for x in prev_switched) | set(switched_to_gradual))

    payload: Dict[str, Any] = {
        "input_dir": str(input_dir),
        "glob": args.glob,
        "hybrid_policy": "instant_for_all_then_gradual_on_gt_failures_choose_best_by(count_err, timing_mismatches)",
        "presets": {"instant": instant_cfg, "gradual": gradual_cfg},
        "results": combined_results,
        "summary": {
            "total_videos": len(combined_results),
            "instant_failed_by_gt": len(failed_ids),
            "switched_to_gradual": len(switched_to_gradual),
            "switched_video_ids": sorted(switched_to_gradual),
        },
    }

    # Compute evaluation_summary using the existing evaluation function (frame_splits mode).
    # We rebuild VideoResult-like dicts into dataclasses by reusing chosen_result fields already present,
    # but simplest is to call the evaluation helper with detect_cuts results; we don't have dataclasses here.
    # Instead, we use _evaluate_against_metadata directly by re-detecting? Avoid. We'll just approximate by
    # calling the helper on the stored outputs is not possible without dataclasses. So we compute via a light re-run:
    # (Still fast: 150 videos, but we're already here; to keep I/O minimal, reuse instant/gradual cached VideoResult objects.)
    #
    # We'll reconstruct minimal objects that mimic VideoResult dataclass shape.
    class _VR:
        def __init__(self, d: Dict[str, Any]):
            self.video = d["video"]
            self.ok = d["ok"]
            self.error = d.get("error")
            self.fps = d.get("fps")
            self.num_scenes = d.get("num_scenes")
            self.has_transition = d.get("has_transition", False)
            self.cuts = [type("C", (), {"frame": c["frame"], "timecode": c.get("timecode", "")}) for c in d.get("cuts", [])]

    vr_list = [_VR(r) for r in combined_results]
    eval_result = _evaluate_against_metadata(
        vr_list,
        metadata_dir=metadata_dir,
        compression_ratio=int(args.compression_ratio),
        single_prompt=False,
    )
    payload = {"evaluation_summary": eval_result["evaluation_summary"], **payload, "evaluation": eval_result}

    write_json(out_json, payload)

    print(f"Wrote: {out_json}")
    print(json.dumps(payload["summary"], indent=2))
    print(json.dumps(payload["evaluation_summary"], indent=2))


if __name__ == "__main__":
    main()

