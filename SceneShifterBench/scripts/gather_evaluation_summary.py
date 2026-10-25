#!/usr/bin/env python3
"""
Gather overall statistics from all evaluation result JSON files produced by evaluate_sceneshifter.sh
"""
import json
import argparse
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import statistics
import math


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def compute_mean(values: List[float]) -> Optional[float]:
    """Compute mean of a list of values."""
    if not values:
        return None
    return statistics.mean(values)


def parse_ids(value: str) -> Optional[Set[int]]:
    """Parse comma-separated ids/ranges like '1,003,10-12'."""
    value = str(value or "").strip()
    if not value:
        return None

    out: Set[int] = set()
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid --ids range: {token}")
            out.update(range(start, end + 1))
        else:
            out.add(int(token))
    return out


def _extract_video_id(value: Any) -> Optional[int]:
    """Extract a numeric video id from common fields/paths/keys."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value)
    m = re.search(r"(\d+)", Path(text).stem)
    if not m:
        m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else None


def _matches_ids(obj: Any, ids: Optional[Set[int]]) -> bool:
    if ids is None:
        return True
    if isinstance(obj, dict):
        for key in ("video_id", "id", "video", "video_path", "metadata_path"):
            vid = _extract_video_id(obj.get(key))
            if vid is not None:
                return vid in ids
        return False
    vid = _extract_video_id(obj)
    return vid in ids if vid is not None else False


def _is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))


def _mean_key(results: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals: List[float] = []
    for r in results:
        v = r.get(key)
        if _is_finite_number(v):
            vals.append(float(v))
    return compute_mean(vals)


def extract_temporal_accuracy(scene_transitions_data: Dict[str, Any], ids: Optional[Set[int]] = None) -> Dict[str, Any]:
    """Temporal Accuracy: scene change success rate + scene timing accuracy (from scene_transitions.json)."""
    out: Dict[str, Any] = {}
    if ids is not None:
        results = scene_transitions_data.get("results")
        evaluation = scene_transitions_data.get("evaluation")
        timing = evaluation.get("timing") if isinstance(evaluation, dict) else None
        per_video = timing.get("per_video") if isinstance(timing, dict) else None
        failures = evaluation.get("failures") if isinstance(evaluation, dict) else None
        if isinstance(results, list):
            selected_results = [r for r in results if _matches_ids(r, ids)]
            selected_ids = {
                vid
                for vid in (_extract_video_id(r.get("video_id", r.get("video"))) for r in selected_results if isinstance(r, dict))
                if vid is not None
            }

            success_ids: Set[int] = set()
            accurate_successful_cuts = 0
            total_gt_transitions = 0
            if isinstance(per_video, list):
                for item in per_video:
                    if not _matches_ids(item, ids):
                        continue
                    vid = _extract_video_id(item.get("video_id", item.get("video"))) if isinstance(item, dict) else None
                    if vid is not None:
                        success_ids.add(vid)
                    cuts = item.get("cuts") if isinstance(item, dict) else None
                    if isinstance(cuts, list):
                        total_gt_transitions += len(cuts)
                        accurate_successful_cuts += sum(1 for c in cuts if isinstance(c, dict) and c.get("in_interval") is True)

            if isinstance(failures, list):
                for item in failures:
                    if not _matches_ids(item, ids) or not isinstance(item, dict):
                        continue
                    expected_cuts = item.get("expected_cuts")
                    if _is_finite_number(expected_cuts):
                        total_gt_transitions += int(expected_cuts)
                    elif isinstance(item.get("frame_splits"), list):
                        total_gt_transitions += len(item["frame_splits"])
                    elif isinstance(item.get("expected_parts"), list):
                        total_gt_transitions += len(item["expected_parts"])

            evaluated = len(selected_results)
            success = len(success_ids & selected_ids) if selected_ids else len(success_ids)
            out["scene_change_success_rate"] = {
                "percentage": round((success / evaluated) * 100, 2) if evaluated else None,
                "fraction": f"{success}/{evaluated}",
            }
            out["scene_timing_accuracy"] = {
                "percentage": round((accurate_successful_cuts / total_gt_transitions) * 100, 2)
                if total_gt_transitions
                else None,
                "fraction": f"{accurate_successful_cuts}/{total_gt_transitions}",
            }
            return out

    eval_summary = scene_transitions_data.get("evaluation_summary")
    if isinstance(eval_summary, dict):
        out["scene_change_success_rate"] = eval_summary.get("scene_change_success_rate", {})
        out["scene_timing_accuracy"] = eval_summary.get("scene_timing_accuracy", {})
    return out


def extract_spatial_subject_consistency_summary(data: Dict[str, Any], ids: Optional[Set[int]] = None) -> Dict[str, Any]:
    """Extract summary from spatial_subject_consistency.json."""
    summary = {}
    if 'videos' in data:
        iou_values: List[float] = []
        total_with_field = 0
        skipped_non_numeric = 0
        for video in data['videos']:
            if not _matches_ids(video, ids):
                continue
            if 'mean_iou_across_cuts' not in video:
                continue
            total_with_field += 1
            v = video.get('mean_iou_across_cuts')
            if _is_finite_number(v):
                iou_values.append(float(v))
            else:
                skipped_non_numeric += 1
        
        if iou_values:
            summary['mean_iou'] = compute_mean(iou_values)
            summary['count_valid'] = len(iou_values)
            summary['count_total_with_field'] = total_with_field
            summary['count_skipped_non_numeric'] = skipped_non_numeric
    return summary


def extract_background_quality_summary(data: Dict[str, Any], ids: Optional[Set[int]] = None) -> Dict[str, Any]:
    """Extract summary from background_quality_scenes_results.json."""
    if ids is not None and isinstance(data.get("videos"), dict):
        agg: Dict[str, Dict[str, Dict[str, List[float]]]] = {
            "all_frames": {},
            "no_transition_frames": {},
        }
        for key, video in data["videos"].items():
            if not _matches_ids({"video_id": key}, ids):
                continue
            scenes = video.get("scenes") if isinstance(video, dict) else None
            if not isinstance(scenes, dict):
                continue
            for scene_key, scene in scenes.items():
                metrics = scene.get("metrics") if isinstance(scene, dict) else None
                if not isinstance(metrics, dict):
                    continue
                for mode_key in ("all_frames", "no_transition_frames"):
                    mode_metrics = metrics.get(mode_key)
                    if not isinstance(mode_metrics, dict):
                        continue
                    bucket = agg.setdefault(mode_key, {}).setdefault(
                        scene_key, {"aesthetic_quality": [], "imaging_quality": []}
                    )
                    for metric_key in ("aesthetic_quality", "imaging_quality"):
                        v = mode_metrics.get(metric_key)
                        if _is_finite_number(v):
                            bucket[metric_key].append(float(v))

        summary: Dict[str, Any] = {"all_frames": {}, "no_transition_frames": {}}
        for mode_key, scenes in agg.items():
            if not scenes:
                summary[mode_key] = None if mode_key == "no_transition_frames" else {}
                continue
            summary[mode_key] = {
                scene_key: {
                    metric_key: compute_mean(vals)
                    for metric_key, vals in buckets.items()
                }
                for scene_key, buckets in scenes.items()
            }
        return summary

    summary = {}
    if 'summary' in data:
        summary = data['summary'].copy()

        # Backward-compat cleanup:
        # Older runs may contain both `after_scene0` and `after_scene0_overall` in the summary.
        # They refer to the same combined (scene1+) segment, so drop `after_scene0` to avoid duplication.
        for mode_key in ("all_frames", "no_transition_frames"):
            mode_obj = summary.get(mode_key)
            if isinstance(mode_obj, dict) and "after_scene0" in mode_obj and "after_scene0_overall" in mode_obj:
                try:
                    del mode_obj["after_scene0"]
                except Exception:
                    pass
    return summary


def extract_background_semantics_summary(data: Dict[str, Any], ids: Optional[Set[int]] = None) -> Dict[str, Any]:
    """Extract summary from background_semantics_vbench2_results.json."""
    if ids is not None and isinstance(data.get("videos"), dict):
        videos = {
            key: value
            for key, value in data["videos"].items()
            if _matches_ids({"video_id": key}, ids)
        }

        def _collect(metric: str, mode: str) -> List[float]:
            vals: List[float] = []
            for video in videos.values():
                avg = video.get("averages", {}).get(mode, {}) if isinstance(video, dict) else {}
                if isinstance(avg, dict):
                    v = avg.get(metric)
                    if _is_finite_number(v):
                        vals.append(float(v))
            return vals

        def _collect_combo_action() -> List[float]:
            vals: List[float] = []
            for video in videos.values():
                combo = video.get("whole_video", {}).get("combo_action", {}) if isinstance(video, dict) else {}
                if isinstance(combo, dict):
                    v = combo.get("score")
                    if _is_finite_number(v):
                        vals.append(float(v))
            return vals

        modes = set()
        for video in videos.values():
            averages = video.get("averages") if isinstance(video, dict) else None
            if isinstance(averages, dict):
                modes.update(averages.keys())
        combo_vals = _collect_combo_action()
        return {
            mode: {
                "background_semantics_presence_mean": compute_mean(_collect("background_semantics_presence", mode)),
                "background_semantics_perceptual_mean": compute_mean(_collect("background_semantics_perceptual", mode)),
                "interaction_naturalness_mean": compute_mean(_collect("interaction_naturalness", mode)),
                "combo_action_mean": compute_mean(combo_vals),
                "count": len(videos),
            }
            for mode in sorted(modes)
        }

    summary = {}
    if 'summary' in data:
        summary = data['summary'].copy()
    return summary


def extract_human_anatomy_summary(data: Dict[str, Any], ids: Optional[Set[int]] = None) -> Dict[str, Any]:
    """Human Anatomy summary (scene0 / scene1+ / overall) from human_anatomy_scene_splits.json."""
    results = data.get("results")
    if not isinstance(results, list):
        return {}
    if ids is not None:
        results = [r for r in results if _matches_ids(r, ids)]

    return {
        "scene0": _mean_key(results, "score_scene0_vbench2_style"),
        "scene1plus": _mean_key(results, "score_scene1plus_vbench2_style"),
        "overall": _mean_key(results, "score_overall_vbench2_style"),
        "count": len(results),
    }


def extract_human_identity_summary(data: Dict[str, Any], ids: Optional[Set[int]] = None) -> Dict[str, Any]:
    """Human Identity summary (scene0 / scene1+ / overall) from human_identity_scene_splits.json."""
    results = data.get("results")
    if not isinstance(results, list) or not results:
        return {}
    if ids is not None:
        results = [r for r in results if _matches_ids(r, ids)]
    if not results:
        return {}

    # Most fields are constant across entries; take from the first.
    first = results[0] if isinstance(results[0], dict) else {}
    return {
        "score_mode": first.get("score_mode") or "mean_similarity",
        "count": len(results),
        "mean_similarity": {
            "scene0": _mean_key(results, "mean_similarity_scene0"),
            "scene1plus": _mean_key(results, "mean_similarity_scene1plus"),
            "overall": _mean_key(results, "mean_similarity_overall"),
        },
    }


def extract_human_clothes_summary(data: Dict[str, Any], ids: Optional[Set[int]] = None) -> Dict[str, Any]:
    """Human Clothes summary from human_clothes_results.json."""
    if ids is not None and isinstance(data.get("results"), list):
        selected = [r for r in data["results"] if _matches_ids(r, ids)]
        valid = [r for r in selected if isinstance(r, dict) and r.get("video_results") != -1]
        passed = [r for r in valid if _is_finite_number(r.get("video_results")) and float(r["video_results"]) > 0]
        return {
            "score": (len(passed) / len(valid)) if valid else None,
            "valid_videos": len(valid),
            "passed_videos": len(passed),
        }

    out: Dict[str, Any] = {}
    for k in ("score", "valid_videos", "passed_videos"):
        if k in data:
            out[k] = data.get(k)
    return out


def extract_lipsync_summary(data: Dict[str, Any], ids: Optional[Set[int]] = None) -> Dict[str, Any]:
    """Extract summary from lipsync_syncnet_align224.json."""
    summary = {}

    if ids is not None and isinstance(data.get("results"), dict):
        conf_values: List[float] = []
        min_dist_values: List[float] = []
        offset_values: List[float] = []
        by_subject_values: Dict[str, Dict[str, List[float]]] = {}
        total = 0
        no_frames = 0

        for key, item in data["results"].items():
            if not _matches_ids({"video_id": key}, ids):
                continue
            total += 1
            syncnet = item.get("syncnet") if isinstance(item, dict) else None
            conf = syncnet.get("conf") if isinstance(syncnet, dict) else None
            min_dist = syncnet.get("min_dist") if isinstance(syncnet, dict) else None
            offset = syncnet.get("offset") if isinstance(syncnet, dict) else None
            if _is_finite_number(conf) and _is_finite_number(min_dist) and _is_finite_number(offset):
                conf_values.append(float(conf))
                min_dist_values.append(float(min_dist))
                offset_values.append(float(offset))
            else:
                no_frames += 1

            subjects = item.get("subjects") if isinstance(item, dict) else None
            if isinstance(subjects, dict):
                for subject_key, subject in subjects.items():
                    subject_sync = subject.get("syncnet") if isinstance(subject, dict) else None
                    if not isinstance(subject_sync, dict):
                        continue
                    bucket = by_subject_values.setdefault(
                        str(subject_key), {"conf": [], "min_dist": [], "offset": []}
                    )
                    for metric_key in ("conf", "min_dist", "offset"):
                        v = subject_sync.get(metric_key)
                        if _is_finite_number(v):
                            bucket[metric_key].append(float(v))

        summary["avg_conf"] = compute_mean(conf_values)
        summary["avg_min_dist"] = compute_mean(min_dist_values)
        summary["avg_offset"] = compute_mean(offset_values)
        summary["num_videos_scored"] = len(conf_values)
        summary["num_videos_total"] = total
        summary["num_videos_no_frames"] = no_frames
        summary["by_subject"] = {
            key: {
                "num_scored": len(vals["conf"]),
                "avg_conf": compute_mean(vals["conf"]),
                "avg_min_dist": compute_mean(vals["min_dist"]),
                "avg_offset": compute_mean(vals["offset"]),
            }
            for key, vals in sorted(by_subject_values.items())
        }
        return summary
    
    # Extract overall summary from meta.syncnet_summary
    if 'meta' in data and 'syncnet_summary' in data['meta']:
        syncnet_summary = data['meta']['syncnet_summary']
        summary['avg_conf'] = syncnet_summary.get('avg_conf')
        summary['avg_min_dist'] = syncnet_summary.get('avg_min_dist')
        summary['avg_offset'] = syncnet_summary.get('avg_offset')
        summary['num_videos_scored'] = syncnet_summary.get('num_videos_scored')
        summary['num_videos_total'] = syncnet_summary.get('num_videos_total')
        summary['num_videos_no_frames'] = syncnet_summary.get('num_videos_no_frames')
        if isinstance(syncnet_summary.get('by_subject'), dict):
            summary['by_subject'] = syncnet_summary.get('by_subject')
    
    return summary


def gather_all_summaries(run_dir: Path, ids: Optional[Set[int]] = None) -> Dict[str, Any]:
    """Gather evaluation summaries from a run directory, organized into 3 categories."""
    results: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "temporal_accuracy": {},
        "human_quality": {},
        "background_quality": {},
    }
    if ids is not None:
        results["ids"] = sorted(ids)

    # --- Temporal Accuracy ---
    scene_transitions_path = run_dir / "scene_transitions.json"
    if scene_transitions_path.exists():
        data = load_json(scene_transitions_path)
        results["temporal_accuracy"] = extract_temporal_accuracy(data, ids)

    # --- Human Quality ---
    spatial_consistency_path = run_dir / "spatial_subject_consistency.json"
    if spatial_consistency_path.exists():
        data = load_json(spatial_consistency_path)
        results["human_quality"]["spatial_subject_consistency"] = extract_spatial_subject_consistency_summary(data, ids)

    vbench_i2v_dir = run_dir / "i2v_scene0_eval" / "vbench_i2v_results"
    human_anatomy_path = vbench_i2v_dir / "human_anatomy_scene_splits.json"
    if human_anatomy_path.exists():
        data = load_json(human_anatomy_path)
        results["human_quality"]["human_anatomy"] = extract_human_anatomy_summary(data, ids)

    human_clothes_path = vbench_i2v_dir / "human_clothes_results.json"
    if human_clothes_path.exists():
        data = load_json(human_clothes_path)
        results["human_quality"]["human_clothes"] = extract_human_clothes_summary(data, ids)

    human_identity_path = vbench_i2v_dir / "human_identity_scene_splits.json"
    if human_identity_path.exists():
        data = load_json(human_identity_path)
        results["human_quality"]["human_identity"] = extract_human_identity_summary(data, ids)

    lipsync_path = run_dir / "lipsync_syncnet_align224.json"
    if lipsync_path.exists():
        data = load_json(lipsync_path)
        results["human_quality"]["lipsync_syncnet"] = extract_lipsync_summary(data, ids)

    # --- Background Quality ---
    bg_quality_path = run_dir / "background_quality_scene1" / "background_quality_scenes_results.json"
    if bg_quality_path.exists():
        data = load_json(bg_quality_path)
        bg_summary = extract_background_quality_summary(data, ids)
        # Aesthetic + Imaging quality for scene1+ (after_scene0_overall) in all_frames mode
        all_frames = bg_summary.get("all_frames") if isinstance(bg_summary, dict) else None
        after_scene0_overall = all_frames.get("after_scene0_overall") if isinstance(all_frames, dict) else None
        if isinstance(after_scene0_overall, dict):
            results["background_quality"]["aesthetic_quality_scene1plus"] = after_scene0_overall.get("aesthetic_quality")
            results["background_quality"]["imaging_quality_scene1plus"] = after_scene0_overall.get("imaging_quality")

    bg_semantics_path = run_dir / "background_semantics" / "background_semantics_vbench2_results.json"
    if bg_semantics_path.exists():
        data = load_json(bg_semantics_path)
        results["background_quality"]["background_semantics"] = extract_background_semantics_summary(data, ids)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Gather overall statistics from all evaluation result JSON files'
    )
    parser.add_argument(
        '--run-dir',
        type=str,
        required=True,
        help='Path to the run directory (e.g., results/sceneshifter_1231)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path (default: <run_dir>/evaluation_summary.json)'
    )
    parser.add_argument(
        '--ids',
        type=str,
        default='',
        help="Optional comma-separated video ids/ranges to summarize, e.g. '1,003,10-12'."
    )
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    
    ids = parse_ids(args.ids)

    # Gather all summaries
    summary = gather_all_summaries(run_dir, ids=ids)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = run_dir / 'evaluation_summary.json'
    
    # Write summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Evaluation summary written to: {output_path}")
    print(f"\nSummary:")
    print(f"  - Temporal Accuracy: {'✓' if summary.get('temporal_accuracy') else '✗'}")
    print(f"  - Human Quality: {'✓' if summary.get('human_quality') else '✗'}")
    print(f"  - Background Quality: {'✓' if summary.get('background_quality') else '✗'}")
    if ids is not None:
        print(f"  - IDs: {','.join(f'{i:03d}' for i in sorted(ids))}")


if __name__ == '__main__':
    main()
