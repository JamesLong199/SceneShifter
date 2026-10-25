#!/usr/bin/env python3
"""
Compute background visual quality for SceneShifter videos by scene segments.

Pipeline:
1) Read cut frames from:
   results/.../videos/scene_transitions.json
   For each video, we treat cuts as scene boundaries:
   - scene0: [0, cut0)
   - scene1: [cut0, cut1)
   - scene2: [cut1, cut2)
   - ...
2) Read per-scene prompts from metadata (best-effort, not required for metrics):
   metadata_jsons_v2_simplified/<id>_metadata.json -> frame_prompt_routing.segment_prompts
3) Build per-scene clips from extracted frames:
   results/.../frames/<id>/*.jpg  -> write mp4 for each scene segment
4) Evaluate VBench metrics on these per-scene clips:
   - Aesthetic Quality (VBench "aesthetic_quality")
   - Image Quality (VBench "imaging_quality")
5) Save a single JSON summary file with:
   - per-video per-scene scores (scene0/scene1/scene2/...)
   - an overall score for scenes after scene0 (scene1+), aggregated from scene1..N
   - aggregate means across videos for each scene key

Notes / gotchas:
- VBench's loaders read videos (.mp4/.gif) or single images (.png). They do NOT
  directly accept a frames folder, so we write per-video mp4 clips.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch
import shutil

_SCRIPTS_DIR = Path(__file__).resolve().parent
_BENCH_ROOT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from eval_resume_utils import (  # noqa: E402
    add_ids_resume_args,
    filter_last_frame_cuts,
    ids_from_dict_keys,
    last_frame_index_from_frames_dir,
    load_json,
    log_resume_plan,
    merge_dict_by_video_key,
    parse_ids,
    resolve_todo_ids,
)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        return iterable


def _natural_sort_key(p: Path) -> Tuple:
    import re as _re

    parts: List[Any] = []
    for chunk in _re.split(r"(\d+)", p.name):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            parts.append(chunk.lower())
    return tuple(parts)


def _require_code_repo(env_name: str, cli_value: str = "") -> Path:
    raw = str(cli_value or "").strip() or os.environ.get(env_name, "").strip()
    if not raw:
        raise SystemExit(f"{env_name} is unset. Export {env_name} to the code repository.")
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"{env_name} does not exist: {path}")
    return path


def _video_id_from_path(video_path: Path) -> Optional[int]:
    m = re.match(r"^(\d+)\.mp4$", video_path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_segment_prompts(metadata_path: Path) -> List[str]:
    try:
        data = _read_json(metadata_path)
    except Exception:
        return []
    routing = data.get("frame_prompt_routing", {})
    prompts = routing.get("segment_prompts", [])
    if not isinstance(prompts, list):
        return []
    out: List[str] = []
    for p in prompts:
        out.append(p if isinstance(p, str) else "")
    return out


def _list_frame_files(frames_dir: Path) -> List[Path]:
    # Accept both 00.jpg / 01.jpg style and 0.jpg / 1.jpg style
    frames = [p for p in frames_dir.glob("*.jpg") if p.is_file()]
    return sorted(frames, key=_natural_sort_key)


def _frame_index_base(frames: List[Path]) -> int:
    """
    Determine whether frames are 0-based (00.jpg) or 1-based (01.jpg).
    Returns the minimum stem integer, typically 0 or 1.
    """
    if not frames:
        return 0
    vals: List[int] = []
    for p in frames:
        try:
            vals.append(int(p.stem))
        except Exception:
            continue
    return min(vals) if vals else 0


def _write_mp4_from_frames(
    frame_paths: List[Path],
    output_mp4: Path,
    *,
    fps: float = 25.0,
) -> None:
    if not frame_paths:
        raise ValueError("No frames provided")

    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    # Read first frame for size
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise ValueError(f"Failed to read frame: {frame_paths[0]}")
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_mp4), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {output_mp4}")

    try:
        for fp in frame_paths:
            img = cv2.imread(str(fp))
            if img is None:
                raise ValueError(f"Failed to read frame: {fp}")
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(img)
    finally:
        writer.release()


@dataclass
class VideoJob:
    video_id: int
    frames_dir: Path
    metadata_path: Path
    cuts: List[int]
    segment_prompts: List[str]
    frame_splits: List[int]


def _build_jobs(
    *,
    scene_transitions_path: Path,
    frames_root: Path,
    metadata_dir: Path,
    only_ids: Optional[List[int]] = None,
) -> List[VideoJob]:
    data = _read_json(scene_transitions_path)
    results = data.get("results", [])
    jobs: List[VideoJob] = []

    for r in results:
        try:
            video_path = Path(r["video"])
        except Exception:
            continue
        vid = _video_id_from_path(video_path)
        if vid is None:
            continue
        if only_ids is not None and vid not in only_ids:
            continue

        cuts_raw = r.get("cuts", []) if isinstance(r, dict) else []
        cuts: List[int] = []
        if isinstance(cuts_raw, list):
            for c in cuts_raw:
                try:
                    cuts.append(int(c["frame"]))
                except Exception:
                    continue
        cuts = sorted(set(cuts))

        frames_dir = frames_root / f"{vid:03d}"
        if not frames_dir.exists():
            continue
        cuts = filter_last_frame_cuts(cuts, last_frame_index_from_frames_dir(frames_dir))

        metadata_path = metadata_dir / f"{vid:03d}_metadata.json"
        segment_prompts = _load_segment_prompts(metadata_path)

        frame_splits = _load_frame_splits(metadata_path)

        jobs.append(
            VideoJob(
                video_id=vid,
                frames_dir=frames_dir,
                metadata_path=metadata_path,
                cuts=cuts,
                segment_prompts=segment_prompts,
                frame_splits=frame_splits,
            )
        )

    return jobs


@dataclass(frozen=True)
class SceneSegment:
    key: str  # e.g. "scene0", "scene1", "scene2", ...
    start_frame: int  # inclusive, dataset frame index space (0-based)
    end_frame_exclusive: int  # exclusive, dataset frame index space (0-based)
    prompt: str


@dataclass(frozen=True)
class SceneClipInfo:
    segment: SceneSegment
    num_frames_selected: int
    clip_path: Path

def _ensure_symlink_or_copy(src: Path, dst: Path) -> None:
    """
    Create dst pointing to src. Prefer symlink; fall back to copy if symlinks are not permitted.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def _load_frame_splits(metadata_path: Path) -> List[int]:
    try:
        data = _read_json(metadata_path)
    except Exception:
        return []
    routing = data.get("frame_prompt_routing", {})
    splits = routing.get("frame_splits", [])
    if not isinstance(splits, list):
        return []
    out: List[int] = []
    for s in splits:
        try:
            out.append(int(s))
        except Exception:
            continue
    return out


def _select_frames_for_dataset_range(
    all_frames: List[Path],
    *,
    base: int,
    start_idx: int,
    end_exclusive_idx: int,
) -> List[Path]:
    """
    Select frame paths where dataset frame index i is in [start_idx, end_exclusive_idx).
    dataset index i is derived from filename stem via: i = int(stem) - base.
    """
    selected: List[Path] = []
    start_stem = start_idx + base
    end_stem_exclusive = end_exclusive_idx + base
    for p in all_frames:
        try:
            stem_i = int(p.stem)
        except Exception:
            continue
        if start_stem <= stem_i < end_stem_exclusive:
            selected.append(p)
    return selected


def _total_frames_from_frame_files(all_frames: List[Path], *, base: int) -> int:
    max_idx = -1
    for p in all_frames:
        try:
            stem_i = int(p.stem)
        except Exception:
            continue
        max_idx = max(max_idx, stem_i - base)
    return max_idx + 1 if max_idx >= 0 else 0


def _append_after_scene0_segment(
    segments: List[SceneSegment], *, cuts: List[int], total_frames: int, prompts: List[str]
) -> List[SceneSegment]:
    """
    Add a combined segment for "scene1+" without modifying VBench:
    - after_scene0_overall: [cut0, total_frames)
    This lets VBench evaluate the combined frames directly (instead of combining per-scene scores).
    """
    if not cuts:
        return segments
    cut0 = int(sorted(cuts)[0])
    if cut0 >= total_frames:
        return segments
    prompt = prompts[1] if len(prompts) > 1 else ""
    return segments + [
        SceneSegment(key="after_scene0_overall", start_frame=cut0, end_frame_exclusive=total_frames, prompt=prompt)
    ]

def _build_segments_for_job(
    *,
    cuts: List[int],
    total_frames: int,
    prompts: List[str],
) -> List[SceneSegment]:
    """
    Always include scene0, plus after_scene0_overall when a cut exists.
    """
    cuts_sorted = sorted([int(c) for c in cuts if c is not None])
    cut0 = cuts_sorted[0] if cuts_sorted else None

    # scene0 is always present. If no cut, it spans the full video.
    scene0_end = int(cut0) if cut0 is not None else int(total_frames)
    segs: List[SceneSegment] = []
    if scene0_end > 0:
        segs.append(
            SceneSegment(
                key="scene0",
                start_frame=0,
                end_frame_exclusive=scene0_end,
                prompt=(prompts[0] if len(prompts) > 0 else ""),
            )
        )

    # Combined scene1+ segment
    segs = _append_after_scene0_segment(segs, cuts=cuts_sorted, total_frames=total_frames, prompts=prompts)
    return segs


def _ensure_clips(
    jobs: List[VideoJob],
    *,
    clips_dir: Path,
    fps: float,
) -> Dict[str, List[SceneClipInfo]]:
    """Write scene0 and after-scene0 clips. Returns a manifest keyed by 3-digit id."""
    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, List[SceneClipInfo]] = {}

    for job in tqdm(jobs, desc="Creating scene clips", unit="video"):
        all_frames = _list_frame_files(job.frames_dir)
        if not all_frames:
            continue

        base = _frame_index_base(all_frames)
        total_frames = _total_frames_from_frame_files(all_frames, base=base)
        if total_frames <= 0:
            continue

        segments = _build_segments_for_job(
            cuts=job.cuts,
            total_frames=total_frames,
            prompts=job.segment_prompts,
        )
        if not segments:
            continue

        vid_key = f"{job.video_id:03d}"
        seg_infos: List[SceneClipInfo] = []
        for seg in segments:
            clip_path = clips_dir / f"{job.video_id:03d}_{seg.key}.mp4"
            selected = _select_frames_for_dataset_range(
                all_frames, base=base, start_idx=seg.start_frame, end_exclusive_idx=seg.end_frame_exclusive
            )
            if not selected:
                continue
            if seg.key == "after_scene0_overall":
                legacy = clips_dir / f"{job.video_id:03d}_after_scene0.mp4"
                if (not clip_path.exists()) and legacy.exists():
                    _ensure_symlink_or_copy(legacy, clip_path)
            if not clip_path.exists():
                _write_mp4_from_frames(selected, clip_path, fps=fps)
            seg_infos.append(
                SceneClipInfo(
                    segment=seg,
                    num_frames_selected=len(selected),
                    clip_path=clip_path,
                )
            )
        if seg_infos:
            manifest[vid_key] = seg_infos
    return manifest


def _write_vbench_full_info(
    clips_manifest: Dict[str, List[SceneClipInfo]],
    full_info_path: Path,
    *,
    include_dimensions: List[str],
) -> None:
    payload: List[Dict[str, Any]] = []
    for _, seg_infos in clips_manifest.items():
        for info in seg_infos:
            clip_path = info.clip_path
            if not clip_path.exists():
                continue
            payload.append(
                {
                    "prompt_en": info.segment.prompt,
                    "dimension": include_dimensions,
                    "video_list": [str(clip_path)],
                }
            )
    full_info_path.parent.mkdir(parents=True, exist_ok=True)
    full_info_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_vbench(
    full_info_path: Path,
    *,
    device: torch.device,
    vbench_repo: Path,
    local: bool,
    imaging_quality_preprocessing_mode: str,
) -> Dict[str, Any]:
    if str(vbench_repo) not in sys.path:
        sys.path.insert(0, str(vbench_repo))

    from vbench.utils import init_submodules  # type: ignore
    from vbench.aesthetic_quality import compute_aesthetic_quality  # type: ignore
    from vbench.imaging_quality import compute_imaging_quality  # type: ignore

    dimensions = ["aesthetic_quality", "imaging_quality"]
    submodules = init_submodules(dimensions, local=local, read_frame=False)

    aes_all, aes_per = compute_aesthetic_quality(str(full_info_path), device, submodules["aesthetic_quality"])
    img_all, img_per = compute_imaging_quality(
        str(full_info_path),
        device,
        submodules["imaging_quality"],
        imaging_quality_preprocessing_mode=imaging_quality_preprocessing_mode,
    )

    return {
        "summary": {
            "aesthetic_quality": (float(aes_all) if aes_all is not None else None),
            "imaging_quality": (float(img_all) if img_all is not None else None),
        },
        "per_video": {
            "aesthetic_quality": aes_per,
            "imaging_quality": img_per,
        },
    }


def _index_results_by_video_path(per_video_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for d in per_video_list:
        vp = d.get("video_path")
        if not vp:
            continue
        out[str(vp)] = d
    return out


def _scene_index_from_key(key: str) -> Optional[int]:
    m = re.match(r"^scene(\d+)$", key)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute background visual quality for scene0 and the frames after scene0.")
    parser.add_argument("--scene-transitions-json", type=str, default="", help="Path to scene_transitions.json. The shell always passes this.")
    parser.add_argument("--frames-root", type=str, default="", help="Root dir containing per-video frame folders.")
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default=str(_BENCH_ROOT / "configs" / "sceneshifter_multitalk"),
        help="Directory containing SceneShifter metadata jsons.",
    )
    parser.add_argument("--output-dir", type=str, default="", help="Directory to write the VBench input json and summary.")
    parser.add_argument(
        "--sub-clip-dir",
        type=str,
        default="",
        help="Shared directory for preprocessed clips. Empty defaults to <output-dir>/../sub_clip.",
    )
    parser.add_argument(
        "--vbench-repo",
        type=str,
        default="",
        help="VBench code repository. Defaults to the VBENCH_REPO environment variable.",
    )
    parser.add_argument(
        "--vbench-cache-dir",
        type=str,
        default=str(_BENCH_ROOT / "vbench"),
        help="Path to the VBench cache dir containing checkpoints (sets VBENCH_CACHE_DIR).",
    )
    parser.add_argument("--fps", type=float, default=25.0, help="FPS for written clips.")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Pass local=True to VBench init_submodules (assumes weights already downloaded in cache).",
    )
    parser.add_argument(
        "--imaging-quality-preprocessing-mode",
        type=str,
        default="longer",
        choices=["shorter", "longer", "shorter_centercrop", "None"],
        help="VBench imaging_quality preprocessing mode.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for VBench evaluation: 'cuda' or 'cpu'.")
    add_ids_resume_args(parser)
    args = parser.parse_args()
    if not str(args.scene_transitions_json).strip() or not str(args.frames_root).strip() or not str(args.output_dir).strip():
        raise SystemExit("--scene-transitions-json, --frames-root, and --output-dir are required")

    vbench_repo = _require_code_repo("VBENCH_REPO", args.vbench_repo)
    if args.vbench_cache_dir.strip():
        os.environ["VBENCH_CACHE_DIR"] = str(Path(args.vbench_cache_dir).expanduser().resolve())

    scene_transitions_path = Path(args.scene_transitions_json).expanduser().resolve()
    frames_root = Path(args.frames_root).expanduser().resolve()
    metadata_dir = Path(args.metadata_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    sub_clip_dir = (
        Path(args.sub_clip_dir).expanduser().resolve()
        if str(args.sub_clip_dir).strip()
        else (output_dir.parent / "sub_clip")
    )
    clips_dir = sub_clip_dir
    legacy_clips_dir = output_dir / "clips"
    if legacy_clips_dir.exists() and legacy_clips_dir.is_dir() and (legacy_clips_dir.resolve() != sub_clip_dir.resolve()):
        for legacy_clip in legacy_clips_dir.glob("*.mp4"):
            if legacy_clip.is_file():
                _ensure_symlink_or_copy(legacy_clip, sub_clip_dir / legacy_clip.name)
    full_info_path = output_dir / "vbench_full_info_scenes.json"
    summary_path = output_dir / "background_quality_scenes_results.json"

    if not scene_transitions_path.exists():
        raise SystemExit(f"Missing scene transitions json: {scene_transitions_path}")
    if not frames_root.exists():
        raise SystemExit(f"Missing frames root: {frames_root}")
    if not metadata_dir.exists():
        raise SystemExit(f"Missing metadata dir: {metadata_dir}")

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    requested_ids = parse_ids(args.ids)
    existing = load_json(summary_path) if args.resume else None
    existing_videos = (
        existing.get("videos", {})
        if isinstance(existing, dict) and isinstance(existing.get("videos"), dict)
        else {}
    )
    done_ids = ids_from_dict_keys(existing_videos) if args.resume else set()
    prefilter = sorted(requested_ids) if requested_ids is not None else None
    jobs_all = _build_jobs(
        scene_transitions_path=scene_transitions_path,
        frames_root=frames_root,
        metadata_dir=metadata_dir,
        only_ids=prefilter,
    )
    available_ids = [int(j.video_id) for j in jobs_all]
    todo_ids, skipped_ids = resolve_todo_ids(
        requested_ids=requested_ids,
        available_ids=available_ids,
        done_ids=done_ids,
        resume=bool(args.resume),
    )
    log_resume_plan(label="bg_quality", todo=todo_ids, skipped=skipped_ids, requested=requested_ids)
    jobs = [j for j in jobs_all if int(j.video_id) in todo_ids]
    print(f"Found {len(jobs)} videos with frames + (optional) transitions + metadata.")
    if not jobs:
        if args.resume and existing is not None:
            print("[bg_quality] nothing to evaluate; keeping existing results")
            summary_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            return
        print("[bg_quality] no jobs to evaluate")
        return

    clips_manifest = _ensure_clips(jobs, clips_dir=clips_dir, fps=float(args.fps))
    print(f"Prepared clips for {len(clips_manifest)} videos.")
    include_dimensions = ["aesthetic_quality", "imaging_quality"]
    _write_vbench_full_info(clips_manifest, full_info_path, include_dimensions=include_dimensions)
    print(f"Wrote VBench full_info JSON: {full_info_path}")
    vbench_out = _run_vbench(
        full_info_path,
        device=device,
        vbench_repo=vbench_repo,
        local=bool(args.local),
        imaging_quality_preprocessing_mode=args.imaging_quality_preprocessing_mode,
    )
    aes_by_path = _index_results_by_video_path(vbench_out["per_video"]["aesthetic_quality"])
    img_by_path = _index_results_by_video_path(vbench_out["per_video"]["imaging_quality"])

    videos_out: Dict[str, Any] = {}
    agg: Dict[str, Dict[str, List[float]]] = {}
    for job in jobs:
        vid_key = f"{job.video_id:03d}"
        seg_infos = clips_manifest.get(vid_key, [])
        if not seg_infos:
            continue
        scenes_out: Dict[str, Any] = {}
        for info in seg_infos:
            scene_key = info.segment.key
            clip_path = str(info.clip_path)
            aes_val = float(aes_by_path.get(clip_path, {}).get("video_results", 0.0))
            img_val = float(img_by_path.get(clip_path, {}).get("video_results", 0.0))
            scenes_out[scene_key] = {
                "prompt": info.segment.prompt,
                "start_frame": info.segment.start_frame,
                "end_frame_exclusive": info.segment.end_frame_exclusive,
                "num_frames_selected": info.num_frames_selected,
                "clip_path": clip_path,
                "metrics": {
                    "all_frames": {
                        "aesthetic_quality": aes_val,
                        "imaging_quality": img_val,
                    },
                },
            }
            bucket = agg.setdefault(scene_key, {"aesthetic_quality": [], "imaging_quality": []})
            bucket["aesthetic_quality"].append(aes_val)
            bucket["imaging_quality"].append(img_val)
        videos_out[vid_key] = {
            "video_id": job.video_id,
            "cuts": job.cuts,
            "num_scenes": len([k for k in scenes_out.keys() if _scene_index_from_key(k) is not None]),
            "frame_splits": job.frame_splits,
            "scenes": scenes_out,
        }

    def _summarize(videos: Dict[str, Any]) -> Dict[str, Any]:
        buckets: Dict[str, Dict[str, List[float]]] = {}
        for video in videos.values():
            scenes = video.get("scenes") if isinstance(video, dict) else None
            if not isinstance(scenes, dict):
                continue
            for scene_key, scene in scenes.items():
                metrics = scene.get("metrics") if isinstance(scene, dict) else None
                mode_metrics = metrics.get("all_frames") if isinstance(metrics, dict) else None
                if not isinstance(mode_metrics, dict):
                    continue
                bucket = buckets.setdefault(scene_key, {"aesthetic_quality": [], "imaging_quality": []})
                for metric_key in ("aesthetic_quality", "imaging_quality"):
                    value = mode_metrics.get(metric_key)
                    if isinstance(value, (int, float)):
                        bucket[metric_key].append(float(value))
        return {
            scene_key: {
                "aesthetic_quality": _mean(vals.get("aesthetic_quality", [])),
                "imaging_quality": _mean(vals.get("imaging_quality", [])),
            }
            for scene_key, vals in buckets.items()
        }

    if args.resume and existing_videos:
        videos_out = merge_dict_by_video_key(existing_videos, videos_out)
    if args.resume and existing_videos:
        summary_all = _summarize(videos_out)
    else:
        summary_all = {
            scene_key: {
                "aesthetic_quality": _mean(vals.get("aesthetic_quality", [])),
                "imaging_quality": _mean(vals.get("imaging_quality", [])),
            }
            for scene_key, vals in agg.items()
        }

    out = {
        "meta": {
            "scene_transitions_json": str(scene_transitions_path),
            "frames_root": str(frames_root),
            "metadata_dir": str(metadata_dir),
            "output_dir": str(output_dir),
            "full_info_json": str(full_info_path),
            "clips_dir": str(clips_dir),
            "sub_clip_dir": str(sub_clip_dir),
            "fps": float(args.fps),
            "device": str(device),
            "imaging_quality_preprocessing_mode": args.imaging_quality_preprocessing_mode,
            "vbench_repo": str(vbench_repo),
        },
        "summary": {"all_frames": summary_all},
        "videos": videos_out,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
    main()
