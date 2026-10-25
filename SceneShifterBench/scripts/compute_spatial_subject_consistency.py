#!/usr/bin/env python3
"""
Compute spatial subject consistency around scene cut frames.

Definition:
- For a cut at frame c, compute IoU(mask(t), mask(t-1)) for t in [c-w, ..., c+w]
  and take the mean across valid t. Default w=0 => IoU(mask(c), mask(c-1)).

Human masks are extracted with the SAM3 image model, one frame at a time. A metadata
prompt that names both people is split into "the man" and "the woman"; each mask is
the highest-scoring detection at the processor confidence threshold, and the two
masks are unioned. Masks are cached so only the needed frames are processed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent
_BENCH_ROOT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from eval_resume_utils import (  # noqa: E402
    add_ids_resume_args,
    extract_video_id,
    filter_last_frame_cuts,
    ids_from_list,
    last_frame_index_from_frames_dir,
    load_json,
    log_resume_plan,
    merge_list_by_video_id,
    parse_ids,
    resolve_todo_ids,
    write_json,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_add_sam3_to_path() -> None:
    """Import sam3 from the active env, otherwise add bench_root/sam3."""
    try:
        import sam3  # noqa: F401
        return
    except Exception:
        pass
    sam3_root = _BENCH_ROOT / "sam3"
    if sam3_root.is_dir():
        sys.path.insert(0, str(sam3_root))
    try:
        import sam3  # noqa: F401
    except Exception as exc:
        raise FileNotFoundError(
            "Could not import sam3 from the active environment or "
            f"{sam3_root}."
        ) from exc


def load_sam3_model(sam3_ckpt: Optional[str] = None):
    """
    Load SAM3 image model and processor.
    """
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    checkpoint_path = sam3_ckpt or str(_BENCH_ROOT / "sam3" / "sam3.pt")
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            "SAM3 checkpoint not found. Provide --sam3_ckpt. Tried: "
            + str(checkpoint_path)
        )

    print(f"Loading SAM3 model from: {checkpoint_path}")
    model = build_sam3_image_model(checkpoint_path=checkpoint_path, load_from_HF=False)
    processor = Sam3Processor(model, confidence_threshold=0.5)
    return model, processor


def _frame_path(frames_dir: Path, frame_idx: int) -> Path:
    """
    Frames in this dataset are named like 01.jpg, 09.jpg, 10.jpg, 81.jpg.
    """
    candidates = [
        frames_dir / f"{frame_idx:02d}.jpg",
        frames_dir / f"{frame_idx}.jpg",
        frames_dir / f"{frame_idx:03d}.jpg",
        frames_dir / f"{frame_idx:04d}.jpg",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Frame file not found for index={frame_idx} under {frames_dir}")


def _video_id_from_video_path(video_path: str) -> Optional[str]:
    """Extract id from a video named '{id}.mp4'."""
    base = Path(video_path).name
    m = re.match(r"^(\d+)\.mp4$", base)
    if not m:
        return None
    return m.group(1).zfill(3)

def _infer_subject_prompt_from_metadata(metadata_dir: Path, video_id: str) -> str:
    """
    Infer a more specific SAM3 prompt from metadata segment_prompts.
    Falls back to "human" if metadata is missing/ambiguous.
    """
    try:
        vid_int = int(video_id)
    except Exception:
        return "human"
    meta_path = metadata_dir / f"{vid_int:03d}_metadata.json"
    if not meta_path.exists():
        return "human"
    try:
        data = _read_json(meta_path)
    except Exception:
        return "human"
    routing = data.get("frame_prompt_routing", {}) if isinstance(data, dict) else {}
    seg_prompts = routing.get("segment_prompts", [])
    if not isinstance(seg_prompts, list) or not seg_prompts:
        return "human"
    text = " ".join([str(p) for p in seg_prompts]).lower()
    # Check ordered multi-subject phrases before single-subject phrases.
    if "the man and the woman" in text:
        return "the man and the woman"
    if "the woman and the man" in text:
        return "the woman and the man"
    if "the woman" in text:
        return "the woman"
    if "the man" in text:
        return "the man"
    return "human"


def _subject_prompt_parts(prompt: str) -> List[str]:
    """
    A phrase that names both people is detected as two image prompts.
    Each part keeps its own highest-scoring mask, and the caller unions them.
    """
    low = prompt.lower()
    if " and " in low and "the man" in low and "the woman" in low:
        return ["the man", "the woman"]
    return [prompt]


def _load_subject_mask(
    frame_img_path: Path,
    processor,
    cache_dir: Path,
    prompt: str = "human",
) -> np.ndarray:
    parts = _subject_prompt_parts(prompt)
    masks = [
        _load_human_mask_with_cache(
            frame_img_path,
            processor,
            cache_dir,
            prompt=part,
        )
        for part in parts
    ]
    if len(masks) == 1:
        return masks[0]
    union = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        union |= mask.astype(bool)
    return union


def _cache_prompt_suffix(prompt: str) -> str:
    suffix = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    return suffix or "human"


def _default_frames_root(scene_transitions_json: Path) -> Path:
    """
    For .../videos/scene_transitions.json, default frames root is .../frames/
    """
    # videos/scene_transitions.json -> results/.../frames
    return scene_transitions_json.parent.parent / "frames"


def _default_cache_dir(scene_transitions_json: Path) -> Path:
    """
    Cache next to the transitions file, under .../videos/sam3_human_masks_cache/
    """
    return scene_transitions_json.parent / "sam3_human_masks_cache"


def _load_human_mask_with_cache(
    frame_img_path: Path,
    processor,
    cache_dir: Path,
    prompt: str = "human",
) -> np.ndarray:
    """
    Returns a boolean mask [H, W] where True indicates human pixels.
    Caches to .npy (uint8 0/1) keyed by the frame filename and SAM prompt.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{frame_img_path.stem}_{_cache_prompt_suffix(prompt)}_mask.npy"
    if cache_path.exists():
        m = np.load(cache_path)
        return (m > 0)

    # Load image as PIL
    image_pil = Image.open(str(frame_img_path)).convert("RGB")
    inference_state = processor.set_image(image_pil)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = output.get("masks", None)  # (num_instances, H, W)
    scores = output.get("scores", None)
    if masks is None or len(masks) == 0:
        # If no human detected, define empty mask
        empty = np.zeros((image_pil.height, image_pil.width), dtype=np.uint8)
        np.save(str(cache_path), empty)
        return empty.astype(bool)

    # scores can be torch or numpy
    try:
        import torch  # local import; script should still be importable without torch

        if torch.is_tensor(scores):
            scores = scores.detach().cpu().numpy()
    except Exception:
        pass

    try:
        import torch

        if torch.is_tensor(masks):
            masks = masks.detach().cpu().numpy()
    except Exception:
        pass

    masks_np = np.squeeze(np.asarray(masks))
    if " and " in prompt.lower():
        human_mask = masks_np.astype(bool) if masks_np.ndim == 2 else np.any(masks_np.astype(bool), axis=0)
    else:
        best_idx = int(np.argmax(scores))
        human_mask = np.squeeze(masks_np[best_idx]).astype(bool) if masks_np.ndim > 2 else masks_np.astype(bool)
    np.save(str(cache_path), human_mask.astype(np.uint8))
    return human_mask


def iou(mask_a: np.ndarray, mask_b: np.ndarray, eps: float = 1e-8) -> float:
    """
    IoU for boolean masks. Both empty means no person was detected, so the score is 0.
    """
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    union = np.logical_or(a, b).sum(dtype=np.float64)
    if union < eps:
        return 0.0
    return float(inter / union)


@dataclass
class PairScore:
    t: int
    t_prev: int
    iou: float


@dataclass
class CutResult:
    cut_frame: int
    window_w: int
    pair_scores: List[PairScore]
    mean_iou: Optional[float]


def compute_cut_consistency(
    frames_dir: Path,
    cut_frame: int,
    w: int,
    processor,
    cache_dir: Path,
    prompt: str = "human",
) -> CutResult:
    """
    Compute mean IoU over pairs (t, t-1) for t in [cut-w, ..., cut+w].
    """
    pair_scores: List[PairScore] = []

    for t in range(cut_frame - w, cut_frame + w + 1):
        t_prev = t - 1
        if t_prev < 1 or t < 1:
            continue

        img_t = _frame_path(frames_dir, t)
        img_prev = _frame_path(frames_dir, t_prev)

        m_t = _load_subject_mask(img_t, processor, cache_dir, prompt=prompt)
        m_prev = _load_subject_mask(img_prev, processor, cache_dir, prompt=prompt)
        score = iou(m_t, m_prev)
        pair_scores.append(PairScore(t=t, t_prev=t_prev, iou=score))

    if len(pair_scores) == 0:
        mean_iou = None
    else:
        mean_iou = float(np.mean([p.iou for p in pair_scores]))

    return CutResult(cut_frame=cut_frame, window_w=w, pair_scores=pair_scores, mean_iou=mean_iou)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scene_transitions_json",
        type=str,
        default="",
        help="Path to scene_transitions.json. The shell always passes this.",
    )
    ap.add_argument(
        "--frames_root",
        type=str,
        default="",
        help="Root directory containing per-video frame folders. Empty derives it from scene_transitions_json.",
    )
    add_ids_resume_args(ap)
    ap.add_argument(
        "--w",
        type=int,
        default=0,
        help="Sliding window radius around the cut (default 0).",
    )
    ap.add_argument(
        "--metadata_dir",
        type=str,
        default=str(_BENCH_ROOT / "configs" / "sceneshifter_multitalk"),
        help="Directory containing <id>_metadata.json. Used to infer the SAM3 subject prompt.",
    )
    ap.add_argument(
        "--sam3_ckpt",
        type=str,
        default=str(_BENCH_ROOT / "sam3" / "sam3.pt"),
        help="Path to SAM3 checkpoint file (sam3.pt).",
    )
    ap.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Where to cache per-frame human masks (.npy). Empty uses a directory next to scene_transitions_json.",
    )
    ap.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Output path for the results JSON. The shell always passes this.",
    )
    args = ap.parse_args()
    if not str(args.scene_transitions_json).strip():
        raise SystemExit("--scene_transitions_json is required")
    if not str(args.output_json).strip():
        raise SystemExit("--output_json is required")

    scene_json_path = Path(args.scene_transitions_json)
    data = json.loads(scene_json_path.read_text())
    results = data.get("results", [])

    frames_root = Path(args.frames_root) if str(args.frames_root).strip() else _default_frames_root(scene_json_path)
    cache_dir = Path(args.cache_dir) if str(args.cache_dir).strip() else _default_cache_dir(scene_json_path)
    metadata_dir = Path(args.metadata_dir).expanduser().resolve()

    # Setup SAM3
    _maybe_add_sam3_to_path()
    _, processor = load_sam3_model(args.sam3_ckpt)

    requested_ids = parse_ids(args.ids)

    available_ids: List[int] = []
    for entry in results:
        vid_s = _video_id_from_video_path(entry.get("video", ""))
        if vid_s:
            available_ids.append(int(vid_s))

    existing = load_json(Path(args.output_json)) if args.output_json else None
    existing_videos = existing.get("videos", []) if isinstance(existing, dict) else []
    done_ids = ids_from_list(existing_videos) if args.resume else set()
    todo_ids, skipped_ids = resolve_todo_ids(
        requested_ids=requested_ids,
        available_ids=available_ids,
        done_ids=done_ids,
        resume=bool(args.resume),
    )
    log_resume_plan(label="spatial", todo=todo_ids, skipped=skipped_ids, requested=requested_ids)

    results = [
        r
        for r in results
        if extract_video_id(r.get("video", "")) in todo_ids
    ]

    new_videos: List[Dict[str, Any]] = []
    for entry in tqdm(results, desc="Videos"):
        video_path = entry.get("video", "")
        ok = entry.get("ok", False)
        if not ok:
            continue

        vid = _video_id_from_video_path(video_path)
        if vid is None:
            continue
        frames_dir = frames_root / vid
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found for video_id={vid}: {frames_dir}")

        prompt = _infer_subject_prompt_from_metadata(metadata_dir, vid)

        last_idx = last_frame_index_from_frames_dir(frames_dir)
        raw_cuts = entry.get("cuts", []) or []
        kept = set(
            filter_last_frame_cuts(
                (c["frame"] for c in raw_cuts if isinstance(c, dict) and "frame" in c),
                last_idx,
            )
        )
        cuts = [c for c in raw_cuts if isinstance(c, dict) and int(c.get("frame", -1)) in kept]
        video_obj: Dict[str, Any] = {
            "video": video_path,
            "video_id": vid,
            "sam3_prompt": prompt,
            "sam3_prompt_parts": _subject_prompt_parts(prompt),
            "cuts": [],
        }

        vid_cache_dir = cache_dir / vid

        for cut in cuts:
            cut_frame = int(cut["frame"])
            cut_res = compute_cut_consistency(
                frames_dir=frames_dir,
                cut_frame=cut_frame,
                w=args.w,
                processor=processor,
                cache_dir=vid_cache_dir,
                prompt=prompt,
            )

            cut_obj: Dict[str, Any] = {
                "cut_frame": cut_res.cut_frame,
                "w": cut_res.window_w,
                "pair_scores": [
                    {"t": ps.t, "t_prev": ps.t_prev, "iou": ps.iou} for ps in cut_res.pair_scores
                ],
                "mean_iou": cut_res.mean_iou,
            }
            video_obj["cuts"].append(cut_obj)

        per_video_cut_scores = [c["mean_iou"] for c in video_obj["cuts"] if c["mean_iou"] is not None]
        video_obj["mean_iou_across_cuts"] = float(np.mean(per_video_cut_scores)) if per_video_cut_scores else None
        new_videos.append(video_obj)

    if args.resume and isinstance(existing, dict):
        merged_videos = merge_list_by_video_id(existing_videos, new_videos)
    else:
        merged_videos = new_videos

    all_cut_scores: List[float] = []
    for v in merged_videos:
        for c in v.get("cuts", []) or []:
            if c.get("mean_iou") is not None:
                all_cut_scores.append(float(c["mean_iou"]))

    out: Dict[str, Any] = {
        "scene_transitions_json": str(scene_json_path),
        "frames_root": str(frames_root),
        "w": args.w,
        "prompt_mode": "auto",
        "metadata_dir": str(metadata_dir),
        "videos": merged_videos,
        "overall": {
            "num_videos_evaluated": len(merged_videos),
            "num_cuts_evaluated": int(sum(len(v.get("cuts", []) or []) for v in merged_videos)),
            "mean_iou_across_cuts": float(np.mean(all_cut_scores)) if all_cut_scores else None,
        },
    }

    print(json.dumps(out["overall"], indent=2))

    out_path = Path(args.output_json)
    write_json(out_path, out)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

