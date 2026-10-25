#!/usr/bin/env python3
"""
Evaluate SceneShifter scenes AFTER scene0 using a VBench-2.0-style VLM+LLM text-alignment pipeline.

We score:
1) Background scene semantics alignment (per scene segment).
2) Subject-environment interaction plausibility (pose/support consistency with environment).

For each scene segment after scene0 (scene1, scene2, ..., sceneN) we score all frames in that segment.

Implementation is aligned to VBench2 dimension implementations (e.g., complex_landscape.py):
- Use LLaVA-Video-7B-Qwen2 to caption a video from 64 uniformly-sampled frames.
- Use Qwen2.5-7B-Instruct to (a) summarize into a numbered list and (b) judge yes/no containment.

Outputs: a JSON summary with per-video, per-scene, per-mode scores and debug artifacts.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from decord import VideoReader, cpu
from tqdm import tqdm
from transformers import AutoTokenizer
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

# Make sibling modules importable when running as a script (no package install required).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_MISC_DIR = _SCRIPT_DIR / "misc"
if _MISC_DIR.exists() and str(_MISC_DIR) not in sys.path:
    sys.path.insert(0, str(_MISC_DIR))

from background_gt_items import (  # type: ignore
    background_checklist_from_prompt,
    expected_pose_from_prompt,
    expected_support_object_from_prompt,
)

# Questionnaire builder (shared with the generator used for manual inspection).
# We intentionally reuse the exact question text + gating metadata.
from generate_background_questionnaire import (  # type: ignore
    build_combo_action_question,
    build_background_semantics_questionnaire,
    build_interaction_naturalness_questionnaire,
    infer_action_info_from_first_prompt,
    infer_subject_phrase_from_prompt,
)


# ---- VBench2 / LLaVA-NeXT imports (aligned with VBench2 dimension code) ----
def _require_code_repo(env_name: str, cli_value: str = "") -> Path:
    raw = str(cli_value or "").strip() or os.environ.get(env_name, "").strip()
    if not raw:
        raise SystemExit(f"{env_name} is unset. Export {env_name} to the code repository.")
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"{env_name} does not exist: {path}")
    return path


def _ensure_llava_importable(vbench2_root: Path) -> None:
    """
    VBench2 vendors LLaVA-NeXT under vbench2/third_party/LLaVA_NeXT.
    Some envs also have llava installed. To be robust, add the vendored path first.
    """
    llava_vendor = vbench2_root / "vbench2" / "third_party" / "LLaVA_NeXT"
    if llava_vendor.exists():
        sys.path.insert(0, str(llava_vendor))

def _ensure_siglip_available_for_llava(llava_dir: Path, cache_dir: Path) -> None:
    """
    LLaVA-Video-7B-Qwen2 config commonly references:
        "mm_vision_tower": "google/siglip-so400m-patch14-384"
    If that model is not in local cache, transformers will attempt to download from huggingface.co.
    We do NOT want that on a CN/offline server.

    Strategy:
    - If cache_dir/<mm_vision_tower>/ exists, rewrite llava config.json to point to that local directory.
    - Else raise with a clear instruction to pre-download SigLIP into cache_dir/google/siglip-so400m-patch14-384.
    """
    cfg_path = llava_dir / "config.json"
    if not cfg_path.exists():
        return
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return

    mm_vt = cfg.get("mm_vision_tower")
    if not isinstance(mm_vt, str) or not mm_vt:
        return
    # Already a local path
    if os.path.isabs(mm_vt) and Path(mm_vt).exists():
        return

    # If it's a hub id like "google/siglip-so400m-patch14-384", check local mirror dir.
    local_vt_dir = cache_dir / mm_vt
    if local_vt_dir.exists() and (local_vt_dir / "config.json").exists():
        cfg["mm_vision_tower"] = str(local_vt_dir)
        cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"[offline] Rewired LLaVA mm_vision_tower to local path: {local_vt_dir}")
        return

    raise FileNotFoundError(
        "Missing required vision tower model for LLaVA:\n"
        f"  mm_vision_tower = {mm_vt!r}\n"
        f"Expected local dir at:\n"
        f"  {local_vt_dir}\n\n"
        "This server cannot reach huggingface.co, so you must pre-download this model once (using a mirror).\n"
        "Recommended (hf-mirror):\n"
        f"  huggingface-cli download {mm_vt} --repo-type model --local-dir {local_vt_dir} --local-dir-use-symlinks False\n"
        "After that, re-run this script."
    )


# ---- generic utils ----
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _natural_sort_key(p: Path) -> Tuple:
    parts: List[Any] = []
    for chunk in re.split(r"(\d+)", p.name):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            parts.append(chunk.lower())
    return tuple(parts)


def _list_frame_files(frames_dir: Path) -> List[Path]:
    frames = [p for p in frames_dir.glob("*.jpg") if p.is_file()]
    return sorted(frames, key=_natural_sort_key)


def _frame_index_base(frames: List[Path]) -> int:
    if not frames:
        return 0
    vals: List[int] = []
    for p in frames:
        try:
            vals.append(int(p.stem))
        except Exception:
            continue
    return min(vals) if vals else 0


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


def _load_segment_prompts(metadata_path: Path) -> List[str]:
    try:
        data = _read_json(metadata_path)
    except Exception:
        return []
    routing = data.get("frame_prompt_routing", {})
    prompts = routing.get("segment_prompts", [])
    if not isinstance(prompts, list):
        return []
    return [p for p in prompts if isinstance(p, str)]


def _strip_accents(s: str) -> str:
    s_norm = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s_norm if not unicodedata.combining(ch))

    # NOTE: background checklist logic is implemented in `background_gt_items.py` for unit testing.


def _write_mp4_from_frames(frame_paths: List[Path], output_mp4: Path, *, fps: float) -> None:
    if not frame_paths:
        raise ValueError("No frames provided")
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
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
                continue
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(img)
    finally:
        writer.release()

def _ensure_symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


# ---- VBench2-aligned VLM video frame sampling ----
def vbench2_sample_video_frames(video_path: str, max_frames_num: int = 64) -> Tuple[np.ndarray, str, float]:
    """
    Directly adapted from VBench2 dimension code (complex_landscape.py / human_interaction.py).
    Returns: (frames_uint8, frame_time_str, video_time_seconds).
    """
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3), dtype=np.uint8), "0.00s", 0.0
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / float(vr.get_avg_fps())
    # force_sample=True in VBench2, so always uniform sample max_frames_num frames.
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frame_time = [i / float(vr.get_avg_fps()) for i in frame_idx]
    frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
    frames = vr.get_batch(frame_idx).asnumpy().astype(np.uint8)
    return frames, frame_time_str, float(video_time)


# ---- VLM yes/no helpers (aligned to VBench2 mechanics.py style) ----
def _parse_yes_no(text: str) -> Optional[bool]:
    """
    Parse a yes/no response from LLaVA.
    Returns:
      True  -> yes
      False -> no
      None  -> unclear
    """
    t = text.strip().lower()
    if re.match(r"^(yes|yes\.|yes,)", t):
        return True
    if re.match(r"^(no|no\.|no,)", t):
        return False
    # fallback: token presence
    if "yes" in t and "no" not in t:
        return True
    if "no" in t and "yes" not in t:
        return False
    return None


def _parse_yes_no_with_explanation(text: str) -> Tuple[Optional[bool], str]:
    """
    Parse yes/no while preserving any brief explanation after the leading token.
    Expected formats:
      - "Yes"
      - "No"
      - "No, ...reason..."
      - "No - ...reason..."
    Returns: (parsed_yes_no, explanation_str_or_empty)
    """
    raw = text.strip()
    t = raw.lower().strip()
    m = re.match(r"^(yes|no)\b[\s\.,:;\-]*([\s\S]*)$", t)
    if not m:
        return _parse_yes_no(raw), ""
    yn = m.group(1)
    rest = m.group(2).strip()
    return (True if yn == "yes" else False), rest

@dataclass
class CachedVlmVideo:
    """
    Cached, preprocessed video tensor and the VBench2-style time instruction metadata.
    This avoids repeated decode + preprocess across multiple yes/no questions for the same clip.
    """
    video: List[torch.Tensor]  # list with one element: (T, C, H, W) bfloat16/float16
    frame_time: str
    video_time: float


def _prepare_vlm_video_cached(
    *,
    video_path: str,
    max_frames_num: int,
    image_processor,
    device: torch.device,
    cache: Dict[Tuple[str, int], CachedVlmVideo],
) -> CachedVlmVideo:
    key = (video_path, int(max_frames_num))
    hit = cache.get(key)
    if hit is not None:
        return hit

    frames, frame_time, video_time = vbench2_sample_video_frames(video_path, max_frames_num=max_frames_num)
    pixel_values = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    # Match VBench2: send to device + bf16 on GPU. On CPU, bf16 may be unsupported, keep fp32.
    if device.type == "cuda":
        pixel_values = pixel_values.to(device).bfloat16()
    else:
        pixel_values = pixel_values.to(device).float()
    cached = CachedVlmVideo(video=[pixel_values], frame_time=frame_time, video_time=video_time)
    cache[key] = cached
    return cached


# ---- Data model ----
@dataclass
class SegmentSpec:
    name: str  # "scene1", "scene2", ...
    prompt: str
    start_frame: int  # in dataset index space (0-based)
    end_frame_exclusive: int  # in dataset index space (0-based)


@dataclass
class VideoSpec:
    video_id: int
    video_path: Path
    frames_dir: Path
    metadata_path: Path
    frame_splits: List[int]
    segments: List[SegmentSpec]
    first_prompt: str
    subject_phrase: str
    action_info: Dict[str, Any]


def _segments_after_scene0(
    *,
    cut_frames: Sequence[int],
    prompts: Sequence[str],
    last_idx: int,
) -> List[SegmentSpec]:
    """
    Build one clip per scene after the opening scene.

    scene k (k >= 1) covers [cut_frames[k-1], cut_frames[k]) and uses prompts[k].
    The scene that starts at the last detected cut runs through the last frame.
    """
    segments: List[SegmentSpec] = []
    n_cuts = len(cut_frames)
    for scene_idx in range(1, len(prompts)):
        start_cut_i = scene_idx - 1
        if start_cut_i >= n_cuts:
            break
        prompt = prompts[scene_idx]
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        start = int(cut_frames[start_cut_i])
        if scene_idx < n_cuts:
            end = int(cut_frames[scene_idx])
        elif scene_idx == n_cuts:
            end = int(last_idx) + 1
        else:
            break
        if end <= start:
            continue
        segments.append(
            SegmentSpec(
                name=f"scene{scene_idx}",
                prompt=prompt,
                start_frame=start,
                end_frame_exclusive=end,
            )
        )
    return segments


def _video_id_from_path(video_path: Path) -> Optional[int]:
    m = re.match(r"^(\d+)\.mp4$", video_path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _parse_pose_choice(text: str) -> Optional[str]:
    """
    Parse a pose choice from the VLM output.
    Expected one of: standing / sitting / squatting (dataset poses).
    """
    t = text.strip().lower()
    # Use first word as primary signal
    first = re.split(r"\s+", t, maxsplit=1)[0].strip(" .,:;")
    if first in {"standing", "sitting", "squatting"}:
        return first
    # Fallback: substring check (guard against verbose answers)
    for pose in ["standing", "sitting", "squatting"]:
        if pose in t:
            return pose
        return None


def _build_video_specs(
    *,
    scene_transitions_path: Path,
    frames_root: Path,
    metadata_dir: Path,
    only_ids: Optional[List[int]],
) -> List[VideoSpec]:
    data = _read_json(scene_transitions_path)
    results = data.get("results", [])
    out: List[VideoSpec] = []

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

        # Only evaluate videos with successful scene-change detection:
        # success definition (aligned with detect_scene_transitions.py):
        #   detected_num_cuts == len(metadata.frame_prompt_routing.frame_splits)
        if not bool(r.get("ok", False)):
            continue
        cuts = r.get("cuts", []) if isinstance(r, dict) else []
        cut_frames: List[int] = []
        if isinstance(cuts, list):
            for c in cuts:
                try:
                    cut_frames.append(int(c["frame"]))
                except Exception:
                    continue
        cut_frames = sorted([c for c in cut_frames if c >= 0])

        frames_dir = frames_root / f"{vid:03d}"
        if not frames_dir.exists():
            continue
        cut_frames = filter_last_frame_cuts(cut_frames, last_frame_index_from_frames_dir(frames_dir))
        frames = _list_frame_files(frames_dir)
        if not frames:
            continue
        base = _frame_index_base(frames)
        last_stem = max(int(p.stem) for p in frames)
        last_idx = last_stem - base

        metadata_path = metadata_dir / f"{vid:03d}_metadata.json"
        prompts = _load_segment_prompts(metadata_path)
        if len(prompts) < 2:
            continue
        first_prompt = prompts[0]
        subject_phrase = infer_subject_phrase_from_prompt(first_prompt)
        action_info = infer_action_info_from_first_prompt(first_prompt)
        frame_splits = _load_frame_splits(metadata_path)
        expected_cuts = len(frame_splits)
        detected_cuts = len(cut_frames)
        if expected_cuts <= 0:
            continue
        if detected_cuts != expected_cuts:
            continue
        if detected_cuts == 0:
            continue

        segments = _segments_after_scene0(
            cut_frames=cut_frames,
            prompts=prompts,
            last_idx=last_idx,
        )

        out.append(
            VideoSpec(
                video_id=vid,
                video_path=video_path,
                frames_dir=frames_dir,
                metadata_path=metadata_path,
                frame_splits=frame_splits,
                segments=segments,
                first_prompt=first_prompt,
                subject_phrase=subject_phrase,
                action_info=action_info,
            )
        )

    return out


def _select_frames_for_segment(
    *,
    frames: List[Path],
    base: int,
    seg: SegmentSpec,
) -> List[Path]:
    start_stem = seg.start_frame + base
    end_stem_excl = seg.end_frame_exclusive + base
    selected: List[Path] = []
    for p in frames:
        try:
            stem_i = int(p.stem)
        except Exception:
            continue
        if stem_i < start_stem or stem_i >= end_stem_excl:
            continue
        selected.append(p)
    return selected


def _build_segment_clips(
    *,
    spec: VideoSpec,
    clips_dir: Path,
    fps: float,
) -> Dict[str, Dict[str, str]]:
    """Return segment_name -> {"all_frames": clip_path}."""
    frames = _list_frame_files(spec.frames_dir)
    base = _frame_index_base(frames)

    out: Dict[str, Dict[str, str]] = {}
    for seg in spec.segments:
        clip_all = clips_dir / f"{spec.video_id:03d}_{seg.name}.mp4"
        if not clip_all.exists():
            selected = _select_frames_for_segment(frames=frames, base=base, seg=seg)
            if selected:
                _write_mp4_from_frames(selected, clip_all, fps=fps)
        seg_out: Dict[str, str] = {}
        if clip_all.exists():
            seg_out["all_frames"] = str(clip_all)
        out[seg.name] = seg_out
    return out


def _llava_generate(
    *,
    video_path: str,
    question_suffix: str,
    llava_model,
    llava_tokenizer,
    image_processor,
    device: torch.device,
    max_frames_num: int,
    max_new_tokens: int = 1024,
    video_cache: Optional[Dict[Tuple[str, int], CachedVlmVideo]] = None,
) -> str:
    # Import LLaVA utilities only when needed.
    from llava.mm_utils import tokenizer_image_token  # type: ignore
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # type: ignore
    from llava.conversation import conv_templates  # type: ignore

    # Cache decode+preprocess to avoid repeated work across multiple questions per clip.
    if video_cache is None:
        video_cache = {}
    cached = _prepare_vlm_video_cached(
        video_path=video_path,
        max_frames_num=max_frames_num,
        image_processor=image_processor,
        device=device,
        cache=video_cache,
    )
    video = cached.video
    frame_time = cached.frame_time
    video_time = cached.video_time

    conv_template = "qwen_1_5"
    time_instruction = (
        f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. "
        f"These frames are located at {frame_time}. {question_suffix}"
    )

    question = DEFAULT_IMAGE_TOKEN + f"{time_instruction}\n"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, llava_tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    cont = llava_model.generate(
        input_ids,
        images=video,
        modalities=["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=max_new_tokens,
    )
    return llava_tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()


def _score_background_alignment_vbench2_style(
    *,
    segment_prompt: str,
    clip_path: str,
    llava_model,
    llava_tokenizer,
    image_processor,
    device: torch.device,
    max_items: int = 5,
    max_frames_num: int = 64,
    video_cache: Optional[Dict[Tuple[str, int], CachedVlmVideo]] = None,
) -> Dict[str, Any]:
    """Presence-only background checks. Perceptual follow-ups are not asked."""
    gt_items = background_checklist_from_prompt(segment_prompt, max_items=max_items)
    q = build_background_semantics_questionnaire(
        prompt=segment_prompt,
        gt_items=gt_items,
        presence_only=True,
    )["background_semantics"]

    # ---- Part 1: presence ----
    presence_instruction = (
        "Please answer yes or no only for the following question. "
        "IMPORTANT: Focus only on the environment/background. Ignore the person/subject, faces, clothing, and actions."
    )
    presence_answers: List[Dict[str, Any]] = []
    yes_cnt = 0
    for item in q["part1_presence"]:
        gt_item = str(item["gt_item"])
        question = str(item["question"])
        resp = _llava_generate(
            video_path=clip_path,
            question_suffix=f"{presence_instruction}\n{question}",
            llava_model=llava_model,
            llava_tokenizer=llava_tokenizer,
            image_processor=image_processor,
            device=device,
            max_frames_num=max_frames_num,
            max_new_tokens=64,
            video_cache=video_cache,
        )
        parsed = _parse_yes_no(resp)
        ok = parsed is True
        if ok:
            yes_cnt += 1
        presence_answers.append({"gt_item": gt_item, "question": question, "answer": resp, "parsed": parsed, "ok": ok})

    presence_score = yes_cnt / max(1, len(q["part1_presence"]))

    return {
        "score": float(presence_score),
        "score_presence": float(presence_score),
        "score_perceptual": None,
        "gt_items": gt_items,
        "presence": {"answers": presence_answers},
        "perceptual": {
            "disabled": True,
            "reason": "presence_only_background_semantics",
            "items": [],
            "total_questions": 0,
        },
    }


def _score_subject_environment_interaction(
    *,
    segment_prompt: str,
    clip_path: str,
    llava_model,
    llava_tokenizer,
    image_processor,
    device: torch.device,
    subject_phrase: str = "the subject",
    max_frames_num: int = 64,
    video_cache: Optional[Dict[Tuple[str, int], CachedVlmVideo]] = None,
) -> Dict[str, Any]:
    """
    Subject-environment interaction questionnaire (sequential gating):
      - Q1 pose classification
      - If expected pose is sitting:
          Q2 Is the subject sitting on the expected support?
          Q3 Seating integration (only if Q2 is satisfactory)
      - If expected pose is standing/squatting:
          plausibility question for that pose

    If an earlier step is unsatisfactory, following steps are forfeited.
    """
    expected_pose = expected_pose_from_prompt(segment_prompt)
    expected_support = expected_support_object_from_prompt(segment_prompt)

    q = build_interaction_naturalness_questionnaire(
        prompt=segment_prompt,
        expected_pose=expected_pose or "",
        expected_support=expected_support,
        subject_phrase=subject_phrase,
    )["interaction_naturalness"]

    answers: List[Dict[str, Any]] = []
    step_scores: List[float] = []

    # ---- Step 1: pose classification ----
    pose_obj = q["questions"][0]
    pose_q = str(pose_obj["question"])
    pose_instruction = "Answer with exactly one word: standing, sitting, or squatting."
    pose_resp = _llava_generate(
            video_path=clip_path,
        question_suffix=f"{pose_instruction}\n{pose_q}",
            llava_model=llava_model,
            llava_tokenizer=llava_tokenizer,
            image_processor=image_processor,
            device=device,
            max_frames_num=max_frames_num,
        max_new_tokens=16,
            video_cache=video_cache,
        )
    predicted_pose = _parse_pose_choice(pose_resp)
    pose_ok = (predicted_pose is not None) and (expected_pose is not None) and (predicted_pose == expected_pose)
    answers.append(
        {
            "id": "pose_classification",
            "question": pose_q,
            "answer": pose_resp,
            "parsed_pose": predicted_pose,
            "expected_pose": expected_pose,
            "ok": pose_ok,
        }
    )
    step_scores.append(float(pose_ok))

    # If pose classification is unsatisfactory, forfeit all follow-ups.
    if not pose_ok:
        # Determine how many follow-up questions exist in the questionnaire for this pose.
        for obj in q["questions"][1:]:
            answers.append(
                {
                    "id": obj.get("id"),
                    "question": obj.get("question"),
                    "skipped": True,
                    "skip_reason": "pose_mismatch",
                    "answer": None,
                    "parsed": None,
                    "ok": False,
                    "explanation": "",
                }
            )
            step_scores.append(0.0)
        score = sum(step_scores) / max(1, len(step_scores))
        return {
            "score": float(score),
            "expected_pose": expected_pose,
            "expected_support": expected_support,
            "subject_phrase": subject_phrase,
            "predicted_pose": predicted_pose,
            "sequential_rule": q.get("sequential_rule"),
            "answers": answers,
        }

    # ---- Follow-ups (pose matched) ----
    yn_explain_instruction = "Answer yes or no. If no, briefly explain (<= 1 sentence)."

    # Sitting: Q2 then Q3 (gated by Q2)
    if expected_pose == "sitting":
        # Q2: sitting on expected support?
        q2_obj = q["questions"][1]
        q2 = str(q2_obj["question"])
        q2_resp = _llava_generate(
            video_path=clip_path,
            question_suffix=f"{yn_explain_instruction}\n{q2}",
            llava_model=llava_model,
            llava_tokenizer=llava_tokenizer,
            image_processor=image_processor,
            device=device,
            max_frames_num=max_frames_num,
            max_new_tokens=96,
            video_cache=video_cache,
        )
        q2_parsed, q2_expl = _parse_yes_no_with_explanation(q2_resp)
        q2_ok = q2_parsed is True
        answers.append(
            {
                "id": "sitting_on_expected_support",
                "question": q2,
                "answer": q2_resp,
                "parsed": q2_parsed,
                "ok": q2_ok,
                "explanation": q2_expl,
            }
        )
        step_scores.append(float(q2_ok))

        # Q3: seating integration (forfeit if Q2 is no/unclear)
        q3_obj = q["questions"][2] if len(q["questions"]) > 2 else None
        if q3_obj is not None:
            q3 = str(q3_obj["question"])
            if not q2_ok:
                answers.append(
                    {
                        "id": "sitting_integration",
                        "question": q3,
                        "skipped": True,
                        "skip_reason": "sitting_support_unsatisfactory",
                        "answer": None,
                        "parsed": None,
                        "ok": False,
                        "explanation": "",
                    }
                )
                step_scores.append(0.0)
            else:
                q3_resp = _llava_generate(
                    video_path=clip_path,
                    question_suffix=f"{yn_explain_instruction}\n{q3}",
                    llava_model=llava_model,
                    llava_tokenizer=llava_tokenizer,
                    image_processor=image_processor,
                    device=device,
                    max_frames_num=max_frames_num,
                    max_new_tokens=128,
                    video_cache=video_cache,
                )
                q3_parsed, q3_expl = _parse_yes_no_with_explanation(q3_resp)
                q3_ok = q3_parsed is True
                answers.append(
                    {
                        "id": "sitting_integration",
                        "question": q3,
                        "answer": q3_resp,
                        "parsed": q3_parsed,
                        "ok": q3_ok,
                        "explanation": q3_expl,
                    }
                )
                step_scores.append(float(q3_ok))

    else:
        # standing/squatting: single plausibility question after pose classification
        if len(q["questions"]) > 1:
            qn_obj = q["questions"][1]
            qn = str(qn_obj["question"])
            qn_resp = _llava_generate(
                video_path=clip_path,
                question_suffix=f"{yn_explain_instruction}\n{qn}",
                llava_model=llava_model,
                llava_tokenizer=llava_tokenizer,
                image_processor=image_processor,
                device=device,
                max_frames_num=max_frames_num,
                max_new_tokens=128,
                video_cache=video_cache,
            )
            qn_parsed, qn_expl = _parse_yes_no_with_explanation(qn_resp)
            qn_ok = qn_parsed is True
            answers.append(
                {
                    "id": qn_obj.get("id"),
                    "question": qn,
                    "answer": qn_resp,
                    "parsed": qn_parsed,
                    "ok": qn_ok,
                    "explanation": qn_expl,
                }
            )
            step_scores.append(float(qn_ok))

    score = sum(step_scores) / max(1, len(step_scores))
    return {
        "score": float(score),
        "expected_pose": expected_pose,
        "expected_support": expected_support,
        "subject_phrase": subject_phrase,
        "predicted_pose": predicted_pose,
        "sequential_rule": q.get("sequential_rule"),
        "answers": answers,
    }


def _score_combo_action(
    *,
    subject_phrase: str,
    combo_action_text: str,
    video_path: str,
    llava_model,
    llava_tokenizer,
    image_processor,
    device: torch.device,
    max_frames_num: int = 64,
) -> Dict[str, Any]:
    question_obj = build_combo_action_question(subject_phrase=subject_phrase, combo_action_text=combo_action_text)
    instruction = "Answer yes or no. If no, briefly explain (<= 1 sentence)."
    cache: Dict[Tuple[str, int], CachedVlmVideo] = {}
    resp = _llava_generate(
        video_path=video_path,
        question_suffix=f"{instruction}\n{question_obj['question']}",
        llava_model=llava_model,
        llava_tokenizer=llava_tokenizer,
        image_processor=image_processor,
        device=device,
        max_frames_num=max_frames_num,
        max_new_tokens=96,
        video_cache=cache,
    )
    parsed, expl = _parse_yes_no_with_explanation(resp)
    ok = parsed is True
    return {
        "score": float(ok),
        "question": question_obj,
        "answer": resp,
        "parsed": parsed,
        "ok": ok,
        "explanation": expl,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="VBench2-style evaluation for SceneShifter background semantics + interaction on every scene after scene0.")
    parser.add_argument("--scene-transitions-json", type=str, default="")
    parser.add_argument("--frames-root", type=str, default="")
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default=str(_BENCH_ROOT / "configs" / "sceneshifter_multitalk"),
    )
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument(
        "--sub-clip-dir",
        type=str,
        default="",
        help="Shared clip directory. Empty defaults to <output-dir>/../sub_clip.",
    )
    parser.add_argument("--fps", type=float, default=25.0)
    add_ids_resume_args(parser)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--vbench2-repo",
        type=str,
        default="",
        help="VBench-2.0 code repository. Defaults to the VBENCH2_REPO environment variable.",
    )
    parser.add_argument("--cache-dir", type=str, default=str(_BENCH_ROOT / "vbench2"), help="Local cache root containing models.")
    parser.add_argument(
        "--llava-dir",
        type=str,
        default=str(_BENCH_ROOT / "vbench2" / "lmms-lab" / "LLaVA-Video-7B-Qwen2"),
    )
    parser.add_argument("--max-frames-num", type=int, default=64, help="Number of uniformly sampled frames for LLaVA.")
    parser.add_argument("--max-items", type=int, default=5, help="Max background checklist items per segment.")

    args = parser.parse_args()
    if not str(args.scene_transitions_json).strip() or not str(args.frames_root).strip() or not str(args.output_dir).strip():
        raise SystemExit("--scene-transitions-json, --frames-root, and --output-dir are required")

    # Offline guard: avoid accidental network calls.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    # Point HF cache to your shared cache root so any lookups stay local.
    os.environ.setdefault("HF_HOME", str(Path(args.cache_dir).expanduser().resolve()))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(args.cache_dir).expanduser().resolve()))

    vbench2_root = _require_code_repo("VBENCH2_REPO", args.vbench2_repo)
    _ensure_llava_importable(vbench2_root)
    # Also add vbench2 root for relative imports if needed.
    sys.path.insert(0, str(vbench2_root))

    scene_transitions_path = Path(args.scene_transitions_json).expanduser().resolve()
    frames_root = Path(args.frames_root).expanduser().resolve()
    metadata_dir = Path(args.metadata_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    out_path = output_dir / "background_semantics_vbench2_results.json"
    sub_clip_dir = (
        Path(args.sub_clip_dir).expanduser().resolve()
        if str(args.sub_clip_dir).strip()
        else (output_dir.parent / "sub_clip")
    )
    clips_dir = sub_clip_dir
    # Backward-compat / reuse: if legacy clips exist under output_dir/clips, mirror them into sub_clip_dir.
    legacy_clips_dir = output_dir / "clips"
    if legacy_clips_dir.exists() and legacy_clips_dir.is_dir() and (legacy_clips_dir.resolve() != sub_clip_dir.resolve()):
        for p in legacy_clips_dir.glob("*.mp4"):
            if not p.is_file():
                continue
            _ensure_symlink_or_copy(p, sub_clip_dir / p.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_ids = parse_ids(args.ids)
    existing_payload = load_json(out_path) if args.resume else None
    existing_videos = (
        existing_payload.get("videos", {})
        if isinstance(existing_payload, dict) and isinstance(existing_payload.get("videos"), dict)
        else {}
    )
    done_ids = ids_from_dict_keys(existing_videos) if args.resume else set()

    prefilter = sorted(requested_ids) if requested_ids is not None else None
    specs_all = _build_video_specs(
        scene_transitions_path=scene_transitions_path,
        frames_root=frames_root,
        metadata_dir=metadata_dir,
        only_ids=prefilter,
    )
    available_ids = [int(s.video_id) for s in specs_all]
    todo_ids, skipped_ids = resolve_todo_ids(
        requested_ids=requested_ids,
        available_ids=available_ids,
        done_ids=done_ids,
        resume=bool(args.resume),
    )
    log_resume_plan(label="bg_semantics", todo=todo_ids, skipped=skipped_ids, requested=requested_ids)
    specs = [s for s in specs_all if int(s.video_id) in todo_ids]
    print(f"Found {len(specs)} videos with scenes after scene0 to evaluate.")

    if not specs:
        if args.resume and existing_payload is not None:
            print("[bg_semantics] nothing to evaluate; keeping existing results")
            out_path.write_text(json.dumps(existing_payload, indent=2), encoding="utf-8")
            return
        print("[bg_semantics] no videos to evaluate")
        return

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    # Load models (aligned to VBench2 implementations)
    from llava.model.builder import load_pretrained_model  # type: ignore

    model_name = "llava_qwen"
    device_map = "auto"
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    llava_dir_path = Path(args.llava_dir).expanduser().resolve()
    _ensure_siglip_available_for_llava(llava_dir_path, cache_dir)
    llava_dir = str(llava_dir_path)
    llava_tokenizer, llava_model, image_processor, _max_length = load_pretrained_model(
        llava_dir, None, model_name, torch_dtype="bfloat16", device_map=device_map
    )
    llava_model.eval()

    # No separate LLM judge: we use VLM yes/no QA per item (VBench2 mechanics-style).

    results: Dict[str, Any] = {
        "meta": {
            "scene_transitions_json": str(scene_transitions_path),
            "frames_root": str(frames_root),
            "metadata_dir": str(metadata_dir),
            "output_dir": str(output_dir),
            "clips_dir": str(clips_dir),
            "sub_clip_dir": str(sub_clip_dir),
            "max_frames_num": int(args.max_frames_num),
            "llava_dir": llava_dir,
            "qa_mode": "vlm_yesno_per_item",
            "background_semantics_mode": "presence_only",
        },
        "videos": {},
    }
    if args.resume and isinstance(existing_payload, dict):
        results = existing_payload
        results.setdefault("meta", {})
        results.setdefault("videos", {})
        if isinstance(results["meta"], dict):
            results["meta"].update(
                {
                    "scene_transitions_json": str(scene_transitions_path),
                    "frames_root": str(frames_root),
                    "metadata_dir": str(metadata_dir),
                    "output_dir": str(output_dir),
                    "clips_dir": str(clips_dir),
                    "sub_clip_dir": str(sub_clip_dir),
                    "max_frames_num": int(args.max_frames_num),
                    "llava_dir": llava_dir,
                    "qa_mode": "vlm_yesno_per_item",
                    "background_semantics_mode": "presence_only",
                }
            )

    for spec in tqdm(specs, desc="videos"):
        clip_map = _build_segment_clips(
            spec=spec,
            clips_dir=clips_dir,
            fps=float(args.fps),
        )

        # Build per-scene evaluation
        vid_key = f"{spec.video_id:03d}"
        video_out: Dict[str, Any] = {
            "video_id": spec.video_id,
            "frame_splits": spec.frame_splits,
            "subject_phrase": spec.subject_phrase,
            "action_mode": spec.action_info.get("action_mode"),
            "actions": spec.action_info.get("actions", []),
            "combo_action_text": spec.action_info.get("combo_action_text", ""),
            "whole_video": {},
            "segments": {},
        }

        is_multi_action = spec.action_info.get("action_mode") == "multi_action"
        last_segment_name = spec.segments[-1].name if spec.segments else None
        if is_multi_action and spec.video_path.exists():
            video_out["whole_video"]["combo_action"] = _score_combo_action(
                subject_phrase=spec.subject_phrase,
                combo_action_text=str(spec.action_info.get("combo_action_text", "")),
                video_path=str(spec.video_path),
                llava_model=llava_model,
                llava_tokenizer=llava_tokenizer,
                image_processor=image_processor,
                device=device,
                max_frames_num=int(args.max_frames_num),
            )

        for seg in spec.segments:
            seg_out: Dict[str, Any] = {
                "prompt": seg.prompt,
                "start_frame": seg.start_frame,
                "end_frame_exclusive": seg.end_frame_exclusive,
                "modes": {},
            }

            clip_path = clip_map.get(seg.name, {}).get("all_frames")
            if not clip_path:
                video_out["segments"][seg.name] = seg_out
                continue
            clip_cache: Dict[Tuple[str, int], CachedVlmVideo] = {}
            bg_score = _score_background_alignment_vbench2_style(
                segment_prompt=seg.prompt,
                clip_path=clip_path,
                llava_model=llava_model,
                llava_tokenizer=llava_tokenizer,
                image_processor=image_processor,
                device=device,
                max_items=int(args.max_items),
                max_frames_num=int(args.max_frames_num),
                video_cache=clip_cache,
            )
            bg_payload = {
                "score": float(bg_score["score"]),
                "score_presence": float(bg_score["score_presence"]),
                "score_perceptual": None,
                "gt_items": bg_score["gt_items"],
                "presence": bg_score["presence"],
                "perceptual": bg_score["perceptual"],
            }

            if is_multi_action and seg.name != last_segment_name:
                int_score = {
                    "score": None,
                    "skipped": True,
                    "skip_reason": "multi_action_interaction_scored_on_last_segment",
                    "expected_pose": None,
                    "expected_support": None,
                    "subject_phrase": spec.subject_phrase,
                    "answers": [],
                }
            else:
                int_score = _score_subject_environment_interaction(
                    segment_prompt=seg.prompt,
                    clip_path=clip_path,
                    llava_model=llava_model,
                    llava_tokenizer=llava_tokenizer,
                    image_processor=image_processor,
                    device=device,
                    subject_phrase=spec.subject_phrase,
                    max_frames_num=int(args.max_frames_num),
                    video_cache=clip_cache,
                )

            seg_out["modes"]["all_frames"] = {
                "clip_path": clip_path,
                "background_semantics": bg_payload,
                "interaction_naturalness": {
                    "score": None if int_score.get("score") is None else float(int_score["score"]),
                    "skipped": bool(int_score.get("skipped", False)),
                    "skip_reason": int_score.get("skip_reason"),
                    "expected_pose": int_score["expected_pose"],
                    "expected_support": int_score["expected_support"],
                    "subject_phrase": int_score.get("subject_phrase", spec.subject_phrase),
                    "answers": int_score.get("answers", []),
                },
            }

            video_out["segments"][seg.name] = seg_out

        # Unweighted mean across every evaluated scene after scene0.
        def _avg_over_scenes(metric_key: str, mode_name: str) -> Optional[float]:
            vals: List[float] = []
            for segname, segv in video_out["segments"].items():
                modes = segv.get("modes", {})
                if mode_name not in modes:
                    continue
                metric_obj = modes[mode_name].get(metric_key, {})
                score = metric_obj.get("score") if isinstance(metric_obj, dict) else None
                if score is None:
                    continue
                vals.append(float(score))
            return (sum(vals) / len(vals)) if vals else None

        def _avg_over_scenes_sub(metric_key: str, subkey: str, mode_name: str) -> Optional[float]:
            vals: List[float] = []
            for segname, segv in video_out["segments"].items():
                modes = segv.get("modes", {})
                if mode_name not in modes:
                    continue
                mk = modes[mode_name].get(metric_key, {})
                if not isinstance(mk, dict):
                    continue
                if mk.get(subkey) is None:
                    continue
                vals.append(float(mk[subkey]))
            return (sum(vals) / len(vals)) if vals else None

        video_out["averages"] = {
            "all_frames": {
                "background_semantics": _avg_over_scenes("background_semantics", "all_frames"),
                "background_semantics_presence": _avg_over_scenes_sub("background_semantics", "score_presence", "all_frames"),
                "background_semantics_perceptual": None,
                "interaction_naturalness": _avg_over_scenes("interaction_naturalness", "all_frames"),
            },
        }

        results["videos"][vid_key] = video_out
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.resume and existing_videos:
        results["videos"] = merge_dict_by_video_key(existing_videos, results.get("videos", {}))

    # Dataset-level summary
    def _collect(metric: str, mode: str) -> List[float]:
        vals: List[float] = []
        for v in results["videos"].values():
            avg = v.get("averages", {}).get(mode, {})
            if avg and avg.get(metric) is not None:
                vals.append(float(avg[metric]))
        return vals

    def _collect_combo_action() -> List[float]:
        vals: List[float] = []
        for v in results["videos"].values():
            combo = v.get("whole_video", {}).get("combo_action", {})
            if isinstance(combo, dict) and combo.get("score") is not None:
                vals.append(float(combo["score"]))
        return vals

    summary = {}
    combo_vals = _collect_combo_action()
    summary_modes = sorted(
        {
            mode
            for v in results["videos"].values()
            if isinstance(v, dict) and isinstance(v.get("averages"), dict)
            for mode in v["averages"].keys()
        }
        or {"all_frames"}
    )
    for mode in summary_modes:
        bg_p = _collect("background_semantics_presence", mode)
        bg_perc = _collect("background_semantics_perceptual", mode)
        inter = _collect("interaction_naturalness", mode)
        summary[mode] = {
            "background_semantics_presence_mean": (sum(bg_p) / len(bg_p)) if bg_p else None,
            "background_semantics_perceptual_mean": (sum(bg_perc) / len(bg_perc)) if bg_perc else None,
            "interaction_naturalness_mean": (sum(inter) / len(inter)) if inter else None,
            "combo_action_mean": (sum(combo_vals) / len(combo_vals)) if combo_vals else None,
            "count": len(results["videos"]),
        }
    results["summary"] = summary

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

