#!/usr/bin/env python3
"""
Evaluate VBench-I2V metrics on SceneShifter *scene 0* only (frames before the first cut).

This script:
1) Reads cut frames from:
   results/.../videos/scene_transitions.json
   For each video, we take the FIRST detected cut frame as the start of scene 1.
2) Builds a scene-0 clip from extracted frames:
   results/.../frames/<id>/*.jpg -> write mp4 with only frames < cut_frame
3) Builds a reference image for I2V evaluation from scene-0 (default: first frame in scene-0 clip):
   images/<id>.jpg
4) Runs VBench-I2V in custom_input mode on i2v_subject.
   Scene0 and later scenes both use the opening-scene image as the reference.

Additionally, for videos with 3+ scenes, we also compute a "scene1plus" score by
evaluating i2v_subject on a *single concatenated clip* containing all frames after scene0
(scene1, scene2, ...). This achieves frame-level weighting over scene1+ without modifying
VBench's metric implementation: we simply change the input clip.

Extended (SceneShifter-specific):
- Optionally evaluate VBench-2.0 "human fidelity" dimensions using SceneShifter's detected
  scene boundaries from scene_transitions.json:
  - Human_Anatomy: compute overall score and also per-split aggregates for scene0 vs scene1+.
    We avoid duplicate inference by computing per-frame counts once and re-aggregating.
  - Human_Identity: compute overall score and also per-split aggregates for scene0 vs scene1+.
    We avoid duplicate inference by computing per-frame consistency flags once and re-aggregating.
  - Human_Clothes: same as VBench2 but with Q1 adapted to "speaking".

Notes:
- VBench-I2V custom_input mode requires that each video file and its paired image share the same stem:
  e.g. clips/001.mp4 <-> images/001.jpg.
- Pretrained weights:
  - i2v_subject: DINO ViT-B/16 (downloaded into VBENCH_CACHE_DIR when --local is used and weights are missing).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch

_SCRIPTS_DIR = Path(__file__).resolve().parent
_BENCH_ROOT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from eval_resume_utils import (  # noqa: E402
    add_ids_resume_args,
    filter_last_frame_cuts,
    ids_from_list,
    last_frame_index_from_frames_dir,
    load_json,
    log_resume_plan,
    merge_list_by_video_id,
    parse_ids,
    resolve_todo_ids,
)

########################################################################################
# Optional VBench2 Human Fidelity evaluation helpers
########################################################################################


def _require_code_repo(env_name: str, cli_value: str = "") -> Path:
    raw = str(cli_value or "").strip() or os.environ.get(env_name, "").strip()
    if not raw:
        raise SystemExit(f"{env_name} is unset. Export {env_name} to the code repository.")
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"{env_name} does not exist: {path}")
    return path


def _maybe_add_vbench2_to_syspath(vbench_repo: Path) -> None:
    """
    Allow importing `vbench2` without installing as a site-package.
    `vbench_repo` should be the VBench-2.0 repository root.
    """
    if vbench_repo.exists() and str(vbench_repo) not in sys.path:
        sys.path.insert(0, str(vbench_repo))

    # YOLO-World configs use `custom_imports=dict(imports=['yolo_world'], ...)`.
    # Ensure `yolo_world` is importable by adding the third_party/YOLO-World root.
    yolo_world_root = (vbench_repo / "vbench2" / "third_party" / "YOLO-World").resolve()
    if yolo_world_root.exists() and str(yolo_world_root) not in sys.path:
        sys.path.insert(0, str(yolo_world_root))


def _split_mask_from_cut(
    frame_indices_0based: List[int],
    cut_frame_0based: Optional[int],
) -> Tuple[List[bool], List[bool]]:
    """
    Given a list of frame indices in 0-based "video frame index" space, build boolean masks
    for scene0 (< cut) and scene1plus (>= cut). If cut is None, scene1plus is empty.
    """
    if cut_frame_0based is None:
        return [True] * len(frame_indices_0based), [False] * len(frame_indices_0based)
    scene0 = [fi < cut_frame_0based for fi in frame_indices_0based]
    scene1p = [fi >= cut_frame_0based for fi in frame_indices_0based]
    return scene0, scene1p


def _sum_where(values: List[int], mask: List[bool]) -> int:
    s = 0
    for v, m in zip(values, mask):
        if m:
            s += int(v)
    return int(s)


def _count_where(values: List[bool], mask: List[bool]) -> int:
    c = 0
    for v, m in zip(values, mask):
        if m and bool(v):
            c += 1
    return int(c)


def _safe_div(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return float(num) / float(den)


def _frames_for_video_id(frames_root: Path, video_id: int) -> List[Path]:
    frames_dir = frames_root / f"{video_id:03d}"
    frames = [p for p in frames_dir.glob("*.jpg") if p.is_file()]
    return sorted(frames, key=_natural_sort_key)


def _metadata_path_for_video(metadata_dir: Path, video_id: int) -> Path:
    return metadata_dir / f"{int(video_id):03d}_metadata.json"


def _resolve_metadata_asset_path(raw_path: Any, *, metadata_path: Path) -> Optional[Path]:
    """Resolve metadata assets such as cond_image, commonly stored as v12_final/xxx.jpg."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    p = Path(raw_path).expanduser()
    if p.is_absolute():
        return p if p.exists() else None

    # Metadata dirs may be either <dataset>/metadata_jsons_v2_simplified... or <dataset>/ablations/<name>.
    # Try nearby dataset roots before giving up.
    candidates = [
        metadata_path.parent / p,
        metadata_path.parent.parent / p,
        metadata_path.parent.parent.parent / p,
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    return None


def _infer_subject_phrase_from_metadata(data: Any) -> str:
    if not isinstance(data, dict):
        return "human"
    routing = data.get("frame_prompt_routing", {})
    seg_prompts = routing.get("segment_prompts", []) if isinstance(routing, dict) else []
    candidates: List[str] = []
    if isinstance(seg_prompts, list):
        candidates.extend(str(p) for p in seg_prompts)
    prompt = data.get("prompt", "")
    if isinstance(prompt, str):
        candidates.append(prompt)
    text = " ".join(candidates).lower()
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


def _infer_subject_metadata(metadata_dir: Optional[Path], video_id: int) -> Dict[str, Any]:
    data: Any = {}
    meta_path: Optional[Path] = None
    if metadata_dir is not None:
        candidate = _metadata_path_for_video(metadata_dir, video_id)
        if candidate.exists():
            meta_path = candidate
            try:
                data = _read_json(candidate)
            except Exception:
                data = {}

    subject_phrase = _infer_subject_phrase_from_metadata(data)
    cond_audio = data.get("cond_audio", {}) if isinstance(data, dict) else {}
    routing = data.get("frame_prompt_routing", {}) if isinstance(data, dict) else {}
    bg_mask = routing.get("background_mask_path", {}) if isinstance(routing, dict) else {}
    audio_type = data.get("audio_type", "") if isinstance(data, dict) else ""

    audio_subjects = len(cond_audio) if isinstance(cond_audio, dict) else 0
    mask_subjects = len(bg_mask) if isinstance(bg_mask, dict) else 0
    phrase_subjects = 2 if subject_phrase in {"the man and the woman", "the woman and the man"} else 1
    expected_subject_count = max(1, audio_subjects, mask_subjects, phrase_subjects)
    if isinstance(audio_type, str) and "add" in audio_type.lower():
        expected_subject_count = max(expected_subject_count, 2)
    expected_subject_count = 2 if expected_subject_count >= 2 else 1

    cond_image_path = None
    if isinstance(data, dict) and meta_path is not None:
        cond_image_path = _resolve_metadata_asset_path(data.get("cond_image"), metadata_path=meta_path)

    return {
        "metadata_path": str(meta_path) if meta_path is not None else None,
        "cond_image_path": str(cond_image_path) if cond_image_path is not None else None,
        "subject_mode": "multi" if expected_subject_count >= 2 else "single",
        "expected_subject_count": int(expected_subject_count),
        "subject_phrase": subject_phrase,
    }


def _subject_metadata_by_video_id(metadata_dir: Optional[Path], jobs: List["VideoJob"]) -> Dict[int, Dict[str, Any]]:
    return {int(job.video_id): _infer_subject_metadata(metadata_dir, int(job.video_id)) for job in jobs}


def _anatomy_eval_one_video_no_duplicate(
    *,
    video_id: int,
    frame_paths: List[Path],
    cut_frame_0based: Optional[int],
    device: str,
    submodules_dict: Dict[str, Any],
    detector: Optional[Any] = None,
    analyzer: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run VBench2 Human_Anatomy-style abnormality detection on frames once, then aggregate:
      - overall (VBench2-style): 1 - total_abnormal_people / total_people
      - plus per-split totals and derived scores for scene0 and scene1+

    We interpret "successful frame" as: frame has >=1 detected person AND abnormal_count==0.
    Frames with no detected people are excluded from the "successful frames" denominator.
    """
    # Lazy imports to keep this script runnable when VBench2 deps aren't installed.
    # IMPORTANT: creating YOLO-World / ViTDetector components repeatedly can crash under some
    # torch/mmengine stacks (e.g. optimizer registry duplication). So we allow callers to
    # construct these once and reuse.
    if detector is None or analyzer is None:
        from vbench2.third_party.ViTDetector.detect import Detector, Analyzer  # type: ignore

        detector = Detector(
            config_file=submodules_dict["detector_config"],
            weight_file=submodules_dict["detector_weights"],
            device=device,
        )
        analyzer = Analyzer(
            model_configs=submodules_dict["analyzer_configs"],
            device=device,
            batch_size=submodules_dict["batch_size"],
            class_thresholds={k: v["threshold"] for k, v in submodules_dict["analyzer_configs"].items()},
        )

    # We avoid video decoding by using extracted frames. We compute detections per frame,
    # then run analyzer.process_frame per frame.
    frame_indices_0based: List[int] = []
    person_counts: List[int] = []
    abnormal_counts: List[int] = []
    success_frames: List[bool] = []

    base = _frame_index_base(frame_paths)
    scene0_mask: List[bool] = []
    scene1p_mask: List[bool] = []

    # Determine split masks in "video frame index" space.
    for fp in frame_paths:
        try:
            stem_i = int(fp.stem)
        except Exception:
            continue
        frame_indices_0based.append(int(stem_i - base))
    scene0_mask, scene1p_mask = _split_mask_from_cut(frame_indices_0based, cut_frame_0based)

    # Build a frame->detections list (matching ViTDetector schema) in a single pass.
    all_detections: Dict[int, List[dict]] = {i: [] for i in range(len(frame_indices_0based))}
    # One-time prompt reparameterization per video evaluation call.
    human_text = "human"
    human_texts = [[t.strip()] for t in human_text.split(",")] + [[" "]]
    face_hand_text = "face,hand"
    face_hand_texts = [[t.strip()] for t in face_hand_text.split(",")] + [[" "]]
    detector.model_human.reparameterize(human_texts)
    detector.model_face_hand.reparameterize(face_hand_texts)

    for local_idx, fp in enumerate(frame_paths):
        bgr = cv2.imread(str(fp))
        if bgr is None:
            continue
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Human/face/hand detections (same as Detector.detect_video but over our frames list).
        result_human = detector.inference_detector(detector.model_human, frame, human_texts, detector.test_pipeline, score_thr=0.1)
        pred_instances = result_human.pred_instances
        human_bboxes = pred_instances.bboxes.cpu().numpy()
        for person_idx, bbox in enumerate(human_bboxes):
            all_detections[local_idx].append(
                {"frame_index": local_idx, "person_index": person_idx, "bbox": bbox.tolist(), "label": "human"}
            )
            x1, y1, x2, y2 = bbox.astype(int)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            result_face_hand = detector.inference_detector(
                detector.model_face_hand, crop, face_hand_texts, detector.test_pipeline, score_thr=0.1
            )
            pred_face_hand = result_face_hand.pred_instances
            fh_bboxes = pred_face_hand.bboxes.cpu().numpy()
            fh_labels = pred_face_hand.labels.cpu().numpy()
            for fh_bbox, label_idx in zip(fh_bboxes, fh_labels):
                adj_bbox = [
                    float(fh_bbox[0] + x1),
                    float(fh_bbox[1] + y1),
                    float(fh_bbox[2] + x1),
                    float(fh_bbox[3] + y1),
                ]
                label = "face" if int(label_idx) == 0 else ("hand" if int(label_idx) == 1 else "unknown")
                all_detections[local_idx].append(
                    {"frame_index": local_idx, "person_index": person_idx, "bbox": adj_bbox, "label": label}
                )

    # Now compute per-frame abnormality (same logic as Analyzer.analyze but frame-by-frame).
    for local_idx, fp in enumerate(frame_paths):
        bgr = cv2.imread(str(fp))
        if bgr is None:
            continue
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        analyzer.frame_cache = {local_idx: frame}
        frame_result = analyzer.process_frame(local_idx, all_detections.get(local_idx, []))

        pc = int(frame_result.get("person_count", 0))
        ac = int(frame_result.get("abnormal_count", 0))
        person_counts.append(pc)
        abnormal_counts.append(ac)
        success_frames.append(bool(pc > 0 and ac == 0))

    # Aggregate VBench2-style score: 1 - total_abnormal / total_people
    total_people = int(sum(person_counts))
    total_abnormal = int(sum(abnormal_counts))
    score_overall = (1.0 - (total_abnormal / total_people)) if total_people > 0 else None

    # Split aggregates
    total_people_s0 = _sum_where(person_counts, scene0_mask)
    total_abnormal_s0 = _sum_where(abnormal_counts, scene0_mask)
    total_people_s1p = _sum_where(person_counts, scene1p_mask)
    total_abnormal_s1p = _sum_where(abnormal_counts, scene1p_mask)

    # Successful frames counts (frames where at least one person detected)
    frames_with_people = [pc > 0 for pc in person_counts]
    frames_with_people_s0 = _count_where(frames_with_people, scene0_mask)
    frames_with_people_s1p = _count_where(frames_with_people, scene1p_mask)

    success_frames_overall = sum(1 for s in success_frames if s)
    success_frames_s0 = _count_where(success_frames, scene0_mask)
    success_frames_s1p = _count_where(success_frames, scene1p_mask)

    score_s0 = (1.0 - (total_abnormal_s0 / total_people_s0)) if total_people_s0 > 0 else None
    score_s1p = (1.0 - (total_abnormal_s1p / total_people_s1p)) if total_people_s1p > 0 else None

    return {
        "video_id": int(video_id),
        "score_overall_vbench2_style": score_overall,
        "score_scene0_vbench2_style": score_s0,
        "score_scene1plus_vbench2_style": score_s1p,
        "total_people_overall": total_people,
        "total_abnormal_overall": total_abnormal,
        "total_people_scene0": total_people_s0,
        "total_abnormal_scene0": total_abnormal_s0,
        "total_people_scene1plus": total_people_s1p,
        "total_abnormal_scene1plus": total_abnormal_s1p,
        "frames_with_people_overall": int(sum(1 for x in frames_with_people if x)),
        "frames_with_people_scene0": frames_with_people_s0,
        "frames_with_people_scene1plus": frames_with_people_s1p,
        "successful_frames_overall": int(success_frames_overall),
        "successful_frames_scene0": int(success_frames_s0),
        "successful_frames_scene1plus": int(success_frames_s1p),
        "success_rate_frames_overall": _safe_div(success_frames_overall, sum(1 for x in frames_with_people if x)),
        "success_rate_frames_scene0": _safe_div(success_frames_s0, frames_with_people_s0),
        "success_rate_frames_scene1plus": _safe_div(success_frames_s1p, frames_with_people_s1p),
    }


def _init_identity_models(
    *,
    device: torch.device,
    submodules_dict: Dict[str, Any],
) -> Tuple[Any, Any, Dict[str, float]]:
    # Lazy imports to keep this script runnable when VBench2 deps aren't installed.
    from retinaface.predict_single import Model  # type: ignore
    from vbench2.third_party.arcface.models import resnet_face18  # type: ignore
    import io
    import zipfile

    device_str = str(device)
    retinaface_confidence_threshold = 0.5
    retinaface_nms_threshold = 0.4
    url = "https://github.com/ternaus/retinaface/releases/download/0.01/retinaface_resnet50_2020-07-20-f168fae3c.zip"
    retina_state_dict: Optional[dict] = None

    def _load_retinaface_from_zip(zip_path: Path) -> dict:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            pths = [n for n in zf.namelist() if n.endswith(".pth")]
            if not pths:
                raise RuntimeError(f"No .pth found inside {zip_path}")
            with zf.open(pths[0], "r") as f:
                buf = io.BytesIO(f.read())
        return torch.load(buf, map_location="cpu")

    vbench2_cache_dir = os.environ.get("VBENCH2_CACHE_DIR", "").strip()
    if vbench2_cache_dir:
        local_zip = (
            Path(vbench2_cache_dir).expanduser().resolve()
            / "retinaface"
            / "retinaface_resnet50_2020-07-20-f168fae3c.zip"
        )
        if local_zip.exists():
            retina_state_dict = _load_retinaface_from_zip(local_zip)
    if retina_state_dict is None:
        raise RuntimeError(
            "RetinaFace checkpoint not found locally. "
            "Please place it at $VBENCH2_CACHE_DIR/retinaface/retinaface_resnet50_2020-07-20-f168fae3c.zip "
            f"(expected URL source: {url})"
        )

    retina_model = Model(max_size=2048, device=device_str)
    retina_model.load_state_dict(retina_state_dict)

    arc_model = resnet_face18(use_se=False)
    state_dict = torch.load(submodules_dict["model"], map_location="cpu")
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    arc_model.load_state_dict(new_state_dict)
    arc_model.to(device).eval()

    return retina_model, arc_model, {
        "retinaface_confidence_threshold": float(retinaface_confidence_threshold),
        "retinaface_nms_threshold": float(retinaface_nms_threshold),
    }


def _identity_eval_one_video_no_duplicate(
    *,
    video_id: int,
    frame_paths: List[Path],
    cut_frame_0based: Optional[int],
    device: torch.device,
    submodules_dict: Dict[str, Any],
    retina_model: Optional[Any] = None,
    arc_model: Optional[Any] = None,
    retinaface_confidence_threshold: float = 0.5,
    retinaface_nms_threshold: float = 0.4,
    subject_mode: str = "single",
    expected_subject_count: int = 1,
    min_valid_frames: int = 20,
    baseline_image_path: Optional[Path] = None,
    baseline_source: str = "metadata_cond_image",
) -> Dict[str, Any]:
    """
    Human identity against the metadata conditioning image, resized to the
    generated resolution. The reported score is the mean ArcFace similarity.
    """
    if retina_model is None or arc_model is None:
        retina_model, arc_model, identity_knobs = _init_identity_models(
            device=device,
            submodules_dict=submodules_dict,
        )
        retinaface_confidence_threshold = identity_knobs["retinaface_confidence_threshold"]
        retinaface_nms_threshold = identity_knobs["retinaface_nms_threshold"]

    subject_count = 2 if str(subject_mode) == "multi" or int(expected_subject_count) >= 2 else 1
    if baseline_image_path is None:
        return {
            "video_id": int(video_id),
            "video_results": -1,
            "reason": "missing_identity_reference_image",
            "baseline_source": str(baseline_source),
            "score_mode": "mean_similarity",
            "subject_mode": str(subject_mode),
            "expected_subject_count": int(subject_count),
        }

    def _extract_face_features(face_rgb: "np.ndarray") -> "np.ndarray":
        import numpy as np

        face_image = cv2.resize(face_rgb, (128, 128))
        image_gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        image = np.dstack((image_gray, np.fliplr(image_gray)))
        image = image.transpose((2, 0, 1))
        image = image[:, np.newaxis, :, :].astype(np.float32, copy=False)
        image -= 127.5
        image /= 127.5
        data = torch.from_numpy(image).to(device)
        with torch.no_grad():
            output = arc_model(data)
        out = output.detach().cpu().numpy()
        fe_1 = out[::2]
        fe_2 = out[1::2]
        feature = np.hstack((fe_1, fe_2))
        return feature.flatten()

    def _sim(x1: "np.ndarray", x2: "np.ndarray") -> float:
        import numpy as np

        return float(np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-12))

    # Build mapping from frame file -> 0-based frame index (matching scene_transitions.json)
    base = _frame_index_base(frame_paths)
    frame_indices_0based: List[int] = []
    for fp in frame_paths:
        try:
            stem_i = int(fp.stem)
        except Exception:
            continue
        frame_indices_0based.append(int(stem_i - base))
    scene0_mask, scene1p_mask = _split_mask_from_cut(frame_indices_0based, cut_frame_0based)

    # Per-subject bookkeeping; baseline frame is excluded from denominators.
    valid_by_subject: List[List[bool]] = [[] for _ in range(subject_count)]
    similarity_by_subject: List[List[Optional[float]]] = [[] for _ in range(subject_count)]
    baseline_feats: List[Any] = []

    def _mean_where(vals: List[Optional[float]], mask: List[bool]) -> Optional[float]:
        acc = 0.0
        n = 0
        for v, m in zip(vals, mask):
            if m and (v is not None):
                acc += float(v)
                n += 1
        return (acc / float(n)) if n > 0 else None

    def _append_invalid_for_all() -> None:
        for subj_idx in range(subject_count):
            valid_by_subject[subj_idx].append(False)
            similarity_by_subject[subj_idx].append(None)

    def _extract_face_detections(frame: "np.ndarray") -> List[Dict[str, Any]]:
        faces = retina_model.predict_jsons(
            frame,
            confidence_threshold=retinaface_confidence_threshold,
            nms_threshold=retinaface_nms_threshold,
        )
        if faces is None:
            faces = []
        detections: List[Dict[str, Any]] = []
        for face in faces:
            box = face.get("bbox")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            x1, y1, x2, y2 = box
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(frame.shape[1], int(x2))
            y2 = min(frame.shape[0], int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            detections.append({"bbox": [x1, y1, x2, y2], "feature": _extract_face_features(face_crop)})
        # Keep matching cheap while still allowing distractor faces.
        return detections[: max(subject_count, 5)]

    def _best_assignment(similarities: List[List[float]], num_detections: int) -> Dict[int, int]:
        if num_detections <= 0:
            return {}
        if num_detections >= subject_count:
            import itertools

            best_perm: Optional[Tuple[int, ...]] = None
            best_score: Optional[float] = None
            for perm in itertools.permutations(range(num_detections), subject_count):
                score = sum(similarities[subj_idx][det_idx] for subj_idx, det_idx in enumerate(perm))
                if best_score is None or score > best_score:
                    best_score = float(score)
                    best_perm = perm
            return {subj_idx: int(det_idx) for subj_idx, det_idx in enumerate(best_perm or tuple())}

        # If fewer faces are detected than expected, use the highest-scoring partial one-to-one match.
        pairs = [
            (float(similarities[subj_idx][det_idx]), subj_idx, det_idx)
            for subj_idx in range(subject_count)
            for det_idx in range(num_detections)
        ]
        assigned_subjects = set()
        assigned_detections = set()
        assignment: Dict[int, int] = {}
        for _, subj_idx, det_idx in sorted(pairs, reverse=True):
            if subj_idx in assigned_subjects or det_idx in assigned_detections:
                continue
            assignment[int(subj_idx)] = int(det_idx)
            assigned_subjects.add(subj_idx)
            assigned_detections.add(det_idx)
        return assignment

    if baseline_image_path is not None:
        ref_bgr = cv2.imread(str(baseline_image_path))
        if ref_bgr is None:
            return {
                "video_id": int(video_id),
                "video_results": -1,
                "reason": "invalid_reference_image",
                "reason_detail": str(baseline_image_path),
                "baseline_source": str(baseline_source),
                "baseline_image_path": str(baseline_image_path),
                "subject_mode": str(subject_mode),
                "expected_subject_count": int(subject_count),
            }
        if frame_paths:
            first_bgr = cv2.imread(str(frame_paths[0]))
            if first_bgr is not None:
                target_h, target_w = first_bgr.shape[:2]
                if ref_bgr.shape[:2] != (target_h, target_w):
                    ref_bgr = cv2.resize(ref_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        ref_frame = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
        ref_detections = _extract_face_detections(ref_frame)
        if len(ref_detections) < subject_count:
            return {
                "video_id": int(video_id),
                "video_results": -1,
                "reason": "invalid_reference_face",
                "reason_detail": f"expected_{subject_count}_faces_found_{len(ref_detections)}",
                "baseline_source": str(baseline_source),
                "baseline_image_path": str(baseline_image_path),
                "subject_mode": str(subject_mode),
                "expected_subject_count": int(subject_count),
                "retinaface_confidence_threshold": retinaface_confidence_threshold,
                "retinaface_nms_threshold": retinaface_nms_threshold,
            }
        baseline_feats = [ref_detections[i]["feature"] for i in range(subject_count)]

    for fp in frame_paths:
        bgr = cv2.imread(str(fp))
        if bgr is None:
            _append_invalid_for_all()
            continue
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        detections = _extract_face_detections(frame)

        if len(baseline_feats) < subject_count:
            return {
                "video_id": int(video_id),
                "video_results": -1,
                "reason": "missing_baseline",
                "baseline_source": str(baseline_source),
            }

        similarities = [
            [_sim(baseline_feats[subj_idx], det["feature"]) for det in detections]
            for subj_idx in range(subject_count)
        ]
        assignment = _best_assignment(similarities, len(detections))
        for subj_idx in range(subject_count):
            det_idx = assignment.get(subj_idx)
            if det_idx is None:
                valid_by_subject[subj_idx].append(False)
                similarity_by_subject[subj_idx].append(None)
                continue
            sim = float(similarities[subj_idx][det_idx])
            valid_by_subject[subj_idx].append(True)
            similarity_by_subject[subj_idx].append(sim)

    subject_results: List[Dict[str, Any]] = []
    for subj_idx in range(subject_count):
        valid_for_scoring = valid_by_subject[subj_idx]
        similarity_for_scoring = similarity_by_subject[subj_idx]
        total_valid = sum(1 for v in valid_for_scoring if v)
        valid_s0 = _count_where(valid_for_scoring, scene0_mask)
        valid_s1p = _count_where(valid_for_scoring, scene1p_mask)
        mean_similarity_overall = _mean_where(similarity_for_scoring, valid_for_scoring)
        mean_similarity_scene0 = _mean_where(
            similarity_for_scoring, [bool(v and m) for v, m in zip(valid_for_scoring, scene0_mask)]
        )
        mean_similarity_scene1plus = _mean_where(
            similarity_for_scoring, [bool(v and m) for v, m in zip(valid_for_scoring, scene1p_mask)]
        )
        subject_results.append(
            {
                "subject_index": int(subj_idx),
                "is_valid_subject": bool(total_valid >= int(min_valid_frames)),
                "mean_similarity_overall": mean_similarity_overall,
                "mean_similarity_scene0": mean_similarity_scene0,
                "mean_similarity_scene1plus": mean_similarity_scene1plus,
                "total_valid_frames_overall": int(total_valid),
                "total_valid_frames_scene0": int(valid_s0),
                "total_valid_frames_scene1plus": int(valid_s1p),
            }
        )

    valid_subject_results = [s for s in subject_results if s["is_valid_subject"]]
    if not valid_subject_results:
        return {
            "video_id": int(video_id),
            "video_results": -1,
            "reason": "too_few_valid_frames",
            "subject_mode": str(subject_mode),
            "expected_subject_count": int(subject_count),
            "subjects": subject_results,
            "score_mode": "mean_similarity",
        }

    def _mean_subject_field(field: str) -> Optional[float]:
        vals = [s.get(field) for s in valid_subject_results if s.get(field) is not None]
        return float(sum(float(v) for v in vals) / len(vals)) if vals else None

    mean_similarity_overall = _mean_subject_field("mean_similarity_overall")
    mean_similarity_scene0 = _mean_subject_field("mean_similarity_scene0")
    mean_similarity_scene1plus = _mean_subject_field("mean_similarity_scene1plus")

    score_overall = mean_similarity_overall
    score_scene0 = mean_similarity_scene0
    score_scene1plus = mean_similarity_scene1plus

    return {
        "video_id": int(video_id),
        "baseline_source": str(baseline_source),
        "baseline_image_path": str(baseline_image_path) if baseline_image_path is not None else None,
        "subject_mode": str(subject_mode),
        "expected_subject_count": int(subject_count),
        "score_mode": "mean_similarity",
        "score_overall": score_overall,
        "score_scene0": score_scene0,
        "score_scene1plus": score_scene1plus,
        "mean_similarity_overall": mean_similarity_overall,
        "mean_similarity_scene0": mean_similarity_scene0,
        "mean_similarity_scene1plus": mean_similarity_scene1plus,
        "total_valid_frames_overall": int(sum(s["total_valid_frames_overall"] for s in subject_results)),
        "total_valid_frames_scene0": int(sum(s["total_valid_frames_scene0"] for s in subject_results)),
        "total_valid_frames_scene1plus": int(sum(s["total_valid_frames_scene1plus"] for s in subject_results)),
        "subjects": subject_results,
    }


def _human_clothes_eval_videos(
    *,
    video_paths: List[Path],
    device: torch.device,
    submodules_dict: Dict[str, Any],
    subject_metadata_by_video_id: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    VBench2 Human_Clothes style (LLaVA-Video multi-question yes/no).
    One-person Q1 asks whether only one person is speaking throughout.
    """
    # Lazy imports to keep script runnable without VBench2 deps.
    from vbench2.third_party.LLaVA_NeXT.llava.model.builder import load_pretrained_model  # type: ignore
    from vbench2.third_party.LLaVA_NeXT.llava.mm_utils import tokenizer_image_token  # type: ignore
    from vbench2.third_party.LLaVA_NeXT.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # type: ignore
    from vbench2.third_party.LLaVA_NeXT.llava.conversation import conv_templates  # type: ignore
    from decord import VideoReader, cpu  # type: ignore
    import numpy as np
    import copy

    def _load_video_for_llava(video_path: str, max_frames_num: int, fps: int = 1, force_sample: bool = False):
        if max_frames_num == 0:
            return np.zeros((1, 336, 336, 3))
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        step = round(vr.get_avg_fps() / fps)
        frame_idx = [i for i in range(0, len(vr), step)]
        frame_time = [i / step for i in frame_idx]
        if len(frame_idx) > max_frames_num or force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
        frame_time_str = ",".join([f"{i:.2f}s" for i in frame_time])
        frames = vr.get_batch(frame_idx).asnumpy()
        return frames, frame_time_str, float(video_time)

    base_question = [
        "Is there only one person speaking in the video throughout?",
        "Is the person in the video the same throughout?",
        "Does the clothes of the person in the video (color, texture) remain consistent throughout?",
    ]
    multi_subject_question = [
        "Are there exactly two main people visible in the video throughout?",
        "Do both main people remain the same people throughout the video?",
        "Do the clothes of both main people (color, texture) remain consistent throughout?",
    ]

    model_name = "llava_qwen"
    device_map = "auto"
    pretrained = submodules_dict.get("llava")
    llava_tokenizer, llava_model, image_processor, _ = load_pretrained_model(
        pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map
    )
    llava_model.eval()

    processed = []
    valid_num = 0
    pass_num = 0
    conv_template = "qwen_1_5"
    for vp in tqdm(video_paths, desc="Human_Clothes (per-video)", unit="video"):
        vid = _video_id_from_path(vp)
        subject_meta = (
            subject_metadata_by_video_id.get(int(vid), {}) if subject_metadata_by_video_id and vid is not None else {}
        )
        subject_mode = str(subject_meta.get("subject_mode", "single"))
        questions = multi_subject_question if subject_mode == "multi" else base_question
        max_frames_num = 64
        video, frame_time, video_time = _load_video_for_llava(str(vp), max_frames_num, 1, force_sample=True)
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(device).bfloat16()
        video = [video]
        time_instruction = (
            f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. "
            f"These frames are located at {frame_time}. Return yes or no only for the following question."
        )
        valid = True
        passed = True
        question_answers: List[Dict[str, Any]] = []
        for i, q in enumerate(questions):
            question = DEFAULT_IMAGE_TOKEN + f"{time_instruction}\n{q}"
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            input_ids = tokenizer_image_token(
                prompt_question, llava_tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)
            cont = llava_model.generate(
                input_ids,
                images=video,
                modalities=["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=128,
            )
            text_outputs = llava_tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip().lower()
            question_answers.append({"question": q, "answer": text_outputs})
            if i == 0 and ("yes" not in text_outputs):
                valid = False
                break
            if i != 0 and ("yes" not in text_outputs):
                passed = False

        if not valid:
            processed.append(
                {
                    "video_path": str(vp),
                    "video_id": int(vid) if vid is not None else None,
                    "video_results": -1,
                    "subject_mode": subject_mode,
                    "expected_subject_count": subject_meta.get("expected_subject_count"),
                    "questions": questions,
                    "question_answers": question_answers,
                }
            )
            continue
        valid_num += 1
        result_obj = {
            "video_path": str(vp),
            "video_id": int(vid) if vid is not None else None,
            "video_results": 1 if passed else 0,
            "subject_mode": subject_mode,
            "expected_subject_count": subject_meta.get("expected_subject_count"),
            "questions": questions,
            "question_answers": question_answers,
        }
        if passed:
            pass_num += 1
        processed.append(result_obj)

    score = (pass_num / valid_num) if valid_num > 0 else None
    return {"score": score, "valid_videos": int(valid_num), "passed_videos": int(pass_num), "results": processed}


def _resolve_vbench2_cache_dir(cli_value: str) -> Optional[Path]:
    """
    Resolve VBench2 cache dir from:
      1) CLI --vbench2-cache-dir
      2) env VBENCH2_CACHE_DIR
    """
    if str(cli_value).strip():
        return Path(cli_value).expanduser().resolve()
    env = os.environ.get("VBENCH2_CACHE_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return None


def _resolve_maybe_relative_path(p: str, *, vbench2_repo: Path, cache_dir: Optional[Path] = None) -> str:
    """
    Resolve a path that may be returned by vbench2.utils.init_submodules as a relative path
    (e.g. 'vbench2/third_party/...'). We try, in order:
      1) if already absolute -> return as-is
      2) relative to vbench2_repo
      3) relative to cache_dir (if provided)
      4) Path.resolve() relative to CWD
    """
    p = str(p).strip()
    if not p:
        return p
    pp = Path(p).expanduser()
    if pp.is_absolute():
        return str(pp)
    cand1 = (vbench2_repo / pp).resolve()
    if cand1.exists():
        return str(cand1)
    if cache_dir is not None:
        cand2 = (cache_dir / pp).resolve()
        if cand2.exists():
            return str(cand2)
    return str(pp.resolve())


def _maybe_patch_yolo_world_text_model_name(
    *,
    detector_config_path: str,
    vbench2_repo: Path,
    cache_dir: Optional[Path],
    scratch_dir: Path,
) -> str:
    """
    YOLO-World configs in VBench2 sometimes hardcode an internal lab path for CLIP:
      text_model_name = '/mnt/petrelfs/.../clip-vit-base-patch32'
    which breaks elsewhere. If detected, rewrite it to a usable value.
    """
    cfg_path = Path(detector_config_path).expanduser().resolve()
    if not cfg_path.is_file():
        return str(cfg_path)

    text = cfg_path.read_text(encoding="utf-8", errors="ignore")
    bad = "/mnt/petrelfs/zhengdian/code/ckpt/clip-vit-base-patch32"
    if bad not in text:
        return str(cfg_path)

    # Prefer a local cached CLIP dir if present under cache_dir; otherwise use HF repo id.
    replacement = "openai/clip-vit-base-patch32"
    if cache_dir is not None:
        # Look for a directory that likely matches a local HF snapshot or local checkout.
        for cand in [
            cache_dir / "openai" / "clip-vit-base-patch32",
            cache_dir / "clip-vit-base-patch32",
        ]:
            if cand.is_dir():
                replacement = str(cand)
                break

    # Patch config content. We operate on quoted string literals and only rewrite when the
    # *string value* starts with known relative prefixes. This avoids accidental double
    # replacement inside already-absolute paths (which previously produced duplicated paths).
    import re

    yolo_world_root = (vbench2_repo / "vbench2" / "third_party" / "YOLO-World").resolve()
    mmyolo_cfg_root = (yolo_world_root / "mmyolo" / "configs").resolve()

    def _maybe_keep_trailing_slash(old: str, new: str) -> str:
        if old.endswith("/") and (not new.endswith("/")):
            return new + "/"
        return new

    def _rewrite_value(val: str) -> str:
        # 1) Fix the hardcoded lab-local CLIP path
        if val == bad:
            return replacement

        # 2) Keep `_base_` and other mmyolo relative paths working when the config is copied
        #    into a scratch dir.
        if val.startswith("mmyolo/configs/"):
            suffix = val[len("mmyolo/configs/") :]
            out = str((mmyolo_cfg_root / suffix).resolve())
            return _maybe_keep_trailing_slash(val, out)

        # 3) Fix vbench2-relative YOLO-World paths (data_root, class_text_path, etc.)
        if val.startswith("vbench2/third_party/YOLO-World/"):
            suffix = val[len("vbench2/third_party/YOLO-World/") :]
            out = str((yolo_world_root / suffix).resolve())
            return _maybe_keep_trailing_slash(val, out)

        # 4) Fix YOLO-World local paths that are relative to the YOLO-World repo root.
        if val.startswith("data/"):
            cand = (yolo_world_root / val).resolve()
            if cand.exists():
                return _maybe_keep_trailing_slash(val, str(cand))

        # 5) Fix ../pretrained_models/... style (relative to config file directory)
        if val.startswith("../"):
            cand = (cfg_path.parent / val).resolve()
            if cand.exists():
                return _maybe_keep_trailing_slash(val, str(cand))

        return val

    def _sub(m: "re.Match") -> str:
        quote = m.group(1)
        val = m.group(2)
        return f"{quote}{_rewrite_value(val)}{quote}"

    patched = re.sub(r"(['\"])([^'\"]+?)\1", _sub, text)

    # Extra safety pass: some configs use the vbench2-relative YOLO-World prefix in many places
    # (e.g. data_root, class_text_path). Ensure these are made absolute even if the generic
    # literal rewriter misses them for any reason.
    patched = re.sub(
        r"(?P<q>['\"])vbench2/third_party/YOLO-World/",
        rf"\g<q>{str(yolo_world_root)}/",
        patched,
    )
    scratch_dir.mkdir(parents=True, exist_ok=True)
    out_cfg = scratch_dir / cfg_path.name
    out_cfg.write_text(patched, encoding="utf-8")
    print(f"[Info] Patched YOLO-World text_model_name in config: {cfg_path} -> {out_cfg} (value='{replacement}')")
    return str(out_cfg.resolve())


def _try_import_init_submodules():
    try:
        from vbench2.utils import init_submodules  # type: ignore
        return init_submodules
    except Exception:
        return None

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

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _list_frame_files(frames_dir: Path) -> List[Path]:
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


def _write_jpg_from_frame(frame_path: Path, output_jpg: Path) -> None:
    output_jpg.parent.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(frame_path))
    if img is None:
        raise ValueError(f"Failed to read frame: {frame_path}")
    ok = cv2.imwrite(str(output_jpg), img)
    if not ok:
        raise RuntimeError(f"Failed to write image: {output_jpg}")


def _ensure_symlink_or_copy(src: Path, dst: Path) -> None:
    """
    Create dst pointing to src. Prefer symlink; fall back to copy if symlinks are not permitted.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Handle the common failure mode where a previous run left a dangling symlink:
    # - Path.exists() is False for broken symlinks, but symlink creation will still raise FileExistsError.
    if dst.is_symlink() and not dst.exists():
        dst.unlink(missing_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src)
    except FileExistsError:
        # Someone else created it concurrently (or it appeared after our check).
        return
    except Exception:
        shutil.copy2(src, dst)


@dataclass
class VideoJob:
    video_id: int
    cut_frames: List[int]
    frames_dir: Path


def _build_jobs(
    *,
    scene_transitions_path: Path,
    frames_root: Path,
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

        cuts = r.get("cuts", []) if isinstance(r, dict) else []
        if not cuts:
            # no detected transitions -> no scene1+; still allow scene0-only runs if user wants
            cut_frames: List[int] = []
        else:
            cut_frames = sorted([int(c["frame"]) for c in cuts if isinstance(c, dict) and "frame" in c])
            cut_frames = [c for c in cut_frames if c >= 0]

        frames_dir = frames_root / f"{vid:03d}"
        if not frames_dir.exists():
            continue
        cut_frames = filter_last_frame_cuts(cut_frames, last_frame_index_from_frames_dir(frames_dir))

        jobs.append(
            VideoJob(
                video_id=vid,
                cut_frames=cut_frames,
                frames_dir=frames_dir,
            )
        )

    return jobs


def _ensure_scene_clips_and_images(
    jobs: List[VideoJob],
    *,
    fps: float,
    image_pick: str,
    sub_clip_dir: Path,
    prepare_scene0_clip: bool,
    prepare_scene1plus: bool,
) -> None:
    """
    Prepare:
      - shared scene0 reference image under sub_clip_dir/i2v_scene0_images/<id>.jpg
      - shared scene0 clip under sub_clip_dir/<id>_scene0.mp4 when --eval-scene0 is on
      - shared scene1+ clip under sub_clip_dir/<id>_after_scene0_overall.mp4 when --eval-scene1plus is on
    Later scenes are scored against the opening-scene image.
    """
    for job in tqdm(jobs, desc="Preparing I2V assets", unit="video"):
        all_frames = _list_frame_files(job.frames_dir)
        if not all_frames:
            continue

        base = _frame_index_base(all_frames)  # usually 0 or 1
        cut_stems = sorted({c + base for c in job.cut_frames}) if job.cut_frames else []
        first_cut_stem = cut_stems[0] if cut_stems else None
        selected0: List[Path] = []
        for p in all_frames:
            try:
                stem_i = int(p.stem)
            except Exception:
                continue
            if stem_i < base:
                continue
            if first_cut_stem is not None and stem_i >= first_cut_stem:
                continue
            selected0.append(p)

        # 1) Always write shared scene0 reference image (needed for any later-scene evaluation)
        scene0_images_dir = sub_clip_dir / "i2v_scene0_images"
        scene0_img = scene0_images_dir / f"{job.video_id:03d}.jpg"
        scene0_clip = sub_clip_dir / f"{job.video_id:03d}_scene0.mp4"
        if (not scene0_img.exists()) and selected0:
            if image_pick == "first":
                ref = selected0[0]
            elif image_pick == "middle":
                ref = selected0[len(selected0) // 2]
            else:
                raise ValueError(f"Unknown --image-pick: {image_pick}")
            _write_jpg_from_frame(ref, scene0_img)

        if prepare_scene0_clip and (not scene0_clip.exists()) and selected0:
            _write_mp4_from_frames(selected0, scene0_clip, fps=fps)

        if prepare_scene1plus and first_cut_stem is not None:
            shared_clip = sub_clip_dir / f"{job.video_id:03d}_after_scene0_overall.mp4"
            # Backward-compat reuse if an older name exists in the same shared folder.
            legacy = sub_clip_dir / f"{job.video_id:03d}_after_scene0.mp4"
            if (not shared_clip.exists()) and legacy.exists():
                _ensure_symlink_or_copy(legacy, shared_clip)

            if not shared_clip.exists():
                selected_1p: List[Path] = []
                for p in all_frames:
                    try:
                        stem_i = int(p.stem)
                    except Exception:
                        continue
                    if stem_i >= first_cut_stem:
                        selected_1p.append(p)
                if selected_1p:
                    _write_mp4_from_frames(selected_1p, shared_clip, fps=fps)


def _run_vbench_i2v(
    *,
    clips_dir: Path,
    images_dir: Path,
    output_dir: Path,
    device: torch.device,
    vbench_cache_dir: Optional[Path],
    vbench_repo: Path,
    local: bool,
    dimensions: List[str],
    ratio: str,
    name: str,
) -> Path:
    if vbench_cache_dir is not None:
        os.environ["VBENCH_CACHE_DIR"] = str(vbench_cache_dir)

    if str(vbench_repo) not in sys.path:
        sys.path.insert(0, str(vbench_repo))

    from vbench2_beta_i2v import VBenchI2V  # type: ignore

    output_dir.mkdir(parents=True, exist_ok=True)
    full_info_json = str((vbench_repo / "vbench2_beta_i2v" / "vbench2_i2v_full_info.json").resolve())
    my_vbench = VBenchI2V(device, full_info_json, str(output_dir))

    my_vbench.evaluate(
        videos_path=str(clips_dir),
        name=name,
        dimension_list=dimensions,
        custom_image_folder=str(images_dir),
        mode="custom_input",
        local=bool(local),
        resolution=ratio,
    )
    return output_dir / f"{name}_eval_results.json"

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VBench-I2V on SceneShifter scene0 (pre-cut) clips.")
    parser.add_argument(
        "--scene-transitions-json",
        type=str,
        default="",
        help="Path to scene_transitions.json. The shell always passes this.",
    )
    parser.add_argument(
        "--frames-root",
        type=str,
        default="",
        help="Root dir containing per-video frame folders (e.g., .../frames/001/*.jpg).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory to write scene0 clips/images and evaluation results.",
    )
    parser.add_argument(
        "--sub-clip-dir",
        type=str,
        default="",
        help="Optional shared directory to store/reuse preprocessed clips across scripts. "
        "If empty, defaults to <run_dir>/sub_clip where run_dir is output_dir's parent.",
    )
    parser.add_argument(
        "--vbench-cache-dir",
        type=str,
        default=str(_BENCH_ROOT / "vbench"),
        help="Path to the VBench cache dir containing checkpoints (sets VBENCH_CACHE_DIR).",
    )
    parser.add_argument("--fps", type=float, default=25.0, help="FPS for written scene0 clips.")
    parser.add_argument(
        "--vbench-repo",
        type=str,
        default="",
        help="VBench code repository. Defaults to the VBENCH_REPO environment variable.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use VBench-I2V local loading (will clone/download DINO weights into VBENCH_CACHE_DIR if missing).",
    )
    parser.add_argument(
        "--dimensions",
        type=str,
        default="i2v_subject",
        help="Comma-separated dimension list (default: i2v_subject).",
    )
    parser.add_argument(
        "--eval-scene1plus",
        action="store_true",
        help="Evaluate i2v_subject on the concatenated scene1+ clip (default: on).",
    )
    # parser.set_defaults(eval_scene1plus=True)
    parser.add_argument(
        "--eval-scene0",
        action="store_true",
        help="Evaluate scene0 i2v_subject. Default: off.",
    )
    parser.add_argument(
        "--ratio",
        type=str,
        default="1-1",
        help="Dummy ratio value required by API (used for standard dataset images; custom_input uses custom images).",
    )
    parser.add_argument(
        "--image-pick",
        type=str,
        default="first",
        choices=["first", "middle"],
        help="Which opening-scene frame to save as the I2V reference image.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for evaluation: 'cuda' or 'cpu'. Default: cuda if available.",
    )
    parser.add_argument(
        "--eval-human-anatomy",
        action="store_true",
        help="Also compute VBench2 Human_Anatomy and per-split (scene0 vs scene1+) aggregates.",
    )
    parser.add_argument(
        "--eval-human-identity",
        action="store_true",
        help="Also compute Human_Identity from the metadata conditioning image using mean ArcFace similarity.",
    )
    parser.add_argument(
        "--eval-human-clothes",
        action="store_true",
        help="Also compute VBench2 Human_Clothes (with Q1 adapted to 'speaking').",
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default=str(_BENCH_ROOT / "configs" / "sceneshifter_multitalk"),
        help="Directory containing <id>_metadata.json. Used to infer single- vs multi-subject evaluation behavior.",
    )
    parser.add_argument(
        "--vbench2-repo",
        type=str,
        default="",
        help="Path to the VBench-2.0 repo. Defaults to the VBENCH2_REPO environment variable.",
    )
    parser.add_argument(
        "--vbench2-cache-dir",
        type=str,
        default=str(_BENCH_ROOT / "vbench2"),
        help="Override for VBENCH2_CACHE_DIR (where VBench2 stores downloaded weights).",
    )
    # Optional explicit weight/model paths (used when vbench2.utils.init_submodules is unavailable)
    parser.add_argument(
        "--human-identity-arcface-weight",
        type=str,
        default=str(_BENCH_ROOT / "vbench2" / "arcface" / "resnet18_110.pth"),
        help="Path to ArcFace weight (resnet18_110.pth).",
    )
    parser.add_argument(
        "--human-clothes-llava-dir",
        type=str,
        default=str(_BENCH_ROOT / "vbench2" / "lmms-lab" / "LLaVA-Video-7B-Qwen2"),
        help="Local dir for the LLaVA-Video model.",
    )
    parser.add_argument(
        "--human-anatomy-detector-config",
        type=str,
        default="",
        help="Optional explicit detector config path for Human_Anatomy (YOLO-World).",
    )
    parser.add_argument(
        "--human-anatomy-detector-weights",
        type=str,
        default="",
        help="Optional explicit detector weight path for Human_Anatomy (YOLO-World).",
    )
    parser.add_argument(
        "--human-anatomy-human-weight",
        type=str,
        default="",
        help="Optional explicit anomaly-detector weight for 'human' category.",
    )
    parser.add_argument(
        "--human-anatomy-face-weight",
        type=str,
        default="",
        help="Optional explicit anomaly-detector weight for 'face' category.",
    )
    parser.add_argument(
        "--human-anatomy-hand-weight",
        type=str,
        default="",
        help="Optional explicit anomaly-detector weight for 'hand' category.",
    )
    add_ids_resume_args(parser)
    args = parser.parse_args()
    if not str(args.scene_transitions_json).strip() or not str(args.frames_root).strip() or not str(args.output_dir).strip():
        raise SystemExit("--scene-transitions-json, --frames-root, and --output-dir are required")

    scene_transitions_path = Path(args.scene_transitions_json).expanduser().resolve()
    frames_root = Path(args.frames_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    metadata_dir = Path(args.metadata_dir).expanduser().resolve() if str(args.metadata_dir).strip() else None
    eval_dir = output_dir / "vbench_i2v_results"
    sub_clip_dir = (
        Path(args.sub_clip_dir).expanduser().resolve()
        if str(args.sub_clip_dir).strip()
        else (output_dir.parent / "sub_clip")
    )

    if not scene_transitions_path.exists():
        raise SystemExit(f"Missing scene transitions json: {scene_transitions_path}")
    if not frames_root.exists():
        raise SystemExit(f"Missing frames root: {frames_root}")

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    vbench_cache_dir: Optional[Path] = None
    if args.vbench_cache_dir.strip():
        vbench_cache_dir = Path(args.vbench_cache_dir).expanduser().resolve()

    # If the user didn't pass --local, but the required local DINO assets already exist in the
    # specified cache dir, auto-enable local mode to avoid torch.hub GitHub downloads.
    if (not bool(args.local)) and (vbench_cache_dir is not None):
        dino_ckpt = vbench_cache_dir / "dino_model" / "dino_vitbase16_pretrain.pth"
        dino_repo = vbench_cache_dir / "dino_model" / "facebookresearch_dino_main"
        dino_hubconf = dino_repo / "hubconf.py"
        if dino_ckpt.is_file() and dino_hubconf.is_file():
            args.local = True
            print(
                f"[Info] Found local DINO assets under {vbench_cache_dir}. "
                "Auto-enabling --local to prevent torch.hub GitHub downloads."
            )

    requested_ids = parse_ids(args.ids)
    only_ids = sorted(requested_ids) if requested_ids is not None else None
    dims = [d.strip() for d in args.dimensions.split(",") if d.strip() == "i2v_subject"]
    if not dims:
        raise SystemExit("--dimensions must be i2v_subject")
    needs_vbench = bool(args.eval_scene0) or bool(args.eval_scene1plus)
    needs_vbench2 = bool(args.eval_human_anatomy) or bool(args.eval_human_identity) or bool(args.eval_human_clothes)
    vbench_repo = _require_code_repo("VBENCH_REPO", args.vbench_repo) if needs_vbench else None
    vbench2_repo = _require_code_repo("VBENCH2_REPO", args.vbench2_repo) if needs_vbench2 else None

    jobs = _build_jobs(
        scene_transitions_path=scene_transitions_path,
        frames_root=frames_root,
        only_ids=only_ids,
    )
    # If only evaluating scene1+, ignore videos with no detected transitions.
    if bool(args.eval_scene1plus) and (not bool(args.eval_scene0)):
        jobs = [j for j in jobs if bool(j.cut_frames)]
    print(f"Found {len(jobs)} videos to evaluate (after filtering).")
    subject_metadata = _subject_metadata_by_video_id(metadata_dir, jobs)

    def _metric_jobs(out_path: Path) -> Tuple[List[Any], Optional[Dict[str, Any]]]:
        existing = load_json(out_path) if args.resume else None
        existing_list = (
            existing.get("results", [])
            if isinstance(existing, dict) and isinstance(existing.get("results"), list)
            else []
        )
        done = ids_from_list(existing_list) if args.resume else set()
        available = [int(j.video_id) for j in jobs]
        todo, skipped = resolve_todo_ids(
            requested_ids=requested_ids,
            available_ids=available,
            done_ids=done,
            resume=bool(args.resume),
        )
        log_resume_plan(label=out_path.stem, todo=todo, skipped=skipped, requested=requested_ids)
        return [j for j in jobs if int(j.video_id) in todo], existing

    _ensure_scene_clips_and_images(
        jobs,
        fps=float(args.fps),
        image_pick=str(args.image_pick),
        sub_clip_dir=sub_clip_dir,
        prepare_scene0_clip=bool(args.eval_scene0),
        prepare_scene1plus=bool(args.eval_scene1plus),
    )

    # Optional: Human fidelity evaluation (Human_Anatomy for now)
    if bool(args.eval_human_anatomy):
        # Configure vbench2 imports + cache dir
        _maybe_add_vbench2_to_syspath(vbench2_repo)
        if str(args.vbench2_cache_dir).strip():
            os.environ["VBENCH2_CACHE_DIR"] = str(Path(args.vbench2_cache_dir).expanduser().resolve())

        cache_dir = _resolve_vbench2_cache_dir(str(args.vbench2_cache_dir))
        init_submodules = _try_import_init_submodules()
        anatomy_submods: Dict[str, Any]
        if init_submodules is not None:
            submods = init_submodules(["Human_Anatomy"], local=bool(args.local))
            anatomy_submods = submods["Human_Anatomy"]
        else:
            # Manual mode: require explicit paths or a populated cache dir.
            if cache_dir is None:
                raise SystemExit(
                    "Human_Anatomy requires VBench2 weights. Either install `gdown` so vbench2 can auto-download, "
                    "or pass --vbench2-cache-dir and/or explicit --human-anatomy-* weight paths."
                )
            # Defaults matching VBench2 init_submodules (paths under cache)
            detector_config = str(args.human_anatomy_detector_config).strip() or str(
                (vbench2_repo / "vbench2/third_party/YOLO-World/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py").resolve()
            )
            detector_weights = str(args.human_anatomy_detector_weights).strip() or str(
                (cache_dir / "YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth").resolve()
            )
            default_cfg = str((vbench2_repo / "vbench2/third_party/ViTDetector/simmim_finetune__vit_base__img224__800ep.yaml").resolve())
            human_w = str(args.human_anatomy_human_weight).strip() or str((cache_dir / "anomaly_detector/human.pth").resolve())
            face_w = str(args.human_anatomy_face_weight).strip() or str((cache_dir / "anomaly_detector/face.pth").resolve())
            hand_w = str(args.human_anatomy_hand_weight).strip() or str((cache_dir / "anomaly_detector/hand.pth").resolve())
            anatomy_submods = {
                "detector_config": detector_config,
                "detector_weights": detector_weights,
                "analyzer_configs": {
                    "human": {"cfg_path": default_cfg, "weight_path": human_w, "threshold": 0.4545454545454546},
                    "face": {"cfg_path": default_cfg, "weight_path": face_w, "threshold": 0.30303030303030304},
                    "hand": {"cfg_path": default_cfg, "weight_path": hand_w, "threshold": 0.3232},
                },
                "batch_size": 128,
            }

        # Normalize vbench2-returned paths (may be relative like 'vbench2/third_party/...').
        anatomy_submods["detector_config"] = _resolve_maybe_relative_path(
            anatomy_submods["detector_config"], vbench2_repo=vbench2_repo, cache_dir=cache_dir
        )
        anatomy_submods["detector_weights"] = _resolve_maybe_relative_path(
            anatomy_submods["detector_weights"], vbench2_repo=vbench2_repo, cache_dir=cache_dir
        )
        for k, cfg in list(anatomy_submods.get("analyzer_configs", {}).items()):
            if isinstance(cfg, dict):
                if "cfg_path" in cfg:
                    cfg["cfg_path"] = _resolve_maybe_relative_path(
                        cfg["cfg_path"], vbench2_repo=vbench2_repo, cache_dir=cache_dir
                    )
                if "weight_path" in cfg:
                    cfg["weight_path"] = _resolve_maybe_relative_path(
                        cfg["weight_path"], vbench2_repo=vbench2_repo, cache_dir=cache_dir
                    )

        # Patch YOLO-World config if it contains a hardcoded CLIP path that doesn't exist here.
        anatomy_submods["detector_config"] = _maybe_patch_yolo_world_text_model_name(
            detector_config_path=anatomy_submods["detector_config"],
            vbench2_repo=vbench2_repo,
            cache_dir=cache_dir,
            scratch_dir=eval_dir / "_patched_configs",
        )

        # Build detector/analyzer ONCE and reuse across all videos.
        # This avoids repeated YOLO-World/mmengine initialization which can crash on some stacks.
        try:
            from vbench2.third_party.ViTDetector.detect import Detector, Analyzer  # type: ignore

            shared_detector = Detector(
                config_file=anatomy_submods["detector_config"],
                weight_file=anatomy_submods["detector_weights"],
                device=str(device),
            )
            shared_analyzer = Analyzer(
                model_configs=anatomy_submods["analyzer_configs"],
                device=str(device),
                batch_size=anatomy_submods["batch_size"],
                class_thresholds={
                    k: v["threshold"] for k, v in anatomy_submods["analyzer_configs"].items()
                },
            )
        except Exception as e:
            raise SystemExit(f"Failed to initialize Human_Anatomy detector/analyzer: {e}")

        out_path = eval_dir / "human_anatomy_scene_splits.json"
        anatomy_jobs, existing_anatomy = _metric_jobs(out_path)
        anatomy_results: List[Dict[str, Any]] = []
        for job in tqdm(anatomy_jobs, desc="Human_Anatomy (per-video)", unit="video"):
            cut0 = job.cut_frames[0] if job.cut_frames else None
            frame_paths = _frames_for_video_id(frames_root, job.video_id)
            if not frame_paths:
                continue
            try:
                r = _anatomy_eval_one_video_no_duplicate(
                    video_id=job.video_id,
                    frame_paths=frame_paths,
                    cut_frame_0based=cut0,
                    device=str(device),
                    submodules_dict=anatomy_submods,
                    detector=shared_detector,
                    analyzer=shared_analyzer,
                )
                r.update(subject_metadata.get(int(job.video_id), {}))
                anatomy_results.append(r)
            except Exception as e:
                anatomy_results.append(
                    {
                        "video_id": int(job.video_id),
                        "error": str(e),
                        "score_overall_vbench2_style": None,
                        **subject_metadata.get(int(job.video_id), {}),
                    }
                )

        if args.resume and isinstance(existing_anatomy, dict):
            merged = merge_list_by_video_id(existing_anatomy.get("results", []), anatomy_results)
        else:
            merged = anatomy_results
        _write_json(out_path, {"metric": "Human_Anatomy", "results": merged})
        print(f"Wrote Human_Anatomy scene-split results: {out_path}")

    if bool(args.eval_human_identity):
        _maybe_add_vbench2_to_syspath(vbench2_repo)
        if str(args.vbench2_cache_dir).strip():
            os.environ["VBENCH2_CACHE_DIR"] = str(Path(args.vbench2_cache_dir).expanduser().resolve())
        cache_dir = _resolve_vbench2_cache_dir(str(args.vbench2_cache_dir))
        init_submodules = _try_import_init_submodules()
        identity_submods: Dict[str, Any]
        if init_submodules is not None:
            submods = init_submodules(["Human_Identity"], local=bool(args.local))
            identity_submods = submods["Human_Identity"]
        else:
            # Manual mode: only need arcface weight path.
            arcface_weight = str(args.human_identity_arcface_weight).strip()
            if not arcface_weight:
                if cache_dir is None:
                    raise SystemExit(
                        "Human_Identity requires ArcFace weights. Either install `gdown` so vbench2 can auto-download, "
                        "or pass --vbench2-cache-dir (with arcface/resnet18_110.pth), or pass --human-identity-arcface-weight."
                    )
                arcface_weight = str((cache_dir / "arcface/resnet18_110.pth").resolve())
            identity_submods = {"model": arcface_weight}

        try:
            shared_retina_model, shared_arc_model, identity_knobs = _init_identity_models(
                device=device,
                submodules_dict=identity_submods,
            )
        except Exception as e:
            raise SystemExit(f"Failed to initialize Human_Identity models: {e}")

        out_path = eval_dir / "human_identity_scene_splits.json"
        identity_jobs, existing_identity = _metric_jobs(out_path)
        identity_results: List[Dict[str, Any]] = []
        for job in tqdm(identity_jobs, desc="Human_Identity (per-video)", unit="video"):
            cut0 = job.cut_frames[0] if job.cut_frames else None
            frame_paths = _frames_for_video_id(frames_root, job.video_id)
            if not frame_paths:
                continue
            subject_meta = subject_metadata.get(int(job.video_id), {})
            baseline_source = "metadata_cond_image"
            cond_image_path = subject_meta.get("cond_image_path")
            baseline_image_path = Path(cond_image_path).expanduser().resolve() if cond_image_path else None
            if baseline_image_path is None:
                identity_results.append(
                    {
                        "video_id": int(job.video_id),
                        "video_results": -1,
                        "reason": "missing_identity_reference_image",
                        "baseline_source": baseline_source,
                        "score_mode": "mean_similarity",
                        "score_overall": None,
                        **subject_meta,
                    }
                )
                continue
            try:
                r = _identity_eval_one_video_no_duplicate(
                    video_id=job.video_id,
                    frame_paths=frame_paths,
                    cut_frame_0based=cut0,
                    device=device,
                    submodules_dict=identity_submods,
                    retina_model=shared_retina_model,
                    arc_model=shared_arc_model,
                    retinaface_confidence_threshold=identity_knobs["retinaface_confidence_threshold"],
                    retinaface_nms_threshold=identity_knobs["retinaface_nms_threshold"],
                    subject_mode=str(subject_meta.get("subject_mode", "single")),
                    expected_subject_count=int(subject_meta.get("expected_subject_count", 1) or 1),
                    baseline_image_path=baseline_image_path,
                    baseline_source=baseline_source,
                )
                identity_results.append(r)
            except Exception as e:
                identity_results.append(
                    {
                        "video_id": int(job.video_id),
                        "error": str(e),
                        "score_mode": "mean_similarity",
                        "score_overall": None,
                        **subject_meta,
                    }
                )

        if args.resume and isinstance(existing_identity, dict):
            merged = merge_list_by_video_id(existing_identity.get("results", []), identity_results)
        else:
            merged = identity_results
        _write_json(out_path, {"metric": "Human_Identity", "results": merged})
        print(f"Wrote Human_Identity scene-split results: {out_path}")

    if bool(args.eval_human_clothes):
        _maybe_add_vbench2_to_syspath(vbench2_repo)
        if str(args.vbench2_cache_dir).strip():
            os.environ["VBENCH2_CACHE_DIR"] = str(Path(args.vbench2_cache_dir).expanduser().resolve())
        cache_dir = _resolve_vbench2_cache_dir(str(args.vbench2_cache_dir))
        init_submodules = _try_import_init_submodules()
        clothes_submods: Dict[str, Any]
        if init_submodules is not None:
            submods = init_submodules(["Human_Clothes"], local=bool(args.local))
            clothes_submods = submods["Human_Clothes"]
        else:
            llava_dir = str(args.human_clothes_llava_dir).strip()
            if not llava_dir:
                if cache_dir is None:
                    raise SystemExit(
                        "Human_Clothes requires a local LLaVA-Video model dir. Either install `gdown` so vbench2 can auto-download, "
                        "or pass --vbench2-cache-dir (with lmms-lab/LLaVA-Video-7B-Qwen2), or pass --human-clothes-llava-dir."
                    )
                llava_dir = str((cache_dir / "lmms-lab/LLaVA-Video-7B-Qwen2").resolve())
            clothes_submods = {"llava": llava_dir}

        out_path = eval_dir / "human_clothes_results.json"
        clothes_jobs, existing_clothes = _metric_jobs(out_path)
        # Use the original generated videos for evaluation.
        video_paths: List[Path] = []
        run_dir = frames_root.parent
        for job in clothes_jobs:
            video_path = run_dir / "videos" / f"{job.video_id:03d}.mp4"
            if video_path.exists():
                video_paths.append(video_path)

        clothes = _human_clothes_eval_videos(
            video_paths=video_paths,
            device=device,
            submodules_dict=clothes_submods,
            subject_metadata_by_video_id=subject_metadata,
        )
        new_results = clothes.get("results", []) if isinstance(clothes, dict) else []
        if args.resume and isinstance(existing_clothes, dict):
            merged_results = merge_list_by_video_id(existing_clothes.get("results", []), new_results)
        else:
            merged_results = new_results
        valid = [r for r in merged_results if isinstance(r, dict) and r.get("video_results") in (0, 1)]
        passed = [r for r in valid if r.get("video_results") == 1]
        score = (len(passed) / len(valid)) if valid else None
        _write_json(
            out_path,
            {
                "metric": "Human_Clothes",
                "score": score,
                "valid_videos": len(valid),
                "passed_videos": len(passed),
                "results": merged_results,
            },
        )
        print(f"Wrote Human_Clothes results: {out_path}")

    # Shared scene0 reference images (must match clip stems).
    scene0_images_ref = sub_clip_dir / "i2v_scene0_images"

    # Optional: Evaluate scene0 (subject + background)
    if bool(args.eval_scene0):
        scene0_clips = output_dir / "scene0_clips"
        # Reuse existing scene0 clips produced by the pipeline under sub_clip_dir/<id>_scene0.mp4.
        # Build a directory of <id>.mp4 links/copies into output_dir/scene0_clips so VBench can consume it.
        scene0_clips.mkdir(parents=True, exist_ok=True)
        for job in jobs:
            src = sub_clip_dir / f"{job.video_id:03d}_scene0.mp4"
            if not src.exists():
                continue
            dst = scene0_clips / f"{job.video_id:03d}.mp4"
            _ensure_symlink_or_copy(src, dst)

        if any(scene0_clips.glob("*.mp4")) and scene0_images_ref.exists() and vbench_repo is not None:
            print("Evaluating scene0...")
            results_json = _run_vbench_i2v(
                clips_dir=scene0_clips,
                images_dir=scene0_images_ref,
                output_dir=eval_dir,
                device=device,
                vbench_cache_dir=vbench_cache_dir,
                vbench_repo=vbench_repo,
                local=bool(args.local),
                dimensions=["i2v_subject"],
                ratio=str(args.ratio),
                name="i2v_scene0",
            )
            print(f"Eval results (scene0): {results_json}")

    if bool(args.eval_scene1plus) and scene0_images_ref.exists() and ("i2v_subject" in dims) and vbench_repo is not None:
        # Build a directory of <id>.mp4 links to shared sub_clip/<id>_after_scene0_overall.mp4
        scene1plus_inputs = output_dir / "scene1plus_clips"
        scene1plus_inputs.mkdir(parents=True, exist_ok=True)
        for job in jobs:
            src = sub_clip_dir / f"{job.video_id:03d}_after_scene0_overall.mp4"
            if not src.exists():
                continue
            dst = scene1plus_inputs / f"{job.video_id:03d}.mp4"
            _ensure_symlink_or_copy(src, dst)

        if any(scene1plus_inputs.glob("*.mp4")):
            print("Evaluating concatenated scene1+ clips...")
            results_json = _run_vbench_i2v(
                clips_dir=scene1plus_inputs,
                images_dir=scene0_images_ref,
                output_dir=eval_dir,
                device=device,
                vbench_cache_dir=vbench_cache_dir,
                vbench_repo=vbench_repo,
                local=bool(args.local),
                dimensions=["i2v_subject"],
                ratio=str(args.ratio),
                name="i2v_scene1plus_concat",
            )
            print(f"Eval results (scene1+ concat): {results_json}")


if __name__ == "__main__":
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
    main()

