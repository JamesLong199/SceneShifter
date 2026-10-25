#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
SceneShifter lip-sync evaluation (SyncNet confidence) + dlib landmark audit.

This script is a SceneShifter-specific wrapper around the SyncNet evaluation code
in `codes/evaluation_eat/code/SyncNetInstance.py`.

Key constraints (from the upstream codebase):
- SyncNet evaluation assumes **25 FPS video** and extracts audio at **16 kHz**.
- For full-body / background videos, you must crop to the head/face region first.

We mirror the preprocessing logic used by `codes/evaluation_eat/code/preprocess.py`:
- force input to 25fps
- run per-frame face alignment crop using dlib 68 landmarks (same geometry as `utils_crop.crop_and_align_224`)

Robustness vs. dlib failures:
- The original `preprocess.py` path *skips* frames when detection fails (dropping frames).
- Here we avoid temporal jitter/length drift by **filling failed frames**:
  - if a previous valid crop exists, reuse it
  - otherwise write a black frame (224x224)
  This keeps frame count stable for SyncNet evaluation.

Checkpoints (you provide these files):
- SyncNet v2 weights: `syncnet_v2.model`
- Dlib landmark model: `shape_predictor_68_face_landmarks.dat`

Defaults live under the bench `checkpoints/` directory.
SyncNet code and the 68-point templates are loaded from EVALUATION_EAT_CODE.

This script runs everything **in one pass**:
- dlib landmark detection audit (success/fail stats)
- cropping (from raw frames)
- SyncNet inference + dataset-level average score
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SCRIPTS_DIR = Path(__file__).resolve().parent
_BENCH_ROOT = _SCRIPTS_DIR.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from eval_resume_utils import (  # noqa: E402
    add_ids_resume_args,
    extract_video_id,
    ids_from_dict_keys,
    load_json,
    log_resume_plan,
    merge_dict_by_video_key,
    parse_ids,
    resolve_todo_ids,
    write_json,
)

import cv2
import numpy as np
from skimage import transform as tf

try:
    import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


_CHECKPOINT_DIR = _BENCH_ROOT / "checkpoints"


def _run(cmd: str) -> None:
    ret = subprocess.call(cmd, shell=True, stdout=None, stderr=None)
    if ret != 0:
        raise RuntimeError(f"Command failed (exit={ret}): {cmd}")


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _video_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _video_id_from_name(path: str) -> Optional[str]:
    m = re.match(r"^(\d+)\.mp4$", os.path.basename(path))
    if not m:
        return None
    return m.group(1).zfill(3)


def _eat_code_dir() -> str:
    raw = os.environ.get("EVALUATION_EAT_CODE", "").strip()
    if not raw:
        raise SystemExit("EVALUATION_EAT_CODE is unset. Export EVALUATION_EAT_CODE to the SyncNet code directory.")
    if not os.path.isdir(raw):
        raise SystemExit(f"EVALUATION_EAT_CODE does not exist: {raw}")
    return raw


def _sorted_frame_paths(frame_dir: str) -> List[str]:
    if not os.path.isdir(frame_dir):
        return []
    frames = glob.glob(os.path.join(frame_dir, "*.jpg"))
    frames += glob.glob(os.path.join(frame_dir, "*.png"))
    def _key(p: str) -> int:
        base = os.path.splitext(os.path.basename(p))[0]
        try:
            return int(base)
        except Exception:
            return 10**18
    frames.sort(key=_key)
    return frames


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _resolve_path(raw_path: Any, *, dataset_root: str, metadata_path: Optional[str] = None) -> Optional[str]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = raw_path.strip()
    if os.path.isabs(path):
        return path
    candidates = []
    if dataset_root:
        candidates.append(os.path.join(dataset_root, path))
    if metadata_path:
        candidates.append(os.path.join(os.path.dirname(metadata_path), path))
    candidates.append(os.path.abspath(path))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0] if candidates else path


def _load_fg_mask(mask_path: Optional[str]) -> Optional[np.ndarray]:
    if not mask_path or not os.path.exists(mask_path):
        return None
    bg_mask = np.load(mask_path)
    bg_mask = np.squeeze(bg_mask)
    if bg_mask.ndim != 2:
        return None
    return np.logical_not(bg_mask.astype(bool))


def _load_subject_specs(
    *,
    metadata_dir: str,
    dataset_root: str,
    vid_id: str,
) -> Tuple[List[SubjectSpec], Optional[str]]:
    if not metadata_dir:
        return [SubjectSpec(key="person1")], None

    meta_path = os.path.join(metadata_dir, f"{int(vid_id):03d}_metadata.json")
    if not os.path.exists(meta_path):
        return [SubjectSpec(key="person1")], None

    data = _read_json(meta_path)
    cond_audio = data.get("cond_audio", {})
    if not isinstance(cond_audio, dict):
        cond_audio = {"person1": cond_audio}

    routing = data.get("frame_prompt_routing", {})
    bg_masks = routing.get("background_mask_path", {}) if isinstance(routing, dict) else {}
    if not isinstance(bg_masks, dict):
        bg_masks = {"0": bg_masks}

    subject_keys = [k for k in ("person1", "person2") if k in cond_audio]
    if not subject_keys:
        subject_keys = ["person1"]

    specs: List[SubjectSpec] = []
    for idx, key in enumerate(subject_keys):
        audio_path = _resolve_path(cond_audio.get(key), dataset_root=dataset_root, metadata_path=meta_path)
        mask_path = _resolve_path(bg_masks.get(str(idx)), dataset_root=dataset_root, metadata_path=meta_path)
        specs.append(
            SubjectSpec(
                key=key,
                audio_path=audio_path,
                fg_mask=_load_fg_mask(mask_path),
                mask_path=mask_path,
            )
        )
    return specs, meta_path


@dataclass
class LandmarkStats:
    total_frames: int = 0
    detected_frames: int = 0
    failed_frames: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return float(self.detected_frames) / float(self.total_frames)


@dataclass
class CropStats(LandmarkStats):
    # Split failures for clarity:
    # - landmark_failed_frames: dlib detector/predictor did not return landmarks
    # - crop_failed_frames: landmarks exist but crop operation failed (rare)
    landmark_failed_frames: int = 0
    crop_failed_frames: int = 0
    filled_with_last: int = 0
    filled_with_black: int = 0


@dataclass
class SubjectSpec:
    key: str
    audio_path: Optional[str] = None
    fg_mask: Optional[np.ndarray] = None
    mask_path: Optional[str] = None


class FaceAlignCropper224:
    """
    Re-implementation of `codes/evaluation_eat/code/utils_crop.crop_and_align_224`
    but with explicit model/template paths (so cwd does not matter).
    """

    def __init__(self, dlib_landmark_model: str, base_open: str, base_close: str):
        import dlib

        if not os.path.exists(dlib_landmark_model):
            raise FileNotFoundError(
                f"Missing dlib landmark model: {dlib_landmark_model}\n"
                f"Expected location (default): {_CHECKPOINT_DIR / 'shape_predictor_68_face_landmarks.dat'}"
            )
        if not os.path.exists(base_open):
            raise FileNotFoundError(f"Missing template: {base_open}")
        if not os.path.exists(base_close):
            raise FileNotFoundError(f"Missing template: {base_close}")

        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(dlib_landmark_model)
        self._base_open = np.load(base_open)
        self._base_close = np.load(base_close)

    @staticmethod
    def _shape_to_np(shape, dtype="int"):
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    def detect_all_landmarks(self, bgr: np.ndarray) -> List[Dict[str, Any]]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rects = self._detector(gray, 1)
        if len(rects) == 0:
            return []

        h, w = bgr.shape[:2]
        detections: List[Dict[str, Any]] = []
        for rect in rects:
            shape = self._predictor(gray, rect)
            landmark = self._shape_to_np(shape)
            x1 = max(0, int(rect.left()))
            y1 = max(0, int(rect.top()))
            x2 = min(w, int(rect.right()))
            y2 = min(h, int(rect.bottom()))
            if x2 <= x1 or y2 <= y1:
                xs = landmark[:, 0]
                ys = landmark[:, 1]
                x1 = max(0, int(xs.min()))
                y1 = max(0, int(ys.min()))
                x2 = min(w, int(xs.max()))
                y2 = min(h, int(ys.max()))
            detections.append({"bbox": [x1, y1, x2, y2], "landmark": landmark})
        detections.sort(key=lambda d: (d["bbox"][0], d["bbox"][1]))
        return detections

    def detect_landmarks(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        detections = self.detect_all_landmarks(bgr)
        if not detections:
            return None
        return detections[0]["landmark"]

    def crop_align_224(self, bgr: np.ndarray, landmark: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if landmark is None:
            landmark = self.detect_landmarks(bgr)
        if landmark is None:
            return None

        if np.sum(landmark[37:39, 1] - landmark[40:42, 1]) < -9:
            template = self._base_open
        else:
            template = self._base_close

        pts2 = np.float32(template[27:45, :])
        mean_x, mean_y = pts2[:, 0].mean(), pts2[:, 1].mean()
        pts2[:, 0] = pts2[:, 0] - mean_x + 112
        pts2[:, 1] = pts2[:, 1] - mean_y + 112
        pts1 = np.float32(landmark[27:45, :])

        tform = tf.SimilarityTransform()
        tform.estimate(pts2, pts1)

        dst = tf.warp(bgr, tform, output_shape=(256, 256))
        dst = np.array(dst * 255, dtype=np.uint8)
        dst = dst[1:225, 1:225, :]
        return dst


def _mux_audio(*, silent_video: str, audio_source: str, out_video: str) -> None:
    _run(
        'ffmpeg -loglevel quiet -y '
        f'-i "{silent_video}" -i "{audio_source}" '
        "-map 0:v:0 -map 1:a:0? "
        "-c:v copy -c:a aac -shortest "
        f'"{out_video}"'
    )


def _crop_from_detection(
    *,
    bgr: np.ndarray,
    detection: Dict[str, Any],
    cropper: FaceAlignCropper224,
    crop_method: str,
) -> Optional[np.ndarray]:
    del crop_method
    landmark = detection.get("landmark")
    if landmark is None:
        return None
    return cropper.crop_align_224(bgr, landmark=landmark)


def _resize_mask(mask: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask.astype(bool)
    resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def _bbox_overlap_with_mask(bbox: List[int], fg_mask: Optional[np.ndarray], frame_shape: Tuple[int, int, int]) -> float:
    if fg_mask is None:
        return 0.0
    h, w = frame_shape[:2]
    mask = _resize_mask(fg_mask, width=w, height=h)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = mask[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    return float(region.sum()) / float(region.size)


def _bbox_iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_continuity_score(prev_bbox: List[int], curr_bbox: List[int], frame_shape: Tuple[int, int, int]) -> float:
    h, w = frame_shape[:2]
    pcx = (prev_bbox[0] + prev_bbox[2]) / 2.0
    pcy = (prev_bbox[1] + prev_bbox[3]) / 2.0
    ccx = (curr_bbox[0] + curr_bbox[2]) / 2.0
    ccy = (curr_bbox[1] + curr_bbox[3]) / 2.0
    diag = math.sqrt(float(w * w + h * h)) + 1e-6
    center_score = 1.0 - (math.sqrt((pcx - ccx) ** 2 + (pcy - ccy) ** 2) / diag)
    return float(_bbox_iou(prev_bbox, curr_bbox) + center_score)


def _best_assignment(scores: List[List[float]], num_detections: int) -> Dict[int, int]:
    import itertools

    num_subjects = len(scores)
    if num_subjects == 0 or num_detections == 0:
        return {}
    if num_detections >= num_subjects:
        best_perm: Optional[Tuple[int, ...]] = None
        best_score: Optional[float] = None
        for perm in itertools.permutations(range(num_detections), num_subjects):
            score = sum(scores[subj_idx][det_idx] for subj_idx, det_idx in enumerate(perm))
            if best_score is None or score > best_score:
                best_score = float(score)
                best_perm = perm
        return {subj_idx: int(det_idx) for subj_idx, det_idx in enumerate(best_perm or tuple())}

    pairs = [
        (float(scores[subj_idx][det_idx]), subj_idx, det_idx)
        for subj_idx in range(num_subjects)
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


def preprocess_to_facecrop_video(
    *,
    in_video: str,
    out_video: str,
    tmp_dir: str,
    cropper: FaceAlignCropper224,
    frames_dir: Optional[str],
    fps: int = 25,
    crop_method: str = "align224",
    audio_source: Optional[str] = None,
    save_frames_dir: Optional[str] = None,
    show_inner_pbar: bool = True,
) -> CropStats:
    """
    Produces a 224x224 @ 25fps cropped mp4 and muxes the original audio back in.

    If `frames_dir` exists, it is used as input (one frame file at a time).
    Otherwise we decode frames from the input video after forcing 25fps.
    """
    if crop_method != "align224":
        raise ValueError("crop_method must be align224")
    _safe_mkdir(tmp_dir)
    tmp_silent = os.path.join(tmp_dir, "video_facecrop_silent.mp4")
    tmp_25fps = os.path.join(tmp_dir, "video_25fps.mp4")

    black = np.zeros((224, 224, 3), dtype=np.uint8)
    last_good = None

    stats = CropStats()

    # Decide input source: raw frames (run dlib + crop_method)
    frame_paths = _sorted_frame_paths(frames_dir) if frames_dir else []

    writer = cv2.VideoWriter(
        tmp_silent,
        cv2.VideoWriter_fourcc("M", "P", "4", "V"),
        fps,
        (224, 224),
    )

    if frame_paths:
        if save_frames_dir:
            _safe_mkdir(save_frames_dir)
        frame_iter = frame_paths
        if tqdm is not None and show_inner_pbar:
            frame_iter = tqdm.tqdm(frame_paths, desc=f"Crop frames {os.path.basename(frames_dir) if frames_dir else ''}", total=len(frame_paths), leave=False, dynamic_ncols=True)

        for fp in frame_iter:
            img = cv2.imread(fp)
            if img is None:
                continue
            stats.total_frames += 1
            landmark = cropper.detect_landmarks(img)
            if landmark is None:
                stats.failed_frames += 1
                stats.landmark_failed_frames += 1
                if last_good is not None:
                    writer.write(last_good)
                    stats.filled_with_last += 1
                    if save_frames_dir:
                        cv2.imwrite(os.path.join(save_frames_dir, os.path.basename(fp)), last_good)
                else:
                    writer.write(black)
                    stats.filled_with_black += 1
                    if save_frames_dir:
                        cv2.imwrite(os.path.join(save_frames_dir, os.path.basename(fp)), black)
                continue
            stats.detected_frames += 1
            crop = cropper.crop_align_224(img, landmark=landmark)
            if crop is None:
                stats.failed_frames += 1
                stats.crop_failed_frames += 1
                if last_good is not None:
                    writer.write(last_good)
                    stats.filled_with_last += 1
                    if save_frames_dir:
                        cv2.imwrite(os.path.join(save_frames_dir, os.path.basename(fp)), last_good)
                else:
                    writer.write(black)
                    stats.filled_with_black += 1
                    if save_frames_dir:
                        cv2.imwrite(os.path.join(save_frames_dir, os.path.basename(fp)), black)
                continue
            writer.write(crop)
            last_good = crop
            if save_frames_dir:
                cv2.imwrite(os.path.join(save_frames_dir, os.path.basename(fp)), crop)
    else:
        # Force to 25fps then decode frames
        _run(f'ffmpeg -loglevel quiet -y -i "{in_video}" -r {fps} "{tmp_25fps}"')
        cap = cv2.VideoCapture(tmp_25fps)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {tmp_25fps}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            stats.total_frames += 1
            landmark = cropper.detect_landmarks(frame)
            if landmark is None:
                stats.failed_frames += 1
                stats.landmark_failed_frames += 1
                if last_good is not None:
                    writer.write(last_good)
                    stats.filled_with_last += 1
                else:
                    writer.write(black)
                    stats.filled_with_black += 1
                continue
            stats.detected_frames += 1
            crop = cropper.crop_align_224(frame, landmark=landmark)
            if crop is None:
                stats.failed_frames += 1
                stats.crop_failed_frames += 1
                if last_good is not None:
                    writer.write(last_good)
                    stats.filled_with_last += 1
                else:
                    writer.write(black)
                    stats.filled_with_black += 1
                continue
            writer.write(crop)
            last_good = crop
        cap.release()

    writer.release()

    # Mux audio back in (audio optional). Defaults to the original generated video
    # for backward compatibility; metadata can override it with subject-specific audio.
    _mux_audio(silent_video=tmp_silent, audio_source=audio_source or in_video, out_video=out_video)

    return stats


def preprocess_to_subject_facecrop_videos(
    *,
    in_video: str,
    subject_specs: List[SubjectSpec],
    out_videos: Dict[str, str],
    tmp_dir: str,
    cropper: FaceAlignCropper224,
    frames_dir: Optional[str],
    fps: int = 25,
    crop_method: str = "align224",
    save_frames_dirs: Optional[Dict[str, str]] = None,
    show_inner_pbar: bool = True,
) -> Dict[str, CropStats]:
    """
    Produce one 224x224 @ 25fps face-crop video per subject.

    Subject identity is initialized from metadata foreground masks on the first usable
    frame, then tracked frame-by-frame from the previously assigned bbox.
    """
    if crop_method != "align224":
        raise ValueError("crop_method must be align224")
    _safe_mkdir(tmp_dir)
    black = np.zeros((224, 224, 3), dtype=np.uint8)
    subject_keys = [s.key for s in subject_specs]
    stats = {s.key: CropStats() for s in subject_specs}
    last_good: Dict[str, Optional[np.ndarray]] = {s.key: None for s in subject_specs}
    prev_bbox: Dict[str, Optional[List[int]]] = {s.key: None for s in subject_specs}
    initialized = False

    silent_videos = {key: os.path.join(tmp_dir, f"{key}_facecrop_silent.mp4") for key in subject_keys}
    writers = {
        key: cv2.VideoWriter(
            silent_videos[key],
            cv2.VideoWriter_fourcc("M", "P", "4", "V"),
            fps,
            (224, 224),
        )
        for key in subject_keys
    }

    if save_frames_dirs:
        for path in save_frames_dirs.values():
            _safe_mkdir(path)

    def _write_fill(subject_key: str, frame_name: str) -> None:
        st = stats[subject_key]
        st.failed_frames += 1
        st.landmark_failed_frames += 1
        fill = last_good[subject_key]
        if fill is not None:
            writers[subject_key].write(fill)
            st.filled_with_last += 1
        else:
            fill = black
            writers[subject_key].write(fill)
            st.filled_with_black += 1
        if save_frames_dirs and subject_key in save_frames_dirs:
            cv2.imwrite(os.path.join(save_frames_dirs[subject_key], frame_name), fill)

    def _write_detection(subject_key: str, detection: Dict[str, Any], frame: np.ndarray, frame_name: str) -> None:
        st = stats[subject_key]
        st.detected_frames += 1
        crop = _crop_from_detection(
            bgr=frame,
            detection=detection,
            cropper=cropper,
            crop_method=crop_method,
        )
        if crop is None:
            st.failed_frames += 1
            st.crop_failed_frames += 1
            fill = last_good[subject_key]
            if fill is not None:
                writers[subject_key].write(fill)
                st.filled_with_last += 1
            else:
                fill = black
                writers[subject_key].write(fill)
                st.filled_with_black += 1
            if save_frames_dirs and subject_key in save_frames_dirs:
                cv2.imwrite(os.path.join(save_frames_dirs[subject_key], frame_name), fill)
            return
        writers[subject_key].write(crop)
        last_good[subject_key] = crop
        prev_bbox[subject_key] = [int(v) for v in detection["bbox"]]
        if save_frames_dirs and subject_key in save_frames_dirs:
            cv2.imwrite(os.path.join(save_frames_dirs[subject_key], frame_name), crop)

    def _iter_frames():
        frame_paths = _sorted_frame_paths(frames_dir) if frames_dir else []
        if frame_paths:
            frame_iter = frame_paths
            if tqdm is not None and show_inner_pbar:
                frame_iter = tqdm.tqdm(
                    frame_paths,
                    desc=f"Crop subjects {os.path.basename(frames_dir) if frames_dir else ''}",
                    total=len(frame_paths),
                    leave=False,
                    dynamic_ncols=True,
                )
            for fp in frame_iter:
                yield os.path.basename(fp), cv2.imread(fp)
            return

        tmp_25fps = os.path.join(tmp_dir, "video_25fps.mp4")
        _run(f'ffmpeg -loglevel quiet -y -i "{in_video}" -r {fps} "{tmp_25fps}"')
        cap = cv2.VideoCapture(tmp_25fps)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {tmp_25fps}")
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield f"{frame_idx:06d}.jpg", frame
                frame_idx += 1
        finally:
            cap.release()

    try:
        for frame_name, frame in _iter_frames():
            if frame is None:
                continue
            for key in subject_keys:
                stats[key].total_frames += 1

            detections = cropper.detect_all_landmarks(frame)
            if not detections:
                for key in subject_keys:
                    _write_fill(key, frame_name)
                continue
            if (not initialized) and len(detections) < len(subject_keys):
                for key in subject_keys:
                    _write_fill(key, frame_name)
                continue

            if not initialized:
                overlap_scores = [
                    [_bbox_overlap_with_mask(det["bbox"], spec.fg_mask, frame.shape) for det in detections]
                    for spec in subject_specs
                ]
                # If masks are unavailable or non-overlapping, deterministic left-to-right
                # assignment is safer than blocking the whole video.
                if max([max(row) if row else 0.0 for row in overlap_scores] or [0.0]) <= 0.0:
                    assignment = {idx: idx for idx in range(min(len(subject_keys), len(detections)))}
                else:
                    assignment = _best_assignment(overlap_scores, len(detections))
                if len(assignment) < len(subject_keys):
                    for key in subject_keys:
                        _write_fill(key, frame_name)
                    continue
                initialized = True
            else:
                continuity_scores = []
                for key in subject_keys:
                    cached_bbox = prev_bbox.get(key)
                    if cached_bbox is None:
                        continuity_scores.append([0.0 for _ in detections])
                    else:
                        continuity_scores.append(
                            [_bbox_continuity_score(cached_bbox, det["bbox"], frame.shape) for det in detections]
                        )
                assignment = _best_assignment(continuity_scores, len(detections))

            for subj_idx, key in enumerate(subject_keys):
                det_idx = assignment.get(subj_idx)
                if det_idx is None:
                    _write_fill(key, frame_name)
                else:
                    _write_detection(key, detections[det_idx], frame, frame_name)
    finally:
        for writer in writers.values():
            writer.release()

    for spec in subject_specs:
        _mux_audio(
            silent_video=silent_videos[spec.key],
            audio_source=spec.audio_path or in_video,
            out_video=out_videos[spec.key],
        )

    return stats


def get_parse():
    p = argparse.ArgumentParser("evaluate_lipsync_syncnet")

    p.add_argument("--videos_glob", type=str, default="", help="Glob of {id}.mp4 videos. The shell always passes this.")
    p.add_argument(
        "--frames_root",
        type=str,
        default="",
        help="If frames exist at frames_root/<id>/00.jpg..., they will be used for landmark detection/cropping.",
    )

    p.add_argument("--out_json", type=str, default="")
    p.add_argument("--work_dir", type=str, default="")
    p.add_argument(
        "--export_crops_dir",
        type=str,
        default="",
        help="Saves 224x224 crop frames to export_crops_dir/align224/<id>/*.jpg.",
    )

    p.add_argument(
        "--syncnet_ckpt",
        type=str,
        default=str(_CHECKPOINT_DIR / "syncnet_v2.model"),
    )
    p.add_argument(
        "--dlib_landmark_model",
        type=str,
        default=str(_CHECKPOINT_DIR / "shape_predictor_68_face_landmarks.dat"),
    )

    p.add_argument("--batch_size", type=int, default=20)
    p.add_argument("--vshift", type=int, default=15)
    p.add_argument(
        "--crop_method",
        type=str,
        default="align224",
        choices=["align224"],
        help="Face crop. align224 is a similarity warp from 68 landmarks.",
    )
    p.add_argument(
        "--metadata_dir",
        type=str,
        default=str(_BENCH_ROOT / "configs" / "sceneshifter_multitalk"),
        help="Directory containing <id>_metadata.json. Enables per-subject audio/mask lip-sync evaluation.",
    )
    p.add_argument(
        "--dataset_root",
        type=str,
        default=str(_BENCH_ROOT),
        help="Bench root used to resolve relative cond_audio and background_mask_path entries.",
    )
    p.add_argument(
        "--no_inner_pbar",
        action="store_true",
        help="Disable per-frame progress bars (keeps only the per-video bar).",
    )
    add_ids_resume_args(p)
    return p


def _stats_to_payload(stats: CropStats) -> Dict[str, Any]:
    return {
        "landmarks": {
            "total_frames": stats.total_frames,
            "detected_frames": stats.detected_frames,
            "failed_frames": stats.landmark_failed_frames,
            "success_rate": (float(stats.detected_frames) / float(stats.total_frames)) if stats.total_frames else 0.0,
        },
        "crop": {
            "total_frames": stats.total_frames,
            "failed_frames_total": stats.failed_frames,
            "failed_frames_landmarks": stats.landmark_failed_frames,
            "failed_frames_crop": stats.crop_failed_frames,
            "filled_with_last": stats.filled_with_last,
            "filled_with_black": stats.filled_with_black,
        },
        "preprocess": {
            "fps": 25,
            "total_frames": stats.total_frames,
            "detected_frames": stats.detected_frames,
            "failed_frames": stats.failed_frames,
            "success_rate": stats.success_rate,
            "filled_with_last": stats.filled_with_last,
            "filled_with_black": stats.filled_with_black,
        },
    }


def _syncnet_to_payload(offset: Any, conf: Any, min_dist: Any) -> Dict[str, Optional[float]]:
    return {
        "offset": None if offset is None else int(offset),
        "conf": None if conf is None else float(conf),
        "min_dist": None if min_dist is None else float(min_dist),
    }


def _mean(values: List[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


def main():
    args = get_parse().parse_args()
    if not str(args.videos_glob).strip() or not str(args.out_json).strip() or not str(args.work_dir).strip():
        raise SystemExit("--videos_glob, --out_json, and --work_dir are required")
    if not str(args.export_crops_dir).strip():
        raise SystemExit("--export_crops_dir is required")
    if args.crop_method != "align224":
        raise SystemExit("--crop_method must be align224")

    videos = sorted(glob.glob(args.videos_glob))
    videos = [v for v in videos if _video_id_from_name(v) is not None]
    if len(videos) == 0:
        raise RuntimeError(f"No videos found: {args.videos_glob}")

    requested_ids = parse_ids(args.ids)
    available_ids = [vid for vid in (extract_video_id(v) for v in videos) if vid is not None]
    existing_payload = load_json(Path(args.out_json)) if args.resume else None
    existing_results = (
        existing_payload.get("results", {})
        if isinstance(existing_payload, dict) and isinstance(existing_payload.get("results"), dict)
        else {}
    )
    done_ids = ids_from_dict_keys(existing_results) if args.resume else set()
    todo_ids, skipped_ids = resolve_todo_ids(
        requested_ids=requested_ids,
        available_ids=available_ids,
        done_ids=done_ids,
        resume=bool(args.resume),
    )
    log_resume_plan(label="lipsync", todo=todo_ids, skipped=skipped_ids, requested=requested_ids)
    videos = [v for v in videos if extract_video_id(v) in todo_ids]
    if not videos:
        if args.resume and existing_payload is not None:
            print("[lipsync] nothing to evaluate; keeping existing results")
            write_json(Path(args.out_json), existing_payload)
            return
        print("[lipsync] no videos to evaluate")
        return

    _safe_mkdir(os.path.dirname(args.out_json))
    _safe_mkdir(args.work_dir)
    if not args.export_crops_dir:
        raise ValueError("--export_crops_dir must be a non-empty path (crop export is always enabled).")
    _safe_mkdir(args.export_crops_dir)

    eat_code_dir = _eat_code_dir()
    if eat_code_dir not in sys.path:
        sys.path.insert(0, eat_code_dir)
    from SyncNetInstance import SyncNetInstance  # noqa: E402

    base_open = os.path.join(eat_code_dir, "base_68.npy")
    base_close = os.path.join(eat_code_dir, "base_68_close.npy")

    cropper = FaceAlignCropper224(
        dlib_landmark_model=args.dlib_landmark_model,
        base_open=base_open,
        base_close=base_close,
    )

    results: Dict[str, Dict] = {}

    def _pbar(iterable, *, desc: str, total: Optional[int] = None, leave: bool = False):
        if tqdm is None:
            return iterable
        return tqdm.tqdm(iterable, desc=desc, total=total, leave=leave, dynamic_ncols=True)
    if not os.path.exists(args.syncnet_ckpt):
        raise FileNotFoundError(
            f"Missing SyncNet checkpoint: {args.syncnet_ckpt}\n"
            f"Put it in: {_CHECKPOINT_DIR} (default) or pass --syncnet_ckpt."
        )

    s = SyncNetInstance()
    s.loadParameters(args.syncnet_ckpt)

    # SyncNetInstance expects an args-like object with tmp_dir/reference/batch_size/vshift
    class _SyncArgs:
        def __init__(self, batch_size: int, vshift: int):
            self.batch_size = batch_size
            self.vshift = vshift
            self.reference = "demo"
            self.tmp_dir = os.path.join(eat_code_dir, "temps")

    sync_args = _SyncArgs(args.batch_size, args.vshift)

    # Aggregate stats for summary
    conf_values: List[float] = []
    min_dist_values: List[float] = []
    offset_values: List[float] = []
    subject_conf_values: Dict[str, List[float]] = {}
    subject_min_dist_values: Dict[str, List[float]] = {}
    subject_offset_values: Dict[str, List[float]] = {}
    num_no_frames = 0

    for in_video in _pbar(videos, desc="Videos"):
        stem = _video_stem(in_video)
        vid_id = _video_id_from_name(in_video)
        if vid_id is None:
            continue
        subject = vid_id
        subject_specs, metadata_path = _load_subject_specs(
            metadata_dir=args.metadata_dir,
            dataset_root=args.dataset_root,
            vid_id=vid_id,
        )

        # Separate work directories per crop method to avoid overwriting.
        one_work = os.path.join(args.work_dir, f"{stem}_{args.crop_method}")
        shutil.rmtree(one_work, ignore_errors=True)
        _safe_mkdir(one_work)

        # Make output video names unique per crop_method.
        out_cropped = os.path.join(one_work, f"{stem}_{args.crop_method}_facecrop_25fps.mp4")

        frame_dir = os.path.join(args.frames_root, vid_id)
        frames_dir = frame_dir if os.path.isdir(frame_dir) else None

        subject_results: Dict[str, Dict[str, Any]] = {}
        video_conf_values: List[float] = []
        video_min_dist_values: List[float] = []
        video_offset_values: List[float] = []

        if len(subject_specs) > 1:
            out_videos = {
                spec.key: os.path.join(one_work, f"{stem}_{args.crop_method}_{spec.key}_facecrop_25fps.mp4")
                for spec in subject_specs
            }
            save_frames_dirs = {
                spec.key: os.path.join(args.export_crops_dir, args.crop_method, vid_id, spec.key)
                for spec in subject_specs
            }
            stats_by_subject = preprocess_to_subject_facecrop_videos(
                in_video=in_video,
                subject_specs=subject_specs,
                out_videos=out_videos,
                tmp_dir=os.path.join(one_work, "tmp_subjects"),
                cropper=cropper,
                frames_dir=frames_dir,
                fps=25,
                crop_method=args.crop_method,
                save_frames_dirs=save_frames_dirs,
                show_inner_pbar=(not args.no_inner_pbar),
            )

            for spec in subject_specs:
                sync_args.tmp_dir = os.path.join(
                    eat_code_dir,
                    "temps",
                    f"sceneshifter_{stem}_{args.crop_method}_{spec.key}",
                )
                shutil.rmtree(sync_args.tmp_dir, ignore_errors=True)
                offset, conf, min_dist = s.evaluate(sync_args, videofile=out_videos[spec.key])
                if conf is not None and offset is not None and min_dist is not None:
                    video_conf_values.append(float(conf))
                    video_min_dist_values.append(float(min_dist))
                    video_offset_values.append(float(offset))
                    subject_conf_values.setdefault(spec.key, []).append(float(conf))
                    subject_min_dist_values.setdefault(spec.key, []).append(float(min_dist))
                    subject_offset_values.setdefault(spec.key, []).append(float(offset))

                subject_payload = {
                    "audio_path": spec.audio_path,
                    "mask_path": spec.mask_path,
                    "preprocessed_video": out_videos[spec.key],
                    "export_crops_dir": save_frames_dirs[spec.key],
                    "syncnet": _syncnet_to_payload(offset, conf, min_dist),
                }
                subject_payload.update(_stats_to_payload(stats_by_subject[spec.key]))
                subject_results[spec.key] = subject_payload
        else:
            spec = subject_specs[0]
            # Always export crops for inspection/reuse later (written under export_crops_dir/<crop_method>/<id>/).
            export_crops_dir = os.path.join(args.export_crops_dir, args.crop_method, vid_id)
            stats = preprocess_to_facecrop_video(
                in_video=in_video,
                out_video=out_cropped,
                tmp_dir=os.path.join(one_work, "tmp"),
                cropper=cropper,
                frames_dir=frames_dir,
                fps=25,
                crop_method=args.crop_method,
                audio_source=spec.audio_path,
                save_frames_dir=export_crops_dir,
                show_inner_pbar=(not args.no_inner_pbar),
            )

            # Per-video SyncNet temp dir to avoid collisions
            sync_args.tmp_dir = os.path.join(eat_code_dir, "temps", f"sceneshifter_{stem}_{args.crop_method}")
            shutil.rmtree(sync_args.tmp_dir, ignore_errors=True)

            offset, conf, min_dist = s.evaluate(sync_args, videofile=out_cropped)
            if conf is not None and offset is not None and min_dist is not None:
                video_conf_values.append(float(conf))
                video_min_dist_values.append(float(min_dist))
                video_offset_values.append(float(offset))
                subject_conf_values.setdefault(spec.key, []).append(float(conf))
                subject_min_dist_values.setdefault(spec.key, []).append(float(min_dist))
                subject_offset_values.setdefault(spec.key, []).append(float(offset))

            subject_payload = {
                "audio_path": spec.audio_path,
                "mask_path": spec.mask_path,
                "preprocessed_video": out_cropped,
                "export_crops_dir": export_crops_dir,
                "syncnet": _syncnet_to_payload(offset, conf, min_dist),
            }
            subject_payload.update(_stats_to_payload(stats))
            subject_results[spec.key] = subject_payload

        video_conf = _mean(video_conf_values)
        video_min_dist = _mean(video_min_dist_values)
        video_offset = _mean(video_offset_values)
        if video_conf is None or video_offset is None or video_min_dist is None:
            num_no_frames += 1
        else:
            conf_values.append(float(video_conf))
            min_dist_values.append(float(video_min_dist))
            offset_values.append(float(video_offset))

        first_subject = subject_specs[0].key
        first_payload = subject_results[first_subject]
        results[subject] = {
            "video": in_video,
            "frames_dir": frames_dir,
            "metadata_path": metadata_path,
            "subject_mode": "multi" if len(subject_specs) > 1 else "single",
            "preprocessed_video": first_payload.get("preprocessed_video"),
            "crop_method": args.crop_method,
            "export_crops_dir": os.path.join(args.export_crops_dir, args.crop_method, vid_id),
            "syncnet": {
                "offset": video_offset,
                "conf": video_conf,
                "min_dist": video_min_dist,
            },
            "subjects": subject_results,
        }
        # Preserve legacy top-level diagnostics for single-subject consumers.
        results[subject].update(
            {
                "landmarks": first_payload.get("landmarks"),
                "crop": first_payload.get("crop"),
                "preprocess": first_payload.get("preprocess"),
            }
        )

    if args.resume and existing_results:
        merged_results = merge_dict_by_video_key(existing_results, results)
    else:
        merged_results = results

    # Recompute summary from the merged payload.
    conf_values = []
    min_dist_values = []
    offset_values = []
    subject_conf_values = {}
    subject_min_dist_values = {}
    subject_offset_values = {}
    num_no_frames = 0
    for item in merged_results.values():
        if not isinstance(item, dict):
            continue
        if item.get("frames_dir") in (None, ""):
            num_no_frames += 1
        sync = item.get("syncnet") if isinstance(item.get("syncnet"), dict) else {}
        if isinstance(sync.get("conf"), (int, float)):
            conf_values.append(float(sync["conf"]))
        if isinstance(sync.get("min_dist"), (int, float)):
            min_dist_values.append(float(sync["min_dist"]))
        if isinstance(sync.get("offset"), (int, float)):
            offset_values.append(float(sync["offset"]))
        subjects = item.get("subjects") if isinstance(item.get("subjects"), dict) else {}
        for skey, spayload in subjects.items():
            if not isinstance(spayload, dict):
                continue
            ssync = spayload.get("syncnet") if isinstance(spayload.get("syncnet"), dict) else {}
            if isinstance(ssync.get("conf"), (int, float)):
                subject_conf_values.setdefault(skey, []).append(float(ssync["conf"]))
            if isinstance(ssync.get("min_dist"), (int, float)):
                subject_min_dist_values.setdefault(skey, []).append(float(ssync["min_dist"]))
            if isinstance(ssync.get("offset"), (int, float)):
                subject_offset_values.setdefault(skey, []).append(float(ssync["offset"]))

    payload = {
        "meta": {
            "videos_glob": args.videos_glob,
            "frames_root": args.frames_root,
            "num_videos": len(merged_results),
            "syncnet_ckpt": args.syncnet_ckpt,
            "dlib_landmark_model": args.dlib_landmark_model,
            "crop_method": args.crop_method,
            "export_crops_dir": args.export_crops_dir,
            "metadata_dir": args.metadata_dir,
            "dataset_root": args.dataset_root,
            "syncnet_summary": {
                "num_videos_total": len(merged_results),
                "num_videos_scored": len(conf_values),
                "num_videos_no_frames": num_no_frames,
                "avg_conf": (sum(conf_values) / len(conf_values)) if conf_values else None,
                "avg_min_dist": (sum(min_dist_values) / len(min_dist_values)) if min_dist_values else None,
                "avg_offset": (sum(offset_values) / len(offset_values)) if offset_values else None,
                "by_subject": {
                    key: {
                        "num_scored": len(subject_conf_values.get(key, [])),
                        "avg_conf": _mean(subject_conf_values.get(key, [])),
                        "avg_min_dist": _mean(subject_min_dist_values.get(key, [])),
                        "avg_offset": _mean(subject_offset_values.get(key, [])),
                    }
                    for key in sorted(subject_conf_values)
                },
            },
        },
        "results": merged_results,
    }

    write_json(Path(args.out_json), payload)


if __name__ == "__main__":
    main()

