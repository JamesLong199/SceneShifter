"""Mask load, polarity (1=BG, 0=FG), downsample, and F10/F7 spatial index helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def parse_mask_entry(value) -> Tuple[int, str]:
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, int):
        value = (value, "full")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"invalid self-attention mask entry: {value!r}")
    active, region = int(value[0]), str(value[1]).lower()
    if active not in (0, 1) or region not in ("full", "fg", "bg"):
        raise ValueError(f"invalid self-attention mask entry: {value!r}")
    return active, region


def normalize_self_attention_mask(mask) -> List:
    out = []
    for row in mask:
        if isinstance(row, dict):
            if set(row) != {"fg", "bg"}:
                raise ValueError(f"split mask row must have fg/bg keys, got {row!r}")
            out.append({region: [parse_mask_entry(v) for v in row[region]] for region in ("fg", "bg")})
        else:
            out.append([parse_mask_entry(v) for v in row])
    return out


def identity_self_attention_mask(num_segments: int) -> List[List[Tuple[int, str]]]:
    return [
        [(1, "full") if i == j else (0, "full") for j in range(num_segments)]
        for i in range(num_segments)
    ]


def _resize_binary(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if mask.ndim == 2:
        mask = mask[None]
    return F.interpolate(mask[:, None].float(), (height, width), mode="nearest")[:, 0] > 0.5


def center_crop_to_aspect(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Match Wan's resize-then-center-crop geometry before downsampling."""
    src_h, src_w = map(int, mask.shape[-2:])
    target_ratio = int(height) / int(width)
    source_ratio = src_h / src_w
    if abs(source_ratio - target_ratio) < 0.01:
        return mask
    if source_ratio > target_ratio:
        crop_h = min(src_h, max(1, round(src_w * target_ratio)))
        top = (src_h - crop_h) // 2
        return mask[..., top : top + crop_h, :]
    crop_w = min(src_w, max(1, round(src_h / target_ratio)))
    left = (src_w - crop_w) // 2
    return mask[..., :, left : left + crop_w]


def downsample_mask_cover_original_2d(
    mask: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Downsample FG while preserving every positive source pixel on round-trip."""
    if mask.ndim != 2:
        raise ValueError(f"expected a 2D mask, got {tuple(mask.shape)}")
    mask = center_crop_to_aspect(mask, height, width)
    src_h, src_w = map(int, mask.shape)
    original = mask > 0.5
    latent = _resize_binary(original, int(height), int(width))[0]
    if (src_h, src_w) == (int(height), int(width)):
        return latent
    owner = F.interpolate(
        torch.arange(
            int(height) * int(width),
            device=mask.device,
            dtype=torch.float32,
        ).view(1, 1, int(height), int(width)),
        size=(src_h, src_w),
        mode="nearest",
    )[0, 0].long()
    uncovered = original & ~_resize_binary(latent, src_h, src_w)[0]
    while bool(uncovered.any()):
        counts = torch.bincount(
            owner[uncovered], minlength=int(height) * int(width)
        )
        best = int(counts.argmax())
        if int(counts[best]) <= 0:
            raise RuntimeError("unable to cover the original mask")
        latent.view(-1)[best] = True
        uncovered = original & ~_resize_binary(latent, src_h, src_w)[0]
    return latent


def downsample_mask_to_latent(
    mask: torch.Tensor,
    height: int,
    width: int,
    *,
    center_crop: bool = False,
    cover_original: bool = False,
) -> torch.Tensor:
    """Nearest downsample of a binary ``[H,W]`` or ``[N,H,W]`` mask."""
    if mask.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D mask, got {tuple(mask.shape)}")
    if cover_original:
        frames = mask[None] if mask.ndim == 2 else mask
        result = torch.stack(
            [
                downsample_mask_cover_original_2d(frame, int(height), int(width))
                for frame in frames
            ]
        )
        return result[0] if mask.ndim == 2 else result
    if center_crop:
        mask = center_crop_to_aspect(mask, int(height), int(width))
    result = _resize_binary(mask, int(height), int(width))
    return result[0] if mask.ndim == 2 else result


def read_mask_file(path: str) -> torch.Tensor:
    if str(path).endswith(".npy"):
        return torch.from_numpy(np.load(path)).float()
    if str(path).endswith(".pt"):
        return torch.load(path, map_location="cpu").float()
    raise ValueError(f"unsupported mask format: {path!r}")


def subject_fg_from_background_paths(
    paths: Dict[str, str],
    num_frames: int,
) -> torch.Tensor:
    """Load 1=BG / 0=FG files and return subject FG bool tensor ``[S,F,H,W]``."""
    if not isinstance(paths, dict) or not paths:
        raise ValueError("background_mask_path must be a non-empty subject-index dict")
    ordered = sorted((int(key), value) for key, value in paths.items())
    if [key for key, _ in ordered] != list(range(len(ordered))):
        raise ValueError("background_mask_path keys must be contiguous from 0")
    masks = []
    source_shape = None
    for _, path in ordered:
        value = read_mask_file(path)
        if value.ndim == 2:
            value = value[None].expand(int(num_frames), -1, -1).contiguous()
        elif value.ndim != 3 or int(value.shape[0]) != int(num_frames):
            raise ValueError(
                f"mask {path} must be 2D or have {num_frames} frame entries"
            )
        if source_shape is None:
            source_shape = tuple(value.shape[-2:])
        elif tuple(value.shape[-2:]) != source_shape:
            raise ValueError("all background masks must share one spatial shape")
        if not torch.isfinite(value).all():
            raise ValueError(f"mask {path} contains non-finite values")
        unique = torch.unique(value)
        if unique.numel() > 2 or not torch.all((unique == 0) | (unique == 1)):
            # Allow soft masks by thresholding, but require values in [0, 1].
            if value.min() < 0 or value.max() > 1:
                raise ValueError(f"mask {path} values must be in [0, 1] (1=BG, 0=FG)")
        masks.append(value <= 0.5)
    return torch.stack(masks)


def downsample_subject_fg(
    subject_fg: torch.Tensor,
    height: int,
    width: int,
    *,
    center_crop: bool = False,
    cover_original: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(subject_fg [S,F,HW], static_bg [F,HW])`` at latent resolution."""
    subjects = []
    for subject in range(subject_fg.shape[0]):
        latent = downsample_mask_to_latent(
            subject_fg[subject].float(),
            height,
            width,
            center_crop=center_crop,
            cover_original=cover_original,
        )
        subjects.append(latent.bool().flatten(1))
    subject_flat = torch.stack(subjects)
    union_fg = subject_flat.any(0)
    return subject_flat, ~union_fg


def downsample_subject_fg_head_union(
    subject_fg: torch.Tensor,
    head_fg: torch.Tensor,
    height: int,
    width: int,
    *,
    center_crop: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nearest human ∪ cover-downsampled head (Fun ``input_mask_cover_head_only``)."""
    if tuple(subject_fg.shape[:2]) != tuple(head_fg.shape[:2]):
        raise ValueError(
            f"head mask [S,F] {tuple(head_fg.shape[:2])} != human [S,F] {tuple(subject_fg.shape[:2])}"
        )
    unions = []
    for subject in range(int(subject_fg.shape[0])):
        human = downsample_mask_to_latent(
            subject_fg[subject].float(),
            height,
            width,
            center_crop=center_crop,
            cover_original=False,
        )
        head = downsample_mask_to_latent(
            head_fg[subject].float(),
            height,
            width,
            center_crop=center_crop,
            cover_original=True,
        )
        unions.append((torch.maximum(human.float(), head.float()) > 0.5).flatten(1))
    subject_flat = torch.stack(unions)
    return subject_flat, ~subject_flat.any(0)


def spatial_token_indices(
    start_frame: int,
    end_frame: int,
    spatial_tokens: int,
    region: str,
    background_by_frame: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    n_frames = int(end_frame) - int(start_frame)
    if region == "full" or background_by_frame is None:
        return torch.arange(n_frames * int(spatial_tokens), device=device)
    frame_bg = background_by_frame[int(start_frame):int(end_frame)]
    selected = frame_bg if region == "bg" else ~frame_bg
    return torch.where(selected.reshape(-1))[0]


def token_union_upgrade(
    prev_keep: Iterable[int],
    prev_frames: Iterable[int],
    context_frames: Iterable[int],
    spatial_tokens: int,
) -> List[int]:
    """Previous-scene tokens: FG-only except context frames which become full."""
    keep = set(int(v) for v in prev_keep)
    context = set(int(v) for v in context_frames)
    L = int(spatial_tokens)
    indices: List[int] = []
    for frame in prev_frames:
        frame = int(frame)
        base = frame * L
        if frame in context:
            indices.extend(range(base, base + L))
        else:
            indices.extend(base + t for t in sorted(keep))
    return indices
