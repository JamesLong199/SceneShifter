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


def downsample_mask_to_latent(
    mask: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Nearest downsample of a binary ``[H,W]`` or ``[N,H,W]`` mask."""
    if mask.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D mask, got {tuple(mask.shape)}")
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(subject_fg [S,F,HW], static_bg [F,HW])`` at latent resolution."""
    subjects = []
    for subject in range(subject_fg.shape[0]):
        latent = downsample_mask_to_latent(subject_fg[subject].float(), height, width)
        subjects.append(latent.bool().flatten(1))
    subject_flat = torch.stack(subjects)
    union_fg = subject_flat.any(0)
    return subject_flat, ~union_fg


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
