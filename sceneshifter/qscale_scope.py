"""F10: spatial scope of Q-scale (new-scene BG-only). Independent of F9."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch


def default_q_scale_bg_only(backend: str) -> bool:
    """Hunyuan is better with F10 false. Other ports historically default true."""
    return backend != "hunyuan_avatar"


def new_scene_scale_token_mask(
    num_tokens: int,
    spatial_tokens: int,
    new_scene_bounds: Sequence[Tuple[int, int]],
    *,
    bg_only: bool,
    background_by_frame: torch.Tensor | None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Boolean mask over noisy tokens that F9 may scale.

    ``new_scene_bounds`` are ``(start_frame, end_frame)`` exclusive-end
    intervals for segments >= 1. Scene 0 is never included.
    """
    if device is None:
        device = (
            background_by_frame.device
            if background_by_frame is not None
            else torch.device("cpu")
        )
    mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    for start, end in new_scene_bounds:
        token_start = int(start) * int(spatial_tokens)
        token_end = int(end) * int(spatial_tokens)
        if not bg_only:
            mask[token_start:token_end] = True
            continue
        if background_by_frame is None:
            mask[token_start:token_end] = True
            continue
        frame_bg = background_by_frame[int(start):int(end)]
        n_frames = int(end) - int(start)
        expected = n_frames * int(spatial_tokens)
        flat = frame_bg.reshape(-1)
        if flat.numel() != expected and n_frames > 0:
            old_spatial = flat.numel() // n_frames
            if old_spatial > 0 and old_spatial * n_frames == flat.numel():
                src = flat.reshape(n_frames, 1, old_spatial).float()
                flat = torch.nn.functional.interpolate(src, size=int(spatial_tokens), mode="nearest").reshape(-1)
            else:
                mask[token_start:token_end] = True
                continue
        mask[token_start:token_end] = flat.to(device=device, dtype=torch.bool)
    return mask


def new_scene_bounds_from_segments(
    segment_bounds: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    return [tuple(bound) for bound in segment_bounds[1:]]
