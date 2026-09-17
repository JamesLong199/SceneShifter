"""F11: live ref-attn mask postprocess on synthetic or model scores."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def _connected_components_4(binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """4-connected labels and areas. Label 0 is background."""
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    areas: list[int] = [0]
    next_label = 1
    for y in range(h):
        for x in range(w):
            if binary[y, x] == 0 or labels[y, x] != 0:
                continue
            stack = [(y, x)]
            labels[y, x] = next_label
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = next_label
                        stack.append((ny, nx))
            areas.append(area)
            next_label += 1
    return labels, np.asarray(areas, dtype=np.int32)


def remove_small_components(mask: np.ndarray, minimum: int) -> np.ndarray:
    if minimum <= 0:
        return mask.astype(np.uint8)
    labels, areas = _connected_components_4(mask.astype(np.uint8))
    keep = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, len(areas)):
        if int(areas[label]) >= int(minimum):
            keep[labels == label] = 1
    return keep


def fill_small_holes(mask: np.ndarray, holes: int) -> np.ndarray:
    if holes <= 0:
        return mask.astype(np.uint8)
    inv = (1 - mask.astype(np.uint8))
    labels, areas = _connected_components_4(inv)
    out = mask.astype(np.uint8).copy()
    border = set(
        np.unique(
            np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
        ).tolist()
    )
    for label in range(1, len(areas)):
        if label not in border and int(areas[label]) <= int(holes):
            out[labels == label] = 1
    return out


def postprocess_live_fg(
    scores: torch.Tensor,
    *,
    drop_low_percent: float = 20.0,
    min_component_size: int = 20,
    hole_size: int = 0,
) -> torch.Tensor:
    """Threshold + 4-connected cleanup.

    ``scores`` is ``[F, H, W]`` or ``[F, HW]`` FG confidence in ``[0, 1]``.
    Returns a boolean FG mask with the same leading frame count, flattened to
    ``[F, HW]``.
    """
    value = scores.detach().float().cpu()
    if value.ndim == 2:
        # Infer a near-square spatial layout for component analysis.
        spatial = int(value.shape[-1])
        height = int(np.sqrt(spatial))
        while spatial % height != 0:
            height -= 1
        value = value.reshape(value.shape[0], height, spatial // height)
    if value.ndim != 3:
        raise ValueError(f"expected scores [F,H,W] or [F,HW], got {tuple(scores.shape)}")
    drop = float(drop_low_percent)
    frames = []
    for frame in value:
        flat = frame.reshape(-1)
        if drop > 0:
            thresh = torch.quantile(flat, drop / 100.0)
            binary = (frame >= thresh).numpy().astype(np.uint8)
        else:
            binary = (frame > 0).numpy().astype(np.uint8)
        binary = remove_small_components(binary, int(min_component_size))
        binary = fill_small_holes(binary, int(hole_size))
        frames.append(torch.from_numpy(binary.reshape(-1) > 0))
    return torch.stack(frames, dim=0)


def live_bg_from_subject_fg(subject_fg: torch.Tensor) -> torch.Tensor:
    """``subject_fg`` is ``[S, F, HW]``. Live BG is the complement of the union."""
    return ~subject_fg.any(0)
