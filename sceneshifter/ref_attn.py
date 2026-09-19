"""F11: live ref-attn mask postprocess on synthetic or model scores."""

from __future__ import annotations

from typing import Optional, Tuple

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


def get_attn_map_with_target(
    visual_q: torch.Tensor,
    ref_k: torch.Tensor,
    shape,
    ref_target_masks: torch.Tensor,
    head_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Official MultiTalk ref-loc score: softmax over first-frame keys, class-mean.

    ``visual_q`` / ``ref_k`` are ``[B, M, H, D]``. ``shape`` is ``(F, H, W)``.
    ``ref_target_masks`` is ``[C, H*W]`` (subjects + overall BG). Returns
    ``[C, B*seq]`` with B=1 → ``[C, seq]``, or ``None`` if no heads are selected.
    """
    if torch.is_tensor(shape):
        n_t, n_h, n_w = (int(shape[0]), int(shape[1]), int(shape[2]))
    else:
        n_t, n_h, n_w = (int(v) for v in shape)
    del n_t
    spatial = int(n_h * n_w)
    ref_k = ref_k[:, :spatial].to(device=visual_q.device, dtype=visual_q.dtype)
    scale = 1.0 / float(visual_q.shape[-1]) ** 0.5
    query = visual_q * scale
    query = query.transpose(1, 2)
    key = ref_k.transpose(1, 2)
    probs = (query @ key.transpose(-2, -1)).softmax(dim=-1)
    targets = ref_target_masks.to(device=visual_q.device, dtype=visual_q.dtype)
    maps = []
    selected = None
    if head_mask is not None:
        selected = (head_mask.to(device=visual_q.device) > 0).to(visual_q.dtype)
        if float(selected.sum().item()) <= 0:
            return None
    for mask in targets:
        weighted = probs * mask.view(1, 1, 1, -1)
        per_head = weighted.sum(-1) / mask.sum().clamp_min(1.0)
        per_head = per_head.permute(0, 2, 1)
        if selected is not None:
            maps.append((per_head * selected.view(1, 1, -1)).sum(-1) / selected.sum())
        else:
            maps.append(per_head.mean(-1))
    return torch.cat(maps, dim=0)


def get_attn_map_with_target_chunked(
    visual_q: torch.Tensor,
    ref_k: torch.Tensor,
    shape,
    ref_target_masks: torch.Tensor,
    *,
    chunk_size: int = 256,
) -> Optional[torch.Tensor]:
    """Wan S2V/Animate selector scores with native FP32 chunked einsum."""
    if torch.is_tensor(shape):
        frames, height, width = (int(shape[0]), int(shape[1]), int(shape[2]))
    else:
        frames, height, width = (int(value) for value in shape)
    spatial = height * width
    sequence = frames * spatial
    if visual_q.shape[0] != 1 or visual_q.shape[1] < sequence:
        return None
    query = visual_q[:, :sequence].float()
    key = ref_k[:, :spatial].to(device=query.device).float()
    targets = ref_target_masks.to(device=query.device, dtype=torch.float32)
    denoms = targets.sum(-1).clamp_min(1.0)
    scale = query.shape[-1] ** -0.5
    chunks = []
    for start in range(0, sequence, int(chunk_size)):
        current = query[:, start : start + int(chunk_size)]
        logits = torch.einsum("bqhd,bkhd->bhqk", current, key) * scale
        probs = logits.softmax(-1)
        classes = []
        for class_idx, target in enumerate(targets):
            per_head = (probs * target.view(1, 1, 1, -1)).sum(-1)
            per_head = per_head / denoms[class_idx]
            classes.append(per_head.mean(1)[0])
        chunks.append(torch.stack(classes, dim=0))
    return torch.cat(chunks, dim=1) if chunks else None


def get_attn_map_with_target_per_head(
    visual_q: torch.Tensor,
    ref_k: torch.Tensor,
    shape,
    ref_target_masks: torch.Tensor,
    head_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Per-head ref-loc maps ``[C, seq, heads]``; unselected heads are NaN."""
    if torch.is_tensor(shape):
        _n_t, n_h, n_w = (int(shape[0]), int(shape[1]), int(shape[2]))
    else:
        _n_t, n_h, n_w = (int(v) for v in shape)
    spatial = int(n_h * n_w)
    ref_k = ref_k[:, :spatial].to(device=visual_q.device, dtype=visual_q.dtype)
    scale = 1.0 / float(visual_q.shape[-1]) ** 0.5
    query = (visual_q * scale).transpose(1, 2)
    key = ref_k.transpose(1, 2)
    probs = (query @ key.transpose(-2, -1)).softmax(dim=-1)
    targets = ref_target_masks.to(device=visual_q.device, dtype=visual_q.dtype)
    heads = int(visual_q.shape[2])
    maps = []
    keep = None
    if head_mask is not None:
        keep = (head_mask.to(device=visual_q.device) > 0)
        if int(keep.sum().item()) <= 0:
            return None
    for mask in targets:
        weighted = probs * mask.view(1, 1, 1, -1)
        per_head = weighted.sum(-1) / mask.sum().clamp_min(1.0)
        per_head = per_head.permute(0, 2, 1)
        if keep is not None:
            nan_fill = torch.full_like(per_head, float("nan"))
            per_head = torch.where(keep.view(1, 1, -1), per_head, nan_fill)
        maps.append(per_head[0] if per_head.shape[0] == 1 else per_head.reshape(-1, heads))
    if not maps:
        return None
    stacked = torch.stack(maps, dim=0)
    return stacked


def derive_live_fg_from_class_scores(
    fg_maps: torch.Tensor,
    bg_map: torch.Tensor,
    spatial_h: int,
    spatial_w: int,
    *,
    drop_low_percent: float = 20.0,
    min_component_size: int = 20,
    hole_size: int = 0,
) -> Optional[torch.Tensor]:
    """Official live mask: per-token argmax over [subjects..., BG], then drop+CC.

    ``fg_maps`` is ``[S, F, L]``, ``bg_map`` is ``[F, L]``. Returns bool
    ``[S, F, L]`` or ``None`` if shapes do not match the spatial grid.
    """
    fg = fg_maps.detach().float().cpu()
    bg = bg_map.detach().float().cpu()
    if fg.ndim == 2:
        fg = fg.unsqueeze(1)
    if bg.ndim == 1:
        bg = bg.unsqueeze(0)
    if fg.ndim != 3 or bg.ndim != 2:
        return None
    num_subjects, num_frames, spatial = [int(v) for v in fg.shape]
    if tuple(bg.shape) != (num_frames, spatial):
        return None
    if spatial != int(spatial_h) * int(spatial_w) or num_frames <= 0 or num_subjects <= 0:
        return None
    drop = float(drop_low_percent)
    out = torch.zeros(num_subjects, num_frames, spatial, dtype=torch.bool)
    for frame_idx in range(num_frames):
        class_scores = torch.cat([fg[:, frame_idx], bg[frame_idx : frame_idx + 1]], dim=0)
        labels = torch.argmax(class_scores, dim=0)
        sorted_scores, _ = torch.sort(class_scores, dim=0)
        confidence = sorted_scores[-1] - (sorted_scores[-2] if class_scores.shape[0] >= 2 else torch.zeros_like(sorted_scores[-1]))
        proposed = []
        for subject_idx in range(num_subjects):
            assigned = labels == subject_idx
            if drop > 0:
                conf_vals = confidence[assigned]
                if conf_vals.numel() > 0:
                    thresh = torch.quantile(conf_vals, drop / 100.0)
                    assigned = assigned & (confidence > thresh)
            binary = assigned.numpy().astype(np.uint8).reshape(int(spatial_h), int(spatial_w))
            binary = remove_small_components(binary, int(min_component_size))
            binary = fill_small_holes(binary, int(hole_size))
            proposed.append(binary)
        stacked = np.stack(proposed, axis=0)
        overlap = stacked.sum(axis=0)
        if np.any(overlap > 1):
            winning = torch.argmax(fg[:, frame_idx], dim=0).numpy().reshape(int(spatial_h), int(spatial_w))
            ys, xs = np.where(overlap > 1)
            for y, x in zip(ys, xs):
                stacked[:, y, x] = 0
                stacked[int(winning[y, x]), y, x] = 1
        out[:, frame_idx] = torch.from_numpy(stacked.reshape(num_subjects, -1) > 0)
    return out


def derive_live_fg_from_class_scores_numpy(
    fg_maps: torch.Tensor,
    bg_map: torch.Tensor,
    spatial_h: int,
    spatial_w: int,
    *,
    drop_low_percent: float = 20.0,
    min_component_size: int = 20,
    hole_size: int = 0,
) -> Optional[torch.Tensor]:
    """MultiTalk's exact NumPy argmax/percentile live-mask implementation."""
    fg = fg_maps.detach().float().cpu().numpy()
    bg = bg_map.detach().float().cpu().numpy()
    if fg.ndim == 2:
        fg = fg[:, None, :]
    if bg.ndim == 1:
        bg = bg[None, :]
    if fg.ndim != 3 or bg.ndim != 2:
        return None
    subjects, frames, spatial = fg.shape
    if bg.shape != (frames, spatial):
        return None
    if spatial != int(spatial_h) * int(spatial_w):
        return None
    result = np.zeros((subjects, frames, spatial), dtype=np.bool_)
    for frame in range(frames):
        classes = np.concatenate([fg[:, frame], bg[None, frame]], axis=0)
        labels = np.argmax(classes, axis=0)
        sorted_scores = np.sort(classes, axis=0)
        confidence = sorted_scores[-1] - sorted_scores[-2]
        proposed = np.zeros(
            (subjects, int(spatial_h), int(spatial_w)), dtype=np.uint8
        )
        for subject in range(subjects):
            assigned = labels == subject
            if float(drop_low_percent) > 0 and assigned.any():
                threshold = np.percentile(
                    confidence[assigned], float(drop_low_percent)
                )
                assigned &= confidence > threshold
            mask = assigned.reshape(int(spatial_h), int(spatial_w)).astype(np.uint8)
            mask = remove_small_components(mask, int(min_component_size))
            mask = fill_small_holes(mask, int(hole_size))
            proposed[subject] = mask
        overlap = proposed.sum(axis=0)
        if np.any(overlap > 1):
            winners = np.argmax(
                fg[:, frame].reshape(subjects, int(spatial_h), int(spatial_w)),
                axis=0,
            )
            for y, x in zip(*np.where(overlap > 1)):
                proposed[:, y, x] = 0
                proposed[int(winners[y, x]), y, x] = 1
        result[:, frame] = proposed.reshape(subjects, spatial) > 0
    return torch.from_numpy(result)


def derive_live_fg_from_union_scores(
    fg_map: torch.Tensor,
    bg_map: torch.Tensor,
    spatial_h: int,
    spatial_w: int,
    *,
    drop_low_percent: float = 20.0,
    min_component_size: int = 20,
    hole_size: int = 0,
) -> Optional[torch.Tensor]:
    """FantasyTalking live mask: union-FG score minus BG score, then cleanup."""
    fg = fg_map.detach().float().cpu().numpy()
    bg = bg_map.detach().float().cpu().numpy()
    if fg.ndim == 1:
        fg = fg[None]
    if bg.ndim == 1:
        bg = bg[None]
    if fg.shape != bg.shape or fg.ndim != 2:
        return None
    frames, spatial = fg.shape
    if spatial != int(spatial_h) * int(spatial_w):
        return None
    output = []
    for frame in range(frames):
        diff = fg[frame] - bg[frame]
        positive = diff > 0
        if positive.any() and float(drop_low_percent) > 0:
            threshold = np.percentile(diff[positive], float(drop_low_percent))
            mask = ((diff > 0) & (diff > threshold)).astype(np.uint8)
        else:
            mask = positive.astype(np.uint8)
        mask = mask.reshape(int(spatial_h), int(spatial_w))
        mask = remove_small_components(mask, int(min_component_size))
        mask = fill_small_holes(mask, int(hole_size))
        output.append(torch.from_numpy(mask.reshape(-1) > 0))
    return torch.stack(output, dim=0).unsqueeze(0)
