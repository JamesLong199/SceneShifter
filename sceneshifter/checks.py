"""Hard-check hooks. Strict by default; set SCENESHIFTER_STRICT=0 to disable."""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Iterable, List, Optional

from .policy import normalize_schedule, schedule_layer_overlaps

STRICT_ENV = "SCENESHIFTER_STRICT"


class SceneShifterCheckError(ValueError):
    """Raised when a SceneShifter invariant fails."""


def strict_enabled() -> bool:
    value = os.environ.get(STRICT_ENV, "1")
    return value not in {"0", "false", "False", "no", "off"}


def _raise(message: str) -> None:
    if strict_enabled():
        raise SceneShifterCheckError(message)
    import warnings

    warnings.warn(message, stacklevel=3)


def warn(message: str) -> None:
    import warnings

    warnings.warn(message, stacklevel=3)


def validate_config(config: Dict[str, Any], *, backend: str = "multitalk") -> None:
    if "enabled" not in config:
        _raise("frame_prompt_routing.enabled is required")
    if not config.get("enabled", False):
        return

    if "transition_segment_indices" in config and config["transition_segment_indices"]:
        _raise("deprecated transition_segment_indices is not empty")

    num_frames = int(config.get("num_frames", 0))
    if num_frames < 1:
        _raise("num_frames must be a positive count of noisy/video latent frames")
    splits = [int(v) for v in config.get("frame_splits", [])]
    if splits != sorted(set(splits)) or any(s <= 0 or s >= num_frames for s in splits):
        _raise("frame_splits must be sorted unique boundaries within (0, num_frames)")
    num_segments = len(splits) + 1
    prompts = config.get("segment_prompts")
    if prompts is None or len(prompts) != num_segments:
        _raise(
            f"segment_prompts must have len(frame_splits)+1 = {num_segments} entries"
        )

    len_trans = int(config.get("len_trans", 0) or 0)
    len_trans_context = int(config.get("len_trans_context", 0) or 0)
    if len_trans < 0 or len_trans_context < 0:
        _raise("len_trans and len_trans_context must be >= 0")
    if bool(len_trans) != bool(len_trans_context):
        _raise("len_trans and len_trans_context must both be positive or both zero")
    raw_range = config.get("trans_timestep_range")
    if raw_range is not None:
        if not len_trans:
            _raise("trans_timestep_range requires transition lengths")
        start, end = int(raw_range[0]), int(raw_range[1])
        if end < start:
            _raise("trans_timestep_range requires START <= END")

    schedule = config.get("attention_control_schedule")
    if schedule is None:
        _raise("attention_control_schedule is required when enabled")
    normalized = normalize_schedule(schedule, num_segments)
    for block in normalized:
        lo, hi = block["timestep_range"]
        if hi < lo:
            _raise(f"invalid timestep_range {[lo, hi]}")
        params = block["params"]
        mask = block["_mask"]
        if len(mask) != num_segments:
            _raise("self_attention_mask must be num_segments x num_segments")
        for row in mask:
            cells = row["fg"] if isinstance(row, dict) else row
            if len(cells) != num_segments:
                _raise("self_attention_mask row width must equal num_segments")
        if backend == "hunyuan_avatar" and not bool(
            params.get("apply_on_uncond_passes", True)
        ):
            _raise("hunyuan_avatar requires apply_on_uncond_passes=true")
        if backend == "wan_s2v" and int(params.get("ref_attn_mask_ext_size", 0) or 0) != 0:
            _raise("wan_s2v requires ref_attn_mask_ext_size=0")
        auto = bool(params.get("auto_q_scale", False))
        bg_only = params.get("q_scale_bg_only", params.get("q_scale_new_scene_bg_only"))
        if bg_only is True and not auto:
            _raise("q_scale_bg_only (F10) is a no-op while auto_q_scale (F9) is off")
        if backend == "hunyuan_avatar" and bg_only is True:
            warn(
                "HunyuanVideo-Avatar is better with q_scale_bg_only=false; "
                "F10=true is allowed but not recommended"
            )
    overlaps = schedule_layer_overlaps(normalized)
    if overlaps:
        _raise("; ".join(overlaps))

    paths = config.get("background_mask_path")
    if paths is not None:
        if not isinstance(paths, dict) or not paths:
            _raise("background_mask_path must be a non-empty subject-index dict")
        keys = sorted(int(k) for k in paths)
        if keys != list(range(len(keys))):
            _raise("background_mask_path keys must be contiguous from 0")
    if bool(config.get("input_mask_cover_head_only", False)):
        head_paths = config.get("head_mask_path")
        if not isinstance(head_paths, dict) or not head_paths:
            _raise("input_mask_cover_head_only=True requires head_mask_path")


def validate_layout(
    *,
    backend: str,
    noisy_frames: int,
    router_frames: int,
    reference_frames: int = 0,
    framepack_tokens: int = 0,
    total_tokens: Optional[int] = None,
    spatial_tokens: Optional[int] = None,
) -> None:
    if noisy_frames != router_frames:
        _raise(
            f"{backend} layout noisy frames {noisy_frames} != router num_frames {router_frames}"
        )
    if backend == "wan_s2v":
        if reference_frames != 1:
            _raise("wan_s2v requires exactly one reference frame")
        if framepack_tokens < 0:
            _raise("framepack_tokens must be >= 0")
        if spatial_tokens is not None and total_tokens is not None:
            expected = router_frames * spatial_tokens + spatial_tokens + framepack_tokens
            if total_tokens != expected:
                _raise(
                    f"wan_s2v total tokens {total_tokens} != noisy+ref+framepack {expected}"
                )
    if backend == "wan_animate" and reference_frames == 0:
        warn("wan_animate num_frames should include the reference slot (frame 0)")


def validate_inject(bound_modules: Iterable[Any], *, expected_min: int = 1) -> None:
    modules = [m for m in bound_modules if getattr(m, "frame_prompt_router", None) is not None]
    if len(modules) < expected_min:
        _raise(
            f"inject_router bound {len(modules)} attention modules; expected >= {expected_min}"
        )


def official_tree_is_clean(path: str) -> None:
    if not os.path.isdir(os.path.join(path, ".git")):
        return
    result = subprocess.run(
        ["git", "-C", path, "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _raise(f"git status failed in {path}: {result.stderr.strip()}")
    if result.stdout.strip():
        _raise(f"official third_party tree is dirty: {path}\n{result.stdout}")
