"""Shared runtime wrapping for official (or toy) attention modules.

Official kernels stay in charge of QKV, RoPE, and flash/sage. SceneShifter only
intercepts the attention call to apply the mask / Q-scale / len_trans contract.
"""

from __future__ import annotations

import contextvars
import types
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..attention import gather_attention, scaled_dot_product
from ..layouts import S2VTokenLayout
from ..parity.trace import trace_tensor
from ..ref_attn import (
    get_attn_map_with_target,
    get_attn_map_with_target_per_head_chunked,
)
from ..router import FramePromptRouter

_CURRENT_MODULE: contextvars.ContextVar[Optional[nn.Module]] = contextvars.ContextVar(
    "sceneshifter_current_attn", default=None
)

_FLASH_NAMES = ("flash_attention", "attention", "sageattn")


def _fun_camera_serial_pass(module: nn.Module, timestep_idx: int) -> str:
    """Official Fun CFG serialization calls each module cond, then uncond."""
    step = int(timestep_idx)
    if getattr(module, "_ss_fun_pass_step", None) != step:
        module._ss_fun_pass_step = step
        module._ss_fun_pass_calls = 0
    calls = int(getattr(module, "_ss_fun_pass_calls", 0))
    module._ss_fun_pass_calls = calls + 1
    return "cond" if calls % 2 == 0 else "uncond"


def find_attention_modules(
    model: Any,
    *,
    self_names=("self_attn", "attn", "attention"),
    cross_names=("cross_attn",),
) -> tuple[List[Any], List[Any]]:
    self_mods: List[Any] = []
    cross_mods: List[Any] = []
    if not hasattr(model, "named_modules"):
        return self_mods, cross_mods
    for name, module in model.named_modules():
        tail = name.rsplit(".", 1)[-1]
        if tail in self_names or (
            tail not in cross_names
            and type(module).__name__
            in {
                "WanSelfAttention",
                "WanS2VSelfAttention",
                "WanAnimateSelfAttention",
                "SelfAttentionLayer",
            }
        ):
            self_mods.append(module)
        elif tail in cross_names or type(module).__name__ in {
            "WanT2VCrossAttention",
            "WanI2VCrossAttention",
            "WanCrossAttention",
            "WanAnimateCrossAttention",
            "CrossAttentionLayer",
        }:
            cross_mods.append(module)
    return list(dict.fromkeys(self_mods)), list(dict.fromkeys(cross_mods))


def find_audio_modules(model: Any) -> List[Any]:
    mods: List[Any] = []
    if not hasattr(model, "named_modules"):
        return mods
    for name, module in model.named_modules():
        tail = name.rsplit(".", 1)[-1]
        if tail == "audio_cross_attn" or type(module).__name__ == "SingleStreamMutiAttention":
            mods.append(module)
    return list(dict.fromkeys(mods))


def wrap_audio_attention(module: nn.Module, router: Optional[FramePromptRouter]) -> nn.Module:
    """Official main always adds audio residual; feature-branch skips it before start."""
    if getattr(module, "_ss_audio_wrapped", False):
        return module
    original = module.forward
    module.frame_prompt_router = router

    def forward(self, x, *args, **kwargs):
        active = getattr(self, "frame_prompt_router", None)
        timestep_idx = int(getattr(self, "_ss_timestep", 0) or 0)
        start = int(getattr(active, "audio_start_timestep", 0) or 0) if active is not None else 0
        if active is not None and int(timestep_idx) < start:
            return torch.zeros_like(x)
        return original(x, *args, **kwargs)

    module.forward = types.MethodType(forward, module)
    module._ss_audio_wrapped = True
    return module


def _project_qkv(module: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_lin = getattr(module, "q", None) or getattr(module, "to_q", None)
    k_lin = getattr(module, "k", None) or getattr(module, "to_k", None)
    v_lin = getattr(module, "v", None) or getattr(module, "to_v", None)
    if q_lin is None or k_lin is None or v_lin is None:
        raise TypeError(f"{type(module).__name__} has no q/k/v projections to wrap")
    q, k, v = q_lin(x), k_lin(x), v_lin(x)
    heads = int(getattr(module, "num_heads", 1))
    dim = q.shape[-1]
    head_dim = dim // heads

    def reshape(t):
        return t.view(t.shape[0], t.shape[1], heads, head_dim)

    q, k, v = reshape(q), reshape(k), reshape(v)
    if hasattr(module, "norm_q"):
        q = module.norm_q(q.flatten(2)).view_as(q) if getattr(module.norm_q, "normalized_shape", None) else q
    return q, k, v


def _maybe_update_spatial(router: FramePromptRouter, module: nn.Module, q: torch.Tensor) -> None:
    grid = getattr(module, "_ss_grid_sizes", None)
    if grid is None:
        return
    try:
        _frames, gh, gw = _grid_fhw(grid, router)
        spatial = gh * gw
        if spatial <= 0 or spatial == router.spatial_tokens:
            return
        old = int(router.spatial_tokens)
        router.patch_height = gh
        router.patch_width = gw
        router.spatial_tokens = spatial
        raw = getattr(router, "raw_subject_fg", None)
        if raw is not None:
            from ..masks import downsample_subject_fg

            router.subject_fg_masks, router.static_bg = downsample_subject_fg(
                raw,
                gh,
                gw,
                center_crop=getattr(router, "backend", None)
                in {"wan_s2v", "wan_animate"},
                cover_original=bool(
                    (getattr(router, "config", {}) or {}).get(
                        "input_mask_cover_original", False
                    )
                ),
            )
            router.live_bg = None
            router.live_subject_fg = None
            return
        for attr in ("static_bg", "live_bg"):
            mask = getattr(router, attr, None)
            if mask is None or mask.numel() == 0 or mask.shape[-1] != old:
                continue
            frames = int(mask.shape[0])
            src = mask.reshape(frames, 1, old).float()
            resized = torch.nn.functional.interpolate(src, size=spatial, mode="nearest")
            setattr(router, attr, resized.reshape(frames, spatial) > 0.5)
        subjects = getattr(router, "subject_fg_masks", None)
        if subjects is not None and subjects.numel() and subjects.shape[-1] == old:
            n_subj, frames = int(subjects.shape[0]), int(subjects.shape[1])
            src = subjects.reshape(n_subj * frames, 1, old).float()
            resized = torch.nn.functional.interpolate(src, size=spatial, mode="nearest")
            router.subject_fg_masks = resized.reshape(n_subj, frames, spatial) > 0.5
        live_fg = getattr(router, "live_subject_fg", None)
        if live_fg is not None and live_fg.numel() and live_fg.shape[-1] == old:
            n_subj, frames = int(live_fg.shape[0]), int(live_fg.shape[1])
            src = live_fg.reshape(n_subj * frames, 1, old).float()
            resized = torch.nn.functional.interpolate(src, size=spatial, mode="nearest")
            router.live_subject_fg = resized.reshape(n_subj, frames, spatial) > 0.5
    except Exception:
        if getattr(router, "backend", None) == "wan_s2v":
            # S2V has reference/FramePack extras; q_len // noisy_frames is not
            # a valid spatial size.
            return
        if q.ndim >= 2 and router.num_frames > 0:
            guessed = int(q.shape[1]) // int(router.num_frames)
            if guessed > 0:
                router.spatial_tokens = guessed


def _extract_qkv(args: Tuple[Any, ...], kwargs: Dict[str, Any]):
    q = kwargs["q"] if "q" in kwargs else (args[0] if len(args) > 0 else None)
    k = kwargs["k"] if "k" in kwargs else (args[1] if len(args) > 1 else None)
    v = kwargs["v"] if "v" in kwargs else (args[2] if len(args) > 2 else None)
    if q is None or k is None or v is None:
        raise TypeError("attention kernel did not receive q/k/v")
    used_kw = "q" in kwargs and "k" in kwargs and "v" in kwargs
    return q, k, v, used_kw


def _grid_fhw(grid: Any, router: FramePromptRouter) -> Tuple[int, int, int]:
    """Resolve (F,H,W), including S2V's nested zero/noisy/reference grids."""
    fallback = (int(router.num_frames), int(router.patch_height), int(router.patch_width))
    if grid is None:
        return fallback

    def triples(value: Any) -> List[Tuple[int, int, int]]:
        if torch.is_tensor(value):
            flat = value.detach().reshape(-1)
            if int(flat.numel()) < 3:
                return []
            usable = int(flat.numel()) - int(flat.numel()) % 3
            return [
                (int(row[0].item()), int(row[1].item()), int(row[2].item()))
                for row in flat[:usable].reshape(-1, 3)
            ]
        if isinstance(value, (list, tuple)):
            found: List[Tuple[int, int, int]] = []
            for item in value:
                found.extend(triples(item))
            return found
        return []

    try:
        candidates = [
            item for item in triples(grid) if item[0] > 0 and item[1] > 0 and item[2] > 0
        ]
        if not candidates:
            return fallback
        for item in candidates:
            if (
                item[0] == int(router.num_frames)
                and item[1] * item[2] == int(router.spatial_tokens)
            ):
                return item
        for item in candidates:
            if item[0] == int(router.num_frames):
                return item
        return candidates[0]
    except Exception:
        return fallback


def _projector_dims(module: nn.Module) -> Tuple[Optional[int], Optional[int]]:
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if not linears:
        return None, None
    return int(linears[0].in_features), int(linears[-1].out_features)


def attach_text_projector(router: Optional[FramePromptRouter], model: Any) -> None:
    """Remember official ``text_embedding`` so raw T5 (4096-d) can be projected."""
    if router is None or model is None:
        return
    proj = getattr(model, "text_embedding", None)
    text_len = getattr(model, "text_len", None)
    if proj is None:
        inner = getattr(model, "model", None) or getattr(model, "dit", None)
        if inner is not None:
            proj = getattr(inner, "text_embedding", None)
            text_len = getattr(inner, "text_len", text_len)
    if proj is None:
        return
    router._text_embedding = proj
    router._text_len = int(text_len or getattr(router, "_text_len", 512) or 512)
    project_router_text_embeddings(router)


def project_router_text_embeddings(router: Optional[FramePromptRouter]) -> None:
    """Pad raw T5 tokens to ``text_len`` and run official ``text_embedding`` (4096→5120)."""
    if router is None or not getattr(router, "text_embeddings", None):
        return
    projector = getattr(router, "_text_embedding", None)
    if projector is None:
        return
    text_len = int(getattr(router, "_text_len", 512) or 512)
    in_dim, out_dim = _projector_dims(projector)
    projected: List[torch.Tensor] = []
    changed = False
    try:
        proj_param = next(projector.parameters())
        proj_device, proj_dtype = proj_param.device, proj_param.dtype
    except StopIteration:
        proj_device, proj_dtype = torch.device("cpu"), torch.float32
    for emb in router.text_embeddings:
        if not torch.is_tensor(emb):
            projected.append(emb)
            continue
        core = emb[0] if emb.ndim == 3 and emb.shape[0] == 1 else emb
        if core.ndim != 2:
            projected.append(emb)
            continue
        if out_dim is not None and int(core.shape[-1]) == out_dim:
            projected.append(emb if emb.ndim == 3 else emb.unsqueeze(0))
            continue
        if in_dim is not None and int(core.shape[-1]) != in_dim:
            projected.append(emb)
            continue
        if core.shape[0] < text_len:
            core = torch.cat([core, core.new_zeros(text_len - core.shape[0], core.shape[1])], dim=0)
        else:
            core = core[:text_len]
        out = projector(core.unsqueeze(0).to(device=proj_device, dtype=proj_dtype))
        projected.append(out)
        changed = True
    if changed:
        router.text_embeddings = projected


def _call_official_flash(
    orig: Callable,
    name: str,
    q,
    k,
    v,
    args,
    kwargs,
    used_kw: bool,
    *,
    subset: bool,
):
    call_kwargs = dict(kwargs)
    call_kwargs.pop("q", None)
    call_kwargs.pop("k", None)
    call_kwargs.pop("v", None)
    if subset:
        # Subset tensors are dense; original packed lengths no longer apply.
        call_kwargs.pop("q_lens", None)
        call_kwargs.pop("k_lens", None)
    if used_kw:
        return orig(q=q, k=k, v=v, **call_kwargs)
    return orig(q, k, v, **call_kwargs)


def _install_rope_stash(globals_dict: Dict[str, Any]) -> bool:
    if globals_dict.get("_ss_rope_patched"):
        return True
    orig = globals_dict.get("rope_apply")
    if not callable(orig):
        return False

    def wrapped(x, grid_sizes, freqs, *args, **kwargs):
        module = _CURRENT_MODULE.get()
        if module is not None:
            count = int(getattr(module, "_ss_rope_n", 0))
            if count == 0:
                module._ss_q_pre_rope = x
            elif count == 1:
                module._ss_k_pre_rope = x
            module._ss_rope_n = count + 1
            module._ss_freqs = freqs
            module._ss_grid_sizes = grid_sizes
            module._ss_rope_fn = orig
        return orig(x, grid_sizes, freqs, *args, **kwargs)

    globals_dict["rope_apply"] = wrapped
    globals_dict["_ss_rope_patched"] = True
    return True


def _should_local_rerope(
    router: FramePromptRouter,
    block: Dict[str, Any],
    segment: int,
    region: Optional[str],
) -> bool:
    """Use the native branch's local-RoPE policy for this backend."""
    if int(segment) <= 0:
        return False
    backend = str(getattr(router, "backend", ""))
    # These native ports keep the model's global post-RoPE queries in their
    # scene-mask kernels. Applying MultiTalk's policy changes every routed step.
    if backend in {"infinitetalk", "fantasytalking", "wan_s2v", "wan_animate"}:
        return False
    # MultiTalk and VideoX-Fun use the same "auto" policy: local only when the
    # query row cannot see scene-0 foreground/full keys.
    try:
        row = router.attention_row(block, int(segment), region)
    except ValueError:
        try:
            row = router.attention_row(block, int(segment), "fg")
        except Exception:
            return False
    for key_seg, (active, key_region) in enumerate(row):
        if active and int(key_seg) == 0 and str(key_region) in ("fg", "full"):
            return False
    return True


def _fun_camera_local_rope(
    x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """VideoX-Fun SceneShifter local RoPE uses float64 complex multiplication."""
    sequence, heads, complex_dim = x.size(1), x.size(2), x.size(3) // 2
    split = [
        complex_dim - 2 * (complex_dim // 3),
        complex_dim // 3,
        complex_dim // 3,
    ]
    freq_parts = freqs.split(split, dim=1)
    output = []
    for row, (frames, height, width) in enumerate(grid_sizes.tolist()):
        frames, height, width = int(frames), int(height), int(width)
        seq_len = frames * height * width
        value = torch.view_as_complex(
            x[row, :sequence].to(torch.float64).reshape(
                sequence, heads, -1, 2
            )
        )
        freq = torch.cat(
            [
                freq_parts[0][:frames]
                .view(frames, 1, 1, -1)
                .expand(frames, height, width, -1),
                freq_parts[1][:height]
                .view(1, height, 1, -1)
                .expand(frames, height, width, -1),
                freq_parts[2][:width]
                .view(1, 1, width, -1)
                .expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        value = torch.view_as_real(value * freq.to(device=value.device)).flatten(2)
        output.append(torch.cat([value, x[row, seq_len:]]))
    return torch.stack(output).float()


def _rerope_queries(
    module: nn.Module,
    router: FramePromptRouter,
    block: Dict[str, Any],
    q: torch.Tensor,
    q_idx: torch.Tensor,
    segment: int,
) -> torch.Tensor:
    rope_fn = getattr(module, "_ss_rope_fn", None)
    q_pre = getattr(module, "_ss_q_pre_rope", None)
    freqs = getattr(module, "_ss_freqs", None)
    grid = getattr(module, "_ss_grid_sizes", None)
    if rope_fn is None or q_pre is None or freqs is None or grid is None:
        return q.index_select(1, q_idx)
    if not torch.is_tensor(grid):
        # Wan S2V stores grid_sizes as a Python list.
        return q.index_select(1, q_idx)
    start, end = router.segment_bounds[int(segment)]
    spatial = int(router.spatial_tokens)
    sl = slice(start * spatial, end * spatial)
    if q_pre.shape[1] < end * spatial:
        return q.index_select(1, q_idx)
    q_seg_pre = q_pre[:, sl].clone()
    token_mask = router.qscale_token_mask(block)
    local_mask = token_mask[sl]
    if bool(local_mask.any()):
        factor = float(router.qscale_factor(int(segment), block))
        q_seg_pre[:, local_mask] = q_seg_pre[:, local_mask] * factor
    grid_seg = grid.clone()
    grid_seg[:, 0] = int(end - start)
    if getattr(router, "backend", None) == "wan_fun_camera":
        q_local = _fun_camera_local_rope(q_seg_pre, grid_seg, freqs)
    else:
        q_local = rope_fn(q_seg_pre, grid_seg, freqs)
    local_idx = q_idx - int(start) * spatial
    local_idx = local_idx[(local_idx >= 0) & (local_idx < q_local.shape[1])]
    if local_idx.numel() != q_idx.numel():
        return q.index_select(1, q_idx)
    return q_local.index_select(1, local_idx)


def _ref_loc_query_for_scene_mask(
    module: nn.Module,
    router: FramePromptRouter,
    block: Dict[str, Any],
    q: torch.Tensor,
) -> torch.Tensor:
    """Official `_forward_with_scene_mask` can score ref-loc on local-rerope Q.

    Kept for contract tests. Live collection stays on the post-RoPE attention Q:
    using this stitch on 032 worsened decoded mean (6.99 -> 9.28).
    """
    q_ref = q
    replaced = False
    for segment in range(1, int(router.num_segments)):
        try:
            q_idx = _query_indices(router, segment, None, q.device)
        except ValueError:
            continue
        if q_idx.numel() == 0:
            continue
        q_local = _rerope_queries(module, router, block, q, q_idx, segment)
        if int(q_local.shape[1]) != int(q_idx.numel()):
            continue
        if not replaced:
            q_ref = q.clone()
            replaced = True
        q_ref.index_copy_(1, q_idx, q_local.to(dtype=q_ref.dtype))
    return q_ref


def _transition_query_indices(
    router: FramePromptRouter, device: torch.device
) -> Dict[int, torch.Tensor]:
    by_segment: Dict[int, torch.Tensor] = {}
    for window in router.get_transition_context_windows():
        frames = [int(f) for f in window["query_frames"]]
        if not frames:
            continue
        parts = [
            torch.arange(
                frame * router.spatial_tokens,
                (frame + 1) * router.spatial_tokens,
                device=device,
                dtype=torch.long,
            )
            for frame in frames
        ]
        by_segment[int(window["new_segment"])] = torch.cat(parts)
    return by_segment


def _s2v_token_layout(router: FramePromptRouter, sequence_length: int) -> S2VTokenLayout:
    noisy = int(router.num_frames) * int(router.spatial_tokens)
    reference = int(router.spatial_tokens)
    framepack = int(sequence_length) - noisy - reference
    if framepack < 0:
        raise ValueError(
            f"S2V sequence has {sequence_length} tokens, expected at least "
            f"{noisy + reference} for noisy+reference"
        )
    return S2VTokenLayout(
        noisy_frames=int(router.num_frames),
        spatial_tokens=int(router.spatial_tokens),
        framepack_tokens=framepack,
    )


def _s2v_reference_indices(
    router: FramePromptRouter,
    layout: S2VTokenLayout,
    region: Optional[str],
    device: torch.device,
) -> torch.Tensor:
    if region in (None, "full"):
        local = torch.arange(int(router.spatial_tokens), device=device)
    else:
        static_bg = getattr(router, "static_bg", None)
        if static_bg is None:
            raise ValueError("S2V reference fg/bg routing requires a static input mask")
        frame_bg = static_bg[0].to(device=device, dtype=torch.bool)
        local = torch.where(frame_bg if region == "bg" else ~frame_bg)[0]
    return local + int(layout.reference_slice.start)


def _ref_target_masks(router: FramePromptRouter, spatial: int, device, dtype) -> Optional[torch.Tensor]:
    subjects = getattr(router, "subject_fg_masks", None)
    if subjects is None or subjects.numel() == 0:
        return None
    subject = subjects[:, 0].to(device=device, dtype=torch.float32)
    if int(subject.shape[-1]) != int(spatial):
        subject = torch.nn.functional.interpolate(
            subject.unsqueeze(1), size=int(spatial), mode="nearest"
        ).squeeze(1)
    background = (~(subject > 0.5).any(0)).to(dtype=torch.float32).unsqueeze(0)
    return torch.cat([subject.to(dtype=dtype), background.to(dtype=dtype)], dim=0)


def _ref_attn_schedule_params(router: FramePromptRouter) -> Optional[Dict[str, Any]]:
    for block in getattr(router, "schedule", []) or []:
        params = (block or {}).get("params") or {}
        if str(params.get("mask_type", "")) == "ref_attn":
            return params
    return None


def _wanted_ref_heads(params: Dict[str, Any], block_idx: Any, num_heads: int) -> List[int]:
    selectors = params.get("ref_attn_mask_selectors") or []
    wanted = [int(head) for layer, head in selectors if int(layer) == int(block_idx)]
    layer_only = params.get("ref_attn_mask_layer")
    head_only = params.get("ref_attn_mask_head")
    if not wanted and layer_only is not None:
        if int(block_idx) != int(layer_only):
            return []
        return list(range(int(num_heads))) if head_only is None else [int(head_only)]
    if not wanted and not selectors:
        return list(range(int(num_heads)))
    return wanted


def _collect_live_ref_attn(
    router: FramePromptRouter,
    module: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    timestep_idx: int,
    block_idx: Any,
    pass_type: str,
) -> None:
    """Store official get_attn_map scores. Do not apply a live mask here.

    ``freeze_live_mask`` freezes the *applied* mask inside a timestep. Scores
    must still be collected so the next step can rebuild from the previous
    step (Fantasy official ``update_ref_attn_mask_after_timestep``).
    """
    if pass_type == "uncond":
        return
    if (
        bool(getattr(router, "freeze_live_mask", False))
        and getattr(router, "backend", None) != "fantasytalking"
    ):
        return
    max_t = int(getattr(router, "ref_loc_max_timesteps", 0) or 0)
    if max_t > 0 and int(timestep_idx) >= max_t:
        return
    if block_idx is None:
        return
    params = _ref_attn_schedule_params(router)
    if params is None:
        return
    publish_after_cond = (
        getattr(router, "backend", None) in {"infinitetalk", "wan_animate"}
        and pass_type == "cond"
        and int(block_idx)
        == int(getattr(router, "_ss_num_self_layers", 0) or 0) - 1
    )
    wanted_heads = _wanted_ref_heads(params, block_idx, int(q.shape[2]))
    if not wanted_heads:
        if publish_after_cond:
            router.complete_denoising_step()
        return
    spatial = int(router.spatial_tokens)
    video = int(router.num_frames) * spatial
    if q.shape[1] < video or k.shape[1] < spatial:
        raise RuntimeError(
            f"{router.backend} ref-attn selector L{int(block_idx)} has "
            f"q/k lengths {int(q.shape[1])}/{int(k.shape[1])}, expected at least "
            f"{video}/{spatial}"
        )
    # Most Wan ports score in v.dtype; FantasyTalking explicitly promotes
    # selector Q/K to FP32 before matmul.
    v = getattr(module, "_ss_v_for_ref", None)
    score_dtype = (
        torch.float32
        if getattr(router, "backend", None)
        in {"fantasytalking", "wan_s2v"}
        else (v.dtype if torch.is_tensor(v) else q.dtype)
    )
    q_score = q.to(dtype=score_dtype)
    k_score = k.to(dtype=score_dtype)
    if (
        getattr(router, "backend", None) == "wan_fun_camera"
        and int(q_score.shape[0]) > 1
    ):
        # Native Fun Camera collects localization from the conditional CFG row.
        q_score = q_score[-1:]
        k_score = k_score[-1:]
    if getattr(router, "backend", None) == "fantasytalking":
        static_bg = getattr(router, "static_bg", None)
        if static_bg is None:
            targets = None
        else:
            bg_ref = static_bg[0].to(device=q.device, dtype=torch.bool)
            targets = torch.stack(
                [
                    (~bg_ref).to(dtype=score_dtype),
                    bg_ref.to(dtype=score_dtype),
                ],
                dim=0,
            )
    else:
        targets = _ref_target_masks(router, spatial, q.device, score_dtype)
    if targets is None:
        raise RuntimeError(
            f"{router.backend} ref-attn selector L{int(block_idx)} has no target masks"
        )
    grid = getattr(module, "_ss_grid_sizes", None)
    shape = _grid_fhw(grid, router)
    spf = int(shape[1]) * int(shape[2])
    if int(targets.shape[-1]) != spf:
        raise RuntimeError(
            f"{router.backend} ref-attn selector L{int(block_idx)} target spatial "
            f"{int(targets.shape[-1])} != grid spatial {spf} from {shape}"
        )
    q_for_scores = q_score[:, :video]
    k_for_scores = k_score
    if getattr(router, "backend", None) == "wan_s2v":
        layout = _s2v_token_layout(router, int(k_score.shape[1]))
        k_for_scores = k_score[:, layout.reference_slice]
    num_subjects = int(targets.shape[0] - 1)
    valid_heads = []
    for head in wanted_heads:
        if head < 0 or head >= int(q.shape[2]):
            raise RuntimeError(
                f"{router.backend} ref-attn selector L{int(block_idx)}H{int(head)} "
                f"outside {int(q.shape[2])} heads"
            )
        valid_heads.append(int(head))
    per_head_backends = {
        "wan_s2v",
        "wan_fun_camera",
    }
    backend = getattr(router, "backend", None)
    individual_scores = None
    try:
        if backend in {"multitalk", "wan_animate"}:
            individual_scores = {}
            for head in valid_heads:
                individual_scores[head] = get_attn_map_with_target(
                    q_for_scores[:, :, head : head + 1],
                    k_for_scores[:, :, head : head + 1],
                    shape,
                    targets,
                )
            scores = None
            per_head_scores = None
        elif backend in per_head_backends:
            compute_heads = (
                valid_heads
                if backend == "wan_s2v"
                else list(range(int(q_for_scores.shape[2])))
            )
            head_index = torch.tensor(
                compute_heads, dtype=torch.long, device=q.device
            )
            selected_q = q_for_scores.index_select(2, head_index)
            selected_k = k_for_scores.index_select(2, head_index)
            per_head_scores = get_attn_map_with_target_per_head_chunked(
                selected_q,
                selected_k,
                shape,
                targets,
                chunk_size=256,
            )
            scores = None
        else:
            head_index = torch.tensor(
                valid_heads, dtype=torch.long, device=q.device
            )
            scores = get_attn_map_with_target(
                q_for_scores.index_select(2, head_index),
                k_for_scores.index_select(2, head_index),
                shape,
                targets,
            )
            per_head_scores = None
    except RuntimeError as exc:
        raise RuntimeError(
            f"{router.backend} ref-attn selector L{int(block_idx)} heads "
            f"{valid_heads} score collection failed"
        ) from exc
    if scores is None and per_head_scores is None and individual_scores is None:
        raise RuntimeError(
            f"{router.backend} ref-attn selector L{int(block_idx)} heads "
            f"{valid_heads} returned no scores"
        )
    if individual_scores is not None:
        for head, head_scores in individual_scores.items():
            if head_scores is None or int(head_scores.shape[1]) != video:
                raise RuntimeError(
                    f"{router.backend} ref-attn selector L{int(block_idx)}H{head} "
                    f"score shape {None if head_scores is None else tuple(head_scores.shape)} "
                    f"does not match video={video}"
                )
            fg = head_scores[:num_subjects].reshape(
                num_subjects, router.num_frames, spatial
            )
            bg = head_scores[num_subjects].reshape(router.num_frames, spatial)
            router.store_ref_loc_head(
                timestep_idx, int(block_idx), int(head), fg, bg
            )
    elif per_head_scores is not None:
        score_shape = per_head_scores.shape
        if int(score_shape[1]) != video or video % spatial != 0:
            raise RuntimeError(
                f"{router.backend} ref-attn selector L{int(block_idx)} heads "
                f"{valid_heads} score shape {tuple(score_shape)} does not match video={video}"
            )
        positions = {head: idx for idx, head in enumerate(compute_heads)}
        for head in valid_heads:
            position = positions[head]
            fg = per_head_scores[:num_subjects, :, position].reshape(
                num_subjects, router.num_frames, spatial
            )
            bg = per_head_scores[num_subjects, :, position].reshape(
                router.num_frames, spatial
            )
            router.store_ref_loc_head(
                timestep_idx, int(block_idx), int(head), fg, bg
            )
    else:
        if int(scores.shape[1]) != video or video % spatial != 0:
            raise RuntimeError(
                f"{router.backend} ref-attn selector L{int(block_idx)} heads "
                f"{valid_heads} score shape {tuple(scores.shape)} does not match video={video}"
            )
        fg = scores[:num_subjects].reshape(
            num_subjects, router.num_frames, spatial
        )
        bg = scores[num_subjects].reshape(router.num_frames, spatial)
        for head in valid_heads:
            router.store_ref_loc_head(
                timestep_idx, int(block_idx), int(head), fg, bg
            )
    if len(valid_heads) != len(wanted_heads):
        raise RuntimeError(
            f"{router.backend} ref-attn selector L{int(block_idx)} stored "
            f"{len(valid_heads)}/{len(wanted_heads)} heads"
        )
    if publish_after_cond:
        router.complete_denoising_step()


def _query_indices(
    router: FramePromptRouter,
    segment: int,
    region: Optional[str],
    device: torch.device,
) -> torch.Tensor:
    start, end = router.segment_bounds[segment]
    if region in (None, "full"):
        return torch.arange(
            start * router.spatial_tokens,
            end * router.spatial_tokens,
            device=device,
            dtype=torch.long,
        )
    local = router.spatial_indices(segment, region, device)
    return local.to(device=device, dtype=torch.long) + int(start) * int(router.spatial_tokens)


def scene_masked_official_flash(
    orig: Callable,
    name: str,
    module: nn.Module,
    router: FramePromptRouter,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Any:
    q, k, v, used_kw = _extract_qkv(args, kwargs)
    module._ss_v_for_ref = v
    _maybe_update_spatial(router, module, q)
    timestep_idx = int(getattr(module, "_ss_timestep", 0) or 0)
    block_idx = getattr(module, "_ss_block_idx", 0)
    pass_type = str(getattr(module, "_ss_pass", "cond"))
    block = router.active_block(timestep_idx, block_idx, pass_type)
    trace_meta = {"backend": router.backend, "pass_type": pass_type}
    trace_tensor(
        "self_q_post_rope",
        q,
        step=timestep_idx,
        layer=block_idx,
        **trace_meta,
    )
    trace_tensor(
        "self_k_post_rope",
        k,
        step=timestep_idx,
        layer=block_idx,
        **trace_meta,
    )
    trace_tensor(
        "self_v",
        v,
        step=timestep_idx,
        layer=block_idx,
        **trace_meta,
    )
    extra_key_fn = getattr(module, "_ss_extra_key_fn", None)
    if block is None:
        # Keep official packed-sequence flash (k_lens=seq_lens). Do not rewrite q/k/v.
        out = orig(*args, **kwargs)
        # Collect vanilla scores so scheduled steps can build the live mask.
        _collect_live_ref_attn(router, module, q, k, timestep_idx, block_idx, pass_type)
        trace_tensor(
            "self_out",
            out[0] if isinstance(out, tuple) else out,
            step=timestep_idx,
            layer=block_idx,
            routed=False,
            **trace_meta,
        )
        return out
    refresh_idx = int(timestep_idx)
    if (
        bool(getattr(router, "freeze_live_mask", False))
        and getattr(router, "backend", None) == "fantasytalking"
        and refresh_idx > 0
    ):
        # Official Fantasy freezes one selector mask for all passes of step t
        # from scores collected at t-1, then rebuilds after the step.
        refresh_idx -= 1
    router.refresh_live_mask_from_stored(refresh_idx, block)
    trace_tensor(
        "live_bg",
        getattr(router, "live_bg", None),
        step=timestep_idx,
        layer=block_idx,
        **trace_meta,
    )

    q = router.apply_q_scale(q, block)
    ref_loc_q = q
    trace_tensor(
        "self_q_scaled",
        q,
        step=timestep_idx,
        layer=block_idx,
        **trace_meta,
    )
    # Official _forward_with_scene_mask accumulates in v.dtype (bf16), not fp32 Q.
    out = q.new_zeros(q.shape, dtype=v.dtype)
    video_tokens = router.num_frames * router.spatial_tokens
    extra = max(0, int(q.shape[1]) - video_tokens)
    s2v_layout = (
        _s2v_token_layout(router, int(q.shape[1]))
        if getattr(router, "backend", None) == "wan_s2v"
        else None
    )

    def _gather(
        q_idx: torch.Tensor,
        k_idx: torch.Tensor,
        segment: int,
        region: Optional[str] = None,
    ) -> None:
        k_idx = k_idx[(k_idx >= 0) & (k_idx < k.shape[1])]
        if s2v_layout is not None:
            row = router.attention_row(block, int(segment), region)
            allow_ref, ref_region = row[0]
            if allow_ref:
                ref_idx = _s2v_reference_indices(
                    router, s2v_layout, ref_region, k_idx.device
                )
                k_idx = torch.cat([k_idx, ref_idx])
            framepack_idx = torch.arange(
                int(s2v_layout.framepack_slice.start),
                int(s2v_layout.framepack_slice.stop),
                device=k_idx.device,
            )
            if framepack_idx.numel():
                k_idx = torch.cat([k_idx, framepack_idx])
            k_idx = torch.unique(k_idx, sorted=True)
        else:
            if extra_key_fn is not None:
                extra_k = extra_key_fn(router, k, segment)
                if extra_k is not None and extra_k.numel():
                    k_idx = torch.unique(torch.cat([k_idx, extra_k.to(k_idx.device)]), sorted=True)
        if extra > 0 and s2v_layout is None:
            extra_idx = torch.arange(video_tokens, q.shape[1], device=q.device)
            k_idx = torch.unique(torch.cat([k_idx, extra_idx]), sorted=True)
        q_limit = (
            int(s2v_layout.reference_slice.stop)
            if s2v_layout is not None
            else int(video_tokens)
        )
        q_idx = q_idx[(q_idx >= 0) & (q_idx < q_limit)]
        if q_idx.numel() == 0 or k_idx.numel() == 0:
            return
        if _should_local_rerope(router, block, segment, region):
            q_piece = _rerope_queries(module, router, block, q, q_idx, segment)
        else:
            q_piece = q.index_select(1, q_idx)
        piece = _call_official_flash(
            orig,
            name,
            q_piece,
            k.index_select(1, k_idx),
            v.index_select(1, k_idx),
            args,
            kwargs,
            used_kw,
            subset=True,
        )
        if isinstance(piece, tuple):
            piece = piece[0]
        out.index_copy_(1, q_idx, piece.to(dtype=out.dtype))

    for segment in range(router.num_segments):
        row = block["_mask"][segment]
        regions: List[Optional[str]]
        if isinstance(row, dict):
            # Official disables fg/bg Q-split when either index set is empty.
            try:
                fg_q = _query_indices(router, segment, "fg", q.device)
                bg_q = _query_indices(router, segment, "bg", q.device)
            except ValueError:
                fg_q = bg_q = None
            if (
                fg_q is None
                or bg_q is None
                or int(fg_q.numel()) == 0
                or int(bg_q.numel()) == 0
            ):
                regions = [None]
            else:
                regions = ["fg", "bg"]
        else:
            regions = [None]
        for region in regions:
            try:
                q_idx = _query_indices(router, segment, region, q.device)
            except ValueError:
                q_idx = _query_indices(router, segment, None, q.device)
            if s2v_layout is not None and int(segment) == 0:
                q_idx = torch.cat(
                    [
                        q_idx,
                        _s2v_reference_indices(
                            router, s2v_layout, region, q.device
                        ),
                    ]
                )
            k_idx = router.noisy_key_indices(
                block,
                segment,
                region,
                q.device,
                timestep_idx=timestep_idx,
                include_transition=False,
            )
            _gather(q_idx, k_idx, segment, region)
    if router.should_apply_transition_context(timestep_idx):
        trans_q = _transition_query_indices(router, q.device)
        for segment, q_idx in trans_q.items():
            trans_region = None if not isinstance(block["_mask"][segment], dict) else "fg"
            k_idx = router.noisy_key_indices(
                block,
                segment,
                trans_region,
                q.device,
                timestep_idx=timestep_idx,
                include_transition=True,
            )
            if isinstance(block["_mask"][segment], dict):
                k_bg = router.noisy_key_indices(
                    block,
                    segment,
                    "bg",
                    q.device,
                    timestep_idx=timestep_idx,
                    include_transition=True,
                )
                k_idx = torch.unique(torch.cat([k_idx, k_bg]), sorted=True)
            _gather(q_idx, k_idx, segment, trans_region)
    if extra > 0:
        if s2v_layout is not None:
            extra_idx = torch.arange(
                int(s2v_layout.framepack_slice.start),
                int(s2v_layout.framepack_slice.stop),
                device=q.device,
            )
        else:
            extra_idx = torch.arange(video_tokens, q.shape[1], device=q.device)
        if extra_idx.numel() == 0:
            _collect_live_ref_attn(
                router, module, ref_loc_q, k, timestep_idx, block_idx, pass_type
            )
            trace_tensor(
                "self_out",
                out,
                step=timestep_idx,
                layer=block_idx,
                routed=True,
                **trace_meta,
            )
            return out
        piece = _call_official_flash(
            orig,
            name,
            q.index_select(1, extra_idx),
            k,
            v,
            args,
            kwargs,
            used_kw,
            subset=s2v_layout is None,
        )
        if isinstance(piece, tuple):
            piece = piece[0]
        out.index_copy_(1, extra_idx, piece.to(dtype=out.dtype))
    _collect_live_ref_attn(
        router, module, ref_loc_q, k, timestep_idx, block_idx, pass_type
    )
    trace_tensor(
        "self_out",
        out,
        step=timestep_idx,
        layer=block_idx,
        routed=True,
        **trace_meta,
    )
    return out


def _install_flash_intercept(globals_dict: Dict[str, Any]) -> bool:
    if globals_dict.get("_ss_flash_patched"):
        return True
    patched = False
    for name in _FLASH_NAMES:
        orig = globals_dict.get(name)
        if not callable(orig):
            continue

        def _make(orig_fn: Callable, kernel_name: str) -> Callable:
            def wrapped(*args, **kwargs):
                module = _CURRENT_MODULE.get()
                router = getattr(module, "frame_prompt_router", None) if module is not None else None
                if router is None:
                    return orig_fn(*args, **kwargs)
                return scene_masked_official_flash(orig_fn, kernel_name, module, router, args, kwargs)

            return wrapped

        globals_dict[name] = _make(orig, name)
        patched = True
    if patched:
        globals_dict["_ss_flash_patched"] = True
    return patched


def _stash_grid_sizes(module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
    grid = kwargs.get("grid_sizes")
    if grid is None and len(args) >= 3:
        grid = args[2]
    if grid is not None:
        module._ss_grid_sizes = grid
    timestep = kwargs.get("timestep_idx", kwargs.get("t_idx"))
    if timestep is not None:
        module._ss_timestep = timestep
    if "pass_type" in kwargs:
        module._ss_pass = kwargs["pass_type"]


def wrap_self_attention(
    module: nn.Module,
    router: Optional[FramePromptRouter],
    *,
    extra_key_fn: Optional[Callable] = None,
) -> nn.Module:
    original = module.forward
    module.frame_prompt_router = router
    module._ss_extra_key_fn = extra_key_fn
    globals_dict = getattr(original, "__globals__", {})
    patched_official = _install_flash_intercept(globals_dict)
    _install_rope_stash(globals_dict)

    if patched_official:

        def forward(self, *args, **kwargs):
            self._ss_rope_n = 0
            _stash_grid_sizes(self, args, kwargs)
            active_router = getattr(self, "frame_prompt_router", None)
            if (
                active_router is not None
                and getattr(active_router, "backend", None) == "wan_fun_camera"
            ):
                self._ss_pass = _fun_camera_serial_pass(
                    self, int(getattr(self, "_ss_timestep", 0) or 0)
                )
            token = _CURRENT_MODULE.set(self)
            try:
                return original(*args, **kwargs)
            finally:
                _CURRENT_MODULE.reset(token)
                # Animate 1280x720 keeps ~0.8 GiB per pre-RoPE tensor; 40 layers would OOM.
                self._ss_q_pre_rope = None
                self._ss_k_pre_rope = None
                self._ss_freqs = None

        module.forward = types.MethodType(forward, module)
        return module

    def toy_forward(self, x, *args, **kwargs):
        active_router = getattr(self, "frame_prompt_router", None)
        timestep_idx = kwargs.pop("timestep_idx", getattr(self, "_ss_timestep", 0))
        block_idx = kwargs.pop("block_idx", getattr(self, "_ss_block_idx", 0))
        pass_type = kwargs.pop("pass_type", getattr(self, "_ss_pass", "cond"))
        if active_router is None:
            return original(x, *args, **kwargs)
        try:
            q, k, v = _project_qkv(self, x)
        except TypeError:
            return original(x, *args, **kwargs)
        block = active_router.active_block(int(timestep_idx), int(block_idx), str(pass_type))
        q = active_router.apply_q_scale(q, block)
        if block is None:
            out = scaled_dot_product(q, k, v)
        else:
            out = q.new_zeros(q.shape)
            for segment in range(active_router.num_segments):
                start, end = active_router.segment_bounds[segment]
                q_idx = torch.arange(
                    start * active_router.spatial_tokens,
                    end * active_router.spatial_tokens,
                    device=q.device,
                )
                k_idx = active_router.noisy_key_indices(
                    block, segment, None, q.device, timestep_idx=int(timestep_idx)
                )
                if extra_key_fn is not None:
                    extra = extra_key_fn(active_router, k, segment)
                    if extra is not None and extra.numel():
                        k_idx = torch.unique(torch.cat([k_idx, extra]), sorted=True)
                if q_idx.numel() and k_idx.numel():
                    gather_attention(q, k, v, q_idx, k_idx, out)
        flat = out.flatten(2)
        if hasattr(self, "o"):
            flat = self.o(flat)
        elif hasattr(self, "out_proj"):
            flat = self.out_proj(flat)
        return flat

    module.forward = types.MethodType(toy_forward, module)
    return module


def _drop_packed_context_lens(args: Tuple[Any, ...], kwargs: Dict[str, Any]):
    """Official SceneShifter text flash uses k_lens=None, not generate's packed length."""
    kwargs = dict(kwargs)
    kwargs.pop("context_lens", None)
    if args:
        return (None,) + tuple(args[1:]), kwargs
    kwargs["context_lens"] = None
    return args, kwargs


def _clip_plus_text(context: torch.Tensor, text: torch.Tensor) -> Optional[torch.Tensor]:
    """Rebuild official I2V context: CLIP 257 + projected T5."""
    if text.ndim == context.ndim - 1:
        text = text.unsqueeze(0)
    if (
        text.ndim != context.ndim
        or text.shape[0] != context.shape[0]
        or text.shape[-1] != context.shape[-1]
    ):
        return None
    if context.shape[1] > text.shape[1]:
        prefix = context[:, : int(context.shape[1] - text.shape[1])]
        return torch.cat([prefix, text.to(device=context.device, dtype=context.dtype)], dim=1)
    if tuple(text.shape) == tuple(context.shape):
        return text
    return None


def wrap_cross_attention(module: nn.Module, router: Optional[FramePromptRouter]) -> nn.Module:
    original = module.forward
    module.frame_prompt_router = router
    processor = getattr(module, "processor", None)
    # FantasyTalking I2V audio views Q as ``[B*latent_frames, ...]``. Slicing x
    # per scene makes that view invalid; the adapter wraps the processor instead.
    if processor is not None and hasattr(processor, "k_proj"):
        return module

    def forward(self, x, context=None, *args, **kwargs):
        active_router = getattr(self, "frame_prompt_router", None)
        timestep_idx = kwargs.pop("timestep_idx", getattr(self, "_ss_timestep", 0))
        pass_type = kwargs.pop("pass_type", getattr(self, "_ss_pass", "cond"))
        if (
            active_router is not None
            and getattr(active_router, "backend", None) == "wan_fun_camera"
            and int(x.shape[0]) == 1
        ):
            pass_type = _fun_camera_serial_pass(self, int(timestep_idx))
            self._ss_pass = pass_type
        kwargs.pop("block_idx", None)
        if active_router is None or context is None:
            if context is None:
                return original(x, *args, **kwargs)
            return original(x, context, *args, **kwargs)
        if active_router.text_embeddings is None:
            if active_router.should_use_segment_prompts(int(timestep_idx), str(pass_type)):
                raise RuntimeError(
                    f"{active_router.backend} segment prompts are active at step "
                    f"{int(timestep_idx)}, but no segment text embeddings were captured"
                )
            return original(x, context, *args, **kwargs)
        project_router_text_embeddings(active_router)
        talking = getattr(active_router, "backend", "") in {"multitalk", "infinitetalk"}
        # Official MultiTalk disables cross-attn routing on uncond (neg prompt stays).
        # Official-main also sets context_lens=None before flash.
        if talking:
            args, kwargs = _drop_packed_context_lens(args, kwargs)
        if not active_router.should_use_segment_prompts(int(timestep_idx), str(pass_type)):
            return original(x, context, *args, **kwargs)
        if getattr(active_router, "backend", None) == "wan_s2v":
            # Native S2V runs baseline cross-attention over the complete token
            # sequence, then overwrites only noisy scene slices with segment T5.
            routed = original(x, context, *args, **kwargs)
            for segment in range(active_router.num_segments):
                start, end = active_router.segment_bounds[segment]
                embed_idx = active_router.selected_text_index(
                    int(timestep_idx), segment, str(pass_type)
                )
                if embed_idx is None:
                    raise RuntimeError("S2V conditional routing has no text index")
                text = active_router.text_embeddings[embed_idx]
                segment_context = _clip_plus_text(context, text)
                if segment_context is None:
                    raise RuntimeError("S2V segment text does not match official context")
                sl = slice(
                    int(start) * int(active_router.spatial_tokens),
                    int(end) * int(active_router.spatial_tokens),
                )
                routed[:, sl] = original(
                    x[:, sl], segment_context, *args, **kwargs
                )
            return routed
        outs = []
        for segment in range(active_router.num_segments):
            start, end = active_router.segment_bounds[segment]
            embed_idx = active_router.selected_text_index(int(timestep_idx), segment, str(pass_type))
            if embed_idx is None:
                return original(x, context, *args, **kwargs)
            text = active_router.text_embeddings[embed_idx]
            if not torch.is_tensor(text) or not torch.is_tensor(context):
                return original(x, context, *args, **kwargs)
            sl = slice(start * active_router.spatial_tokens, end * active_router.spatial_tokens)
            if (
                getattr(active_router, "backend", None) == "wan_fun_camera"
                and int(x.shape[0]) == 2
                and int(context.shape[0]) == 2
            ):
                # Fun Camera performs CFG in one [uncond, cond] batch. Keep the
                # negative context on row 0 and route only row 1 by segment.
                cond_text = _clip_plus_text(context[1:], text)
                if cond_text is None:
                    raise RuntimeError("Fun Camera segment text does not match cond context")
                uncond_piece = original(x[:1, sl], context[:1], *args, **kwargs)
                cond_piece = original(x[1:, sl], cond_text, *args, **kwargs)
                piece = torch.cat([uncond_piece, cond_piece], dim=0)
            else:
                text = _clip_plus_text(context, text)
                if text is None:
                    return original(x, context, *args, **kwargs)
                piece = original(x[:, sl], text, *args, **kwargs)
            outs.append(piece)
        out = torch.cat(outs, dim=1)
        video = int(active_router.num_frames) * int(active_router.spatial_tokens)
        extra = max(0, int(x.shape[1]) - video)
        if extra > 0:
            try:
                tail = original(x[:, video:], context, *args, **kwargs)
            except TypeError:
                tail = original(x[:, video:], context)
            out = torch.cat([out, tail], dim=1)
        if int(out.shape[1]) != int(x.shape[1]):
            return original(x, context, *args, **kwargs)
        return out

    module.forward = types.MethodType(forward, module)
    return module


def thread_timestep(model: Any) -> None:
    if not hasattr(model, "forward") or getattr(model, "_ss_thread_installed", False):
        return
    original = model.forward
    unseen = object()
    state = {"t_idx": 0, "last_t": unseen, "pass": 0}

    def _t_key(args, kwargs):
        t = kwargs.get("t", kwargs.get("timestep", kwargs.get("timesteps")))
        if t is None and len(args) > 1:
            t = args[1]
        if t is None:
            return None
        if torch.is_tensor(t):
            return float(t.reshape(-1)[0].detach().float().cpu())
        try:
            return float(t)
        except (TypeError, ValueError):
            return None

    def forward(self, *args, **kwargs):
        timestep_idx = kwargs.get("timestep_idx", kwargs.get("t_idx"))
        pass_type = kwargs.get("pass_type")
        t_key = _t_key(args, kwargs)
        stepped = False
        if timestep_idx is None:
            if t_key is not None and t_key != state["last_t"]:
                if state["last_t"] is not unseen:
                    state["t_idx"] += 1
                    stepped = True
                state["last_t"] = t_key
                state["pass"] = 0
            else:
                state["pass"] += 1
            timestep_idx = state["t_idx"]
        router = getattr(self, "frame_prompt_router", None)
        if router is None:
            for module in find_attention_modules(self)[0]:
                router = getattr(module, "frame_prompt_router", None)
                if router is not None:
                    break
        if pass_type is None and router is not None:
            context = kwargs.get("context")
            null = getattr(router, "null_text_embeddings", None)
            ctx0 = context[0] if isinstance(context, (list, tuple)) and context else context
            null0 = null[0] if isinstance(null, (list, tuple)) and null else null
            if null0 is not None and ctx0 is null0:
                pass_type = "uncond"
            else:
                pass_type = "cond"
        if pass_type is None:
            # Official MultiTalk/InfiniteTalk run cond first, then drop-audio / uncond.
            pass_type = "cond" if state["pass"] == 0 else "uncond"
        if router is not None:
            if (
                stepped
                and getattr(router, "backend", None)
                in {"wan_s2v", "wan_fun_camera"}
            ):
                router.complete_denoising_step()
            if (
                stepped
                and bool(getattr(router, "freeze_live_mask", False))
                and getattr(router, "backend", None) == "fantasytalking"
            ):
                # Allow the next step to rebuild from the previous step's scores.
                router.live_bg = None
                router.live_subject_fg = None
            router._ss_timestep = timestep_idx
            router._ss_pass = pass_type
            start = int(getattr(router, "audio_start_timestep", 0) or 0)
            if int(timestep_idx) < start:
                kwargs = dict(kwargs)
                # FantasyTalking official main gates audio with audio_scale, not `audio`.
                if "audio_scale" in kwargs:
                    kwargs["audio_scale"] = 0.0
        kwargs.setdefault("timestep_idx", timestep_idx)
        kwargs.setdefault("pass_type", pass_type)
        self_mods, cross_mods = find_attention_modules(self)
        audio_mods = find_audio_modules(self)
        for idx, module in enumerate(self_mods + cross_mods):
            module._ss_timestep = timestep_idx
            module._ss_pass = pass_type
            module._ss_block_idx = idx if module in self_mods else idx - len(self_mods)
        for module in audio_mods:
            module._ss_timestep = timestep_idx
            module._ss_pass = pass_type
            if router is not None:
                module.frame_prompt_router = router
        # Official Wan/Hunyuan forwards do not accept these bookkeeping kwargs.
        kwargs.pop("timestep_idx", None)
        kwargs.pop("pass_type", None)
        kwargs.pop("t_idx", None)
        kwargs.pop("block_idx", None)
        kwargs.pop("audio_start_timestep", None)
        return original(*args, **kwargs)

    model.forward = types.MethodType(forward, model)
    model._ss_thread_installed = True


class Adapter:
    name = "base"

    def inject(self, model: Any, router: Optional[FramePromptRouter]) -> List[Any]:
        raise NotImplementedError
