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
from ..router import FramePromptRouter

_CURRENT_MODULE: contextvars.ContextVar[Optional[nn.Module]] = contextvars.ContextVar(
    "sceneshifter_current_attn", default=None
)

_FLASH_NAMES = ("flash_attention", "attention", "sageattn")


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
        row = grid[0]
        gh, gw = int(row[1]), int(row[2])
        spatial = gh * gw
        if spatial > 0 and spatial != router.spatial_tokens:
            old = int(router.spatial_tokens)
            router.patch_height = gh
            router.patch_width = gw
            router.spatial_tokens = spatial
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
        if q.ndim >= 2 and router.num_frames > 0:
            guessed = int(q.shape[1]) // int(router.num_frames)
            if guessed > 0:
                router.spatial_tokens = guessed


def _extract_qkv(args: Tuple[Any, ...], kwargs: Dict[str, Any]):
    if "q" in kwargs:
        return kwargs["q"], kwargs["k"], kwargs["v"], True
    if len(args) >= 3:
        return args[0], args[1], args[2], False
    raise TypeError("attention kernel did not receive q/k/v")


def _call_official_flash(orig: Callable, name: str, q, k, v, args, kwargs, used_kw: bool):
    call_kwargs = dict(kwargs)
    # Subset tensors are dense; original packed lengths no longer apply.
    call_kwargs.pop("q_lens", None)
    call_kwargs.pop("k_lens", None)
    if used_kw:
        call_kwargs["q"] = q
        call_kwargs["k"] = k
        call_kwargs["v"] = v
        return orig(**call_kwargs)
    return orig(q, k, v, *args[3:], **call_kwargs)


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


def _maybe_update_live_ref_attn(
    router: FramePromptRouter,
    module: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    timestep_idx: int,
    block_idx: Any,
    pass_type: str,
) -> None:
    if pass_type == "uncond" or bool(getattr(router, "freeze_live_mask", False)):
        return
    max_t = int(getattr(router, "ref_loc_max_timesteps", 0) or 0)
    if max_t > 0 and int(timestep_idx) >= max_t:
        return
    block = router.active_block(int(timestep_idx), block_idx, pass_type)
    params = (block or {}).get("params") or {}
    if str(params.get("mask_type", "")) != "ref_attn":
        params = {}
        for scheduled in getattr(router, "schedule", []) or []:
            candidate = scheduled.get("params") or {}
            if str(candidate.get("mask_type", "")) == "ref_attn":
                params = candidate
                break
    if str(params.get("mask_type", "")) != "ref_attn":
        return
    selectors = params.get("ref_attn_mask_selectors") or []
    if block_idx is None:
        return
    wanted_heads = [
        int(head) for layer, head in selectors if int(layer) == int(block_idx)
    ]
    layer_only = params.get("ref_attn_mask_layer")
    head_only = params.get("ref_attn_mask_head")
    if not wanted_heads and layer_only is not None:
        if int(block_idx) != int(layer_only):
            return
        wanted_heads = list(range(int(q.shape[2]))) if head_only is None else [int(head_only)]
    if not wanted_heads and not selectors:
        # Layer-headavg: current layer, all heads.
        wanted_heads = list(range(int(q.shape[2])))
    if not wanted_heads:
        return
    spatial = int(router.spatial_tokens)
    video = int(router.num_frames) * spatial
    if q.shape[1] < video or k.shape[1] < spatial:
        return
    targets = _ref_target_masks(router, spatial, q.device, q.dtype)
    if targets is None:
        return
    scale = 1.0 / float(q.shape[-1]) ** 0.5
    video_q = q[:, :video].float() * scale
    ref_k = k[:, :spatial].float()
    num_subjects = int(targets.shape[0] - 1)
    collected = []
    for head in wanted_heads:
        if head < 0 or head >= int(q.shape[2]):
            continue
        logits = torch.matmul(
            video_q[:, :, head],
            ref_k[:, :, head].transpose(-2, -1),
        )
        probs = logits.softmax(dim=-1)
        denom = targets.sum(dim=-1).clamp_min(1.0).view(1, -1, 1)
        scores = torch.einsum("bqr,cr->bcq", probs, targets.to(dtype=probs.dtype)) / denom
        collected.append(scores[0, :num_subjects])
    if not collected:
        return
    stacked = torch.stack(collected, dim=0).mean(0)
    fg = stacked.view(num_subjects, router.num_frames, spatial)
    router.update_live_mask_from_scores(
        fg,
        drop_low_percent=float(params.get("ref_attn_mask_drop_low_percent", 10.0)),
        min_component_size=int(params.get("ref_attn_mask_min_component_size", 20)),
        hole_size=int(params.get("ref_attn_mask_hole_size", 0)),
    )


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
    _maybe_update_spatial(router, module, q)
    timestep_idx = int(getattr(module, "_ss_timestep", 0) or 0)
    block_idx = getattr(module, "_ss_block_idx", 0)
    pass_type = str(getattr(module, "_ss_pass", "cond"))
    block = router.active_block(timestep_idx, block_idx, pass_type)
    q = router.apply_q_scale(q, block)
    extra_key_fn = getattr(module, "_ss_extra_key_fn", None)
    if block is None:
        out = _call_official_flash(orig, name, q, k, v, args, kwargs, used_kw)
        _maybe_update_live_ref_attn(router, module, q, k, timestep_idx, block_idx, pass_type)
        return out

    out = q.new_zeros(q.shape)
    video_tokens = router.num_frames * router.spatial_tokens
    extra = max(0, int(q.shape[1]) - video_tokens)

    def _gather(q_idx: torch.Tensor, k_idx: torch.Tensor, segment: int) -> None:
        k_idx = k_idx[(k_idx >= 0) & (k_idx < k.shape[1])]
        if extra_key_fn is not None:
            extra_k = extra_key_fn(router, k, segment)
            if extra_k is not None and extra_k.numel():
                k_idx = torch.unique(torch.cat([k_idx, extra_k.to(k_idx.device)]), sorted=True)
        if extra > 0:
            extra_idx = torch.arange(video_tokens, q.shape[1], device=q.device)
            k_idx = torch.unique(torch.cat([k_idx, extra_idx]), sorted=True)
        q_idx = q_idx[(q_idx >= 0) & (q_idx < video_tokens)]
        if q_idx.numel() == 0 or k_idx.numel() == 0:
            return
        piece = _call_official_flash(
            orig,
            name,
            q.index_select(1, q_idx),
            k.index_select(1, k_idx),
            v.index_select(1, k_idx),
            args,
            kwargs,
            used_kw,
        )
        if isinstance(piece, tuple):
            piece = piece[0]
        out.index_copy_(1, q_idx, piece)

    for segment in range(router.num_segments):
        row = block["_mask"][segment]
        regions: List[Optional[str]]
        if isinstance(row, dict):
            regions = ["fg", "bg"]
        else:
            regions = [None]
        for region in regions:
            try:
                q_idx = _query_indices(router, segment, region, q.device)
            except ValueError:
                q_idx = _query_indices(router, segment, None, q.device)
            k_idx = router.noisy_key_indices(
                block,
                segment,
                region,
                q.device,
                timestep_idx=timestep_idx,
                include_transition=False,
            )
            _gather(q_idx, k_idx, segment)
    if router.should_apply_transition_context(timestep_idx):
        trans_q = _transition_query_indices(router, q.device)
        for segment, q_idx in trans_q.items():
            k_idx = router.noisy_key_indices(
                block,
                segment,
                None if not isinstance(block["_mask"][segment], dict) else "fg",
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
            _gather(q_idx, k_idx, segment)
    if extra > 0:
        extra_idx = torch.arange(video_tokens, q.shape[1], device=q.device)
        piece = _call_official_flash(
            orig,
            name,
            q.index_select(1, extra_idx),
            k,
            v,
            args,
            kwargs,
            used_kw,
        )
        if isinstance(piece, tuple):
            piece = piece[0]
        out.index_copy_(1, extra_idx, piece)
    _maybe_update_live_ref_attn(router, module, q, k, timestep_idx, block_idx, pass_type)
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
    patched_official = _install_flash_intercept(getattr(original, "__globals__", {}))

    if patched_official:

        def forward(self, *args, **kwargs):
            _stash_grid_sizes(self, args, kwargs)
            token = _CURRENT_MODULE.set(self)
            try:
                return original(*args, **kwargs)
            finally:
                _CURRENT_MODULE.reset(token)

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


def wrap_cross_attention(module: nn.Module, router: Optional[FramePromptRouter]) -> nn.Module:
    original = module.forward
    module.frame_prompt_router = router

    def forward(self, x, context=None, *args, **kwargs):
        active_router = getattr(self, "frame_prompt_router", None)
        timestep_idx = kwargs.pop("timestep_idx", getattr(self, "_ss_timestep", 0))
        pass_type = kwargs.pop("pass_type", getattr(self, "_ss_pass", "cond"))
        kwargs.pop("block_idx", None)
        if active_router is None or context is None or active_router.text_embeddings is None:
            if context is None:
                return original(x, *args, **kwargs)
            return original(x, context, *args, **kwargs)
        if not active_router.should_use_segment_prompts(int(timestep_idx), str(pass_type)):
            return original(x, context, *args, **kwargs)
        outs = []
        for segment in range(active_router.num_segments):
            start, end = active_router.segment_bounds[segment]
            embed_idx = active_router.selected_text_index(int(timestep_idx), segment, str(pass_type))
            if embed_idx is None:
                return original(x, context, *args, **kwargs)
            text = active_router.text_embeddings[embed_idx]
            if not torch.is_tensor(text) or not torch.is_tensor(context):
                return original(x, context, *args, **kwargs)
            if text.ndim == context.ndim - 1:
                text = text.unsqueeze(0)
            if (
                text.ndim != context.ndim
                or text.shape[0] != context.shape[0]
                or text.shape[-1] != context.shape[-1]
            ):
                return original(x, context, *args, **kwargs)
            if context.shape[1] > text.shape[1]:
                prefix = context[:, : int(context.shape[1] - text.shape[1])]
                text = torch.cat(
                    [prefix, text.to(device=context.device, dtype=context.dtype)], dim=1
                )
            elif tuple(text.shape) != tuple(context.shape):
                return original(x, context, *args, **kwargs)
            sl = slice(start * active_router.spatial_tokens, end * active_router.spatial_tokens)
            try:
                piece = original(x[:, sl], text, *args, **kwargs)
            except TypeError:
                piece = original(x[:, sl], text)
            outs.append(piece)
        return torch.cat(outs, dim=1)

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
        if timestep_idx is None:
            if t_key is not None and t_key != state["last_t"]:
                if state["last_t"] is not unseen:
                    state["t_idx"] += 1
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
            start = int(getattr(router, "audio_start_timestep", 0) or 0)
            audio = kwargs.get("audio")
            if int(timestep_idx) < start and torch.is_tensor(audio):
                kwargs = dict(kwargs)
                kwargs["audio"] = torch.zeros_like(audio[:1] if audio.shape[0] > 1 else audio)
        kwargs.setdefault("timestep_idx", timestep_idx)
        kwargs.setdefault("pass_type", pass_type)
        self_mods, cross_mods = find_attention_modules(self)
        for idx, module in enumerate(self_mods + cross_mods):
            module._ss_timestep = timestep_idx
            module._ss_pass = pass_type
            module._ss_block_idx = idx if module in self_mods else idx - len(self_mods)
        # Official Wan/Hunyuan forwards do not accept these bookkeeping kwargs.
        kwargs.pop("timestep_idx", None)
        kwargs.pop("pass_type", None)
        kwargs.pop("t_idx", None)
        kwargs.pop("block_idx", None)
        return original(*args, **kwargs)

    model.forward = types.MethodType(forward, model)
    model._ss_thread_installed = True


class Adapter:
    name = "base"

    def inject(self, model: Any, router: Optional[FramePromptRouter]) -> List[Any]:
        raise NotImplementedError
