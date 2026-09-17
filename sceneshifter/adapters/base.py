"""Shared runtime wrapping for official (or toy) attention modules."""

from __future__ import annotations

import types
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn

from ..attention import gather_attention, scaled_dot_product
from ..router import FramePromptRouter


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


def wrap_self_attention(
    module: nn.Module,
    router: Optional[FramePromptRouter],
    *,
    extra_key_fn: Optional[Callable] = None,
) -> nn.Module:
    original = module.forward
    module.frame_prompt_router = router

    def forward(self, x, *args, **kwargs):
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

    module.forward = types.MethodType(forward, module)
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
            sl = slice(start * active_router.spatial_tokens, end * active_router.spatial_tokens)
            # Run original on the slice if possible; else project locally.
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

    def forward(self, *args, **kwargs):
        timestep_idx = kwargs.get("timestep_idx", kwargs.get("t_idx", 0))
        pass_type = kwargs.get("pass_type", "cond")
        self_mods, cross_mods = find_attention_modules(self)
        for idx, module in enumerate(self_mods + cross_mods):
            module._ss_timestep = timestep_idx
            module._ss_pass = pass_type
            module._ss_block_idx = idx if module in self_mods else idx - len(self_mods)
        return original(*args, **kwargs)

    model.forward = types.MethodType(forward, model)
    model._ss_thread_installed = True


class Adapter:
    name = "base"

    def inject(self, model: Any, router: Optional[FramePromptRouter]) -> List[Any]:
        raise NotImplementedError
