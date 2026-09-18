from __future__ import annotations

import types
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

from ..router import FramePromptRouter
from .base import (
    Adapter,
    find_attention_modules,
    project_router_text_embeddings,
    thread_timestep,
    wrap_self_attention,
)

_CLIP_TOKENS = 257


def _attn_kernels(original) -> Tuple[Any, Any]:
    globals_dict = getattr(original, "__globals__", {}) or {}
    flash = globals_dict.get("flash_attention")
    attn_fn = globals_dict.get("attention")
    if callable(flash) and callable(attn_fn):
        return flash, attn_fn
    try:
        from diffsynth.models.wan_video_dit import attention, flash_attention  # type: ignore

        return flash_attention, attention
    except Exception:  # noqa: BLE001
        def _flash(q, k, v, k_lens=None):
            return q

        return _flash, _flash


def _segment_text(text: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    if text.ndim == context.ndim - 1:
        text = text.unsqueeze(0)
    if text.ndim != context.ndim:
        return text
    clip_len = _CLIP_TOKENS if context.size(1) > _CLIP_TOKENS else 0
    if clip_len and text.size(1) == context.size(1):
        text = text[:, clip_len:]
    return text


def wrap_fantasy_i2v_processor(processor: nn.Module, router: Optional[FramePromptRouter]) -> nn.Module:
    """Route segment text on full-sequence Q so audio can still view 21 latent frames.

    Official ``WanCrossAttentionProcessor`` computes Q from the full token list,
    then (on the feature branch) flashes each scene's Q slice against that
    scene's T5 context. Unified must not slice ``x`` before the processor.
    """
    if getattr(processor, "_ss_fantasy_cross_wrapped", False):
        processor.frame_prompt_router = router
        return processor
    original = processor.__call__
    processor.frame_prompt_router = router
    flash_attention, attention = _attn_kernels(original)

    def __call__(
        self,
        attn,
        x,
        context,
        context_lens,
        audio_proj,
        audio_context_lens,
        latents_num_frames: int = 21,
        audio_scale: float = 1.0,
        **kwargs,
    ):
        active = getattr(self, "frame_prompt_router", None)
        timestep_idx = int(getattr(attn, "_ss_timestep", 0) or 0)
        pass_type = str(getattr(attn, "_ss_pass", "cond") or "cond")
        if (
            active is None
            or getattr(active, "text_embeddings", None) is None
            or not active.should_use_segment_prompts(timestep_idx, pass_type)
        ):
            return original(
                attn,
                x,
                context,
                context_lens,
                audio_proj,
                audio_context_lens,
                latents_num_frames=latents_num_frames,
                audio_scale=audio_scale,
                **kwargs,
            )
        project_router_text_embeddings(active)
        clip_len = _CLIP_TOKENS if context.size(1) > _CLIP_TOKENS else 0
        context_img = context[:, :clip_len] if clip_len else None
        b, n, d = x.size(0), attn.num_heads, attn.head_dim
        q = attn.norm_q(attn.q(x)).view(b, -1, n, d)
        spatial = int(getattr(active, "spatial_tokens", 0) or 0)
        if spatial <= 0 or q.size(1) != int(active.num_frames) * spatial:
            if int(active.num_frames) > 0 and q.size(1) % int(active.num_frames) == 0:
                spatial = int(q.size(1)) // int(active.num_frames)
                active.spatial_tokens = spatial
        text_x = None
        for segment in range(active.num_segments):
            start, end = active.segment_bounds[segment]
            embed_idx = active.selected_text_index(timestep_idx, segment, pass_type)
            if embed_idx is None:
                return original(
                    attn,
                    x,
                    context,
                    context_lens,
                    audio_proj,
                    audio_context_lens,
                    latents_num_frames=latents_num_frames,
                    audio_scale=audio_scale,
                    **kwargs,
                )
            text = active.text_embeddings[embed_idx]
            if not torch.is_tensor(text):
                return original(
                    attn,
                    x,
                    context,
                    context_lens,
                    audio_proj,
                    audio_context_lens,
                    latents_num_frames=latents_num_frames,
                    audio_scale=audio_scale,
                    **kwargs,
                )
            text = _segment_text(text, context).to(device=x.device, dtype=x.dtype)
            if text.size(0) != b:
                text = text.expand(b, *text.shape[1:])
            k_seg = attn.norm_k(attn.k(text)).view(b, -1, n, d)
            v_seg = attn.v(text).view(b, -1, n, d)
            sl = slice(int(start) * spatial, int(end) * spatial)
            out_seg = flash_attention(q[:, sl], k_seg, v_seg, k_lens=None)
            if text_x is None:
                text_x = q.new_zeros(q.shape, dtype=out_seg.dtype)
            text_x[:, sl] = out_seg.to(text_x.dtype)
        if text_x is None:
            return original(
                attn,
                x,
                context,
                context_lens,
                audio_proj,
                audio_context_lens,
                latents_num_frames=latents_num_frames,
                audio_scale=audio_scale,
                **kwargs,
            )
        if not getattr(active, "_debug_fantasy_cross_printed", False):
            print(
                "[parity-unified] fantasy I2V cross-attn routed "
                f"segments={active.num_segments} seq={int(q.size(1))} spatial={spatial} "
                f"step={timestep_idx}",
                flush=True,
            )
            active._debug_fantasy_cross_printed = True
        x_out = text_x.flatten(2)
        if context_img is not None and context_img.numel():
            k_img = attn.norm_k_img(attn.k_img(context_img)).view(b, -1, n, d)
            v_img = attn.v_img(context_img).view(b, -1, n, d)
            img_x = flash_attention(q, k_img, v_img, k_lens=None).flatten(2)
            x_out = x_out + img_x
        if audio_proj is None:
            audio_x = torch.zeros_like(x_out)
        elif len(audio_proj.shape) == 4:
            audio_q = q.view(b * int(latents_num_frames), -1, n, d)
            ip_key = self.k_proj(audio_proj).view(b * int(latents_num_frames), -1, n, d)
            ip_value = self.v_proj(audio_proj).view(b * int(latents_num_frames), -1, n, d)
            audio_x = attention(audio_q, ip_key, ip_value, k_lens=audio_context_lens)
            audio_x = audio_x.view(b, q.size(1), n, d).flatten(2)
        elif len(audio_proj.shape) == 3:
            ip_key = self.k_proj(audio_proj).view(b, -1, n, d)
            ip_value = self.v_proj(audio_proj).view(b, -1, n, d)
            audio_x = attention(q, ip_key, ip_value, k_lens=audio_context_lens).flatten(2)
        else:
            audio_x = torch.zeros_like(x_out)
        x_out = x_out + audio_x * audio_scale
        return attn.o(x_out)

    processor.__call__ = types.MethodType(__call__, processor)
    processor._ss_fantasy_cross_wrapped = True
    return processor


class FantasyTalkingAdapter(Adapter):
    name = "fantasytalking"

    def inject(self, model: Any, router: Optional[FramePromptRouter]) -> List[Any]:
        self_mods, cross_mods = find_attention_modules(model)
        bound = [wrap_self_attention(m, router) for m in self_mods]
        seen_proc = set()
        for module in cross_mods:
            module.frame_prompt_router = router
            bound.append(module)
            processor = getattr(module, "processor", None)
            if processor is not None:
                wrap_fantasy_i2v_processor(processor, router)
                bound.append(processor)
                seen_proc.add(id(processor))
        if hasattr(model, "named_modules"):
            for _, module in model.named_modules():
                processor = getattr(module, "processor", None)
                if processor is None or id(processor) in seen_proc:
                    continue
                wrap_fantasy_i2v_processor(processor, router)
                bound.append(processor)
                seen_proc.add(id(processor))
        thread_timestep(model)
        return bound
