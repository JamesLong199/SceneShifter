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


def _fallback(original, attn, x, context, context_lens, audio_proj, audio_context_lens, latents_num_frames, audio_scale, kwargs):
    if audio_proj is None:
        try:
            return original(x, context, context_lens, **kwargs)
        except TypeError:
            return original(x, context, context_lens)
    try:
        return original(
            x,
            context,
            context_lens,
            audio_proj,
            audio_context_lens,
            latents_num_frames,
            audio_scale,
            **kwargs,
        )
    except TypeError:
        return original(
            x,
            context,
            context_lens,
            audio_proj,
            audio_context_lens,
            latents_num_frames,
            audio_scale,
        )


def wrap_fantasy_i2v_processor(processor: Any, router: Optional[FramePromptRouter]) -> Any:
    """Keep the official processor object; routing is installed on the parent module."""
    if processor is not None:
        processor.frame_prompt_router = router
    return processor


def wrap_fantasy_i2v_cross_module(module: nn.Module, router: Optional[FramePromptRouter]) -> nn.Module:
    """Route segment text in ``WanI2VCrossAttention.forward`` on the full Q sequence.

    Official DiT calls ``self.processor(...)``, which uses the processor *class*
    ``__call__``. Replacing a registered ``nn.Module`` processor also fails.
    Intercepting ``module.forward`` keeps audio's 21-frame Q view intact.
    """
    if getattr(module, "_ss_fantasy_cross_wrapped", False):
        module.frame_prompt_router = router
        return module
    original = module.forward
    module.frame_prompt_router = router
    processor = getattr(module, "processor", None)
    kernel_src = type(processor).__call__ if processor is not None else original
    flash_attention, attention = _attn_kernels(kernel_src)

    def forward(
        self,
        x,
        context,
        context_lens=None,
        audio_proj=None,
        audio_context_lens=None,
        latents_num_frames: int = 21,
        audio_scale: float = 1.0,
        **kwargs,
    ):
        active = getattr(self, "frame_prompt_router", None)
        timestep_idx = int(
            getattr(self, "_ss_timestep", getattr(active, "_ss_timestep", 0)) or 0
        )
        pass_type = str(getattr(self, "_ss_pass", getattr(active, "_ss_pass", "cond")) or "cond")
        if (
            active is None
            or not active.should_use_segment_prompts(timestep_idx, pass_type)
        ):
            return _fallback(
                original,
                self,
                x,
                context,
                context_lens,
                audio_proj,
                audio_context_lens,
                latents_num_frames,
                audio_scale,
                kwargs,
            )
        if getattr(active, "text_embeddings", None) is None:
            raise RuntimeError(
                f"fantasytalking segment prompts are active at step {timestep_idx}, "
                "but no segment text embeddings were captured"
            )
        project_router_text_embeddings(active)
        clip_len = _CLIP_TOKENS if context.size(1) > _CLIP_TOKENS else 0
        context_img = context[:, :clip_len] if clip_len else None
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
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
                return _fallback(
                    original,
                    self,
                    x,
                    context,
                    context_lens,
                    audio_proj,
                    audio_context_lens,
                    latents_num_frames,
                    audio_scale,
                    kwargs,
                )
            text = active.text_embeddings[embed_idx]
            if not torch.is_tensor(text):
                return _fallback(
                    original,
                    self,
                    x,
                    context,
                    context_lens,
                    audio_proj,
                    audio_context_lens,
                    latents_num_frames,
                    audio_scale,
                    kwargs,
                )
            text = _segment_text(text, context).to(device=x.device, dtype=x.dtype)
            if text.size(0) != b:
                text = text.expand(b, *text.shape[1:])
            k_seg = self.norm_k(self.k(text)).view(b, -1, n, d)
            v_seg = self.v(text).view(b, -1, n, d)
            sl = slice(int(start) * spatial, int(end) * spatial)
            out_seg = flash_attention(q[:, sl], k_seg, v_seg, k_lens=None)
            if text_x is None:
                text_x = q.new_zeros(q.shape, dtype=out_seg.dtype)
            text_x[:, sl] = out_seg.to(text_x.dtype)
        if text_x is None:
            return _fallback(
                original,
                self,
                x,
                context,
                context_lens,
                audio_proj,
                audio_context_lens,
                latents_num_frames,
                audio_scale,
                kwargs,
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
            k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
            v_img = self.v_img(context_img).view(b, -1, n, d)
            img_x = flash_attention(q, k_img, v_img, k_lens=None).flatten(2)
            x_out = x_out + img_x
        proc = getattr(self, "processor", None)
        if audio_proj is None or proc is None or not hasattr(proc, "k_proj"):
            audio_x = torch.zeros_like(x_out)
        elif len(audio_proj.shape) == 4:
            audio_q = q.view(b * int(latents_num_frames), -1, n, d)
            ip_key = proc.k_proj(audio_proj).view(b * int(latents_num_frames), -1, n, d)
            ip_value = proc.v_proj(audio_proj).view(b * int(latents_num_frames), -1, n, d)
            audio_x = attention(audio_q, ip_key, ip_value, k_lens=audio_context_lens)
            audio_x = audio_x.view(b, q.size(1), n, d).flatten(2)
        elif len(audio_proj.shape) == 3:
            ip_key = proc.k_proj(audio_proj).view(b, -1, n, d)
            ip_value = proc.v_proj(audio_proj).view(b, -1, n, d)
            audio_x = attention(q, ip_key, ip_value, k_lens=audio_context_lens).flatten(2)
        else:
            audio_x = torch.zeros_like(x_out)
        return self.o(x_out + audio_x * audio_scale)

    module.forward = types.MethodType(forward, module)
    module._ss_fantasy_cross_wrapped = True
    return module


class FantasyTalkingAdapter(Adapter):
    name = "fantasytalking"

    def inject(self, model: Any, router: Optional[FramePromptRouter]) -> List[Any]:
        self_mods, cross_mods = find_attention_modules(model)
        bound = [wrap_self_attention(m, router) for m in self_mods]
        for module in cross_mods:
            wrap_fantasy_i2v_cross_module(module, router)
            bound.append(module)
        thread_timestep(model)
        return bound
