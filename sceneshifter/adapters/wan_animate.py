from __future__ import annotations

import types
from typing import Any, List, Optional

import torch
import torch.nn as nn

from ..router import FramePromptRouter
from .base import (
    Adapter,
    find_attention_modules,
    project_router_text_embeddings,
    thread_timestep,
    wrap_cross_attention,
    wrap_self_attention,
)


def wrap_animate_cross_attention(
    module: nn.Module, router: Optional[FramePromptRouter]
) -> nn.Module:
    """Mirror native Animate: one full CLIP branch plus segmented T5 branches."""
    if getattr(module, "_ss_animate_cross_wrapped", False):
        module.frame_prompt_router = router
        return module
    original = module.forward
    flash = getattr(original, "__globals__", {}).get("flash_attention")
    if not callable(flash):
        raise RuntimeError("Wan Animate cross-attention has no official flash_attention")
    module.frame_prompt_router = router

    def forward(self, x, context, context_lens=None, *args, **kwargs):
        active = getattr(self, "frame_prompt_router", None)
        timestep = int(getattr(self, "_ss_timestep", 0) or 0)
        block_idx = int(getattr(self, "_ss_block_idx", -1))
        pass_type = str(getattr(self, "_ss_pass", "cond"))
        block = (
            active.active_block(timestep, block_idx, pass_type)
            if active is not None
            else None
        )
        if (
            active is None
            or block is None
            or pass_type == "uncond"
            or not active.should_use_segment_prompts(timestep, pass_type)
        ):
            return original(x, context, context_lens, *args, **kwargs)
        if active.text_embeddings is None:
            raise RuntimeError(
                f"wan_animate segment prompts are active at step {timestep}, "
                "but no segment text embeddings were captured"
            )
        project_router_text_embeddings(active)
        use_img = bool(getattr(self, "use_img_emb", True))
        context_img = context[:, :257] if use_img else None
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        text_out = q.new_zeros(q.shape)
        spatial = int(active.spatial_tokens)
        for segment, (start, end) in enumerate(active.segment_bounds):
            text = active.text_embeddings[segment]
            if text.ndim == 2:
                text = text.unsqueeze(0)
            text = text.to(device=x.device, dtype=x.dtype)
            if text.shape[0] != b:
                text = text.expand(b, *text.shape[1:])
            k = self.norm_k(self.k(text)).view(b, -1, n, d)
            v = self.v(text).view(b, -1, n, d)
            sl = slice(int(start) * spatial, int(end) * spatial)
            text_out[:, sl] = flash(q[:, sl], k, v, k_lens=None)
        output = text_out.flatten(2)
        if context_img is not None:
            k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
            v_img = self.v_img(context_img).view(b, -1, n, d)
            output = output + flash(q, k_img, v_img, k_lens=None).flatten(2)
        return self.o(output)

    module.forward = types.MethodType(forward, module)
    module._ss_animate_cross_wrapped = True
    return module


class WanAnimateAdapter(Adapter):
    name = "wan_animate"

    def inject(self, model: Any, router: Optional[FramePromptRouter]) -> List[Any]:
        self_mods, cross_mods = find_attention_modules(model)
        if router is not None:
            router._ss_num_self_layers = len(self_mods)
        bound = [wrap_self_attention(m, router) for m in self_mods]
        bound.extend(
            wrap_animate_cross_attention(m, router)
            if type(m).__name__ == "WanAnimateCrossAttention"
            else wrap_cross_attention(m, router)
            for m in cross_mods
        )
        thread_timestep(model)
        return bound
