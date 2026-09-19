from __future__ import annotations

import types
from typing import Any, List, Optional

import torch

from ..router import FramePromptRouter
from .base import (
    Adapter,
    find_attention_modules,
    find_audio_modules,
    thread_timestep,
    wrap_cross_attention,
    wrap_self_attention,
)


def _subject_rope_positions(
    x_ref_attn_map: torch.Tensor,
    human_num: int,
    class_range: int,
    class_interval: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    centers = torch.linspace(
        float(class_interval) / 2.0,
        float(class_range) - float(class_interval) / 2.0,
        steps=max(int(human_num), 1),
        device=device,
        dtype=dtype,
    )
    background = torch.tensor(float(class_range // 2), device=device, dtype=dtype)
    labels = x_ref_attn_map.argmax(dim=0).to(torch.long)
    visual = torch.full(
        (int(x_ref_attn_map.shape[1]),),
        float(background.item()),
        device=device,
        dtype=dtype,
    )
    for subject in range(min(int(human_num), int(x_ref_attn_map.shape[0]) - 1)):
        visual[labels == subject] = centers[subject]
    return visual, centers


def wrap_multitalk_audio_attention(
    module: torch.nn.Module, router: Optional[FramePromptRouter]
) -> torch.nn.Module:
    """Match transition's always-on multi-subject audio RoPE assignment."""
    if getattr(module, "_ss_multitalk_audio_wrapped", False):
        module.frame_prompt_router = router
        return module
    original = module.forward
    globals_dict = getattr(original, "__globals__", {})
    rearrange = globals_dict.get("rearrange")
    xformers = globals_dict.get("xformers")
    if rearrange is None or xformers is None:
        raise RuntimeError("MultiTalk audio attention globals are unavailable")
    module.frame_prompt_router = router

    def forward(
        self,
        x,
        encoder_hidden_states,
        shape=None,
        x_ref_attn_map=None,
        subject_token_assignments=None,
        human_num=None,
    ):
        active = getattr(self, "frame_prompt_router", None)
        timestep = int(getattr(self, "_ss_timestep", 0) or 0)
        start = int(getattr(active, "audio_start_timestep", 0) or 0)
        if active is not None and timestep < start:
            return torch.zeros_like(x)
        if int(human_num or 1) <= 1 or x_ref_attn_map is None:
            return original(
                x,
                encoder_hidden_states,
                shape=shape,
                x_ref_attn_map=x_ref_attn_map,
                human_num=human_num,
            )
        encoder_hidden_states = encoder_hidden_states.squeeze(0)
        frames, _, _ = shape
        x = rearrange(x, "B (T S) C -> (B T) S C", T=frames)
        batch, tokens, channels = x.shape
        q = self.q_linear(x).view(
            batch, tokens, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        if self.qk_norm:
            q = self.q_norm(q)
        visual_pos, centers = _subject_rope_positions(
            x_ref_attn_map,
            int(human_num),
            int(self.class_range),
            int(self.class_interval),
            dtype=q.dtype,
            device=q.device,
        )
        if subject_token_assignments is not None:
            assignments = subject_token_assignments.reshape(-1).to(q.device)
            if int(assignments.numel()) == int(visual_pos.numel()):
                visual_pos.fill_(float(self.rope_bak))
                for subject in range(int(human_num)):
                    visual_pos[assignments == subject] = centers[subject]
        q = rearrange(q, "(B T) H S C -> B H (T S) C", T=frames)
        q = self.rope_1d(q, visual_pos)
        q = rearrange(q, "B H (T S) C -> (B T) H S C", T=frames)

        _, audio_tokens, _ = encoder_hidden_states.shape
        encoder_kv = self.kv_linear(encoder_hidden_states).view(
            batch, audio_tokens, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        encoder_k, encoder_v = encoder_kv.unbind(0)
        if self.qk_norm:
            encoder_k = self.add_k_norm(encoder_k)
        per_frame = torch.full(
            (audio_tokens,),
            float(self.rope_bak),
            dtype=encoder_k.dtype,
            device=encoder_k.device,
        )
        chunks = torch.chunk(
            torch.arange(audio_tokens, device=encoder_k.device),
            max(int(human_num), 1),
        )
        for subject, indices in enumerate(chunks[: int(human_num)]):
            if indices.numel():
                per_frame[indices] = centers[subject].to(encoder_k.dtype)
        encoder_pos = torch.cat([per_frame] * int(frames))
        encoder_k = rearrange(
            encoder_k, "(B T) H S C -> B H (T S) C", T=frames
        )
        encoder_k = self.rope_1d(encoder_k, encoder_pos)
        encoder_k = rearrange(
            encoder_k, "B H (T S) C -> (B T) H S C", T=frames
        )

        q = rearrange(q, "B H M K -> B M H K")
        encoder_k = rearrange(encoder_k, "B H M K -> B M H K")
        encoder_v = rearrange(encoder_v, "B H M K -> B M H K")
        output = xformers.ops.memory_efficient_attention(
            q, encoder_k, encoder_v, attn_bias=None, op=None
        )
        output = rearrange(output, "B M H K -> B H M K")
        output = output.transpose(1, 2).reshape(batch, tokens, channels)
        output = self.proj_drop(self.proj(output))
        return rearrange(output, "(B T) S C -> B (T S) C", T=frames)

    module.forward = types.MethodType(forward, module)
    module._ss_multitalk_audio_wrapped = True
    return module


class MultiTalkAdapter(Adapter):
    name = "multitalk"

    def inject(self, model: Any, router: Optional[FramePromptRouter]) -> List[Any]:
        self_mods, cross_mods = find_attention_modules(model)
        bound = []
        for module in self_mods:
            wrap_self_attention(module, router)
            bound.append(module)
        for module in cross_mods:
            wrap_cross_attention(module, router)
            bound.append(module)
        for module in find_audio_modules(model):
            wrap_multitalk_audio_attention(module, router)
            bound.append(module)
        thread_timestep(model)
        return bound
