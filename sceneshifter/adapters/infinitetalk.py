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
    wrap_audio_attention,
    wrap_cross_attention,
    wrap_self_attention,
)
from .talking_audio import install_reference_ref_map


def _latent_frame_tokens(shape: Any) -> int:
    """Spatial token count of one latent frame from ``grid_sizes[0]``."""
    if shape is None:
        return 0
    if torch.is_tensor(shape):
        values = [int(v) for v in shape.detach().flatten().tolist()]
    else:
        try:
            values = [int(v) for v in shape]
        except TypeError:
            return 0
    if len(values) < 2:
        return 0
    return int(values[-2]) * int(values[-1])


def _sequence_parallel_rank() -> tuple[int, int]:
    try:
        from xfuser.core.distributed import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
        )

        return int(get_sequence_parallel_world_size()), int(get_sequence_parallel_rank())
    except Exception:
        return 1, 0


def zero_first_latent_frame_audio(module: torch.nn.Module) -> torch.nn.Module:
    """Drop the audio residual on latent frame 0.

    Official InfiniteTalk keeps that frame equal to the clean condition latent.
    The single-clip reference removes that pin, then zeros frame-0 audio so the
    condition frame is not blurred while later frames still receive audio.
    """
    if getattr(module, "_ss_it_frame0_audio", False):
        return module
    original = module.forward

    def forward(self, x, *args, **kwargs):
        out = original(x, *args, **kwargs)
        if not torch.is_tensor(out) or out.ndim < 2:
            return out
        shape = kwargs.get("shape")
        if shape is None and len(args) >= 2:
            shape = args[1]
        tokens = _latent_frame_tokens(shape)
        if tokens <= 0:
            return out
        sp_size, sp_rank = _sequence_parallel_rank()
        if sp_size > 1 and sp_rank != 0:
            return out
        n_zero = min(int(tokens), int(out.shape[1]))
        if n_zero <= 0:
            return out
        out = out.clone()
        out[:, :n_zero] = 0
        return out

    module.forward = types.MethodType(forward, module)
    module._ss_it_frame0_audio = True
    return module


class InfiniteTalkAdapter(Adapter):
    name = "infinitetalk"

    def inject(self, model: Any, router: Optional[FramePromptRouter]) -> List[Any]:
        self_mods, cross_mods = find_attention_modules(model)
        for module in self_mods:
            install_reference_ref_map(module, split_num=2)
        bound = [wrap_self_attention(m, router) for m in self_mods]
        bound.extend(wrap_cross_attention(m, router) for m in cross_mods)
        for module in find_audio_modules(model):
            wrap_audio_attention(module, router)
            bound.append(zero_first_latent_frame_audio(module))
        thread_timestep(model)
        print("[infinitetalk] zero audio residual on latent frame 0", flush=True)
        print("[infinitetalk] ref map uses reference bf16 query-length reduction", flush=True)
        return bound
