from __future__ import annotations

from typing import Any, List, Optional

import torch

from ..layouts import S2VTokenLayout
from ..router import FramePromptRouter
from .base import Adapter, find_attention_modules, thread_timestep, wrap_cross_attention, wrap_self_attention


def framepack_keys(router: FramePromptRouter, k: torch.Tensor, segment: int) -> torch.Tensor:
    layout = getattr(router, "_s2v_layout", None)
    if layout is None:
        extra = k.shape[1] - router.num_frames * router.spatial_tokens
        if extra <= 0:
            return k.new_empty(0, dtype=torch.long)
        layout = S2VTokenLayout(
            noisy_frames=router.num_frames,
            spatial_tokens=router.spatial_tokens,
            framepack_tokens=max(0, extra - router.spatial_tokens),
        )
        router._s2v_layout = layout
    start = layout.framepack_slice.start
    stop = layout.framepack_slice.stop
    if stop <= start:
        return k.new_empty(0, dtype=torch.long)
    return torch.arange(start, min(stop, k.shape[1]), device=k.device)


class WanS2VAdapter(Adapter):
    name = "wan_s2v"

    def inject(self, model: Any, router: Optional[FramePromptRouter]) -> List[Any]:
        self_mods, cross_mods = find_attention_modules(model)
        bound = [
            wrap_self_attention(module, router, extra_key_fn=framepack_keys)
            for module in self_mods
        ]
        bound.extend(wrap_cross_attention(module, router) for module in cross_mods)
        thread_timestep(model)
        return bound
