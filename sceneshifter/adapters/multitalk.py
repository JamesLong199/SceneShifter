from __future__ import annotations

from typing import Any, List, Optional

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
from .talking_audio import align_multitalk_audio_rope, install_reference_ref_map


class MultiTalkAdapter(Adapter):
    name = "multitalk"

    def inject(self, model: Any, router: Optional[FramePromptRouter]) -> List[Any]:
        self_mods, cross_mods = find_attention_modules(model)
        bound = []
        for module in self_mods:
            install_reference_ref_map(module, split_num=1)
            wrap_self_attention(module, router)
            bound.append(module)
        for module in cross_mods:
            wrap_cross_attention(module, router)
            bound.append(module)
        for module in find_audio_modules(model):
            align_multitalk_audio_rope(module)
            wrap_audio_attention(module, router)
            bound.append(module)
        print("[multitalk] audio rope uses reference subject centers", flush=True)
        thread_timestep(model)
        return bound
