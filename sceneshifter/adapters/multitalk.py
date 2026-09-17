from __future__ import annotations

from typing import Any, List, Optional

from ..router import FramePromptRouter
from .base import Adapter, find_attention_modules, thread_timestep, wrap_cross_attention, wrap_self_attention


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
        thread_timestep(model)
        return bound
