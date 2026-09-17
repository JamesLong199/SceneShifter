from __future__ import annotations

from typing import Any, List, Optional

from ..router import FramePromptRouter
from .base import Adapter, find_attention_modules, thread_timestep, wrap_cross_attention, wrap_self_attention


class WanAnimateAdapter(Adapter):
    name = "wan_animate"

    def inject(self, model: Any, router: Optional[FramePromptRouter]) -> List[Any]:
        self_mods, cross_mods = find_attention_modules(model)
        bound = [wrap_self_attention(m, router) for m in self_mods]
        bound.extend(wrap_cross_attention(m, router) for m in cross_mods)
        thread_timestep(model)
        return bound
