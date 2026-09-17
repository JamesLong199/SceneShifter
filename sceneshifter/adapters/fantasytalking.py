from __future__ import annotations

from typing import Any, List, Optional

from ..router import FramePromptRouter
from .base import Adapter, find_attention_modules, thread_timestep, wrap_cross_attention, wrap_self_attention


class FantasyTalkingAdapter(Adapter):
    name = "fantasytalking"

    def inject(self, model: Any, router: Optional[FramePromptRouter]) -> List[Any]:
        self_mods, cross_mods = find_attention_modules(model)
        bound = [wrap_self_attention(m, router) for m in self_mods]
        bound.extend(wrap_cross_attention(m, router) for m in cross_mods)
        # DiffSynth processors expose __call__ on attn.processor
        if hasattr(model, "named_modules"):
            for _, module in model.named_modules():
                processor = getattr(module, "processor", None)
                if processor is not None:
                    processor.frame_prompt_router = router
                    bound.append(processor)
        thread_timestep(model)
        return bound
