from __future__ import annotations

from typing import Any, List, Optional

from ..router import FramePromptRouter
from .base import Adapter, find_attention_modules, thread_timestep, wrap_cross_attention, wrap_self_attention


class HunyuanAvatarAdapter(Adapter):
    name = "hunyuan_avatar"

    def inject(self, model: Any, router: Optional[FramePromptRouter]) -> List[Any]:
        # Joint MM-DiT still binds the shared router; V-T isolation is adapter-side
        # grouping on top of noisy-axis key sets (see scene_masked_attention ports).
        self_mods, cross_mods = find_attention_modules(
            model,
            self_names=("self_attn", "attn", "double_block", "single_block"),
        )
        if not self_mods and hasattr(model, "named_modules"):
            for _, module in model.named_modules():
                if hasattr(module, "Wqkv") or hasattr(module, "q"):
                    self_mods.append(module)
        bound = []
        for module in self_mods:
            wrap_self_attention(module, router)
            bound.append(module)
        for module in cross_mods:
            wrap_cross_attention(module, router)
            bound.append(module)
        thread_timestep(model)
        return bound
