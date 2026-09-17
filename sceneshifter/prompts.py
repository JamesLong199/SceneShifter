"""Prompt encoding protocol. Adapters implement backbone-specific projection."""

from __future__ import annotations

from typing import List, Protocol, Sequence

import torch


class PromptEncoder(Protocol):
    def encode_segment_prompts(self, prompts: Sequence[str]) -> List[torch.Tensor]:
        ...
