"""SceneShifter metadata / batch helpers."""

from __future__ import annotations

import json
from typing import Any, Dict


def load_metadata(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def is_sceneshifter_metadata(data: Dict[str, Any]) -> bool:
    return "frame_prompt_routing" in data
