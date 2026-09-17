"""Load and unwrap ``frame_prompt_routing`` JSON."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict, Optional

from .checks import validate_config


def unwrap_routing(data: Dict[str, Any]) -> Dict[str, Any]:
    if "frame_prompt_routing" in data:
        routing = deepcopy(data["frame_prompt_routing"])
    else:
        routing = deepcopy(data)
    return routing


def resolve_mask_paths(routing: Dict[str, Any], asset_root: Optional[str]) -> Dict[str, Any]:
    if not asset_root:
        return routing
    for key in ("background_mask_path", "head_mask_path"):
        paths = routing.get(key)
        if not isinstance(paths, dict):
            continue
        resolved = {}
        for subject, path in paths.items():
            path = str(path)
            resolved[str(subject)] = (
                path if os.path.isabs(path) else os.path.normpath(os.path.join(asset_root, path))
            )
        routing[key] = resolved
    return routing


def load_frame_prompt_config(
    source: str | Dict[str, Any],
    *,
    backend: str = "multitalk",
    asset_root: Optional[str] = None,
    validate: bool = True,
) -> Dict[str, Any]:
    if isinstance(source, str):
        with open(source, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        root = asset_root or os.path.dirname(os.path.abspath(source))
    else:
        data = source
        root = asset_root
    routing = resolve_mask_paths(unwrap_routing(data), root)
    if validate:
        validate_config(routing, backend=backend)
    return routing
