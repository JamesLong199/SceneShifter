from __future__ import annotations

import os
import sys
from typing import Optional

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

OFFICIAL_DIR = {
    "multitalk": "multitalk",
    "infinitetalk": "infinitetalk",
    "fantasytalking": "fantasy-talking",
    "hunyuan_avatar": "hunyuanvideo-avatar",
    "wan_s2v": "wan2.2",
    "wan_animate": "wan2.2",
    "wan_fun_camera": "videox-fun",
}


def repo_root() -> str:
    return _REPO_ROOT


def official_root(backend: str, override: Optional[str] = None) -> str:
    if override:
        return os.path.abspath(override)
    return os.path.join(_REPO_ROOT, "third_party", OFFICIAL_DIR[backend])


def push_official_path(path: str) -> None:
    path = os.path.abspath(path)
    if path not in sys.path:
        sys.path.insert(0, path)
