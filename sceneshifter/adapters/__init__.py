from __future__ import annotations

from typing import Dict, Type

from .base import Adapter
from .fantasytalking import FantasyTalkingAdapter
from .hunyuan_avatar import HunyuanAvatarAdapter
from .infinitetalk import InfiniteTalkAdapter
from .multitalk import MultiTalkAdapter
from .wan_animate import WanAnimateAdapter
from .wan_fun_camera import WanFunCameraAdapter
from .wan_s2v import WanS2VAdapter

ADAPTERS: Dict[str, Type[Adapter]] = {
    "multitalk": MultiTalkAdapter,
    "infinitetalk": InfiniteTalkAdapter,
    "fantasytalking": FantasyTalkingAdapter,
    "hunyuan_avatar": HunyuanAvatarAdapter,
    "wan_s2v": WanS2VAdapter,
    "wan_animate": WanAnimateAdapter,
    "wan_fun_camera": WanFunCameraAdapter,
}


def get_adapter(name: str) -> Adapter:
    try:
        return ADAPTERS[name]()
    except KeyError as exc:
        raise KeyError(f"unknown adapter {name!r}; choose from {sorted(ADAPTERS)}") from exc
