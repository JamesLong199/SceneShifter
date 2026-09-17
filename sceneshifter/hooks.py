"""Runtime inject: wrap official modules in memory, never edit files."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from . import checks
from .adapters import get_adapter
from .router import FramePromptRouter


def iter_named_modules(root: Any):
    if hasattr(root, "named_modules"):
        yield from root.named_modules()
        return
    yield "", root


def inject_router(
    model: Any,
    router: Optional[FramePromptRouter],
    *,
    adapter: str,
    expected_min: int = 1,
    official_root: Optional[str] = None,
) -> List[Any]:
    """Bind ``router`` onto official attention modules via the named adapter."""
    impl = get_adapter(adapter)
    bound = impl.inject(model, router)
    if router is not None:
        checks.validate_inject(bound, expected_min=expected_min)
    if official_root:
        checks.official_tree_is_clean(official_root)
    return list(bound)


def bind_router_attribute(modules: Iterable[Any], router: Optional[FramePromptRouter]) -> List[Any]:
    bound = []
    for module in modules:
        module.frame_prompt_router = router
        bound.append(module)
    return bound
