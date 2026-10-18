from __future__ import annotations

import contextlib
import types
from typing import Any, List, Optional

import numpy as np
import torch

from ..router import FramePromptRouter
from .base import Adapter, find_attention_modules, thread_timestep, wrap_cross_attention, wrap_self_attention


class _KeepFeatureTimeDtype:
    """Delay only the official ``e.to(model_dtype)`` conversion."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def to(self, *args, **kwargs):
        dtype = kwargs.get("dtype")
        if args and isinstance(args[0], torch.dtype):
            dtype = args[0]
        if self.tensor.dtype == torch.float32 and dtype in {
            torch.bfloat16,
            torch.float16,
        }:
            return self.tensor
        return self.tensor.to(*args, **kwargs)


def _feature_fp32_context(args):
    value = next((item for item in args if torch.is_tensor(item)), None)
    if value is not None and value.is_cuda:
        return torch.cuda.amp.autocast(dtype=torch.float32)
    return contextlib.nullcontext()


def _closed_over_function(
    value: Any,
    name: str,
    seen: Optional[set[int]] = None,
) -> Optional[Any]:
    function = getattr(value, "__func__", value)
    seen = set() if seen is None else seen
    if id(function) in seen:
        return None
    seen.add(id(function))
    closure = getattr(function, "__closure__", None)
    freevars = getattr(getattr(function, "__code__", None), "co_freevars", ())
    if not closure:
        return None
    if name in freevars:
        return closure[freevars.index(name)].cell_contents
    for cell in closure:
        candidate = cell.cell_contents
        if callable(candidate):
            found = _closed_over_function(candidate, name, seen)
            if found is not None:
                return found
    return None


def install_fun_camera_batched_cfg(model: Any) -> None:
    """Replace locked official serial CFG with the feature batch-2 contract."""
    if getattr(model, "_ss_fun_batched_cfg_installed", False):
        return
    decorated = getattr(model, "forward", None)
    feature_forward = _closed_over_function(decorated, "func")
    if not callable(feature_forward):
        # Tiny test models and future official models may already be batched.
        return

    def batched_cfg_forward(self, x, *args, **kwargs):
        batch = len(x)
        skip_cfg = (
            batch >= 2
            and self.cfg_skip_ratio is not None
            and self.current_steps
            >= self.num_inference_steps * (1 - self.cfg_skip_ratio)
        )
        if skip_cfg:
            half = int(batch // 2)
            x = x[half:]
            args = tuple(
                value[half:]
                if isinstance(value, (torch.Tensor, list, tuple, np.ndarray))
                else value
                for value in args
            )
            kwargs = {
                key: (
                    value[half:]
                    if isinstance(value, (torch.Tensor, list, tuple, np.ndarray))
                    else value
                )
                for key, value in kwargs.items()
            }
        result = feature_forward(self, x, *args, **kwargs)
        return torch.cat([result, result], dim=0) if skip_cfg else result

    model.forward = types.MethodType(batched_cfg_forward, model)
    model._ss_fun_batched_cfg_installed = True


def install_fun_camera_fp32_time_embedding(model: Any) -> None:
    """Match feature Fun's FP32 time-embedding/projection block in memory.

    Locked official Fun casts ``time_embedding(...)`` to the latent dtype before
    ``time_projection``.  The feature branch keeps both operations in its FP32
    autocast block and passes FP32 ``e/e0`` into every DiT block and the head.
    """
    if getattr(model, "_ss_fun_fp32_time_installed", False):
        return
    time_embedding = getattr(model, "time_embedding", None)
    time_projection = getattr(model, "time_projection", None)
    if time_embedding is None or time_projection is None:
        if hasattr(model, "blocks"):
            raise TypeError("Wan Fun Camera model has no time embedding/projection modules")
        return

    original_embedding = time_embedding.forward
    original_projection = time_projection.forward

    def embedding_forward(self, *args, **kwargs):
        with _feature_fp32_context(args):
            result = original_embedding(*args, **kwargs)
        if not torch.is_tensor(result):
            raise TypeError("Wan Fun Camera time embedding did not return a tensor")
        return _KeepFeatureTimeDtype(result)

    def projection_forward(self, *args, **kwargs):
        with _feature_fp32_context(args):
            return original_projection(*args, **kwargs)

    time_embedding.forward = types.MethodType(embedding_forward, time_embedding)
    time_projection.forward = types.MethodType(projection_forward, time_projection)
    model._ss_fun_fp32_time_installed = True


class WanFunCameraAdapter(Adapter):
    name = "wan_fun_camera"

    def inject(self, model: Any, router: Optional[FramePromptRouter]) -> List[Any]:
        # Camera residual (y_camera / SimpleAdapter) stays on the official path.
        install_fun_camera_batched_cfg(model)
        install_fun_camera_fp32_time_embedding(model)
        self_mods, cross_mods = find_attention_modules(model)
        bound = [wrap_self_attention(m, router) for m in self_mods]
        bound.extend(wrap_cross_attention(m, router) for m in cross_mods)
        thread_timestep(model)
        return bound
