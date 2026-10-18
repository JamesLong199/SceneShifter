from __future__ import annotations

import contextlib
import types
from typing import Any, List, Optional

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
    original_model_forward = getattr(model, "forward", None)

    def feature_cfg_batch(args):
        value = next((item for item in args if torch.is_tensor(item)), None)
        outer_batch = int(getattr(model, "_ss_fun_outer_cfg_batch", 0) or 0)
        if (
            value is not None
            and value.ndim > 0
            and int(value.shape[0]) == 1
            and outer_batch > 1
        ):
            return value.expand(outer_batch, *value.shape[1:]).contiguous(), 1
        return value, None

    def embedding_forward(self, *args, **kwargs):
        value, trim = feature_cfg_batch(args)
        call_args = (value, *args[1:]) if value is not None else args
        with _feature_fp32_context(call_args):
            result = original_embedding(*call_args, **kwargs)
        if not torch.is_tensor(result):
            raise TypeError("Wan Fun Camera time embedding did not return a tensor")
        if trim is not None:
            result = result[:trim]
        return _KeepFeatureTimeDtype(result)

    def projection_forward(self, *args, **kwargs):
        value, trim = feature_cfg_batch(args)
        call_args = (value, *args[1:]) if value is not None else args
        with _feature_fp32_context(call_args):
            result = original_projection(*call_args, **kwargs)
        return result[:trim] if trim is not None else result

    def model_forward(self, *args, **kwargs):
        value = kwargs.get("x")
        if value is None and args:
            value = args[0]
        model._ss_fun_outer_cfg_batch = (
            int(value.shape[0]) if torch.is_tensor(value) and value.ndim > 0 else 0
        )
        try:
            return original_model_forward(*args, **kwargs)
        finally:
            model._ss_fun_outer_cfg_batch = 0

    time_embedding.forward = types.MethodType(embedding_forward, time_embedding)
    time_projection.forward = types.MethodType(projection_forward, time_projection)
    if callable(original_model_forward):
        model.forward = types.MethodType(model_forward, model)
    model._ss_fun_fp32_time_installed = True


class WanFunCameraAdapter(Adapter):
    name = "wan_fun_camera"

    def inject(self, model: Any, router: Optional[FramePromptRouter]) -> List[Any]:
        # Camera residual (y_camera / SimpleAdapter) stays on the official path.
        install_fun_camera_fp32_time_embedding(model)
        self_mods, cross_mods = find_attention_modules(model)
        bound = [wrap_self_attention(m, router) for m in self_mods]
        bound.extend(wrap_cross_attention(m, router) for m in cross_mods)
        thread_timestep(model)
        return bound
