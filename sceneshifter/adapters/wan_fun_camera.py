from __future__ import annotations

import contextlib
import types
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn

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


def _closed_over_function(value: Any, name: str) -> Optional[Any]:
    function = getattr(value, "__func__", value)
    closure = getattr(function, "__closure__", None)
    freevars = getattr(getattr(function, "__code__", None), "co_freevars", ())
    if not closure or name not in freevars:
        return None
    return closure[freevars.index(name)].cell_contents


def _unwrap_callable(value: Any) -> Any:
    """Follow bound methods and ``functools.partial`` to the underlying function."""
    current = value
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        inner = getattr(current, "__func__", None)
        if callable(inner) and inner is not current:
            current = inner
            continue
        partial_func = getattr(current, "func", None)
        if (
            callable(partial_func)
            and partial_func is not current
            and not hasattr(current, "__code__")
        ):
            current = partial_func
            continue
        return current
    return value


def _is_serial_cfg_wrapper(value: Any) -> bool:
    if not callable(value):
        return False
    return callable(_closed_over_function(value, "func"))


def _bind_like(template: Any, function: Any) -> Any:
    owner = getattr(template, "__self__", None)
    if owner is None:
        return function
    return types.MethodType(function, owner)


def _closure_cells(value: Any) -> List[Any]:
    function = _unwrap_callable(value)
    closure = getattr(function, "__closure__", None)
    if not closure:
        return []
    return list(closure)


def _serial_cfg_targets(model: Any, forward: Any) -> List[Any]:
    """Slots whose current callable is the official serial CFG decorator.

    When ``forward`` itself is that decorator, replace it. When an offload hook
    already replaced ``forward``, patch ``_old_forward`` or the hook closure so
    the hook stays in front and calls the batch-2 forward.
    """
    if _is_serial_cfg_wrapper(forward):
        return [(forward, lambda new: setattr(model, "forward", new))]

    targets = []
    old = getattr(model, "_old_forward", None)
    if _is_serial_cfg_wrapper(old):
        targets.append((old, lambda new: setattr(model, "_old_forward", new)))
    if forward is None:
        return targets
    for cell in _closure_cells(forward):
        try:
            contents = cell.cell_contents
        except ValueError:
            continue
        if not _is_serial_cfg_wrapper(contents):
            continue

        def _set_cell(new, cell=cell):
            cell.cell_contents = new

        targets.append((contents, _set_cell))
    return targets


def install_fun_camera_batched_cfg(model: Any) -> None:
    """Replace locked official serial CFG with the feature batch-2 contract."""
    if getattr(model, "_ss_fun_batched_cfg_installed", False):
        return
    forward = getattr(model, "forward", None)
    targets = _serial_cfg_targets(model, forward)
    if not targets:
        # Tiny test models have no ``blocks`` and may already be batched.
        # A real Wan Fun transformer always has ``blocks``; skipping it leaves
        # the locked official decorator on two batch-1 forwards.
        if hasattr(model, "blocks"):
            raise TypeError("Wan Fun Camera batched CFG forward was not installed")
        return
    feature_forward = _closed_over_function(targets[0][0], "func")
    if not callable(feature_forward):
        raise TypeError("Wan Fun Camera batched CFG forward was not installed")

    def batched_cfg_forward(self, x, *args, **kwargs):
        _reject_cuda_fun_encoders(self)
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

    for template, setter in targets:
        setter(_bind_like(template, batched_cfg_forward))
    model._ss_fun_batched_cfg_installed = True


def _cuda_tensors(module: Any) -> List[Any]:
    tensors: List[Any] = []
    parameters = getattr(module, "parameters", None)
    buffers = getattr(module, "buffers", None)
    if callable(parameters):
        tensors.extend(parameters())
    if callable(buffers):
        tensors.extend(buffers())
    return [tensor for tensor in tensors if bool(getattr(tensor, "is_cuda", False))]


def module_has_cuda_storage(module: Any) -> bool:
    if module is None:
        return False
    return bool(_cuda_tensors(module))


def release_module_to_cpu(module: Any) -> bool:
    """Move one module to CPU when any parameter or buffer is still on CUDA."""
    if not module_has_cuda_storage(module):
        return False
    module.to("cpu")
    return True


def release_fun_camera_encoders(*modules: Any) -> None:
    """Return Fun Camera UMT5, CLIP, and VAE weights to CPU after they run."""
    moved = False
    for module in modules:
        moved = release_module_to_cpu(module) or moved
    if moved and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _reject_cuda_fun_encoders(model: Any) -> None:
    """Fail the batch-2 forward if UMT5 or CLIP is still resident on CUDA."""
    guard = getattr(model, "_ss_fun_encoder_guard", None)
    if not guard:
        return
    names = ("text_encoder", "clip_image_encoder", "vae")
    resident = [
        name
        for name, module in zip(names, guard)
        if module_has_cuda_storage(module)
    ]
    if resident:
        raise RuntimeError(
            "Wan Fun Camera encoders still on CUDA at batch-2 forward: "
            + ", ".join(resident)
        )


def _wrap_leave_cpu(module: Any, method_name: str) -> None:
    """Call the hooked method, then leave the module on CPU."""
    if module is None:
        return
    flag = f"_ss_fun_leave_{method_name}"
    if getattr(module, flag, False):
        return
    original = getattr(module, method_name, None)
    if not callable(original):
        return

    def wrapped(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        finally:
            release_fun_camera_encoders(module)

    setattr(module, method_name, wrapped)
    setattr(module, flag, True)


def install_fun_camera_encoder_offload(pipeline: Any) -> None:
    """Keep UMT5, CLIP, and VAE off the GPU except while their own forward runs.

    The accelerate hook on ``transformer.forward`` stays in place. A pre-hook
    releases the encoders before that forward, and each encoder method returns
    its weights to CPU when it finishes.
    """
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None or not hasattr(transformer, "register_forward_pre_hook"):
        raise TypeError("Wan Fun Camera transformer cannot register the encoder offload hook")
    text = getattr(pipeline, "text_encoder", None)
    clip = getattr(pipeline, "clip_image_encoder", None)
    vae = getattr(pipeline, "vae", None)
    transformer._ss_fun_encoder_guard = (text, clip, vae)
    _wrap_leave_cpu(text, "forward")
    _wrap_leave_cpu(clip, "forward")
    for method_name in ("encode", "decode", "forward"):
        _wrap_leave_cpu(vae, method_name)
    if getattr(transformer, "_ss_fun_encoder_release_hook", False):
        return

    def _release_before_dit(module, args, kwargs):
        guard = getattr(module, "_ss_fun_encoder_guard", ())
        release_fun_camera_encoders(*guard)
        return args, kwargs

    transformer.register_forward_pre_hook(_release_before_dit, with_kwargs=True)
    transformer._ss_fun_encoder_release_hook = True


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


def _feature_wan_layernorm_forward(self, x):
    """Return native LayerNorm output, matching feature Fun.

    CUDA autocast keeps that output in FP32. Locked official Fun then applies
    ``.type_as(x)``, which casts it back to the activation dtype.
    """
    return nn.LayerNorm.forward(self, x)


def install_fun_camera_fp32_layernorm(model: Any) -> None:
    """Stop Fun Camera WanLayerNorm from casting autocast FP32 back to BF16.

    Only instances on this model are replaced, so other backbones keep the
    locked official ``.type_as(x)`` LayerNorm.
    """
    if getattr(model, "_ss_fun_fp32_layernorm_installed", False):
        return
    modules = getattr(model, "modules", None)
    patched = []
    if callable(modules):
        for module in modules():
            if type(module).__name__ != "WanLayerNorm":
                continue
            module.forward = types.MethodType(_feature_wan_layernorm_forward, module)
            patched.append(module)
    if not patched:
        if hasattr(model, "blocks"):
            raise TypeError("Wan Fun Camera model has no WanLayerNorm modules")
        return
    model._ss_fun_fp32_layernorm_installed = True


def _feature_wan_rmsnorm_forward(self, x):
    """Match feature Fun RMSNorm under autocast.

    Feature computes the norm in the incoming dtype and casts only the rsqrt
    factor with ``.to(x.dtype)``. Locked official Fun normalizes ``x.float()``
    and then ``.type_as(x)``, which changes Q/K after a matching linear.
    """
    scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)
    return x * scale * self.weight


def install_fun_camera_feature_rmsnorm(model: Any) -> None:
    """Replace Fun Camera WanRMSNorm with the feature autocast formula.

    Only instances on this model are replaced, so other backbones keep the
    locked official ``x.float()`` then ``.type_as(x)`` RMSNorm.
    """
    if getattr(model, "_ss_fun_feature_rmsnorm_installed", False):
        return
    modules = getattr(model, "modules", None)
    patched = []
    if callable(modules):
        for module in modules():
            if type(module).__name__ != "WanRMSNorm":
                continue
            module.forward = types.MethodType(_feature_wan_rmsnorm_forward, module)
            patched.append(module)
    if not patched:
        if hasattr(model, "blocks"):
            raise TypeError("Wan Fun Camera model has no WanRMSNorm modules")
        return
    model._ss_fun_feature_rmsnorm_installed = True


class WanFunCameraAdapter(Adapter):
    name = "wan_fun_camera"

    def inject(self, model: Any, router: Optional[FramePromptRouter]) -> List[Any]:
        # Camera residual (y_camera / SimpleAdapter) stays on the official path.
        install_fun_camera_batched_cfg(model)
        install_fun_camera_fp32_time_embedding(model)
        install_fun_camera_fp32_layernorm(model)
        install_fun_camera_feature_rmsnorm(model)
        self_mods, cross_mods = find_attention_modules(model)
        bound = [wrap_self_attention(m, router) for m in self_mods]
        bound.extend(wrap_cross_attention(m, router) for m in cross_mods)
        thread_timestep(model)
        return bound
