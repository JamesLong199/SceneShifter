"""Opt-in tensor fingerprints for locating the first parity divergence."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Set

import torch


def _parse_int_set(raw: str, default: Set[int]) -> Set[int]:
    if not raw.strip():
        return set(default)
    if raw.strip().lower() == "all":
        return set()
    return {int(value.strip()) for value in raw.split(",") if value.strip()}


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    if tensor.dtype == torch.bfloat16:
        payload = tensor.view(torch.uint16).numpy().tobytes()
    else:
        payload = tensor.numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


class ParityTracer:
    def __init__(self, directory: str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "trace.jsonl"
        self.steps = _parse_int_set(os.environ.get("SCENESHIFTER_TRACE_STEPS", "0"), {0})
        self.layers = _parse_int_set(
            os.environ.get("SCENESHIFTER_TRACE_LAYERS", "0,37"), {0, 37}
        )
        self._lock = threading.Lock()
        self._ordinal = 0

    def enabled_for(self, step: Optional[int], layer: Optional[int]) -> bool:
        if step is not None and self.steps and int(step) not in self.steps:
            return False
        if layer is not None and self.layers and int(layer) not in self.layers:
            return False
        return True

    def record(
        self,
        event: str,
        tensor: torch.Tensor,
        *,
        step: Optional[int] = None,
        layer: Optional[int] = None,
        **metadata: Any,
    ) -> None:
        if not self.enabled_for(step, layer):
            return
        value = tensor.detach()
        payload: Dict[str, Any] = {
            "ordinal": self._ordinal,
            "event": str(event),
            "step": None if step is None else int(step),
            "layer": None if layer is None else int(layer),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": tensor_sha256(value),
            "metadata": _json_value(metadata),
        }
        self._ordinal += 1
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")


_TRACER: Optional[ParityTracer] = None
_TRACER_DIR: Optional[str] = None


def get_tracer() -> Optional[ParityTracer]:
    global _TRACER, _TRACER_DIR
    directory = os.environ.get("SCENESHIFTER_TRACE_DIR", "").strip()
    if not directory:
        return None
    if _TRACER is None or _TRACER_DIR != directory:
        _TRACER = ParityTracer(directory)
        _TRACER_DIR = directory
    return _TRACER


def trace_tensor(
    event: str,
    tensor: Any,
    *,
    step: Optional[int] = None,
    layer: Optional[int] = None,
    **metadata: Any,
) -> None:
    tracer = get_tracer()
    if tracer is None or not torch.is_tensor(tensor):
        return
    tracer.record(event, tensor, step=step, layer=layer, **metadata)
