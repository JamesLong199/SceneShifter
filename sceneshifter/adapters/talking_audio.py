"""Match feature-branch talking-head audio details from outside the locked tree.

MultiTalk reference audio RoPE snaps each query to a subject center. Its
person map is one mean over every head (``split_num=1``). InfiniteTalk keeps
two head chunks (``split_num=2``). Both then rescale that mean in bf16 by the
query length: ``calculate_x_ref_attn_map`` transposes ``[B, seq, heads, dim]``
and, when no head mask is passed, uses ``shape[2]``. That is not the official
``(mean0 + mean1) / 2`` and not a rescale by the head count.

Once the schedule is on, both references build that person map from the
Q-scaled query. MultiTalk also replaces the map's argmax with the live
per-subject mask when the active mask type is ``ref_attn``.
"""

from __future__ import annotations

import math
import types
from typing import Any, Optional

import torch


def publish_scaled_query_to_audio_map(
    original: torch.Tensor,
    scaled: torch.Tensor,
    backend: Optional[str],
) -> None:
    """Leave the scaled query in the tensor official self-attn maps after flash.

    The schedule scales a clone for attention. Reference ``get_attn_map_with_target``
    then reads that same scaled query. Official code reads the unscaled local
    ``q`` after the kernel returns, so copy the scaled values back for the
    talking-head backends. Other backends keep the unscaled storage their
    ref-loc collectors alias.
    """
    if backend not in {"multitalk", "infinitetalk"}:
        return
    if not torch.is_tensor(original) or not torch.is_tensor(scaled):
        return
    if scaled.data_ptr() == original.data_ptr() or tuple(scaled.shape) != tuple(original.shape):
        return
    original.copy_(scaled.to(dtype=original.dtype))


def _ref_attn_schedule_block(router: Any, timestep: int) -> Optional[dict]:
    match = None
    for block in getattr(router, "schedule", None) or []:
        if not isinstance(block, dict):
            continue
        span = block.get("timestep_range")
        if not isinstance(span, (list, tuple)) or len(span) < 2:
            continue
        if not (int(span[0]) <= int(timestep) <= int(span[1])):
            continue
        params = block.get("params") or {}
        if str(params.get("mask_type", "")) == "ref_attn":
            match = block
    return match


def multitalk_live_subject_assignments(
    router: Any,
    *,
    human_num: int,
    n_t: int,
    spatial: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Reference assignments from the live per-subject ref-attn mask.

    Foreground tokens take ``argmax`` over subjects. Every other token stays
    at ``human_num``, the background center. Steps outside a ``ref_attn``
    schedule keep the person-map argmax by returning ``None``.
    """
    if router is None or int(human_num) <= 1:
        return None
    timestep = getattr(router, "_ss_timestep", None)
    if timestep is None:
        return None
    block = _ref_attn_schedule_block(router, int(timestep))
    if block is None:
        return None
    serial = getattr(router, "_ss_forward_serial", None)
    if serial is not None:
        key = (int(timestep), int(serial))
        cached = getattr(router, "_ss_audio_assign_cache", None)
        if isinstance(cached, tuple) and cached[0] == key:
            return cached[1]
    refresh = getattr(router, "refresh_live_mask_from_stored", None)
    if callable(refresh):
        refresh(int(timestep), block)
    fg = getattr(router, "live_subject_fg", None)
    result: Optional[torch.Tensor] = None
    if (
        torch.is_tensor(fg)
        and fg.ndim == 3
        and int(fg.shape[0]) >= int(human_num)
        and int(fg.shape[1]) == int(n_t)
        and int(fg.shape[2]) == int(spatial)
    ):
        # Reference stores per-subject background and uses ``1 - (bg > 0.5)``.
        # ``live_subject_fg`` is already that foreground mask.
        flat = (fg[: int(human_num)] > 0.5).to(dtype=torch.float32).reshape(int(human_num), -1)
        result = torch.full((int(flat.shape[1]),), int(human_num), dtype=torch.long)
        any_fg = flat > 0.5
        covered = any_fg.any(dim=0)
        if bool(covered.any()):
            result[covered] = torch.argmax(flat[:, covered], dim=0).to(dtype=torch.long)
        result = result.to(device=device)
    if serial is not None:
        router._ss_audio_assign_cache = ((int(timestep), int(serial)), result)
    return result


def reference_audio_map_weight(visual_q_chunk: torch.Tensor) -> int:
    """Token count used when the audio call does not pass a head mask.

    Reference ``calculate_x_ref_attn_map`` receives ``[B, seq, heads, dim]``,
    transposes it to ``[B, heads, seq, dim]``, and reads ``shape[2]``.
    """
    return int(visual_q_chunk.shape[1])


def accumulate_reference_ref_map(chunk_means: list[torch.Tensor], weights: list[int]) -> Optional[torch.Tensor]:
    """Reference ``get_attn_map_with_target`` reduction, including bf16 rounding."""
    if not chunk_means:
        return None
    acc = torch.zeros_like(chunk_means[0])
    total = 0
    for maps, count in zip(chunk_means, weights):
        count = int(count)
        if count <= 0:
            continue
        acc += maps * float(count)
        total += count
    if total <= 0:
        return None
    return acc / float(total)


def install_reference_ref_map(module: Any, *, split_num: int) -> None:
    """Point official self-attn at the reference person-map average."""
    forward = module.forward
    func = getattr(forward, "__func__", forward)
    globals_dict = getattr(func, "__globals__", None)
    if not isinstance(globals_dict, dict):
        return
    current = globals_dict.get("get_attn_map_with_target")
    if current is None:
        print(f"[talking-audio] ref-map patch skipped; no get_attn_map on {type(module).__name__}", flush=True)
        return
    if getattr(current, "_ss_reference_ref_map", False):
        return
    calculate = current.__globals__.get("calculate_x_ref_attn_map")
    if calculate is None:
        print("[talking-audio] ref-map patch skipped; calculate_x_ref_attn_map missing", flush=True)
        return
    get_sp_group = current.__globals__.get("get_sp_group")
    default_split = int(split_num)

    def get_attn_map_with_target(
        visual_q,
        ref_k,
        shape,
        ref_target_masks=None,
        split_num=default_split,
        enable_sp=False,
        head_mask=None,
    ):
        del head_mask  # audio calls omit it; the reference then weights by query length
        _n_t, n_h, n_w = shape
        if enable_sp and get_sp_group is not None:
            ref_k = get_sp_group().all_gather(ref_k, dim=1)
        x_seqlens = n_h * n_w
        ref_k = ref_k[:, :x_seqlens]
        _batch, seq_lens, heads, _dim = visual_q.shape
        class_num, _ = ref_target_masks.shape
        acc = torch.zeros(class_num, seq_lens).to(visual_q.device).to(visual_q.dtype)
        chunk = int(math.ceil(float(heads) / float(split_num)))
        total = 0
        for head_start in range(0, int(heads), chunk):
            head_end = min(int(heads), head_start + chunk)
            query_chunk = visual_q[:, :, head_start:head_end, :]
            result = calculate(
                query_chunk,
                ref_k[:, :, head_start:head_end, :],
                ref_target_masks,
            )
            if isinstance(result, tuple):
                maps, selected = result
                selected = int(selected.item()) if torch.is_tensor(selected) else int(selected)
            else:
                maps = result
                selected = reference_audio_map_weight(query_chunk)
            if selected <= 0:
                continue
            acc += maps * float(selected)
            total += selected
        if total <= 0:
            return None
        return acc / float(total)

    get_attn_map_with_target._ss_reference_ref_map = True  # type: ignore[attr-defined]
    get_attn_map_with_target._ss_reference_split_num = default_split  # type: ignore[attr-defined]
    globals_dict["get_attn_map_with_target"] = get_attn_map_with_target
    print(f"[talking-audio] ref-map split_num={default_split} installed", flush=True)


def multitalk_subject_centers(module: Any, human_num: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.linspace(
        float(module.class_interval) / 2.0,
        float(module.class_range) - float(module.class_interval) / 2.0,
        steps=max(int(human_num), 1),
        device=device,
        dtype=dtype,
    )


def multitalk_audio_query_positions(
    module: Any,
    *,
    n_t: int,
    spatial: int,
    human_num: int,
    x_ref_attn_map: Optional[torch.Tensor],
    subject_token_assignments: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Reference query RoPE positions. ``None`` means use the single-person path."""
    centers = multitalk_subject_centers(module, human_num, device, dtype)
    background = float(module.rope_bak)
    if subject_token_assignments is not None:
        assignments = subject_token_assignments.to(device=device)
        if assignments.ndim != 1:
            assignments = assignments.reshape(-1)
        if int(assignments.numel()) == int(n_t) * int(spatial):
            assignments = assignments.to(torch.long)
            positions = torch.full((assignments.numel(),), background, device=device, dtype=dtype)
            for subject_idx in range(int(human_num)):
                positions[assignments == int(subject_idx)] = centers[subject_idx]
            return positions
    if x_ref_attn_map is None:
        return None
    max_indices = x_ref_attn_map.argmax(dim=0).to(torch.long)
    positions = torch.full((x_ref_attn_map.size(1),), background, device=device, dtype=dtype)
    for subject_idx in range(min(int(human_num), int(x_ref_attn_map.size(0)) - 1)):
        positions[max_indices == int(subject_idx)] = centers[subject_idx]
    return positions


def multitalk_audio_key_positions(
    module: Any,
    *,
    n_audio: int,
    n_t: int,
    human_num: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    centers = multitalk_subject_centers(module, human_num, device, dtype)
    background = float(module.rope_bak)
    per_frame = torch.full((int(n_audio),), background, dtype=dtype, device=device)
    chunks = torch.chunk(torch.arange(int(n_audio), device=device), max(int(human_num), 1))
    for subject_idx, token_idx in enumerate(chunks[: int(human_num)]):
        if token_idx.numel() > 0:
            per_frame[token_idx] = centers[subject_idx].to(dtype=dtype)
    return torch.cat([per_frame] * int(n_t), dim=0)


def _single_stream_forward(module: Any, x: torch.Tensor, encoder_hidden_states: torch.Tensor, shape: Any) -> torch.Tensor:
    for cls in type(module).__mro__[1:]:
        if cls.__name__ == "SingleStreamAttention":
            return cls.forward(module, x, encoder_hidden_states, shape)
    raise RuntimeError(f"{type(module).__name__} has no SingleStreamAttention parent")


def align_multitalk_audio_rope(module: torch.nn.Module) -> torch.nn.Module:
    """Replace official two-person score scaling with reference subject centers."""
    if getattr(module, "_ss_mt_audio_rope", False):
        return module
    from einops import rearrange

    def forward(
        self,
        x,
        encoder_hidden_states=None,
        shape=None,
        x_ref_attn_map=None,
        human_num=None,
        subject_token_assignments=None,
        **_kwargs,
    ):
        encoder_hidden_states = encoder_hidden_states.squeeze(0)
        if human_num == 1:
            return _single_stream_forward(self, x, encoder_hidden_states, shape)

        n_t, _, _ = shape
        x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=n_t)
        batch, spatial, _channels = x.shape
        query = self.q_linear(x)
        query = query.view(batch, spatial, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.qk_norm:
            query = self.q_norm(query)

        if subject_token_assignments is None:
            subject_token_assignments = multitalk_live_subject_assignments(
                getattr(self, "frame_prompt_router", None),
                human_num=int(human_num),
                n_t=int(n_t),
                spatial=int(spatial),
                device=query.device,
            )
        positions = multitalk_audio_query_positions(
            self,
            n_t=int(n_t),
            spatial=int(spatial),
            human_num=int(human_num),
            x_ref_attn_map=x_ref_attn_map,
            subject_token_assignments=subject_token_assignments,
            device=query.device,
            dtype=query.dtype,
        )
        if positions is None:
            return _single_stream_forward(self, x, encoder_hidden_states, shape)

        import xformers

        query = rearrange(query, "(B N_t) H S C -> B H (N_t S) C", N_t=n_t)
        query = self.rope_1d(query, positions)
        query = rearrange(query, "B H (N_t S) C -> (B N_t) H S C", N_t=n_t)

        _batch_audio, n_audio, _dim = encoder_hidden_states.shape
        encoder_kv = self.kv_linear(encoder_hidden_states)
        encoder_kv = encoder_kv.view(batch, n_audio, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        encoder_k, encoder_v = encoder_kv.unbind(0)
        if self.qk_norm:
            encoder_k = self.add_k_norm(encoder_k)

        key_positions = multitalk_audio_key_positions(
            self,
            n_audio=int(n_audio),
            n_t=int(n_t),
            human_num=int(human_num),
            device=encoder_k.device,
            dtype=encoder_k.dtype,
        )
        encoder_k = rearrange(encoder_k, "(B N_t) H S C -> B H (N_t S) C", N_t=n_t)
        encoder_k = self.rope_1d(encoder_k, key_positions)
        encoder_k = rearrange(encoder_k, "B H (N_t S) C -> (B N_t) H S C", N_t=n_t)

        query = rearrange(query, "B H M K -> B M H K")
        encoder_k = rearrange(encoder_k, "B H M K -> B M H K")
        encoder_v = rearrange(encoder_v, "B H M K -> B M H K")
        attended = xformers.ops.memory_efficient_attention(query, encoder_k, encoder_v, attn_bias=None, op=None)
        attended = rearrange(attended, "B M H K -> B H M K")
        attended = attended.transpose(1, 2).reshape(batch, spatial, _channels)
        attended = self.proj_drop(self.proj(attended))
        return rearrange(attended, "(B N_t) S C -> B (N_t S) C", N_t=n_t)

    module.forward = types.MethodType(forward, module)
    module._ss_mt_audio_rope = True
    return module
