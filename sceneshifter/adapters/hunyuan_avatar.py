"""Hunyuan Avatar MMDiT adapter: official flash_attn_varlen_func + scene groups."""

from __future__ import annotations

import math
import sys
import types
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from ..layouts import HunyuanMMDiTLayout
from ..router import FramePromptRouter
from .base import (
    Adapter,
    _CURRENT_MODULE,
    find_attention_modules,
    thread_timestep,
    wrap_cross_attention,
    wrap_self_attention,
)


HY_BLOCK_NAMES = {"DoubleStreamBlock", "SingleStreamBlock"}
PROMPT_PREFIX = "Authentic, Realistic, Natural, High-quality, Lens-Fixed, "


def _empty(device: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.long, device=device)


def _resize_frame_mask(mask: torch.Tensor, frames: int, spatial: int) -> torch.Tensor:
    value = mask.to(dtype=torch.float32)
    if value.numel() == frames * spatial:
        return value.reshape(frames, spatial) > 0.5
    old_spatial = value.numel() // max(frames, 1)
    if old_spatial <= 0 or old_spatial * frames != value.numel():
        raise ValueError(f"cannot reshape mask of {tuple(mask.shape)} to [{frames}, {spatial}]")
    src = value.reshape(frames, 1, old_spatial)
    return torch.nn.functional.interpolate(src, size=spatial, mode="nearest").reshape(frames, spatial) > 0.5


def _resize_center_crop_mask(mask: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Official Hunyuan ``_resize_center_crop_mask``: cover-resize + center crop."""
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if mask.ndim != 2:
        raise ValueError(f"expected 2D mask, got {tuple(mask.shape)}")
    height, width = int(mask.shape[0]), int(mask.shape[1])
    scale = max(target_h / max(height, 1), target_w / max(width, 1))
    resized_h = max(round(height * scale), target_h)
    resized_w = max(round(width * scale), target_w)
    resized = torch.nn.functional.interpolate(
        mask[None, None].float(), size=(resized_h, resized_w), mode="nearest"
    )[0, 0]
    top = (resized_h - target_h) // 2
    left = (resized_w - target_w) // 2
    return resized[top : top + target_h, left : left + target_w]


def apply_hunyuan_official_masks(router: FramePromptRouter) -> None:
    """Reload subject FG with official cover-crop so ref/video tokens match exp."""
    raw = getattr(router, "raw_subject_fg", None)
    if raw is None:
        return
    patch_h, patch_w = int(router.patch_height), int(router.patch_width)
    if patch_h <= 0 or patch_w <= 0:
        return
    subjects = []
    union = None
    for subject in range(int(raw.shape[0])):
        fg = raw[subject, 0].float()
        subjects.append((_resize_center_crop_mask(fg, (patch_h, patch_w)) > 0.5).flatten())
        union = fg if union is None else torch.maximum(union, fg)
    union_lat = (_resize_center_crop_mask(union, (patch_h, patch_w)) > 0.5).flatten()
    stacked = torch.stack(subjects, dim=0)
    frames = int(router.num_frames)
    router.subject_fg_masks = stacked[:, None].expand(-1, frames, -1).contiguous()
    router.static_bg = (~union_lat)[None].expand(frames, -1).contiguous()
    router.live_bg = None
    router.live_subject_fg = None
    print(
        f"[hunyuan-adapter] official cover-crop masks subjects={tuple(stacked.shape)} "
        f"latent={patch_h}x{patch_w} union_fg={int(union_lat.sum())}",
        flush=True,
    )


def _fg_maps(router: FramePromptRouter, device: torch.device, frames: int, spatial: int):
    live = router.live_subject_fg
    if live is not None:
        video_fg = _resize_frame_mask(live.any(0).to(device=device), frames, spatial)
        return video_fg[0], video_fg
    if router.subject_fg_masks is not None:
        video_fg = _resize_frame_mask(router.subject_fg_masks.any(0).to(device=device), frames, spatial)
        return video_fg[0], video_fg
    bg = router.background_mask(device)
    if bg is None:
        ones = torch.ones(spatial, dtype=torch.bool, device=device)
        return ones, ones[None].expand(frames, -1)
    video_fg = _resize_frame_mask((~bg.to(device=device)), frames, spatial)
    return video_fg[0], video_fg


def build_hunyuan_groups(
    router: FramePromptRouter,
    block: Optional[Dict[str, Any]],
    row_mode: str,
    img_len: int,
    txt_len: int,
    timestep: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Build exp-port KV groups on the official ``[ref | video | text]`` axis."""
    layout = HunyuanMMDiTLayout(
        video_frames=router.num_frames,
        spatial_tokens=router.spatial_tokens,
        text_tokens=txt_len,
    )
    if img_len != layout.img_seq_len:
        raise ValueError(
            f"Hunyuan img_len={img_len} does not match ref+video={layout.img_seq_len}"
        )
    ref_len = layout.ref_length
    tokens_per_frame = layout.spatial_tokens
    ref_fg, video_fg = _fg_maps(router, device, layout.video_frames, tokens_per_frame)
    ref_fg_idx = torch.nonzero(ref_fg, as_tuple=False).flatten()
    ref_bg_idx = torch.nonzero(~ref_fg, as_tuple=False).flatten()
    full_idx = torch.arange(tokens_per_frame, device=device)
    empty = _empty(device)
    apply_vv = block is not None
    trans_active = apply_vv and router.should_apply_transition_context(timestep)
    windows_by_seg = {}
    if trans_active:
        windows_by_seg = {
            int(window["new_segment"]): window
            for window in router.get_transition_context_windows()
        }

    def ref_spatial(region: str) -> torch.Tensor:
        if region == "full":
            return full_idx
        return ref_fg_idx if region == "fg" else ref_bg_idx

    def video_idx_frames(frames, region: str) -> torch.Tensor:
        parts = []
        for frame in frames:
            if region == "full":
                spatial = full_idx
            else:
                frame_fg = video_fg[int(frame)]
                selected = frame_fg if region == "fg" else ~frame_fg
                spatial = torch.nonzero(selected, as_tuple=False).flatten()
            parts.append(ref_len + int(frame) * tokens_per_frame + spatial)
        return torch.cat(parts) if parts else empty

    cond_segments = getattr(router, "_hy_cond_text_segments", None)
    uncond_len = getattr(router, "_hy_uncond_text_len", None)
    if row_mode == "routed":
        if not cond_segments:
            cond_segments = [(0, txt_len)] * router.num_segments
        text_ranges = cond_segments
    else:
        end = int(uncond_len if uncond_len is not None else txt_len)
        text_ranges = [(0, end)]

    def text_idx(seg: int) -> torch.Tensor:
        start, end = text_ranges[seg if row_mode == "routed" else 0]
        start, end = int(start), int(end)
        if end <= start:
            return empty
        return torch.arange(img_len + start, img_len + end, device=device)

    groups: List[Dict[str, Any]] = []

    def add_group(q_idx, k_idx, scale=None, scale_positions=None):
        if q_idx.numel() == 0 or k_idx.numel() == 0:
            return
        groups.append(
            {
                "q": q_idx,
                "k": k_idx,
                "scale": scale,
                "scale_positions": scale_positions,
            }
        )

    def visual_key_parts(entries, window=None):
        key_parts = []
        allow0, spatial0 = entries[0]
        if allow0:
            key_parts.append(ref_spatial(spatial0))
        prev_segment = None if window is None else int(window["new_segment"]) - 1
        context_set = set() if window is None else {int(f) for f in window["context_frames"]}
        for key_seg, (allow, spatial) in enumerate(entries):
            if not allow:
                continue
            upgrade_prev = (
                window is not None
                and key_seg == prev_segment
                and spatial in ("fg", "bg")
                and context_set
            )
            start, end = router.segment_bounds[key_seg]
            if upgrade_prev:
                non_ctx = [frame for frame in range(start, end) if frame not in context_set]
                if non_ctx:
                    key_parts.append(video_idx_frames(non_ctx, spatial))
                key_parts.append(video_idx_frames(sorted(context_set), "full"))
            else:
                key_parts.append(video_idx_frames(range(start, end), spatial))
        return key_parts

    def bg_scale_positions(frames):
        positions = []
        for local_frame, frame in enumerate(frames):
            frame_bg = torch.nonzero(~video_fg[int(frame)], as_tuple=False).flatten()
            positions.append(local_frame * tokens_per_frame + frame_bg)
        return torch.cat(positions) if positions else empty

    if apply_vv:
        mask = block["_mask"]
        for seg, row in enumerate(mask):
            scale = router.qscale_factor(seg, block)
            if abs(scale - 1.0) < 1e-6:
                scale = None
            bg_only = bool((block.get("params") or {}).get("q_scale_bg_only", False))
            window = windows_by_seg.get(seg)
            start, end = router.segment_bounds[seg]
            seg_frames = list(range(start, end))
            if window is not None:
                query_set = {int(f) for f in window["query_frames"]}
                frame_splits = [
                    ([f for f in seg_frames if f in query_set], window),
                    ([f for f in seg_frames if f not in query_set], None),
                ]
            else:
                frame_splits = [(seg_frames, None)]
            region_rows = row.items() if isinstance(row, dict) else [("all", row)]
            for region_key, entries in region_rows:
                q_region = "full" if region_key == "all" else region_key
                for frames, upgrade_window in frame_splits:
                    if not frames:
                        continue
                    q_idx = video_idx_frames(frames, q_region)
                    if seg == 0:
                        q_idx = torch.cat([ref_spatial(q_region), q_idx])
                    key_parts = visual_key_parts(entries, upgrade_window)
                    key_parts.append(text_idx(seg if row_mode == "routed" else 0))
                    k_idx = torch.cat(key_parts)
                    group_scale = None
                    scale_positions = None
                    if scale is not None:
                        if not bg_only or region_key == "bg":
                            group_scale = scale
                        elif region_key == "all":
                            group_scale = scale
                            scale_positions = bg_scale_positions(frames)
                    add_group(q_idx, k_idx, group_scale, scale_positions)
    else:
        all_visual = torch.arange(img_len, device=device)
        for seg in range(router.num_segments):
            start, end = router.segment_bounds[seg]
            q_idx = video_idx_frames(range(start, end), "full")
            if seg == 0:
                q_idx = torch.cat([ref_spatial("full"), q_idx])
            add_group(q_idx, torch.cat([all_visual, text_idx(seg if row_mode == "routed" else 0)]))

    if row_mode == "routed":
        for seg in range(router.num_segments):
            start, end = router.segment_bounds[seg]
            key_parts = [video_idx_frames(range(start, end), "full"), text_idx(seg)]
            if seg == 0:
                key_parts.insert(0, ref_spatial("full"))
            add_group(text_idx(seg), torch.cat(key_parts))
    else:
        add_group(text_idx(0), torch.cat([torch.arange(img_len, device=device), text_idx(0)]))
    return groups


def _maybe_update_live_mask(
    router: FramePromptRouter,
    q_row: torch.Tensor,
    k_row: torch.Tensor,
    img_len: int,
    timestep: int,
    block_idx: Optional[int],
    pass_type: str,
) -> None:
    if pass_type == "uncond":
        return
    block = router.active_block(int(timestep), block_idx, pass_type)
    if block is None:
        return
    params = block.get("params") or {}
    if str(params.get("mask_type", "")) != "ref_attn":
        return
    layer = params.get("ref_attn_mask_layer")
    if layer is not None and block_idx is not None and int(block_idx) != int(layer):
        return
    ref_len = router.spatial_tokens
    if img_len <= ref_len:
        return
    video_q = q_row[ref_len:img_len].unsqueeze(0)
    ref_k = k_row[:ref_len].unsqueeze(0)
    if router.subject_fg_masks is None:
        return
    subject = router.subject_fg_masks[:, 0].to(device=q_row.device, dtype=torch.float32)
    if subject.shape[-1] != ref_k.shape[1]:
        subject = torch.nn.functional.interpolate(
            subject.unsqueeze(1), size=int(ref_k.shape[1]), mode="nearest"
        ).squeeze(1)
    background = (~subject.any(0)).float().unsqueeze(0)
    targets = torch.cat([subject, background], dim=0)
    head = params.get("ref_attn_mask_head")
    heads = list(range(video_q.shape[2])) if head is None else [int(head)]
    scale = 1.0 / math.sqrt(float(video_q.shape[-1]))
    scores = []
    mask_denom = targets.sum(dim=-1).clamp_min(1.0).view(1, -1, 1)
    with torch.no_grad():
        for hid in heads:
            logits = torch.matmul(
                video_q[:, :, hid].float() * scale,
                ref_k[:, :, hid].float().transpose(-2, -1),
            )
            probs = logits.softmax(dim=-1)
            scores.append(torch.einsum("bqr,cr->bcq", probs, targets) / mask_denom)
    stacked = torch.stack(scores, dim=2)[0]
    if head is None:
        live = stacked.float().mean(dim=1)
    else:
        live = stacked[:, 0].float()
    num_subjects = int(router.subject_fg_masks.shape[0])
    if int(live.shape[0]) < num_subjects + 1:
        return
    spatial = int(router.spatial_tokens)
    frames = int(router.num_frames)
    if int(live.shape[-1]) != frames * spatial:
        return
    # Official derive_subject_fg_masks: per-subject FG vs BG argmax, not mean FG.
    router.update_live_mask_from_class_scores(
        live[:num_subjects].view(num_subjects, frames, spatial),
        live[num_subjects].view(frames, spatial),
        drop_low_percent=float(params.get("ref_attn_mask_drop_low_percent", 10.0)),
        min_component_size=int(params.get("ref_attn_mask_min_component_size", 20)),
        hole_size=int(params.get("ref_attn_mask_hole_size", 0)),
    )


def hunyuan_scene_varlen(
    orig,
    module: nn.Module,
    router: FramePromptRouter,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    extra_args,
    extra_kwargs,
) -> torch.Tensor:
    heads, dim = q.shape[-2], q.shape[-1]
    seq = int(max_seqlen_q)
    batch = q.shape[0] // seq
    q4 = q.view(batch, seq, heads, dim)
    k4 = k.view(batch, seq, heads, dim)
    v4 = v.view(batch, seq, heads, dim)
    img_len = int(getattr(module, "_ss_img_len", 0) or 0)
    if img_len <= 0 or img_len >= seq:
        return orig(q, k, v, *extra_args, **extra_kwargs)
    txt_len = seq - img_len
    expected = router.spatial_tokens * (router.num_frames + 1)
    if img_len != expected:
        spatial = img_len // (router.num_frames + 1)
        if spatial > 0 and spatial * (router.num_frames + 1) == img_len:
            old = int(router.spatial_tokens)
            router.spatial_tokens = spatial
            router.patch_height = spatial
            router.patch_width = 1
            if router.static_bg is not None and router.static_bg.shape[-1] == old:
                router.static_bg = _resize_frame_mask(router.static_bg, router.num_frames, spatial)
        else:
            print(
                f"[hunyuan-adapter] img_len={img_len} != ref+video={expected}; official flash",
                flush=True,
            )
            return orig(q, k, v, *extra_args, **extra_kwargs)
    timestep = int(getattr(module, "_ss_timestep", 0) or 0)
    pass_type = str(getattr(module, "_ss_pass", "cond"))
    block_idx = getattr(module, "_ss_block_idx", None)
    block = router.active_block(timestep, block_idx, pass_type)
    out = q4.new_zeros(q4.shape)
    for row in range(batch):
        row_pass = pass_type if batch == 1 else ("uncond" if row == 0 else "cond")
        row_mode = "global" if row_pass == "uncond" and timestep < 10 else "routed"
        _maybe_update_live_mask(
            router, q4[row], k4[row], img_len, timestep, block_idx, row_pass
        )
        groups = build_hunyuan_groups(
            router, block, row_mode, img_len, txt_len, timestep, q.device
        )
        for group in groups:
            q_g = q4[row].index_select(0, group["q"])
            if group["scale"] is not None:
                q_g = q_g.clone()
                if group["scale_positions"] is not None and group["scale_positions"].numel():
                    q_g[group["scale_positions"]] *= group["scale"]
                else:
                    q_g *= group["scale"]
            k_g = k4[row].index_select(0, group["k"])
            v_g = v4[row].index_select(0, group["k"])
            nq, nk = int(q_g.shape[0]), int(k_g.shape[0])
            attn = orig(
                q_g,
                k_g,
                v_g,
                torch.tensor([0, nq], dtype=torch.int32, device=q.device),
                torch.tensor([0, nk], dtype=torch.int32, device=q.device),
                nq,
                nk,
                **extra_kwargs,
            )
            if attn.ndim == 2:
                attn = attn.view(nq, heads, dim)
            out[row].index_copy_(0, group["q"], attn.to(dtype=out.dtype))
    return out.reshape(batch * seq, heads, dim)


def install_hunyuan_flash_intercept(globals_dict: Dict[str, Any]) -> bool:
    orig = globals_dict.get("flash_attn_varlen_func")
    if not callable(orig) or getattr(orig, "_ss_hy_patched", False):
        return False

    def wrapped(q, k, v, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None, *args, **kwargs):
        module = _CURRENT_MODULE.get()
        router = getattr(module, "frame_prompt_router", None) if module is not None else None
        if router is None or max_seqlen_q is None:
            return orig(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *args, **kwargs)
        return hunyuan_scene_varlen(
            orig,
            module,
            router,
            q,
            k,
            v,
            int(max_seqlen_q),
            (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, *args),
            kwargs,
        )

    wrapped._ss_hy_patched = True
    globals_dict["flash_attn_varlen_func"] = wrapped
    return True


def _wrap_block(module: nn.Module, router: FramePromptRouter, block_idx: int) -> None:
    original = module.forward
    module.frame_prompt_router = router
    module._ss_block_idx = block_idx

    def forward(self, *args, **kwargs):
        name = type(self).__name__
        if name == "DoubleStreamBlock" and len(args) >= 2:
            self._ss_img_len = int(args[0].shape[1])
            self._ss_txt_len = int(args[1].shape[1])
        elif name == "SingleStreamBlock" and args:
            txt_len = kwargs.get("txt_len", args[2] if len(args) > 2 else None)
            if txt_len is not None:
                self._ss_txt_len = int(txt_len)
                self._ss_img_len = int(args[0].shape[1]) - int(txt_len)
        token = _CURRENT_MODULE.set(self)
        try:
            return original(*args, **kwargs)
        finally:
            _CURRENT_MODULE.reset(token)

    module.forward = types.MethodType(forward, module)


def _find_hy_transformer(model: Any) -> Any:
    seen = set()
    queue = [model]
    while queue:
        obj = queue.pop(0)
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        if type(obj).__name__ == "HYVideoDiffusionTransformer":
            return obj
        for attr in ("model", "dit", "transformer", "pipeline", "pipe", "noise_model"):
            queue.append(getattr(obj, attr, None))
        if hasattr(obj, "named_modules"):
            try:
                for _, module in obj.named_modules():
                    if type(module).__name__ == "HYVideoDiffusionTransformer":
                        return module
            except Exception:
                continue
    return model


def _iter_hy_blocks(model: Any) -> List[Any]:
    found: List[Any] = []
    for attr in ("double_blocks", "single_blocks"):
        blocks = getattr(model, attr, None)
        if blocks is None:
            continue
        try:
            found.extend(list(blocks))
        except TypeError:
            continue
    if found:
        return found
    if not hasattr(model, "named_modules"):
        return found
    for _, module in model.named_modules():
        if type(module).__name__ in HY_BLOCK_NAMES:
            found.append(module)
    return found


def install_hunyuan_text_refine(transformer: Any, router: FramePromptRouter) -> None:
    """Official exp ``_refine_text_segments``: refine each scene independently."""
    orig = getattr(transformer, "txt_in", None)
    if orig is None or getattr(transformer, "_ss_hy_txt_refined", False):
        return
    hidden = int(getattr(transformer, "hidden_size", 3072) or 3072)

    def refined(txt, t, mask=None):
        cond_segments = getattr(router, "_hy_cond_text_segments", None)
        uncond_len = getattr(router, "_hy_uncond_text_len", None)
        if not cond_segments:
            return orig(txt, t, mask)
        timestep = int(getattr(transformer, "_ss_timestep", 0) or 0)
        pieces = []
        for row in range(int(txt.shape[0])):
            if int(txt.shape[0]) == 2:
                row_mode = "global" if row == 0 and timestep < 10 else "routed"
            else:
                pass_type = str(getattr(transformer, "_ss_pass", "cond"))
                row_mode = "global" if pass_type == "uncond" and timestep < 10 else "routed"
            if row_mode == "routed":
                ranges = cond_segments
            else:
                end = int(uncond_len if uncond_len is not None else txt.shape[1])
                ranges = [(0, min(end, int(txt.shape[1])))]
            for start, end in ranges:
                start, end = int(start), min(int(end), int(txt.shape[1]))
                if end <= start:
                    continue
                pieces.append(
                    (row, start, end, orig(txt[row : row + 1, start:end], t[row : row + 1], None))
                )
        if not pieces:
            return orig(txt, t, mask)
        refined_txt = torch.zeros(
            txt.shape[0],
            txt.shape[1],
            hidden,
            dtype=pieces[0][3].dtype,
            device=txt.device,
        )
        for row, start, end, piece in pieces:
            refined_txt[row : row + 1, start:end] = piece
        return refined_txt

    transformer.txt_in = refined
    transformer._ss_hy_txt_refined = True
    print("[hunyuan-adapter] per-scene SingleTokenRefiner enabled", flush=True)


def wrap_hunyuan_transformer(model: Any, router: Optional[FramePromptRouter]) -> List[Any]:
    target = _find_hy_transformer(model)
    if router is not None:
        apply_hunyuan_official_masks(router)
    blocks = _iter_hy_blocks(target)
    print(
        f"[hunyuan-adapter] wrap target={type(target).__name__} "
        f"from={type(model).__name__} blocks={len(blocks)}",
        flush=True,
    )
    for idx, module in enumerate(blocks):
        _wrap_block(module, router, idx)
    for _name, mod in list(sys.modules.items()):
        if str(_name).endswith("models_audio") and hasattr(mod, "flash_attn_varlen_func"):
            install_hunyuan_flash_intercept(vars(mod))
    if router is not None:
        install_hunyuan_text_refine(target, router)
    if hasattr(target, "forward") and blocks:
        original = target.forward
        unseen = object()
        state = {"t_idx": 0, "last_t": unseen, "pass": 0}

        def _t_key(args, kwargs):
            t = kwargs.get("t", kwargs.get("timestep", kwargs.get("timesteps")))
            if t is None and len(args) > 1:
                t = args[1]
            if t is None:
                return None
            if torch.is_tensor(t):
                return float(t.reshape(-1)[0].detach().float().cpu())
            try:
                return float(t)
            except (TypeError, ValueError):
                return None

        def forward(self, *args, **kwargs):
            kwargs["is_cache"] = False
            t_key = _t_key(args, kwargs)
            if t_key is not None and t_key != state["last_t"]:
                if state["last_t"] is not unseen:
                    state["t_idx"] += 1
                state["last_t"] = t_key
                state["pass"] = 0
            else:
                state["pass"] += 1
            timestep_idx = state["t_idx"]
            pass_type = "uncond" if state["pass"] == 0 else "cond"
            self._ss_timestep = timestep_idx
            self._ss_pass = pass_type
            for module in blocks:
                module._ss_timestep = timestep_idx
                module._ss_pass = pass_type
            return original(*args, **kwargs)

        target.forward = types.MethodType(forward, target)
    else:
        thread_timestep(target)
    return blocks


def install_hunyuan_prompt_routing(pipeline: Any, router: FramePromptRouter, routing: Dict[str, Any], meta: Dict[str, Any]) -> None:
    """Encode per-scene LLaVA tokens and CLIP-L common_prompt like the exp port."""
    orig = pipeline.encode_prompt_audio_text_base
    segments = [PROMPT_PREFIX + str(p) for p in (routing.get("segment_prompts") or [])]
    common = meta.get("common_prompt") or routing.get("common_prompt")

    def _pack(embeds, mask):
        if mask is None:
            packed_mask = torch.ones(embeds.shape[:2], device=embeds.device, dtype=torch.int64)
            return embeds, packed_mask
        valid = mask.to(device=embeds.device, dtype=torch.bool)
        packed = torch.stack([embeds[row, valid[row]] for row in range(embeds.shape[0])], dim=0)
        packed_mask = torch.ones(
            (embeds.shape[0], packed.shape[1]), dtype=mask.dtype, device=mask.device
        )
        return packed, packed_mask

    def _pad(embeds, target_len):
        pad = target_len - embeds.shape[1]
        return torch.nn.functional.pad(embeds, (0, 0, 0, pad)) if pad > 0 else embeds

    def wrapped(*args, **kwargs):
        text_encoder = kwargs.get("text_encoder")
        pixel = kwargs.get("pixel_value_llava")
        if pixel is None and common and text_encoder is getattr(pipeline, "text_encoder_2", None):
            kwargs = dict(kwargs)
            kwargs["prompt"] = common
            print(f"[hunyuan-adapter] CLIP-L common_prompt={common!r}", flush=True)
            return orig(*args, **kwargs)
        if (
            kwargs.get("do_classifier_free_guidance")
            and pixel is not None
            and segments
            and (text_encoder is None or text_encoder is pipeline.text_encoder)
        ):
            device = kwargs.get("device")
            shared = dict(kwargs)
            # Official encode_prompt_audio_text_base only assigns negative_* when CFG=True.
            shared["do_classifier_free_guidance"] = True
            shared["prompt_embeds"] = None
            shared["negative_prompt_embeds"] = None
            shared["negative_prompt"] = kwargs.get("negative_prompt") or ""
            shared["uncond_prompt"] = kwargs.get("uncond_prompt") or shared["negative_prompt"]
            scene_embeds = []
            scene_lens = []
            neg = neg_mask = None
            for prompt in segments:
                shared["prompt"] = prompt
                embeds, neg, mask, neg_mask = orig(**shared)
                embeds, mask = _pack(embeds, mask)
                scene_embeds.append(embeds)
                scene_lens.append(int(embeds.shape[1]))
            routed = torch.cat(scene_embeds, dim=1)
            neg, neg_mask = _pack(neg, neg_mask)
            target = max(int(routed.shape[1]), int(neg.shape[1]))
            routed = _pad(routed, target)
            neg = _pad(neg, target)
            routed_mask = torch.zeros((routed.shape[0], target), dtype=neg_mask.dtype, device=device)
            routed_mask[:, : sum(scene_lens)] = 1
            neg_mask = torch.nn.functional.pad(neg_mask, (0, target - neg_mask.shape[1]))
            cursor = 0
            ranges = []
            for length in scene_lens:
                ranges.append((cursor, cursor + length))
                cursor += length
            router._hy_cond_text_segments = ranges
            router._hy_uncond_text_len = int(neg_mask.sum().item() if neg_mask is not None else neg.shape[1])
            print(
                f"[hunyuan-adapter] routed LLaVA segments={ranges} uncond_len={router._hy_uncond_text_len}",
                flush=True,
            )
            return routed, neg, routed_mask, neg_mask
        return orig(*args, **kwargs)

    pipeline.encode_prompt_audio_text_base = wrapped


class HunyuanAvatarAdapter(Adapter):
    name = "hunyuan_avatar"

    def inject(self, model: Any, router: Optional[FramePromptRouter]) -> List[Any]:
        bound = wrap_hunyuan_transformer(model, router)
        if bound:
            return bound
        self_mods, cross_mods = find_attention_modules(model)
        for module in self_mods:
            wrap_self_attention(module, router)
            bound.append(module)
        for module in cross_mods:
            wrap_cross_attention(module, router)
            bound.append(module)
        if bound:
            thread_timestep(model)
        return bound
