"""Shared FramePromptRouter implementing F1–F12 on the noisy/video axis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from . import checks
from .config import load_frame_prompt_config
from .masks import (
    downsample_subject_fg,
    downsample_subject_fg_head_union,
    identity_self_attention_mask,
    spatial_token_indices,
    subject_fg_from_background_paths,
    upgrade_context_token_indices,
)
from .policy import active_block as policy_active_block
from .policy import attention_row, normalize_schedule
from .qscale import qscale_value
from .qscale_scope import (
    default_q_scale_bg_only,
    new_scene_bounds_from_segments,
    new_scene_scale_token_mask,
)
from .ref_attn import (
    derive_fun_live_fg_from_score_diff,
    derive_live_fg_from_class_scores,
    derive_live_fg_from_class_scores_global,
    derive_live_fg_from_class_scores_numpy,
    derive_live_fg_from_class_scores_s2v,
    derive_live_fg_from_union_scores,
    live_bg_from_subject_fg,
    postprocess_live_fg,
)


def scene_frame_bounds(num_frames: int, frame_splits: Sequence[int]) -> List[Tuple[int, int]]:
    bounds = [0, *[int(v) for v in frame_splits], int(num_frames)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def resolve_transition_context_windows(
    segment_bounds: Sequence[Tuple[int, int]],
    len_trans: int,
    len_trans_context: int,
) -> List[Dict[str, Any]]:
    if int(len_trans) <= 0:
        return []
    windows = []
    for new_segment in range(1, len(segment_bounds)):
        split = segment_bounds[new_segment][0]
        prev_start, _ = segment_bounds[new_segment - 1]
        _, new_end = segment_bounds[new_segment]
        queries = list(range(split, min(new_end, split + int(len_trans))))
        context = list(range(max(prev_start, split - int(len_trans_context)), split))
        if queries and context:
            windows.append(
                {
                    "new_segment": new_segment,
                    "query_frames": queries,
                    "context_frames": context,
                }
            )
    return windows


class FramePromptRouter:
    def __init__(
        self,
        config: Dict[str, Any],
        *,
        backend: str = "multitalk",
        patch_height: int = 2,
        patch_width: int = 2,
        lat_h: Optional[int] = None,
        lat_w: Optional[int] = None,
        num_frames: Optional[int] = None,
        frame_splits: Optional[Sequence[int]] = None,
        enabled: Optional[bool] = None,
        self_attention_mask=None,
        attention_control_schedule=None,
        len_trans: Optional[int] = None,
        len_trans_context: Optional[int] = None,
        trans_timestep_range=None,
        segment_prompt_start_timestep: Optional[int] = None,
    ):
        cfg = dict(config)
        if enabled is not None:
            cfg["enabled"] = enabled
        if num_frames is not None:
            cfg["num_frames"] = num_frames
        if frame_splits is not None:
            cfg["frame_splits"] = list(frame_splits)
        if len_trans is not None:
            cfg["len_trans"] = len_trans
        if len_trans_context is not None:
            cfg["len_trans_context"] = len_trans_context
        if trans_timestep_range is not None:
            cfg["trans_timestep_range"] = trans_timestep_range
        if segment_prompt_start_timestep is not None:
            cfg["segment_prompt_start_timestep"] = segment_prompt_start_timestep
        if attention_control_schedule is not None:
            cfg["attention_control_schedule"] = attention_control_schedule
        if self_attention_mask is not None and not cfg.get("attention_control_schedule"):
            cfg["attention_control_schedule"] = [
                {
                    "timestep_range": [0, 10**9],
                    "self_attention_mask": self_attention_mask,
                    "params": {},
                }
            ]

        self.backend = backend
        self.enabled = bool(cfg.get("enabled", True))
        self.config = cfg
        self.num_frames = int(cfg["num_frames"])
        self.frame_splits = [int(v) for v in cfg.get("frame_splits", [])]
        self.segment_bounds = scene_frame_bounds(self.num_frames, self.frame_splits)
        self.num_segments = len(self.segment_bounds)
        self.frame_to_segment = [
            next(i for i, (start, end) in enumerate(self.segment_bounds) if start <= frame < end)
            for frame in range(self.num_frames)
        ]
        self.segment_prompt_start_timestep = int(cfg.get("segment_prompt_start_timestep", 0))
        self.patch_height = int(lat_h if lat_h is not None else patch_height)
        self.patch_width = int(lat_w if lat_w is not None else patch_width)
        self.spatial_tokens = self.patch_height * self.patch_width
        self.text_embeddings: Optional[List[torch.Tensor]] = None

        self.len_trans = int(cfg.get("len_trans", 0) or 0)
        self.len_trans_context = int(cfg.get("len_trans_context", 0) or 0)
        raw_range = cfg.get("trans_timestep_range")
        self.trans_timestep_range = (
            None if raw_range is None else (int(raw_range[0]), int(raw_range[1]))
        )
        if self.len_trans < 0 or self.len_trans_context < 0:
            raise ValueError("len_trans and len_trans_context must be >= 0")
        if bool(self.len_trans) != bool(self.len_trans_context):
            raise ValueError(
                "len_trans and len_trans_context must both be positive or both zero"
            )
        if self.trans_timestep_range is not None:
            if not self.len_trans:
                raise ValueError("trans_timestep_range requires transition lengths")
            if self.trans_timestep_range[1] < self.trans_timestep_range[0]:
                raise ValueError("trans_timestep_range requires START <= END")

        schedule = cfg.get("attention_control_schedule")
        if schedule is None:
            mask = self_attention_mask or identity_self_attention_mask(self.num_segments)
            schedule = [
                {
                    "timestep_range": [0, 10**9],
                    "self_attention_mask": mask,
                    "params": {},
                }
            ]
        self.schedule = normalize_schedule(schedule, self.num_segments)

        self.subject_fg_masks: Optional[torch.Tensor] = None
        self.raw_subject_fg: Optional[torch.Tensor] = None
        self.raw_head_fg: Optional[torch.Tensor] = None
        self.static_bg: Optional[torch.Tensor] = None
        self.live_bg: Optional[torch.Tensor] = None
        self.live_subject_fg: Optional[torch.Tensor] = None
        self.audio_start_timestep = int(cfg.get("audio_start_timestep", 0) or 0)
        self.ref_loc_max_timesteps = int(cfg.get("ref_loc_max_timesteps", 15) or 0)
        self.freeze_live_mask = bool(cfg.get("freeze_live_mask", False))
        self.null_text_embeddings: Optional[Any] = None
        paths = cfg.get("background_mask_path")
        if paths:
            raw_fg = subject_fg_from_background_paths(paths, self.num_frames)
            self.raw_subject_fg = raw_fg
            head_paths = cfg.get("head_mask_path")
            if head_paths:
                self.raw_head_fg = subject_fg_from_background_paths(
                    head_paths, self.num_frames
                )
            self.rebuild_latent_masks()

        self._selector_scores: Dict[Any, Any] = {}
        self._pending_selector_scores: Dict[Any, Any] = {}
        self._ref_loc_scores: Dict[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._pending_ref_loc_scores: Dict[
            Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._s2v_frozen_live_masks: Dict[
            Tuple[Any, ...], Tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def rebuild_latent_masks(self) -> None:
        """Downsample stored pixel masks onto the current latent grid."""
        raw = self.raw_subject_fg
        if raw is None:
            return
        center_crop = self.backend == "wan_s2v"
        s2v_defaults = self.backend == "wan_s2v"
        cover_original = bool(
            self.config.get("input_mask_cover_original", s2v_defaults)
        )
        cover_head_only = bool(
            self.config.get(
                "input_mask_cover_head_only",
                s2v_defaults and self.raw_head_fg is not None,
            )
        )
        if cover_head_only and not cover_original:
            cover_head_only = False
        if cover_head_only:
            if self.raw_head_fg is None:
                raise ValueError(
                    "input_mask_cover_head_only=True requires head_mask_path "
                    '{"0": ..., "1": ...} with SAM3 head BG masks.'
                )
            self.subject_fg_masks, self.static_bg = downsample_subject_fg_head_union(
                raw,
                self.raw_head_fg,
                self.patch_height,
                self.patch_width,
                center_crop=center_crop,
            )
            return
        self.subject_fg_masks, self.static_bg = downsample_subject_fg(
            raw,
            self.patch_height,
            self.patch_width,
            center_crop=center_crop,
            cover_original=cover_original,
        )

    def set_text_embeddings(self, embeddings: List[torch.Tensor]) -> None:
        if len(embeddings) != self.num_segments:
            raise ValueError(f"expected {self.num_segments} segment embeddings")
        self.text_embeddings = embeddings

    def set_background_mask(self, background: torch.Tensor) -> None:
        """``background`` is bool ``[F, HW]`` with True=BG."""
        if background.ndim == 3:
            background = background.reshape(background.shape[0], -1)
        self.static_bg = background.bool()
        self.live_bg = None

    def scene_frame_bounds(self) -> List[Tuple[int, int]]:
        return list(self.segment_bounds)

    def segment_ranges(self, spatial_tokens: Optional[int] = None) -> List[Tuple[int, int, int]]:
        L = int(self.spatial_tokens if spatial_tokens is None else spatial_tokens)
        return [(a * L, b * L, i) for i, (a, b) in enumerate(self.segment_bounds)]

    def get_transition_context_windows(self) -> List[Dict[str, Any]]:
        return resolve_transition_context_windows(
            self.segment_bounds, self.len_trans, self.len_trans_context
        )

    resolve_transition_context_windows = get_transition_context_windows

    def is_transition_latent_frame(self, frame: int) -> bool:
        return any(int(frame) in window["query_frames"] for window in self.get_transition_context_windows())

    def should_apply_transition_context(self, timestep_idx: Optional[int]) -> bool:
        if self.len_trans <= 0:
            return False
        if self.trans_timestep_range is None:
            return True
        if timestep_idx is None:
            raise ValueError("trans_timestep_range requires timestep_idx")
        lo, hi = self.trans_timestep_range
        return lo <= int(timestep_idx) <= hi

    def should_use_segment_prompts(self, timestep_idx: int, pass_type: str = "cond") -> bool:
        if pass_type == "uncond":
            return False
        if self.backend == "hunyuan_avatar":
            return True
        return int(timestep_idx) >= self.segment_prompt_start_timestep

    def selected_text_index(
        self, timestep_idx: int, segment: int, pass_type: str = "cond"
    ) -> Optional[int]:
        if pass_type == "uncond":
            return None
        if not self.should_use_segment_prompts(timestep_idx, pass_type):
            return 0
        return int(segment)

    def active_block(
        self,
        timestep_idx: int,
        block_idx: Optional[int] = None,
        pass_type: str = "cond",
    ) -> Optional[Dict[str, Any]]:
        return policy_active_block(self.schedule, timestep_idx, block_idx, pass_type)

    def get_active_attention_control_block(
        self,
        timestep_idx: int,
        block_idx: Optional[int] = None,
        pass_type: str = "cond",
    ) -> Optional[Dict[str, Any]]:
        return self.active_block(timestep_idx, block_idx, pass_type)

    def get_active_attention_control_blocks(
        self,
        timestep_idx: int,
        pass_type: str = "cond",
    ) -> List[Dict[str, Any]]:
        block = self.active_block(int(timestep_idx), None, pass_type)
        return [] if block is None else [block]

    def should_route(self) -> bool:
        return bool(self.enabled)

    def should_apply_scene_mask(
        self,
        timestep_idx: Optional[int],
        block_idx: Optional[int] = None,
        pass_type: str = "cond",
    ) -> bool:
        if timestep_idx is None:
            return False
        return self.active_block(int(timestep_idx), block_idx, pass_type) is not None

    def should_apply_schedule_to_block(
        self,
        timestep_idx: Optional[int],
        block_idx: Optional[int],
        pass_type: str = "cond",
    ) -> bool:
        return self.should_apply_scene_mask(timestep_idx, block_idx, pass_type)

    def get_self_attention_mask(
        self,
        timestep_idx: int,
        block_idx: Optional[int] = None,
        pass_type: str = "cond",
    ):
        block = self.active_block(timestep_idx, block_idx, pass_type)
        return None if block is None else block["_mask"]

    def attention_row(self, block: Dict[str, Any], segment: int, query_region: Optional[str] = None):
        return attention_row(block, segment, query_region)

    def _key_row(self, block: Dict[str, Any], segment: int, query_region: Optional[str]):
        """Schedule row for key gathering. A collapsed split uses the reference merge."""
        row = block["_mask"][int(segment)]
        if isinstance(row, dict) and query_region is None:
            merged = []
            for fg, bg in zip(row["fg"], row["bg"]):
                fg_on, fg_region = fg
                bg_on, bg_region = bg
                if fg_on == 0 and bg_on == 0:
                    merged.append((0, "full"))
                elif fg_on == 1 and bg_on == 1:
                    merged.append((1, fg_region if fg_region == bg_region else "full"))
                else:
                    merged.append((1, fg_region if fg_on == 1 else bg_region))
            return merged
        return self.attention_row(block, segment, query_region)

    def background_mask(self, device: torch.device) -> Optional[torch.Tensor]:
        mask = self.live_bg if self.live_bg is not None else self.static_bg
        return None if mask is None else mask.to(device=device)

    def spatial_indices(
        self,
        segment: int,
        region: str,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        device = device or torch.device("cpu")
        start, end = self.segment_bounds[int(segment)]
        return spatial_token_indices(
            start, end, self.spatial_tokens, region, self.background_mask(device), device
        )

    def noisy_key_indices(
        self,
        block: Dict[str, Any],
        query_segment: int,
        query_region: Optional[str],
        device: Optional[torch.device] = None,
        *,
        timestep_idx: Optional[int] = None,
        include_transition: bool = True,
    ) -> torch.Tensor:
        """Noisy-axis K/V indices. Adapters union extras (FramePack, text)."""
        device = device or torch.device("cpu")
        row = self._key_row(block, int(query_segment), query_region)
        bg = self.background_mask(device)
        context_frames: List[int] = []
        if (
            include_transition
            and timestep_idx is not None
            and self.should_apply_transition_context(timestep_idx)
        ):
            for window in self.get_transition_context_windows():
                if int(window["new_segment"]) == int(query_segment):
                    context_frames.extend(int(frame) for frame in window["context_frames"])
        prev_segment = int(query_segment) - 1
        selected = []
        for key_segment, (active, region) in enumerate(row):
            if not active:
                continue
            start, end = self.segment_bounds[key_segment]
            local = spatial_token_indices(
                start, end, self.spatial_tokens, region, bg, device
            )
            absolute = local.to(device=device, dtype=torch.long) + int(start) * int(
                self.spatial_tokens
            )
            if context_frames and int(key_segment) == prev_segment and str(region) in ("fg", "bg"):
                absolute = upgrade_context_token_indices(
                    absolute, context_frames, self.spatial_tokens
                )
            if str(region) in ("fg", "bg") and int(absolute.numel()) == 0:
                # MultiTalk keeps one token so the flash call still runs.
                # InfiniteTalk omits an empty key segment.
                if self.backend == "multitalk":
                    absolute = torch.tensor(
                        [int(start) * int(self.spatial_tokens)],
                        device=device,
                        dtype=torch.long,
                    )
                else:
                    continue
            selected.append(absolute)
        nonempty = [idx for idx in selected if idx.numel()]
        if not nonempty:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.unique(torch.cat(nonempty), sorted=True)

    def qscale_token_mask(self, block: Optional[Dict[str, Any]]) -> torch.Tensor:
        """F9+F10: which noisy tokens receive a scale != 1."""
        num_tokens = self.num_frames * self.spatial_tokens
        empty = torch.zeros(num_tokens, dtype=torch.bool)
        if block is None:
            return empty
        params = block.get("params") or {}
        if not bool(params.get("auto_q_scale", False)):
            return empty
        bg_only = params.get("q_scale_bg_only", params.get("q_scale_new_scene_bg_only"))
        if bg_only is None:
            bg_only = default_q_scale_bg_only(self.backend)
        bg = self.static_bg if self.live_bg is None else self.live_bg
        return new_scene_scale_token_mask(
            num_tokens,
            self.spatial_tokens,
            new_scene_bounds_from_segments(self.segment_bounds),
            bg_only=bool(bg_only),
            background_by_frame=bg,
        )

    def qscale_factor(self, segment: int, block: Optional[Dict[str, Any]]) -> float:
        if block is None or int(segment) < 1:
            return 1.0
        params = block.get("params") or {}
        if not bool(params.get("auto_q_scale", False)):
            return 1.0
        start, end = self.segment_bounds[int(segment)]
        override = float(params["q_scale"]) if "q_scale" in params else None
        return qscale_value(end - start, self.backend, override)

    def apply_q_scale(self, q: torch.Tensor, block: Optional[Dict[str, Any]]) -> torch.Tensor:
        token_mask = self.qscale_token_mask(block)
        if not bool(token_mask.any()):
            return q
        # Use the first new-scene factor when F9 is on (one factor per scene applied below).
        scaled = q.clone()
        params = block.get("params") or {}
        override = float(params["q_scale"]) if "q_scale" in params else None
        for segment in range(1, self.num_segments):
            start, end = self.segment_bounds[segment]
            factor = qscale_value(end - start, self.backend, override)
            sl = slice(start * self.spatial_tokens, end * self.spatial_tokens)
            local = token_mask[sl]
            if local.any():
                global_idx = (start * self.spatial_tokens) + torch.where(local)[0]
                scaled[:, global_idx] = scaled[:, global_idx] * factor
        return scaled

    def update_live_mask_from_scores(
        self,
        scores: torch.Tensor,
        *,
        drop_low_percent: float = 20.0,
        min_component_size: int = 20,
        hole_size: int = 0,
        bg_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if bool(self.freeze_live_mask) and self.live_bg is not None:
            return self.live_bg
        if bg_scores is not None:
            return self.update_live_mask_from_class_scores(
                scores,
                bg_scores,
                drop_low_percent=drop_low_percent,
                min_component_size=min_component_size,
                hole_size=hole_size,
            )
        value = scores.detach()
        if value.ndim == 2:
            value = value.unsqueeze(0)
        if value.ndim != 3:
            raise ValueError(f"expected live scores [F,HW] or [S,F,HW], got {tuple(scores.shape)}")
        live_fg = torch.stack(
            [
                postprocess_live_fg(
                    value[subject],
                    drop_low_percent=drop_low_percent,
                    min_component_size=min_component_size,
                    hole_size=hole_size,
                )
                for subject in range(int(value.shape[0]))
            ],
            dim=0,
        )
        self.live_subject_fg = live_fg
        self.live_bg = live_bg_from_subject_fg(self.live_subject_fg)
        return self.live_bg

    def update_live_mask_from_class_scores(
        self,
        fg_scores: torch.Tensor,
        bg_scores: torch.Tensor,
        *,
        drop_low_percent: float = 20.0,
        min_component_size: int = 20,
        hole_size: int = 0,
    ) -> torch.Tensor:
        if bool(self.freeze_live_mask) and self.live_bg is not None:
            return self.live_bg
        if self.backend == "multitalk":
            derive = derive_live_fg_from_class_scores_numpy
        elif self.backend == "wan_s2v":
            derive = derive_live_fg_from_class_scores_s2v
        elif self.backend == "wan_animate":
            derive = derive_live_fg_from_class_scores_global
        else:
            derive = derive_live_fg_from_class_scores
        live_fg = derive(
            fg_scores,
            bg_scores,
            self.patch_height,
            self.patch_width,
            drop_low_percent=drop_low_percent,
            min_component_size=min_component_size,
            hole_size=hole_size,
        )
        if live_fg is None:
            return self.update_live_mask_from_scores(
                fg_scores,
                drop_low_percent=drop_low_percent,
                min_component_size=min_component_size,
                hole_size=hole_size,
            )
        self.live_subject_fg = live_fg
        self.live_bg = live_bg_from_subject_fg(self.live_subject_fg)
        return self.live_bg

    def store_ref_loc_head(
        self,
        timestep_idx: int,
        layer_idx: int,
        head_idx: int,
        fg: torch.Tensor,
        bg: torch.Tensor,
    ) -> None:
        target = (
            self._pending_ref_loc_scores
            if self.backend
            in {"multitalk", "infinitetalk", "wan_s2v", "wan_animate", "wan_fun_camera"}
            else self._ref_loc_scores
        )
        target[(int(timestep_idx), int(layer_idx), int(head_idx))] = (
            fg.detach().float().cpu(),
            bg.detach().float().cpu(),
        )

    def latest_ref_loc_timestep(self, timestep_idx: int) -> Optional[int]:
        times = {ts for ts, _, _ in self._ref_loc_scores if int(ts) <= int(timestep_idx)}
        return max(times) if times else None

    def selector_avg_scores(
        self,
        timestep_idx: int,
        selectors: Sequence[Sequence[int]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        latest = self.latest_ref_loc_timestep(timestep_idx)
        if latest is None:
            return None, None
        fgs: List[torch.Tensor] = []
        bgs: List[torch.Tensor] = []
        for pair in selectors:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            key = (int(latest), int(pair[0]), int(pair[1]))
            stored = self._ref_loc_scores.get(key)
            if stored is None:
                continue
            fgs.append(stored[0])
            bgs.append(stored[1])
        if not fgs:
            return None, None
        return torch.stack(fgs, dim=0).mean(0), torch.stack(bgs, dim=0).mean(0)

    def refresh_live_mask_from_stored(
        self,
        timestep_idx: int,
        block: Optional[Dict[str, Any]],
    ) -> Optional[torch.Tensor]:
        if block is None:
            return self.live_bg
        params = block.get("params") or {}
        if str(params.get("mask_type", "")) != "ref_attn":
            return self.live_bg
        selectors = params.get("ref_attn_mask_selectors") or []
        s2v_mask_key = None
        if self.backend == "wan_s2v" and self.ref_loc_max_timesteps > 0:
            s2v_mask_key = (
                tuple(
                    tuple(int(value) for value in selector[:2])
                    for selector in selectors
                    if isinstance(selector, (list, tuple)) and len(selector) >= 2
                ),
                float(params.get("ref_attn_mask_drop_low_percent", 10.0)),
                int(params.get("ref_attn_mask_min_component_size", 20)),
                int(params.get("ref_attn_mask_hole_size", 0)),
            )
            frozen = self._s2v_frozen_live_masks.get(s2v_mask_key)
            if frozen is not None and int(timestep_idx) >= self.ref_loc_max_timesteps:
                self.live_bg, self.live_subject_fg = frozen
                return self.live_bg
        fg, bg = (None, None)
        score_timestep = None
        if selectors:
            score_timestep = self.latest_ref_loc_timestep(timestep_idx)
            fg, bg = self.selector_avg_scores(timestep_idx, selectors)
        if fg is None or bg is None:
            return self.live_bg
        if self.backend == "fantasytalking":
            union_fg = fg[0] if fg.ndim == 3 and int(fg.shape[0]) == 1 else fg
            live_fg = derive_live_fg_from_union_scores(
                union_fg,
                bg,
                self.patch_height,
                self.patch_width,
                drop_low_percent=float(
                    params.get("ref_attn_mask_drop_low_percent", 10.0)
                ),
                min_component_size=int(
                    params.get("ref_attn_mask_min_component_size", 20)
                ),
                hole_size=int(params.get("ref_attn_mask_hole_size", 0)),
            )
            if live_fg is None:
                return self.live_bg
            self.live_subject_fg = live_fg
            self.live_bg = live_bg_from_subject_fg(live_fg)
            return self.live_bg
        live_bg = self.update_live_mask_from_class_scores(
            fg,
            bg,
            drop_low_percent=float(params.get("ref_attn_mask_drop_low_percent", 10.0)),
            min_component_size=int(params.get("ref_attn_mask_min_component_size", 20)),
            hole_size=int(params.get("ref_attn_mask_hole_size", 0)),
        )
        if (
            s2v_mask_key is not None
            and score_timestep is not None
            and score_timestep < self.ref_loc_max_timesteps
            and self.live_subject_fg is not None
        ):
            self._s2v_frozen_live_masks[s2v_mask_key] = (
                live_bg,
                self.live_subject_fg,
            )
        return live_bg

    def install_fun_ref_loc_mask(
        self,
        timestep_idx: int,
        block: Optional[Dict[str, Any]],
    ) -> Optional[torch.Tensor]:
        """Install one Fun mask from averaged selector scores.

        Feature keeps a token in the background when ``fg == bg``, then drops
        the low tail of ``fg - bg``. The shared argmax path treats that tie as
        foreground and changes the next step's q-scale.
        """
        if block is None:
            return self.live_bg
        params = block.get("params") or {}
        if str(params.get("mask_type", "")) != "ref_attn":
            return self.live_bg
        selectors = params.get("ref_attn_mask_selectors") or []
        fg, bg = self.selector_avg_scores(int(timestep_idx), selectors) if selectors else (None, None)
        if fg is None or bg is None:
            return self.live_bg
        fg_map = fg[0] if fg.ndim == 3 else fg
        bg_map = bg[0] if bg.ndim == 3 else bg
        live_fg = derive_fun_live_fg_from_score_diff(
            fg_map,
            bg_map,
            int(self.patch_height),
            int(self.patch_width),
            drop_low_percent=float(params.get("ref_attn_mask_drop_low_percent", 10.0)),
            min_component_size=int(params.get("ref_attn_mask_min_component_size", 20)),
            hole_size=int(params.get("ref_attn_mask_hole_size", 0) or 0),
            ext_size=int(params.get("ref_attn_mask_ext_size", 0) or 0),
        )
        if live_fg is None or int(live_fg.sum().item()) <= 0:
            return self.live_bg
        self.live_subject_fg = live_fg.unsqueeze(0)
        self.live_bg = ~live_fg
        return self.live_bg

    def ref_attn_timestep_window(self) -> Optional[Tuple[int, int]]:
        """Inclusive ``(t_min, t_max)`` over ref-attn schedule blocks."""
        bounds = []
        for block in self.schedule or []:
            params = (block or {}).get("params") or {}
            if str(params.get("mask_type", "")).strip().lower() != "ref_attn":
                continue
            span = block.get("timestep_range")
            if isinstance(span, (list, tuple)) and len(span) == 2:
                bounds.append((int(span[0]), int(span[1])))
        if not bounds:
            return None
        return min(lo for lo, _hi in bounds), max(hi for _lo, hi in bounds)

    def publish_ref_loc_scores(self) -> None:
        """Make this forward's ref-loc scores visible to the next forward.

        MultiTalk publishes at the end of the model forward, so later layers
        in that forward still use the previous step. InfiniteTalk does the
        same during the conditional pass, then the unconditional pass sees
        the scores collected there.
        """
        if self._pending_ref_loc_scores:
            self._ref_loc_scores.update(self._pending_ref_loc_scores)
            self._pending_ref_loc_scores = {}

    def complete_denoising_step(self) -> None:
        """Publish backend scores only after every CFG pass in the step."""
        self._selector_scores = dict(self._pending_selector_scores)
        self._pending_selector_scores = {}
        self.publish_ref_loc_scores()
        if self.backend == "wan_fun_camera":
            # Feature installs one mask after the forward that collected it.
            # The next step keeps that mask; this step still used the previous one.
            finished = getattr(self, "_ss_timestep", None)
            window = self.ref_attn_timestep_window()
            if (
                finished is not None
                and window is not None
                and window[0] <= int(finished) <= window[1]
            ):
                block = self.active_block(int(finished), None, "cond")
                self.install_fun_ref_loc_mask(int(finished), block)
            return
        if self.backend in {"wan_s2v", "wan_animate"}:
            self.live_bg = None
            self.live_subject_fg = None

    def observe_ref_attention(self, key: Any, scores: Any) -> None:
        self._pending_selector_scores[key] = scores


def create_router(
    config: Dict[str, Any] | str,
    *,
    backend: str = "multitalk",
    patch_height: int = 2,
    patch_width: int = 2,
    lat_h: Optional[int] = None,
    lat_w: Optional[int] = None,
    validate: bool = True,
    **kwargs,
) -> Optional[FramePromptRouter]:
    if isinstance(config, str):
        routing = load_frame_prompt_config(config, backend=backend, validate=validate)
    else:
        routing = dict(config)
        if validate:
            checks.validate_config(routing, backend=backend)
    if not routing.get("enabled", False):
        return None
    return FramePromptRouter(
        routing,
        backend=backend,
        patch_height=patch_height,
        patch_width=patch_width,
        lat_h=lat_h,
        lat_w=lat_w,
        **kwargs,
    )


create_frame_prompt_router_from_config = create_router
