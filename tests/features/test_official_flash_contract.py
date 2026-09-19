import pytest
import torch
import torch.nn as nn

from sceneshifter.adapters import base as base_adapter
from sceneshifter.adapters.base import (
    _grid_fhw,
    _maybe_update_spatial,
    _ref_loc_query_for_scene_mask,
    _rerope_queries,
    _should_local_rerope,
    project_router_text_embeddings,
    scene_masked_official_flash,
    wrap_audio_attention,
    wrap_cross_attention,
)
from sceneshifter.router import FramePromptRouter
from tests.conftest import canonical_config


class _Attn(nn.Module):
    pass


def test_rerope_queries_skips_list_grid_sizes():
    router = FramePromptRouter(canonical_config(), patch_height=2, patch_width=2)
    module = _Attn()
    module._ss_rope_fn = lambda *args, **kwargs: args[0]
    module._ss_q_pre_rope = torch.randn(1, 32, 2, 2)
    module._ss_freqs = torch.randn(2)
    module._ss_grid_sizes = [[8, 2, 2]]
    q = torch.randn(1, 32, 2, 2)
    q_idx = torch.arange(4)
    out = _rerope_queries(module, router, {"_mask": []}, q, q_idx, 0)
    assert out.shape[1] == 4


def test_local_rerope_policy_is_backend_specific():
    cfg = canonical_config()
    cfg["attention_control_schedule"][0]["self_attention_mask"] = [
        [[1, "full"], [0, "full"]],
        [[0, "full"], [1, "full"]],
    ]
    for backend in ("infinitetalk", "fantasytalking", "wan_s2v", "wan_animate"):
        router = FramePromptRouter(cfg, backend=backend, patch_height=2, patch_width=2)
        block = router.active_block(3, 0, "cond")
        assert block is not None
        assert not _should_local_rerope(router, block, 1, None)

    fun = FramePromptRouter(cfg, backend="wan_fun_camera", patch_height=2, patch_width=2)
    fun_block = fun.active_block(3, 0, "cond")
    assert fun_block is not None
    assert _should_local_rerope(fun, fun_block, 1, None)

    multitalk = FramePromptRouter(cfg, backend="multitalk", patch_height=2, patch_width=2)
    mt_block = multitalk.active_block(3, 0, "cond")
    assert mt_block is not None
    assert _should_local_rerope(multitalk, mt_block, 1, None)


def test_grid_fhw_flattens_s2v_nonscalar_row():
    router = FramePromptRouter(canonical_config(), patch_height=2, patch_width=2)
    # int(row[0]) raises when each axis is a multi-element tensor.
    grid = [torch.tensor([[21, 44, 64], [21, 44, 64]])]
    assert _grid_fhw(grid, router) == (21, 44, 64)
    assert _grid_fhw(torch.tensor([[8, 2, 2]]), router) == (8, 2, 2)


def test_s2v_nested_grid_does_not_infer_spatial_from_extra_tokens():
    router = FramePromptRouter(
        canonical_config(num_frames=21),
        backend="wan_s2v",
        patch_height=44,
        patch_width=64,
    )
    grid = torch.tensor([[21, 44, 64]])
    module = _Attn()
    module._ss_grid_sizes = [[torch.zeros_like(grid), grid, grid]]
    q = torch.zeros(1, 61952, 1, 1)
    assert _grid_fhw(module._ss_grid_sizes, router) == (21, 44, 64)
    _maybe_update_spatial(router, module, q)
    assert (router.patch_height, router.patch_width, router.spatial_tokens) == (
        44,
        64,
        2816,
    )


def test_unrestricted_flash_keeps_k_lens():
    cfg = canonical_config()
    cfg["attention_control_schedule"][0]["timestep_range"] = [10, 20]
    router = FramePromptRouter(cfg, patch_height=2, patch_width=2)
    captured = {}

    def fake_flash(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return args[0]

    q = torch.randn(1, 32, 2, 2)
    k = q.clone()
    v = q.clone()
    k_lens = torch.tensor([32])
    module = _Attn()
    module.frame_prompt_router = router
    module._ss_timestep = 0
    module._ss_block_idx = 0
    module._ss_pass = "cond"
    out = scene_masked_official_flash(
        fake_flash,
        "flash_attention",
        module,
        router,
        (q, k, v),
        {"k_lens": k_lens},
    )
    assert out is q
    assert captured["kwargs"]["k_lens"] is k_lens


def test_scene_mask_flash_drops_packed_lens():
    router = FramePromptRouter(canonical_config(), patch_height=2, patch_width=2)
    captured = []

    def fake_flash(q, k, v, *args, **kwargs):
        captured.append(kwargs.get("k_lens", "missing"))
        return q

    q = torch.randn(1, 32, 2, 2)
    k = q.clone()
    v = q.clone()
    module = _Attn()
    module.frame_prompt_router = router
    module._ss_timestep = 3
    module._ss_block_idx = 0
    module._ss_pass = "cond"
    scene_masked_official_flash(
        fake_flash,
        "flash_attention",
        module,
        router,
        (q, k, v),
        {"k_lens": torch.tensor([32])},
    )
    assert captured
    assert all(item is None or item == "missing" for item in captured)


def test_s2v_scene_mask_preserves_reference_and_framepack_contract(bg_mask_path):
    cfg = canonical_config(
        background_mask_path=bg_mask_path,
        len_trans=0,
        len_trans_context=0,
        trans_timestep_range=None,
    )
    cfg["attention_control_schedule"][0]["params"]["auto_q_scale"] = False
    router = FramePromptRouter(cfg, backend="wan_s2v", patch_height=2, patch_width=2)
    captured = []

    def fake_flash(q, k, v, *args, **kwargs):
        captured.append(
            {
                "q": q[0, :, 0, 0].to(torch.int64).tolist(),
                "k": k[0, :, 0, 0].to(torch.int64).tolist(),
                "k_lens": kwargs.get("k_lens"),
            }
        )
        return q

    # [32 noisy | 4 immutable reference | 3 FramePack]
    values = torch.arange(39, dtype=torch.float32).view(1, 39, 1, 1)
    module = _Attn()
    module.frame_prompt_router = router
    module._ss_timestep = 3
    module._ss_block_idx = 0
    module._ss_pass = "cond"
    scene_masked_official_flash(
        fake_flash,
        "flash_attention",
        module,
        router,
        (values, values.clone(), values.clone()),
        {"k_lens": torch.tensor([39])},
    )

    scene0, scene1, framepack = captured
    assert scene0["q"] == [*range(12), *range(32, 36)]
    assert scene0["k"] == [*range(12), *range(32, 39)]
    assert scene1["q"] == list(range(12, 32))
    assert scene1["k"] == [0, 4, 8, *range(12, 32), 32, 36, 37, 38]
    assert framepack["q"] == [36, 37, 38]
    assert framepack["k"] == list(range(39))
    assert torch.equal(framepack["k_lens"], torch.tensor([39]))


def test_project_raw_t5_through_text_embedding():
    router = FramePromptRouter(canonical_config(), patch_height=2, patch_width=2)
    router._text_embedding = nn.Sequential(
        nn.Linear(8, 16, bias=False),
        nn.GELU(),
        nn.Linear(16, 16, bias=False),
    )
    router._text_len = 4
    router.set_text_embeddings([torch.randn(3, 8), torch.randn(2, 8)])
    project_router_text_embeddings(router)
    assert router.text_embeddings[0].shape == (1, 4, 16)
    assert router.text_embeddings[1].shape == (1, 4, 16)
    # Second call is a no-op once embeddings are already in model space.
    before = router.text_embeddings[0].clone()
    project_router_text_embeddings(router)
    assert torch.equal(router.text_embeddings[0], before)


def test_audio_residual_skipped_before_start():
    cfg = canonical_config()
    cfg["audio_start_timestep"] = 20
    router = FramePromptRouter(cfg, patch_height=2, patch_width=2)

    class Audio(nn.Module):
        def forward(self, x, **kwargs):
            return x + 1

    module = Audio()
    wrap_audio_attention(module, router)
    module._ss_timestep = 3
    x = torch.zeros(1, 4, 2)
    assert torch.equal(module(x), torch.zeros_like(x))
    module._ss_timestep = 20
    assert torch.equal(module(x), x + 1)


def test_unrestricted_collects_ref_loc_without_updating_mask():
    cfg = canonical_config()
    cfg["attention_control_schedule"][0]["timestep_range"] = [3, 39]
    cfg["attention_control_schedule"][0]["params"] = {
        "mask_type": "ref_attn",
        "ref_attn_mask_selectors": [[0, 0]],
        "ref_attn_mask_drop_low_percent": 0.0,
        "ref_attn_mask_min_component_size": 1,
        "ref_attn_mask_hole_size": 0,
    }
    cfg["background_mask_path"] = None
    router = FramePromptRouter(cfg, patch_height=2, patch_width=2)
    router.subject_fg_masks = torch.zeros(1, 8, 4, dtype=torch.bool)
    router.subject_fg_masks[0, :, 0] = True
    router.static_bg = ~router.subject_fg_masks.any(0)
    captured = {}

    def fake_flash(*args, **kwargs):
        captured["k_lens"] = kwargs.get("k_lens")
        return args[0]

    q = torch.randn(1, 32, 2, 4)
    k = q.clone()
    v = q.clone()
    k_lens = torch.tensor([32])
    module = _Attn()
    module.frame_prompt_router = router
    module._ss_timestep = 0
    module._ss_block_idx = 0
    module._ss_pass = "cond"
    module._ss_grid_sizes = torch.tensor([[8, 2, 2]])
    scene_masked_official_flash(
        fake_flash,
        "flash_attention",
        module,
        router,
        (q, k, v),
        {"k_lens": k_lens},
    )
    assert captured["k_lens"] is k_lens
    assert router.live_bg is None
    assert (0, 0, 0) in router._ref_loc_scores


def test_freeze_live_mask_still_collects_scores():
    cfg = canonical_config()
    cfg["freeze_live_mask"] = True
    cfg["attention_control_schedule"][0]["timestep_range"] = [3, 39]
    cfg["attention_control_schedule"][0]["params"] = {
        "mask_type": "ref_attn",
        "ref_attn_mask_selectors": [[0, 0]],
        "ref_attn_mask_drop_low_percent": 0.0,
        "ref_attn_mask_min_component_size": 1,
        "ref_attn_mask_hole_size": 0,
    }
    cfg["background_mask_path"] = None
    router = FramePromptRouter(cfg, backend="fantasytalking", patch_height=2, patch_width=2)
    router.subject_fg_masks = torch.zeros(1, 8, 4, dtype=torch.bool)
    router.subject_fg_masks[0, :, 0] = True
    router.static_bg = ~router.subject_fg_masks.any(0)
    router.ref_loc_max_timesteps = 9

    def fake_flash(*args, **kwargs):
        return args[0]

    q = torch.randn(1, 32, 2, 4)
    module = _Attn()
    module.frame_prompt_router = router
    module._ss_timestep = 0
    module._ss_block_idx = 0
    module._ss_pass = "cond"
    module._ss_grid_sizes = torch.tensor([[8, 2, 2]])
    scene_masked_official_flash(
        fake_flash,
        "flash_attention",
        module,
        router,
        (q, q.clone(), q.clone()),
        {"k_lens": torch.tensor([32])},
    )
    assert (0, 0, 0) in router._ref_loc_scores


@pytest.mark.parametrize("backend", ["wan_s2v", "wan_animate", "wan_fun_camera"])
def test_ref_scores_publish_only_after_complete_step(backend):
    router = FramePromptRouter(
        canonical_config(), backend=backend, patch_height=2, patch_width=2
    )
    fg = torch.ones(1, 8, 4)
    bg = torch.zeros(8, 4)
    router.store_ref_loc_head(0, 37, 32, fg, bg)
    assert (0, 37, 32) not in router._ref_loc_scores
    assert (0, 37, 32) in router._pending_ref_loc_scores
    router.complete_denoising_step()
    assert (0, 37, 32) in router._ref_loc_scores
    assert router._pending_ref_loc_scores == {}


def test_collect_skips_s2v_grid_mask_mismatch():
    cfg = canonical_config()
    cfg["attention_control_schedule"][0]["timestep_range"] = [3, 39]
    cfg["attention_control_schedule"][0]["params"] = {
        "mask_type": "ref_attn",
        "ref_attn_mask_selectors": [[0, 0]],
        "ref_attn_mask_drop_low_percent": 0.0,
        "ref_attn_mask_min_component_size": 1,
        "ref_attn_mask_hole_size": 0,
    }
    router = FramePromptRouter(cfg, backend="wan_s2v", patch_height=2, patch_width=2)
    router.subject_fg_masks = torch.zeros(1, 8, 4, dtype=torch.bool)
    router.subject_fg_masks[0, :, 0] = True
    router.static_bg = ~router.subject_fg_masks.any(0)
    router.ref_loc_max_timesteps = 9

    def fake_flash(*args, **kwargs):
        return args[0]

    q = torch.randn(1, 32, 2, 4)
    module = _Attn()
    module.frame_prompt_router = router
    module._ss_timestep = 0
    module._ss_block_idx = 0
    module._ss_pass = "cond"
    module._ss_grid_sizes = torch.tensor([[21, 21, 21]])
    scene_masked_official_flash(
        fake_flash,
        "flash_attention",
        module,
        router,
        (q, q.clone(), q.clone()),
        {"k_lens": torch.tensor([32])},
    )
    assert router._ref_loc_scores == {}


def test_scene_mask_ref_loc_q_uses_local_new_scene_rope():
    """Official scores ref-loc with locally re-roped new-scene Q, even if auto rerope is off."""
    cfg = canonical_config()
    cfg["attention_control_schedule"][0]["self_attention_mask"] = [
        [[1, "full"], [1, "fg"]],
        [[1, "fg"], [1, "full"]],
    ]
    cfg["attention_control_schedule"][0]["params"]["auto_q_scale"] = False
    router = FramePromptRouter(cfg, patch_height=2, patch_width=2)
    q = torch.randn(1, 32, 2, 4)
    module = _Attn()
    module._ss_q_pre_rope = q * 3
    module._ss_freqs = torch.ones(1)
    module._ss_grid_sizes = torch.tensor([[8, 2, 2]])
    module._ss_rope_fn = lambda x, grid_sizes, freqs, *args, **kwargs: x
    block = router.active_block(3, 0, "cond")
    assert block is not None
    q_ref = _ref_loc_query_for_scene_mask(module, router, block, q)
    spatial = 4
    scene0 = slice(0, 3 * spatial)
    scene1 = slice(3 * spatial, 8 * spatial)
    assert torch.equal(q_ref[:, scene0], q[:, scene0])
    assert torch.equal(q_ref[:, scene1], (q * 3)[:, scene1])


def test_talking_i2v_uncond_keeps_official_neg_context():
    cfg = canonical_config()
    cfg["segment_prompt_start_timestep"] = 3
    router = FramePromptRouter(cfg, backend="multitalk", patch_height=2, patch_width=2)
    clip = torch.randn(1, 257, 8)
    shared = torch.randn(1, 4, 8)
    router.set_text_embeddings([shared, torch.randn(1, 4, 8)])
    captured = {}

    class I2V(nn.Module):
        def forward(self, x, context, context_lens=None):
            captured["context"] = context
            captured["context_lens"] = context_lens
            return x

    module = I2V()
    wrap_cross_attention(module, router)
    module._ss_timestep = 0
    module._ss_pass = "uncond"
    x = torch.randn(1, 8, 8)
    official = torch.cat([clip, torch.randn(1, 4, 8)], dim=1)
    out = module(x, official, torch.tensor([3]))
    assert out is x
    # Official disables cross-attn routing on uncond; keep generate's neg prompt.
    assert captured["context_lens"] is None
    assert captured["context"] is official


def test_active_segment_routing_fails_without_text_embeddings():
    cfg = canonical_config()
    cfg["segment_prompt_start_timestep"] = 3
    router = FramePromptRouter(cfg, backend="multitalk", patch_height=2, patch_width=2)

    class Cross(nn.Module):
        def forward(self, x, context, context_lens=None):
            return x

    module = Cross()
    wrap_cross_attention(module, router)
    module._ss_timestep = 3
    module._ss_pass = "cond"
    with pytest.raises(RuntimeError, match="no segment text embeddings"):
        module(torch.randn(1, 8, 4), torch.randn(1, 6, 4))


def test_fun_camera_batched_cfg_routes_only_cond_row():
    cfg = canonical_config()
    router = FramePromptRouter(
        cfg, backend="wan_fun_camera", patch_height=2, patch_width=2
    )
    router.set_text_embeddings(
        [torch.full((1, 3, 4), 10.0), torch.full((1, 3, 4), 20.0)]
    )
    seen = []

    class Cross(nn.Module):
        def forward(self, x, context, context_lens=None):
            seen.append(context.clone())
            return x

    module = Cross()
    wrap_cross_attention(module, router)
    module._ss_timestep = 3
    module._ss_pass = "cond"
    x = torch.randn(2, 32, 4)
    context = torch.stack(
        [torch.full((3, 4), -1.0), torch.full((3, 4), 1.0)], dim=0
    )
    out = module(x, context)
    assert out.shape == x.shape
    assert [float(value[0, 0, 0]) for value in seen] == [-1.0, 10.0, -1.0, 20.0]


def test_fun_camera_ref_loc_collects_cond_cfg_row(monkeypatch):
    cfg = canonical_config()
    cfg["attention_control_schedule"][0]["params"] = {
        "mask_type": "ref_attn",
        "ref_attn_mask_selectors": [[0, 0]],
    }
    router = FramePromptRouter(
        cfg, backend="wan_fun_camera", patch_height=2, patch_width=2
    )
    router.subject_fg_masks = torch.zeros(1, 8, 4, dtype=torch.bool)
    router.subject_fg_masks[0, :, 0] = True
    router.static_bg = ~router.subject_fg_masks.any(0)
    seen = {}

    def scores(visual_q, ref_k, shape, targets):
        seen["batch"] = int(visual_q.shape[0])
        seen["value"] = float(visual_q[0, 0, 0, 0])
        return torch.zeros(2, 32)

    monkeypatch.setattr(base_adapter, "get_attn_map_with_target", scores)
    module = _Attn()
    module._ss_grid_sizes = torch.tensor([[8, 2, 2]])
    module._ss_v_for_ref = torch.zeros(2, 32, 1, 1)
    q = torch.stack(
        [torch.zeros(32, 1, 1), torch.ones(32, 1, 1)], dim=0
    )
    base_adapter._collect_live_ref_attn(router, module, q, q, 0, 0, "cond")
    assert seen == {"batch": 1, "value": 1.0}


def test_cross_attn_keeps_s2v_extra_tokens():
    cfg = canonical_config()
    cfg["segment_prompt_start_timestep"] = 0
    router = FramePromptRouter(cfg, backend="wan_s2v", patch_height=2, patch_width=2)
    text = torch.randn(1, 4, 8)
    router.set_text_embeddings([text, text])
    captured = []

    class Cross(nn.Module):
        def forward(self, x, context, context_lens=None):
            captured.append(int(x.shape[1]))
            return x

    module = Cross()
    wrap_cross_attention(module, router)
    module._ss_timestep = 3
    module._ss_pass = "cond"
    x = torch.randn(1, 34, 8)
    context = torch.cat([torch.randn(1, 257, 8), text], dim=1)
    out = module(x, context)
    assert out.shape[1] == 34
    assert captured == [34, 12, 20]
