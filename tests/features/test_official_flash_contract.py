import torch
import torch.nn as nn

from sceneshifter.adapters.base import (
    _grid_fhw,
    _ref_loc_query_for_scene_mask,
    project_router_text_embeddings,
    scene_masked_official_flash,
    wrap_audio_attention,
    wrap_cross_attention,
)
from sceneshifter.router import FramePromptRouter
from tests.conftest import canonical_config


class _Attn(nn.Module):
    pass


def test_grid_fhw_flattens_s2v_nonscalar_row():
    router = FramePromptRouter(canonical_config(), patch_height=2, patch_width=2)
    # int(row[0]) raises when each axis is a multi-element tensor.
    grid = [torch.tensor([[21, 44, 64], [21, 44, 64]])]
    assert _grid_fhw(grid, router) == (21, 44, 64)
    assert _grid_fhw(torch.tensor([[8, 2, 2]]), router) == (8, 2, 2)


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
