import pytest
import torch
import torch.nn as nn

from sceneshifter.adapters.infinitetalk import zero_first_latent_frame_audio
from sceneshifter.adapters import base as base_adapter
from sceneshifter.adapters.base import (
    _grid_fhw,
    _fun_camera_serial_pass,
    _maybe_update_spatial,
    project_router_text_embeddings,
    scene_masked_official_flash,
    wrap_audio_attention,
    wrap_cross_attention,
)
from sceneshifter.masks import downsample_subject_fg
from sceneshifter.router import FramePromptRouter
from tests.conftest import canonical_config


class _Attn(nn.Module):
    pass


def test_fun_camera_serial_pass_resets_each_timestep():
    module = _Attn()
    assert _fun_camera_serial_pass(module, 3) == "cond"
    assert _fun_camera_serial_pass(module, 3) == "uncond"
    assert _fun_camera_serial_pass(module, 4) == "cond"


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


def test_animate_grid_reconciliation_uses_direct_mask_resize():
    router = FramePromptRouter(
        canonical_config(), backend="wan_animate", patch_height=4, patch_width=6
    )
    raw = torch.zeros(1, 8, 24, 16, dtype=torch.bool)
    raw[..., :, 0] = True
    router.raw_subject_fg = raw
    module = _Attn()
    module._ss_grid_sizes = torch.tensor([[8, 7, 4]])
    q = torch.zeros(1, 8 * 7 * 4, 1, 1)
    expected_subjects, expected_bg = downsample_subject_fg(
        raw, 7, 4, center_crop=False
    )
    cropped_subjects, _ = downsample_subject_fg(raw, 7, 4, center_crop=True)
    assert not torch.equal(expected_subjects, cropped_subjects)
    _maybe_update_spatial(router, module, q)
    assert torch.equal(router.subject_fg_masks, expected_subjects)
    assert torch.equal(router.static_bg, expected_bg)


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


def test_wrapped_self_attention_releases_ref_value_stash(monkeypatch):
    from sceneshifter.adapters.base import wrap_self_attention

    def fake_flash(q, k, v, **kwargs):
        return q

    class Official(nn.Module):
        def forward(self, q, k, v, grid_sizes=None):
            return flash_attention(q, k, v)

    monkeypatch.setitem(Official.forward.__globals__, "flash_attention", fake_flash)
    router = FramePromptRouter(
        canonical_config(), backend="wan_animate", patch_height=2, patch_width=2
    )
    module = Official()
    wrap_self_attention(module, router)
    module._ss_timestep = 3
    module._ss_block_idx = 0
    module._ss_pass = "cond"
    q = torch.randn(1, 32, 1, 2)
    module(q, q.clone(), q.clone(), grid_sizes=torch.tensor([[8, 2, 2]]))
    assert module._ss_v_for_ref is None


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


def _tiny_text_projector(device: torch.device, dtype: torch.dtype) -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(8, 16, bias=False),
        nn.GELU(),
        nn.Linear(16, 16, bias=False),
    ).to(device=device, dtype=dtype)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cpu_text_projector_runs_on_cuda_and_restores():
    """CPU-resident text_embedding must matmul on CUDA, then return to CPU."""
    router = FramePromptRouter(canonical_config(), patch_height=2, patch_width=2)
    projector = _tiny_text_projector(torch.device("cpu"), torch.bfloat16)
    router._text_embedding = projector
    router._text_len = 4
    first = torch.randn(3, 8, dtype=torch.bfloat16)
    second = torch.randn(2, 8, dtype=torch.bfloat16)
    router.set_text_embeddings([first.clone(), second.clone()])
    weights_before = [param.detach().clone() for param in projector.parameters()]
    seen = []

    def _record(_module, inputs, _output):
        seen.append((inputs[0].device.type, next(_module.parameters()).device.type))

    handles = [
        module.register_forward_hook(_record)
        for module in projector.modules()
        if isinstance(module, nn.Linear)
    ]
    try:
        project_router_text_embeddings(router)
    finally:
        for handle in handles:
            handle.remove()

    assert seen == [("cuda", "cuda")] * 4
    assert all(param.device.type == "cpu" and param.dtype == torch.bfloat16 for param in projector.parameters())
    for before, after in zip(weights_before, projector.parameters()):
        assert torch.equal(before, after.detach())

    reference = _tiny_text_projector(torch.device("cpu"), torch.bfloat16)
    reference.load_state_dict(projector.state_dict())
    reference.to(device=torch.device("cuda"))

    def _cuda_project(tokens: torch.Tensor) -> torch.Tensor:
        core = tokens
        if core.shape[0] < 4:
            core = torch.cat([core, core.new_zeros(4 - core.shape[0], core.shape[1])], dim=0)
        else:
            core = core[:4]
        return reference(core.unsqueeze(0).to(device=torch.device("cuda"), dtype=torch.bfloat16))

    assert torch.equal(router.text_embeddings[0].cpu(), _cuda_project(first).cpu())
    assert torch.equal(router.text_embeddings[1].cpu(), _cuda_project(second).cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_text_projector_stays_on_cuda():
    router = FramePromptRouter(canonical_config(), patch_height=2, patch_width=2)
    projector = _tiny_text_projector(torch.device("cuda"), torch.bfloat16)
    router._text_embedding = projector
    router._text_len = 4
    router.set_text_embeddings(
        [
            torch.randn(3, 8, dtype=torch.bfloat16),
            torch.randn(2, 8, dtype=torch.bfloat16),
        ]
    )
    storage = [param.data_ptr() for param in projector.parameters()]
    moves = []
    original_to = projector.to

    def _track_to(*args, **kwargs):
        moves.append((args, kwargs))
        return original_to(*args, **kwargs)

    projector.to = _track_to
    project_router_text_embeddings(router)
    assert moves == []
    assert [param.data_ptr() for param in projector.parameters()] == storage
    assert all(param.device.type == "cuda" and param.dtype == torch.bfloat16 for param in projector.parameters())
    assert router.text_embeddings[0].device.type == "cuda"


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


def test_audio_residual_skipped_without_routing_router():
    class Audio(nn.Module):
        def forward(self, x, **kwargs):
            return x + 1

    module = Audio()
    wrap_audio_attention(module, None)
    module._ss_audio_start_timestep = 1000
    module._ss_timestep = 0
    x = torch.zeros(1, 4, 2)
    assert torch.equal(module(x), torch.zeros_like(x))


def test_disabled_talking_routing_still_gates_audio_residual():
    from sceneshifter.parity.unified import _maybe_gate_talking_audio_cfg

    class Audio(nn.Module):
        def forward(self, x, **kwargs):
            return x + 1

    class DiT(nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_cross_attn = Audio()

        def forward(self, x, **kwargs):
            return x

    dit = DiT()
    wrap_audio_attention(dit.audio_cross_attn, None)
    _maybe_gate_talking_audio_cfg(
        dit,
        None,
        "multitalk",
        {"enabled": False, "audio_start_timestep": 1000},
    )
    dit.audio_cross_attn._ss_timestep = 0
    x = torch.zeros(1, 4, 2)
    assert dit.audio_cross_attn._ss_audio_start_timestep == 1000
    assert torch.equal(dit.audio_cross_attn(x), torch.zeros_like(x))


def test_infinitetalk_zeros_audio_on_first_latent_frame():
    class Audio(nn.Module):
        def forward(self, x, encoder_hidden_states=None, shape=None, **kwargs):
            return torch.ones_like(x)

    module = Audio()
    zero_first_latent_frame_audio(module)
    x = torch.ones(1, 6, 2)
    out = module(x, shape=torch.tensor([3, 2, 1]))
    assert torch.equal(out[:, :2], torch.zeros(1, 2, 2))
    assert torch.equal(out[:, 2:], torch.ones(1, 4, 2))
    again = module(x, encoder_hidden_states=torch.zeros(1, 1, 1), shape=(3, 2, 1))
    assert torch.equal(again, out)


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


@pytest.mark.parametrize(
    "backend", ["wan_s2v", "wan_animate", "wan_fun_camera"]
)
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


def test_animate_publishes_scores_at_last_cond_layer():
    cfg = canonical_config()
    cfg["attention_control_schedule"][0]["params"] = {
        "mask_type": "ref_attn",
        "ref_attn_mask_selectors": [[0, 0]],
    }
    router = FramePromptRouter(
        cfg, backend="wan_animate", patch_height=2, patch_width=2
    )
    router._ss_num_self_layers = 2
    router.subject_fg_masks = torch.zeros(1, 8, 4, dtype=torch.bool)
    router.subject_fg_masks[0, :, 0] = True
    router.static_bg = ~router.subject_fg_masks.any(0)
    module = _Attn()
    module._ss_grid_sizes = torch.tensor([[8, 2, 2]])
    module._ss_v_for_ref = torch.zeros(1, 32, 1, 2)
    q = torch.randn(1, 32, 1, 2)

    base_adapter._collect_live_ref_attn(router, module, q, q, 0, 0, "cond")
    assert (0, 0, 0) in router._pending_ref_loc_scores
    assert (0, 0, 0) not in router._ref_loc_scores
    base_adapter._collect_live_ref_attn(router, module, q, q, 0, 1, "cond")
    assert router._pending_ref_loc_scores == {}
    assert (0, 0, 0) in router._ref_loc_scores


def test_collect_fails_loud_on_s2v_grid_mask_mismatch():
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
    with pytest.raises(RuntimeError, match="ref-attn selector"):
        scene_masked_official_flash(
            fake_flash,
            "flash_attention",
            module,
            router,
            (q, q.clone(), q.clone()),
            {"k_lens": torch.tensor([32])},
        )
    assert router._ref_loc_scores == {}


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


def test_fun_camera_serialized_cfg_routes_first_call_only():
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
            seen.append(float(context[0, 0, 0]))
            return x

    module = Cross()
    wrap_cross_attention(module, router)
    module._ss_timestep = 3
    module._ss_pass = "uncond"  # Outer CFG wrapper is intentionally misleading.
    x = torch.randn(1, 32, 4)
    module(x, torch.ones(1, 3, 4))
    module(x, torch.full((1, 3, 4), -1.0))
    assert seen == [10.0, 20.0, -1.0]


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


def test_fantasy_ref_loc_scores_union_fg_in_fp32(monkeypatch):
    cfg = canonical_config()
    cfg["attention_control_schedule"][0]["params"] = {
        "mask_type": "ref_attn",
        "ref_attn_mask_selectors": [[0, 0]],
    }
    router = FramePromptRouter(
        cfg, backend="fantasytalking", patch_height=2, patch_width=2
    )
    router.subject_fg_masks = torch.zeros(2, 8, 4, dtype=torch.bool)
    router.subject_fg_masks[0, :, 0] = True
    router.subject_fg_masks[1, :, 1] = True
    router.static_bg = ~router.subject_fg_masks.any(0)
    seen = {}

    def scores(visual_q, ref_k, shape, targets):
        seen["dtype"] = visual_q.dtype
        seen["targets"] = targets.clone()
        return torch.zeros(2, 32)

    monkeypatch.setattr(base_adapter, "get_attn_map_with_target", scores)
    module = _Attn()
    module._ss_grid_sizes = torch.tensor([[8, 2, 2]])
    module._ss_v_for_ref = torch.zeros(1, 32, 1, 1, dtype=torch.bfloat16)
    q = torch.randn(1, 32, 1, 1, dtype=torch.bfloat16)
    base_adapter._collect_live_ref_attn(router, module, q, q, 0, 0, "cond")
    assert seen["dtype"] == torch.float32
    assert tuple(seen["targets"].shape) == (2, 4)
    assert seen["targets"][0].bool().tolist() == [True, True, False, False]


def test_infinitetalk_ref_loc_scores_in_fp32_when_v_is_bf16(monkeypatch):
    cfg = canonical_config()
    cfg["attention_control_schedule"][0]["params"] = {
        "mask_type": "ref_attn",
        "ref_attn_mask_selectors": [[0, 0]],
    }
    router = FramePromptRouter(
        cfg, backend="infinitetalk", patch_height=2, patch_width=2
    )
    router.subject_fg_masks = torch.zeros(1, 8, 4, dtype=torch.bool)
    router.subject_fg_masks[0, :, 0] = True
    router.static_bg = ~router.subject_fg_masks.any(0)
    seen = {}

    def scores(visual_q, ref_k, shape, targets):
        seen["q_dtype"] = visual_q.dtype
        seen["k_dtype"] = ref_k.dtype
        seen["targets_dtype"] = targets.dtype
        return torch.zeros(2, 32)

    monkeypatch.setattr(base_adapter, "get_attn_map_with_target", scores)
    module = _Attn()
    module._ss_grid_sizes = torch.tensor([[8, 2, 2]])
    module._ss_v_for_ref = torch.zeros(1, 32, 1, 1, dtype=torch.bfloat16)
    q = torch.randn(1, 32, 1, 1, dtype=torch.float32)
    k = torch.randn(1, 32, 1, 1, dtype=torch.float32)
    base_adapter._collect_live_ref_attn(router, module, q, k, 0, 0, "cond")
    assert seen["q_dtype"] == torch.float32
    assert seen["k_dtype"] == torch.float32
    assert seen["targets_dtype"] == torch.float32
    assert (0, 0, 0) in router._ref_loc_scores


def test_fantasy_collects_unscaled_q_but_attention_uses_scaled_q(monkeypatch):
    cfg = canonical_config()
    router = FramePromptRouter(
        cfg, backend="fantasytalking", patch_height=2, patch_width=2
    )
    router.apply_q_scale = lambda q, block: q * 2
    collected = {}
    attended = []

    def collect(router, module, q, k, timestep_idx, block_idx, pass_type):
        collected["q"] = q.clone()

    monkeypatch.setattr(base_adapter, "_collect_live_ref_attn", collect)

    def flash(q, k, v, *args, **kwargs):
        attended.append(q.clone())
        return q

    q = torch.ones(1, 32, 1, 2)
    module = _Attn()
    module.frame_prompt_router = router
    module._ss_timestep = 3
    module._ss_block_idx = 0
    module._ss_pass = "cond"
    scene_masked_official_flash(
        flash,
        "flash_attention",
        module,
        router,
        (q, q.clone(), q.clone()),
        {"k_lens": torch.tensor([32])},
    )
    assert torch.equal(collected["q"], q)
    assert attended
    assert all(torch.equal(value, torch.full_like(value, 2.0)) for value in attended)


def test_multitalk_selector_scores_use_reference_head_chunks(monkeypatch):
    cfg = canonical_config()
    cfg["attention_control_schedule"][0]["params"] = {
        "mask_type": "ref_attn",
        "ref_attn_mask_selectors": [[0, 0], [0, 1]],
    }
    router = FramePromptRouter(
        cfg, backend="multitalk", patch_height=2, patch_width=2
    )
    router.subject_fg_masks = torch.zeros(1, 8, 4, dtype=torch.bool)
    router.subject_fg_masks[0, :, 0] = True
    router.static_bg = ~router.subject_fg_masks.any(0)
    seen = []

    def scores(visual_q, ref_k, shape, targets, *, split_num=2):
        seen.append((int(visual_q.shape[2]), int(split_num), visual_q.dtype))
        return torch.zeros(2, 32, int(visual_q.shape[2]), dtype=visual_q.dtype)

    monkeypatch.setattr(
        base_adapter, "get_attn_map_with_target_per_head_split", scores
    )
    module = _Attn()
    module._ss_grid_sizes = torch.tensor([[8, 2, 2]])
    module._ss_v_for_ref = torch.zeros(1, 32, 2, 2, dtype=torch.bfloat16)
    q = torch.randn(1, 32, 2, 2)
    base_adapter._collect_live_ref_attn(router, module, q, q, 0, 0, "cond")
    assert seen == [(2, 2, torch.bfloat16)]
    assert (0, 0, 0) in router._ref_loc_scores
    assert (0, 0, 1) in router._ref_loc_scores


def test_animate_uses_memory_safe_model_dtype_selector_calls(monkeypatch):
    cfg = canonical_config()
    cfg["attention_control_schedule"][0]["params"] = {
        "mask_type": "ref_attn",
        "ref_attn_mask_selectors": [[0, 2], [0, 3]],
    }
    router = FramePromptRouter(
        cfg, backend="wan_animate", patch_height=2, patch_width=2
    )
    router._ss_num_self_layers = 2
    router.subject_fg_masks = torch.zeros(1, 8, 4, dtype=torch.bool)
    router.subject_fg_masks[0, :, 0] = True
    router.static_bg = ~router.subject_fg_masks.any(0)
    seen = []

    def scores(visual_q, ref_k, shape, targets):
        seen.append((int(visual_q.shape[2]), visual_q.dtype))
        return torch.zeros(2, 32)

    monkeypatch.setattr(base_adapter, "get_attn_map_with_target", scores)
    module = _Attn()
    module._ss_grid_sizes = torch.tensor([[8, 2, 2]])
    module._ss_v_for_ref = torch.zeros(1, 32, 4, 2, dtype=torch.bfloat16)
    q = torch.randn(1, 32, 4, 2)
    base_adapter._collect_live_ref_attn(router, module, q, q, 0, 0, "cond")
    assert seen == [(1, torch.bfloat16), (1, torch.bfloat16)]
    assert (0, 0, 2) in router._pending_ref_loc_scores
    assert (0, 0, 3) in router._pending_ref_loc_scores


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


def test_multitalk_cross_attn_matches_reference_call_structure(monkeypatch):
    cfg = canonical_config(segment_prompt_start_timestep=3)
    router = FramePromptRouter(cfg, backend="multitalk", patch_height=2, patch_width=2)
    router.set_text_embeddings(
        [
            torch.full((1, 4, 4), 3.0),
            torch.full((1, 4, 4), 7.0),
        ]
    )
    calls = []

    def fake_flash(q, k, v, k_lens=None):
        calls.append((int(q.shape[1]), int(k.shape[1]), float(k.mean())))
        return q

    class Cross(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Identity()
            self.k = nn.Identity()
            self.v = nn.Identity()
            self.k_img = nn.Identity()
            self.v_img = nn.Identity()
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
            self.norm_k_img = nn.Identity()
            self.o = nn.Identity()
            self.num_heads = 1
            self.head_dim = 4

        def forward(self, x, context, context_lens=None):
            return x

    monkeypatch.setitem(Cross.forward.__globals__, "flash_attention", fake_flash)
    module = Cross()
    wrap_cross_attention(module, router)
    module._ss_pass = "cond"
    x = torch.randn(1, 32, 4)
    context = torch.cat(
        [torch.full((1, 257, 4), 11.0), torch.zeros(1, 4, 4)], dim=1
    )

    module._ss_timestep = 0
    out = module(x, context)
    assert torch.equal(out, 2 * x)
    assert calls == [(12, 4, 3.0), (20, 4, 3.0), (32, 257, 11.0)]

    calls.clear()
    module._ss_timestep = 3
    module(x, context)
    assert calls == [(12, 4, 3.0), (20, 4, 7.0), (32, 257, 11.0)]

    calls.clear()
    module._ss_pass = "uncond"
    assert module(x, context) is x
    assert calls == []
