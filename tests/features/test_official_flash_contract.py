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
from sceneshifter.adapters.wan_fun_camera import (
    install_fun_camera_batched_cfg,
    install_fun_camera_encoder_offload,
    install_fun_camera_feature_rmsnorm,
    install_fun_camera_fp32_layernorm,
    install_fun_camera_fp32_time_embedding,
)
from sceneshifter.masks import downsample_subject_fg
from sceneshifter.router import FramePromptRouter
from tests.conftest import canonical_config


class _Attn(nn.Module):
    pass


def test_fun_camera_preserves_feature_fp32_time_modulation():
    class TimeEmbedding(nn.Module):
        def forward(self, value):
            return value * 1.25 + 0.125 + value.shape[0] * 1e-4

    class TimeProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_dtype = None

        def forward(self, value):
            self.seen_dtype = value.dtype
            return value * 2.0 + value.shape[0] * 1e-4

    class Model:
        def __init__(self):
            self.time_embedding = TimeEmbedding()
            self.time_projection = TimeProjection()

        def forward(self, value):
            e = self.time_embedding(value).to(torch.bfloat16)
            return self.time_projection(e)

    value = torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float32)
    old_e = TimeEmbedding()(value[:1]).to(torch.bfloat16)
    assert old_e.dtype == torch.bfloat16

    model = Model()
    install_fun_camera_fp32_time_embedding(model)
    install_fun_camera_fp32_time_embedding(model)  # idempotent
    e0 = model.forward(value)
    expected_e = TimeEmbedding()(value)
    expected_e0 = TimeProjection()(expected_e)
    assert model.time_projection.seen_dtype == torch.float32
    assert e0.dtype == torch.float32
    torch.testing.assert_close(e0, expected_e0, rtol=0, atol=0)


def test_fun_camera_layernorm_keeps_native_output_and_leaves_other_backends(monkeypatch):
    class WanLayerNorm(nn.LayerNorm):
        def forward(self, x):
            return nn.LayerNorm.forward(self, x.float()).type_as(x)

    class WanRMSNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))

        def forward(self, x):
            return x.float().type_as(x) * self.weight

    class Fun(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = WanLayerNorm(4, elementwise_affine=False)
            self.norm3 = WanLayerNorm(4, elementwise_affine=True)
            self.norm_q = WanRMSNorm(4)
            self.blocks = nn.ModuleList([self.norm1])

    class Other(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = WanLayerNorm(4, elementwise_affine=False)
            self.blocks = nn.ModuleList([self.norm1])

    torch.manual_seed(0)
    fun = Fun()
    other = Other()
    other.norm1.load_state_dict(fun.norm1.state_dict())
    x = torch.randn(2, 5, 4, dtype=torch.bfloat16)
    casts = []
    real_type_as = torch.Tensor.type_as

    def spy(self, other_tensor):
        casts.append((self.dtype, other_tensor.dtype))
        return real_type_as(self, other_tensor)

    monkeypatch.setattr(torch.Tensor, "type_as", spy)
    install_fun_camera_fp32_layernorm(fun)
    install_fun_camera_fp32_layernorm(fun)  # idempotent

    casts.clear()
    norm1 = fun.norm1(x)
    assert casts == []
    assert fun.norm1.forward.__func__.__name__ == "_feature_wan_layernorm_forward"
    feature_norm1 = nn.LayerNorm.forward(fun.norm1, x)
    assert norm1.dtype == feature_norm1.dtype
    torch.testing.assert_close(norm1, feature_norm1, rtol=0, atol=0)

    casts.clear()
    norm3 = fun.norm3(x)
    assert casts == []
    torch.testing.assert_close(norm3, nn.LayerNorm.forward(fun.norm3, x), rtol=0, atol=0)

    casts.clear()
    rms = fun.norm_q(x)
    assert rms.dtype == torch.bfloat16
    assert casts == [(torch.float32, torch.bfloat16)]

    casts.clear()
    other_out = other.norm1(x)
    assert other.norm1.forward.__func__ is WanLayerNorm.forward
    assert other_out.dtype == torch.bfloat16
    assert casts == [(torch.float32, torch.bfloat16)]
    torch.testing.assert_close(
        other_out,
        nn.LayerNorm.forward(other.norm1, x.float()).to(dtype=x.dtype),
        rtol=0,
        atol=0,
    )


def test_fun_camera_layernorm_requires_modules_on_a_real_transformer():
    class Model:
        blocks = []

    with pytest.raises(TypeError, match="WanLayerNorm"):
        install_fun_camera_fp32_layernorm(Model())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fun_camera_layernorm_stays_fp32_under_cuda_autocast():
    class WanLayerNorm(nn.LayerNorm):
        def forward(self, x):
            return nn.LayerNorm.forward(self, x.float()).type_as(x)

    class Fun(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = WanLayerNorm(8, elementwise_affine=False)
            self.blocks = nn.ModuleList([self.norm1])

    class Other(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = WanLayerNorm(8, elementwise_affine=False)

    fun = Fun().cuda()
    other = Other().cuda()
    x = torch.randn(2, 4, 8, device="cuda", dtype=torch.bfloat16)
    install_fun_camera_fp32_layernorm(fun)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = fun.norm1(x)
        native = nn.LayerNorm.forward(fun.norm1, x)
        official = other.norm1(x)
    assert native.dtype == torch.float32
    assert out.dtype == torch.float32
    assert official.dtype == torch.bfloat16
    torch.testing.assert_close(out, native, rtol=0, atol=0)


def _feature_rmsnorm(module, x):
    scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + module.eps).to(dtype=x.dtype)
    return x * scale * module.weight


def _official_rmsnorm(module, x):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + module.eps)
    return y.to(dtype=x.dtype) * module.weight


def test_fun_camera_rmsnorm_keeps_feature_formula_and_leaves_other_backends(monkeypatch):
    class WanRMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.randn(dim, dtype=torch.bfloat16))

        def forward(self, x):
            y = x.float()
            y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return y.type_as(x) * self.weight

    class WanLayerNorm(nn.LayerNorm):
        def forward(self, x):
            return nn.LayerNorm.forward(self, x.float()).type_as(x)

    class Fun(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_q = WanRMSNorm(4)
            self.norm_k = WanRMSNorm(4)
            self.norm1 = WanLayerNorm(4, elementwise_affine=False)
            self.blocks = nn.ModuleList([self.norm_q, self.norm_k])

    class Other(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_q = WanRMSNorm(4)
            self.blocks = nn.ModuleList([self.norm_q])

    torch.manual_seed(0)
    fun = Fun()
    other = Other()
    other.norm_q.load_state_dict(fun.norm_q.state_dict())
    x = torch.randn(2, 5, 4, dtype=torch.bfloat16)
    casts = []
    real_type_as = torch.Tensor.type_as

    def spy(self, other_tensor):
        casts.append((self.dtype, other_tensor.dtype))
        return real_type_as(self, other_tensor)

    monkeypatch.setattr(torch.Tensor, "type_as", spy)
    install_fun_camera_feature_rmsnorm(fun)
    install_fun_camera_feature_rmsnorm(fun)  # idempotent

    feature_q = _feature_rmsnorm(fun.norm_q, x)
    official_q = _official_rmsnorm(fun.norm_q, x)
    assert not torch.equal(feature_q, official_q)

    casts.clear()
    norm_q = fun.norm_q(x)
    assert casts == []
    assert fun.norm_q.forward.__func__.__name__ == "_feature_wan_rmsnorm_forward"
    assert norm_q.dtype == feature_q.dtype
    torch.testing.assert_close(norm_q, feature_q, rtol=0, atol=0)
    assert not torch.equal(norm_q, official_q)

    casts.clear()
    norm_k = fun.norm_k(x)
    assert casts == []
    torch.testing.assert_close(norm_k, _feature_rmsnorm(fun.norm_k, x), rtol=0, atol=0)

    casts.clear()
    layernorm = fun.norm1(x)
    assert layernorm.dtype == torch.bfloat16
    assert casts == [(torch.float32, torch.bfloat16)]
    assert fun.norm1.forward.__func__ is WanLayerNorm.forward

    casts.clear()
    other_out = other.norm_q(x)
    assert other.norm_q.forward.__func__ is WanRMSNorm.forward
    assert other_out.dtype == torch.bfloat16
    assert casts
    torch.testing.assert_close(other_out, _official_rmsnorm(other.norm_q, x), rtol=0, atol=0)


def test_fun_camera_rmsnorm_requires_modules_on_a_real_transformer():
    class Model:
        blocks = []

    with pytest.raises(TypeError, match="WanRMSNorm"):
        install_fun_camera_feature_rmsnorm(Model())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fun_camera_rmsnorm_matches_feature_under_cuda_autocast():
    class WanRMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.randn(dim, dtype=torch.bfloat16))

        def forward(self, x):
            y = x.float()
            y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return y.type_as(x) * self.weight

    class Fun(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_q = WanRMSNorm(8)
            self.blocks = nn.ModuleList([self.norm_q])

    class Other(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_q = WanRMSNorm(8)

    torch.manual_seed(0)
    fun = Fun().cuda()
    other = Other().cuda()
    other.norm_q.load_state_dict(fun.norm_q.state_dict())
    x = torch.randn(2, 4, 8, device="cuda", dtype=torch.bfloat16)
    install_fun_camera_feature_rmsnorm(fun)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = fun.norm_q(x)
        feature = _feature_rmsnorm(fun.norm_q, x)
        official = other.norm_q(x)
    assert out.dtype == torch.bfloat16
    assert feature.dtype == torch.bfloat16
    assert official.dtype == torch.bfloat16
    torch.testing.assert_close(out, feature, rtol=0, atol=0)
    assert not torch.equal(out, official)


def test_fun_camera_replaces_serial_cfg_with_feature_batch_order():
    serial_batches = []
    batched_batches = []

    def official_serial_cfg(func):
        def wrapper(self, x, context):
            half = len(x) // 2
            cond = func(self, x[half:], context[half:])
            uncond = func(self, x[:half], context[:half])
            return torch.cat([uncond, cond])

        return wrapper

    class SerialModel:
        cfg_skip_ratio = 0
        current_steps = 0
        num_inference_steps = 50

        @official_serial_cfg
        def forward(self, x, context):
            serial_batches.append(len(x))
            return context

    x = torch.tensor([[1.0], [2.0]])
    context = torch.tensor([[-1.0], [1.0]])
    serial = SerialModel()
    torch.testing.assert_close(serial.forward(x, context), context)
    assert serial_batches == [1, 1]

    class BatchedModel:
        cfg_skip_ratio = 0
        current_steps = 0
        num_inference_steps = 50

        @official_serial_cfg
        def forward(self, x, context):
            batched_batches.append(len(x))
            return context

    batched = BatchedModel()
    install_fun_camera_batched_cfg(batched)
    install_fun_camera_batched_cfg(batched)  # idempotent
    torch.testing.assert_close(batched.forward(x, context), context)
    assert batched_batches == [2]


def _official_serial_cfg(func):
    def wrapper(self, x, context):
        half = len(x) // 2
        cond = func(self, x[half:], context[half:])
        uncond = func(self, x[:half], context[:half])
        return torch.cat([uncond, cond])

    return wrapper


def _serial_cfg_model(batches):
    class Model:
        cfg_skip_ratio = 0
        current_steps = 0
        num_inference_steps = 50
        blocks = []

        @_official_serial_cfg
        def forward(self, x, context):
            batches.append(len(x))
            return context

    return Model()


def test_fun_camera_batched_cfg_requires_install_on_a_real_transformer():
    class Model:
        blocks = []

        def forward(self, x, context):
            return context

    with pytest.raises(TypeError, match="batched CFG"):
        install_fun_camera_batched_cfg(Model())


def test_fun_camera_batched_cfg_skips_tiny_models_without_serial_cfg():
    class Tiny:
        def forward(self, x):
            return x

    model = Tiny()
    install_fun_camera_batched_cfg(model)
    assert model.forward(torch.ones(1)).shape == (1,)
    assert not getattr(model, "_ss_fun_batched_cfg_installed", False)


def test_fun_camera_batched_cfg_installs_under_existing_offload_hook():
    import functools

    batches = []
    model = _serial_cfg_model(batches)
    old_forward = model.forward
    model._old_forward = old_forward

    def new_forward(module, *args, **kwargs):
        return module._old_forward(*args, **kwargs)

    hooked = functools.update_wrapper(
        functools.partial(new_forward, model), old_forward
    )
    model.forward = hooked
    x = torch.tensor([[1.0], [2.0]])
    context = torch.tensor([[-1.0], [1.0]])
    torch.testing.assert_close(model.forward(x, context), context)
    assert batches == [1, 1]

    install_fun_camera_batched_cfg(model)
    install_fun_camera_batched_cfg(model)  # idempotent
    assert model.forward is hooked
    batches.clear()
    torch.testing.assert_close(model.forward(x, context), context)
    assert batches == [2]


class _CudaResident:
    """Reports CUDA storage until ``to('cpu')``."""

    def __init__(self):
        self.device = "cuda"
        self.cpu_calls = 0
        self.ran_on = None

    def parameters(self):
        owner = self

        class _Tensor:
            @property
            def is_cuda(self):
                return owner.device == "cuda"

        yield _Tensor()

    def buffers(self):
        return iter(())

    def to(self, device, *args, **kwargs):
        if str(device) == "cpu" or getattr(device, "type", None) == "cpu":
            self.device = "cpu"
            self.cpu_calls += 1
        else:
            self.device = "cuda"
        return self

    def forward(self, x):
        self.to("cuda")
        self.ran_on = self.device
        return x

    def encode(self, x):
        self.to("cuda")
        self.ran_on = self.device
        return x


def test_fun_camera_batch2_starts_with_encoders_on_cpu():
    batches = []

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList()
            self.cfg_skip_ratio = 0
            self.current_steps = 0
            self.num_inference_steps = 50
            self.marker = nn.Parameter(torch.zeros(1))

        @_official_serial_cfg
        def forward(self, x, context):
            batches.append(len(x))
            return x

    model = Model()
    install_fun_camera_batched_cfg(model)
    batched = model.forward
    hook_calls = {"n": 0}

    def accelerate_forward(*args, **kwargs):
        hook_calls["n"] += 1
        return batched(*args, **kwargs)

    model.forward = accelerate_forward
    text = _CudaResident()
    clip = _CudaResident()
    vae = _CudaResident()
    pipeline = type("Pipe", (), {})()
    pipeline.transformer = model
    pipeline.text_encoder = text
    pipeline.clip_image_encoder = clip
    pipeline.vae = vae
    install_fun_camera_encoder_offload(pipeline)
    install_fun_camera_encoder_offload(pipeline)
    assert model.forward is accelerate_forward

    x = torch.tensor([[1.0], [2.0]])
    context = torch.tensor([[-1.0], [1.0]])
    torch.testing.assert_close(model(x, context=context), x)
    assert hook_calls["n"] == 1
    assert batches == [2]
    assert text.device == "cpu"
    assert clip.device == "cpu"
    assert vae.device == "cpu"

    text.device = "cuda"
    clip.device = "cuda"
    model._forward_pre_hooks.clear()
    if hasattr(model, "_forward_pre_hooks_with_kwargs"):
        model._forward_pre_hooks_with_kwargs.clear()
    with pytest.raises(RuntimeError, match="still on CUDA"):
        model(x, context=context)


def test_fun_camera_encoder_forward_returns_to_cpu_and_other_backends_skip_guard():
    text = _CudaResident()
    clip = _CudaResident()
    vae = _CudaResident()

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return x

    kept = Tiny().forward
    pipeline = type("Pipe", (), {})()
    pipeline.transformer = Tiny()
    pipeline.transformer.forward = kept
    pipeline.text_encoder = text
    pipeline.clip_image_encoder = clip
    pipeline.vae = vae
    install_fun_camera_encoder_offload(pipeline)
    assert pipeline.transformer.forward is kept

    text.forward(torch.ones(1))
    assert text.ran_on == "cuda"
    assert text.device == "cpu"
    assert text.cpu_calls == 1
    clip.forward(torch.ones(1))
    vae.encode(torch.ones(1))
    assert clip.device == "cpu"
    assert vae.device == "cpu"

    plain = Tiny()
    plain.forward(torch.ones(1))
    assert not hasattr(plain, "_ss_fun_encoder_guard")

    import types

    previous = text.forward

    def capture(self, value):
        return previous(value)

    text.device = "cuda"
    text.forward = types.MethodType(capture, text)
    text.forward(torch.ones(1))
    assert text.ran_on == "cuda"
    assert text.device == "cpu"


def test_fun_camera_batched_cfg_installs_through_hook_closure():
    batches = []
    model = _serial_cfg_model(batches)
    original = model.forward

    def hooked(*args, **kwargs):
        return original(*args, **kwargs)

    model.forward = hooked
    x = torch.tensor([[1.0], [2.0]])
    context = torch.tensor([[-1.0], [1.0]])
    install_fun_camera_batched_cfg(model)
    assert model.forward is hooked
    torch.testing.assert_close(model.forward(x, context), context)
    assert batches == [2]


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
    assert scene1["k"] == [0, 4, 8, 32, *range(12, 32), 36, 37, 38]
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
    assert (0, 0, 0) in router._pending_ref_loc_scores
    assert (0, 0, 0) not in router._ref_loc_scores


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


def _single_subject_scores(foreground_token):
    fg = torch.zeros(1, 8, 4)
    bg = torch.ones(8, 4)
    fg[:, :, foreground_token] = 2
    bg[:, foreground_token] = 0
    return fg, bg


def _s2v_ref_attn_router(backend="wan_s2v"):
    cfg = canonical_config(ref_loc_max_timesteps=3)
    cfg["attention_control_schedule"][0]["params"] = {
        "mask_type": "ref_attn",
        "ref_attn_mask_selectors": [[37, 32]],
        "ref_attn_mask_drop_low_percent": 0.0,
        "ref_attn_mask_min_component_size": 0,
        "ref_attn_mask_hole_size": 0,
    }
    return FramePromptRouter(cfg, backend=backend, patch_height=2, patch_width=2)


def test_s2v_freezes_last_applied_mask_at_ref_loc_cutoff():
    router = _s2v_ref_attn_router()
    block = router.active_block(0, 0, "cond")

    for score_timestep, foreground_token in ((0, 0), (1, 1)):
        router.store_ref_loc_head(
            score_timestep, 37, 32, *_single_subject_scores(foreground_token)
        )
        router.complete_denoising_step()
        router.refresh_live_mask_from_stored(score_timestep + 1, block)

    expected_bg = router.live_bg.clone()
    expected_fg = router.live_subject_fg.clone()
    assert torch.all(expected_fg[:, :, 1])

    router.store_ref_loc_head(2, 37, 32, *_single_subject_scores(2))
    router.complete_denoising_step()
    assert router.live_bg is None

    router.refresh_live_mask_from_stored(3, block)
    assert torch.equal(router.live_bg, expected_bg)
    assert torch.equal(router.live_subject_fg, expected_fg)
    assert torch.all(router.live_subject_fg[:, :, 1])
    assert not torch.any(router.live_subject_fg[:, :, 2])


@pytest.mark.parametrize("backend", ["wan_animate", "wan_fun_camera"])
def test_s2v_ref_loc_freeze_does_not_change_other_wan_backends(backend):
    router = _s2v_ref_attn_router(backend)
    block = router.active_block(0, 0, "cond")

    for score_timestep, foreground_token in ((0, 0), (1, 1), (2, 2)):
        router.store_ref_loc_head(
            score_timestep, 37, 32, *_single_subject_scores(foreground_token)
        )
        router.complete_denoising_step()
        router.refresh_live_mask_from_stored(score_timestep + 1, block)

    assert torch.all(router.live_subject_fg[:, :, 2])
    assert not torch.any(router.live_subject_fg[:, :, 1])


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


def test_fun_camera_batched_cfg_splits_text_kv_and_other_backends_do_not():
    cfg = canonical_config(segment_prompt_start_timestep=3)
    router = FramePromptRouter(
        cfg, backend="wan_fun_camera", patch_height=2, patch_width=2
    )
    router.set_text_embeddings(
        [torch.full((1, 3, 4), 10.0), torch.full((1, 3, 4), 20.0)]
    )

    class Remember(nn.Module):
        def __init__(self):
            super().__init__()
            self.shapes = []

        def forward(self, value):
            self.shapes.append(tuple(value.shape))
            return value

    class FunCross(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 1
            self.head_dim = 4
            self.q = Remember()
            self.k = Remember()
            self.v = Remember()
            self.k_img = Remember()
            self.v_img = Remember()
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
            self.norm_k_img = nn.Identity()
            self.o = nn.Identity()
            self.full_batches = []

        def forward(self, x, context, context_lens=None, dtype=None, t=0):
            self.full_batches.append(int(context.shape[0]))
            return x

    fun = FunCross()
    wrap_cross_attention(fun, router)
    fun._ss_pass = "uncond"  # Batched CFG is not a pure uncond pass.
    x = torch.randn(2, 32, 4)
    context = torch.zeros(2, 260, 4)
    fun._ss_timestep = 0
    fun(x, context)
    assert fun.full_batches == [2]
    assert fun.k.shapes == []

    fun.full_batches.clear()
    fun._ss_timestep = 3
    fun(x, context)
    assert fun.full_batches == []
    assert fun.q.shapes == [(2, 32, 4)]
    assert fun.k.shapes == [(1, 3, 4)] * 4
    assert fun.v.shapes == [(1, 3, 4)] * 4
    assert fun.k_img.shapes == [(2, 257, 4)]

    other_router = FramePromptRouter(
        cfg, backend="multitalk", patch_height=2, patch_width=2
    )
    other_router.set_text_embeddings(
        [torch.full((1, 3, 4), 10.0), torch.full((1, 3, 4), 20.0)]
    )
    seen = []

    class Other(nn.Module):
        def forward(self, x, context, context_lens=None):
            seen.append(int(context.shape[0]))
            return x

    other = Other()
    wrap_cross_attention(other, other_router)
    other._ss_timestep = 3
    other._ss_pass = "uncond"
    other(x, context)
    assert seen == [2]


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


def _fun_ref_attn_config():
    cfg = canonical_config()
    cfg["attention_control_schedule"][0]["timestep_range"] = [3, 39]
    cfg["attention_control_schedule"][0]["params"] = {
        "mask_type": "ref_attn",
        "ref_attn_mask_selectors": [[0, 0]],
        "ref_attn_mask_drop_low_percent": 0.0,
        "ref_attn_mask_min_component_size": 1,
        "ref_attn_mask_hole_size": 0,
        "auto_q_scale": True,
        "q_scale_bg_only": True,
    }
    cfg["background_mask_path"] = None
    return cfg


def _seed_ref_router(backend: str):
    router = FramePromptRouter(
        _fun_ref_attn_config(), backend=backend, patch_height=2, patch_width=2
    )
    router.ref_loc_max_timesteps = 15
    router.subject_fg_masks = torch.zeros(1, 8, 4, dtype=torch.bool)
    router.subject_fg_masks[0, :, 0] = True
    router.static_bg = ~router.subject_fg_masks.any(0)
    return router


def test_fun_ref_loc_keeps_selector_heads_separate():
    cfg = _fun_ref_attn_config()
    cfg["attention_control_schedule"][0]["params"]["ref_attn_mask_selectors"] = [[0, 0], [0, 1]]
    router = FramePromptRouter(cfg, backend="wan_fun_camera", patch_height=2, patch_width=2)
    router.subject_fg_masks = torch.zeros(1, 8, 4, dtype=torch.bool)
    router.subject_fg_masks[0, :, 0] = True
    router.static_bg = ~router.subject_fg_masks.any(0)
    module = _Attn()
    module._ss_grid_sizes = torch.tensor([[8, 2, 2]])
    module._ss_v_for_ref = torch.zeros(1, 32, 2, 4)
    q = torch.zeros(1, 32, 2, 4)
    q[:, :, 1, :] = 1
    k = torch.zeros(1, 32, 2, 4)
    k[:, 0, 1, :] = 4
    base_adapter._collect_live_ref_attn(router, module, q, k, 3, 0, "cond")
    first = router._pending_ref_loc_scores[(3, 0, 0)][0]
    second = router._pending_ref_loc_scores[(3, 0, 1)][0]
    assert not torch.equal(first, second)


def test_fun_ref_loc_mask_keeps_score_ties_in_background():
    cfg = canonical_config(num_frames=1, frame_splits=[])
    cfg["attention_control_schedule"][0]["timestep_range"] = [3, 39]
    cfg["attention_control_schedule"][0]["params"] = {
        "mask_type": "ref_attn",
        "ref_attn_mask_selectors": [[37, 32], [37, 35]],
        "ref_attn_mask_drop_low_percent": 0.0,
        "ref_attn_mask_min_component_size": 1,
        "ref_attn_mask_hole_size": 0,
    }
    fg = torch.tensor([[[0.50, 0.20, 0.10, 0.00]]])
    bg = torch.tensor([[0.40, 0.20, 0.30, 0.80]])
    fun = FramePromptRouter(cfg, backend="wan_fun_camera", patch_height=2, patch_width=2)
    other = FramePromptRouter(cfg, backend="multitalk", patch_height=2, patch_width=2)
    for router in (fun, other):
        router.store_ref_loc_head(3, 37, 32, fg, bg)
        router.store_ref_loc_head(3, 37, 35, fg.clone(), bg.clone())
        router._ss_timestep = 3
    previous = torch.ones(1, 4, dtype=torch.bool)
    fun.live_bg = previous
    fun.store_ref_loc_head(3, 37, 32, torch.zeros_like(fg), torch.ones_like(bg))
    fun.store_ref_loc_head(3, 37, 35, torch.zeros_like(fg), torch.ones_like(bg))
    fun.complete_denoising_step()
    assert fun.live_bg is previous

    fun._pending_ref_loc_scores = {}
    fun._ref_loc_scores = {}
    fun.live_bg = None
    fun.store_ref_loc_head(3, 37, 32, fg, bg)
    fun.store_ref_loc_head(3, 37, 35, fg.clone(), bg.clone())
    fun.complete_denoising_step()
    assert fun.live_bg.tolist() == [[False, True, True, True]]

    other.publish_ref_loc_scores()
    other.refresh_live_mask_from_stored(3, other.active_block(3, None, "cond"))
    assert bool(other.live_bg[0, 1].item()) is False


def test_fun_ref_loc_waits_for_schedule_and_skips_per_layer_refresh():
    fun = _seed_ref_router("wan_fun_camera")
    other = _seed_ref_router("multitalk")
    module = _Attn()
    module._ss_grid_sizes = torch.tensor([[8, 2, 2]])
    module._ss_v_for_ref = torch.zeros(1, 32, 1, 4)
    q = torch.randn(1, 32, 1, 4)

    base_adapter._collect_live_ref_attn(fun, module, q, q, 2, 0, "cond")
    assert fun._pending_ref_loc_scores == {}
    base_adapter._collect_live_ref_attn(other, module, q, q, 2, 0, "cond")
    assert (2, 0, 0) in other._pending_ref_loc_scores

    base_adapter._collect_live_ref_attn(fun, module, q, q, 3, 0, "cond")
    assert (3, 0, 0) in fun._pending_ref_loc_scores
    assert (3, 0, 0) not in fun._ref_loc_scores
    base_adapter._collect_live_ref_attn(fun, module, q, q, 40, 0, "cond")
    assert (40, 0, 0) not in fun._pending_ref_loc_scores

    sentinel = torch.zeros(8, 4, dtype=torch.bool)
    fun.live_bg = sentinel
    fun._ref_loc_scores[(0, 0, 0)] = (
        torch.ones(1, 8, 4),
        torch.zeros(8, 4),
    )

    def fake_flash(*args, **kwargs):
        return args[0]

    module.frame_prompt_router = fun
    module._ss_timestep = 3
    module._ss_block_idx = 0
    module._ss_pass = "cond"
    scene_masked_official_flash(
        fake_flash,
        "flash_attention",
        module,
        fun,
        (q, q.clone(), q.clone()),
        {"k_lens": torch.tensor([32])},
    )
    assert fun.live_bg is sentinel

    fun._ss_timestep = 2
    fun.complete_denoising_step()
    assert fun.live_bg is sentinel
    fun._ss_timestep = 3
    fun.complete_denoising_step()
    assert (3, 0, 0) in fun._ref_loc_scores
    assert fun.live_bg is not None


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
    seen = []

    def scores(visual_q, ref_k, shape, targets, *, split_num=2):
        seen.append(
            (
                int(visual_q.shape[0]),
                int(visual_q.shape[1]),
                int(visual_q.shape[2]),
                int(split_num),
                float(visual_q[0, 0, 0, 0]),
            )
        )
        return torch.zeros(
            int(targets.shape[0]),
            int(visual_q.shape[1]),
            int(visual_q.shape[2]),
        )

    monkeypatch.setattr(
        base_adapter, "get_attn_map_with_target_per_head_split", scores
    )
    module = _Attn()
    module._ss_grid_sizes = torch.tensor([[8, 2, 2]])
    module._ss_v_for_ref = torch.zeros(2, 32, 1, 1)
    q = torch.stack(
        [torch.zeros(32, 1, 1), torch.ones(32, 1, 1)], dim=0
    )
    base_adapter._collect_live_ref_attn(router, module, q, q, 0, 0, "cond")
    # Canonical split is frame 3: 12 tokens, then 20. Cond row only.
    assert seen == [(1, 12, 1, 2, 1.0), (1, 20, 1, 2, 1.0)]


def test_fun_ref_loc_scores_compiled_chunks_per_segment_not_selector_heads(monkeypatch):
    """The old Fun mask scored the globally rotated query for every segment."""
    from sceneshifter.rope import rope_apply

    cfg = _fun_ref_attn_config()
    cfg["attention_control_schedule"][0]["params"]["ref_attn_mask_selectors"] = [
        [0, 2],
        [0, 3],
    ]
    router = FramePromptRouter(
        cfg, backend="wan_fun_camera", patch_height=2, patch_width=2
    )
    router.subject_fg_masks = torch.zeros(1, 8, 4, dtype=torch.bool)
    router.subject_fg_masks[0, :, 0] = True
    router.static_bg = ~router.subject_fg_masks.any(0)
    seen = []

    def scores(visual_q, ref_k, shape, targets, *, split_num=2):
        seen.append(visual_q.detach().clone())
        return torch.zeros(
            int(targets.shape[0]),
            int(visual_q.shape[1]),
            int(visual_q.shape[2]),
        )

    monkeypatch.setattr(
        base_adapter, "get_attn_map_with_target_per_head_split", scores
    )
    module = _Attn()
    module._ss_grid_sizes = torch.tensor([[8, 2, 2]])
    module._ss_v_for_ref = torch.zeros(1, 32, 4, 6, dtype=torch.bfloat16)
    q = torch.zeros(1, 32, 4, 6, dtype=torch.bfloat16)
    pre = torch.ones(1, 32, 4, 6, dtype=torch.bfloat16)
    freqs = torch.randn(16, 3, dtype=torch.float64) + 1j * torch.randn(
        16, 3, dtype=torch.float64
    )
    module._ss_q_pre_rope = pre
    module._ss_freqs = freqs
    base_adapter._collect_live_ref_attn(router, module, q, q, 3, 0, "cond")
    assert [(int(item.shape[1]), int(item.shape[2])) for item in seen] == [
        (12, 4),
        (20, 4),
    ]
    assert torch.equal(seen[0], q[:, :12])
    block = router.active_block(3, 0, "cond")
    scaled = router.apply_q_scale(pre, block)
    expected = rope_apply(
        scaled[:, 12:32],
        torch.tensor([[5, 2, 2]], dtype=torch.long),
        freqs,
    ).to(dtype=q.dtype)
    assert torch.equal(seen[1], expected)
    assert not torch.equal(seen[1], q[:, 12:32])
    assert (3, 0, 2) in router._pending_ref_loc_scores
    assert (3, 0, 3) in router._pending_ref_loc_scores


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
    # Same-forward layers must not see these scores. The next pass publishes them.
    assert (0, 0, 0) not in router._ref_loc_scores
    assert (0, 0, 0) in router._pending_ref_loc_scores
    assert router.selector_avg_scores(0, [[0, 0]]) == (None, None)
    router.publish_ref_loc_scores()
    assert (0, 0, 0) in router._ref_loc_scores
    assert router._pending_ref_loc_scores == {}
    fg, bg = router.selector_avg_scores(0, [[0, 0]])
    assert fg is not None and bg is not None


@pytest.mark.parametrize("backend", ["multitalk", "infinitetalk"])
def test_talking_head_person_map_q_is_the_scaled_query(monkeypatch, backend):
    cfg = canonical_config()
    router = FramePromptRouter(cfg, backend=backend, patch_height=2, patch_width=2)
    router.apply_q_scale = lambda q, block: q * 2
    collected = {}

    def collect(router, module, q, k, timestep_idx, block_idx, pass_type):
        collected["q"] = q
        collected["ptr"] = q.data_ptr()

    monkeypatch.setattr(base_adapter, "_collect_live_ref_attn", collect)

    def flash(q, k, v, *args, **kwargs):
        return torch.zeros_like(q)

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
    assert torch.equal(q, torch.full_like(q, 2.0))
    assert torch.equal(collected["q"], torch.full_like(q, 2.0))
    assert collected["ptr"] != q.data_ptr()


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
    assert (0, 0, 0) in router._pending_ref_loc_scores
    assert (0, 0, 1) in router._pending_ref_loc_scores
    assert (0, 0, 0) not in router._ref_loc_scores
    router.publish_ref_loc_scores()
    assert (0, 0, 0) in router._ref_loc_scores
    assert (0, 0, 1) in router._ref_loc_scores


def test_multitalk_new_scene_scores_use_local_rope(monkeypatch):
    from sceneshifter.rope import rope_apply

    cfg = canonical_config()
    cfg["attention_control_schedule"][0]["params"] = {
        "mask_type": "ref_attn",
        "ref_attn_mask_selectors": [[0, 0]],
        "auto_q_scale": True,
        "q_scale_bg_only": False,
        "q_scale": 0.5,
    }
    router = FramePromptRouter(cfg, backend="multitalk", patch_height=2, patch_width=2)
    router.subject_fg_masks = torch.zeros(1, 8, 4, dtype=torch.bool)
    router.static_bg = ~router.subject_fg_masks.any(0)
    pre = torch.randn(1, 32, 2, 6)
    global_q = torch.randn(1, 32, 2, 6)
    freqs = torch.randn(16, 3, dtype=torch.float64) + 1j * torch.randn(16, 3, dtype=torch.float64)
    seen = []

    def scores(visual_q, ref_k, shape, targets, *, split_num=2):
        seen.append(visual_q.detach().clone())
        return torch.zeros(
            2, int(visual_q.shape[1]), int(visual_q.shape[2]), dtype=visual_q.dtype
        )

    monkeypatch.setattr(base_adapter, "get_attn_map_with_target_per_head_split", scores)
    module = _Attn()
    module._ss_grid_sizes = torch.tensor([[8, 2, 2]])
    module._ss_v_for_ref = torch.zeros(1, 32, 2, 6, dtype=torch.bfloat16)
    module._ss_q_pre_rope = pre
    module._ss_freqs = freqs
    base_adapter._collect_live_ref_attn(router, module, global_q, global_q, 0, 0, "cond")

    assert [int(q.shape[1]) for q in seen] == [12, 20]
    assert torch.equal(seen[0], global_q.to(torch.bfloat16)[:, :12])
    scaled = pre.clone()
    scaled[:, 12:] *= 0.5
    expected = rope_apply(scaled[:, 12:], torch.tensor([[5, 2, 2]]), freqs).to(torch.bfloat16)
    assert torch.equal(seen[1], expected)
    assert (0, 0, 0) in router._pending_ref_loc_scores


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

    monkeypatch.setattr(base_adapter, "get_attn_map_with_target_animate", scores)
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
