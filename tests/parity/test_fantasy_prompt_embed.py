import pytest
import torch
import torch.nn as nn

from sceneshifter.adapters.base import thread_timestep
from sceneshifter.parity.unified import (
    _filter_official_argv,
    _inject,
    _install_encode_prompt_hook,
    _install_fantasy_audio_cfg_gate,
    _unwrap_prompt_embed,
)
from sceneshifter.router import FramePromptRouter
from tests.conftest import TinyDiT, canonical_config


def test_unwrap_diffsynth_context_dict():
    tensor = torch.ones(1, 4, 8)
    assert _unwrap_prompt_embed({"context": tensor}) is tensor
    assert _unwrap_prompt_embed([tensor]) is tensor


def test_encode_prompt_hook_fun_camera_signature():
    cfg = canonical_config()
    router = FramePromptRouter(cfg, backend="wan_fun_camera", patch_height=2, patch_width=2)
    pos = {"a": torch.randn(1, 3, 4), "b": torch.randn(1, 3, 4)}
    neg = torch.randn(1, 3, 4)

    class Pipe:
        def encode_prompt(self, prompt, negative_prompt=None, do_classifier_free_guidance=True):
            if do_classifier_free_guidance:
                return pos.get(prompt, pos["a"]), neg
            return pos[prompt], None

    pipe = Pipe()
    _install_encode_prompt_hook(pipe, router, {"segment_prompts": ["a", "b"]})
    out = pipe.encode_prompt("a", negative_prompt="bad")
    assert out[0] is pos["a"]
    assert router.text_embeddings is not None
    assert router.text_embeddings[0] is pos["a"]
    assert router.null_text_embeddings is neg


def test_encode_prompt_hook_captures_null_and_segments():
    cfg = canonical_config()
    router = FramePromptRouter(cfg, backend="fantasytalking", patch_height=2, patch_width=2)
    pos = {"a": torch.randn(1, 3, 4), "b": torch.randn(1, 3, 4)}
    neg = torch.randn(1, 3, 4)

    class Pipe:
        def encode_prompt(self, prompt, positive=True):
            if not positive:
                return {"context": neg}
            return {"context": pos[prompt]}

    pipe = Pipe()
    _install_encode_prompt_hook(pipe, router, {"segment_prompts": ["a", "b"]})
    first = pipe.encode_prompt("a", positive=True)
    assert first["context"] is pos["a"]
    assert router.text_embeddings is not None
    assert router.text_embeddings[0] is pos["a"]
    assert router.text_embeddings[1] is pos["b"]
    pipe.encode_prompt("neg", positive=False)
    assert router.null_text_embeddings is neg


def test_text_capture_is_noop_when_routing_disabled():
    from sceneshifter.parity.unified import _capture_null_context

    class Pipe:
        def __init__(self):
            self.encode_prompt = lambda prompt: prompt

    pipe = Pipe()
    original = pipe.encode_prompt
    _install_encode_prompt_hook(pipe, None, {})
    _capture_null_context(pipe, None)
    assert pipe.encode_prompt is original


def test_thread_timestep_zeros_audio_scale_before_start():
    cfg = canonical_config()
    cfg["audio_start_timestep"] = 15
    router = FramePromptRouter(cfg, backend="fantasytalking", patch_height=2, patch_width=2)

    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, x, **kwargs):
            self.seen.append(dict(kwargs))
            return x

    model = Dummy()
    model.self_attn = nn.Identity()
    model.frame_prompt_router = router
    thread_timestep(model)
    x = torch.zeros(1)
    model(x, t=1.0, audio_scale=1.0, audio_start_timestep=15, context=torch.ones(1, 2))
    assert model.seen[0]["audio_scale"] == 0.0
    assert "audio_start_timestep" not in model.seen[0]
    for _ in range(15):
        model(x, t=float(len(model.seen) + 1), audio_scale=1.0, context=torch.ones(1, 2))
    model(x, t=99.0, audio_scale=1.0, context=torch.ones(1, 2))
    assert model.seen[-1]["audio_scale"] == 1.0


def test_record_single_prompt_t5_sets_segments_then_null():
    from sceneshifter.parity.unified import _record_official_text_encode

    cfg = canonical_config()
    router = FramePromptRouter(cfg, backend="wan_animate", patch_height=2, patch_width=2)
    store = {
        "scene 0": torch.randn(1, 3, 4),
        "scene 1": torch.randn(1, 3, 4),
        "neg": torch.randn(1, 3, 4),
    }

    def encode_fn(texts, *args, **kwargs):
        return store[texts[0]]

    _record_official_text_encode(router, encode_fn, ["scene 0"], store["scene 0"], (), {})
    assert router.text_embeddings[0] is store["scene 0"]
    assert router.text_embeddings[1] is store["scene 1"]
    _record_official_text_encode(router, encode_fn, ["neg"], store["neg"], (), {})
    assert router.null_text_embeddings is store["neg"]


@pytest.mark.parametrize("backend", ["wan_s2v", "wan_animate"])
def test_wan_records_segments_in_one_batch_and_replaces_baseline(backend):
    from sceneshifter.parity.unified import _record_official_text_encode

    cfg = canonical_config()
    router = FramePromptRouter(cfg, backend=backend, patch_height=2, patch_width=2)
    store = {
        "scene 0": torch.randn(3, 4),
        "scene 1": torch.randn(5, 4),
        "original": torch.randn(3, 4),
    }
    calls = []

    def encode_fn(texts, *args, **kwargs):
        calls.append(list(texts))
        return [store[text] for text in texts]

    replacement = _record_official_text_encode(
        router,
        encode_fn,
        ["scene 0"],
        [store["original"]],
        (),
        {},
    )
    assert calls == [["scene 0", "scene 1"]]
    assert isinstance(replacement, list)
    assert replacement[0] is store["scene 0"]
    assert router.text_embeddings is not None
    assert router.text_embeddings[0] is store["scene 0"]
    assert router.text_embeddings[1] is store["scene 1"]


def test_fantasy_cross_attn_keeps_full_sequence_for_audio():
    from sceneshifter.adapters.base import wrap_cross_attention
    from sceneshifter.adapters.fantasytalking import (
        FantasyTalkingAdapter,
        wrap_fantasy_i2v_cross_module,
    )

    cfg = canonical_config(num_frames=4, frame_splits=[2], segment_prompts=["a", "b"])
    router = FramePromptRouter(cfg, backend="fantasytalking", patch_height=2, patch_width=2)
    dim, heads, frames, spatial = 8, 2, 4, 4
    router.set_text_embeddings([torch.randn(1, 3, dim), torch.randn(1, 3, dim)])

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = heads
            self.head_dim = dim // heads
            self.q = nn.Linear(dim, dim, bias=False)
            self.k = nn.Linear(dim, dim, bias=False)
            self.v = nn.Linear(dim, dim, bias=False)
            self.o = nn.Linear(dim, dim, bias=False)
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
            self.k_img = nn.Linear(dim, dim, bias=False)
            self.v_img = nn.Linear(dim, dim, bias=False)
            self.norm_k_img = nn.Identity()

        def forward(self, x, context, *args, **kwargs):
            return self.processor(
                self,
                x,
                context,
                None,
                kwargs["audio_proj"],
                None,
                latents_num_frames=frames,
                audio_scale=1.0,
            )

    class Processor(nn.Module):
        def __init__(self):
            super().__init__()
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)

        def __call__(
            self,
            attn,
            x,
            context,
            context_lens,
            audio_proj,
            audio_context_lens,
            latents_num_frames=4,
            audio_scale=1.0,
            **kwargs,
        ):
            b, n, d = x.size(0), attn.num_heads, attn.head_dim
            q = attn.norm_q(attn.q(x)).view(b, -1, n, d)
            audio_q = q.view(b * latents_num_frames, -1, n, d)
            return attn.o(audio_q.reshape(b, q.size(1), -1))

    attn = Attn()
    attn.processor = Processor()
    wrap_cross_attention(attn, router)
    assert attn.forward.__func__ is Attn.forward

    x = torch.randn(1, frames * spatial, dim)
    context = torch.randn(1, 3, dim)
    audio = torch.randn(1, frames, 2, dim)
    wrap_fantasy_i2v_cross_module(attn, router)
    out = attn(
        x,
        context,
        None,
        audio,
        None,
        frames,
        1.0,
    )
    assert out.shape == x.shape
    assert attn._ss_fantasy_cross_wrapped

    model = nn.Module()
    model.cross_attn = Attn()
    model.cross_attn.processor = Processor()
    model.self_attn = nn.Identity()
    bound = FantasyTalkingAdapter().inject(model, router)
    assert any(getattr(m, "_ss_fantasy_cross_wrapped", False) for m in bound)


def test_encode_prompt_hook_keeps_raw_t5():
    cfg = canonical_config()
    router = FramePromptRouter(cfg, backend="fantasytalking", patch_height=2, patch_width=2)
    raw = {"a": torch.randn(5, 8), "b": torch.randn(3, 8)}

    class Pipe(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_embedding = nn.Sequential(nn.Linear(8, 16, bias=False))
            self.text_len = 4

        def encode_prompt(self, prompt, positive=True):
            return {"context": raw[prompt]}

    pipe = Pipe()
    _install_encode_prompt_hook(pipe, router, {"segment_prompts": ["a", "b"]})
    pipe.encode_prompt("a", positive=True)
    assert router.text_embeddings[0].shape == (5, 8)
    assert router.text_embeddings[1].shape == (3, 8)


def test_fantasy_audio_cfg_gate_reuses_posi_before_start():
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.n = 0
            self.audio_scales = []

        def forward(self, x, timestep=0.0, **kwargs):
            self.n += 1
            self.audio_scales.append(kwargs.get("audio_scale"))
            return x + self.n

    model = Dummy()
    _install_fantasy_audio_cfg_gate(model, audio_start=2)
    first = model(torch.zeros(1), timestep=9.0, audio_scale=1.0)
    second = model(torch.ones(1), timestep=9.0, audio_scale=1.0)
    third = model(torch.zeros(1), timestep=8.0, audio_scale=1.0)
    model(torch.zeros(1), timestep=7.0, audio_scale=1.0)
    assert torch.equal(first, second)
    assert model.n == 3
    assert not torch.equal(third, first)
    assert model.audio_scales == [0.0, 0.0, 1.0]


def test_extract_qkv_mixed_v_keyword():
    from sceneshifter.adapters.base import _call_official_flash, _extract_qkv

    q, k, v = torch.zeros(1), torch.ones(1), torch.full((1,), 2.0)
    gq, gk, gv, used_kw = _extract_qkv((q, k), {"v": v, "k_lens": 3})
    assert gq is q and gk is k and gv is v
    assert used_kw is False
    seen = {}

    def orig(qq, kk, vv, k_lens=None):
        seen["args"] = (qq, kk, vv, k_lens)

    _call_official_flash(orig, "attention", q, k, v, (q, k), {"v": v, "k_lens": 3}, False, subset=False)
    assert seen["args"] == (q, k, v, 3)


def test_allow_missing_flex_attention_is_safe():
    from sceneshifter.parity.unified import _allow_missing_flex_attention

    _allow_missing_flex_attention()
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

    assert BlockMask is not None
    assert callable(create_block_mask)
    assert callable(flex_attention)


def test_encoder_device_prefers_offloaded_t5():
    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2))

    class Pipe:
        device = "cuda"
        text_encoder = Enc()

    from sceneshifter.parity.unified import _encoder_device

    assert _encoder_device(Pipe()) == torch.device("cpu")


def test_official_argv_rejects_silently_dropped_flags(tmp_path):
    entrypoint = tmp_path / "generate.py"
    entrypoint.write_text('parser.add_argument("--known")\n')

    class Module:
        __file__ = str(entrypoint)

    assert _filter_official_argv(Module, ["--known", "value"]) == ["--known", "value"]
    with pytest.raises(RuntimeError, match="--missing"):
        _filter_official_argv(Module, ["--missing", "value"])


def test_mux_fantasy_audio_uses_env_ffmpeg(tmp_path, monkeypatch):
    from pathlib import Path

    from sceneshifter.parity.unified import _mux_fantasy_audio

    src = tmp_path / "tmp.mp4"
    dest = tmp_path / "out.mp4"
    audio = tmp_path / "a.wav"
    src.write_bytes(b"video")
    audio.write_bytes(b"audio")
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "ffmpeg").write_text("#!/bin/sh\nexit 1\n")
    (fake / "ffmpeg").chmod(0o755)
    monkeypatch.setattr("sys.executable", str(fake / "python"))
    monkeypatch.delenv("PATH", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "missing"))
    _mux_fantasy_audio(src, str(audio), dest)
    assert dest.read_bytes() == b"video"


def test_fun_clip_image_uses_uncropped_cond():
    import inspect

    from sceneshifter.parity import unified

    src = inspect.getsource(unified.run_fun_official)
    assert "clip_image = cropped" not in src
    assert 'Image.open(image).convert("RGB")' in src
    assert "attach_text_projector(router, pipeline.transformer)" in src
    assert 'meta.get("prompt")' in src


def test_fun_official_enables_model_cpu_offload():
    import inspect

    from sceneshifter.parity import unified
    from sceneshifter.parity.common import load_parity_config

    src = inspect.getsource(unified.run_fun_official)
    assert "enable_model_cpu_offload" in src
    assert 'pipeline.enable_model_cpu_offload(device="cuda")' in src
    assert 'pipeline.to("cuda")' not in src
    assert "pipeline.to('cuda')" not in src
    assert 'pipeline.to(device="cuda")' not in src
    assert "pipeline.to(device='cuda')" not in src
    extra = load_parity_config("wan_fun_camera")["reference"]["extra_args"]
    assert extra[extra.index("--gpu_memory_mode") + 1] == "model_cpu_offload"


def test_fun_official_injects_batched_cfg_before_cpu_offload():
    import inspect

    from sceneshifter.parity import unified

    src = inspect.getsource(unified.run_fun_official)
    inject_at = src.index('_inject(pipeline.transformer, router, "wan_fun_camera")')
    offload_at = src.index('pipeline.enable_model_cpu_offload(device="cuda")')
    release_at = src.index("install_fun_camera_encoder_offload(pipeline)")
    nograd_at = src.index("with torch.no_grad():")
    embed_at = src.index("_maybe_set_text_embeddings(router, pipeline, routing)")
    call_at = src.index("video = pipeline(")
    assert inject_at < offload_at < release_at < nograd_at < embed_at < call_at
    assert "transformer.forward =" not in src[offload_at:]


def test_fantasy_auto_bucket_handles_null_image_size(tmp_path):
    from PIL import Image

    from sceneshifter.parity.unified import _resize_fantasy_image

    path = tmp_path / "three_by_two.png"
    Image.new("RGB", (300, 200)).save(path)
    resized = _resize_fantasy_image(str(path), None)
    assert resized.size == (768, 512)


def test_offload_dit_before_vae_moves_noise_model():
    from sceneshifter.parity.unified import _offload_dit_before_vae

    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.moved = False

        def cpu(self):
            self.moved = True
            return self

    class Vae:
        def decode(self, x):
            return ("decoded", x)

    class Pipe:
        noise_model = Dummy()
        vae = Vae()

    pipe = Pipe()
    _offload_dit_before_vae(pipe)
    assert pipe.vae.decode("latents") == ("decoded", "latents")
    assert pipe.noise_model.moved is True
    _offload_dit_before_vae(pipe)
    assert pipe.vae._ss_decode_offload is True


def test_inject_finds_wan_animate_noise_model():
    class Wrapper:
        def __init__(self):
            self.noise_model = TinyDiT()

    router = FramePromptRouter(canonical_config(), patch_height=2, patch_width=2)
    bound = _inject(Wrapper(), router, "wan_animate")
    assert len(bound) >= 1


def test_animate_prompt_is_forwarded_to_official_generate():
    import types

    from sceneshifter.parity.unified import _install_animate_prompt_forwarding

    class WanAnimate:
        def generate(self, **kwargs):
            return kwargs

    module = types.SimpleNamespace(WanAnimate=WanAnimate)
    _install_animate_prompt_forwarding(module, "English parity prompt")
    output = WanAnimate().generate(src_root_path="/tmp/process")
    assert output["input_prompt"] == "English parity prompt"
    explicit = WanAnimate().generate(input_prompt="explicit")
    assert explicit["input_prompt"] == "explicit"


def test_animate_cross_attention_runs_clip_branch_once(monkeypatch):
    from sceneshifter.adapters.wan_animate import wrap_animate_cross_attention

    calls = []

    def flash_attention(q, k, v, k_lens=None):
        calls.append((int(q.shape[1]), int(k.shape[1])))
        return q

    class Attn(nn.Module):
        use_img_emb = True
        num_heads = 2
        head_dim = 4

        def __init__(self):
            super().__init__()
            self.q = nn.Linear(8, 8, bias=False)
            self.k = nn.Linear(8, 8, bias=False)
            self.v = nn.Linear(8, 8, bias=False)
            self.k_img = nn.Linear(8, 8, bias=False)
            self.v_img = nn.Linear(8, 8, bias=False)
            self.o = nn.Linear(8, 8, bias=False)
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
            self.norm_k_img = nn.Identity()

        def forward(self, x, context, context_lens=None):
            return x

    monkeypatch.setitem(Attn.forward.__globals__, "flash_attention", flash_attention)
    cfg = canonical_config()
    router = FramePromptRouter(
        cfg, backend="wan_animate", patch_height=2, patch_width=2
    )
    router.set_text_embeddings([torch.randn(1, 4, 8), torch.randn(1, 4, 8)])
    module = Attn()
    wrap_animate_cross_attention(module, router)
    module._ss_timestep = 3
    module._ss_block_idx = 0
    module._ss_pass = "cond"
    out = module(torch.randn(1, 32, 8), torch.randn(1, 261, 8))
    assert out.shape == (1, 32, 8)
    assert calls == [(12, 4), (20, 4), (32, 257)]
