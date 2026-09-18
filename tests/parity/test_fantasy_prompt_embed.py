import torch
import torch.nn as nn

from sceneshifter.adapters.base import thread_timestep
from sceneshifter.parity.unified import (
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

        def forward(self, x, timestep=0.0, **kwargs):
            self.n += 1
            return x + self.n

    model = Dummy()
    _install_fantasy_audio_cfg_gate(model, audio_start=2)
    first = model(torch.zeros(1), timestep=9.0)
    second = model(torch.ones(1), timestep=9.0)
    third = model(torch.zeros(1), timestep=8.0)
    assert torch.equal(first, second)
    assert model.n == 2
    assert not torch.equal(third, first)


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


def test_inject_finds_wan_animate_noise_model():
    class Wrapper:
        def __init__(self):
            self.noise_model = TinyDiT()

    router = FramePromptRouter(canonical_config(), patch_height=2, patch_width=2)
    bound = _inject(Wrapper(), router, "wan_animate")
    assert len(bound) >= 1
