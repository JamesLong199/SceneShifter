import torch
import torch.nn as nn

from sceneshifter.adapters.base import thread_timestep
from sceneshifter.parity.unified import _install_encode_prompt_hook, _unwrap_prompt_embed
from sceneshifter.router import FramePromptRouter
from tests.conftest import canonical_config


def test_unwrap_diffsynth_context_dict():
    tensor = torch.ones(1, 4, 8)
    assert _unwrap_prompt_embed({"context": tensor}) is tensor
    assert _unwrap_prompt_embed([tensor]) is tensor


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
