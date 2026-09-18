import torch
import torch.nn as nn

from sceneshifter.adapters.base import (
    project_router_text_embeddings,
    scene_masked_official_flash,
)
from sceneshifter.router import FramePromptRouter
from tests.conftest import canonical_config


class _Attn(nn.Module):
    pass


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
