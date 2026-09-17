"""Layer C: 2-step toy forward per backbone adapter (no official weights)."""

from __future__ import annotations

import pytest
import torch

from sceneshifter.hooks import inject_router
from sceneshifter.router import FramePromptRouter
from tests.conftest import TinyDiT, canonical_config

BACKENDS = [
    "multitalk",
    "infinitetalk",
    "fantasytalking",
    "hunyuan_avatar",
    "wan_s2v",
    "wan_animate",
    "wan_fun_camera",
]


@pytest.mark.pipeline
@pytest.mark.parametrize("backend", BACKENDS)
def test_toy_two_step_forward(backend):
    cfg = canonical_config()
    if backend == "hunyuan_avatar":
        cfg["attention_control_schedule"][0]["params"]["apply_on_uncond_passes"] = True
        cfg["attention_control_schedule"][0]["params"]["q_scale_bg_only"] = False
    router = FramePromptRouter(cfg, backend=backend, patch_height=2, patch_width=2)
    model = TinyDiT()
    bound = inject_router(model, router, adapter=backend, expected_min=1)
    assert bound
    x = torch.randn(1, 32, 16)
    for step in range(2):
        out = model(x, timestep_idx=step, pass_type="cond")
        assert out.shape == x.shape
        x = out
    assert model.self_attn.frame_prompt_router is router


@pytest.mark.pipeline
def test_disabled_router_keeps_baseline_forward():
    model = TinyDiT()
    x = torch.randn(1, 32, 16)
    baseline = model(x).detach()
    inject_router(model, None, adapter="multitalk", expected_min=0)
    out = model(x)
    assert out.shape == baseline.shape
