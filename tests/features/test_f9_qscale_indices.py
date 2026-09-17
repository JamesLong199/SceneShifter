import pytest
import torch

from sceneshifter.qscale import formula_hunyuan_avatar, formula_multitalk, formula_wan_s2v
from sceneshifter.router import FramePromptRouter
from tests.conftest import canonical_config


@pytest.mark.feature
def test_new_scene_all_tokens_when_f10_false():
    router = FramePromptRouter(canonical_config(), patch_height=2, patch_width=2)
    block = router.active_block(0, 0, "cond")
    mask = router.qscale_token_mask(block)
    L = 4
    assert not mask[: 3 * L].any()
    assert mask[3 * L :].all()


@pytest.mark.feature
def test_apply_override_factor():
    router = FramePromptRouter(canonical_config(), patch_height=2, patch_width=2)
    block = router.active_block(0, 0, "cond")
    q = torch.ones(1, 32, 2, 2)
    scaled = router.apply_q_scale(q, block)
    assert torch.allclose(scaled[:, :12], torch.ones_like(scaled[:, :12]))
    assert torch.allclose(scaled[:, 12:], torch.full_like(scaled[:, 12:], 0.5))


@pytest.mark.feature
def test_formulas_differ():
    assert formula_multitalk(7) != formula_wan_s2v(7)
    assert formula_wan_s2v(7) != formula_hunyuan_avatar(7)
