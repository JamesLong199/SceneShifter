import pytest
import torch

from sceneshifter.checks import SceneShifterCheckError, validate_config
from sceneshifter.qscale_scope import default_q_scale_bg_only
from sceneshifter.router import FramePromptRouter
from tests.conftest import canonical_config


@pytest.mark.feature
def test_defaults_are_independent():
    assert default_q_scale_bg_only("hunyuan_avatar") is False
    assert default_q_scale_bg_only("multitalk") is True


@pytest.mark.feature
def test_toggle_scope(bg_mask_path):
    cfg = canonical_config(background_mask_path=bg_mask_path)
    cfg["attention_control_schedule"][0]["params"]["q_scale_bg_only"] = True
    router = FramePromptRouter(cfg, patch_height=2, patch_width=2)
    block = router.active_block(0, 0, "cond")
    mask_true = router.qscale_token_mask(block)
    cfg["attention_control_schedule"][0]["params"]["q_scale_bg_only"] = False
    router_false = FramePromptRouter(cfg, patch_height=2, patch_width=2)
    mask_false = router_false.qscale_token_mask(router_false.active_block(0, 0, "cond"))
    L = 4
    assert mask_false[3 * L :].all()
    assert mask_true[3 * L :].any()
    assert int(mask_true[3 * L :].sum()) < int(mask_false[3 * L :].sum())
    # FG token 0 of each new-scene frame is unscaled when F10 is true
    assert bool(mask_true[3 * L]) is False
    assert bool(mask_false[3 * L]) is True


@pytest.mark.feature
def test_f10_noop_without_f9_is_hard_error():
    cfg = canonical_config()
    cfg["attention_control_schedule"][0]["params"]["auto_q_scale"] = False
    cfg["attention_control_schedule"][0]["params"]["q_scale_bg_only"] = True
    with pytest.raises(SceneShifterCheckError):
        validate_config(cfg)


@pytest.mark.feature
def test_hunyuan_f10_false_is_recommended_not_fatal():
    cfg = canonical_config()
    cfg["attention_control_schedule"][0]["params"]["q_scale_bg_only"] = True
    cfg["attention_control_schedule"][0]["params"]["apply_on_uncond_passes"] = True
    validate_config(cfg, backend="hunyuan_avatar")
