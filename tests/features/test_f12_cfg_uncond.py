import pytest

from sceneshifter.checks import SceneShifterCheckError, validate_config
from sceneshifter.router import FramePromptRouter
from tests.conftest import canonical_config, canonical_mask


@pytest.mark.feature
def test_uncond_skips_mask_by_default():
    router = FramePromptRouter(canonical_config())
    assert router.active_block(0, 0, "cond") is not None
    assert router.active_block(0, 0, "uncond") is None


@pytest.mark.feature
def test_uncond_applies_when_flag_true():
    cfg = canonical_config()
    cfg["attention_control_schedule"][0]["params"]["apply_on_uncond_passes"] = True
    router = FramePromptRouter(cfg)
    assert router.active_block(0, 0, "uncond") is not None


@pytest.mark.feature
def test_hunyuan_rejects_false():
    cfg = canonical_config()
    cfg["attention_control_schedule"][0]["params"]["apply_on_uncond_passes"] = False
    with pytest.raises(SceneShifterCheckError):
        validate_config(cfg, backend="hunyuan_avatar")
