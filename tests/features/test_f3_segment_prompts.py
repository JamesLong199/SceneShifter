import pytest

from sceneshifter.router import FramePromptRouter
from tests.conftest import canonical_config


@pytest.mark.feature
def test_cond_uses_segment_after_start():
    router = FramePromptRouter(canonical_config(segment_prompt_start_timestep=4))
    assert router.selected_text_index(3, 1, "cond") == 0
    assert router.selected_text_index(4, 1, "cond") == 1
    assert router.selected_text_index(4, 0, "cond") == 0


@pytest.mark.feature
def test_uncond_never_routed():
    router = FramePromptRouter(canonical_config())
    assert router.should_use_segment_prompts(10, "uncond") is False
    assert router.selected_text_index(10, 1, "uncond") is None


@pytest.mark.feature
def test_hunyuan_text_always_per_scene():
    router = FramePromptRouter(
        canonical_config(segment_prompt_start_timestep=9),
        backend="hunyuan_avatar",
    )
    assert router.should_use_segment_prompts(0, "cond") is True
    assert router.selected_text_index(0, 1, "cond") == 1
