import pytest
import torch

from sceneshifter.masks import upgrade_context_token_indices
from sceneshifter.router import FramePromptRouter, resolve_transition_context_windows
from tests.conftest import canonical_config


@pytest.mark.feature
def test_fixture_windows():
    router = FramePromptRouter(canonical_config())
    wins = router.get_transition_context_windows()
    assert wins == [{"new_segment": 1, "query_frames": [3, 4], "context_frames": [2]}]
    assert router.should_apply_transition_context(2) is True
    assert router.should_apply_transition_context(5) is True
    assert router.should_apply_transition_context(1) is False
    assert router.should_apply_transition_context(6) is False


@pytest.mark.feature
def test_multitalk_three_scene_windows():
    bounds = [(0, 4), (4, 17), (17, 21)]
    wins = resolve_transition_context_windows(bounds, 2, 1)
    assert wins[0]["query_frames"] == [4, 5]
    assert wins[0]["context_frames"] == [3]
    assert wins[1]["query_frames"] == [17, 18]
    assert wins[1]["context_frames"] == [16]


@pytest.mark.feature
def test_context_upgrade_keeps_per_frame_tokens():
    # Frames 0 and 1 keep their own FG tokens. Frame 2 is context and becomes full.
    keep = torch.tensor([0, 5, 8])
    upgraded = upgrade_context_token_indices(keep, [2], 4)
    assert upgraded.tolist() == [0, 5, 8, 9, 10, 11]


@pytest.mark.feature
def test_range_none_and_none_timestep():
    cfg = canonical_config()
    cfg.pop("trans_timestep_range")
    router = FramePromptRouter(cfg)
    assert router.should_apply_transition_context(None) is True


@pytest.mark.feature
def test_range_requires_timestep():
    router = FramePromptRouter(canonical_config())
    with pytest.raises(ValueError):
        router.should_apply_transition_context(None)
