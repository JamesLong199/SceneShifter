import pytest

from sceneshifter.router import scene_frame_bounds
from tests.conftest import canonical_config
from sceneshifter.router import FramePromptRouter


@pytest.mark.feature
def test_scene_frame_bounds_fixture():
    assert scene_frame_bounds(8, [3]) == [(0, 3), (3, 8)]


@pytest.mark.feature
def test_noisy_token_ranges():
    router = FramePromptRouter(canonical_config(), patch_height=2, patch_width=2)
    L = 4
    assert router.segment_ranges() == [(0, 3 * L, 0), (3 * L, 8 * L, 1)]
    assert router.frame_to_segment[0] == 0
    assert router.frame_to_segment[3] == 1
    assert router.frame_to_segment[7] == 1
