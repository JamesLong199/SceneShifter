import pytest
import torch

from sceneshifter.router import FramePromptRouter
from tests.conftest import canonical_config


@pytest.mark.feature
def test_fg_bg_full_key_sets(bg_mask_path):
    router = FramePromptRouter(
        canonical_config(background_mask_path=bg_mask_path),
        patch_height=2,
        patch_width=2,
    )
    full = set(router.spatial_indices(0, "full").tolist())
    fg = set(router.spatial_indices(0, "fg").tolist())
    bg = set(router.spatial_indices(0, "bg").tolist())
    assert full == set(range(12))  # 3 frames * 4 tokens
    assert fg.isdisjoint(bg)
    assert fg | bg == full
    assert 0 in fg  # first token of frame 0
