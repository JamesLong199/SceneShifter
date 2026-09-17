import numpy as np
import pytest
import torch

from sceneshifter.masks import subject_fg_from_background_paths
from sceneshifter.router import FramePromptRouter
from tests.conftest import canonical_config


@pytest.mark.feature
def test_polarity_and_broadcast(bg_mask_path):
    fg = subject_fg_from_background_paths(bg_mask_path, num_frames=8)
    assert fg.shape[0] == 1
    assert fg.shape[1] == 8
    # token 0 is FG
    assert bool(fg[0, 0, 0, 0]) is True
    assert bool(fg[0, 0, 0, 1]) is False


@pytest.mark.feature
def test_router_loads_mask(bg_mask_path):
    cfg = canonical_config(background_mask_path=bg_mask_path)
    router = FramePromptRouter(cfg, patch_height=2, patch_width=2)
    assert router.static_bg is not None
    assert router.static_bg.shape == (8, 4)
    assert bool(router.static_bg[0, 0]) is False  # FG
    assert bool(router.static_bg[0, 1]) is True   # BG
