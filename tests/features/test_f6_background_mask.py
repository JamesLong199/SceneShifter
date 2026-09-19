import numpy as np
import pytest
import torch

from sceneshifter.masks import downsample_mask_to_latent, subject_fg_from_background_paths
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


@pytest.mark.feature
def test_s2v_mask_center_crop_matches_native_geometry():
    mask = torch.zeros(4, 8)
    mask[:, 2:6] = 1
    cropped = downsample_mask_to_latent(mask, 2, 2, center_crop=True)
    stretched = downsample_mask_to_latent(mask, 2, 2)
    assert cropped.all()
    assert not stretched.all()


@pytest.mark.feature
def test_cover_original_keeps_sparse_foreground():
    mask = torch.zeros(8, 8)
    mask[1, 1] = 1
    normal = downsample_mask_to_latent(mask, 2, 2)
    covered = downsample_mask_to_latent(mask, 2, 2, cover_original=True)
    assert not normal.any()
    assert covered.any()
