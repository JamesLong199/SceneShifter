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


@pytest.mark.feature
def test_s2v_defaults_to_cover_original_without_changing_other_backends(tmp_path):
    foreground = torch.zeros(8, 8)
    foreground[1, 1] = 1
    path = tmp_path / "sparse.npy"
    np.save(path, (~foreground.bool()).float().numpy())
    cfg = canonical_config(background_mask_path={"0": str(path)})

    s2v = FramePromptRouter(
        cfg, backend="wan_s2v", patch_height=2, patch_width=2
    )
    multitalk = FramePromptRouter(
        cfg, backend="multitalk", patch_height=2, patch_width=2
    )
    assert s2v.subject_fg_masks.any()
    assert not multitalk.subject_fg_masks.any()

    cfg["input_mask_cover_original"] = False
    s2v_override = FramePromptRouter(
        cfg, backend="wan_s2v", patch_height=2, patch_width=2
    )
    assert not s2v_override.subject_fg_masks.any()


@pytest.mark.feature
def test_cover_head_only_unions_nearest_human_and_cover_head(tmp_path):
    from sceneshifter.masks import downsample_subject_fg, downsample_subject_fg_head_union

    human = torch.zeros(1, 8, 8, 8)
    human[..., :4, :4] = 1
    head = torch.zeros(1, 8, 8, 8)
    head[..., 6, 6] = 1
    nearest, _ = downsample_subject_fg(human.bool(), 2, 2)
    head_cover, _ = downsample_subject_fg(head.bool(), 2, 2, cover_original=True)
    full_cover, _ = downsample_subject_fg(human.bool(), 2, 2, cover_original=True)
    union, bg = downsample_subject_fg_head_union(human.bool(), head.bool(), 2, 2)
    assert nearest.any()
    assert head_cover.any()
    assert torch.equal(union, nearest | head_cover)
    assert int(union.sum().item()) > int(nearest.sum().item())
    assert not torch.equal(union, full_cover)
    assert bool((union | bg).all())

    human_path = tmp_path / "human.npy"
    head_path = tmp_path / "head.npy"
    # Files are 1=BG, 0=FG.
    np.save(human_path, (~human[0, 0].bool()).float().numpy())
    np.save(head_path, (~head[0, 0].bool()).float().numpy())
    cfg = canonical_config(
        background_mask_path={"0": str(human_path)},
        num_frames=8,
    )
    cfg["head_mask_path"] = {"0": str(head_path)}
    cfg["input_mask_cover_original"] = True
    cfg["input_mask_cover_head_only"] = True
    router = FramePromptRouter(cfg, patch_height=2, patch_width=2)
    assert torch.equal(router.subject_fg_masks, union)
    assert torch.equal(router.static_bg, bg)
