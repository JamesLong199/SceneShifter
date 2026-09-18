import torch

from sceneshifter.adapters.hunyuan_avatar import (
    apply_hunyuan_official_masks,
    build_hunyuan_groups,
    _fg_maps,
    _resize_center_crop_mask,
)
from sceneshifter.layouts import HunyuanMMDiTLayout
from sceneshifter.router import FramePromptRouter
from tests.conftest import canonical_config


def test_hunyuan_layout_ref_then_video():
    layout = HunyuanMMDiTLayout(video_frames=8, spatial_tokens=4, text_tokens=6)
    assert layout.ref_length == 4
    assert layout.img_seq_len == 36
    assert layout.total_tokens == 42


def test_hunyuan_groups_offset_ref_and_isolate_text(bg_mask_path):
    cfg = canonical_config(background_mask_path=bg_mask_path)
    cfg["attention_control_schedule"][0]["params"]["apply_on_uncond_passes"] = True
    router = FramePromptRouter(cfg, backend="hunyuan_avatar", patch_height=2, patch_width=2)
    router._hy_cond_text_segments = [(0, 3), (3, 7)]
    router._hy_uncond_text_len = 4
    block = router.active_block(0, 0, "cond")
    groups = build_hunyuan_groups(
        router,
        block,
        "routed",
        img_len=36,
        txt_len=7,
        timestep=0,
        device=torch.device("cpu"),
    )
    assert groups
    all_q = torch.cat([g["q"] for g in groups])
    assert int(all_q.min()) >= 0
    assert int(all_q.max()) < 43
    # Scene-0 queries include the reference prefix [0, 4).
    scene0 = next(g for g in groups if (g["q"] < 4).any())
    assert (scene0["q"] < 4).any()
    # Routed text keys stay inside their own packed range.
    text_groups = [g for g in groups if (g["q"] >= 36).all()]
    assert text_groups
    for group in text_groups:
        text_keys = group["k"][group["k"] >= 36]
        owners = {0 if int(i) < 39 else 1 for i in text_keys.tolist()}
        q_owner = 0 if int(group["q"][0]) < 39 else 1
        assert owners <= {q_owner}


def test_hunyuan_cover_crop_matches_official_formula():
    mask = torch.zeros(16, 24)
    mask[:, :8] = 1
    cropped = _resize_center_crop_mask(mask, (2, 3))
    assert tuple(cropped.shape) == (2, 3)
    # Same-aspect cover-resize: left third stays FG.
    assert float(cropped[:, 0].min()) == 1.0
    assert float(cropped[:, 1:].max()) == 0.0


def test_hunyuan_official_masks_use_cover_crop(tmp_path):
    import numpy as np

    # File polarity is 1=BG, 0=FG. Left third of a 16x24 mask is FG.
    mask = np.ones((16, 24), dtype=np.float32)
    mask[:, :8] = 0.0
    path = tmp_path / "sub0.npy"
    np.save(path, mask)
    cfg = canonical_config(background_mask_path={"0": str(path)})
    router = FramePromptRouter(cfg, backend="hunyuan_avatar", patch_height=2, patch_width=3)
    apply_hunyuan_official_masks(router)
    assert router.subject_fg_masks is not None
    assert tuple(router.subject_fg_masks.shape) == (1, 8, 6)
    assert bool(router.subject_fg_masks[0, 0, 0])
    assert not bool(router.subject_fg_masks[0, 0, 1])
    assert getattr(router, "hy_fg_spatial", None) is not None
    assert int(router.hy_fg_spatial.sum()) == 2


def test_hunyuan_ref_fg_stays_static_after_live(tmp_path):
    import numpy as np

    mask = np.ones((16, 24), dtype=np.float32)
    mask[:, :8] = 0.0
    path = tmp_path / "sub0.npy"
    np.save(path, mask)
    cfg = canonical_config(background_mask_path={"0": str(path)})
    cfg["attention_control_schedule"][0]["params"]["mask_type"] = "ref_attn"
    router = FramePromptRouter(cfg, backend="hunyuan_avatar", patch_height=2, patch_width=3)
    apply_hunyuan_official_masks(router)
    static = router.hy_fg_spatial.clone()
    live = torch.zeros_like(router.subject_fg_masks)
    live[..., 5] = True
    router.live_subject_fg = live
    router.live_bg = ~live.any(0)
    block = router.active_block(0, 0, "cond")
    ref_fg, video_fg = _fg_maps(router, torch.device("cpu"), router.num_frames, 6, block)
    assert torch.equal(ref_fg.cpu(), static.cpu())
    assert bool(video_fg[0, 5])
    assert not bool(ref_fg[5])
