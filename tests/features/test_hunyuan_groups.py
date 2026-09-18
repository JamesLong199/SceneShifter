import torch

from sceneshifter.adapters.hunyuan_avatar import build_hunyuan_groups
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
