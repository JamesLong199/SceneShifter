import pytest
import torch

from sceneshifter.adapters.base import _query_indices
from sceneshifter.router import FramePromptRouter
from tests.conftest import canonical_config


@pytest.mark.feature
def test_split_query_indices_are_global(bg_mask_path):
    router = FramePromptRouter(
        canonical_config(background_mask_path=bg_mask_path),
        patch_height=2,
        patch_width=2,
    )
    # Segment 1 starts at frame 3; local FG/BG ids must be shifted by 3*4.
    fg = _query_indices(router, 1, "fg", torch.device("cpu"))
    bg = _query_indices(router, 1, "bg", torch.device("cpu"))
    full = _query_indices(router, 1, "full", torch.device("cpu"))
    assert set(full.tolist()) == set(range(12, 32))
    assert set(fg.tolist()).isdisjoint(set(bg.tolist()))
    assert set(fg.tolist()) | set(bg.tolist()) == set(full.tolist())
    assert all(int(i) >= 12 for i in fg.tolist())


@pytest.mark.feature
def test_noisy_keys_transition_is_optional():
    router = FramePromptRouter(canonical_config(), patch_height=2, patch_width=2)
    block = router.active_block(3, 0, "cond")
    base = router.noisy_key_indices(block, 1, None, torch.device("cpu"), timestep_idx=3, include_transition=False)
    with_trans = router.noisy_key_indices(block, 1, None, torch.device("cpu"), timestep_idx=3, include_transition=True)
    assert int(with_trans.numel()) >= int(base.numel())


def test_transition_keeps_per_frame_fg_and_splits_query_rows():
    cfg = canonical_config()
    cfg["num_frames"] = 6
    cfg["frame_splits"] = [3]
    cfg["attention_control_schedule"][0]["self_attention_mask"] = [
        [[1, "full"], [0, "full"]],
        {
            "fg": [[1, "fg"], [1, "full"]],
            "bg": [[1, "bg"], [1, "full"]],
        },
    ]
    router = FramePromptRouter(cfg, patch_height=2, patch_width=2)
    # Spatial 4. Frame 0 FG is token 0, frame 1 FG is token 1, frame 2 FG is token 2.
    live = torch.ones(6, 4, dtype=torch.bool)
    live[0, 0] = False
    live[1, 1] = False
    live[2, 2] = False
    router.live_bg = live
    block = router.active_block(3, 0, "cond")
    fg = router.noisy_key_indices(
        block, 1, "fg", torch.device("cpu"), timestep_idx=3, include_transition=True
    )
    bg = router.noisy_key_indices(
        block, 1, "bg", torch.device("cpu"), timestep_idx=3, include_transition=True
    )
    # Context frame 2 is full. Frames 0 and 1 keep their own FG tokens, not a copied set.
    assert 0 in fg.tolist() and 5 in fg.tolist()
    assert 4 not in fg.tolist()
    assert set(range(8, 12)).issubset(set(fg.tolist()))
    assert set(fg.tolist()).isdisjoint({1, 2, 3, 6, 7})
    # BG row keeps BG tokens on frames 0 and 1, and the same full context frame.
    assert 0 not in bg.tolist() and 5 not in bg.tolist()
    assert {1, 2, 3, 4, 6, 7}.issubset(set(bg.tolist()))
    assert set(range(8, 12)).issubset(set(bg.tolist()))
