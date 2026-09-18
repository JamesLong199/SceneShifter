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
