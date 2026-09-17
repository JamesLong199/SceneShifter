import pytest
import torch

from sceneshifter.ref_attn import live_bg_from_subject_fg, postprocess_live_fg


@pytest.mark.feature
def test_synthetic_scores_to_binary():
    scores = torch.zeros(2, 4, 4)
    scores[:, 1:3, 1:3] = 1.0
    fg = postprocess_live_fg(scores, drop_low_percent=0, min_component_size=1, hole_size=0)
    assert fg.shape == (2, 16)
    subject = fg.unsqueeze(0)
    bg = live_bg_from_subject_fg(subject)
    assert torch.equal(bg, ~fg)
    assert bool(bg[0].sum() + fg[0].sum() == 16)


@pytest.mark.feature
def test_small_component_drop():
    scores = torch.zeros(1, 8, 8)
    scores[0, 0, 0] = 1.0
    fg = postprocess_live_fg(scores, drop_low_percent=0, min_component_size=4, hole_size=0)
    assert int(fg.sum()) == 0
