import pytest
import torch

from sceneshifter.ref_attn import (
    derive_live_fg_from_class_scores,
    derive_live_fg_from_union_scores,
    get_attn_map_with_target,
    live_bg_from_subject_fg,
    postprocess_live_fg,
)


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


@pytest.mark.feature
def test_get_attn_map_class_mean_matches_manual():
    q = torch.zeros(1, 4, 1, 4)
    k = torch.zeros(1, 2, 1, 4)
    q[0, 0, 0, 0] = 1.0
    k[0, 0, 0, 0] = 1.0
    k[0, 1, 0, 1] = 1.0
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    scores = get_attn_map_with_target(q, k, (2, 1, 2), targets)
    assert scores is not None
    assert scores.shape[0] == 2
    # Query 0 is aligned with key 0 (class 0 / FG).
    assert float(scores[0, 0]) > float(scores[1, 0])


@pytest.mark.feature
def test_argmax_live_mask_uses_bg_class():
    fg = torch.zeros(1, 1, 4)
    bg = torch.zeros(1, 4)
    fg[0, 0, 0] = 0.9
    bg[0, 1:] = 0.8
    mask = derive_live_fg_from_class_scores(
        fg, bg, 2, 2, drop_low_percent=0.0, min_component_size=1, hole_size=0
    )
    assert mask is not None
    assert bool(mask[0, 0, 0])
    assert not bool(mask[0, 0, 1])


@pytest.mark.feature
def test_fantasy_union_fg_uses_strict_fp32_diff_threshold():
    fg = torch.tensor([[0.9, 0.5, 0.6, 0.1]], dtype=torch.float32)
    bg = torch.tensor([[0.1, 0.5, 0.2, 0.8]], dtype=torch.float32)
    mask = derive_live_fg_from_union_scores(
        fg,
        bg,
        2,
        2,
        drop_low_percent=0.0,
        min_component_size=1,
        hole_size=0,
    )
    assert mask is not None
    assert mask.shape == (1, 1, 4)
    assert mask[0, 0].tolist() == [True, False, True, False]
