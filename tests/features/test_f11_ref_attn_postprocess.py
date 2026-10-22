import pytest
import torch

from sceneshifter.ref_attn import (
    derive_live_fg_from_class_scores,
    derive_live_fg_from_class_scores_global,
    derive_live_fg_from_class_scores_numpy,
    derive_live_fg_from_class_scores_s2v,
    derive_live_fg_from_union_scores,
    get_attn_map_with_target,
    get_attn_map_with_target_animate,
    get_attn_map_with_target_chunked,
    get_attn_map_with_target_per_head,
    get_attn_map_with_target_per_head_chunked,
    get_attn_map_with_target_per_head_split,
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
def test_animate_scores_scale_bfloat16_logits_after_qk():
    torch.manual_seed(29)
    q = torch.randn(1, 7, 1, 6, dtype=torch.bfloat16)
    k = torch.randn(1, 3, 1, 6, dtype=torch.bfloat16)
    targets = torch.tensor(
        [[1, 0, 0], [0, 1, 1]], dtype=torch.bfloat16
    )
    actual = get_attn_map_with_target_animate(q, k, (1, 1, 3), targets)

    logits = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-2, -1)
    probs = (logits * (q.shape[-1] ** -0.5)).softmax(-1)
    expected = torch.cat(
        [
            (
                probs[..., target.bool()].sum(-1)
                / target.bool().sum()
            ).permute(0, 2, 1)
            for target in targets
        ],
        dim=0,
    )
    legacy_probs = (
        (q * (q.shape[-1] ** -0.5)).transpose(1, 2)
        @ k.transpose(1, 2).transpose(-2, -1)
    ).softmax(-1)
    legacy = torch.cat(
        [
            (
                legacy_probs[..., target.bool()].sum(-1)
                / target.bool().sum()
            ).permute(0, 2, 1)
            for target in targets
        ],
        dim=0,
    )

    assert actual is not None
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, legacy)
    assert logits.dtype == torch.bfloat16


@pytest.mark.feature
def test_chunked_ref_scores_match_full_fp32_scores():
    torch.manual_seed(7)
    q = torch.randn(1, 12, 2, 4)
    k = torch.randn(1, 12, 2, 4)
    targets = torch.tensor([[1.0, 0, 1, 0], [0.0, 1, 0, 1]])
    full = get_attn_map_with_target(q, k, (3, 2, 2), targets)
    chunked = get_attn_map_with_target_chunked(
        q, k, (3, 2, 2), targets, chunk_size=3
    )
    assert full is not None and chunked is not None
    assert torch.allclose(chunked, full, atol=1e-6, rtol=1e-6)


@pytest.mark.feature
def test_chunked_per_head_scores_match_full_per_head():
    torch.manual_seed(11)
    q = torch.randn(1, 12, 3, 4)
    k = torch.randn(1, 12, 3, 4)
    targets = torch.tensor([[1.0, 0, 1, 0], [0.0, 1, 0, 1]])
    full = get_attn_map_with_target_per_head(q, k, (3, 2, 2), targets)
    chunked = get_attn_map_with_target_per_head_chunked(
        q, k, (3, 2, 2), targets, chunk_size=3
    )
    assert full is not None and chunked is not None
    assert torch.allclose(chunked, full, atol=1e-6, rtol=1e-6)


@pytest.mark.feature
def test_chunked_masked_key_reduction_matches_wan_native_order():
    torch.manual_seed(13)
    q = torch.randn(1, 8, 2, 4)
    k = torch.randn(1, 8, 2, 4)
    targets = torch.tensor([[1.0, 0, 1, 0], [0.0, 1, 0, 1]])
    actual = get_attn_map_with_target_per_head_chunked(
        q,
        k,
        (2, 2, 2),
        targets,
        chunk_size=8,
        select_masked_keys=True,
    )
    logits = torch.einsum("bqhd,bkhd->bhqk", q.float(), k[:, :4].float())
    probs = (logits * (4**-0.5)).softmax(-1)
    expected = []
    for target in targets:
        selected = target.bool()
        per_head = probs[..., selected].sum(-1) / selected.sum()
        expected.append(per_head[0].permute(1, 0))
    assert actual is not None
    assert torch.equal(actual, torch.stack(expected))


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
def test_s2v_confidence_drop_uses_one_subject_threshold_across_frames():
    fg = torch.tensor(
        [
            [
                [0.4963, 0.7682, 0.0885, 0.1320],
                [0.3074, 0.6341, 0.4901, 0.8964],
            ]
        ]
    )
    bg = torch.tensor(
        [
            [0.4556, 0.6323, 0.3489, 0.4017],
            [0.0223, 0.1689, 0.2939, 0.5185],
        ]
    )
    mask = derive_live_fg_from_class_scores_s2v(
        fg,
        bg,
        2,
        2,
        drop_low_percent=50.0,
        min_component_size=0,
        hole_size=0,
    )
    assert mask is not None
    assert mask.tolist() == [
        [
            [False, False, False, False],
            [True, True, False, True],
        ]
    ]


@pytest.mark.feature
def test_animate_drop_percentile_is_subject_global_across_frames():
    fg = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [100.0, 101.0, 102.0, 103.0]]]
    )
    bg = torch.zeros(2, 4)
    global_mask = derive_live_fg_from_class_scores_global(
        fg,
        bg,
        1,
        4,
        drop_low_percent=25.0,
        min_component_size=1,
        hole_size=0,
    )
    per_frame_mask = derive_live_fg_from_class_scores(
        fg,
        bg,
        1,
        4,
        drop_low_percent=25.0,
        min_component_size=1,
        hole_size=0,
    )

    assert global_mask is not None and per_frame_mask is not None
    assert global_mask[0].tolist() == [
        [False, False, True, True],
        [True, True, True, True],
    ]
    assert per_frame_mask[0].tolist() == [
        [False, True, True, True],
        [False, True, True, True],
    ]


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


@pytest.mark.feature
def test_multitalk_numpy_class_mask_is_mutually_exclusive():
    fg = torch.tensor(
        [[[0.9, 0.2, 0.1, 0.1]], [[0.1, 0.8, 0.2, 0.1]]],
        dtype=torch.float32,
    )
    bg = torch.tensor([[0.1, 0.1, 0.7, 0.9]], dtype=torch.float32)
    mask = derive_live_fg_from_class_scores_numpy(
        fg,
        bg,
        2,
        2,
        drop_low_percent=0.0,
        min_component_size=1,
        hole_size=0,
    )
    assert mask is not None
    assert mask[:, 0].tolist() == [
        [True, False, False, False],
        [False, True, False, False],
    ]


def test_reference_head_chunks_isolate_groups():
    torch.manual_seed(0)
    q = torch.randn(1, 8, 4, 4, dtype=torch.bfloat16)
    k = torch.randn(1, 8, 4, 4, dtype=torch.bfloat16)
    targets = torch.tensor([[1, 0, 0, 0], [0, 1, 1, 1]], dtype=torch.bfloat16)
    shape = (2, 2, 2)
    full = get_attn_map_with_target_per_head_split(
        q, k, shape, targets, split_num=2
    )
    q_other = q.clone()
    q_other[:, :, 2:] = torch.randn_like(q_other[:, :, 2:])
    other = get_attn_map_with_target_per_head_split(
        q_other, k, shape, targets, split_num=2
    )
    assert full.shape == (2, 8, 4)
    assert torch.equal(full[:, :, :2], other[:, :, :2])
    assert not torch.equal(full[:, :, 2:], other[:, :, 2:])
