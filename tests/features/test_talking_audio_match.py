import torch

from sceneshifter.adapters.talking_audio import (
    accumulate_reference_ref_map,
    multitalk_audio_key_positions,
    multitalk_audio_query_positions,
    reference_audio_map_weight,
)


class _Rope:
    class_interval = 4
    class_range = 24
    rope_bak = 12


def test_audio_map_weight_is_the_query_length():
    chunk = torch.zeros(1, 32256, 20, 8)
    assert reference_audio_map_weight(chunk) == 32256


def test_query_length_bf16_rescale_differs_from_head_count():
    torch.manual_seed(0)
    first = torch.randn(3, 8, dtype=torch.bfloat16)
    second = torch.randn(3, 8, dtype=torch.bfloat16)
    by_tokens = accumulate_reference_ref_map([first, second], [32256, 32256])
    by_heads = accumulate_reference_ref_map([first, second], [20, 20])
    official = (first + second) / 2
    assert not torch.equal(by_tokens, by_heads)
    assert not torch.equal(by_tokens, official)


def test_reference_ref_map_reduction_matches_bf16_rescale():
    torch.manual_seed(0)
    first = torch.randn(3, 5, dtype=torch.bfloat16)
    second = torch.randn(3, 5, dtype=torch.bfloat16)
    reduced = accumulate_reference_ref_map([first, second], [20, 20])
    official = (first + second) / 2
    manual = torch.zeros_like(first)
    manual += first * 20.0
    manual += second * 20.0
    manual = manual / 40.0
    assert torch.equal(reduced, manual)
    assert not torch.equal(reduced, official)


def test_multitalk_audio_queries_snap_to_subject_centers():
    module = _Rope()
    # Rows are person 0, person 1, background. Argmax is background, person 0, person 1, background.
    ref_map = torch.tensor(
        [
            [0.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 9.0, 0.0],
            [1.0, 0.0, 0.0, 3.0],
        ],
        dtype=torch.bfloat16,
    )
    positions = multitalk_audio_query_positions(
        module,
        n_t=1,
        spatial=4,
        human_num=2,
        x_ref_attn_map=ref_map,
        subject_token_assignments=None,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert torch.equal(positions, torch.tensor([12.0, 2.0, 22.0, 12.0], dtype=torch.bfloat16))


def test_multitalk_audio_assignments_override_the_score_map():
    module = _Rope()
    positions = multitalk_audio_query_positions(
        module,
        n_t=1,
        spatial=4,
        human_num=2,
        x_ref_attn_map=None,
        subject_token_assignments=torch.tensor([1, 0, 2, 1]),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.equal(positions, torch.tensor([22.0, 2.0, 12.0, 22.0]))


def test_multitalk_audio_keys_chunk_by_speaker():
    module = _Rope()
    keys = multitalk_audio_key_positions(
        module,
        n_audio=4,
        n_t=2,
        human_num=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.equal(keys, torch.tensor([2.0, 2.0, 22.0, 22.0, 2.0, 2.0, 22.0, 22.0]))
