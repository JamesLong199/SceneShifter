import pytest

from sceneshifter.masks import parse_mask_entry
from sceneshifter.policy import attention_row, normalize_schedule


@pytest.mark.feature
def test_legacy_and_spatial_cells():
    assert parse_mask_entry(1) == (1, "full")
    assert parse_mask_entry([1, "fg"]) == (1, "fg")
    assert parse_mask_entry([0, "bg"]) == (0, "bg")
    with pytest.raises(ValueError):
        parse_mask_entry([1, "left"])


@pytest.mark.feature
def test_split_fg_bg_rows():
    mask = [
        [[1, "full"], [0, "full"]],
        {"fg": [[1, "fg"], [1, "full"]], "bg": [[1, "bg"], [1, "full"]]},
    ]
    blocks = normalize_schedule(
        [{"timestep_range": [0, 1], "self_attention_mask": mask, "params": {}}],
        num_segments=2,
    )
    assert attention_row(blocks[0], 1, "fg")[0] == (1, "fg")
    assert attention_row(blocks[0], 1, "bg")[0] == (1, "bg")
    with pytest.raises(ValueError):
        attention_row(blocks[0], 1, None)
