import pytest

from sceneshifter.checks import SceneShifterCheckError, validate_config
from sceneshifter.router import FramePromptRouter
from tests.conftest import canonical_config, canonical_mask


@pytest.mark.feature
def test_timestep_and_layer_gating():
    cfg = canonical_config()
    cfg["attention_control_schedule"] = [
        {
            "timestep_range": [0, 5],
            "dit_layer_indices": [0],
            "self_attention_mask": canonical_mask(),
            "params": {"auto_q_scale": False},
        }
    ]
    router = FramePromptRouter(cfg)
    assert router.active_block(3, 0, "cond") is not None
    assert router.active_block(3, 1, "cond") is None
    assert router.active_block(6, 0, "cond") is None


@pytest.mark.feature
def test_last_match_wins():
    cfg = canonical_config()
    first = {
        "timestep_range": [0, 10],
        "dit_layer_indices": [0],
        "self_attention_mask": canonical_mask(),
        "params": {"auto_q_scale": False, "tag": "a"},
    }
    second = {
        "timestep_range": [11, 20],
        "dit_layer_indices": [1],
        "self_attention_mask": canonical_mask(),
        "params": {"auto_q_scale": False, "tag": "b"},
    }
    cfg["attention_control_schedule"] = [first, second]
    router = FramePromptRouter(cfg)
    assert router.active_block(12, 1, "cond")["params"]["tag"] == "b"


@pytest.mark.feature
def test_overlap_rejected():
    cfg = canonical_config()
    cfg["attention_control_schedule"] = [
        {
            "timestep_range": [0, 10],
            "dit_layer_indices": None,
            "self_attention_mask": canonical_mask(),
            "params": {"auto_q_scale": True, "q_scale_bg_only": False},
        },
        {
            "timestep_range": [5, 15],
            "dit_layer_indices": None,
            "self_attention_mask": canonical_mask(),
            "params": {"auto_q_scale": True, "q_scale_bg_only": False},
        },
    ]
    with pytest.raises(SceneShifterCheckError):
        validate_config(cfg)
