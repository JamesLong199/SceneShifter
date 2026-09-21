import os

import pytest

from sceneshifter.checks import SceneShifterCheckError, official_tree_is_clean, validate_config
from tests.conftest import canonical_config


@pytest.mark.strict
def test_deprecated_transition_indices():
    cfg = canonical_config(transition_segment_indices=[1])
    with pytest.raises(SceneShifterCheckError):
        validate_config(cfg)


@pytest.mark.strict
def test_s2v_ext_size():
    cfg = canonical_config()
    cfg["attention_control_schedule"][0]["params"]["ref_attn_mask_ext_size"] = 2
    with pytest.raises(SceneShifterCheckError):
        validate_config(cfg, backend="wan_s2v")


@pytest.mark.strict
def test_cover_head_only_requires_head_mask_path():
    cfg = canonical_config()
    cfg["input_mask_cover_head_only"] = True
    with pytest.raises(SceneShifterCheckError):
        validate_config(cfg)


@pytest.mark.strict
def test_prompt_count():
    cfg = canonical_config(segment_prompts=["only-one"])
    with pytest.raises(SceneShifterCheckError):
        validate_config(cfg)


@pytest.mark.strict
def test_official_tree_clean(tmp_path):
    repo = tmp_path / "official"
    repo.mkdir()
    os.system(f"git -C {repo} init -b main >/dev/null")
    (repo / "README").write_text("ok\n")
    os.system(f"git -C {repo} add README && git -C {repo} -c user.email=t@t -c user.name=t commit -m init >/dev/null")
    official_tree_is_clean(str(repo))
    (repo / "dirty.txt").write_text("nope")
    with pytest.raises(SceneShifterCheckError):
        official_tree_is_clean(str(repo))
