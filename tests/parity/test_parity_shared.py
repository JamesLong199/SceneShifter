"""Layer B: unified router vs locked goldens (and jameslong199 if present)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sceneshifter.router import FramePromptRouter
from tests.conftest import canonical_config

REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "reference"


@pytest.mark.parity
@pytest.mark.parametrize(
    "backend",
    [
        "multitalk",
        "infinitetalk",
        "fantasytalking",
        "hunyuan_avatar",
        "wan_s2v",
        "wan_animate",
        "wan_fun_camera",
    ],
)
def test_windows_and_prompts_match_contract(backend):
    cfg = canonical_config()
    if backend == "hunyuan_avatar":
        cfg["attention_control_schedule"][0]["params"]["apply_on_uncond_passes"] = True
        cfg["attention_control_schedule"][0]["params"]["q_scale_bg_only"] = False
    router = FramePromptRouter(cfg, backend=backend, patch_height=2, patch_width=2)
    assert router.get_transition_context_windows()[0]["query_frames"] == [3, 4]
    assert router.selected_text_index(0, 1, "uncond") is None
    if backend == "hunyuan_avatar":
        assert router.qscale_token_mask(router.active_block(0, 0, "cond"))[12:].all()


@pytest.mark.parity
def test_reference_checkout_optional():
    if not REFERENCE_ROOT.exists():
        pytest.skip("tests/reference not cloned; run setup_backbones.sh --with-reference")
    assert any(REFERENCE_ROOT.iterdir())


@pytest.mark.parity
def test_every_parity_config_locks_both_revisions():
    config_root = Path(__file__).resolve().parents[2] / "configs" / "parity"
    configs = [json.loads(path.read_text()) for path in sorted(config_root.glob("*.json"))]
    assert len(configs) == 7
    for cfg in configs:
        locks = cfg.get("locks") or {}
        assert len(str(locks.get("reference_feature_sha") or "")) == 40, cfg["backbone"]
        assert len(str(locks.get("reference_base_sha") or "")) == 40, cfg["backbone"]
        assert len(str(locks.get("official_sha") or "")) == 40, cfg["backbone"]


def test_parity_launcher_exposes_current_env_media_tools(monkeypatch):
    import os
    import sys

    from scripts.parity_generate import _prepend_python_bin_to_path

    monkeypatch.setenv("PATH", "/usr/bin")
    expected = str(Path(sys.executable).resolve().parent)
    assert _prepend_python_bin_to_path() == expected
    assert os.environ["PATH"].split(os.pathsep)[0] == expected
