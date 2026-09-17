import os
import subprocess
from pathlib import Path

import pytest

from sceneshifter.checks import official_tree_is_clean
from sceneshifter.hooks import inject_router
from sceneshifter.router import FramePromptRouter
from tests.conftest import TinyDiT, canonical_config


def _git(repo: Path, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.strict
def test_inject_leaves_official_git_clean(tmp_path):
    repo = tmp_path / "third_party_model"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    (repo / "model.py").write_text("print('official')\n")
    _git(repo, "add", "model.py")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "init")

    model = TinyDiT()
    router = FramePromptRouter(canonical_config(), patch_height=2, patch_width=2)
    inject_router(model, router, adapter="multitalk", official_root=str(repo))
    official_tree_is_clean(str(repo))
    assert (repo / "model.py").read_text() == "print('official')\n"
    status = _git(repo, "status", "--porcelain").stdout
    assert status.strip() == ""
