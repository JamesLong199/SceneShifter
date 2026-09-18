"""Shared paths and metadata helpers for parity jobs."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


PARITY_ROOT_DEFAULT = (
    "/data/home/jzlong/data/SceneShifter_dataset/results/sceneshifter_unified_parity_0918"
)
DATASET_ROOT_DEFAULT = "/data/home/jzlong/data/SceneShifter_dataset"
CONDA_SH = "/data/home/jzlong/miniconda3/etc/profile.d/conda.sh"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_parity_config(backbone: str) -> Dict[str, Any]:
    path = repo_root() / "configs" / "parity" / f"{backbone}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing parity config {path}")
    return json.loads(path.read_text())


def output_root(override: Optional[str] = None) -> Path:
    return Path(override or os.environ.get("SCENESHIFTER_PARITY_ROOT") or PARITY_ROOT_DEFAULT)


def side_dir(backbone: str, side: str, *, root: Optional[str] = None) -> Path:
    path = output_root(root) / backbone / side
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_metadata(cfg: Dict[str, Any]) -> Path:
    meta = Path(cfg["metadata"])
    if meta.is_absolute():
        return meta
    if str(meta).startswith("configs/"):
        return repo_root() / meta
    return Path(cfg.get("dataset_root") or DATASET_ROOT_DEFAULT) / meta


def git_head(repo: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "HEAD"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def write_run_args(out_dir: Path, payload: Dict[str, Any]) -> None:
    (out_dir / "run_args.json").write_text(json.dumps(payload, indent=2, default=str))


def conda_python(env_name: str) -> str:
    candidate = Path("/data/home/jzlong/miniconda3/envs") / env_name / "bin" / "python"
    if candidate.is_file():
        return str(candidate)
    return "python"


def find_videos(directory: Path, glob_pat: str) -> List[Path]:
    hits = sorted(
        p
        for p in directory.rglob("*")
        if p.suffix.lower() == ".mp4" and "tmp_" not in p.name
    )
    if glob_pat and glob_pat != "*.mp4":
        named = list(directory.glob(glob_pat)) + list((directory / "videos").glob(glob_pat) if (directory / "videos").exists() else [])
        if named:
            return sorted({p.resolve() for p in named if p.suffix.lower() == ".mp4"})
    videos = [p for p in hits if (directory / "videos") in p.parents or p.parent == directory]
    return videos or hits


def overlay_routing(routing: Dict[str, Any], unified: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(routing)
    for key in ("len_trans", "len_trans_context", "trans_timestep_range"):
        if unified.get(key) is not None:
            out[key] = unified[key]
    return out
