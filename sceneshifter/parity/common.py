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


_DEMO_PATH_KEYS = {"src_root_path", "control_camera_txt"}


def default_unified_weights(backbone: str) -> Dict[str, str]:
    """Checkpoint paths created by scripts/download_checkpoints.py."""
    root = repo_root() / "weights" / backbone
    if backbone == "multitalk":
        return {
            "ckpt_dir": str(root / "Wan2.1-I2V-14B-480P"),
            "wav2vec_dir": str(root / "chinese-wav2vec2-base"),
        }
    if backbone == "infinitetalk":
        return {
            "ckpt_dir": str(root / "Wan2.1-I2V-14B-480P"),
            "wav2vec_dir": str(root / "chinese-wav2vec2-base"),
            "infinitetalk_dir": str(root / "InfiniteTalk" / "multi" / "infinitetalk.safetensors"),
        }
    if backbone == "fantasytalking":
        return {
            "wan_model_dir": str(root / "Wan2.1-I2V-14B-720P"),
            "fantasytalking_model_path": str(root / "fantasytalking_model.ckpt"),
            "wav2vec_model_dir": str(root / "wav2vec2-base-960h"),
        }
    if backbone == "wan_s2v":
        return {"ckpt_dir": str(root / "Wan2.2-S2V-14B")}
    if backbone == "wan_animate":
        return {"ckpt_dir": str(root / "Wan2.2-Animate-14B")}
    if backbone == "wan_fun_camera":
        return {"ckpt": str(root / "Wan2.1-Fun-V1.1-14B-Control-Camera")}
    if backbone == "hunyuan_avatar":
        return {
            "model_base": str(root),
            "ckpt": str(
                root
                / "ckpts"
                / "hunyuan-video-t2v-720p"
                / "transformers"
                / "mp_rank_00_model_states_fp8.pt"
            ),
        }
    raise KeyError(f"no default weights for {backbone}")


def apply_demo_config(cfg: Dict[str, Any], demo: str | Path) -> Dict[str, Any]:
    """Point a parity config at the shared examples/demos tree and weights/<backbone>."""
    demo_dir = Path(demo)
    if not demo_dir.is_absolute():
        demo_dir = repo_root() / demo_dir
    demo_dir = demo_dir.resolve()
    backbone = str(cfg["backbone"])
    metadata = demo_dir / "configs" / f"{backbone}.json"
    if not metadata.is_file():
        metadata = demo_dir / "metadata.json"
    if not metadata.is_file():
        raise FileNotFoundError(f"demo metadata missing: {metadata}")
    updated = dict(cfg)
    updated["dataset_root"] = str(demo_dir)
    updated["metadata"] = str(metadata)
    unified = dict(updated.get("unified") or {})
    unified.update(default_unified_weights(backbone))
    demo_cfg = demo_dir / "configs" / f"{backbone}.demo.json"
    if not demo_cfg.is_file():
        demo_cfg = demo_dir / "demo.json"
    if demo_cfg.is_file():
        extra = json.loads(demo_cfg.read_text())
        if extra.get("sample_id"):
            updated["sample_id"] = str(extra["sample_id"])
        for key, value in (extra.get("unified") or {}).items():
            if (
                key in _DEMO_PATH_KEYS
                and isinstance(value, str)
                and value
                and not Path(value).is_absolute()
            ):
                value = str((demo_dir / value).resolve())
            unified[key] = value
    updated["unified"] = unified
    return updated


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
    for key in (
        "len_trans",
        "len_trans_context",
        "trans_timestep_range",
        "audio_start_timestep",
        "ref_loc_max_timesteps",
        "freeze_live_mask",
        "infer_frames",
        "input_mask_cover_original",
        "input_mask_cover_head_only",
    ):
        if unified.get(key) is not None:
            out[key] = unified[key]
    return out
