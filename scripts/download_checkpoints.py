#!/usr/bin/env python3
"""Download the checkpoints required to run one SceneShifter backbone."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BACKBONES = [
    "multitalk",
    "infinitetalk",
    "fantasytalking",
    "wan_s2v",
    "wan_animate",
    "wan_fun_camera",
    "hunyuan_avatar",
]


def _snapshot(
    repo_id: str,
    dest: Path,
    *,
    allow: list[str] | None = None,
    ignore: list[str] | None = None,
) -> None:
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    print(f"download {repo_id} -> {dest}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        allow_patterns=allow,
        ignore_patterns=ignore,
    )


def _file(repo_id: str, filename: str, dest: Path, revision: str) -> None:
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    print(f"download {repo_id}:{filename} ({revision}) -> {dest}", flush=True)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=str(dest),
    )


def _link_multitalk(root: Path) -> None:
    wan = root / "Wan2.1-I2V-14B-480P"
    talk = root / "MeiGen-MultiTalk"
    index = wan / "diffusion_pytorch_model.safetensors.index.json"
    backup = wan / "diffusion_pytorch_model.safetensors.index.json_old"
    if index.exists() and not index.is_symlink() and not backup.exists():
        index.rename(backup)
    for name in (
        "diffusion_pytorch_model.safetensors.index.json",
        "multitalk.safetensors",
    ):
        source = talk / name
        target = wan / name
        if not source.is_file():
            raise FileNotFoundError(f"MultiTalk weight missing: {source}")
        if target.is_symlink() or target.exists():
            target.unlink()
        try:
            target.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, target)
        print(f"linked {target.name} -> {source}", flush=True)


def _wav2vec(root: Path) -> None:
    dest = root / "chinese-wav2vec2-base"
    # The fairseq checkpoint and pytorch_model.bin duplicate model.safetensors.
    _snapshot(
        "TencentGameMate/chinese-wav2vec2-base",
        dest,
        ignore=[
            "chinese-wav2vec2-base-fairseq-ckpt.pt",
            "pytorch_model.bin",
        ],
    )
    _file(
        "TencentGameMate/chinese-wav2vec2-base",
        "model.safetensors",
        dest,
        "refs/pr/1",
    )


def download_backbone(backbone: str) -> Path:
    root = ROOT / "weights" / backbone
    root.mkdir(parents=True, exist_ok=True)
    if backbone == "multitalk":
        _snapshot("Wan-AI/Wan2.1-I2V-14B-480P", root / "Wan2.1-I2V-14B-480P")
        _wav2vec(root)
        _snapshot(
            "MeiGen-AI/MeiGen-MultiTalk",
            root / "MeiGen-MultiTalk",
            allow=[
                "multitalk.safetensors",
                "diffusion_pytorch_model.safetensors.index.json",
            ],
        )
        _link_multitalk(root)
    elif backbone == "infinitetalk":
        _snapshot("Wan-AI/Wan2.1-I2V-14B-480P", root / "Wan2.1-I2V-14B-480P")
        _wav2vec(root)
        _snapshot(
            "MeiGen-AI/InfiniteTalk",
            root / "InfiniteTalk",
            allow=["multi/infinitetalk.safetensors"],
        )
    elif backbone == "fantasytalking":
        _snapshot("Wan-AI/Wan2.1-I2V-14B-720P", root / "Wan2.1-I2V-14B-720P")
        _snapshot(
            "facebook/wav2vec2-base-960h",
            root / "wav2vec2-base-960h",
            ignore=["pytorch_model.bin", "tf_model.h5", "flax_model.msgpack"],
        )
        _file("acvlab/FantasyTalking", "fantasytalking_model.ckpt", root, "main")
    elif backbone == "wan_s2v":
        _snapshot(
            "Wan-AI/Wan2.2-S2V-14B",
            root / "Wan2.2-S2V-14B",
            ignore=[
                "wav2vec2-large-xlsr-53-english/pytorch_model.bin",
                "wav2vec2-large-xlsr-53-english/flax_model.msgpack",
                "wav2vec2-large-xlsr-53-english/language_model/*",
            ],
        )
    elif backbone == "wan_animate":
        _snapshot(
            "Wan-AI/Wan2.2-Animate-14B",
            root / "Wan2.2-Animate-14B",
            ignore=[
                "relighting_lora.ckpt",
                "relighting_lora/*",
                "process_checkpoint/*",
                "xlm-roberta-large/model.safetensors",
                "xlm-roberta-large/pytorch_model.bin",
                "xlm-roberta-large/flax_model.msgpack",
                "xlm-roberta-large/tf_model.h5",
                "xlm-roberta-large/onnx/*",
            ],
        )
    elif backbone == "wan_fun_camera":
        _snapshot(
            "alibaba-pai/Wan2.1-Fun-V1.1-14B-Control-Camera",
            root / "Wan2.1-Fun-V1.1-14B-Control-Camera",
        )
    elif backbone == "hunyuan_avatar":
        _snapshot(
            "tencent/HunyuanVideo-Avatar",
            root,
            ignore=[
                # The demo loads the fp8 transformer, not this bf16 copy.
                "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
                "ckpts/text_encoder_2/pytorch_model.bin",
                "ckpts/text_encoder_2/flax_model.msgpack",
                "ckpts/text_encoder_2/tf_model.h5",
                "ckpts/whisper-tiny/pytorch_model.bin",
                "ckpts/whisper-tiny/flax_model.msgpack",
                "ckpts/whisper-tiny/tf_model.h5",
            ],
        )
    else:
        raise SystemExit(f"unknown backbone {backbone}")
    print(f"weights ready: {root}", flush=True)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description="Download SceneShifter backbone weights")
    parser.add_argument(
        "--backbone",
        required=True,
        choices=[*BACKBONES, "all"],
        help="Backbone to download. 'all' downloads every backbone into its own weights directory.",
    )
    args = parser.parse_args()
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        raise SystemExit(
            'huggingface_hub is not installed. Run: python -m pip install "huggingface_hub"'
        )
    selected = BACKBONES if args.backbone == "all" else [args.backbone]
    for backbone in selected:
        download_backbone(backbone)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
