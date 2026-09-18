"""Run official-main generate with SceneShifter inject_router."""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import overlay_routing, repo_root, resolve_metadata


def resolve_official_root(cfg: Dict[str, Any]) -> Path:
    raw = (cfg.get("unified") or {}).get("official_root") or ""
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root() / path
    return path


def _push_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _load_routing(cfg: Dict[str, Any]) -> Dict[str, Any]:
    from sceneshifter.config import load_frame_prompt_config

    meta_path = resolve_metadata(cfg)
    asset_root = Path(cfg.get("dataset_root") or meta_path.parent)
    routing = load_frame_prompt_config(
        str(meta_path),
        backend=cfg["backbone"],
        asset_root=str(asset_root),
        validate=False,
    )
    return overlay_routing(routing, cfg.get("unified") or {})


def _resolve_media(dataset_root: Path, value: Any) -> Any:
    if isinstance(value, str):
        path = Path(value)
        return str(path if path.is_absolute() else (dataset_root / path).resolve())
    if isinstance(value, dict):
        return {k: _resolve_media(dataset_root, v) for k, v in value.items()}
    return value


def _metadata(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(resolve_metadata(cfg).read_text())


def _write_input_json(cfg: Dict[str, Any], out_dir: Path) -> Path:
    meta = _metadata(cfg)
    root = Path(cfg.get("dataset_root"))
    routing = meta.get("frame_prompt_routing") or {}
    prompt = meta.get("prompt") or (routing.get("segment_prompts") or ["a person talking"])[0]
    image = _resolve_media(root, meta.get("cond_image"))
    payload = {
        "prompt": prompt,
        "cond_image": image,
        "cond_video": image,
        "cond_audio": _resolve_media(root, meta.get("cond_audio")),
        "audio_type": meta.get("audio_type", "para"),
    }
    path = out_dir / "input.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _create_router(cfg: Dict[str, Any], routing: Dict[str, Any], patch_h: int = 30, patch_w: int = 52):
    from sceneshifter.router import create_router

    return create_router(
        routing,
        backend=cfg["backbone"],
        patch_height=patch_h,
        patch_width=patch_w,
        validate=False,
        len_trans=routing.get("len_trans"),
        len_trans_context=routing.get("len_trans_context"),
        trans_timestep_range=routing.get("trans_timestep_range"),
    )


def _inject(model: Any, router: Any, adapter: str) -> List[Any]:
    from sceneshifter.hooks import inject_router

    targets = [model]
    for attr in ("model", "dit", "transformer", "pipe", "net", "pipeline"):
        inner = getattr(model, attr, None)
        if inner is not None:
            targets.append(inner)
    last_err: Optional[Exception] = None
    for target in targets:
        try:
            bound = inject_router(target, router, adapter=adapter, expected_min=1)
            print(f"[parity-unified] injected {len(bound)} modules on {type(target).__name__}", flush=True)
            return bound
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"inject_router failed for adapter={adapter}: {last_err}")


def _maybe_set_text_embeddings(router: Any, pipeline: Any, routing: Dict[str, Any]) -> None:
    prompts = routing.get("segment_prompts") or []
    if not prompts or not hasattr(router, "set_text_embeddings"):
        return
    try:
        embeds = []
        if hasattr(pipeline, "encode_prompt"):
            for prompt in prompts:
                out = pipeline.encode_prompt(prompt)
                embeds.append(out[0] if isinstance(out, (tuple, list)) else out)
        elif hasattr(pipeline, "text_encoder") and callable(pipeline.text_encoder):
            for prompt in prompts:
                embeds.append(pipeline.text_encoder([prompt])[0])
        if embeds:
            router.set_text_embeddings(embeds)
    except Exception as exc:  # noqa: BLE001
        print(f"[parity-unified] text embed skip: {exc}", flush=True)


def _wrap_named_classes(mod: Any, names: List[str], router: Any, adapter: str) -> None:
    for name in names:
        cls = getattr(mod, name, None)
        if cls is None or not inspect.isclass(cls):
            continue
        original_init = cls.__init__

        def make_init(orig):
            def wrapped(self, *args, **kwargs):
                orig(self, *args, **kwargs)
                target = getattr(self, "model", None) or getattr(self, "transformer", None) or self
                try:
                    _inject(target, router, adapter)
                    _maybe_set_text_embeddings(router, self, getattr(router, "config", {}))
                except Exception as exc:  # noqa: BLE001
                    print(f"[parity-unified] inject after {name}.__init__ failed: {exc}", flush=True)

            return wrapped

        cls.__init__ = make_init(original_init)
        print(f"[parity-unified] wrapped {mod.__name__}.{name}.__init__", flush=True)


def _wrap_create_pipeline(mod: Any, router: Any, adapter: str, routing: Dict[str, Any]) -> None:
    if not hasattr(mod, "create_pipeline"):
        return
    original = mod.create_pipeline

    def wrapped(*args, **kwargs):
        pipe = original(*args, **kwargs)
        _inject(pipe, router, adapter)
        _maybe_set_text_embeddings(router, pipe, routing)
        return pipe

    mod.create_pipeline = wrapped


def _filter_official_argv(mod: Any, argv: List[str]) -> List[str]:
    """Drop CLI flags the official parser does not declare."""
    source_files = []
    for candidate in (getattr(mod, "__file__", None),):
        if candidate and Path(candidate).is_file():
            source_files.append(Path(candidate).read_text(errors="ignore"))
    blob = "\n".join(source_files)
    filtered: List[str] = []
    skip_next = False
    i = 0
    while i < len(argv):
        tok = argv[i]
        if skip_next:
            skip_next = False
            i += 1
            continue
        if tok.startswith("--"):
            key = tok.split("=", 1)[0]
            if key not in blob:
                print(f"[parity-unified] drop unsupported official flag {tok}", flush=True)
                if "=" not in tok and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                    skip_next = True
                i += 1
                continue
        filtered.append(tok)
        i += 1
    return filtered


def _invoke_generate(mod: Any, script: str, argv: List[str]) -> None:
    argv = _filter_official_argv(mod, argv)
    old = sys.argv
    sys.argv = [script, *argv]
    try:
        if hasattr(mod, "parse_args") and hasattr(mod, "generate"):
            try:
                args = mod.parse_args(argv)
            except TypeError:
                args = mod.parse_args()
            pipeline = None
            if hasattr(mod, "create_pipeline"):
                try:
                    pipeline = None
                except Exception:
                    pipeline = None
            try:
                mod.generate(args, pipeline=pipeline) if pipeline is not None else mod.generate(args)
            except TypeError:
                mod.generate(args)
            return
        if hasattr(mod, "_parse_args") and hasattr(mod, "generate"):
            args = mod._parse_args()
            mod.generate(args)
            return
        if hasattr(mod, "main"):
            if inspect.signature(mod.main).parameters:
                try:
                    args = mod.parse_args() if hasattr(mod, "parse_args") else None
                    if args is not None:
                        return mod.main(args)
                except Exception:
                    pass
            return mod.main()
        raise RuntimeError(f"{script} has no generate/main entrypoint")
    finally:
        sys.argv = old


def run_talking_official(cfg: Dict[str, Any], out_dir: Path, official: Path) -> None:
    unified = cfg["unified"]
    input_json = _write_input_json(cfg, out_dir)
    routing = _load_routing(cfg)
    router = _create_router(cfg, routing)
    _push_path(official)
    os.chdir(official)
    script = unified.get("official_script", "generate_multitalk.py")
    if "infinitetalk" in script:
        import generate_infinitetalk as gm  # type: ignore
        adapter = "infinitetalk"
    else:
        import generate_multitalk as gm  # type: ignore
        adapter = "multitalk"
    _wrap_create_pipeline(gm, router, adapter, routing)
    try:
        import wan  # type: ignore

        _wrap_named_classes(
            wan,
            ["MultiTalkPipeline", "InfiniteTalkPipeline", "WanT2V", "WanI2V"],
            router,
            adapter,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[parity-unified] wan class wrap skip: {exc}", flush=True)

    save = out_dir / "videos" / f"{cfg['sample_id']}_sceneshifter"
    save.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        "--ckpt_dir", str(unified["ckpt_dir"]),
        "--wav2vec_dir", str(unified["wav2vec_dir"]),
        "--input_json", str(input_json),
        "--base_seed", str(unified.get("base_seed", 43)),
        "--save_file", str(save),
    ]
    if unified.get("infinitetalk_dir"):
        argv += ["--infinitetalk_dir", str(unified["infinitetalk_dir"])]
    if unified.get("size"):
        argv += ["--size", str(unified["size"])]
    if unified.get("mode"):
        argv += ["--mode", str(unified["mode"])]
    if unified.get("frame_num"):
        argv += ["--frame_num", str(unified["frame_num"])]
    if unified.get("audio_start_timestep") is not None:
        argv += ["--audio_start_timestep", str(unified["audio_start_timestep"])]
    _invoke_generate(gm, script, argv)


def run_fantasy_official(cfg: Dict[str, Any], out_dir: Path, official: Path) -> None:
    import librosa
    import torch
    from PIL import Image

    unified = cfg["unified"]
    routing = _load_routing(cfg)
    router = _create_router(cfg, routing)
    meta = _metadata(cfg)
    root = Path(cfg["dataset_root"])
    _push_path(official)
    os.chdir(official)
    from diffsynth import ModelManager, WanVideoPipeline  # type: ignore
    from model import FantasyTalkingAudioConditionModel  # type: ignore
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    from utils import get_audio_features, resize_image_by_longest_edge, save_video  # type: ignore

    wan_dir = unified["wan_model_dir"]
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            [f"{wan_dir}/diffusion_pytorch_model-0000{i}-of-00007.safetensors" for i in range(1, 8)],
            f"{wan_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            f"{wan_dir}/models_t5_umt5-xxl-enc-bf16.pth",
            f"{wan_dir}/Wan2.1_VAE.pth",
        ],
        torch_dtype=torch.bfloat16,
    )
    pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
    fantasytalking = FantasyTalkingAudioConditionModel(pipe.dit, 768, 2048).to("cuda")
    fantasytalking.load_audio_processor(unified["fantasytalking_model_path"], pipe.dit)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    wav2vec_processor = Wav2Vec2Processor.from_pretrained(unified["wav2vec_model_dir"])
    wav2vec = Wav2Vec2Model.from_pretrained(unified["wav2vec_model_dir"]).to("cuda")
    _inject(pipe.dit, router, "fantasytalking")
    _maybe_set_text_embeddings(router, pipe, routing)

    image_path = _resolve_media(root, meta.get("cond_image"))
    audio = meta.get("cond_audio")
    if isinstance(audio, dict):
        audio = next(iter(audio.values()))
    audio_path = _resolve_media(root, audio)
    prompt = meta.get("prompt") or routing.get("segment_prompts", [""])[0]
    duration = librosa.get_duration(filename=audio_path)
    fps = 23
    num_frames = min(int(fps * duration // 4) * 4 + 5, 81)
    audio_wav2vec_fea = get_audio_features(wav2vec, wav2vec_processor, audio_path, fps, num_frames)
    image = resize_image_by_longest_edge(image_path, None)
    width, height = image.size
    audio_proj_fea = fantasytalking.get_proj_fea(audio_wav2vec_fea)
    pos_idx_ranges = fantasytalking.split_audio_sequence(audio_proj_fea.size(1), num_frames=num_frames)
    audio_proj_split, audio_context_lens = fantasytalking.split_tensor_with_padding(
        audio_proj_fea, pos_idx_ranges, expand_length=4
    )
    video = pipe(
        prompt=prompt,
        negative_prompt="人物静止不动，静止，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=image,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=30,
        seed=int(unified.get("base_seed", 43)),
        tiled=True,
        audio_scale=1.0,
        audio_start_timestep=int(unified.get("audio_start_timestep", 15)),
        cfg_scale=5.0,
        audio_cfg_scale=5.0,
        audio_proj=audio_proj_split,
        audio_context_lens=audio_context_lens,
        latents_num_frames=(num_frames - 1) // 4 + 1,
    )
    save = out_dir / "videos" / f"{cfg['sample_id']}_sceneshifter.mp4"
    save.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "videos" / f"tmp_{cfg['sample_id']}.mp4"
    save_video(video, str(tmp), fps=fps, quality=5)
    os.system(
        f'ffmpeg -y -i "{tmp}" -i "{audio_path}" -c:v libx264 -c:a aac -shortest "{save}"'
    )
    tmp.unlink(missing_ok=True)
    _ = Image


def run_s2v_official(cfg: Dict[str, Any], out_dir: Path, official: Path) -> None:
    unified = cfg["unified"]
    meta = _metadata(cfg)
    root = Path(cfg["dataset_root"])
    routing = _load_routing(cfg)
    router = _create_router(cfg, routing)
    _push_path(official)
    os.chdir(official)
    import generate as gm  # type: ignore

    _wrap_create_pipeline(gm, router, "wan_s2v", routing)
    try:
        import wan  # type: ignore

        _wrap_named_classes(wan, ["WanS2V", "WanI2V", "WanT2V"], router, "wan_s2v")
    except Exception as exc:  # noqa: BLE001
        print(f"[parity-unified] wan s2v wrap skip: {exc}", flush=True)

    image = _resolve_media(root, meta.get("cond_image"))
    audio = meta.get("cond_audio")
    if isinstance(audio, dict):
        audio = next(iter(audio.values()))
    audio = _resolve_media(root, audio)
    prompt = meta.get("prompt") or routing.get("segment_prompts", [""])[0]
    save = out_dir / "videos" / f"{cfg['sample_id']}_sceneshifter.mp4"
    save.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        "--task", "s2v-14B",
        "--ckpt_dir", str(unified["ckpt_dir"]),
        "--image", str(image),
        "--audio", str(audio),
        "--prompt", str(prompt),
        "--base_seed", str(unified.get("base_seed", 43)),
        "--save_file", str(save),
        "--offload_model", "True",
        "--convert_model_dtype",
    ]
    _invoke_generate(gm, "generate.py", argv)


def run_animate_official(cfg: Dict[str, Any], out_dir: Path, official: Path) -> None:
    unified = cfg["unified"]
    routing = _load_routing(cfg)
    router = _create_router(cfg, routing, patch_h=45, patch_w=80)
    _push_path(official)
    os.chdir(official)
    import generate as gm  # type: ignore

    _wrap_create_pipeline(gm, router, "wan_animate", routing)
    try:
        import wan  # type: ignore

        _wrap_named_classes(wan, ["WanAnimate", "WanI2V", "WanT2V"], router, "wan_animate")
    except Exception as exc:  # noqa: BLE001
        print(f"[parity-unified] wan animate wrap skip: {exc}", flush=True)

    save = out_dir / "videos" / f"{cfg['sample_id']}_sceneshifter.mp4"
    save.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        "--task", "animate-14B",
        "--size", unified.get("size", "1280*720"),
        "--ckpt_dir", str(unified["ckpt_dir"]),
        "--src_root_path", str(unified["src_root_path"]),
        "--refert_num", "1",
        "--prompt", unified.get("prompt", ""),
        "--base_seed", str(unified.get("base_seed", 4)),
        "--sample_guide_scale", str(unified.get("sample_guide_scale", 5)),
        "--save_file", str(save),
        "--offload_model", "True",
    ]
    _invoke_generate(gm, "generate.py", argv)


def run_fun_official(cfg: Dict[str, Any], out_dir: Path, official: Path) -> None:
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    unified = cfg["unified"]
    routing = _load_routing(cfg)
    meta = _metadata(cfg)
    root = Path(cfg["dataset_root"])
    _push_path(official)
    os.chdir(official)
    from videox_fun.models import (  # type: ignore
        AutoencoderKLWan,
        CLIPModel,
        WanT5EncoderModel,
        WanTransformer3DModel,
    )
    from videox_fun.pipeline import WanFunControlPipeline  # type: ignore
    from videox_fun.data.dataset_image_video import process_pose_file  # type: ignore
    from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # type: ignore
    from videox_fun.utils.utils import get_image_latent, save_videos_grid  # type: ignore

    config_path = official / "config/wan2.1/wan_civitai.yaml"
    config = OmegaConf.load(str(config_path))
    ckpt = Path(unified["ckpt"])
    weight_dtype = torch.bfloat16
    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(ckpt, config["transformer_additional_kwargs"].get("transformer_subpath", "transformer")),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        torch_dtype=weight_dtype,
    )
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(ckpt, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(weight_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(ckpt, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"))
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(ckpt, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
    ).to(weight_dtype)
    clip = CLIPModel.from_pretrained(
        os.path.join(ckpt, config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"))
    ).to(weight_dtype)
    scheduler = FlowUniPCMultistepScheduler(
        **OmegaConf.to_container(config["scheduler_kwargs"])
    )
    pipeline = WanFunControlPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip,
    )
    pipeline.to("cuda")
    router = _create_router(cfg, routing, patch_h=40, patch_w=40)
    _inject(pipeline.transformer, router, "wan_fun_camera")
    _maybe_set_text_embeddings(router, pipeline, routing)

    image = _resolve_media(root, meta.get("cond_image"))
    sample_size = unified.get("sample_size") or [640, 640]
    camera_txt = unified.get("control_camera_txt", "asset/Zoom_Out.txt")
    if not Path(camera_txt).is_absolute():
        camera_txt = str(official / camera_txt)
    control = process_pose_file(camera_txt, sample_size[1], sample_size[0])
    start = get_image_latent(image, sample_size=sample_size)
    generator = torch.Generator(device="cuda").manual_seed(int(unified.get("seed", 43)))
    prompts = routing.get("segment_prompts") or [meta.get("prompt") or ""]
    video = pipeline(
        prompt=prompts[0],
        video_length=81,
        height=sample_size[0],
        width=sample_size[1],
        generator=generator,
        num_inference_steps=int(unified.get("num_inference_steps", 50)),
        start_image=start,
        control_camera_video=control,
    ).videos
    save = out_dir / "videos" / f"{cfg['sample_id']}_sceneshifter.mp4"
    save.parent.mkdir(parents=True, exist_ok=True)
    save_videos_grid(video, str(save), fps=16)


def run_hunyuan_official(cfg: Dict[str, Any], out_dir: Path, official: Path) -> None:
    import csv
    import subprocess
    from pathlib import Path as P

    import imageio
    import numpy as np
    import torch
    from einops import rearrange
    from torch.utils.data import DataLoader

    unified = cfg["unified"]
    routing = _load_routing(cfg)
    router = _create_router(cfg, routing, patch_h=44, patch_w=44)
    meta = _metadata(cfg)
    root = Path(cfg["dataset_root"])
    _push_path(official)
    os.chdir(official)
    os.environ["MODEL_BASE"] = str(unified["model_base"])
    os.environ["DISABLE_SP"] = "1"
    from hymm_sp.config import add_extra_args, sanity_check_args  # type: ignore
    from hymm_sp.data_kits.audio_dataset import VideoAudioTextLoaderVal  # type: ignore
    from hymm_sp.data_kits.face_align import AlignImage  # type: ignore
    from hymm_sp.sample_inference_audio import HunyuanVideoSampler  # type: ignore
    from transformers import AutoFeatureExtractor, WhisperModel
    import argparse

    image = _resolve_media(root, meta.get("cond_image"))
    audio = meta.get("cond_audio")
    if isinstance(audio, dict):
        audio = next(iter(audio.values()))
    audio = _resolve_media(root, audio)
    prompt = meta.get("prompt") or routing.get("segment_prompts", [""])[0]
    save_dir = out_dir / "videos"
    save_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        "--ckpt", os.path.join(unified["model_base"], "ckpts"),
        "--sample-n-frames", "129",
        "--seed", str(unified.get("base_seed", 43)),
        "--image-size", "704",
        "--cfg-scale", str(unified.get("cfg_scale", 7.5)),
        "--infer-steps", str(unified.get("infer_steps", 50)),
        "--use-deepcache", "1",
        "--save-path", str(save_dir),
        "--use-fp8",
    ]
    parser = argparse.ArgumentParser(add_help=False)
    parser = add_extra_args(parser)
    model_args = sanity_check_args(parser.parse_args(argv))
    device = torch.device("cuda")
    sampler = HunyuanVideoSampler.from_pretrained(model_args.ckpt, args=model_args, device=device)
    _inject(sampler.pipeline.transformer, router, "hunyuan_avatar")
    _maybe_set_text_embeddings(router, sampler, routing)

    whisper_path = os.path.join(unified["model_base"], "ckpts", "whisper-tiny")
    wav2vec = WhisperModel.from_pretrained(whisper_path).to(device=device, dtype=torch.float32)
    wav2vec.requires_grad_(False)
    feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_path)
    detector_path = os.path.join(unified["model_base"], "ckpts", "det_align", "detface.pt")
    align_instance = AlignImage("cuda", det_path=detector_path)

    csv_path = out_dir / "task.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["videoid", "image", "audio", "prompt", "fps"])
        writer.writeheader()
        writer.writerow(
            {
                "videoid": cfg["sample_id"],
                "image": image,
                "audio": audio,
                "prompt": prompt,
                "fps": 25.0,
            }
        )
    dataset = VideoAudioTextLoaderVal(
        image_size=704,
        meta_file=str(csv_path),
        text_encoder=sampler.text_encoder,
        text_encoder_2=sampler.text_encoder_2,
        feature_extractor=feature_extractor,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    model_args.seed = int(unified.get("base_seed", 43))
    samples = sampler.predict(model_args, batch, wav2vec, feature_extractor, align_instance)
    sample = samples["samples"][0]
    frame_count = min(int(batch["audio_len"][0]), sample.shape[1])
    video = rearrange(sample[:, :frame_count], "c f h w -> f h w c")
    video = (video * 255.0).clamp(0, 255).cpu().numpy().astype(np.uint8)
    raw_path = save_dir / f"{cfg['sample_id']}.video-only.mp4"
    output_path = save_dir / f"{cfg['sample_id']}_audio.mp4"
    imageio.mimsave(raw_path, video, fps=25.0)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(raw_path), "-i", audio,
            "-c:v", "copy", "-c:a", "aac", "-shortest", str(output_path),
        ],
        check=True,
    )
    raw_path.unlink(missing_ok=True)
    _ = P


def run_unified(cfg: Dict[str, Any], out_dir: Path) -> None:
    official = resolve_official_root(cfg)
    if not official.is_dir():
        raise FileNotFoundError(
            f"Official clone missing at {official}. Run scripts/setup_backbones.sh first."
        )
    dispatch = {
        "multitalk": run_talking_official,
        "infinitetalk": run_talking_official,
        "fantasytalking": run_fantasy_official,
        "wan_s2v": run_s2v_official,
        "wan_animate": run_animate_official,
        "wan_fun_camera": run_fun_official,
        "hunyuan_avatar": run_hunyuan_official,
    }
    fn = dispatch.get(cfg["backbone"])
    if fn is None:
        raise KeyError(cfg["backbone"])
    fn(cfg, out_dir, official)
