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
    # Animate parity metadata lives under configs/ and stores masks next to it.
    if str(cfg.get("metadata") or "").startswith("configs/"):
        asset_root = meta_path.parent
    else:
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


def _talking_latent_hw(image_path: str) -> tuple[int, int]:
    """Match InfiniteTalk/MultiTalk 480p ASPECT_RATIO_627 → DiT tokens (H/16, W/16)."""
    from PIL import Image

    buckets = {
        0.26: (320, 1216),
        0.38: (384, 1024),
        0.50: (448, 896),
        0.67: (512, 768),
        0.75: (576, 768),
        0.82: (576, 704),
        1.00: (640, 640),
        1.22: (704, 576),
        1.33: (768, 576),
        1.50: (768, 512),
        1.86: (832, 448),
        2.00: (896, 448),
        2.50: (960, 384),
        2.83: (1088, 384),
        3.60: (1152, 320),
        3.80: (1216, 320),
        4.00: (1280, 320),
    }
    with Image.open(image_path) as image:
        width, height = image.size
    ratio = height / max(width, 1)
    key = min(buckets, key=lambda item: abs(float(item) - ratio))
    target_h, target_w = buckets[key]
    return target_h // 16, target_w // 16


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

    targets = []
    for attr in ("model", "dit", "transformer", "pipe", "net", "pipeline", "noise_model"):
        inner = getattr(model, attr, None)
        if inner is not None:
            targets.append(inner)
    targets.append(model)
    last_err: Optional[Exception] = None
    for target in targets:
        try:
            bound = inject_router(target, router, adapter=adapter, expected_min=1)
            print(f"[parity-unified] injected {len(bound)} modules on {type(target).__name__}", flush=True)
            return bound
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise RuntimeError(f"inject_router failed for adapter={adapter}: {last_err}")


def _encoder_device(pipeline: Any):
    """Use the T5 parameter device. Wan S2V offload keeps the encoder on CPU."""
    encoder = getattr(pipeline, "text_encoder", None)
    if encoder is not None:
        try:
            return next(encoder.parameters()).device
        except (StopIteration, Exception):
            pass
    for attr in ("device",):
        value = getattr(pipeline, attr, None)
        if value is not None:
            return value
    for attr in ("model", "dit", "transformer"):
        inner = getattr(pipeline, attr, None)
        if inner is None:
            continue
        try:
            return next(inner.parameters()).device
        except StopIteration:
            continue
    return "cpu"


def _unwrap_prompt_embed(out: Any) -> Any:
    """DiffSynth encode_prompt returns ``{"context": tensor}``; Wan T5 returns a list."""
    if isinstance(out, dict):
        for key in ("context", "prompt_embeds", "encoder_hidden_states"):
            if key in out:
                return _unwrap_prompt_embed(out[key])
        return out
    if isinstance(out, (tuple, list)) and out:
        return _unwrap_prompt_embed(out[0])
    return out


def _is_text_embed(value: Any) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _commit_text_embeddings(
    router: Any,
    pipeline: Any,
    embeds: List[Any],
    *,
    source: str,
    project: bool = True,
) -> None:
    usable = [_unwrap_prompt_embed(item) for item in embeds]
    if not usable or not all(_is_text_embed(item) for item in usable):
        print(f"[parity-unified] skip non-tensor segment embeds from {source}", flush=True)
        return
    router.set_text_embeddings(usable)
    if project:
        from sceneshifter.adapters.base import attach_text_projector, project_router_text_embeddings

        attach_text_projector(router, pipeline)
        project_router_text_embeddings(router)
    from sceneshifter.parity.trace import trace_tensor

    for index, item in enumerate(router.text_embeddings):
        trace_tensor(
            "segment_text",
            item,
            backend=getattr(router, "backend", None),
            segment=index,
            source=source,
        )
    dims = [tuple(item.shape) if hasattr(item, "shape") else None for item in router.text_embeddings]
    print(f"[parity-unified] encoded {len(usable)} segment prompts via {source} {dims}", flush=True)


def _install_encode_prompt_hook(pipeline: Any, router: Any, routing: Dict[str, Any]) -> None:
    """Capture official DiffSynth null context and encode segments after T5 is onloaded."""
    if router is None:
        return
    original = getattr(pipeline, "encode_prompt", None)
    if original is None or getattr(pipeline, "_ss_encode_prompt_hooked", False):
        return
    state = {"busy": False}

    try:
        params = inspect.signature(original).parameters
    except (TypeError, ValueError):
        params = {}

    def _call_original(prompt, *, positive: bool = True, **kwargs):
        call_kwargs = dict(kwargs)
        if "positive" in params:
            call_kwargs["positive"] = positive
        elif not positive and "do_classifier_free_guidance" in params:
            call_kwargs.setdefault("do_classifier_free_guidance", False)
        return original(prompt, **call_kwargs)

    def wrapped(*args, **kwargs):
        prompt = args[0] if args else kwargs.get("prompt")
        positive = kwargs.pop("positive", True)
        extra = args[1:]
        call_kwargs = dict(kwargs)
        if extra and "positive" in params:
            out = original(prompt, *extra, **call_kwargs)
        elif extra:
            # Fun Camera: encode_prompt(prompt, negative_prompt=...) uses keywords only.
            out = original(prompt, *extra, **call_kwargs)
        else:
            out = _call_original(prompt, positive=positive, **call_kwargs)
        if not positive:
            router.null_text_embeddings = _unwrap_prompt_embed(out)
            print("[parity-unified] captured official null prompt embed", flush=True)
        elif isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
            router.null_text_embeddings = _unwrap_prompt_embed(out[1])
        if positive and not state["busy"] and getattr(router, "text_embeddings", None) is None:
            prompts = routing.get("segment_prompts") or []
            if prompts:
                state["busy"] = True
                try:
                    embeds = [_call_original(seg, positive=True) for seg in prompts]
                    # Keep raw T5 (4096). Official DiT projects after it onloads.
                    _commit_text_embeddings(
                        router, pipeline, embeds, source="encode_prompt hook", project=False
                    )
                    try:
                        import torch

                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                except Exception as exc:  # noqa: BLE001
                    print(f"[parity-unified] encode_prompt hook skip: {exc}", flush=True)
        return out

    pipeline.encode_prompt = wrapped
    pipeline._ss_encode_prompt_hooked = True


def _maybe_set_text_embeddings(router: Any, pipeline: Any, routing: Dict[str, Any]) -> None:
    prompts = routing.get("segment_prompts") or []
    if not prompts or not hasattr(router, "set_text_embeddings"):
        return
    _install_encode_prompt_hook(pipeline, router, routing)
    try:
        _capture_null_context(pipeline, router)
    except Exception as exc:  # noqa: BLE001
        print(f"[parity-unified] capture skip: {exc}", flush=True)
    try:
        embeds = []
        device = _encoder_device(pipeline)
        if hasattr(pipeline, "encode_prompt"):
            try:
                for prompt in prompts:
                    out = pipeline.encode_prompt(prompt)
                    embeds.append(_unwrap_prompt_embed(out))
            except Exception as exc:  # noqa: BLE001
                print(f"[parity-unified] encode_prompt skip: {exc}", flush=True)
                embeds = []
        if not embeds and hasattr(pipeline, "text_encoder") and callable(pipeline.text_encoder):
            encoder = pipeline.text_encoder
            for prompt in prompts:
                try:
                    out = encoder([prompt], device)
                except TypeError:
                    out = encoder([prompt])
                embeds.append(_unwrap_prompt_embed(out))
            neg = getattr(pipeline, "sample_neg_prompt", None)
            if neg:
                try:
                    null = encoder([neg], device)
                except TypeError:
                    null = encoder([neg])
                router.null_text_embeddings = _unwrap_prompt_embed(null)
        if embeds:
            _commit_text_embeddings(router, pipeline, embeds, source="encode_prompt")
    except Exception as exc:  # noqa: BLE001
        print(f"[parity-unified] text embed skip: {exc}", flush=True)


def _record_official_text_encode(router: Any, encode_fn, texts, out, args, kwargs):
    if not isinstance(texts, (list, tuple)) or not texts:
        return
    prompts = (getattr(router, "config", {}) or {}).get("segment_prompts") or []
    if len(texts) >= 2 and isinstance(out, (list, tuple)):
        router.null_text_embeddings = out[1]
        if getattr(router, "text_embeddings", None) is not None:
            return
        neg = texts[1]
        embeds = []
        for prompt in prompts:
            encoded = encode_fn([prompt, neg], *args, **kwargs)
            embeds.append(_unwrap_prompt_embed(encoded))
        if embeds and all(_is_text_embed(item) for item in embeds):
            router.set_text_embeddings(embeds)
            from sceneshifter.adapters.base import project_router_text_embeddings
            from sceneshifter.parity.trace import trace_tensor

            project_router_text_embeddings(router)
            for index, item in enumerate(router.text_embeddings):
                trace_tensor(
                    "segment_text",
                    item,
                    backend=getattr(router, "backend", None),
                    segment=index,
                    source="official T5 pair",
                )
            dims = [tuple(t.shape) if hasattr(t, "shape") else None for t in router.text_embeddings]
            print(f"[parity-unified] encoded {len(embeds)} segment prompts on official T5 {dims}", flush=True)
        return
    # Native S2V encodes every positive segment in one T5 batch and uses the
    # first result as its baseline context before routing starts.
    if (
        getattr(router, "backend", None) == "wan_s2v"
        and getattr(router, "text_embeddings", None) is None
        and prompts
    ):
        encoded = encode_fn(list(prompts), *args, **kwargs)
        if isinstance(encoded, (list, tuple)) and len(encoded) >= len(prompts):
            raw_embeds = list(encoded[: len(prompts)])
            if all(_is_text_embed(item) for item in raw_embeds):
                router.set_text_embeddings(raw_embeds)
                from sceneshifter.adapters.base import project_router_text_embeddings

                project_router_text_embeddings(router)
                dims = [
                    tuple(item.shape) if hasattr(item, "shape") else None
                    for item in router.text_embeddings
                ]
                print(
                    f"[parity-unified] encoded {len(raw_embeds)} S2V prompts "
                    f"in one T5 batch {dims}",
                    flush=True,
                )
                first = raw_embeds[0]
                if isinstance(out, tuple):
                    return (first, *out[1:])
                if isinstance(out, list):
                    return [first, *out[1:]]
                return first
    # Wan Animate encodes one prompt at a time: first call is cond, second is null.
    if getattr(router, "text_embeddings", None) is None and prompts:
        embeds = []
        for prompt in prompts:
            encoded = encode_fn([prompt], *args, **kwargs)
            embeds.append(_unwrap_prompt_embed(encoded))
        if embeds and all(_is_text_embed(item) for item in embeds):
            router.set_text_embeddings(embeds)
            from sceneshifter.adapters.base import project_router_text_embeddings
            from sceneshifter.parity.trace import trace_tensor

            project_router_text_embeddings(router)
            for index, item in enumerate(router.text_embeddings):
                trace_tensor(
                    "segment_text",
                    item,
                    backend=getattr(router, "backend", None),
                    segment=index,
                    source="official T5 single",
                )
            dims = [tuple(t.shape) if hasattr(t, "shape") else None for t in router.text_embeddings]
            print(
                f"[parity-unified] encoded {len(embeds)} segment prompts on single-prompt T5 {dims}",
                flush=True,
            )
        return
    if getattr(router, "null_text_embeddings", None) is None:
        router.null_text_embeddings = _unwrap_prompt_embed(out)
        print("[parity-unified] captured official null from single-prompt T5", flush=True)


class _TextEncoderProxy:
    """Preserve T5EncoderModel.model while capturing the official null prompt."""

    def __init__(self, inner: Any, router: Any):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_router", router)
        object.__setattr__(self, "_ss_null_captured", True)

    def __call__(self, texts, *args, **kwargs):
        out = self._inner(texts, *args, **kwargs)
        replacement = _record_official_text_encode(
            self._router, self._inner, texts, out, args, kwargs
        )
        return out if replacement is None else replacement

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


def _capture_null_context(pipeline: Any, router: Any) -> None:
    if router is None:
        return
    _install_encode_prompt_hook(pipeline, router, getattr(router, "config", {}) or {})
    encoder = getattr(pipeline, "text_encoder", None)
    if encoder is None or not callable(encoder) or getattr(encoder, "_ss_null_captured", False):
        return
    modules = getattr(pipeline, "_modules", None)
    if isinstance(modules, dict) and modules.get("text_encoder") is encoder:
        import types

        original_forward = encoder.forward

        def wrapped(self, texts, *args, **kwargs):
            out = original_forward(texts, *args, **kwargs)
            replacement = _record_official_text_encode(
                router, original_forward, texts, out, args, kwargs
            )
            return out if replacement is None else replacement

        encoder.forward = types.MethodType(wrapped, encoder)
        encoder._ss_null_captured = True
        return
    try:
        pipeline.text_encoder = _TextEncoderProxy(encoder, router)
    except TypeError:
        encoder._ss_null_captured = True


def _offload_dit_before_vae(pipeline: Any) -> None:
    """Official Animate/S2V leave the 14B DiT on GPU through VAE decode.

    After SceneShifter routing, an 80GB A800 can OOM on ``vae.decode``
    (~1.9 GiB) while 76 GiB is still held by the transformer. Move DiT
    weights to CPU and empty the cache immediately before decode.
    Official kernels stay; decoded pixels are unchanged.
    """
    vae = getattr(pipeline, "vae", None)
    if vae is None or not hasattr(vae, "decode") or getattr(vae, "_ss_decode_offload", False):
        return

    original = vae.decode

    def decode(*args, **kwargs):
        import torch

        for attr in ("noise_model", "model", "dit", "transformer"):
            module = getattr(pipeline, attr, None)
            if module is None or not hasattr(module, "cpu"):
                continue
            try:
                module.cpu()
                print(f"[parity-unified] offloaded {attr} before VAE decode", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[parity-unified] {attr} cpu skip: {exc}", flush=True)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return original(*args, **kwargs)

    vae.decode = decode
    vae._ss_decode_offload = True


def _wrap_named_classes(mod: Any, names: List[str], router: Any, adapter: str) -> None:
    for name in names:
        cls = getattr(mod, name, None)
        if cls is None or not inspect.isclass(cls):
            continue
        original_init = cls.__init__

        def make_init(orig, class_name):
            def wrapped(self, *args, **kwargs):
                orig(self, *args, **kwargs)
                target = (
                    getattr(self, "model", None)
                    or getattr(self, "dit", None)
                    or getattr(self, "transformer", None)
                    or getattr(self, "noise_model", None)
                    or self
                )
                try:
                    _inject(target, router, adapter)
                    _maybe_gate_talking_audio_cfg(target, router, adapter)
                    _maybe_set_text_embeddings(router, self, getattr(router, "config", {}))
                    _capture_null_context(self, router)
                    _offload_dit_before_vae(self)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[parity-unified] inject after {class_name}.__init__ failed: {exc}",
                        flush=True,
                    )

            return wrapped

        cls.__init__ = make_init(original_init, name)
        print(f"[parity-unified] wrapped {mod.__name__}.{name}.__init__", flush=True)


def _wrap_create_pipeline(mod: Any, router: Any, adapter: str, routing: Dict[str, Any]) -> None:
    if not hasattr(mod, "create_pipeline"):
        return
    original = mod.create_pipeline

    def wrapped(*args, **kwargs):
        pipe = original(*args, **kwargs)
        _inject(pipe, router, adapter)
        _maybe_gate_talking_audio_cfg(pipe, router, adapter)
        _maybe_set_text_embeddings(router, pipe, routing)
        _capture_null_context(pipe, router)
        _offload_dit_before_vae(pipe)
        return pipe

    mod.create_pipeline = wrapped


def _filter_official_argv(mod: Any, argv: List[str]) -> List[str]:
    """Validate flags before invoking an official parser.

    Silently dropping a parity flag changes the generation contract while still
    producing a plausible video. Sidecar-only options must be consumed before
    this function; every remaining flag is required to exist in the official
    entrypoint.
    """
    source_files = []
    for candidate in (getattr(mod, "__file__", None),):
        if candidate and Path(candidate).is_file():
            source_files.append(Path(candidate).read_text(errors="ignore"))
    blob = "\n".join(source_files)
    filtered: List[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            key = tok.split("=", 1)[0]
            if key not in blob:
                raise RuntimeError(
                    f"official entrypoint {getattr(mod, '__file__', mod)!r} "
                    f"does not declare required parity flag {key}"
                )
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
            params = inspect.signature(mod.generate).parameters
            if "pipeline" in params:
                mod.generate(args, pipeline=None)
            else:
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
    meta = _metadata(cfg)
    cond_image = _resolve_media(Path(cfg.get("dataset_root")), meta.get("cond_image"))
    try:
        patch_h, patch_w = _talking_latent_hw(str(cond_image))
    except Exception:
        patch_h, patch_w = 32, 48
    router = _create_router(cfg, routing, patch_h=patch_h, patch_w=patch_w)
    print(f"[parity-unified] talking router tokens {patch_h}x{patch_w}", flush=True)
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
        "--offload_model", "True",
        "--color_correction_strength", "0",
    ]
    if unified.get("infinitetalk_dir"):
        argv += ["--infinitetalk_dir", str(unified["infinitetalk_dir"])]
    if unified.get("size"):
        argv += ["--size", str(unified["size"])]
    if unified.get("mode"):
        argv += ["--mode", str(unified["mode"])]
    if unified.get("frame_num"):
        argv += ["--frame_num", str(unified["frame_num"])]
    # Official main has no SceneShifter audio-start CLI. The in-memory audio
    # residual and CFG wrappers consume this sidecar-only setting.
    _invoke_generate(gm, script, argv)


def _maybe_gate_talking_audio_cfg(target: Any, router: Any, adapter: str) -> None:
    if adapter not in {"multitalk", "infinitetalk"}:
        return
    audio_start = int(getattr(router, "audio_start_timestep", 0) or 0)
    if audio_start <= 0:
        return
    dit = (
        getattr(target, "model", None)
        or getattr(target, "dit", None)
        or getattr(target, "transformer", None)
        or target
    )
    _install_talking_audio_cfg_gate(dit, audio_start)


def _install_talking_audio_cfg_gate(dit: Any, audio_start: int) -> None:
    """Official-main always runs cond, drop_text, uncond.

    Exp uses 2-pass before ``audio_start``: ``uncond + text*(cond-uncond)``.
    Official-main uses ``uncond + text*(cond-drop_text) + audio*(drop_text-uncond)``
    where drop_text is null-text + audio. After audio residual is zeroed, reuse
    the drop_text output for uncond so the audio term is exactly 0.
    """
    import types

    import torch

    if dit is None or not hasattr(dit, "forward") or getattr(dit, "_ss_talking_audio_cfg_gated", False):
        return
    original = dit.forward
    state = {"t": object(), "step": -1, "calls": 0, "drop_text": None}

    def forward(self, *args, **kwargs):
        t = kwargs.get("t", kwargs.get("timestep", kwargs.get("timesteps")))
        if t is None and len(args) > 1:
            t = args[1]
        if torch.is_tensor(t):
            key = float(t.reshape(-1)[0].detach().float().cpu())
        else:
            try:
                key = float(t)
            except (TypeError, ValueError):
                key = t
        if key != state["t"]:
            state["t"] = key
            state["step"] += 1
            state["calls"] = 0
            state["drop_text"] = None
        state["calls"] += 1
        if (
            int(state["step"]) < int(audio_start)
            and state["calls"] == 3
            and state["drop_text"] is not None
        ):
            return state["drop_text"]
        out = original(*args, **kwargs)
        if int(state["step"]) < int(audio_start) and state["calls"] == 2:
            state["drop_text"] = out
        return out

    dit.forward = types.MethodType(forward, dit)
    dit._ss_talking_audio_cfg_gated = True
    print(
        f"[parity-unified] talking audio CFG gated: reuse drop_text as uncond until step {int(audio_start)}",
        flush=True,
    )


def _install_fantasy_audio_cfg_gate(dit: Any, audio_start: int) -> None:
    """Official-main always runs 3-pass audio CFG; exp uses 2-pass before audio_start.

    Reuse the cond prediction for the no-audio call so ``audio_cfg*(posi-noaudio)``
    is exactly 0 and thread_timestep only sees cond + uncond, matching exp.
    """
    import types

    import torch

    if dit is None or getattr(dit, "_ss_fantasy_audio_cfg_gated", False):
        return
    original = dit.forward
    state = {"t": object(), "step": -1, "calls": 0, "posi": None}

    def forward(self, *args, **kwargs):
        t = kwargs.get("timestep", kwargs.get("t"))
        if torch.is_tensor(t):
            key = float(t.reshape(-1)[0].detach().float().cpu())
        else:
            try:
                key = float(t)
            except (TypeError, ValueError):
                key = t
        if key != state["t"]:
            state["t"] = key
            state["step"] += 1
            state["calls"] = 0
            state["posi"] = None
        state["calls"] += 1
        if (
            int(state["step"]) < int(audio_start)
            and state["calls"] == 2
            and state["posi"] is not None
        ):
            return state["posi"]
        out = original(*args, **kwargs)
        if state["calls"] == 1:
            state["posi"] = out
        return out

    dit.forward = types.MethodType(forward, dit)
    dit._ss_fantasy_audio_cfg_gated = True
    print(
        f"[parity-unified] fantasy audio CFG gated: 2-pass until step {int(audio_start)}",
        flush=True,
    )


def _mux_fantasy_audio(video_mp4: Path, audio_path: str, dest: Path) -> None:
    """Mux with the env ffmpeg next to sys.executable. Keep the video if mux fails."""
    import shutil
    import subprocess

    dest.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg") or str(Path(sys.executable).resolve().parent / "ffmpeg")
    if Path(ffmpeg).is_file():
        cmd = [
            str(ffmpeg), "-y",
            "-i", str(video_mp4),
            "-i", str(audio_path),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-shortest",
            str(dest),
        ]
        print(f"[parity-unified] mux {' '.join(cmd)}", flush=True)
        try:
            subprocess.check_call(cmd)
            video_mp4.unlink(missing_ok=True)
            return
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"[parity-unified] ffmpeg mux failed: {exc}", flush=True)
    else:
        print(f"[parity-unified] ffmpeg missing at {ffmpeg}; keeping video-only", flush=True)
    shutil.copy2(video_mp4, dest)


def _resize_fantasy_image(image_path: str, image_size: Any):
    """Match FantasyTalking exp auto buckets without modifying official utils."""
    from PIL import Image

    if image_size is not None:
        from utils import resize_image_by_longest_edge  # type: ignore

        return resize_image_by_longest_edge(image_path, int(image_size))
    buckets = {
        "1:1": (640, 640),
        "2:3": (512, 768),
        "3:2": (768, 512),
        "4:3": (768, 576),
        "3:4": (576, 768),
    }
    image = Image.open(image_path).convert("RGB")
    ratio = float(image.width) / float(image.height)
    label, (width, height) = min(
        buckets.items(),
        key=lambda item: abs(ratio - float(item[1][0]) / float(item[1][1])),
    )
    print(
        f"[parity-unified] Fantasy auto bucket {label} -> {width}x{height}",
        flush=True,
    )
    return image.resize((width, height), Image.LANCZOS)


def run_fantasy_official(cfg: Dict[str, Any], out_dir: Path, official: Path) -> None:
    import librosa
    import torch
    from PIL import Image

    unified = cfg["unified"]
    routing = _load_routing(cfg)
    meta = _metadata(cfg)
    root = Path(cfg["dataset_root"])
    _push_path(official)
    os.chdir(official)
    from diffsynth import ModelManager, WanVideoPipeline  # type: ignore
    from model import FantasyTalkingAudioConditionModel  # type: ignore
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    from utils import get_audio_features, save_video  # type: ignore

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
    persist = unified.get("num_persistent_param_in_dit", None)
    pipe.enable_vram_management(
        num_persistent_param_in_dit=None if persist in (None, "", "none") else int(persist)
    )
    wav2vec_processor = Wav2Vec2Processor.from_pretrained(unified["wav2vec_model_dir"])
    wav2vec = Wav2Vec2Model.from_pretrained(unified["wav2vec_model_dir"]).to("cuda")

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
    wav2vec.to("cpu")
    torch.cuda.empty_cache()
    image = _resize_fantasy_image(image_path, unified.get("image_size"))
    width, height = image.size
    router = _create_router(cfg, routing, patch_h=max(1, height // 16), patch_w=max(1, width // 16))
    print(
        f"[parity-unified] fantasytalking router spatial={height}//16 x {width}//16 "
        f"-> {height // 16}x{width // 16}",
        flush=True,
    )
    _inject(pipe.dit, router, "fantasytalking")
    from sceneshifter.adapters.base import attach_text_projector

    attach_text_projector(router, pipe.dit)
    _install_fantasy_audio_cfg_gate(pipe.dit, int(unified.get("audio_start_timestep", 15)))
    # Encode segments later, when official __call__ has only T5 onloaded.
    _install_encode_prompt_hook(pipe, router, routing)
    try:
        _capture_null_context(pipe, router)
    except Exception as exc:  # noqa: BLE001
        print(f"[parity-unified] capture skip: {exc}", flush=True)
    torch.cuda.empty_cache()
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
    _mux_fantasy_audio(tmp, audio_path, save)
    _ = Image


def run_s2v_official(cfg: Dict[str, Any], out_dir: Path, official: Path) -> None:
    unified = cfg["unified"]
    meta = _metadata(cfg)
    root = Path(cfg["dataset_root"])
    routing = _load_routing(cfg)
    size = str(unified.get("size", "1024*704"))
    parts = [int(part) for part in size.replace("x", "*").split("*") if part]
    width, height = (parts + [1024, 704])[:2]
    router = _create_router(
        cfg, routing, patch_h=max(1, height // 16), patch_w=max(1, width // 16)
    )
    _push_path(official)
    os.chdir(official)
    import generate as gm  # type: ignore

    _wrap_create_pipeline(gm, router, "wan_s2v", routing)
    try:
        import wan  # type: ignore

        _wrap_named_classes(
            wan,
            ["WanS2V", "WanModel_S2V", "WanI2V", "WanT2V", "WanAnimate"],
            router,
            "wan_s2v",
        )
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
        "--size", str(unified.get("size", "1024*704")),
        "--ckpt_dir", str(unified["ckpt_dir"]),
        "--image", str(image),
        "--audio", str(audio),
        "--prompt", str(prompt),
        "--base_seed", str(unified.get("base_seed", 43)),
        "--save_file", str(save),
        "--offload_model", "True",
        "--convert_model_dtype",
        "--num_clip", str(unified.get("num_clip", 1)),
        "--infer_frames", str(unified.get("infer_frames", 84)),
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


def _allow_missing_flex_attention() -> None:
    """Official-main videox-fun imports lingbot/self-forcing at package import."""
    try:
        from torch.nn.attention.flex_attention import (  # noqa: F401
            BlockMask,
            create_block_mask,
            flex_attention,
        )
        return
    except Exception:
        pass
    import sys
    import types

    def _noop(*args, **kwargs):
        return None

    flex = types.ModuleType("torch.nn.attention.flex_attention")
    flex.BlockMask = type("BlockMask", (), {})
    flex.create_block_mask = _noop
    flex.flex_attention = _noop
    sys.modules["torch.nn.attention.flex_attention"] = flex


def _fun_center_crop_pil(image_path: str, sample_size: List[int]):
    """Match VideoX-Fun SceneShifter ``resize_and_centercrop_pil`` (bilinear cover crop)."""
    import math

    import numpy as np
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    target_h, target_w = int(sample_size[0]), int(sample_size[1])
    scale = max(target_h / float(image.height), target_w / float(image.width))
    final_h = max(target_h, int(math.ceil(scale * image.height)))
    final_w = max(target_w, int(math.ceil(scale * image.width)))
    resized = image.resize((final_w, final_h), resample=Image.BILINEAR)
    tensor = torch.from_numpy(np.array(resized))[None, ...].permute(0, 3, 1, 2).contiguous()
    cropped = TF.center_crop(tensor, [target_h, target_w])
    arr = cropped[0].permute(1, 2, 0).cpu().numpy().astype("uint8")
    return Image.fromarray(arr)


def run_fun_official(cfg: Dict[str, Any], out_dir: Path, official: Path) -> None:
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from omegaconf import OmegaConf
    from PIL import Image
    from transformers import AutoTokenizer

    unified = cfg["unified"]
    routing = _load_routing(cfg)
    meta = _metadata(cfg)
    root = Path(cfg["dataset_root"])
    _push_path(official)
    os.chdir(official)
    _allow_missing_flex_attention()
    # Import submodules directly. Official-main videox_fun.models.__init__
    # pulls lingbot/self-forcing which need torch.nn.attention.flex_attention.
    from videox_fun.models.wan_image_encoder import CLIPModel  # type: ignore
    from videox_fun.models.wan_text_encoder import WanT5EncoderModel  # type: ignore
    from videox_fun.models.wan_transformer3d import WanTransformer3DModel  # type: ignore
    from videox_fun.models.wan_vae import AutoencoderKLWan  # type: ignore
    from videox_fun.pipeline.pipeline_wan_fun_control import WanFunControlPipeline  # type: ignore
    from videox_fun.data.dataset_image_video import process_pose_file  # type: ignore
    from videox_fun.utils.utils import get_image_latent, save_videos_grid  # type: ignore

    try:
        from videox_fun.utils.utils import filter_kwargs  # type: ignore
    except Exception:  # noqa: BLE001
        filter_kwargs = lambda cls, kwargs: kwargs  # noqa: E731

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
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()
    clip = CLIPModel.from_pretrained(
        os.path.join(ckpt, config["image_encoder_kwargs"].get("image_encoder_subpath", "image_encoder"))
    ).to(weight_dtype).eval()
    # Official Fun Camera 092 uses Flow (Euler) + pipeline shift=3, not UniPC.
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(
            FlowMatchEulerDiscreteScheduler,
            OmegaConf.to_container(config["scheduler_kwargs"]),
        )
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
    sample_size = list(unified.get("sample_size") or [640, 640])
    router = _create_router(
        cfg, routing, patch_h=max(1, int(sample_size[0]) // 16), patch_w=max(1, int(sample_size[1]) // 16)
    )
    _inject(pipeline.transformer, router, "wan_fun_camera")
    from sceneshifter.adapters.base import attach_text_projector

    attach_text_projector(router, pipeline.transformer)
    _maybe_set_text_embeddings(router, pipeline, routing)

    image = _resolve_media(root, meta.get("cond_image"))
    camera_txt = unified.get("control_camera_txt", "asset/Zoom_Out.txt")
    if not Path(camera_txt).is_absolute():
        camera_txt = str(official / camera_txt)
    video_length = int(unified.get("video_length", 81))
    control = process_pose_file(camera_txt, sample_size[1], sample_size[0])
    control = control[:video_length].permute([3, 0, 1, 2]).unsqueeze(0)
    # Official 092: VAE start uses resize_mode=centercrop; CLIP gets the raw cond image.
    cropped = _fun_center_crop_pil(image, sample_size)
    start = get_image_latent(cropped, sample_size=sample_size)
    clip_image = Image.open(image).convert("RGB")
    generator = torch.Generator(device="cuda").manual_seed(int(unified.get("seed", 43)))
    prompt = meta.get("prompt") or (routing.get("segment_prompts") or [""])[0]
    negative = unified.get(
        "negative_prompt",
        "Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code, blurry, low quality, watermark, subtitles.",
    )
    video = pipeline(
        prompt,
        num_frames=video_length,
        negative_prompt=negative,
        height=sample_size[0],
        width=sample_size[1],
        generator=generator,
        guidance_scale=float(unified.get("guidance_scale", 6.0)),
        num_inference_steps=int(unified.get("num_inference_steps", 50)),
        start_image=start,
        control_camera_video=control,
        clip_image=clip_image,
        shift=int(unified.get("shift", 3)),
    ).videos
    save = out_dir / "videos" / f"{cfg['sample_id']}_sceneshifter.mp4"
    save.parent.mkdir(parents=True, exist_ok=True)
    save_videos_grid(video, str(save), fps=16)


def _sceneshifter_aspect_bucket(image_path: str) -> tuple[int, int]:
    """Match Hunyuan batch_generate_sceneshifter ASPECT_BUCKETS (H, W)."""
    import math

    from PIL import Image

    buckets = {
        "1_1": (640, 640),
        "2_3": (768, 512),
        "3_2": (512, 768),
        "4_3": (576, 768),
        "3_4": (768, 576),
    }
    values = {
        "1_1": 1.0,
        "2_3": 2.0 / 3.0,
        "3_2": 3.0 / 2.0,
        "4_3": 4.0 / 3.0,
        "3_4": 3.0 / 4.0,
    }
    with Image.open(image_path) as image:
        width, height = image.size
    ratio = width / max(height, 1)
    label = min(values, key=lambda name: abs(math.log(ratio / values[name])))
    return buckets[label]


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
    meta = _metadata(cfg)
    root = Path(cfg["dataset_root"])
    _push_path(official)
    os.chdir(official)
    os.environ["MODEL_BASE"] = str(unified["model_base"])
    os.environ["DISABLE_SP"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from hymm_sp.config import add_extra_args, sanity_check_args  # type: ignore
    from hymm_sp.data_kits.audio_dataset import VideoAudioTextLoaderVal  # type: ignore
    from hymm_sp.data_kits.face_align import AlignImage  # type: ignore
    from hymm_sp.sample_inference_audio import HunyuanVideoSampler  # type: ignore
    import hymm_sp.sample_inference_audio as hunyuan_inference  # type: ignore
    from transformers import AutoFeatureExtractor, WhisperModel
    import argparse
    from sceneshifter.adapters.hunyuan_avatar import hunyuan_multiface_mask

    hunyuan_inference.get_facemask = hunyuan_multiface_mask
    print("[parity-unified] Hunyuan multi-face union mask enabled", flush=True)

    image = _resolve_media(root, meta.get("cond_image"))
    audio = meta.get("cond_audio")
    if isinstance(audio, dict):
        audio = next(iter(audio.values()))
    audio = _resolve_media(root, audio)
    prompt = meta.get("prompt") or routing.get("segment_prompts", [""])[0]
    target_h, target_w = _sceneshifter_aspect_bucket(image)
    # Hunyuan VAE is 16x on the SceneShifter 480-class buckets (512x768 -> 32x48).
    router = _create_router(cfg, routing, patch_h=target_h // 16, patch_w=target_w // 16)
    save_dir = out_dir / "videos"
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt = unified.get("ckpt") or os.path.join(
        unified["model_base"],
        "ckpts",
        "hunyuan-video-t2v-720p",
        "transformers",
        "mp_rank_00_model_states_fp8.pt",
    )
    argv = [
        "--ckpt", ckpt,
        "--sample-n-frames", "129",
        "--seed", str(unified.get("base_seed", 43)),
        "--image-size", "512",
        "--cfg-scale", str(unified.get("cfg_scale", 7.5)),
        "--infer-steps", str(unified.get("infer_steps", 50)),
        "--flow-shift-eval-video", str(unified.get("flow_shift", 5.0)),
        "--use-deepcache", "1",
        "--save-path", str(save_dir),
        "--use-fp8",
    ]
    parser = argparse.ArgumentParser(add_help=False)
    parser = add_extra_args(parser)
    model_args = sanity_check_args(parser.parse_args(argv))
    device = torch.device("cuda")
    sampler = HunyuanVideoSampler.from_pretrained(model_args.ckpt, args=model_args, device=device)
    from hymm_sp.vae.vae import DiagonalGaussianDistribution  # type: ignore
    from sceneshifter.adapters.hunyuan_avatar import (
        install_hunyuan_prompt_routing,
        install_hunyuan_seeded_vae_sampling,
    )

    install_hunyuan_seeded_vae_sampling(
        sampler, DiagonalGaussianDistribution, int(unified.get("base_seed", 43))
    )
    install_hunyuan_prompt_routing(sampler.pipeline, router, routing, meta)
    bound = _inject(sampler, router, "hunyuan_avatar")
    print(
        f"[parity-unified] hunyuan bound {len(bound)} MMDiT blocks "
        f"target={target_h}x{target_w}",
        flush=True,
    )
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
    dataset_kwargs = dict(
        image_size=512,
        meta_file=str(csv_path),
        text_encoder=sampler.text_encoder,
        text_encoder_2=sampler.text_encoder_2,
        feature_extractor=feature_extractor,
    )
    try:
        dataset = VideoAudioTextLoaderVal(target_size=(target_h, target_w), **dataset_kwargs)
    except TypeError:
        dataset = VideoAudioTextLoaderVal(**dataset_kwargs)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
    # Feature-branch --infer-min default: standard 129-frame (~5s) clip.
    batch["audio_len"][0] = 129
    model_args.seed = int(unified.get("base_seed", 43))
    torch.cuda.empty_cache()
    samples = sampler.predict(model_args, batch, wav2vec, feature_extractor, align_instance)
    sample = samples["samples"][0]
    frame_count = min(129, int(batch["audio_len"][0]), sample.shape[1])
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
