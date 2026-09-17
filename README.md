# SceneShifter

Training-free multi-scene temporal control for pretrained DiT video backbones. SceneShifter is a **sidecar package**: it wraps official `main` checkouts at runtime and does not fork or rewrite those trees.

Supported adapters: MultiTalk, InfiniteTalk, FantasyTalking, HunyuanVideo-Avatar, Wan2.2 S2V, Wan2.2 Animate, Wan Fun camera control.

## Install

```bash
git clone <this-repo> && cd sceneshifter
pip install -e ".[dev]"
bash scripts/setup_backbones.sh          # official mains into third_party/
# bash scripts/setup_backbones.sh --with-reference   # jameslong199 ports for parity tests
bash scripts/setup_envs.sh --family talking
conda activate ss-talking
pip install -e .
```

Do **not** `pip install` a backbone as a site package. Put one official clone on `PYTHONPATH` per process.

## Shared features (F1–F12)

| ID | Feature |
|---|---|
| F1 | Enable / factory |
| F2 | `num_frames` + `frame_splits` segment map |
| F3 | Per-segment text prompts |
| F4 | `attention_control_schedule` |
| F5 | Self-attention scene mask (`full`/`fg`/`bg`) |
| F6 | Background masks (`1=BG`, `0=FG`) |
| F7 | Spatial KV filter |
| F8 | `len_trans` / `len_trans_context` / `trans_timestep_range` |
| F9 | Auto Q-scale **magnitude** (per-backbone formula) |
| F10 | New-scene BG-only **scope** (independent of F9; Hunyuan prefers `false`) |
| F11 | Live `ref_attn` mask postprocess |
| F12 | CFG / `apply_on_uncond_passes` |

## Use with an official pipeline

```python
from sceneshifter import create_router, inject_router, load_frame_prompt_config

cfg = load_frame_prompt_config("configs/two_scene.json", backend="wan_s2v")
router = create_router(cfg, backend="wan_s2v", patch_height=30, patch_width=52)
# pipe = official constructor from third_party/wan2.2, unmodified
inject_router(pipe, router, adapter="wan_s2v")
pipe.generate(...)
```

Hard checks run from `create_router` / `inject_router`. Disable with `SCENESHIFTER_STRICT=0`.

## Tests

```bash
pytest -m feature
pytest -m "feature or parity or pipeline or strict"
```

Layer C toy pipelines use a tiny DiT (no 14B weights). Full official generate still needs the family conda env and checkpoints.

## Environments

One env per family (`envs/ss-*.yml`): `ss-talking`, `ss-wan22`, `ss-hva`, `ss-fantasytalking`, `ss-videox-fun`. Official `wan` packages cannot share a process.

## Official clones

Pinned in `third_party/backbones.lock`. Runtime trees are upstream `main` only. jameslong199 feature branches are test/reference sources, not community clones.
