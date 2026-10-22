# SceneShifter

Training-free multi-scene temporal control for pretrained DiT video backbones. SceneShifter wraps an official checkout at runtime and injects the router into that model. It does not fork or rewrite the official tree.

Supported adapters: MultiTalk, InfiniteTalk, FantasyTalking, HunyuanVideo-Avatar, Wan2.2 S2V, Wan2.2 Animate, Wan Fun camera control.

## Install

```bash
git clone git@github.com:JamesLong199/SceneShifter.git
cd SceneShifter
python -m pip install -e ".[dev]"
bash scripts/setup_backbones.sh
bash scripts/setup_envs.sh --family all
```

`setup_backbones.sh` clones the official `main` revisions pinned in `third_party/backbones.lock`. One official tree is imported per process. Do not `pip install` a backbone as a site package.

Family environments live in `envs/`:

| Family | Conda env | Backbones |
|---|---|---|
| `talking` | `ss-talking` | MultiTalk, InfiniteTalk |
| `fantasytalking` | `ss-fantasytalking` | FantasyTalking |
| `wan22` | `ss-wan22` | Wan2.2 S2V, Wan2.2 Animate |
| `videox-fun` | `ss-videox-fun` | Wan Fun camera |
| `hva` | `ss-hva` | HunyuanVideo-Avatar |

Activate the family env and install this package into it:

```bash
conda activate ss-talking
python -m pip install -e .
```

A full 14B demo also needs that backbone's checkpoint and the Python packages its official script imports. On the lab machine those environments are `multitalk`, `infinitetalk`, `wans2v`, and `HunyuanVideo-Avatar`. Point `configs/parity/<backbone>.json` `unified` checkpoint fields at the local weights before generating. Checkpoints are not part of this repo.

## Demo inputs

`examples/demos/<backbone>/` is one self-contained sample per backbone. The sample id is the first qualitative example for that backbone in `SceneShifter_dataset/demo_example/demo_bookkeeping.json`. FantasyTalking, Wan2.2 S2V, and Wan Fun camera do not have their own bookkeeping section, so those demos use the bookkeeping id that the parity config already runs (032, 032, and 092).

| Backbone | Demo directory | Sample | Bookkeeping source |
|---|---|---|---|
| MultiTalk | `examples/demos/multitalk` | 023 | MultiTalk vs Baselines, Demo 1 |
| InfiniteTalk | `examples/demos/infinitetalk` | 159 | InfiniteTalk, Demo 1 |
| FantasyTalking | `examples/demos/fantasytalking` | 032 | MultiTalk Demo 2; FantasyTalking parity sample |
| HunyuanVideo-Avatar | `examples/demos/hunyuan_avatar` | 005 | Hunyuan, Demo 1 |
| Wan2.2 S2V | `examples/demos/wan_s2v` | 032 | MultiTalk Demo 2; Wan S2V parity sample |
| Wan2.2 Animate | `examples/demos/wan_animate` | 012 | Wan-Animate, Demo 1 |
| Wan Fun camera | `examples/demos/wan_fun_camera` | 092 | MultiTalk Demo 6; Fun Camera parity sample |

Each directory contains `metadata.json`, `demo.json`, the conditioning image, audio, and background masks named the way the metadata expects. Wan Animate also includes the `process/` pose, face, and reference inputs. Wan Fun camera includes `camera/Zoom_Out.txt`.

## Run a demo

From the repo root, with the matching environment active and checkpoints filled in:

```bash
python scripts/parity_generate.py \
  --side unified \
  --backbone multitalk \
  --gpu 0 \
  --demo examples/demos/multitalk \
  --output_root outputs/demos
```

Replace `--backbone` and `--demo` together. `--gpu` is the CUDA device index visible to this process. The video is written under `outputs/demos/<backbone>/unified/videos/`.

| Backbone | Command |
|---|---|
| MultiTalk | `python scripts/parity_generate.py --side unified --backbone multitalk --gpu 0 --demo examples/demos/multitalk --output_root outputs/demos` |
| InfiniteTalk | `python scripts/parity_generate.py --side unified --backbone infinitetalk --gpu 0 --demo examples/demos/infinitetalk --output_root outputs/demos` |
| FantasyTalking | `python scripts/parity_generate.py --side unified --backbone fantasytalking --gpu 0 --demo examples/demos/fantasytalking --output_root outputs/demos` |
| HunyuanVideo-Avatar | `python scripts/parity_generate.py --side unified --backbone hunyuan_avatar --gpu 0 --demo examples/demos/hunyuan_avatar --output_root outputs/demos` |
| Wan2.2 S2V | `python scripts/parity_generate.py --side unified --backbone wan_s2v --gpu 0 --demo examples/demos/wan_s2v --output_root outputs/demos` |
| Wan2.2 Animate | `python scripts/parity_generate.py --side unified --backbone wan_animate --gpu 0 --demo examples/demos/wan_animate --output_root outputs/demos` |
| Wan Fun camera | `python scripts/parity_generate.py --side unified --backbone wan_fun_camera --gpu 0 --demo examples/demos/wan_fun_camera --output_root outputs/demos` |

MultiTalk, InfiniteTalk, Wan2.2 S2V, and Wan2.2 Animate run with `--offload_model True`. Wan Fun camera uses Diffusers CPU offload: UMT5, CLIP, and the VAE stay on CPU during the DiT loop. Leave that path in place. Loading the 14B Fun Camera transformer, text encoder, and CLIP together needs more than a shared 80 GB GPU.

Checkpoint fields to set in `configs/parity/<backbone>.json` before the first run:

| Backbone | Fields |
|---|---|
| MultiTalk | `unified.ckpt_dir`, `unified.wav2vec_dir` |
| InfiniteTalk | `unified.ckpt_dir`, `unified.infinitetalk_dir`, `unified.wav2vec_dir` |
| FantasyTalking | `unified.wan_model_dir`, `unified.fantasytalking_model_path`, `unified.wav2vec_model_dir` |
| HunyuanVideo-Avatar | `unified.model_base`, `unified.ckpt` |
| Wan2.2 S2V | `unified.ckpt_dir` |
| Wan2.2 Animate | `unified.ckpt_dir` |
| Wan Fun camera | `unified.ckpt` |

Dry-check that the router and official clone resolve, without loading a checkpoint:

```bash
python scripts/generate.py \
  --backbone multitalk \
  --frame_prompt_config examples/demos/multitalk/metadata.json \
  --dry_run
```

## Use inside an official pipeline

```python
from sceneshifter import create_router, inject_router, load_frame_prompt_config

cfg = load_frame_prompt_config("examples/demos/wan_s2v/metadata.json", backend="wan_s2v")
router = create_router(cfg, backend="wan_s2v", patch_height=30, patch_width=52)
# pipe = official constructor from third_party/wan2.2
inject_router(pipe, router, adapter="wan_s2v")
pipe.generate(...)
```

Hard checks run from `create_router` / `inject_router`. Disable with `SCENESHIFTER_STRICT=0`.

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

## Tests

```bash
pytest -m feature
pytest -m "feature or parity or pipeline or strict"
```

Layer C toy pipelines use a tiny DiT and do not need 14B weights.
