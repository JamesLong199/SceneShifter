# SceneShifter

## Installation

```bash
# Clone the repository
git clone git@github.com:JamesLong199/SceneShifter.git
cd SceneShifter

# Clone the official backbone code
bash scripts/setup_backbones.sh --backbone multitalk

# Create environment
bash scripts/setup_envs.sh --family multitalk
conda activate sceneshifter_multitalk

# Install dependencies
python -m pip install -e .
```

`--backbone` selects which official repository to clone. It uses the same names as the demo. `wan_s2v` and `wan_animate` both clone Wan2.2. `all` clones every backbone.

`--family` creates the conda environment in that row. The environment runs the backbones in the last column.

| `--family` | Environment to activate | `--backbone` |
|---|---|---|
| `multitalk` | `sceneshifter_multitalk` | `multitalk`, `infinitetalk` |
| `fantasy` | `sceneshifter_fantasy` | `fantasytalking` |
| `wans2v` | `sceneshifter_wans2v` | `wan_s2v`, `wan_animate` |
| `videoxfun` | `sceneshifter_videoxfun` | `wan_fun_camera` |
| `hunyuan` | `sceneshifter_hunyuan` | `hunyuan_avatar` |
| `all` | every environment above | every backbone |

Activate the environment from the row you created, then install this package into it with `python -m pip install -e .`.

## Download checkpoints

Each backbone has its own directory under `weights/`. The download includes the base model and the audio or text modules that backbone needs, so one backbone does not depend on another's weights.

```bash
python scripts/download_checkpoints.py --backbone multitalk
```

`--backbone` accepts `multitalk`, `infinitetalk`, `fantasytalking`, `wan_s2v`, `wan_animate`, `wan_fun_camera`, `hunyuan_avatar`, or `all`.

| `--backbone` | Directory |
|---|---|
| `multitalk` | `weights/multitalk` |
| `infinitetalk` | `weights/infinitetalk` |
| `fantasytalking` | `weights/fantasytalking` |
| `wan_s2v` | `weights/wan_s2v` |
| `wan_animate` | `weights/wan_animate` |
| `wan_fun_camera` | `weights/wan_fun_camera` |
| `hunyuan_avatar` | `weights/hunyuan_avatar` |

The demo command uses these directories. You do not pass a checkpoint path.

## Demo

Demo images, audio, and masks are in `examples/demos/`. Each backbone reads its own config in `examples/demos/configs/`.

Activate the matching environment, then run:

```bash
conda activate sceneshifter_multitalk
python scripts/parity_generate.py \
  --backbone multitalk \
  --demo examples/demos \
  --output_root outputs/demos
```

| Backbone | Environment | Command |
|---|---|---|
| MultiTalk | `sceneshifter_multitalk` | `python scripts/parity_generate.py --backbone multitalk --demo examples/demos --output_root outputs/demos` |
| InfiniteTalk | `sceneshifter_multitalk` | `python scripts/parity_generate.py --backbone infinitetalk --demo examples/demos --output_root outputs/demos` |
| FantasyTalking | `sceneshifter_fantasy` | `python scripts/parity_generate.py --backbone fantasytalking --demo examples/demos --output_root outputs/demos` |
| Wan2.2 S2V | `sceneshifter_wans2v` | `python scripts/parity_generate.py --backbone wan_s2v --demo examples/demos --output_root outputs/demos` |
| Wan2.2 Animate | `sceneshifter_wans2v` | `python scripts/parity_generate.py --backbone wan_animate --demo examples/demos --output_root outputs/demos` |
| Wan Fun camera | `sceneshifter_videoxfun` | `python scripts/parity_generate.py --backbone wan_fun_camera --demo examples/demos --output_root outputs/demos` |
| HunyuanVideo-Avatar | `sceneshifter_hunyuan` | `python scripts/parity_generate.py --backbone hunyuan_avatar --demo examples/demos --output_root outputs/demos` |

The video is written to `outputs/demos/<backbone>/unified/videos/`.
