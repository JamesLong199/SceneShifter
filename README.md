# SceneShifter

## Demo

Videos below are the SceneShifter clips from the rebuttal page. Each image is a GIF, following the [DiTCtrl](https://github.com/TencentARC/DiTCtrl) README.

### MultiTalk

<div style="overflow-x: auto;">

| Demo 1 | Demo 2 | Demo 3 | Demo 4 | Demo 5 | Demo 6 |
|---|---|---|---|---|---|
| <img src="assets/demos/multitalk/023.gif" width="240" alt="MultiTalk Demo 1"> | <img src="assets/demos/multitalk/032.gif" width="240" alt="MultiTalk Demo 2"> | <img src="assets/demos/multitalk/036.gif" width="240" alt="MultiTalk Demo 3"> | <img src="assets/demos/multitalk/062.gif" width="240" alt="MultiTalk Demo 4"> | <img src="assets/demos/multitalk/063.gif" width="240" alt="MultiTalk Demo 5"> | <img src="assets/demos/multitalk/092.gif" width="240" alt="MultiTalk Demo 6"> |

| Demo 7 | Demo 8 | Demo 9 | Demo 10 | Demo 11 |
|---|---|---|---|---|
| <img src="assets/demos/multitalk/104.gif" width="240" alt="MultiTalk Demo 7"> | <img src="assets/demos/multitalk/110.gif" width="240" alt="MultiTalk Demo 8"> | <img src="assets/demos/multitalk/120.gif" width="240" alt="MultiTalk Demo 9"> | <img src="assets/demos/multitalk/163.gif" width="240" alt="MultiTalk Demo 10"> | <img src="assets/demos/multitalk/172.gif" width="240" alt="MultiTalk Demo 11"> |

</div>

### Wan-Animate

<div style="overflow-x: auto;">

| Demo 1 | Demo 2 | Demo 3 | Demo 4 | Demo 5 |
|---|---|---|---|---|
| <img src="assets/demos/wan_animate/012.gif" width="240" alt="Wan-Animate Demo 1"> | <img src="assets/demos/wan_animate/047.gif" width="240" alt="Wan-Animate Demo 2"> | <img src="assets/demos/wan_animate/051.gif" width="240" alt="Wan-Animate Demo 3"> | <img src="assets/demos/wan_animate/099.gif" width="240" alt="Wan-Animate Demo 4"> | <img src="assets/demos/wan_animate/117.gif" width="240" alt="Wan-Animate Demo 5"> |

</div>

### Wan Fun Camera

<div style="overflow-x: auto;">

| Pan Left | Pan Right | Zoom Out | Clockwise | Anti-Clockwise |
|---|---|---|---|---|
| <img src="assets/demos/wan_fun_camera/068_pan_left.gif" width="240" alt="Pan Left"> | <img src="assets/demos/wan_fun_camera/096_pan_right.gif" width="240" alt="Pan Right"> | <img src="assets/demos/wan_fun_camera/126_zoom_out.gif" width="240" alt="Zoom Out"> | <img src="assets/demos/wan_fun_camera/071_clockwise.gif" width="240" alt="Clockwise"> | <img src="assets/demos/wan_fun_camera/153_anti_clockwise.gif" width="240" alt="Anti-Clockwise"> |

</div>

## Installation

```bash
# Clone the repository
git clone git@github.com:JamesLong199/SceneShifter.git
cd SceneShifter

# Clone the official backbone code
bash scripts/setup_backbones.sh --backbone [BACKBONE] # e.g. multitalk

# Create environment
bash scripts/setup_envs.sh --family [BACKBONE] # e.g. multitalk
conda activate [ENV] # e.g. sceneshifter_multitalk

# Install dependencies
python -m pip install -e .
```

`--backbone` selects which official repository to clone. It uses the same names as the demo. `wan_s2v` and `wan_animate` both clone Wan2.2. `all` clones every backbone.

`--family` creates the conda environment in that row. The environment runs the backbones in the last column.

| `--family` | Environment | `--backbone` |
|---|---|---|
| `multitalk` | `sceneshifter_multitalk` | `multitalk`, `infinitetalk` |
| `fantasy` | `sceneshifter_fantasy` | `fantasytalking` |
| `wans2v` | `sceneshifter_wans2v` | `wan_s2v`, `wan_animate` |
| `videoxfun` | `sceneshifter_videoxfun` | `wan_fun_camera` |
| `hunyuan` | `sceneshifter_hunyuan` | `hunyuan_avatar` |
| `all` | every environment above | every backbone |

Activate the environment from the row you created, then install this package into it with `python -m pip install -e .`.

## Download checkpoints

Each backbone has its own directory under `weights/`. The download includes the base model and auxiliary models such as audio and text encoders.

```bash
python scripts/download_checkpoints.py --backbone [BACKBONE] # e.g. multitalk
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

## Run a demo

Demo images, audio, and masks are in `examples/demos/`. Each backbone reads its own config in `examples/demos/configs/`.

| Backbone | Command |
|---|---|
| MultiTalk | `conda activate sceneshifter_multitalk && python scripts/generate.py --backbone multitalk --demo examples/demos --output_root outputs/demos` |
| InfiniteTalk | `conda activate sceneshifter_multitalk && python scripts/generate.py --backbone infinitetalk --demo examples/demos --output_root outputs/demos` |
| FantasyTalking | `conda activate sceneshifter_fantasy && python scripts/generate.py --backbone fantasytalking --demo examples/demos --output_root outputs/demos` |
| Wan2.2 S2V | `conda activate sceneshifter_wans2v && python scripts/generate.py --backbone wan_s2v --demo examples/demos --output_root outputs/demos` |
| Wan2.2 Animate | `conda activate sceneshifter_wans2v && python scripts/generate.py --backbone wan_animate --demo examples/demos --output_root outputs/demos` |
| Wan Fun camera | `conda activate sceneshifter_videoxfun && python scripts/generate.py --backbone wan_fun_camera --demo examples/demos --output_root outputs/demos` |
| HunyuanVideo-Avatar | `conda activate sceneshifter_hunyuan && python scripts/generate.py --backbone hunyuan_avatar --demo examples/demos --output_root outputs/demos` |

The video is written to `outputs/demos/<backbone>/videos/`.
