# SceneShifter

## Demo

### Audio-driven Human Animation (MultiTalk)

<div style="overflow-x: auto;">

<table>
<tr>
<td><img src="assets/demos/multitalk/023.gif" width="240" alt="urban street to library"></td>
<td><img src="assets/demos/multitalk/032.gif" width="240" alt="courtyard to rooftop terrace to parking garage"></td>
<td><img src="assets/demos/multitalk/036.gif" width="240" alt="urban street to bus stop to hotel lobby to park to café"></td>
<td><img src="assets/demos/multitalk/062.gif" width="240" alt="parking garage to library to bus stop to conference hall"></td>
<td><img src="assets/demos/multitalk/063.gif" width="240" alt="classroom to elevator lobby to park to parking garage"></td>
<td><img src="assets/demos/multitalk/092.gif" width="240" alt="parking garage to office"></td>
</tr>
<tr>
<td>urban street → library</td>
<td>courtyard → rooftop terrace → parking garage</td>
<td>urban street → bus stop → hotel lobby → park → café</td>
<td>parking garage → library → bus stop → conference hall</td>
<td>classroom → elevator lobby → park → parking garage</td>
<td>parking garage → office</td>
</tr>
</table>

<table>
<tr>
<td><img src="assets/demos/multitalk/104.gif" width="240" alt="urban street to airport"></td>
<td><img src="assets/demos/multitalk/110.gif" width="240" alt="courtyard to airport"></td>
<td><img src="assets/demos/multitalk/120.gif" width="240" alt="airport to supermarket to elevator lobby to parking garage to urban street"></td>
<td><img src="assets/demos/multitalk/163.gif" width="240" alt="café to parking garage"></td>
<td><img src="assets/demos/multitalk/172.gif" width="240" alt="classroom to conference hall"></td>
</tr>
<tr>
<td>urban street → airport</td>
<td>courtyard → airport</td>
<td>airport → supermarket → elevator lobby → parking garage → urban street</td>
<td>café → parking garage</td>
<td>classroom → conference hall</td>
</tr>
</table>

</div>

### Video-driven Human Animation (Wan-Animate)

<div style="overflow-x: auto;">

<table>
<tr>
<td><img src="assets/demos/wan_animate/012.gif" width="240" alt="supermarket to conference hall to parking garage"></td>
<td><img src="assets/demos/wan_animate/047.gif" width="240" alt="hotel lobby to urban street"></td>
<td><img src="assets/demos/wan_animate/051.gif" width="240" alt="supermarket to podcast room to subway to courtyard"></td>
<td><img src="assets/demos/wan_animate/099.gif" width="240" alt="airport to courtyard to office"></td>
<td><img src="assets/demos/wan_animate/117.gif" width="240" alt="hotel lobby to rooftop terrace to covered walkway to supermarket"></td>
</tr>
<tr>
<td>supermarket → conference hall → parking garage</td>
<td>hotel lobby → urban street</td>
<td>supermarket → podcast room → subway → courtyard</td>
<td>airport → courtyard → office</td>
<td>hotel lobby → rooftop terrace → covered walkway → supermarket</td>
</tr>
</table>

</div>

### Camera Control (Wan Fun Camera)

<div style="overflow-x: auto;">

<table>
<tr>
<td><img src="assets/demos/wan_fun_camera/068_pan_left.gif" width="240" alt="park to hospital"></td>
<td><img src="assets/demos/wan_fun_camera/096_pan_right.gif" width="240" alt="courtyard to elevator lobby to workshop"></td>
<td><img src="assets/demos/wan_fun_camera/126_zoom_out.gif" width="240" alt="museum to parking garage"></td>
<td><img src="assets/demos/wan_fun_camera/071_clockwise.gif" width="240" alt="supermarket to conference hall to urban street to museum"></td>
<td><img src="assets/demos/wan_fun_camera/153_anti_clockwise.gif" width="240" alt="bus stop to train station to supermarket to beach"></td>
</tr>
<tr>
<td>pan left<br>park → hospital</td>
<td>pan right<br>courtyard → elevator lobby → workshop</td>
<td>zoom out<br>museum → parking garage</td>
<td>clockwise<br>supermarket → conference hall → urban street → museum</td>
<td>anti-clockwise<br>bus stop → train station → supermarket → beach</td>
</tr>
</table>

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
