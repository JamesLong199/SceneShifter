# SceneShifterBench

## Download Dataset

Download the dataset from [Google Drive](https://drive.google.com/DATASET_LINK).

Unzip it from the repository root:

```bash
unzip /path/to/dataset.zip -d SceneShifterBench
```

## Download Evaluation Model Checkpoints

### Google Drive Download

Download the first part of the evaluation checkpoints from [Google Drive](https://drive.google.com/EVALUATION_MODELS_LINK).

Unzip them from the repository root:

```bash
unzip /path/to/evaluation_models.zip -d SceneShifterBench
```

### HuggingFace Download

Download the second part of the evaluation checkpoints from HuggingFace.

SAM3 is a separate download. Request access at [facebook/sam3](https://huggingface.co/facebook/sam3), then:

```bash
huggingface-cli download facebook/sam3 --local-dir SceneShifterBench/sam3
```

```bash
huggingface-cli download lmms-lab/LLaVA-Video-7B-Qwen2 \
  --local-dir SceneShifterBench/vbench2/lmms-lab/LLaVA-Video-7B-Qwen2
huggingface-cli download google/siglip-so400m-patch14-384 \
  --local-dir SceneShifterBench/vbench2/google/siglip-so400m-patch14-384
```

## Environment Installation

Create one conda environment for each scorer.

| Variable | Environment | Role |
|---|---|---|
| `ENV_LIPSYNC` | `sceneshifter_multitalk` | Temporal accuracy and lip sync |
| `ENV_SAM3` | `sam3` | Spatial subject consistency |
| `ENV_VBENCH` | `vbench` | Aesthetic quality and imaging quality |
| `ENV_VBENCH2` | `vbench2` | Human identity, clothes, and background semantics |

### Sam3 Installation

SAM3, following the [SAM 3 install](https://github.com/facebookresearch/sam3#installation):

```bash
conda create -n sam3 python=3.12 -y
conda activate sam3
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

### VBench Installation

VBench, for aesthetic quality and imaging quality:

```bash
conda create -n vbench python=3.10 -y
conda activate vbench
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install vbench
```

### VBench-2.0 Installation

VBench-2.0, for human identity, clothes, and background semantics. This follows the [VBench-2.0 install](https://github.com/Vchitect/VBench/tree/master/VBench-2.0#installation).

```bash
git clone https://github.com/Vchitect/VBench.git
cd VBench/VBench-2.0
conda create -n vbench2 python=3.10 -y
conda activate vbench2
conda install psutil
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install ninja
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.2.post1
pip install -r requirement.txt
pip install retinaface_pytorch==0.0.8 --no-deps
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html --no-cache-dir
```

If `mmcv` reports that its version exceeds 2.1.0, raise `mmcv_maximum_version` to `2.3.0` in the installed `mmdet` and `mmyolo` packages.

### DINO Source Code

Clone DINO source code.

```bash
git clone https://github.com/facebookresearch/dino \
  SceneShifterBench/vbench/dino_model/facebookresearch_dino_main
```

## Test on SceneShifterBench

Export the scorer environment names.

```bash
export ENV_LIPSYNC=sceneshifter_multitalk
export ENV_SAM3=sam3
export ENV_VBENCH=vbench
export ENV_VBENCH2=vbench2
```

The batch script runs MultiTalk, InfiniteTalk, FantasyTalking, WanS2V, or HunyuanVideo-Avatar over a metadata folder. One process is started per GPU. A clip is skipped when `videos/<id>.mp4` is already in the output folder.

| Backbone | Command |
|---|---|
| MultiTalk | `conda activate sceneshifter_multitalk && python scripts/batch_generate.py --backbone multitalk --dataset_root SceneShifterBench --metadata_glob 'configs/sceneshifter_multitalk/*_metadata.json' --output_root outputs/batch/multitalk --gpus 0,1` |
| InfiniteTalk | `conda activate sceneshifter_multitalk && python scripts/batch_generate.py --backbone infinitetalk --dataset_root SceneShifterBench --metadata_glob 'configs/sceneshifter_infinitetalk/*_metadata.json' --output_root outputs/batch/infinitetalk --gpus 0,1` |
| FantasyTalking | `conda activate sceneshifter_fantasy && python scripts/batch_generate.py --backbone fantasytalking --dataset_root SceneShifterBench --metadata_glob 'configs/sceneshifter_fantasytalking/*_metadata.json' --output_root outputs/batch/fantasytalking --gpus 0,1` |
| WanS2V | `conda activate sceneshifter_wans2v && python scripts/batch_generate.py --backbone wan_s2v --dataset_root SceneShifterBench --metadata_glob 'configs/sceneshifter_wans2v/*_metadata.json' --output_root outputs/batch/wan_s2v --gpus 0,1` |
| HunyuanVideo-Avatar | `conda activate sceneshifter_hunyuan && python scripts/batch_generate.py --backbone hunyuan_avatar --dataset_root SceneShifterBench --metadata_glob 'configs/sceneshifter_hunyuan/*_metadata.json' --output_root outputs/batch/hunyuan_avatar --gpus 0,1` |

## Evaluate Generation Results

Generated videos are `videos/{id}.mp4` inside the run directory. Run the commands below from the repository root. `--ids 023,032` and `--resume` limit or continue a run. Metric files are written inside the run directory.

SceneShifter Wan-based models are MultiTalk, InfiniteTalk, FantasyTalking, and Wan2.2 S2V. These clips are 81 frames. Pass the matching config:

```bash
bash SceneShifterBench/scripts/evaluate_sceneshifter.sh \
  outputs/batch/multitalk \
  --config sceneshifter_multitalk
```

The other Wan-based configs are `sceneshifter_infinitetalk`, `sceneshifter_fantasytalking`, and `sceneshifter_wans2v`.

SceneShifter HunyuanVideo-Avatar clips are 129 frames:

```bash
bash SceneShifterBench/scripts/evaluate_sceneshifter.sh \
  outputs/batch/hunyuan_avatar \
  --config sceneshifter_hunyuan
```

Base-model runs use a single prompt. Wan-based base runs (MultiTalk, InfiniteTalk, FantasyTalking, and Wan2.2 S2V) use `base_model_wan`:

```bash
bash SceneShifterBench/scripts/evaluate_sceneshifter.sh \
  outputs/batch_base/multitalk \
  --config base_model_wan
```

The HunyuanVideo-Avatar base run uses `base_model_hunyuan`:

```bash
bash SceneShifterBench/scripts/evaluate_sceneshifter.sh \
  outputs/batch_base/hunyuan_avatar \
  --config base_model_hunyuan
```
