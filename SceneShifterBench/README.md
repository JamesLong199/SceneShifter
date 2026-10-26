# SceneShifterBench

Evaluation for the 150-id SceneShifter benchmark. The scripts live in this directory. The images, audio, masks, configs, and several checkpoints are downloaded separately and unpacked here, next to `scripts/`.

## Dataset

`dataset.zip` unpacks into this directory:

```text
SceneShifterBench/
  configs/
    base_model_wan/
    base_model_hunyuan/
    sceneshifter_multitalk/
    sceneshifter_infinitetalk/
    sceneshifter_fantasytalking/
    sceneshifter_wans2v/
    sceneshifter_hunyuan/
  images/
  audios/
  masks/
```

`base_model_wan` is the shared single-prompt config for MultiTalk, InfiniteTalk, FantasyTalking, and Wan2.2 S2V. `base_model_hunyuan` is the HunyuanVideo-Avatar base config. Each `sceneshifter_*` folder is that backbone's SceneShifter config. The generation CLI ids stay `wan_s2v` and `hunyuan_avatar`.

Audio files are the first 5.2 seconds of the source clips, which covers an 81-frame Wan clip at 16 fps and a 129-frame Hunyuan clip at 25 fps.

The archive prepared on the GPU server is `/mnt/whuscs/ljz/sceneshifter_release/dataset.zip`. Unzip it from the repository root:

```bash
unzip /path/to/dataset.zip -d SceneShifterBench
```

## Evaluation Checkpoints

`evaluation_models.zip` unpacks into this same directory. The paths match the ones the scripts read:

```text
checkpoints/syncnet_v2.model
checkpoints/shape_predictor_68_face_landmarks.dat
vbench/dino_model/dino_vitbase16_pretrain.pth
vbench/clip_model/ViT-L-14.pt
vbench/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth
vbench/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth
vbench2/arcface/resnet18_110.pth
vbench2/retinaface/retinaface_resnet50_2020-07-20-f168fae3c.zip
vbench2/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth
vbench2/anomaly_detector/human.pth
vbench2/anomaly_detector/face.pth
vbench2/anomaly_detector/hand.pth
```

The archive prepared on the GPU server is `/mnt/whuscs/ljz/sceneshifter_release/evaluation_models.zip`.

```bash
unzip /path/to/evaluation_models.zip -d SceneShifterBench
```

These three models are downloaded by each user. They are not in the zip.

SAM3 requires access approval, then:

```bash
huggingface-cli download facebook/sam3 --local-dir SceneShifterBench/sam3
```

The spatial-consistency script loads `SceneShifterBench/sam3/sam3.pt` with `load_from_HF=False`.

Clothes and background questions use LLaVA-Video:

```bash
huggingface-cli download lmms-lab/LLaVA-Video-7B-Qwen2 \
  --local-dir SceneShifterBench/vbench2/lmms-lab/LLaVA-Video-7B-Qwen2
```

Its vision tower is SigLIP. The semantics script expects this directory and does not call the Hub at runtime:

```bash
huggingface-cli download google/siglip-so400m-patch14-384 \
  --local-dir SceneShifterBench/vbench2/google/siglip-so400m-patch14-384
```

Qwen2.5-7B-Instruct, DreamSim, CLIP ViT-B/32, and tag2text are not used.

VBench and VBench-2.0 are code checkouts, not files in the zip. Set `VBENCH_REPO` and `VBENCH2_REPO` to those clones before evaluation. Lip sync also needs the SyncNet code directory in `EVALUATION_EAT_CODE` (the directory that contains `SyncNetInstance.py`). The generation environments (`sceneshifter_*`) are separate from the evaluation environments `sam3`, `vbench`, and `vbench2`. Frame extraction, scene timing, and lip sync use `ENV_LIPSYNC` (default `multitalk`).

## Score a Run

Generated videos are `videos/{id}.mp4` inside the run directory. SceneShifter runs use the matching `sceneshifter_*` config. Base-model runs use `base_model_wan` or `base_model_hunyuan`; those two configs select single-prompt timing. A 129-frame clip uses the Hunyuan buckets, and an 81-frame clip uses the Wan buckets.

```bash
bash SceneShifterBench/scripts/evaluate_sceneshifter.sh \
  outputs/batch/multitalk \
  --config sceneshifter_multitalk
```

```bash
bash SceneShifterBench/scripts/evaluate_sceneshifter.sh \
  outputs/batch_base/hunyuan_avatar \
  --config base_model_hunyuan
```

`--ids 023,032` and `--resume` limit or continue a run. Metric files are written inside the run directory.
