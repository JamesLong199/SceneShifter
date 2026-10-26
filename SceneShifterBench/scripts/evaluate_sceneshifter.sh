#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CALLER_PWD="$(pwd)"
cd "${BENCH_ROOT}"

eval "$(conda shell.bash hook)"

conda_activate() {
  set +u
  conda activate "$1"
  set -u
}

conda_deactivate() {
  set +u
  conda deactivate
  set -u
}

ENV_LIPSYNC="${ENV_LIPSYNC:-multitalk}"
ENV_SAM3="${ENV_SAM3:-sam3}"
ENV_VBENCH2="${ENV_VBENCH2:-vbench2}"
ENV_VBENCH="${ENV_VBENCH:-vbench}"

usage() {
  echo "Usage: bash scripts/evaluate_sceneshifter.sh RUN_DIR --config CONFIG [--resume] [--ids SPEC]" >&2
  exit 1
}

if [[ $# -lt 1 ]]; then
  usage
fi

RUN_DIR="$1"
shift

CONFIG=""
RESUME_MODE=""
IDS_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      shift
      if [[ $# -lt 1 ]]; then
        echo "Missing value for --config" >&2
        exit 1
      fi
      CONFIG="$1"
      shift
      ;;
    --config=*)
      CONFIG="${1#--config=}"
      shift
      ;;
    --resume)
      RESUME_MODE="--resume"
      shift
      ;;
    --ids)
      shift
      if [[ $# -lt 1 ]]; then
        echo "Missing value for --ids" >&2
        exit 1
      fi
      IDS_ARG="$1"
      shift
      ;;
    --ids=*)
      IDS_ARG="${1#--ids=}"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${CONFIG}" ]]; then
  echo "Missing --config" >&2
  usage
fi

case "${CONFIG}" in
  sceneshifter_multitalk|sceneshifter_infinitetalk|sceneshifter_fantasytalking|sceneshifter_wans2v|sceneshifter_hunyuan|base_model_wan|base_model_hunyuan)
    ;;
  *)
    echo "Unknown config '${CONFIG}'." >&2
    echo "Expected one of: sceneshifter_multitalk sceneshifter_infinitetalk sceneshifter_fantasytalking sceneshifter_wans2v sceneshifter_hunyuan base_model_wan base_model_hunyuan" >&2
    exit 1
    ;;
esac

if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${CALLER_PWD}/${RUN_DIR}"
fi

METADATA_DIR="${BENCH_ROOT}/configs/${CONFIG}"
if [[ ! -d "${METADATA_DIR}" ]]; then
  echo "Missing metadata dir: ${METADATA_DIR}" >&2
  exit 1
fi

VIDEO_GLOB="*.mp4"
SUB_CLIP_DIR="${RUN_DIR}/sub_clip"
mkdir -p "${SUB_CLIP_DIR}"

IDS_FLAGS=()
if [[ -n "${IDS_ARG}" ]]; then
  IDS_FLAGS=(--ids "${IDS_ARG}")
fi
RESUME_FLAGS=()
if [[ -n "${RESUME_MODE}" ]]; then
  RESUME_FLAGS=(--resume)
fi

echo "[evaluate_sceneshifter] run=${RUN_DIR} config=${CONFIG} ids=${IDS_ARG:-all} resume=${RESUME_MODE:-off}"

conda_activate "${ENV_LIPSYNC}"

if [[ -n "${RESUME_MODE}" ]] || [[ -n "${IDS_ARG}" ]] || [[ ! -d "${RUN_DIR}/frames" ]] || [[ "$(ls "${RUN_DIR}/frames" 2>/dev/null | wc -l)" -lt 1 ]]; then
  python scripts/extract_frames.py \
    --input-dir "${RUN_DIR}" \
    --output-dir "${RUN_DIR}" \
    --fps 25.0 \
    --glob "${VIDEO_GLOB}" \
    "${IDS_FLAGS[@]}" \
    "${RESUME_FLAGS[@]}"
else
  echo "Frames already present; skipping extract_frames.py"
fi

if [[ -n "${RESUME_MODE}" ]] || [[ -n "${IDS_ARG}" ]] || [[ ! -f "${RUN_DIR}/scene_transitions.json" ]]; then
  if [[ "${CONFIG}" == "base_model_wan" || "${CONFIG}" == "base_model_hunyuan" ]]; then
    HUNYUAN_FLAGS=()
    if [[ "${CONFIG}" == "base_model_hunyuan" ]]; then
      HUNYUAN_FLAGS=(--hunyuan)
    fi
    python scripts/run_multitalk_scene_transitions_hybrid.py \
      --input-dir "${RUN_DIR}" \
      --glob "${VIDEO_GLOB}" \
      --config-dir "${METADATA_DIR}" \
      "${HUNYUAN_FLAGS[@]}" \
      --output-json "${RUN_DIR}/scene_transitions.json" \
      "${IDS_FLAGS[@]}" \
      "${RESUME_FLAGS[@]}"
  else
    python scripts/run_sceneshifter_scene_transitions_hybrid.py \
      --input-dir "${RUN_DIR}" \
      --glob "${VIDEO_GLOB}" \
      --metadata-dir "${METADATA_DIR}" \
      --guided-gradual \
      --output-json "${RUN_DIR}/scene_transitions.json" \
      "${IDS_FLAGS[@]}" \
      "${RESUME_FLAGS[@]}"
  fi
else
  echo "scene_transitions.json already present; skipping transition detection"
fi

conda_deactivate
conda_activate "${ENV_SAM3}"

python scripts/compute_spatial_subject_consistency.py \
  --scene_transitions_json "${RUN_DIR}/scene_transitions.json" \
  --frames_root "${RUN_DIR}/frames" \
  --metadata_dir "${METADATA_DIR}" \
  --sam3_ckpt "${BENCH_ROOT}/sam3/sam3.pt" \
  --w 1 \
  --output_json "${RUN_DIR}/spatial_subject_consistency.json" \
  "${IDS_FLAGS[@]}" \
  "${RESUME_FLAGS[@]}"

conda_deactivate

conda_activate "${ENV_VBENCH2}"

python scripts/evaluate_i2v_scene0_human_fidelity.py \
  --scene-transitions-json "${RUN_DIR}/scene_transitions.json" \
  --frames-root "${RUN_DIR}/frames" \
  --output-dir "${RUN_DIR}/i2v_scene0_eval" \
  --sub-clip-dir "${SUB_CLIP_DIR}" \
  --vbench-cache-dir "${BENCH_ROOT}/vbench" \
  --local \
  --device cuda \
  --dimensions i2v_subject \
  --eval-scene1plus \
  --eval-human-identity \
  --eval-human-clothes \
  --eval-human-anatomy \
  --eval-scene0 \
  --metadata-dir "${METADATA_DIR}" \
  --vbench2-cache-dir "${BENCH_ROOT}/vbench2" \
  --human-identity-arcface-weight "${BENCH_ROOT}/vbench2/arcface/resnet18_110.pth" \
  --human-clothes-llava-dir "${BENCH_ROOT}/vbench2/lmms-lab/LLaVA-Video-7B-Qwen2" \
  "${IDS_FLAGS[@]}" \
  "${RESUME_FLAGS[@]}"

conda_deactivate
conda_activate "${ENV_VBENCH}"

python scripts/compute_background_quality.py \
  --scene-transitions-json "${RUN_DIR}/scene_transitions.json" \
  --frames-root "${RUN_DIR}/frames" \
  --metadata-dir "${METADATA_DIR}" \
  --output-dir "${RUN_DIR}/background_quality_scene1" \
  --sub-clip-dir "${SUB_CLIP_DIR}" \
  --vbench-cache-dir "${BENCH_ROOT}/vbench" \
  "${IDS_FLAGS[@]}" \
  "${RESUME_FLAGS[@]}"

conda_deactivate
conda_activate "${ENV_VBENCH2}"

python scripts/evaluate_background_semantics.py \
  --scene-transitions-json "${RUN_DIR}/scene_transitions.json" \
  --frames-root "${RUN_DIR}/frames" \
  --metadata-dir "${METADATA_DIR}" \
  --output-dir "${RUN_DIR}/background_semantics" \
  --sub-clip-dir "${SUB_CLIP_DIR}" \
  --cache-dir "${BENCH_ROOT}/vbench2" \
  --llava-dir "${BENCH_ROOT}/vbench2/lmms-lab/LLaVA-Video-7B-Qwen2" \
  "${IDS_FLAGS[@]}" \
  "${RESUME_FLAGS[@]}"

conda_deactivate
conda_activate "${ENV_LIPSYNC}"

python scripts/evaluate_lipsync_syncnet.py \
  --videos_glob "${RUN_DIR}/videos/${VIDEO_GLOB}" \
  --frames_root "${RUN_DIR}/frames" \
  --out_json "${RUN_DIR}/lipsync_syncnet_align224.json" \
  --work_dir "${RUN_DIR}/tmp_syncnet" \
  --export_crops_dir "${RUN_DIR}/crops_224" \
  --metadata_dir "${METADATA_DIR}" \
  --dataset_root "${BENCH_ROOT}" \
  --syncnet_ckpt "${BENCH_ROOT}/checkpoints/syncnet_v2.model" \
  --dlib_landmark_model "${BENCH_ROOT}/checkpoints/shape_predictor_68_face_landmarks.dat" \
  --crop_method align224 \
  "${IDS_FLAGS[@]}" \
  "${RESUME_FLAGS[@]}"

GATHER_IDS_FLAGS=()
if [[ -n "${IDS_ARG}" && -z "${RESUME_MODE}" ]]; then
  GATHER_IDS_FLAGS=(--ids "${IDS_ARG}")
fi
python scripts/gather_evaluation_summary.py \
  --run-dir "${RUN_DIR}" \
  --output "${RUN_DIR}/evaluation_summary.json" \
  "${GATHER_IDS_FLAGS[@]}"
