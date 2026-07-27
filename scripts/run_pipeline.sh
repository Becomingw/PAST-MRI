#!/usr/bin/env bash
set -Eeuo pipefail

ACTION="${1:-help}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
GPU_ID="${GPU_ID:-0}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/processed}"
export PAST_MRI_DATA_ROOT="${PAST_MRI_DATA_ROOT:-${DATA_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs}"
STAGE1_CONFIG="${STAGE1_CONFIG:-${ROOT}/configs/stage1_pretrain.yaml}"
STAGE2_CONFIG="${STAGE2_CONFIG:-${ROOT}/configs/stage2_translation.yaml}"
STAGE1_DIR="${STAGE1_DIR:-${OUTPUT_ROOT}/stage1}"
STAGE2_DIR="${STAGE2_DIR:-${OUTPUT_ROOT}/stage2}"
STAGE1_INITIALIZER="${STAGE1_INITIALIZER:-${STAGE1_DIR}/terminal_encoder_middle_core_initializer.ckpt}"
STAGE2_MANIFEST="${STAGE2_MANIFEST:-${DATA_ROOT}/manifests/stage2_train.jsonl}"
STAGE2_EMA="${STAGE2_EMA:-${STAGE2_DIR}/ema_step_040000.ckpt}"
PREDICTIONS="${PREDICTIONS:-${OUTPUT_ROOT}/predictions}"
SUBMISSION="${SUBMISSION:-${OUTPUT_ROOT}/past_mri_submission.zip}"

cd "${ROOT}"

checkpoint_step() {
  "${PYTHON}" -c 'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("step", -1)))' "$1"
}

prepare_data() {
  : "${UNPAIRED_ROOT:?Set UNPAIRED_ROOT to the retrospective NIfTI root}"
  : "${PAIRED_ROOT:?Set PAIRED_ROOT to the paired NIfTI root}"
  "${PYTHON}" -m past_mri.prepare_data \
    --unpaired-root "${UNPAIRED_ROOT}" \
    --paired-root "${PAIRED_ROOT}" \
    --output-root "${DATA_ROOT}"
}

run_stage1() {
  mkdir -p "${STAGE1_DIR}"
  if [[ -s "${STAGE1_DIR}/last.ckpt" ]]; then
    step="$(checkpoint_step "${STAGE1_DIR}/last.ckpt")"
    if (( step >= 40000 )); then
      echo "[stage1] complete at step ${step}"
      return
    fi
    resume=(--resume "${STAGE1_DIR}/last.ckpt")
  else
    resume=()
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" -m past_mri.train_stage1 \
    --config "${STAGE1_CONFIG}" \
    --train-manifest "${DATA_ROOT}/manifests/stage1_train.jsonl" \
    --val-manifest "${DATA_ROOT}/manifests/stage1_val.jsonl" \
    --work-dir "${STAGE1_DIR}" \
    --device cuda \
    "${resume[@]}" \
    2>&1 | tee -a "${STAGE1_DIR}/train.outer.log"
}

run_stage2() {
  test -s "${STAGE1_INITIALIZER}"
  test -s "${STAGE2_MANIFEST}"
  mkdir -p "${STAGE2_DIR}"
  if [[ -s "${STAGE2_DIR}/last.ckpt" ]]; then
    step="$(checkpoint_step "${STAGE2_DIR}/last.ckpt")"
    if (( step >= 40000 )); then
      echo "[stage2] complete at step ${step}"
      return
    fi
    initialization=(--resume "${STAGE2_DIR}/last.ckpt")
  else
    initialization=(--init-model "${STAGE1_INITIALIZER}" --init-model-partial)
  fi
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" -m past_mri.train_stage2 \
    --config "${STAGE2_CONFIG}" \
    --manifest "${STAGE2_MANIFEST}" \
    --work-dir "${STAGE2_DIR}" \
    --device cuda \
    --max-steps 40000 \
    --assert-power batch_ge=16 lr_no_early_decay=true seeds_ge=1 ema_decay=0.9999 \
    "${initialization[@]}" \
    2>&1 | tee -a "${STAGE2_DIR}/train.outer.log"
}

run_inference() {
  : "${VALIDATION_ROOT:?Set VALIDATION_ROOT to the source-only validation NIfTI root}"
  test -s "${STAGE2_EMA}"
  for contrast in T1W T2W T2FLAIR; do
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" -m past_mri.infer \
      --config "${STAGE2_CONFIG}" \
      --checkpoint "${STAGE2_EMA}" \
      --input-root "${VALIDATION_ROOT}" \
      --out-root "${PREDICTIONS}" \
      --contrast "${contrast}" \
      --device cuda \
      --skip-existing
  done
}

package_predictions() {
  : "${VALIDATION_ROOT:?Set VALIDATION_ROOT for affine verification}"
  "${PYTHON}" -m scripts.package_submission \
    --predictions "${PREDICTIONS}" \
    --output "${SUBMISSION}" \
    --reference-input-root "${VALIDATION_ROOT}"
}

case "${ACTION}" in
  prepare)
    prepare_data
    ;;
  stage1)
    run_stage1
    ;;
  stage2)
    run_stage2
    ;;
  infer)
    run_inference
    ;;
  package)
    package_predictions
    ;;
  all)
    prepare_data
    run_stage1
    run_stage2
    run_inference
    package_predictions
    ;;
  *)
    echo "usage: $0 {prepare|stage1|stage2|infer|package|all}"
    ;;
esac
