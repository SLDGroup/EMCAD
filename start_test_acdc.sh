#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

CONDA_BASE="${CONDA_BASE:-/base/mambaforge}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sld_emcad_251}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
export PYTHONUNBUFFERED=1

DATASET="ACDC"
IMG_SIZE="${IMG_SIZE:-224}"
NUM_WORKERS="${NUM_WORKERS:-0}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
Z_SPACING="${Z_SPACING:-10.0}"
MAX_CASES="${MAX_CASES:-0}"
SEED="${SEED:-2222}"

LIST_DIR="${PROJECT_DIR}/../data/ACDC/lists/lists_ACDC"
ROOT_PATH="${PROJECT_DIR}/../data/ACDC"
CKPT="${CKPT:-}"

if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "[ERROR] CKPT is missing or does not exist: ${CKPT}"
  echo "[ERROR] Example: CKPT=/absolute/path/to/best.pth bash start_test_acdc.sh"
  exit 1
fi
test -d "${ROOT_PATH}/test" || { echo "[ERROR] test directory not found: ${ROOT_PATH}/test"; exit 1; }
test -f "${LIST_DIR}/test.txt" || { echo "[ERROR] test list not found: ${LIST_DIR}/test.txt"; exit 1; }

CKPT_DIR="$(cd "$(dirname "${CKPT}")" && pwd)"
TEST_SAVE_DIR="${CKPT_DIR}/predictions"
OUTPUT_CSV="${CKPT_DIR}/test_metrics.csv"

TS="$(date +%F_%H%M%S)"
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"
LOG_FILE="${LOG_DIR}/test_${DATASET}__img${IMG_SIZE}_${TS}.log"
RUN_ID="test_${DATASET}_${TS}_gpu${CUDA_VISIBLE_DEVICES}_SEED${SEED}_RAND${RAND}"
PID_FILE="${RUN_ID}.pid"

echo "[INFO] PROJECT_DIR=${PROJECT_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] DATASET=${DATASET}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] ROOT_PATH=${ROOT_PATH}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] LIST_DIR=${LIST_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] VOLUME_PATH=${ROOT_PATH}/test" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] CKPT=${CKPT}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] IMG_SIZE=${IMG_SIZE} NUM_CLASSES=4" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] INFERENCE_BATCH_SIZE=${INFERENCE_BATCH_SIZE} MAX_CASES=${MAX_CASES}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}"

echo "---------------------------ready to test----------------------------------" | tee -a "${LOG_FILE}" > /dev/null

nohup env RUN_ID="${RUN_ID}" "${PYTHON_BIN}" -u test_ACDC.py \
  --checkpoint "${CKPT}" \
  --root_path "${ROOT_PATH}" \
  --list_dir "${LIST_DIR}" \
  --output_dir "${TEST_SAVE_DIR}" \
  --output_csv "${OUTPUT_CSV}" \
  --encoder pvt_v2_b2 \
  --kernel_sizes 1 3 5 \
  --expansion_factor 2 \
  --lgag_ks 3 \
  --activation_mscb relu6 \
  --img_size "${IMG_SIZE}" \
  --inference_batch_size "${INFERENCE_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --z_spacing "${Z_SPACING}" \
  --seed "${SEED}" \
  --max_cases "${MAX_CASES}" \
  --device auto \
  --save_nii \
  --save_npz \
  >> "${LOG_FILE}" 2>&1 < /dev/null &

PID=$!
echo "[INFO] PID=${PID}"
echo "${PID}" > "${PID_FILE}"
echo "[INFO] PID_FILE=${PID_FILE}"
echo "[INFO] LOG_FILE=${LOG_FILE}"
echo "[INFO] OUTPUT_CSV=${OUTPUT_CSV}"
