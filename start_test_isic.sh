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

DATASET="ISIC"
DATASET_NAME="${DATASET_NAME:-ISIC2018}"
SPLIT="${SPLIT:-test}"

INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
THRESHOLD="${THRESHOLD:-0.5}"
MAX_CASES="${MAX_CASES:-0}"
DETERMINISTIC="${DETERMINISTIC:-1}"
SEED="${SEED:-2222}"
DEVICE="${DEVICE:-auto}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/../data/isic/target}"
CKPT="${CKPT:-}"

case "${DATASET_NAME}" in
  ISIC2017|ISIC2018)
    ;;
  *)
    echo "[ERROR] DATASET_NAME must be ISIC2017 or ISIC2018"
    exit 1
    ;;
esac

case "${SPLIT}" in
  val|test)
    ;;
  *)
    echo "[ERROR] SPLIT must be val or test"
    exit 1
    ;;
esac

if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "[ERROR] CKPT is missing or does not exist: ${CKPT}"
  echo "[ERROR] Example:"
  echo "CKPT=/absolute/path/to/best.pth DATASET_NAME=ISIC2018 bash start_test_isic.sh"
  exit 1
fi

test -d "${DATA_ROOT}/${DATASET_NAME}/${SPLIT}/images" || {
  echo "[ERROR] images directory not found:"
  echo "${DATA_ROOT}/${DATASET_NAME}/${SPLIT}/images"
  exit 1
}

test -d "${DATA_ROOT}/${DATASET_NAME}/${SPLIT}/masks" || {
  echo "[ERROR] masks directory not found:"
  echo "${DATA_ROOT}/${DATASET_NAME}/${SPLIT}/masks"
  exit 1
}

CKPT_DIR="$(cd "$(dirname "${CKPT}")" && pwd)"
CKPT="${CKPT_DIR}/$(basename "${CKPT}")"
CONFIG_FILE="${CKPT_DIR}/config.json"

test -f "${CONFIG_FILE}" || {
  echo "[ERROR] checkpoint config not found:"
  echo "${CONFIG_FILE}"
  exit 1
}

TEST_SAVE_DIR="${TEST_SAVE_DIR:-${CKPT_DIR}/${SPLIT}_${DATASET_NAME}_outputs}"
OUTPUT_CSV="${OUTPUT_CSV:-${TEST_SAVE_DIR}/${SPLIT}_metrics.csv}"

TS="$(date +%F_%H%M%S)"
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"

LOG_FILE="${LOG_DIR}/test_${DATASET}_${DATASET_NAME}_${SPLIT}_${TS}.log"
RUN_ID="test_${DATASET}_${DATASET_NAME}_${SPLIT}_${TS}_gpu${CUDA_VISIBLE_DEVICES}_SEED${SEED}_RAND${RAND}"
PID_FILE="${PROJECT_DIR}/${RUN_ID}.pid"

echo "[INFO] PROJECT_DIR=${PROJECT_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] DATASET_NAME=${DATASET_NAME}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] SPLIT=${SPLIT}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] DATA_ROOT=${DATA_ROOT}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] CKPT=${CKPT}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] CONFIG_FILE=${CONFIG_FILE}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] TEST_SAVE_DIR=${TEST_SAVE_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] OUTPUT_CSV=${OUTPUT_CSV}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] THRESHOLD=${THRESHOLD}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}"

echo "---------------------------ready to test----------------------------------" | tee -a "${LOG_FILE}" > /dev/null

nohup env RUN_ID="${RUN_ID}" "${PYTHON_BIN}" -u test_isic.py \
  --checkpoint "${CKPT}" \
  --data_root "${DATA_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --split "${SPLIT}" \
  --output_dir "${TEST_SAVE_DIR}" \
  --output_csv "${OUTPUT_CSV}" \
  --inference_batch_size "${INFERENCE_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --threshold "${THRESHOLD}" \
  --seed "${SEED}" \
  --deterministic "${DETERMINISTIC}" \
  --max_cases "${MAX_CASES}" \
  --device "${DEVICE}" \
  --save_probabilities \
  >> "${LOG_FILE}" 2>&1 < /dev/null &

PID=$!

echo "[INFO] PID=${PID}"
echo "${PID}" > "${PID_FILE}"
echo "[INFO] PID_FILE=${PID_FILE}"
echo "[INFO] LOG_FILE=${LOG_FILE}"
echo "[INFO] OUTPUT_CSV=${OUTPUT_CSV}"
echo "[INFO] PREDICTIONS=${TEST_SAVE_DIR}"