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

DATASET="BUSI"
DATASET_NAME="${DATASET_NAME:-BUSI}"
SPLIT="${SPLIT:-test}"

INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_CASES="${MAX_CASES:-0}"
DETERMINISTIC="${DETERMINISTIC:-1}"
SEED="${SEED:-2222}"
DEVICE="${DEVICE:-auto}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/../data/busi/target}"
CKPT="${CKPT:-}"

if [[ "${DATASET_NAME}" != "BUSI" ]]; then
  echo "[ERROR] DATASET_NAME must be BUSI"
  exit 1
fi

case "${SPLIT}" in
  val|test)
    ;;
  *)
    echo "[ERROR] SPLIT must be val or test"
    exit 1
    ;;
esac

if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "[ERROR] CKPT is missing or does not exist:"
  echo "${CKPT}"
  echo "[ERROR] Example:"
  echo 'CKPT="/absolute/path/to/best.pth" bash start_test_busi.sh'
  exit 1
fi

for REQUIRED in \
  "${DATA_ROOT}/BUSI/${SPLIT}/images" \
  "${DATA_ROOT}/BUSI/${SPLIT}/masks" \
  "${DATA_ROOT}/BUSI/manifest.csv" \
  "${DATA_ROOT}/BUSI/split_summary.json"
do
  if [[ ! -e "${REQUIRED}" ]]; then
    echo "[ERROR] required BUSI path not found:"
    echo "${REQUIRED}"
    exit 1
  fi
done

CKPT_DIR="$(cd "$(dirname "${CKPT}")" && pwd)"
CKPT="${CKPT_DIR}/$(basename "${CKPT}")"
CONFIG_FILE="${CKPT_DIR}/config.json"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "[ERROR] checkpoint config not found:"
  echo "${CONFIG_FILE}"
  echo "[ERROR] best.pth and config.json must be in the same directory"
  exit 1
fi

TEST_SAVE_DIR="${TEST_SAVE_DIR:-${CKPT_DIR}/${SPLIT}_BUSI_outputs}"
OUTPUT_CSV="${OUTPUT_CSV:-${TEST_SAVE_DIR}/${SPLIT}_metrics.csv}"

TS="$(date +%F_%H%M%S)"
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"

RUN_ID="test_${DATASET}_${SPLIT}_${TS}_gpu${CUDA_VISIBLE_DEVICES}_SEED${SEED}_RAND${RAND}"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"
PID_FILE="${PROJECT_DIR}/${RUN_ID}.pid"

echo "[INFO] PROJECT_DIR=${PROJECT_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] DATA_ROOT=${DATA_ROOT}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] SPLIT=${SPLIT}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] CKPT=${CKPT}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] CONFIG_FILE=${CONFIG_FILE}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] TEST_SAVE_DIR=${TEST_SAVE_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] OUTPUT_CSV=${OUTPUT_CSV}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] MODEL_CONFIG=loaded_from_checkpoint_config" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] MAX_CASES=${MAX_CASES}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}" | tee -a "${LOG_FILE}" > /dev/null

echo "[INFO] RUN_ID=${RUN_ID}"
echo "---------------------------ready to test----------------------------------" | tee -a "${LOG_FILE}" > /dev/null

nohup env RUN_ID="${RUN_ID}" "${PYTHON_BIN}" -u test_busi.py \
  --checkpoint "${CKPT}" \
  --data_root "${DATA_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --split "${SPLIT}" \
  --output_dir "${TEST_SAVE_DIR}" \
  --output_csv "${OUTPUT_CSV}" \
  --inference_batch_size "${INFERENCE_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}" \
  --deterministic "${DETERMINISTIC}" \
  --max_cases "${MAX_CASES}" \
  --device "${DEVICE}" \
  --save_probabilities \
  >> "${LOG_FILE}" 2>&1 < /dev/null &

PID=$!

echo "${PID}" > "${PID_FILE}"

echo "[INFO] PID=${PID}"
echo "[INFO] PID_FILE=${PID_FILE}"
echo "[INFO] LOG_FILE=${LOG_FILE}"
echo "[INFO] OUTPUT_CSV=${OUTPUT_CSV}"
echo "[INFO] PREDICTIONS=${TEST_SAVE_DIR}/predictions"
echo "[INFO] PROBABILITIES=${TEST_SAVE_DIR}/probabilities"