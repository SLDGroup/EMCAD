#!/usr/bin/env bash
set -Eeuo pipefail

# Required: CKPT=/absolute/path/to/model_pth/acdc_RUN/best.pth
CONDA_BASE="${CONDA_BASE:-$HOME/lzqEnvs/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sld_emcad}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CKPT="${CKPT:-}"
NUM_WORKERS="${NUM_WORKERS:-1}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "[ERROR] set CKPT=/absolute/path/to/best.pth"; exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export PYTHONUNBUFFERED=1
ROOT_PATH="${PROJECT_DIR}/data/ACDC"
LIST_DIR="${ROOT_PATH}/lists/lists_ACDC"
OUTPUT_CSV="$(dirname "${CKPT}")/test_metrics.csv"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"
RUN_NAME="test_acdc_$(date +%Y%m%d_%H%M%S)_gpu${CUDA_DEVICE}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
PID_FILE="${RUN_NAME}.pid"
ARGS=(
  --checkpoint "${CKPT}"
  --root_path "${ROOT_PATH}"
  --list_dir "${LIST_DIR}"
  --output_csv "${OUTPUT_CSV}"
  --num_workers "${NUM_WORKERS}"
  --inference_batch_size "${INFERENCE_BATCH_SIZE}"
  --device auto
)
if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
  ARGS+=(--save_predictions_dir "$(dirname "${CKPT}")/predictions")
fi

echo "[INFO] checkpoint=${CKPT} log=${LOG_FILE}"
nohup env RUN_NAME="${RUN_NAME}" "${PYTHON_BIN}" -u test_acdc.py "${ARGS[@]}" \
  >> "${LOG_FILE}" 2>&1 < /dev/null &
PID=$!
echo "${PID}" > "${PID_FILE}"
echo "[INFO] PID=${PID}; CSV=${OUTPUT_CSV}"
