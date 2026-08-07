#!/usr/bin/env bash
set -Eeuo pipefail

# Edit this block for the server. Every value can also be overridden as an
# environment variable, e.g. BATCH_SIZE=12 bash start_train_acdc.sh.
CONDA_BASE="${CONDA_BASE:-$HOME/lzqEnvs/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sld_emcad}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
IMG_SIZE="${IMG_SIZE:-224}"
BATCH_SIZE="${BATCH_SIZE:-6}"
MAX_EPOCHS="${MAX_EPOCHS:-150}"
BASE_LR="${BASE_LR:-1e-4}"
SUPERVISION="${SUPERVISION:-deep_supervision}"
NUM_WORKERS="${NUM_WORKERS:-4}"
AMP="${AMP:-1}"
SEED="${SEED:-2222}"
RUN_NAME="${RUN_NAME:-acdc_$(date +%Y%m%d_%H%M%S)_gpu${CUDA_DEVICE}}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # This is skipped when the caller already provides a usable Python.
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export PYTHONUNBUFFERED=1
ROOT_PATH="${PROJECT_DIR}/data/ACDC"
LIST_DIR="${ROOT_PATH}/lists/lists_ACDC"
OUTPUT_DIR="${PROJECT_DIR}/model_pth"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

test -f "${LIST_DIR}/train.txt" || { echo "[ERROR] missing ${LIST_DIR}/train.txt"; exit 1; }
test -f "${LIST_DIR}/valid.txt" || { echo "[ERROR] missing ${LIST_DIR}/valid.txt"; exit 1; }
test -f "${PROJECT_DIR}/pretrained_pth/pvt/pvt_v2_b2.pth" || {
  echo "[ERROR] missing pretrained_pth/pvt/pvt_v2_b2.pth (or set --no_pretrain)"; exit 1;
}

LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
PID_FILE="${RUN_NAME}.pid"
ARGS=(
  --root_path "${ROOT_PATH}"
  --list_dir "${LIST_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --run_name "${RUN_NAME}"
  --img_size "${IMG_SIZE}"
  --batch_size "${BATCH_SIZE}"
  --max_epochs "${MAX_EPOCHS}"
  --base_lr "${BASE_LR}"
  --supervision "${SUPERVISION}"
  --num_workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --device auto
)
if [[ "${AMP}" == "1" ]]; then ARGS+=(--amp); else ARGS+=(--no-amp); fi

echo "[INFO] run=${RUN_NAME} log=${LOG_FILE}"
nohup env RUN_NAME="${RUN_NAME}" "${PYTHON_BIN}" -u train_acdc.py "${ARGS[@]}" \
  >> "${LOG_FILE}" 2>&1 < /dev/null &
PID=$!
echo "${PID}" > "${PID_FILE}"
echo "[INFO] PID=${PID}; tail -f ${LOG_FILE}"

