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
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_EPOCHS="${MAX_EPOCHS:-400}"
BASE_LR="${BASE_LR:-1e-4}"
SUPERVISION="${SUPERVISION:-mutation}"
NUM_WORKERS="${NUM_WORKERS:-0}"
N_GPU="${N_GPU:-1}"
DETERMINISTIC="${DETERMINISTIC:-1}"
SEED="${SEED:-2222}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VALID_VOLUMES="${MAX_VALID_VOLUMES:-0}"

LIST_DIR="${PROJECT_DIR}/../data/ACDC/lists/lists_ACDC"
ROOT_PATH="${PROJECT_DIR}/../data/ACDC"
OUTPUT_DIR="${PROJECT_DIR}/model_pth/ACDC"

test -d "${ROOT_PATH}/train" || { echo "[ERROR] ROOT_PATH/train not found: ${ROOT_PATH}/train"; exit 1; }
test -d "${ROOT_PATH}/valid" || { echo "[ERROR] ROOT_PATH/valid not found: ${ROOT_PATH}/valid"; exit 1; }
test -f "${LIST_DIR}/train.txt" || { echo "[ERROR] train list not found: ${LIST_DIR}/train.txt"; exit 1; }
test -f "${LIST_DIR}/valid.txt" || { echo "[ERROR] valid list not found: ${LIST_DIR}/valid.txt"; exit 1; }
test -f "${PROJECT_DIR}/pretrained_pth/pvt/pvt_v2_b2.pth" || {
  echo "[ERROR] pretrained model not found: ${PROJECT_DIR}/pretrained_pth/pvt/pvt_v2_b2.pth"
  echo "[ERROR] This Synapse-style launcher uses the pretrained EMCAD encoder."
  exit 1
}

TS="$(date +%F_%H%M%S)"
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"
LOG_FILE="${LOG_DIR}/train_${DATASET}__imgSize${IMG_SIZE}_batchSize${BATCH_SIZE}_lr${BASE_LR}_epo${MAX_EPOCHS}_${TS}.log"
RUN_ID="train_${DATASET}_${TS}_gpu${CUDA_VISIBLE_DEVICES}_SEED${SEED}_RAND${RAND}"
PID_FILE="${RUN_ID}.pid"

echo "[INFO] PROJECT_DIR=${PROJECT_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] DATASET=${DATASET}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] ROOT_PATH=${ROOT_PATH}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] LIST_DIR=${LIST_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] IMG_SIZE=${IMG_SIZE} NUM_CLASSES=4" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] BATCH_SIZE=${BATCH_SIZE} MAX_EPOCHS=${MAX_EPOCHS} BASE_LR=${BASE_LR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] SUPERVISION=${SUPERVISION} NUM_WORKERS=${NUM_WORKERS}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] SEED=${SEED} N_GPU=${N_GPU}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}"

echo "---------------------------ready to train---------------------------------" | tee -a "${LOG_FILE}" > /dev/null

nohup env RUN_ID="${RUN_ID}" "${PYTHON_BIN}" -u train_ACDC.py \
  --root_path "${ROOT_PATH}" \
  --list_dir "${LIST_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_ID}" \
  --encoder pvt_v2_b2 \
  --kernel_sizes 1 3 5 \
  --expansion_factor 2 \
  --lgag_ks 3 \
  --activation_mscb relu6 \
  --supervision "${SUPERVISION}" \
  --img_size "${IMG_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_epochs "${MAX_EPOCHS}" \
  --base_lr "${BASE_LR}" \
  --num_workers "${NUM_WORKERS}" \
  --n_gpu "${N_GPU}" \
  --deterministic "${DETERMINISTIC}" \
  --seed "${SEED}" \
  --max_train_batches "${MAX_TRAIN_BATCHES}" \
  --max_valid_volumes "${MAX_VALID_VOLUMES}" \
  --device auto \
  >> "${LOG_FILE}" 2>&1 < /dev/null &

PID=$!
echo "[INFO] PID=${PID}"
echo "${PID}" > "${PID_FILE}"
echo "[INFO] PID_FILE=${PID_FILE}"
echo "[INFO] LOG_FILE=${LOG_FILE}"
