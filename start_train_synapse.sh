#!/usr/bin/env bash
set -euo pipefail

#conda路径、环境名参数化
CONDA_BASE="/base/mambaforge"
CONDA_ENV_NAME="sld_emcad_251"


PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"


export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

DATASET="Synapse"
IMG_SIZE=256
BATCH_SIZE=20
MAX_EPOCHS=300
BASE_LR=1e-4
SUPERVISION="mutation"

LIST_DIR="${PROJECT_DIR}/../data/Synapse/lists/lists_Synapse"
ROOT_PATH="../data/Synapse/train_npz"
VOLUME_PATH="../data/Synapse/test_vol_h5"
DETERMINISTIC=1
SEED=2222

RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"
#TS="$(date +%F_%H%M)"
TS="$(date +%F_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_${DATASET}__imgSize${IMG_SIZE}_batchSize${BATCH_SIZE}_lr${BASE_LR}_epo${MAX_EPOCHS}_${TS}.log"
RUN_ID="train_${DATASET}_${TS}_gpu${CUDA_VISIBLE_DEVICES}_SEED${SEED}_RAND${RAND}"

PID_FILE="${RUN_ID}.pid"

echo "[INFO] PROJECT_DIR=${PROJECT_DIR}" | tee -a "${LOG_FILE}" > /dev/null 
echo "[INFO] DATASET=${DATASET}"  | tee -a "${LOG_FILE}" > /dev/null
#echo "[INFO] IMG_SIZE=${IMG_SIZE} NUM_CLASSES=${NUM_CLASSES}"  | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] BATCH_SIZE=${BATCH_SIZE} MAX_EPOCHS=${MAX_EPOCHS} BASE_LR=${BASE_LR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] LIST_DIR=${LIST_DIR}"  | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] SEED=${SEED}"  | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}"  | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}"

echo "---------------------------ready to train---------------------------------" | tee -a "${LOG_FILE}" > /dev/null

nohup env RUN_ID="${RUN_ID}" python train_synapse.py \
  --root_path "${ROOT_PATH}" \
  --list_dir "${LIST_DIR}" \
  --volume_path "${VOLUME_PATH}" \
  --dataset "${DATASET}" \
  --img_size "${IMG_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --max_epochs "${MAX_EPOCHS}" \
  --base_lr "${BASE_LR}" \
  --seed "${SEED}" \
  --deterministic "${DETERMINISTIC}" \
  >> "${LOG_FILE}" 2>&1 < /dev/null &

PID=$!
echo "[INFO] PID=${PID}"
echo "${PID}" > "${PID_FILE}"
