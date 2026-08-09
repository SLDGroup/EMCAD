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

IMG_SIZE="${IMG_SIZE:-352}"
BATCH_SIZE="${BATCH_SIZE:-16}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
BASE_LR="${BASE_LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
CLIP="${CLIP:-0.5}"
THRESHOLD="${THRESHOLD:-0.5}"
SCHEDULER="${SCHEDULER:-constant}"
MIN_LR="${MIN_LR:-1e-6}"

NUM_WORKERS="${NUM_WORKERS:-0}"
N_GPU="${N_GPU:-1}"
DETERMINISTIC="${DETERMINISTIC:-1}"
SEED="${SEED:-2222}"
VALIDATE_EVERY="${VALIDATE_EVERY:-1}"
SAVE_EVERY="${SAVE_EVERY:-50}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VALID_CASES="${MAX_VALID_CASES:-0}"
DEVICE="${DEVICE:-auto}"

ENCODER="${ENCODER:-pvt_v2_b2}"
EXPANSION_FACTOR="${EXPANSION_FACTOR:-2}"
LGAG_KS="${LGAG_KS:-3}"
ACTIVATION_MSCB="${ACTIVATION_MSCB:-relu6}"
SUPERVISION="${SUPERVISION:-paper}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/../data/isic/target}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/model_pth/ISIC}"
PRETRAINED_DIR="${PRETRAINED_DIR:-${PROJECT_DIR}/pretrained_pth/pvt}"

case "${DATASET_NAME}" in
  ISIC2017|ISIC2018)
    ;;
  *)
    echo "[ERROR] DATASET_NAME must be ISIC2017 or ISIC2018"
    exit 1
    ;;
esac

test -d "${DATA_ROOT}/${DATASET_NAME}/train/images" || {
  echo "[ERROR] train images not found:"
  echo "${DATA_ROOT}/${DATASET_NAME}/train/images"
  exit 1
}

test -d "${DATA_ROOT}/${DATASET_NAME}/train/masks" || {
  echo "[ERROR] train masks not found:"
  echo "${DATA_ROOT}/${DATASET_NAME}/train/masks"
  exit 1
}

test -d "${DATA_ROOT}/${DATASET_NAME}/val/images" || {
  echo "[ERROR] val images not found:"
  echo "${DATA_ROOT}/${DATASET_NAME}/val/images"
  exit 1
}

test -d "${DATA_ROOT}/${DATASET_NAME}/val/masks" || {
  echo "[ERROR] val masks not found:"
  echo "${DATA_ROOT}/${DATASET_NAME}/val/masks"
  exit 1
}

test -f "${DATA_ROOT}/${DATASET_NAME}/split_manifest.csv" || {
  echo "[ERROR] split_manifest.csv not found"
  exit 1
}

test -f "${DATA_ROOT}/${DATASET_NAME}/split_summary.json" || {
  echo "[ERROR] split_summary.json not found"
  exit 1
}

if [[ "${ENCODER}" == pvt_v2_* ]]; then
  test -f "${PRETRAINED_DIR}/${ENCODER}.pth" || {
    echo "[ERROR] pretrained model not found:"
    echo "${PRETRAINED_DIR}/${ENCODER}.pth"
    exit 1
  }
fi

mkdir -p "${OUTPUT_DIR}"

TS="$(date +%F_%H%M%S)"
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"

LOG_FILE="${LOG_DIR}/train_${DATASET}_${DATASET_NAME}__imgSize${IMG_SIZE}_batchSize${BATCH_SIZE}_lr${BASE_LR}_epo${MAX_EPOCHS}_${TS}.log"
RUN_ID="train_${DATASET}_${DATASET_NAME}_${TS}_gpu${CUDA_VISIBLE_DEVICES}_SEED${SEED}_RAND${RAND}"
PID_FILE="${PROJECT_DIR}/${RUN_ID}.pid"

echo "[INFO] PROJECT_DIR=${PROJECT_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] DATASET_NAME=${DATASET_NAME}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] DATA_ROOT=${DATA_ROOT}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] IMG_SIZE=${IMG_SIZE}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] BATCH_SIZE=${BATCH_SIZE} VAL_BATCH_SIZE=${VAL_BATCH_SIZE}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] MAX_EPOCHS=${MAX_EPOCHS} BASE_LR=${BASE_LR} WEIGHT_DECAY=${WEIGHT_DECAY}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] SUPERVISION=${SUPERVISION} NUM_WORKERS=${NUM_WORKERS}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] SEED=${SEED} DETERMINISTIC=${DETERMINISTIC}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}"

echo "---------------------------ready to train---------------------------------" | tee -a "${LOG_FILE}" > /dev/null

nohup env RUN_ID="${RUN_ID}" "${PYTHON_BIN}" -u train_isic.py \
  --data_root "${DATA_ROOT}" \
  --dataset_name "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_ID}" \
  --encoder "${ENCODER}" \
  --kernel_sizes 1 3 5 \
  --expansion_factor "${EXPANSION_FACTOR}" \
  --lgag_ks "${LGAG_KS}" \
  --activation_mscb "${ACTIVATION_MSCB}" \
  --supervision "${SUPERVISION}" \
  --pretrained_dir "${PRETRAINED_DIR}" \
  --img_size "${IMG_SIZE}" \
  --batch_size "${BATCH_SIZE}" \
  --val_batch_size "${VAL_BATCH_SIZE}" \
  --max_epochs "${MAX_EPOCHS}" \
  --base_lr "${BASE_LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --clip "${CLIP}" \
  --scheduler "${SCHEDULER}" \
  --min_lr "${MIN_LR}" \
  --scale_rates 0.75 1.0 1.25 \
  --num_workers "${NUM_WORKERS}" \
  --n_gpu "${N_GPU}" \
  --seed "${SEED}" \
  --deterministic "${DETERMINISTIC}" \
  --validate_every "${VALIDATE_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --threshold "${THRESHOLD}" \
  --max_train_batches "${MAX_TRAIN_BATCHES}" \
  --max_valid_cases "${MAX_VALID_CASES}" \
  --device "${DEVICE}" \
  >> "${LOG_FILE}" 2>&1 < /dev/null &

PID=$!

echo "[INFO] PID=${PID}"
echo "${PID}" > "${PID_FILE}"
echo "[INFO] PID_FILE=${PID_FILE}"
echo "[INFO] LOG_FILE=${LOG_FILE}"
echo "[INFO] RUN_OUTPUT=${OUTPUT_DIR}/${DATASET_NAME}/${RUN_ID}"