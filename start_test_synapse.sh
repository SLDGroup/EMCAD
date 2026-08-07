#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"


LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"


CONDA_BASE="/base/mambaforge"
CONDA_ENV_NAME="sld_emcad_251"


source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1


IMG_SIZE=256
DATASET="Synapse"
VOLUME_PATH="../data/Synapse/test_vol_h5"
LIST_PATH="../data/Synapse/lists/lists_Synapse"

#本项目ckpt无需作为启动参数，项目中写死了位置，best.pth文件必须放在项目的根目录下
#CKPT="${PROJECT_DIR}/model_pth/SimMPNetSynapse....."

SEED=2222
#TS="$(date +%F_%H%M)"	
TS="$(date +%F_%H%M%S)"
RAND="$(head -c 6 /dev/urandom | od -An -tx1 | tr -d ' \n')"
LOG_FILE="${LOG_DIR}/test_${DATASET}__img${IMG_SIZE}_${TS}.log"

RUN_ID="test_${DATASET}_${TS}_gpu${CUDA_VISIBLE_DEVICES}_SEED${SEED}_RAND${RAND}"
PID_FILE="${RUN_ID}.pid"
echo "[INFO] PROJECT_DIR=${PROJECT_DIR}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] VOLUME_PATH=${VOLUME_PATH}" | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}"  | tee -a "${LOG_FILE}" > /dev/null
echo "[INFO] RUN_ID=${RUN_ID}"

# 硬性检查：路径不存在就直接退出（比跑半天才发现强太多）  ||：逻辑或，含义是：如果左边失败，就执行右边
#test -f "${CKPT}" || { echo "[ERROR] CKPT not found: ${CKPT}" | tee -a "${LOG_FILE}"; exit 1; }
test -d "${VOLUME_PATH}" || { echo "[ERROR] VOLUME_PATH not found: ${VOLUME_PATH}" | tee -a "${LOG_FILE}"; exit 1; }

nohup env RUN_ID="${RUN_ID}"   python test_synapse.py \
  --volume_path "${VOLUME_PATH}" \
  --dataset "${DATASET}" \
  --img_size "${IMG_SIZE}" \
  --list_dir "${LIST_PATH}" \
  >> "${LOG_FILE}" 2>&1 &

PID=$!
echo "[INFO] PID=${PID}"
echo "${PID}" > "${PID_FILE}"

