#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

RUN_ID="${1:-${RUN_ID:-}}"

if [[ -z "${RUN_ID}" ]]; then
  echo "[ERROR] usage: bash stop_train_busi.sh <RUN_ID>"
  exit 1
fi

PID_FILE="${PROJECT_DIR}/${RUN_ID}.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "[ERROR] PID file not found: ${PID_FILE}"
  exit 2
fi

PID="$(tr -d ' \n' < "${PID_FILE}")"

if [[ ! "${PID}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] invalid PID in ${PID_FILE}: ${PID}"
  exit 3
fi

if ! kill -0 "${PID}" 2>/dev/null; then
  rm -f "${PID_FILE}"
  echo "[INFO] process already ended; PID file removed"
  exit 0
fi

ENV_RUN_ID=""

if [[ -r "/proc/${PID}/environ" ]]; then
  ENV_RUN_ID="$(
    tr '\0' '\n' < "/proc/${PID}/environ" |
      sed -n 's/^RUN_ID=//p' |
      head -n 1 || true
  )"
fi

if [[ "${ENV_RUN_ID}" != "${RUN_ID}" ]]; then
  echo "[ERROR] RUN_ID does not match PID ${PID}"
  exit 4
fi

kill -TERM "${PID}" 2>/dev/null || {
  rm -f "${PID_FILE}"
  exit 0
}

for _ in {1..10}; do
  if ! kill -0 "${PID}" 2>/dev/null; then
    rm -f "${PID_FILE}"
    echo "[INFO] stopped train_busi.py: RUN_ID=${RUN_ID} PID=${PID}"
    exit 0
  fi

  sleep 1
done

kill -KILL "${PID}" 2>/dev/null || true
rm -f "${PID_FILE}"

echo "[INFO] force-stopped train_busi.py: RUN_ID=${RUN_ID} PID=${PID}"