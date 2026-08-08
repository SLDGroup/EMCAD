#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

RUN_ID="${1:-${RUN_ID:-}}"

[ -n "${RUN_ID}" ] || {
  echo "[ERROR] usage: bash stop_test_polyp.sh <RUN_ID>"
  exit 1
}

PID_FILE="${PROJECT_DIR}/${RUN_ID}.pid"

[ -f "${PID_FILE}" ] || {
  echo "[ERROR] PID file not found: ${PID_FILE}"
  exit 2
}

PID="$(tr -d ' \n' < "${PID_FILE}")"

[[ "${PID}" =~ ^[0-9]+$ ]] || {
  echo "[ERROR] invalid PID in ${PID_FILE}: ${PID}"
  exit 3
}

kill -0 "${PID}" 2>/dev/null || {
  rm -f "${PID_FILE}"
  exit 0
}

ENV_RUN_ID=""

if [[ -r "/proc/${PID}/environ" ]]; then
  ENV_RUN_ID="$(tr '\0' '\n' < "/proc/${PID}/environ" |
    sed -n 's/^RUN_ID=//p' |
    head -n 1 || true)"
fi

[ "${ENV_RUN_ID}" = "${RUN_ID}" ] || {
  echo "[ERROR] RUN_ID does not match PID ${PID}"
  exit 4
}

kill -TERM "${PID}" 2>/dev/null || {
  rm -f "${PID_FILE}"
  exit 0
}

for _ in {1..10}; do
  kill -0 "${PID}" 2>/dev/null || {
    rm -f "${PID_FILE}"
    echo "[INFO] stopped test_polyp.py: RUN_ID=${RUN_ID} PID=${PID}"
    exit 0
  }
  sleep 1
done

kill -KILL "${PID}" 2>/dev/null || true
rm -f "${PID_FILE}"

echo "[INFO] force-stopped test_polyp.py: RUN_ID=${RUN_ID} PID=${PID}"