#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${ROOT_DIR}/weights"

mkdir -p "${WEIGHTS_DIR}"

RTMPOSE_URL="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth"
RTMPOSE_DST="${WEIGHTS_DIR}/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth"

echo "Downloading RTMPose weights..."
curl -L -o "${RTMPOSE_DST}" "${RTMPOSE_URL}"
echo "Saved: ${RTMPOSE_DST}"
