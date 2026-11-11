# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV VENV_DIR=/opt/venv
ENV PATH=${VENV_DIR}/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST=12.0
ENV FORCE_CUDA=1

SHELL ["/bin/bash", "-lc"]
WORKDIR /app

# ----------------------------
# System dependencies
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git ffmpeg \
    build-essential pkg-config \
    python3 python3-dev python3-pip python3-venv \
    libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Python venv + pip
# ----------------------------
RUN python3 -m venv ${VENV_DIR} \
 && pip install --upgrade pip setuptools wheel

# ----------------------------
# Copy project + install deps
# ----------------------------
COPY . /app

RUN mkdir -p /app/weights \
 && curl -L -o /app/weights/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth \
    https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth

RUN MMCV_WITH_OPS=1 \
    CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++ \
    pip install -r requirements.txt --no-binary mmcv --no-build-isolation

# ----------------------------
# Runtime
# ----------------------------
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
ENV PYTHONUNBUFFERED=1

CMD ["bash", "-lc", "python server.py"]
