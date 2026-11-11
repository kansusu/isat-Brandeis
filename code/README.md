# multimodal-attention-analyzer
A multimodal analyzer of attention and posture.

## Requirements
- NVIDIA GPU + recent driver
- CUDA runtime compatible with the PyTorch build
- ffmpeg

## Setup (local)
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-binary mmcv --no-build-isolation
```

## Download model weights
```bash
./scripts/download_weights.sh
```

## Run
```bash
python server.py
```

Optional overrides:
```bash
export MMPOSE_CONFIG=/path/to/rtmpose-config.py
export MMPOSE_CHECKPOINT=/path/to/rtmpose-weights.pth
```

## Docker (GPU)
```bash
docker build -t mma-gpu .
docker run --gpus all --rm -it \
  -v "$PWD":/app \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e AWS_DEFAULT_REGION \
  mma-gpu
```
