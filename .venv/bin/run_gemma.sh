#!/bin/bash

MODEL="google/gemma-4-26B-A4B-it"
PORT=8000
CACHE_DIR="${HOME}/.cache/huggingface"

docker pull vllm/vllm-openai:nightly

docker run --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p ${PORT}:${PORT} \
    -v "${CACHE_DIR}:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    --entrypoint python3 \
    vllm/vllm-openai:nightly \
    -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host 0.0.0.0 \
    --port ${PORT} \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser functiongemma
