#!/bin/bash
# Head (rank 0): serve the API and drive both ranks. Run on the head Spark.
set -euo pipefail
source "$(dirname "$0")/env.sh"
export H3_TP2_ROLE=head VLLM_HOST_IP=$HEAD_IP
exec "$H3_VENV/bin/vllm" serve "$H3_MODEL" --omni --trust-remote-code \
  --host "$H3_API_HOST" --port "$H3_API_PORT" --init-timeout 3600 \
  --num-gpus 2 --tensor-parallel-size 1 --text-encoder-tp-size 2 --usp 2 --ring 1 \
  --vae-patch-parallel-size "${VAE_PP:-2}" --vae-parallel-mode tile --vae-use-tiling \
  --quantization fp8 --enforce-eager --diffusion-attention-backend CUDNN_ATTN \
  --distributed-executor-backend h3_tp2.executor.RemoteReaderExecutor "$@"
