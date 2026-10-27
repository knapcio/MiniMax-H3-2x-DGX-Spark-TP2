# Shared settings for the two-Spark MiniMax H3 recipe. Source from the other scripts.
# Head = rank 0 (serves the API), worker = rank 1. Fabric = direct ConnectX-7 (both twins of the cabled port).
# Everything here can be overridden from the environment; put site values in ~/.h3-tp2.env (sourced if present).
[ -r "$HOME/.h3-tp2.env" ] && . "$HOME/.h3-tp2.env"
H3_ROOT=${H3_ROOT:-$HOME/h3-base-20260916}
H3_VENV=${H3_VENV:-$H3_ROOT/runtime/.venv}
H3_MODEL=${H3_MODEL:-$H3_ROOT/weights/FL2VA}
H3_API_HOST=${H3_API_HOST:-127.0.0.1}
H3_API_PORT=${H3_API_PORT:-18093}

HEAD_IP=${HEAD_IP:-10.100.96.6}          # head rail-1 address (NVIDIA Sync gives 10.100.96.0/24 to the cabled port)
WORKER_IP=${WORKER_IP:-10.100.96.5}      # worker rail-1 address
AGENT_PORT=${AGENT_PORT:-18095}
FABRIC_IF=${FABRIC_IF:-enp1s0f0np0}      # netdev of rail 1 (NCCL/Gloo bootstrap)
IB_HCA=${IB_HCA:-rocep1s0f0,roceP2p1s0f0}   # both twins of the cabled QSFP port
IB_GID_INDEX=${IB_GID_INDEX:-3}          # RoCE v2, IPv4

export H3_TP2_HEAD_IP=$HEAD_IP H3_TP2_WORKER_IP=$WORKER_IP H3_TP2_WORKER_AGENT=$WORKER_IP:$AGENT_PORT
export H3_TP2_MASTER_ADDR=$HEAD_IP H3_TP2_WORLD_SIZE=2
# optional shared secret for the control channel (same file content on both Sparks): HMAC on every frame
[ -r "$HOME/.h3-tp2-secret" ] && export H3_TP2_SECRET="$(tr -d '\n' < "$HOME/.h3-tp2-secret")"
export NCCL_SOCKET_IFNAME=$FABRIC_IF GLOO_SOCKET_IFNAME=$FABRIC_IF
export NCCL_IB_HCA=$IB_HCA NCCL_IB_GID_INDEX=$IB_GID_INDEX NCCL_IB_ROCE_VERSION_NUM=2
export NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_IB_MERGE_NICS=${NCCL_IB_MERGE_NICS:-1} NCCL_CROSS_NIC=0
export NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0 NCCL_NET_GDR_LEVEL=0 NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_ADDR_FAMILY=AF_INET
# vLLM-Omni runtime knobs (same as the single-node unit)
export HF_HUB_OFFLINE=1 VLLM_OMNI_VIDEO_SYNC_TIMEOUT=${VLLM_OMNI_VIDEO_SYNC_TIMEOUT:-14400}
export VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=${VLLM_OMNI_DLO_DP_WAVE_TIMEOUT:-14400}
export H3_TP2_REQUEST_TIMEOUT_S=${H3_TP2_REQUEST_TIMEOUT_S:-14400}
export H3_TP2_EMPTY_CACHE_BEFORE_DECODE=${H3_TP2_EMPTY_CACHE_BEFORE_DECODE:-1}   # return cached pages before the VAE peak (bit-identical)
export VAE_PP=${VAE_PP:-2}   # VAE decode tiles split across both GPUs (bit-identical, measured)
export TRITON_CACHE_DIR=$H3_ROOT/runtime/cache/triton FLASHINFER_WORKSPACE_BASE=$H3_ROOT/runtime/cache/flashinfer
export CPATH="$H3_ROOT/runtime/debs/extracted/usr/include/python3.12:$H3_ROOT/runtime/debs/extracted/usr/include/aarch64-linux-gnu/python3.12:$H3_ROOT/runtime/debs/extracted/usr/include"
export PATH="$H3_VENV/bin:/usr/local/bin:/usr/bin:/bin"
