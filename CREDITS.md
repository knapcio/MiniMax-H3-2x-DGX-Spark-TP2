# Credits

- **MiniMax** for MiniMax H3 and the FL2VA checkpoint ([model card](https://huggingface.co/MiniMaxAI/MiniMax-H3)).
- **vLLM-Omni** (vllm-project/vllm-omni) for the engine: the diffusion executor/worker pair this recipe extends, the
  text-encoder TP and Ulysses SP implementation of H3, and the GB10 recipe
  ([MiniMax-H3-Spark-GB10.md](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3-Spark-GB10.md)).
- **Bojiang Li (@bojiang-li)** for the first two-Spark measurement of exactly this layout (text-encoder TP2 + USP2 at commit
  `e1aa6eae`, [vllm-omni #7343](https://github.com/vllm-project/vllm-omni/pull/7343)); our 5 s / 10 s runs use the same
  workload so the numbers compare.
- **Joey Rodriguez (@joeynyc)** for the first public two-Spark H3 server
  ([MiniMax-H3-2x-DGX-Spark](https://github.com/joeynyc/MiniMax-H3-2x-DGX-Spark), Apache-2.0): the RoCE v2 GID preflight
  checks, the media acceptance script (`scripts/verify-output.sh` follows his), the fail-closed executor rule, and the
  warning that MP4 bytes differ between identical renders (we hash decoded frames instead).
- **brandneway** for vllm-omni #7241 (freeing the allocator's cached pages at the denoise/decode boundary; bit-identical).
- **NVIDIA** DGX Spark playbooks (Connect Two Sparks, NCCL for two Sparks, the 175 Gbps bandwidth gate in
  `spark_cluster_setup.py`) and the ConnectX-7 twin-port measurements.
- **tonyd2wild** for the MemAvailable guard pattern on GB10 (`h3_guard.sh`, MIT); ours restarts the systemd unit instead.
- **vLLM** project: the `MessageQueue` (shm_broadcast) remote-reader mode over ZMQ is what lets rank 1 live on another host
  without Ray.
- The NCCL/RoCE environment is carried over from our own DeepSeek V4.1 and GLM-5.3 TP4 recipes.
