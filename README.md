# MiniMax H3 on 2x NVIDIA DGX Spark (text-encoder TP2 + Ulysses SP2)

MiniMax H3-Base (FL2VA: first/last frame + prompt -> video with audio) served by vLLM-Omni across two DGX Sparks
connected back to back over ConnectX-7. The text encoder is tensor-parallel over the two GPUs, the DiT runs Ulysses
sequence parallel, so each Spark holds about two thirds of what a single Spark holds today and the diffusion steps are
shared. Same maths as the single-Spark recipe (online FP8 DiT, BF16 encoder and VAEs, cuDNN attention): the point is
longer clips (10 s at 1344x768, which one Spark cannot hold) and twice the speed, not a different model.

vLLM-Omni cannot launch diffusion workers on another host (`vllm serve` starts every rank locally and pins
`MASTER_ADDR=localhost`). This recipe adds `h3_tp2`, a small package that keeps the stock executor and only moves rank 1
to the second Spark: the executor's broadcast message queue gets a ZMQ remote reader, the worker's result queue is read
back the same way, and a tiny agent on the worker Spark starts the rank on request. No Ray, no fork of vLLM-Omni.

## Speed and memory

Same prompt, seed 1101, flow shift 12, 24 fps, T2VA (video + audio), online FP8 DiT, cuDNN attention. Times are wall
clock of `POST /v1/videos/sync` after a warm-up request; "peak" is the engine's `X-Peak-Memory-MB` (reserved pool, so an
upper bound); "min free" is the lowest MemAvailable sampled every 2 s on that host during the request. Raw files in
`results/`.

| Clip | One Spark (stock recipe) | Two Sparks (this recipe) |
|---|---|---|
| 5 s 960x576, 10 steps | 290.9 s, peak 97.2 GB, min free 15.7 GB | 153.8 s, peak 75.2 GB, min free 38.8 GB (head) / 39.6 GB (worker) |
| 5 s 1344x768, 20 steps | 1399.0 s, peak 103.3 GB, min free 6.0 GB | 643.0 s, peak 75.2 GB, min free 38.8 GB / 39.3 GB |
| 10 s 1344x768, 20 steps | does not fit: the host runs out of memory and livelocks | 2119.0 s, peak 91.0 GB, min free 10.2 GB / 22.9 GB (first boot: FP32 output path, VAE on one GPU) |
| 10 s 1344x768 from a still (FL2VA), 20 steps | does not fit | 1940.3 s, peak 91.2 GB, min free 19.9 GB / 29.3 GB (uint8 output path, VAE on one GPU) |
| 15 s 1344x768, 20 steps | does not fit | did not fit either: head out of host memory after 7796 s (stock FP32 output path, see Levers); <!-- TP2-15s --> |

On one Spark the 5 s 960x576 render took 362.8 s when run right after the 1344x768 one (same frames, bit for bit): the
allocator keeps the bigger pool and the host is under pressure, which is also why the single-Spark recipe is limited to
5 s. The two-Spark numbers come from fresh boots with memory compacted first; the 5 s rows are the final configuration
(uint8 output path, VAE split over both GPUs, cache freed before decode), the 10 s rows predate the last two levers.

## Quality checks

Tensor parallelism changes the order of floating-point reductions (all-reduce over two GPUs, sequence-parallel
attention), so the two-Spark render of a given seed is not bit-identical to the one-Spark render of the same seed; it
is the same model, the same steps and the same precision. What we measure, same seed and prompt on both layouts:

| Case | Decoded frames | SSIM | PSNR | Audio level (one / two Sparks) |
|---|---|---|---|---|
| 5 s 960x576, 10 steps | differ | 0.927 | 29.5 dB | -27.7 / -31.2 dB mean |
| 5 s 1344x768, 20 steps | differ | 0.959 | 37.2 dB | -46.4 / -46.9 dB mean |

Read this as "the same clip with a different noise realisation of the fine detail", which is what a change in
floating-point reduction order does to a 10 to 20 step diffusion; fewer steps diverge more. There is no step, precision
or model change between the two columns.

Run-to-run on one layout is deterministic (two one-Spark renders of the 5 s 960x576 case gave the same decoded frames,
`framemd5 b58a4612`). Every lever in the table below that claims "bit-identical" was checked by rendering the same seed
with and without it and comparing the SHA-256 of the uint8 frames that the plugin logs (`h3_tp2 frames sha256 ...` in
the head journal) and `scripts/verify-output.sh`'s framemd5 of the decoded video.

## Levers

| Lever | Setting | What it does | Lossless | Credit |
|---|---|---|---|---|
| Text-encoder TP2 | `--text-encoder-tp-size 2` | the 50-layer Qwen3-VL encoder (51.5 GB BF16) is sharded over both Sparks: ~26 GB per rank instead of 52 GB on one | same model and precision; reduction order changes (Quality checks) | vLLM-Omni |
| Ulysses SP2 | `--usp 2` | the DiT splits the video sequence across the two GPUs; all-to-alls around every attention layer over RoCE | same model and precision; reduction order changes | vLLM-Omni |
| Cross-host executor | `--distributed-executor-backend h3_tp2.executor.RemoteReaderExecutor` | rank 1 on the second Spark through vLLM's remote-reader message queues | yes | ours; vLLM shm_broadcast |
| Both ConnectX-7 twins | `NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0` | all-to-all over two rails: 19.5 GB/s bus bandwidth instead of 12.8 | yes | NVIDIA playbook, our LLM recipes |
| Online FP8 DiT | `--quantization fp8` | the only way the DiT fits; encoder and VAEs stay BF16 | numerics of the single-Spark recipe | vLLM-Omni recipe |
| cuDNN attention | `--diffusion-attention-backend CUDNN_ATTN` | the GB10-validated backend | numerics only | vLLM-Omni recipe |
| Tiled VAE, split over both GPUs | `--vae-use-tiling --vae-parallel-mode tile --vae-patch-parallel-size 2` | the 3x6 pixel tiles of each temporal chunk are decoded round-robin by the two ranks and gathered; each tile runs the unchanged decoder, blend order unchanged. Measured: same frames (framemd5 `c36d17dc` / `8b3e06a8` with 1 and 2), 5 s 1344x768 681.8 -> 643.0 s, peak 77.5 -> 75.2 GB | bit-identical (measured) | vLLM-Omni |
| Free cached pages before decode | `H3_TP2_EMPTY_CACHE_BEFORE_DECODE=1` (default) | `gc.collect()` + `torch.cuda.empty_cache()` right before the VAE decode so its peak does not stack on the allocator's denoise-time cache; no tensor arithmetic touched | bit-identical (measured) | brandneway, vllm-omni #7241 |
| uint8 output path | `H3_TP2_U8_OUTPUT=1` (default) | frames leave the GPU as uint8 (clamp, x255, round-half-even on the GPU: the same FP32 operations the stock path runs later in numpy) instead of FP32; every copy on the head shrinks 4x, and a SHA-256 of the frames is logged as the identity check | bit-identical (`tests/test_u8_identity.py`, same-seed frame digest) | ours |

Tried, not adopted (lossy, so out of scope here): Cache-DiT / caching profiles, Turbo or distilled step LoRAs, NVFP4 or
INT8 DiT, CPU offload (unified memory: frees nothing).

## The model

MiniMax H3-Base FL2VA: DiT 62 GB BF16 (31 GB after online FP8), Qwen3-VL text encoder with 50 retained layers
(51.5 GB BF16), video VAE 9.8 GB, audio VAE. 24 fps, 4 to 15 s, native 768p. Weights: `MiniMaxAI/MiniMax-H3`, folder
`FL2VA`, identical path on both Sparks.

## Hardware

Two DGX Spark (GB10, 128 GB unified memory, DGX OS kernel 6.17.0-1032-nvidia, driver 580.178.04), one QSFP cable
between the top ConnectX-7 ports (both twins active: `rocep1s0f0` and `roceP2p1s0f0`, 200 Gb/s each). Management over
Wi-Fi or RJ-45 is fine; only NCCL and the rank-1 control channel use the fabric. Power both Sparks off before cabling: a
cable plugged into a running Spark brings the NIC back, but RDMA runs at ~13 Gb/s until the next reboot.

## Running it

Prerequisites on **both** Sparks, at identical paths: the single-Spark H3 runtime (vLLM-Omni at commit `e1aa6eae`, vLLM
0.26.0, Python 3.12 venv) and the FL2VA weights. Defaults are `~/h3-base-20260916/{runtime/.venv,weights/FL2VA}`;
override `H3_ROOT`, `H3_VENV`, `H3_MODEL`, the two fabric addresses (`HEAD_IP`, `WORKER_IP`), the netdev of rail 1
(`FABRIC_IF`) and the two HCAs (`IB_HCA`) in `~/.h3-tp2.env` (sourced by `scripts/env.sh`). Optional but recommended:
the same random string in `~/.h3-tp2-secret` on both Sparks authenticates the rank-1 control channel (see
docs/ARCHITECTURE.md, Trust boundary).

```bash
# on each Spark
git clone https://github.com/knapcio/MiniMax-H3-2x-DGX-Spark-TP2 ~/h3-tp2
~/h3-base-20260916/runtime/.venv/bin/pip install -e ~/h3-tp2
bash ~/h3-tp2/scripts/preflight.sh            # links ACTIVE, RoCE v2 GID + IPv4 per HCA, >=110 GiB free, venv/package/weights present; same versions on both
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches; echo 1 | sudo tee /proc/sys/vm/compact_memory   # before every boot: a fragmented host makes rank 0 slower

# worker Spark (rank 1)
cp ~/h3-tp2/systemd/h3-tp2-{agent,memguard-worker}.service ~/.config/systemd/user/ && systemctl --user daemon-reload
systemctl --user enable --now h3-tp2-agent.service h3-tp2-memguard-worker.service   # agent listens on WORKER_IP:18095, fabric only

# head Spark (rank 0, API on 127.0.0.1:18093)
cp ~/h3-tp2/systemd/h3-tp2-{head,memguard-head}.service ~/.config/systemd/user/ && systemctl --user daemon-reload
systemctl --user enable --now h3-tp2-head.service h3-tp2-memguard-head.service
journalctl --user -u h3-tp2-head.service -f     # ~7 min to health 200 with a warm page cache; rank-1 log in ~/h3-tp2-logs on the worker
bash ~/h3-tp2/scripts/status.sh                  # both units, health, MemAvailable on both nodes
```

First request after a boot pays lazy initialisation; the recipe warm-up is one 5 s, 2-step render:

```bash
bash ~/h3-tp2/scripts/render.sh ~/warmup.mp4 5 1344 768 2 1101 "a quiet valley at dawn"
bash ~/h3-tp2/scripts/render.sh ~/clip.mp4 10 1344 768 20 1101 "<prompt>" ~/still.jpg first   # FL2VA from a still
bash ~/h3-tp2/scripts/verify-output.sh ~/clip.mp4 10 1344 768                                   # streams, size, duration, decode, non-silent audio, framemd5; exit 1 on failure
bash ~/h3-tp2/scripts/bench-matrix.sh "5:1344:768:50:1101 10:1344:768:50:1101"                  # the table above, with memory sampling on both nodes; STATUS per case
```

The API is the stock vLLM-Omni one (`POST /v1/videos/sync`, `extra_params` with `task`, `duration`, `frame_indices`),
so existing clients only need the host. The memory guards kill the rank when MemAvailable stays under 4 GiB for two
samples: on GB10 the NVIDIA driver cannot fail an allocation gracefully and the whole host livelocks instead (that is
how we lost an afternoon to a 10 s render on one Spark). The head then fails closed and systemd restarts it; a stuck
request is bounded by `H3_TP2_REQUEST_TIMEOUT_S` (4 h) and the API by `VLLM_OMNI_VIDEO_SYNC_TIMEOUT`.

`python -m unittest discover -s tests` (any machine, no GPU) exercises the control channel and the agent's failure paths.

## Next

- 15 s at 1344x768: with the uint8 output path the head keeps ~10 GB more during a 10 s render; a guarded 15 s run is
  in progress and its row above will say whether two Sparks reach the model's maximum length. <!-- 15S -->
- Attention is 57 % of a denoising step at 10 s and 66 % at 15 s (fit of step time against sequence length:
  t ≈ 4.9e-4·S + 8.8e-9·S² s); an exact, faster attention kernel for SM121 is the only lever above 10 %.
- Ulysses sequence parallel issues three all-to-alls per attention layer (q, k, v) with a permute copy each; one fused
  all-to-all without the copies is bit-identical and worth ~1-2 s per step. The fabric itself is not the bottleneck
  (≈ 53 GB per step per direction at 19.5 GB/s is 3-6 % of a step).
- Phase timers (`h3_tp2 phase encode_prompt / diffuse / decode` in the head journal) for the 5 s 1344x768 20-step
  render: prompt encoding 0.2 s, denoising 598.9 s (30 s per step), VAE decode 42.1 s with the decoder split over both
  GPUs; the rest (audio decode, uint8 conversion, muxing) is about 2 s. Denoising is the whole game above 5 s.
- Regional `torch.compile` of the eager modulation and elementwise work (drop `--enforce-eager`) is 5-10 % of the
  non-attention time but changes floating-point association, so it stays out of a lossless recipe.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to shrink the reserved-but-unused pool: numerics-neutral, GB10
  driver support untested.

## Licences

Code: MIT (see LICENSE, NOTICE). vLLM-Omni and vLLM: Apache-2.0. MiniMax H3 weights and outputs: MiniMax H3 Community
License (see MODEL-LICENSE.md). Credits in CREDITS.md.
