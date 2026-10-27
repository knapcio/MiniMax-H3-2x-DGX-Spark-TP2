# How the two-Spark layout works

## Stock vLLM-Omni (one host)

`vllm serve --omni` starts the API server, a stage engine process and, inside it, `MultiprocDiffusionExecutor`:

1. the executor creates one broadcast `MessageQueue` (vLLM `shm_broadcast`: a shared-memory ring buffer plus a ZMQ
   XPUB socket) and spawns one `WorkerProc.worker_main` process per rank with `multiprocessing` (spawn);
2. each worker builds its own result `MessageQueue` (one local reader: the executor) and sends the handle back through
   an `mp.Pipe` together with `{"status": "ready"}`;
3. the executor subscribes to every result queue, starts the result pump and a monitor thread on the process sentinels;
4. `DiffusionWorker.init_device` sets `MASTER_ADDR=localhost`, `RANK`, `WORLD_SIZE`, picks `cuda:<rank>` and calls
   `init_distributed_environment`, then `initialize_model_parallel` builds the text-encoder TP and Ulysses SP groups.

The stage-init code also refuses to start when `world_size > torch.cuda.device_count()`.

## What `h3_tp2` changes

Nothing in the vLLM-Omni tree. Two pieces are installed into the same venv on both Sparks:

**`h3_tp2.plugin`** (a `vllm.general_plugins` entry point, so vLLM-Omni loads it in the API server, the stage engine and
every worker; a no-op unless `H3_TP2_ROLE` is set):

| role | patch | why |
|---|---|---|
| head | `get_device_count()` reports the world size | the one-GPU head must accept `world_size=2` |
| both | `get_torch_device()` returns `cuda:0` | one GPU per host |
| both | `init_distributed_environment` wrapper sets `MASTER_ADDR`/`MASTER_PORT` from `H3_TP2_MASTER_ADDR`/`_PORT` | rendezvous on the fabric address instead of `localhost` |
| worker | `DiffusionWorker(local_rank=0, rank=1)` | NVML and device bookkeeping use the local index |
| worker | the async D2H output thread opens its side stream on local device 0 (`gpu_id` is the rank) | one GPU per host |
| worker | the result `MessageQueue` is created with a remote reader bound to `H3_TP2_RESULT_CONNECT_IP` | the head reads it over ZMQ |
| both, opt-in | `empty_cache()` before `MiniMaxH3Pipeline.decode` (`H3_TP2_EMPTY_CACHE_BEFORE_DECODE=1`) | returns cached allocator pages before the VAE peak; outputs unchanged |
| all, default | `h3_tp2.h3_output`: `decode` returns uint8 frames, `_minimax_h3_post_process` passes them through, `_coerce_video_to_uint8_frames` skips the float round trip and logs a SHA-256 of the frames; `encode_prompt`/`diffuse`/`decode` are timed | the stock FP32 output path holds up to six 4.5 GB copies of a 15 s clip on the head (GPU, pinned, /dev/shm, API copy, clamp copy, float normalisation); this is what ran the head out of host memory. Same arithmetic, same bytes |

**`h3_tp2.executor.RemoteReaderExecutor`** (selected with `--distributed-executor-backend`), a subclass of the stock
executor that overrides three methods:

- `_init_broadcast_queue`: `n_reader=2, n_local_reader=1, connect_ip=<head fabric IP>`, plus a thread that completes
  the XPUB handshake with the remote subscriber (receive its subscription frame, send `READY`). PUB/SUB drops messages
  published before a subscription is active, and the stock code never waits because shared memory does not need it.
- `_launch_workers`: rank 0 exactly as upstream; rank 1 as a job (`od_config`, broadcast handle, env) sent to the
  worker agent over TCP on the fabric. The agent starts `h3_tp2.remote_worker`, which runs the stock
  `WorkerProc.worker_main` with a TCP shim in place of the pipe and, before the busy loop, handshakes both ZMQ channels.
- `_init_result_queue`: for the remote handle, wait for the worker's `READY` after subscribing.

Startup is bounded by one deadline (`H3_TP2_WORKER_READY_TIMEOUT`, default 3600 s) that covers rank 0, rank 1 and both
ZMQ handshakes; the executor waits on the rank-0 pipe and the agent socket at the same time, so a rank that dies while
the other is still loading is reported at once, and any failure terminates whatever was already started.

Liveness, both ways: the agent sends a heartbeat every 5 s and `dead` with the exit code if the rank exits; the head
pings the agent every 15 s and the agent terminates the worker after 90 s of silence (`H3_TP2_HEAD_SILENCE_S`), so a
black-holed head cannot leave a stale rank 1 holding 66 GB on the worker. Both sockets carry TCP keepalives and
`TCP_USER_TIMEOUT` (30 s). A request pending for longer than `H3_TP2_REQUEST_TIMEOUT_S` (default 14400) fails the
executor closed, which covers a rank that is alive but stuck. Failure handling is the stock one (`_is_failed`,
shutdown, failure callbacks), serialized so callbacks fire once and never during an intentional stop.

Shutdown lets the stock `SHUTDOWN` message reach both ranks through the broadcast queue first, then tells the agent to
terminate rank 1 (SIGTERM, 15 s, SIGKILL on the process group).

**`h3_tp2.agent`** runs as a user service on the worker Spark, bound to its fabric address only. One thread owns writes
to the head socket (an outbound queue), one owns reads; the worker's `ready` message and the heartbeats are enqueued,
never written concurrently. Any session end, intended or not, terminates the worker.

### Trust boundary

The control channel carries pickles, so anyone who can reach the agent port can run code as the agent's user. The
recipe binds it to the fabric address of a point-to-point cable only. Set the same `H3_TP2_SECRET` on both Sparks
(`~/.h3-tp2-secret`, read by `scripts/env.sh`) and every frame is authenticated with HMAC-SHA256; a frame with a bad tag
is rejected before it is unpickled. The ZMQ queues and NCCL itself remain unauthenticated, as upstream.

What rank 1 inherits from the head is an allowlist (`NCCL_*`, `GLOO_*`, `H3_TP2_*`, `VLLM_OMNI_*`, `TORCH_*`,
`OMP_*`, `PYTORCH_*`, `HF_HUB_OFFLINE`); device, cache, path and credential variables are never forwarded and never
land in the job file.

## Message flow during a render

```
API -> stage engine -> executor.collective_rpc --broadcast MQ--> rank 0 (shm)   -> result MQ 0 (shm)   -> pump
                                               \--ZMQ tcp------> rank 1 (05)    -> result MQ 1 (ZMQ)  -> pump
rank 0 <--NCCL over RoCE (text-encoder all-reduce, Ulysses all-to-all per attention layer)--> rank 1
```

Only rank 0 returns video tensors; rank 1 replies with acknowledgements and errors. The 1.5 GB FP32 video never crosses
the fabric.

## Tests

`python -m unittest discover -s tests` runs CPU-only tests of the control channel (frame HMAC, agent session state
machine: ready delivery, heartbeats, death before/after ready, head loss, lease expiry, SIGKILL path, consecutive
sessions) with a fake worker, of the uint8 output conversion against the stock numpy arithmetic (all batch elements,
exact rounding ties and their float neighbours, infinities; NaN is undefined on both paths) and of the grouped
all-to-all layout used by the overlap patch against the stock permutation; no GPU and no vLLM needed. The fabric
itself is tested by `scripts/preflight.sh` and the NCCL pair test in the README. GPU identity of any lever is the
same-seed frame digest (`h3_tp2 frames sha256` in the head journal, framemd5 in `results/`).

## Opt-in measurement and experiment switches

- `H3_TP2_A2A_TIMING=1`: CUDA-event totals of the Ulysses all-to-alls (`all_to_all_4D` calls and the collectives
  inside them) logged at the end of every denoising phase, per rank. Measurement only.
- `H3_TP2_A2A_OVERLAP=<groups>`: `h3_tp2.ulysses_overlap` replaces the attention layer's forward for the strict
  Ulysses path (no joint tensors, no masks, no Ring): local heads are split into `groups` head groups, the forward
  all-to-alls of all groups go to a second CUDA stream, the attention kernel runs per group on the compute stream and
  each group's backward all-to-all is issued as soon as its attention is done. Every NCCL call is issued from the comm
  stream, so both ranks keep the same collective order. The arithmetic per head is unchanged; whether the kernel picks
  the same plan for fewer heads is verified only by the frame digest for the configurations listed in the README.
- `H3_TP2_MAX_OUTPUT_SECONDS=<s>`: lifts the pipeline's 15 s cap (model specification is 4-15 s; above it the DiT
  extrapolates its temporal RoPE and the VAE runs lengths it was not validated on). Experiment only.

## vLLM-Omni versions

The recipe is measured on commit `e1aa6eae` (vLLM 0.26.0). The same package loads on v0.30.0 (vLLM 0.30.0, torch
2.13): every hook it uses still exists there with the same name, and the differences are handled at import time:
upstream v0.30 quantises the video to uint8 on the GPU itself and releases the stage cache before the VAE decode, so
`h3_output` keeps only its phase timers and frame digest; the worker is constructed with keyword arguments; the
executor's newer shutdown reads a few more attributes, which `_init_broadcast_queue` pre-creates; the duration limit
moved to `model_executor.models.minimax_h3.long_video` (15 s, or 30 s / 300 s with `long_video=true` in full /
continuation mode). Continuation is Ref2VA-only: windows of 107..345 frames (17n+5) with a 22-frame latent-tail
overlap, `diffuse` is called once per window and `decode` once for the whole clip.
