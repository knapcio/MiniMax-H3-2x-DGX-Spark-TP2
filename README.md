# MiniMax H3 on 2x NVIDIA DGX Spark (text-encoder TP2 + Ulysses SP2)

MiniMax H3-Base (FL2VA: first/last frame + prompt -> video with audio) served by vLLM-Omni across two DGX Sparks
connected back to back over ConnectX-7. The text encoder is tensor-parallel over the two GPUs, the DiT runs Ulysses
sequence parallel, so each Spark holds about two thirds of what a single Spark holds today and the diffusion steps are
shared. Same maths as the single-Spark recipe (online FP8 DiT, BF16 encoder and VAEs, cuDNN attention): the point is
longer clips (10 s and the model's maximum 15 s at 1344x768, which one Spark cannot hold) and twice the speed, not a
different model.

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
| 10 s 1344x768, 20 steps | does not fit: the host runs out of memory and livelocks | 1690.5 s, peak 84.2 GB, min free 26.1 GB / 29.6 GB (first boot with the stock FP32 output path and the VAE on one GPU: 2119.0 s, min free 10.2 GB) |
| 10 s 1344x768 from a still (FL2VA), 20 steps | does not fit | 1940.3 s, peak 91.2 GB, min free 19.9 GB / 29.3 GB (uint8 output path, VAE on one GPU) |
| 15 s 1344x768 from a still (FL2VA), 20 steps | does not fit | 3166.3 s, peak 84.4 GB, min free 24.1 GB / 26.5 GB |
| 15 s 1344x768, 20 steps | does not fit | 3079.9 s, peak 85.6 GB, min free 25.8 GB / 27.2 GB (final configuration). With the stock FP32 output path and the VAE on one GPU the same render ran the head out of host memory after 7796 s |

On one Spark the 5 s 960x576 render took 362.8 s when run right after the 1344x768 one (same frames, bit for bit): the
allocator keeps the bigger pool and the host is under pressure, which is also why the single-Spark recipe is limited to
5 s. The two-Spark numbers come from fresh boots with memory compacted first; all rows except the 10 s FL2VA one are
the final configuration (uint8 output path, VAE split over both GPUs, cache freed before decode). Phases: 10 s T2VA
prompt encoding 0.2 s, denoising 1602.6 s (80 s per step), VAE decode 84.2 s; 15 s T2VA 0.2 / 2946.7 (147 s per step)
/ 125.8 s; 15 s FL2VA 0.9 / 3030.9 / 125.9 s.

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
| Two-pass FP8 GEMM (default since 2026-10-29; `H3_TP2_FP8_GEMM=stock` restores the CUTLASS kernel) | `H3_TP2_FP8_GEMM=tensor32` | the per-row-scaled CUTLASS FP8 kernel runs at ~59 TFLOPS on the GB10 while the same GEMM with scalar scales runs at ~189; the plugin runs the scalar GEMM with fp32 output and applies the per-token and per-channel scales, bias and rounding in one Triton pass in vLLM's epilogue order. Warm 5 s 960x576 render: 451.6 s against 592.3 s (-24 %), first request after boot 437.8 against 771.1 s; at the serving cell (FL2VA 5 s 1344x768) 558 against 688 s (-19 %) with a byte-identical MP4, 10 s -15 % | same operations, different reduction order inside the GEMM; deterministic; at FL2VA 5 s 1344x768 the output is byte-identical to stock, at 960x576 Ref2VA and at 10 s the frames differ (same look) | ours; GB10 FP8 ceiling confirmed by NVlabs' sol-engine GB10 port |
| Fused epilogue + SwiGLU level 1 (default since 2026-10-29; `H3_TP2_FP8_FUSE_SWIGLU=0` disables, `2` experimental) | `H3_TP2_FP8_FUSE_SWIGLU=1` | fc1's fp32 product is read once by a kernel that applies the epilogue and `silu_and_mul`; the bf16 [M, 2*ffn] intermediate is neither written nor read | bit-identical to the two-pass path in nine 5 s renders (same latents and frames), denoising -5..-7 % at the median with a ~10 % boot-to-boot spread (`results/20261029-reproducibility/`); level 2 (per-token quant in the same kernel) stays experimental | ours; the fusion sites follow NVlabs' GB10 lossless tier |

Tried, not adopted (lossy, so out of scope here): Cache-DiT / caching profiles, Turbo or distilled step LoRAs, NVFP4 or
INT8 DiT, CPU offload (unified memory: frees nothing).

## vLLM-Omni v0.30.0

The same package runs on vLLM-Omni v0.30.0 (vLLM 0.30.0, torch 2.13; the first release with the MiniMax-H3 long-video
continuation). Measured on the pair with the FL2VA partition, 5 s 1344x768 20 steps: 645.0 s against 643.0 s on the
pinned commit, peak 62.2 GB against 75.2 GB, 53 GB free on both hosts during the request, same quality by eye and by
SSIM (`results/20261027-v030-fl2va/`). A second runtime directory selected with `scripts/select-stack.sh` keeps both
stacks installed side by side.

Ref2VA on the same stack: 5 s 1344x768 in 1248.6 s (1.94x FL2VA, as upstream documents), peak 65.6 GB. The reference
image is subject/style conditioning, not a first frame.

**Clips longer than 15 s with real continuity**: upstream's latent-tail continuation (Ref2VA only, `long_video=true`)
renders the clip in windows of 277 frames with a 22-frame overlap, each window conditioned on the previous window's
latent tail (video and audio). First measurement on the pair: 30 s 960x576 20 steps in 3615 s (3 windows, peak
71.1 GB, 42 GB free on both hosts), no scene cut in the whole clip; the picture is continuous across both seams but
the camera changed speed and direction at the second one, which is visible when watched
(`results/20261027-v030-continuation-30s/`). A 60 s clip (6 windows, 7986 s, peak 69.4 GB) shows the limit of
the plain mode: with one fixed prompt the apparent motion dies down window by window (inter-frame change 7x smaller
by 53 s), the audio mean level fades by 18 dB and the model inserts shot changes it was not asked for; the picture itself stays sharp and consistent
(`results/20261027-v030-continuation-60s/`). At 1344x768 the same 30 s request takes 10114 s (2.8 h, peak 78.8 GB, 26 GB free) and its first seam came out as a
hard cut in the one sample so far (`results/20261027-v030-continuation-30s-1344/`). Asking for "one single continuous
take, no cuts, no transitions" in the
prompt removes the shot changes and the hand-over bumps at the seams (`results/20261027-v030-continuation-prompt/`,
+25 % time for the longer prompt); the motion and audio decay remain. A driving soundtrack (`audio_mode=lock_source`, `audio_reference`) removes the audio fade by construction, and its
samples also showed no motion decay and no shot changes over 30 s with the plain prompt
(`results/20261027-v030-continuation-audiolock/`), at the price of not generating the audio; the same two settings carried a 60 s clip through six windows with no cut,
smooth seams, flat audio and no motion decay in 7313 s (`results/20261028-v030-continuation-60s-best/`). One prompt
per window (`continuation_prompts`) keeps the motion and the
shot continuous at no extra cost but the generated audio still fades about 8 dB over 30 s
(`results/20261027-v030-continuation-winprompts/`). Local-time windows and a different seed per window were tested and
change nothing (`results/20261028-v030-continuation-toff0/`, `results/20261028-v030-continuation-seedstep/`). Repeats
after 16 h of continuous rendering are bit-identical and within 1 % in time, so the continuation is deterministic on the
pair and there is no sustained-load slowdown; a 345-frame window (the upstream maximum) gave smooth seams and a 1.7 dB
instead of 7 dB audio fade over 30 s at +8 % time in one sample (`results/20261028-v030-repeats-and-window345/`). Request shape in `scripts/render.sh` (`TASK=ref2va`, `EXTRA_JSON`).

**Ref2VA cost**: the 1.94x comes from the reference rows (upstream resizes the reference image to a 2048 short edge:
~7300 VAE rows plus the vision tokens in every DiT step). `H3_TP2_REF_SHORT_EDGE=1024` cut the 5 s Ref2VA render
from 1026.7 s (warm repeat; the first request after the boot took 1248.6 s) to 752.9 s (-27 %) with the same quality
by eye in one sample (`results/20261028-v030-ref2va-edge1024/`);
it changes the conditioning input, not the model; continuation runs with it are being measured.

**Long clips, where this stands after the 2026-10-27/29 campaign** (one sample per cell unless noted; all 960x576
unless noted; `results/20261028-clip-metrics/` has the per-second numbers):

| Goal | Lever | Result |
|---|---|---|
| Speed of the continuation | `H3_TP2_REF_SHORT_EDGE=1024` | -24 % at 960x576, -3 % at 1344x768, no other effect (same fades and detail): recommended |
| Motion decay across windows | per-window prompts (`CONT_PROMPTS=auto`) | fixed, free |
| Generated-audio fade (-8..-18 dB over 30-60 s) | a driving soundtrack (`audio_mode: lock_source`) or the largest window (345 frames) | both keep the level flat; `audio_flow_shift` 1 makes it worse |
| Texture blur over 60 s | the largest window (345) | no blur, but over-sharpening instead at the end |
| Unrequested cuts and dissolves inside windows | prompt wording, light-lock wording, an opening-frames anchor, a tail anchor, H3's own looped audio as the track | none works: the cuts land on the same frames whatever the prompt (sampling events of the per-window seed), the anchors add cuts, the loop runs the motion away. One 60 s clip with a music track had no cuts; one sample. |

So a 30 s 960x576 clip with reference 1024 + per-window prompts + a soundtrack came out clean (one sample,
`results/20261028-v030-recipe-30s-960/`); at 1344x768 and at 60 s no combination tried is reliably free of the
model's own scene changes, and the final detail drifts (blur with 277-frame windows, over-sharpening with 345). The
30 s 960x576 recipe is the honest offer; longer or larger is possible (the mode allows 300 s, about 2.2 h per 60 s)
but comes with the model's own cuts. Not yet tested on other prompts and references.

### Long clips from a client or an agent: the request

**Verdict of 2026-10-29 evening (owner, by eye): the continuation is not usable for real work.** The first hand-over
at 11.5 s shows as a light step (luma 54 -> 64 within six frames, +19 %) with a texture change and the subject
appears twice, with or without per-window prompts; a 60 s clip washes out (luma 47 -> 88) and the subject's action
loops window after window. The metrics in this repo did not catch any of it, the eye did. **What is stable: one
window, 4-15 s, the still as the first frame (FL2VA), the serving configuration of `## Running it`.** The request
below stays documented for experiments only.

The server must run the v0.30.0 stack on the **Ref2VA** partition (`scripts/select-stack.sh long Ref2VA` on both
Sparks, agent restarted before head) with `H3_TP2_REF_SHORT_EDGE=1024` in `~/.h3-tp2.env` on the head; FL2VA (the image
as the first frame) cannot continue a clip. On Ref2VA the image is a **reference for look and subject**, not the first
frame: the clip resembles the picture, it does not start from it. One request at a time; the call is synchronous and
returns the MP4 (with audio) in the response body.

```bash
curl -sS --max-time 14400 -X POST http://127.0.0.1:18093/v1/videos/sync \
  -F "prompt=An old forest with a small stream. Leaves sway in a light breeze, the stream ripples and glistens, a woman in a white dress walks slowly along it. Steady camera, one continuous take, natural forest ambience." \
  -F "width=960" -F "height=576" -F "aspect_ratio=16:9" -F "fps=24" \
  -F "num_inference_steps=20" -F "flow_shift=12" -F "seed=1101" \
  -F 'extra_params={"task": "ref2va", "duration": 30, "audio_flow_shift": 3.0, "long_video": true, "long_video_mode": "continuation"}' \
  -F "input_reference=@picture.jpg" -o clip.mp4 -w "HTTP %{http_code}\n"
```

What to expect and what to tell the user:

| | |
|---|---|
| Lengths | 4-15 s: one window (same fields without the two `long_video` keys also work on Ref2VA). 16-30 s: windows of 277 frames continued from each other. Longer is possible (the mode allows 300 s) but the one 60 s sample was not usable (colour drift, the subject's action looping per window). |
| Size | 960x576 for long clips (1344x768 takes 2.8x longer at 30 s and its first seam cut in the one sample). |
| Time on the pair (two-pass GEMM + fused SwiGLU, reference 1024) | 30 s: 35-36 min measured (2097-2163 s, peak 68.7 GB); 60 s: 75 min measured (4470 s, peak 67.9 GB). Windows after the first denoise about 17 % slower than the first. |
| What 30 s and 60 s look like | Not usable by eye (owner, 2026-10-29): light and texture step at the first seam and a duplicated subject at 30 s; wash-out and a looping action at 60 s. The earlier "seam invisible" reading of frame strips was wrong: the luma step is +10 within six frames (`results/20261029-long-clips-agent-shape/`). |
| Prompt | Describe the motion of what is **in** the picture (this water, these clouds, this person); a prompt that asks for something the picture does not contain makes the model move the camera to find it. End with "Steady camera, one continuous take". |
| Per-window prompts | Generated by the server when the request sends none (since 9a65815, `H3_TP2_CONT_PROMPTS=off` disables): the first window gets the prompt plus "one single continuous take", the following windows "the same continuous shot goes on without any cut or transition" plus the prompt. Without them every window re-introduces the prompt's subject (a second woman after the first seam) and the hand-over shows as a light bump. A request may still send its own `continuation_prompts` (one string per window). |
| Known limits | The model may insert a cut of its own inside a window (a seed-dependent sampling event, not a prompt error); the generated audio may fade over 30-60 s; motion calms down window by window. A clip with a cut is best redone with another `seed`. A driving soundtrack (`audio_reference`, `audio_mode: lock_source`) keeps the audio level flat by construction, see the table above. |
| Reference size | The server resizes the reference to a 1024 px short edge; a larger picture is fine. |

`clients/h3-generate.py --long --duration 30 --image picture.jpg "prompt"` in the workspace sends the same request
through an SSH tunnel.

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

### Second runtime: vLLM-Omni v0.30.0 (long clips)

The same units can run the newer stack from a second directory; nothing in the first one is touched, and
`scripts/select-stack.sh` switches per host.

```bash
# on each Spark (the weights of the Ref2VA partition are 134 GiB; the FL2VA ones can be a symlink to the existing copy)
mkdir -p ~/h3-long && cd ~/h3-long
python3.12 -m venv .venv && .venv/bin/pip install -q --upgrade pip uv
.venv/bin/uv pip install --python .venv/bin/python "vllm==0.30.0" --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
git clone https://github.com/vllm-project/vllm-omni && (cd vllm-omni && git checkout v0.30.0) && .venv/bin/uv pip install --python .venv/bin/python -e vllm-omni
.venv/bin/uv pip install --python .venv/bin/python --no-deps -e ~/h3-tp2
mkdir -p runtime/cache/{triton,flashinfer} weights && ln -s ../.venv runtime/.venv && ln -s ../vllm-omni runtime/vllm-omni && ln -s ~/h3-base-20260916/runtime/debs runtime/debs
ln -s ~/h3-base-20260916/weights/FL2VA weights/FL2VA
HF_HUB_DISABLE_XET=1 .venv/bin/hf download MiniMaxAI/MiniMax-H3 --include "Ref2VA/*" --local-dir ~/h3-long/weights
bash ~/h3-tp2/scripts/select-stack.sh long Ref2VA      # or: long FL2VA, or: base   (then restart the agent, then the head)
```

A 30 s clip with continuity:

```bash
TASK=ref2va EXTRA_JSON='{"long_video": true, "long_video_mode": "continuation"}' \
  bash ~/h3-tp2/scripts/render.sh ~/long.mp4 30 960 576 20 1101 "<prompt>" ~/reference.jpg
```

## Next

- Longer than 15 s is a model limit (duration range 4 to 15 s), not a memory one: with the cap lifted
  (`H3_TP2_MAX_OUTPUT_SECONDS=20`, experiment only) a 20 s 1344x768 clip rendered in 6740 s with 20 GB still free on
  both hosts and looked coherent to the last frame on a calm landscape prompt (`results/20261027-20s-out-of-spec/`).
  One sample outside the model's training range; the serving configuration keeps the 15 s cap.
- Measured all-to-all share of a denoising step at 20 s: 2.9 % including the permute copies, 2.1 % inside the
  collectives. That is the ceiling for `H3_TP2_A2A_OVERLAP`.
- Attention is 57 % of a denoising step at 10 s and 66 % at 15 s (fit of step time against sequence length:
  t ≈ 4.9e-4·S + 8.8e-9·S² s); an exact, faster attention kernel for SM121 is the only lever above 10 %. Of the exact
  kernels installed here cuDNN is already the fastest: vLLM's FlashAttention-2 is 3 % slower and torch's flash SDPA
  4-7 % slower on the per-rank shapes (`scripts/attn-bench.py`, `results/20261027-attn-bench/`).
- Ulysses sequence parallel issues three all-to-alls per attention layer (q, k, v) and one back, and the GPU idles
  during them: 2.9 % of a denoising step at 20 s (`H3_TP2_A2A_TIMING=1`). Hiding them behind the attention kernel on a
  second stream (`H3_TP2_A2A_OVERLAP=<groups>`, off, kept for reference) gives identical frames but is 4.5 % slower:
  NCCL kernels queue behind the attention kernel on this GPU instead of running beside it
  (`results/20261027-a2a-overlap/`). Closed.
- Phase timers (`h3_tp2 phase encode_prompt / diffuse / decode` in the head journal) for the 5 s 1344x768 20-step
  render: prompt encoding 0.2 s, denoising 598.9 s (30 s per step), VAE decode 42.1 s with the decoder split over both
  GPUs; the rest (audio decode, uint8 conversion, muxing) is about 2 s. Denoising is the whole game above 5 s.
- Regional `torch.compile` of the eager modulation and elementwise work (drop `--enforce-eager`) is 5-10 % of the
  non-attention time but changes floating-point association, so it stays out of a lossless recipe.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: identical frames, peak 75.2 -> 74.1 GB, but 3.3 % slower
  (VAE decode 42 -> 50 s). Off (`results/20261027-expandable-segments/`). Closed.

## Licences

Code: MIT (see LICENSE, NOTICE). vLLM-Omni and vLLM: Apache-2.0. MiniMax H3 weights and outputs: MiniMax H3 Community
License (see MODEL-LICENSE.md). Credits in CREDITS.md.
