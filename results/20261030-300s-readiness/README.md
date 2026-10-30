# 300-second continuation: readiness and guarded experiments (2026-10-30)

The owner wants good videos at 10, 15, 30, 60 and up to 300 seconds, and faster generation. Existing 10/15 s renders
and the owner's approved 16/30 s painting samples establish useful shorter controls; they do not establish quality
for every image or duration. The current 60 s painting request is still running, with a four-hour timeout risk. A
300 s render has **not yet been completed or visually assessed**.

For the current 24 fps, 277-frame/22-frame-overlap configuration, 300 requested seconds align to **7,213 frames**:
28 full windows and a short 73-frame last window, 29 windows total. At the recent 120 s/step (20 steps) on full
1344x768 windows, the 28 full windows alone imply about 18.7 hours, before the last window, decode and retries.
This is an extrapolation, not a benchmark. The staged 8-hour executor limit cannot accommodate that projection.
One decoded 1344x768 RGB uint8 copy would occupy 20.8 GiB; copies, VAE activations, audio and model state are
additional. This arithmetic is not a measurement of peak memory.

The vLLM-Omni stack includes an opt-in `preencode_mp4` path that emits VAE chunks to the encoder without materializing
the whole uncompressed video. Its two-rank correctness, visual equivalence, memory and speed for this deployment
remain unverified. The `/v1/videos` async API can outlive a client connection, but the job registry is process-local:
a head restart loses the job metadata and does not resume completed windows. The continuation loop retains accumulated
audio/video latents in memory and does not checkpoint them. A failure near the end can therefore lose hours of work.
Do not present a configured timeout, an async job, or frame-count arithmetic as a successful 300 s test.

## Duration-field correctness

The running stack's `resolve_minimax_h3_shape` selects `target.duration_seconds`, then `duration_seconds`, then
`duration`, with an explicitly present null taking the `num_frames`/default path. The continuation-prompt helper and
`scripts/render.sh` previously used a different precedence: for example, a request containing `duration=30` and
`target.duration_seconds=300` could generate three text prompts while the model planned 29 windows, and be rejected.
The local fix matches upstream precedence and does not invent a duration for explicit nulls. The script's timing JSON
also records `effective_duration: null` in that case. No change is loaded into the live serving processes yet.

When a request supplies `num_frames` without an effective duration, upstream uses that frame count rather than a
duration. The local plugin now generates the matching number of continuation prompts in this case too: for example,
7,200 requested frames require 29 prompts. Without an explicit duration or frame count, the encoder's short default
is left alone. Caller-supplied continuation prompts remain intact.

Verification: 17,766 plans covering 4.0–300.0 s at 0.1 s increments, three window sizes and two overlaps matched the
upstream planner copied from Spark_06; 42,630 direct-frame plans and 80 duration-field combinations, including nulls,
also matched its source. An isolated test of the plugin's actual prompt wrapper checks 7,200 frames, duration
precedence and supplied prompts.
The actual `render.sh` was also exercised with fake `curl`: 300 s produces 29 prompts, explicit null produces no
wrong-length list, invalid target is rejected before submission, and supplied prompts remain intact. This is a CPU
test, not an end-to-end 300 s render.

## Deferred follow-up on the pair

[`queue-quality.sh`](queue-quality.sh) is the exact script running on Spark_06 as
`~/h3-tp2-results/queue-quality-20261030.sh` (SHA-256 `8abffafb7d2a7b0d1d77cd4bc7a334a96b6152f58cc451158607264565fd2f8a`,
process recorded in the workspace `SETUP.md`). Its fake-host tests cover failed guards, predecessor verification,
no-overwrite, stopping on failure, and the conditional 300 s decision. The queue waits for the
earlier 60 s chain to finish **with a verified MP4**. It verifies both services and memory guards, API request gauges,
GPU idleness and available memory on both hosts before every experiment; it stops on failures and never restarts a
service or overwrites an output. The planned controls on the same boot are: the original 16 s painting request, the
same request with 175-frame windows, the original request with only the figures told to move very slowly, and 5/16 s
tests of the chunked MP4 path. A 16 s 960x576 chunked test combines the soft-light and slow-figures prompts.
`compare_speed.py` waits on Spark_06 for both verified 16 s window controls, checks that prompt, seed, image and
other request fields match, and writes `quality-speed-comparison-20261030.json` with wall time, per-window times,
memory minima and diagnostic quality signals. A single comparison does not justify changing the default.

Only if those tests succeed, the 960 test has no detected hard cut or repeated global luma drops, both memory traces
stay above 8 GiB, and 29 first-window times plus a 35% margin and 30 minutes for decode fit within 95% of the active
executor/wave limit, the queue attempts an **experimental** 300 s 960x576 chunked render. The model can still drift,
shadows can look wrong, and the guard does not predict all memory peaks or guarantee completion. It does not create
restart checkpoints. All outputs need visual review, especially the final minute and motion of figures. Quality
metrics are diagnostic, not an automatic promotion criterion.

## Staged prompt fix and deferred deployment

The files at `fc14580` are staged outside the live trees in `~/h3-tp2-next-fc14580` on both nodes; the live trees
still report `0ed86ba`. The prompt-wrapper, reference-lock and disabled-by-default VAE trace tests passed on CPU on
each node with CUDA hidden (11/11 per node). `file_txn.py` checks old and staged SHA-256s, backs up only the five
affected files, performs atomic file replacements and refuses to overwrite unexpected changes during rollback. Its
tests cover a partial failure, bytecode-cache invalidation, damaged inputs and rollback refusal.

`deploy-prompt-fc14580.sh` is queued separately on Spark_06 after the quality chain. It requires a verified 960
control, a completed or safely skipped 300 s experiment, release of the quality queue's lock, healthy/idle services
and memory guards on both nodes, and an empty async-video job store before stopping the head. It installs the five
staged files on both nodes, leaves the trace off, restarts the head and performs a 5 s smoke plus a real 16 s request
whose `target.duration_seconds=16` overrides `duration_seconds=30`. That request must produce the two expected
window prompts. Startup failures trigger a verified rollback before the old head is restarted; a failure after the
head becomes healthy leaves the service untouched on HOLD. The serving tree's `.synced-rev` remains `0ed86ba` because
this is a five-file patch, not a full-tree sync. The deployment is **queued, not yet performed or GPU-qualified**.
The idleness checks cannot reserve the API against an unrelated client arriving between the final check and stop.
