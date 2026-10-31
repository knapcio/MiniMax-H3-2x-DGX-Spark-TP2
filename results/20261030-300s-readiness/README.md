# 300-second continuation: readiness and guarded experiments (2026-10-30)

The owner wants good videos at 10, 15, 30, 60 and up to 300 seconds, and faster generation. Existing 10/15 s renders
and the owner's approved 16/30 s painting samples establish useful shorter controls; they do not establish quality
for every image or duration. The 60 s painting request completed in 13,444.6 s and passed media validation, but the
global luma screen detected two repeated-drop intervals and owner review is pending. A 300 s render has **not yet
been completed or visually assessed**.

For the current 24 fps, 277-frame/22-frame-overlap configuration, 300 requested seconds align to **7,213 frames**:
28 full windows and a short 73-frame last window, 29 windows total. At the recent 120 s/step (20 steps) on full
1344x768 windows, the 28 full windows alone imply about 18.7 hours, before the last window, decode and retries.
This is an extrapolation, not a benchmark. The staged 8-hour executor limit cannot accommodate that projection.
One decoded 1344x768 RGB uint8 copy would occupy 20.8 GiB; copies, VAE activations, audio and model state are
additional. This arithmetic is not a measurement of peak memory.

The vLLM-Omni stack includes an opt-in `preencode_mp4` path that emits VAE chunks to the encoder without materializing
the whole uncompressed video. One two-rank 16 s test is described below; memory and correctness at 300 s remain
unverified. The `/v1/videos` async API can outlive a client connection, but the job registry is process-local:
a head restart loses the job metadata and does not resume completed windows. The continuation loop retains accumulated
audio/video latents in memory and does not checkpoint them. A failure near the end can therefore lose hours of work.
Do not present a configured timeout, an async job, or frame-count arithmetic as a successful 300 s test.

## 2026-10-31: preserve the painting's aspect ratio

The painting source is 2316x1320 (1.7545:1). The deployed FL2VA encoder directly resizes the reference to the
requested canvas (`_prepare_encoder_images` in `encoder_processing.py`), so the 960x576 experiment compressed it
horizontally by about 5%. The 16 s test still completed and is useful as a speed/quality reference: 886.1 s wall,
585.9/212.6 s diffusion, seam ratio 0.99, no detected hard cut or repeated global luma drops, and minimum available
head/worker memory 49,377/50,574 MiB. These diagnostics and a shorter runtime do not establish subjective quality.

Before the 960x576 60 s request started, its waiting quality and fallback queues were stopped deliberately and their
rc=143 markers kept as history. The replacement is **960x544**, 0.58% wider relative to the source ratio, with both
dimensions divisible by 32 as required by the deployed model. The 5 s smoke completed successfully (195.5 s wall;
head/worker minimum available 49,229/50,560 MiB). Its native 124-frame output passed full audio/video decode and
global and regional repeated-flicker screens; an exact 120-frame/5.000 s copy was verified and collected. Those
screens and one still do not establish temporal quality. The 16 s test completed in 814.8 s (535.9/197.7 s diffusion),
8.05% faster wall time than the earlier 960x576 sample with the same prompt and seed on the same head process. This
changes the spatial shape, so it is a single resolution comparison, not an output-preserving acceleration. Its seam
ratio was 1.00, no hard cut was detected and head/worker minimum available memory was 49,165/50,123 MiB. However,
two successive >=2 YAVG drops were detected at 12.04–12.75 s, and the regional screen flagged repeated changes through
15.58 s; the 300 s guard therefore cannot pass on this raw 16 s control. The 60 s test started at 15:02 CET.
[`run-aspect-control-20261031.sh`](run-aspect-control-20261031.sh) runs the 5 s and 16 s controls in sequence; the
latter requires a successful smoke and matching source-image SHA-256. The
[`queue-aspect-20261031.sh`](queue-aspect-20261031.sh) then checks the 16 s request and source hash before running
60 s, with quality/memory/time gates for a conditional 300 s. The active v2 queues require a clean repeated global
luma-drop screen on **both** 16 s and 60 s controls before the 300 s attempt; the 60 s detector can still miss
regional flicker, and this is not a visual verdict. The separate
[`queue-aspect-long300-20261031.sh`](queue-aspect-long300-20261031.sh) applies the existing guarded 16 h fallback
only for a time-only skip and after rechecking both controls and their matching source hashes. Their output/event
names are separate; the 960x576 jobs must not be resumed. The aspect controller, queue and fallback passed 4, 5 and
7 fake-host tests respectively; these do not establish GPU validity or visual quality. A separate Mac collector and
duration comparator will verify and copy successful outputs while the Mac is online.

After a verified patch, [`queue-aspect-vae-trace-20261031.sh`](queue-aspect-vae-trace-20261031.sh) waits for idle
nodes and enables the trace in a runtime-only systemd drop-in. A 5 s smoke must actually produce usable trace lines
before a 16 s 960x544, seed-1101 streamed request with the same prompt and source bytes. It records approximate
per-overlap luma, full media and flicker diagnostics, then restores and verifies a trace-off head. A busy or uncertain
head is left untouched on HOLD. Fourteen fake-host tests cover the guards, missing trace and restoration; the GPU
trace remains untested until it runs. The measurements are diagnostic, not proof of the flicker cause.

[`queue-aspect-short-20261031.sh`](queue-aspect-short-20261031.sh) waits for the trace to finish and the normal head
to be restored, checks both deployed file sets and source-image hash, and then runs separate direct 10 s and 15 s
FL2VA requests at 960x544. These do not enable continuation, so their results test the short-request path rather than
trimming the 16 s clip. They use the same image, prompt, seed and step count; each requires an idle, guarded pair and
an empty async-job store. Seven fake-host tests cover the predecessor, source, trace-off, idle, no-overwrite and
failure gates. The direct requests are queued, not yet rendered or visually evaluated.

An independent [`queue-aspect-experimental300-20261031.sh`](queue-aspect-experimental300-20261031.sh) waits for both
direct controls and the trace-off head. The short 16 s control already has measured flicker, so the existing 300 s
**quality** gate remains failed. This separate, explicitly labelled structural experiment asks whether 300 s can
complete and produce decodable media despite that known quality issue; it does not promote settings or assert that
the result looks good. It requires matched 16/60 s inputs, no detected hard cuts, memory samples and guards on both
nodes, six measured 60 s diffusion phases, an 8 h timeout projection with margin, sufficient disk space, verified
deployed files, three idle checks and no queued API jobs. It does not change timeouts or restart either node. The
raw result, if any, is preserved with its flicker and memory measurements. Six fake-host tests cover predecessor,
memory/cut/input, time, busy-node, failed-render and output-validation gates; the 300 s GPU request remains unrun.

The current automatic later-window prompt says "the camera keeps moving the same way" even when the base prompt
specifies a locked tripod; the live 60 s journal confirms that wording. Whether it causes any visible defect is
unknown. After the structural 300 s stage, [`queue-camera-speed-ab-20261031.sh`](queue-camera-speed-ab-20261031.sh)
can run three 30 s, 960x544 requests on the same checked head/agent processes: default prompt/20 steps, an explicit
locked-camera continuation prompt/20 steps, and that same explicit prompt/10 steps. The first comparison changes
the continuation wording; the second changes only step count. The first-window wording and base prompt remain the
same. It requires verified predecessors, the same image SHA, unchanged deployed prompt files, trace off, no custom
server prompt template, memory/disk margin and a guarded idle pair before each request. Five fake-host tests cover
those gates, changes of service PID, request failures and the prompt fields. The GPU comparisons are queued, not
completed; none of the wording or speed settings is promoted without assessment.

After the aspect queue and its 300 s fallback finish or safely skip, the independent
[`deploy-aspect-prompt-20261031.sh`](deploy-aspect-prompt-20261031.sh) can install the staged duration/prompt fix on
both nodes. It requires verified 16/60 s outputs, released queue locks, matching staged/live hashes, healthy idle
nodes, no asynchronous jobs and the expected 8 h or verified 16 h server limits. It then restarts the head and runs
two smokes, including a real 16 s request where `target.duration_seconds=16` must take precedence over
`duration_seconds=30`. Failure handling restores files only in a safely stopped or idle state; an uncertain or
active head is left on HOLD. Sixteen fake-host tests cover the gate, both timeout configurations, busy jobs,
nonquiescent service and rollback. The queue is waiting; the patch and GPU precedence test are not yet complete.

## 2026-10-31: resume after the owner's stop

The owner asked to stop the experiments on October 30 and to continue on October 31. The stopped queues and their
HOLD markers are historical; the resumed run used new output names and a separate event log. Both memory guards,
the head and agent were active, the API was idle and both GPUs were idle before the 16 s 960x576 control started
at 14:29 CET. That test combines instructions for slower, smaller steps, softer ground light without strong moving
cloud shadows, and broad painted cloud shapes. It is a combined experiment, not proof of what caused a change.

The 16 s `quality16-stream` request completed just before the stop, but its queue success marker was never written.
Its HTTP 200, successful STATUS, full video/audio decode and SHA-256 were rechecked before collection: its native MP4
is **byte-for-byte identical** to `quality16-base` (`439b8987208a0a25a344f177e948e10c28545f01bdbd6353ceb2ceece20a502c`,
11,694,843 bytes). The baseline reported `decode 139.2 s`; the stream run reported `decode_to_mp4 141.2 s` and the
API logged a pre-encoded response. Both window latent digests matched. Minimum available head/worker memory was
39,597/43,935 MiB for baseline and 39,677/43,216 MiB for streaming. This establishes the streaming path for this
one 16 s pair, not its peak memory or visual quality on a longer clip.

On the same prior boot, 10 steps instead of 20 cut wall time from 2,423.3 s to 1,237.0 s (48.95%); 175-frame windows
took 1,949.5 s (19.55% less than the baseline). Both are single experimental samples. The shorter windows also
changed the clouds and measured motion more, so neither result has been promoted as a default. The owner's feedback
on earlier variants was that regional deflicker looked better, figures moved a little too fast, and ground cloud
shadows looked questionable. Visual review of the new controls remains pending.
For a closer look, `~/Downloads/h3-prod-smoke/porownanie_20_vs_10_krokow_16s.mp4` shows the two step-count samples
side by side, `porownanie_okien_277_vs_175_16s.mp4` compares the window sizes, and `porownanie_ruchu_postaci_16s.mp4`
enlarges the road in the original and slow-figure-prompt clips.
On sampled stills the figure's path looks similar in those two clips; the prompt alone has not established a clear
improvement in walking speed or naturalness. Both comparisons use the left-hand clip's audio and remain review aids.
Two separately saved exact-length excerpts of the earlier approved 16 s painting clip are also in the same Downloads
folder as `d10_koninck_fragment_z_16s.mp4` and `d15_koninck_fragment_z_16s.mp4`. They contain 240 and 360 frames
respectively; retained decoded frames and AAC packet prefixes were checked. These are trims, not independent 10 s
or 15 s generation tests.

[`run-resume-control.sh`](run-resume-control.sh) started the new 16 s test after three checks of both nodes' health,
GPU idleness, memory guards, available RAM and API gauges, plus an empty async-video job list. It never overwrites an
output. [`queue-resume-20261031.sh`](queue-resume-20261031.sh) waits for that verified output, checks its HTTP result,
prompt, image, shape, seed and streaming settings, and runs a matching 60 s control. It tries a 300 s 960x576 render
only when both controls meet the existing cut, global-flicker, memory and 8 h projection guards. Both queues use
`resume-events-20261031.log`, leaving the earlier stopped queue untouched. The waiting 960x576 60/300 s queues were
later canceled as described above, and replaced by the 960x544 plan. The 16/60/300 s tests and all motion and
light settings are experimental; no duration is marked successful until its output is actually verified.

[`queue-resume-long300-20261031.sh`](queue-resume-long300-20261031.sh) waits independently for that decision.
If and only if the 300 s experiment was skipped for insufficient time, it rechecks both controls, the byte-identical
earlier streaming baseline, memory, quality, job idleness and a 16 h projection. If all pass, it can extend four
timeouts with the existing hash-checked transaction, restart the idle head, and try one 300 s experiment. It restores
the previous limits on failure only when the head is safely idle or stopped; otherwise it holds for diagnosis. A
completed 300 s attempt retains the 16 h limits. There is still no restart checkpoint. The resumed control,
quality queue and fallback passed 5, 5 and 7 fake-host tests respectively; these tests do not substitute for GPU
or long-duration validation. A Mac watcher collects successful clips after checking source/copy hashes, full media
decode and duration; a second watcher records validated events in the workspace `SETUP.md` while the Mac is online.

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

On October 30, [`queue-quality.sh`](queue-quality.sh) was run on Spark_06 as
`~/h3-tp2-results/queue-quality-20261030-v5.sh` (SHA-256 `36c4d6e6a6cfacd06cf5cc7d911a52f1fb0e579f3ee7bc81a441ab69d2430467`,
process recorded in the workspace `SETUP.md`). The owner subsequently stopped it; it is not running now. Its
fake-host tests cover failed guards, predecessor verification,
no-overwrite, stopping on failure, and the conditional 300 s decision. The queue waits for the
earlier 60 s chain to finish **with a verified MP4**. It verifies both services and memory guards, API request gauges,
GPU idleness and available memory on both hosts before every experiment; it stops on failures and never restarts a
service or overwrites an output. The planned controls on the same boot are: the original 16 s painting request, the
same request with 10 rather than 20 denoising steps, the same request with 175-frame windows, the original request
with only the figures told to move very slowly, and 5/16 s
tests of the chunked MP4 path. Matched 16 s and 60 s 960x576 chunked tests combine soft light, slower figures and
an instruction to keep the clouds' broad painted shapes, giving a longer quality and memory check before the 300 s
decision. This combined prompt is an experiment; it does not isolate which wording caused any difference.
`compare_speed.py` was used for the matched window-size and step-count controls. The comparators
check that prompt, seed, image and other request fields match, and write `quality-speed-comparison-20261030.json`
and `quality-steps-comparison-20261030.json` with wall time, per-window times, memory minima and diagnostic quality
signals. Ten steps might be faster but may degrade the image; neither single comparison justifies changing a default.
[`compare_length.py`](compare_length.py) was designed for the matched 960x576 16/60 s streaming controls, checks that
only duration differs, and reports their first 277-frame window times in `quality-length-comparison-20261030.json`.
This is one same-boot pair with potentially different cache/runtime state, so it cannot by itself establish why the
original 1344x768 60 s request slowed down. The comparator is staged on Spark_06 as
`compare-quality-length-20261030.py` (SHA-256 `fd82725a97cf5f200dd3e196272bee1d8ec1555ae21d7375bbfcb10c339d5246`).

Only if those tests succeed, both 960 clips have no detected hard cut, the 16 s clip has no repeated global luma
drops, both nodes' traces for both clips stay at or above 8 GiB, and 29 times the slowest measured 60 s window plus a 35%
margin and 30 minutes for decode fit within 95% of the active
executor/wave limit, the queue attempts an **experimental** 300 s 960x576 chunked render. The model can still drift,
shadows can look wrong, and the guard does not predict all memory peaks or guarantee completion. It does not create
restart checkpoints. All outputs need visual review, especially the final minute and motion of figures. Quality
metrics are diagnostic, not an automatic promotion criterion.

The first queued version had a guard bug: the live `0ed86ba` `clip-metrics.sh` does not produce the newer periodic
luma report, so its 300 s test would always have been skipped. Before the quality queue started any request, the
waiting process was replaced by v2. That version runs the separately staged, SHA-256-pinned `luma_flicker.py` on
the 960 control with `--minimum 2`, records `luma-flicker.txt` and permits the attempt only on a confirmed clean
report. Missing, modified or failed detector means skip, not pass. Fake-host tests exercise the actual detector with
clean frames, two repeated drops, a failed decoder and a missing detector; the other limits still apply. A clean
global screen does not prove that regional flicker or cloud shadows look right.
On the already completed 60 s original, the standalone detector with this two-drop threshold found an additional
short interval at 34.71–35.42 s, in addition to the two intervals reported with the original three-drop threshold.
Before it started any quality request, v2 was replaced by v3 to add the matched 10-step control; v4 adds the 60 s
960 control and uses its slowest window for the 300 s projection. Before the quality queue started, v5 added the
cloud-shape wording after a still-frame inspection of the original 60 s clip. The 60 s luma screen is saved for
inspection, but is not a pass/fail criterion for the 300 s attempt. The first
speed-comparison watcher had the same missing-luma-report assumption; the v2 comparator instead runs the pinned
standalone detector itself on both verified clips. It was checked against the real 60 s clip and the old metrics
format, and it reports rather than promotes either setting.

## Staged prompt fix and deferred deployment

The files at `fc14580` are staged outside the live trees in `~/h3-tp2-next-fc14580` on both nodes; the live trees
still report `0ed86ba`. The prompt-wrapper, reference-lock and disabled-by-default VAE trace tests passed on CPU on
each node with CUDA hidden (11/11 per node). `file_txn.py` checks old and staged SHA-256s, backs up only the five
affected files, performs atomic file replacements and refuses to overwrite unexpected changes during rollback. Its
tests cover a partial failure, bytecode-cache invalidation, damaged inputs and rollback refusal.

`deploy-prompt-fc14580.sh` was queued separately on Spark_06 as `deploy-prompt-fc14580-v3.sh` after the quality chain;
it was stopped at the owner's request and is not running.
It requires a verified 960
control, a completed or safely skipped 300 s experiment, release of the quality queue's lock, healthy/idle services
and memory guards on both nodes, and an empty async-video job store before stopping the head. It installs the five
staged files on both nodes, leaves the trace off, restarts the head and performs a 5 s smoke plus a real 16 s request
whose `target.duration_seconds=16` overrides `duration_seconds=30`. That request must produce the two expected
window prompts. Before touching files, the queue requires systemd to report the head inactive or failed with MainPID 0.
Startup failures trigger a verified rollback only if the new head is stopped or verifiably idle with an empty async
job store; an activating or uncertain head is left untouched on HOLD. A failure after the head becomes healthy also
leaves the service untouched on HOLD. The serving tree's `.synced-rev` remains `0ed86ba` because
this is a five-file patch, not a full-tree sync. The deployment was stopped and is **not performed or GPU-qualified**.
The idleness checks cannot reserve the API against an unrelated client arriving between the final check and stop.

## Deferred 16-hour fallback for a 300 s test

[`queue-long300.sh`](queue-long300.sh) was waiting as `queue-long300-20261030-v3.sh` on Spark_06 (SHA-256
`bf3a22f9e776a65b39252a74bf04702e5290a836224b3215070a4ba9a6579f6a`); the owner stopped it. If the first 300 s experiment succeeds,
it does nothing. If that experiment is skipped for quality, memory or a missing detector, it also does nothing.
Only when it was skipped for insufficient time does the fallback recheck the verified 5/16/60 s streaming controls,
the 16/60 s 960x576 controls' exact prompt/settings, no hard cuts, no two successive large global luma drops in the
16 s control, and both pairs of memory traces. It requires `29 * slowest 60 s window * 1.35 + 1800 s` to fit within
95% of a 16-hour limit. This is an
estimate, not a guarantee against slowdown, drift or memory exhaustion.

After the prompt patch succeeds and both nodes are verified idle with an empty async job store, the fallback uses
[`timeout_env_txn.py`](timeout_env_txn.py) to change only four numeric timeouts from 8 h to 16 h plus staggered
API/client margins. It restarts the head, verifies the effective limits and tries the 960x576 300 s chunked request.
The transaction metadata contains only timeouts and environment-file hashes, not the other configuration contents;
it refuses unknown changes and can reconstruct the original file byte-for-byte. On render or startup failure the
old timeouts are restored only when safely possible; on an uncertain or busy head it holds without stopping it.
After a verified 300 s render and diagnostic reports, it retains the 16 h timeouts for subsequent long requests.
The 12 fake-host queue tests and 6 transaction tests cover quality/time gates, no-overwrite, busy jobs, rollback,
changed configuration/helper and nonquiescent startup. The actual 300 s test is still conditional, has no restart
checkpoint, and neither a completed file nor a clean global screen proves the long-term image looks good.

## Deferred VAE diagnostic

[`queue-vae-trace.sh`](queue-vae-trace.sh) was waiting on Spark_06 as `queue-vae-trace-20261030-v2.sh` (SHA-256
`8c86b948eb62836b26b971866596fce9780ba2d471f9e2be43fd66428db6de43`); the owner stopped it. It runs only after the patch and both of
its verified smoke clips finish, and after the 300 s fallback finishes or is skipped and both queue locks are released.
It validates the deployed files on both hosts,
requires an empty async-job store plus healthy, idle, sufficiently free nodes, and temporarily enables the trace via
a runtime-only systemd drop-in. A 5 s smoke must actually emit usable blend logs before it runs the user's 30 s,
345/124-frame history case. It saves decoded video, luma and trace reports. The sampled tile luma and approximate
frame numbers are investigative evidence, not final-frame measurements or proof of a flicker cause.

On success the queue removes its own verified drop-in and restarts the normal head with tracing off. On failure it
attempts the same restoration only if the head is stopped or verifiably idle and no async jobs exist; a changed
drop-in, busy/uncertain head or failed restoration is left untouched on HOLD. The temporary systemd setting does
not survive host reboot. An uncontrolled termination at the wrong moment can still require manual recovery, and
no idle check can reserve the API against a new unrelated request. The 13 fake-host tests cover those guards,
restoration, startup failure and missing trace; they do not verify the GPU hook. The Mac collector watches for the
verified diagnostic video and reports while it is online. The trace and diagnostic render are not new quality defaults.
