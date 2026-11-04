# 2026-11-03: Ulysses exchange without the removable permute copies, stock layouts kept (`H3_TP2_ULYSSES_EXACT`)

Speed plan item 5, second attempt. The first attempt (`H3_TP2_ULYSSES_NOCOPY`, `h3_tp2/ulysses_nocopy.py`) was closed
on 2026-11-02 because its 5 s 1344x768 T2VA latents differed from stock (rel-l2 0.199, every element) and it was no
faster. This attempt keeps every operand that reaches cuDNN or NCCL in exactly the layout the stock path produces and
moves only the producers. **Status: implemented, opt-in, every piece bitwise on the GPU offline (18 GPU tests, 15 CPU
tests); the serving proof (digest `4d6d69e4aad0`) is still to run, steps below.**

Spark_06, GB10 sm_121, torch 2.13.0+cu130, cuDNN 9.20.0.48 (the pinned one), vLLM-Omni v0.30.0 `a8576ccb7`. All GPU
work ran next to the serving head (the head was idle at first, then rendered queue56's 960x544 job; nothing was
restarted, clocks untouched). Peak GPU memory per run under 2.3 GB, except one cuDNN probe at S = 86016 (8.2 GB, a few
seconds).

## Why the nocopy exchange differed: not what the 11-02 note guessed

The 11-02 note blamed either cuDNN reading the strided packed-QKV views or the per-destination launches of the Triton
norm+RoPE kernel. Both are bitwise identical to stock on the GPU, and so is nocopy's whole exchange:

| check (Spark_06 GPU, bf16, 28 local heads x 128) | result |
|---|---|
| cuDNN SDPA (`SDPBackend.CUDNN_ATTENTION`, the serving backend's call) on q/k/v as strided views of a packed [S, 3, 28, 128] buffer vs contiguous copies, S = 4096, 8192, 43008, 86016, with and without the `valid_kv_length` K/V slice (S-37) | **bit-equal**, every case; output strides identical (dense BSHD) |
| the same, S = 37504, 38000, 38106, 38400, 38912, 39000 (around the 5 s calibration's length), K/V slices S, S-1, S-37, S-600 | **bit-equal**, 24 of 24 |
| q-only strided, k/v-only strided | bit-equal |
| `_rms_norm_rope_kernel` launched once per destination (28 heads into the [M, 3, 28, 128] slot, nocopy's launch) vs the stock single launch (56 heads) + stock pack, rope table fp32 and bf16, M = 4096 | **bit-equal** (q, k, v slots) |
| nocopy's full exchange (producer redirect, one combined all-to-all, cuDNN on the strided views, reverse) vs vLLM-Omni's own `all_to_all_4D` + `CuDNNAttentionImpl.forward_cuda` on two thread-ranks with the collective emulated byte for byte, M = 4096 and 2048 per rank, K/V slice | **bit-equal**, both ranks (`test_nocopy_exchange_is_bitwise_too`) |

So the 0.199 does not come from the exchange math, the layout, the head-slice order or cuDNN's treatment of strides.
What the offline runs cannot model is the live two-node path: the real NCCL all-to-all over RoCE (nocopy sent one
message per peer three times the stock size, ~410 MB at the calibration) and the in-model wiring (redirected producer,
stride-0 placeholders, the S-major reverse). The shape of the difference also argues against a layout bug: it is
uniform over the 37 latent frames and the 48x84 positions (0.18-0.26 per frame, 0.10-0.26 per row), the audio latents
differ by only 0.035, and the clip statistics (motion, light, colour per second) match stock to the digit; a wrong head
or row mapping wrecks the frames. It looks like a small numerical perturbation amplified over 20 steps, the class of
the cuDNN 9.20 -> 9.27 kernel change (`results/20261028-cudnn-927`, SSIM 0.848), but the source was not found offline.
Root cause: **not identified**; the candidates that were blamed are excluded, the remaining ones are only reachable in
serving, and the proof below bisects them.

## What the exact module does (`h3_tp2/ulysses_exact.py`)

Stock strict Ulysses at batch 1, per attention layer and rank (M local rows, P = 2, 56 heads, hl = 28, d = 128; one
[M, 56, 128] bf16 tensor = 14336*M bytes = 614 MB at M = 42848, ~273 MB at the 5 s calibration's M ~ 19k):

| # | stock copy (`vllm_omni/diffusion/distributed/comm.py`) | exact |
|---|---|---|
| 1 | q pack `:44` rope output -> `input_t` [P, M, hl, d] | **gone**: the q producer writes straight into `send_q` = stock's `input_t` |
| 2 | k pack `:44` | **gone**, the same |
| 3 | v pack `:44` from the strided v view of the qkv GEMM output | **gone** with the consumer-fused producer (v's scale pass writes `send_v`); kept (stock's copy) with the upstream producer |
| - | the three receives `:57-60` | views for batch 1 in both (q/k/v reach cuDNN as contiguous [1, P*M, hl, d]: same tensors, same strides as stock) |
| 4 | reverse pack `:74-80` [1, P*M, hl, d] -> [P, hl, M, 1, d] | **gone**: the attention output is sent as it is (its row blocks are the per-destination chunks; inside a chunk the bytes are row-major instead of head-major) |
| 5 | reverse unpack `:93-96` [P, hl, M, d] -> [1, M, 56, d] | **kept** (one copy into the out-projection's input): the out-projection reads rows that interleave both sources, which `all_to_all_single` cannot write; removing it needs the out-projection's activation quant to read the receive buffer (item 6 territory, the fp8 modules) |

Removed per layer and rank with the serving producer: 4 x 14336*M bytes written and as many read (2.46 + 2.46 GB per
layer at M = 42848, 123 + 123 GB per 50-layer DiT forward; at the calibration ~1.1 + 1.1 GB per layer). With the
upstream producer (consumer fuse off) 3 of the 4. The q/k producer output also no longer lives through the attention
(the model holds stride-0 placeholders), so the attention-time peak should drop by ~2 x 14336*M bytes; the proof records
`x-peak-memory-mb`. The NCCL traffic is unchanged: three forward all-to-alls and one reverse per layer, the same
message sizes in the same order as stock (asserted in `tests/test_ulysses_exact.py`). Extra cost: 3 more small kernel
launches per layer (per-destination producer launches).

Producers. Since 2026-11-03 the serving qkv producer is the consumer-fused one (`fp8_consumers.qk_norm_rope_fused`:
q/k read from the fp32 GEMM product by a copy of the upstream kernel, v scale-passed into the GEMM placeholder). The
exact module wraps that function and makes the same three launches (`_scale_rms_norm_rope_kernel` for q and k,
`_scale_cols_kernel` for v), once per destination block (28 heads / 3584 columns each, the same kernels, launch
shapes and constexprs) into `send_q` / `send_k` / `send_v`. Without the consumer fuse it wraps the transformer's
`fused_qk_norm_rope` and launches the upstream `_rms_norm_rope_kernel` per destination (`ulysses_nocopy._triton_launch`,
shown bitwise above). The hooks are attached on the first wrapped attention call, after the fp8 sections of the plugin
have run their own source checks (`fp8_consumers` refuses its qkv and out sites when the rope producer or the
attention forward is already wrapped at install time; with the lazy hook its sites install as usual). The
consumer-fuse self-check (64 rows inside the armed forward) is never redirected (row-count guard, tested).

Modes: `H3_TP2_ULYSSES_EXACT=1` (all of the above), `=fwd` (forward side only; the reverse is stock's
`all_to_all_4D`), `=verify` (diagnostic: producers stay stock; every eligible call runs the stock exchange + attention +
reverse AND the exact one on the same q/k/v, compares the received q, k, v, the attention output and the result
bitwise, logs every differing call and a tally every 500, and returns the STOCK result). The plugin (section 6e)
installs it instead of the overlap when both are set; the env is in `scripts/queue-lib.sh`'s STRIP list.
Fallbacks: anything not strict Ulysses / batch 1 / no mask / no joint tensors / no ring / no paged KV / no VSA gate /
heads % P == 0 runs the stock forward; a redirected layer whose attention turns out ineligible first materialises
q, k (and v into the model's view) exactly (logged once as `first materialize`).

## GPU and CPU tests

GPU, Spark_06 (`tests/test_ulysses_exact_gpu.py`, from an isolated copy of the tree, PYTHONPATH pointing at it, the
serving env sourced): **18 passed in 11.2 s** (with the CPU file in the same session: 33 passed).

- upstream producer per destination vs `fused_qk_norm_rope` + stock pack, M = 4096 / 4100, table fp32 / bf16: bitwise;
- consumer-fused producer per destination vs `fp8_consumers.qk_norm_rope_fused` + stock pack (q, k and v), with and
  without bias, M = 4096 / 4100: bitwise;
- two thread-ranks, collective emulated: exact forward vs vLLM-Omni's `all_to_all_4D` + `CuDNNAttentionImpl.forward_cuda`
  with a `valid_kv_length` slice, for producers {pack, upstream, consumer-fused} x reverse {exact, stock}: bitwise, 4
  collectives per rank; `verify_forward` reports all five tensors equal and returns stock's result;
- the two nocopy suspects (table above): bitwise.

CPU (`tests/test_ulysses_exact.py`, Mac torch 2.15 and Spark_06 torch 2.13): **15 passed** (with
`tests/test_ulysses_nocopy.py`: 28 passed). Exchange views equal stock's tensors including strides; the send buffer is
stock's `input_t`; reverse; the producers' slicing (the consumer-fused kernels replaced by eager stand-ins with the
same pointer arithmetic); end to end for all producers and both reverses incl. message sizes; verify flags a wrong
reverse; wiring (redirect, self-check guard, fallback materialising q/k/v, unarmed / VSA / odd-heads cases). Three
mutations of the module (v column offset, reverse unpack order, rope head slice) each fail 4-11 tests. An install smoke
test against the real vLLM-Omni modules (CPU) confirms the wrappers attach once, the consumer wrapper calls the
patched module global, and a mode change is refused.

Commands:

    # Mac (CPU), any torch env with pytest
    python -m pytest tests/test_ulysses_exact.py tests/test_ulysses_nocopy.py -q
    # Spark_06 (GPU), from a copy of the tree that is NOT the synced serving tree
    set -a; . ~/.h3-tp2.env; . ~/h3-tp2/scripts/env.sh; set +a; export PATH=/usr/local/cuda/bin:$PATH PYTHONPATH=/tmp/ux-exact
    cd /tmp/ux-exact && ~/h3-long/.venv/bin/python -m pytest tests/test_ulysses_exact_gpu.py -q

## Serving proof: PASS (queue59, 2026-11-03 14:04-15:08, both Sparks, gated compacted boots)

| arm | boot | 5 s 1344x768 T2VA calibration (seed 0, 20 steps) | digest | s/step | diffuse / decode |
|---|---|---|---|---|---|
| A defaults | 14:17 | q59-A-default | `4d6d69e4aad0` | 21.21 | 424.3 s / 26.5 s |
| B `H3_TP2_ULYSSES_EXACT=1`, request 1 | 14:39 | q59-B-exact-1 | `4d6d69e4aad0` | 20.16 | 403.2 s / 26.2 s |
| B, request 2 (same boot) | | q59-B-exact-2 | `4d6d69e4aad0` | 20.45 | 409.1 s / 26.3 s |

- `latents_diff.py A B1`: video latents (1, 24, 37, 48, 84) and audio latents (2, 32, 207) **bit-equal** (max|d| 0).
  The second exact request is bit-identical as well (same digest, same framemd5 `c6f54e83`), so unlike the cutlass
  GEMM level the exact exchange has no second-request hazard.
- Journal (head): `Ulysses exact exchange installed (mode 1)`, `producer = consumer-fused qkv (q, k and v written per
  destination)`, `first redirect_fused`, `consumer fuse requested=1, installed sites: qkv, out, fc2` unchanged, exactly
  one `first fallback` (the token refiner, as expected), no `first materialize` / `first out_copy` / `pack`.
- Speed: -3.6 % and -5.0 % against the defaults boot of the same queue. The two boots were different boots (the 11-02
  compacted-boot spread was 1.2 %), so the gain is real but its size is 3-5 %, to be pinned by the same-boot pair of
  the default gate below. The motion/light/detail metrics, the 17-frame flicker check and the boundary steps of the
  three clips are identical.
- Boot E restored the defaults (probe 8.63 / warm 8.22 s/step, `bf6e85687a6a`), serving healthy.

Default gate (queue63, after the X clip): 16 s 1344x768 FL2VA continuation (koninck, seed 1101) on a defaults boot
and on an exact boot, framemd5 + per-window latents digests equal; the exact boot's 960x544 probe digest must be
`bf6e85687a6a`. Pass = `H3_TP2_ULYSSES_EXACT=1` becomes the default.

## Serving proof: the script as run (queue59 = this with Q58 -> Q59 and the LALKA-B-DONE wait)

Note first: `H3_TP2_A2A_OVERLAP` is not in queue-lib's STRIP list, so the `export H3_TP2_A2A_OVERLAP=2` that queue50's
P arm wrote is still in `~/.h3-tp2.env`: every head boot since 2026-11-02 19:14 runs the overlap with 2 head groups
(journal: `Ulysses all-to-all overlap installed`). It is bit-equal to stock (queue50) and all `4d6d69e4aad0` digests
since were made with it, so the defaults arm below includes it; with `H3_TP2_ULYSSES_EXACT` set the plugin installs the
exact exchange instead. Whether the overlap should stay in serving is a separate decision (it measured no speed gain).

Before: sync the reviewed tree to both Sparks while no render runs (`H3_VENV=/home/knapcio/h3-long/.venv
scripts/sync-to-nodes.sh Spark_06 Spark_05`). Then on the head, `~/h3-tp2-results/queue58-ulysses-exact.sh`:

```bash
#!/bin/bash
# queue58-ulysses-exact (2026-11-03): serving proof of H3_TP2_ULYSSES_EXACT (speed plan item 5, second attempt).
#   A: defaults, gated boot, 5 s 1344x768 T2VA calibration (seed 0, 20 steps): digest 4d6d69e4aad0, s/step, latents
#   B: + H3_TP2_ULYSSES_EXACT=1, gated boot, the calibration twice: digest unchanged both times, s/step from the second,
#      latents bit-equal to A (latents_diff), exact/fp8 log lines
#   only if B's digest differs: C: =fwd (forward side only), D: =verify (per-call stock-vs-exact comparison in the log)
#   E: defaults again
. ~/h3-tp2/scripts/queue-lib.sh
DEFAULT_ENV="export H3_TP2_SAVE_LATENTS=/home/knapcio/h3-tp2-results/latents
export H3_TP2_WINDOW_SEED_STEP=1"
WANT=4d6d69e4aad0
PY=~/h3-long/.venv/bin/python
unset CONT_PROMPTS CONT_PROMPTS_EXTRA AUDIO_URL EXTRA_JSON; export TASK=t2va
export PROMPT="A cinematic shot of a forest stream at golden hour, leaves swaying in a light breeze, sunlight flickering through the canopy, gentle water sounds."
calib() {   # calib <name>: one calibration; sets DIG / RATE / LAT
  local T; T=$(date "+%Y-%m-%d %H:%M:%S"); run "$1" 5 "5:1344:768:20:0"
  RATE=$(journalctl --user -u h3-tp2-head.service -a --since "$T" --no-pager -o short | tr "\r" "\n" | grep -o -E "20/20 \[[0-9:]+<00:00, +[0-9.]+s/it\]" | tail -1)
  DIG=$(journalctl --user -u h3-tp2-head.service --since "$T" --no-pager | grep -o -E "latents sha256 [0-9a-f]{12}" | tail -1 | awk '{print $3}')
  LAT=$(journalctl --user -u h3-tp2-head.service --since "$T" --no-pager | grep -o "h3_tp2 latents saved to [^\" ]*" | tail -1 | awk '{print $5}')
  echo "CALIB $1: $RATE latents $DIG $LAT $(journalctl --user -u h3-tp2-head.service --since "$T" --no-pager | grep -o -E "phase (diffuse|decode)[_a-z0-9]* [0-9.]+ s" | tr "\n" " ")" | tee -a gen.log
}
lines() {   # exchange + fp8 consumer lines of the running boot (head; rank 1 logs to ~/h3-tp2-logs on Spark_05)
  journalctl --user -u h3-tp2-head.service --since "-30min" --no-pager | grep -o -E "Ulysses (exact|all-to-all overlap)[^\"]{0,90}|ulysses exact[^\"]{0,220}|consumer fuse requested=[^\"]{0,60}" | sort | uniq -c | tail -14
}
for i in $(seq 1 900); do grep -q "${WAIT_FOR:-CUTLASS-PROOF-DONE}" gen.log 2>/dev/null && break; sleep 30; done
grep -q "${WAIT_FOR:-CUTLASS-PROOF-DONE}" gen.log || { echo "Q58 gave up waiting $(date +%H:%M:%S)"; exit 1; }
echo "Q58-START $(date +%H:%M:%S)" | tee -a gen.log
rm -f ~/h3-tp2-results/.head-rev            # new code at the same rev: force the first restart
restart_head_gated "$DEFAULT_ENV"; echo "Q58-BOOT-A rc $? $(date +%H:%M:%S)" | tee -a gen.log; lines
OK=1
if [ "$(health)" = 200 ]; then calib q58-A-default; DA=$DIG; LA=$LAT
  [ "$DA" = "$WANT" ] || { echo "Q58 STOP: default digest $DA != $WANT" | tee -a gen.log; OK=0; }
else echo "Q58 STOP: head not healthy after boot A" | tee -a gen.log; OK=0; fi
if [ $OK = 1 ]; then
  restart_head_gated "$DEFAULT_ENV
export H3_TP2_ULYSSES_EXACT=1"; echo "Q58-BOOT-B rc $? $(date +%H:%M:%S)" | tee -a gen.log; lines
  if [ "$(health)" = 200 ]; then
    calib q58-B-exact-1; D1=$DIG; L1=$LAT
    echo "A vs B1:"; $PY ~/h3-tp2/scripts/latents_diff.py "$LA" "$L1"
    if [ "$D1" = "$WANT" ]; then calib q58-B-exact-2; D2=$DIG
      [ "$D2" = "$WANT" ] || echo "Q58 STOP: exact digest (second) $D2 != $WANT" | tee -a gen.log
      lines
    else
      echo "Q58: exact digest $D1 != $WANT, bisecting" | tee -a gen.log
      restart_head_gated "$DEFAULT_ENV
export H3_TP2_ULYSSES_EXACT=fwd"; lines
      [ "$(health)" = 200 ] && { calib q58-C-fwd; echo "A vs C:"; $PY ~/h3-tp2/scripts/latents_diff.py "$LA" "$LAT"; }
      restart_head_gated "$DEFAULT_ENV
export H3_TP2_ULYSSES_EXACT=verify"; lines
      [ "$(health)" = 200 ] && { T=$(date "+%Y-%m-%d %H:%M:%S"); calib q58-D-verify
        journalctl --user -u h3-tp2-head.service --since "$T" --no-pager | grep -o -E "ulysses exact verify[^\"]{0,200}" | sort | uniq -c | sort -rn | head -12; }
    fi
  else echo "Q58 STOP: head not healthy after boot B" | tee -a gen.log; fi
fi
restart_head_gated "$DEFAULT_ENV"; echo "Q58-BOOT-E (defaults restored) rc $? health $(health) $(date +%H:%M:%S)" | tee -a gen.log
echo "Q58-DONE $(date +%H:%M:%S)" | tee -a gen.log
echo "ULYSSES-EXACT-PROOF-DONE $(date +%H:%M:%S)" >> gen.log
```

Pass criteria for B: digest `4d6d69e4aad0` on both calibrations, `latents_diff.py A B1` bit-equal True for every tensor;
in the log `Ulysses exact exchange installed (mode 1)`, `producer = consumer-fused qkv`, `first redirect_fused`,
`consumer fuse requested=1, installed sites: qkv, out, fc2` (unchanged), `calls so far redirect_fused=...` with
`fallback` only for the token refiner, `pack` at most 1 (only if a process's very first attention call is a DiT block:
the hooks attach during that call) and NO `first materialize` or `first out_copy`. Speed: B's second
s/step against A (same queue, compacted gated boots; boot-to-boot spread was 1.2 % on 11-02, so a gain below ~1 % needs
a second A/B pair). If B differs: C equal to A localises the difference to the reverse (S-major send); D's per-call
log names the tensor (q/k/v = the exchange under real NCCL, attn = cuDNN, result = the reverse) while its digest must
be A's (it returns stock results).

## Default gate: PASS (queue63, 2026-11-04 00:00-02:19, both Sparks, gated compacted boots)

Arm A = serving defaults, arm B = `H3_TP2_ULYSSES_EXACT=1`; each arm on its own gated boot: the 960x544 boot probe, the
5 s 1344x768 calibration (seed 1101) and a 16 s 1344x768 FL2VA continuation (two windows, the painting prompt, seed 1101,
`EXTRA_JSON` continuation).

| check | A (defaults) | B (exact) |
|---|---|---|
| boot probe | PASS, warm 8.84 s/step, latents `bf6e85687a6a` | PASS, warm 8.22 s/step, latents `bf6e85687a6a` |
| 1344 calibration digest / framemd5 | `4d6d69e4aad0` / `c6f54e83dac8` | `4d6d69e4aad0` / `c6f54e83dac8` |
| 1344 calibration s/step | 26.59 (slow boot: probe 9.18 first) | 22.05 |
| 16 s continuation digests (window 1, 2) / framemd5 | `200e38721c9e` `4138eba5a964` / `e46129c05cc6` | `200e38721c9e` `4138eba5a964` / `e46129c05cc6` |
| continuation window 1 / 2 s/step, diffuse s | 89.0 / 27.1, 1795.7 + 542.1 | 85.1 / 29.4, 1705.2 + 589.0 |

Every digest and every decoded frame is identical between the arms, on the probe, the calibration and both continuation
windows, so the exchange is exact in serving on a continuation request too (the hand-over path, the id-lock and the seam
retries all read the same latents). Speed on these two boots: window 1 -4.4 %, calibration -17 % (arm A landed on a slow
boot, so only the queue59 pair, 20.16/20.45 vs 21.21 s/step on compacted boots, is the speed number). Default since
2026-11-04 02:25 (`h3_tp2/plugin.py`, `H3_TP2_ULYSSES_EXACT=0` restores the stock exchange).

## What is left

- DONE 15:08: the serving proof passed; DONE 11-04 02:19: the default gate passed (tables above); default since 02:25.
- Copy 5 (reverse unpack) stays; removing it means the out-projection's activation quant reading the receive buffer
  [P, M, hl*d] directly (with the fp8 modules; per-tensor/per-token amax is order-independent, so it can be exact).
- If B fails: the nocopy root cause is then in the live path too, and D's log says where.
- `ulysses_nocopy.py` stays as the measured negative (its helpers `_vllm_strategy_info` / `_triton_launch` are reused).
