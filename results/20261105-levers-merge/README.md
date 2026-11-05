# 2026-11-05: the two out-projection levers on one branch (`levers`): `H3_TP2_ULYSSES_EXACT=2` + `H3_TP2_FP8_QUANT_FUSE`

Branch `levers` from main 50b5a9d: `quant-fuse` (db53042, `results/20261104-quant-fuse/`) merged first, then `copy5`
(4a1962c, `results/20261104-ulysses-copy5/`). Both are opt-in; defaults unchanged. **Status: merged, combination rule
implemented, CPU tests pass (13 new, both levers' suites unchanged); no GPU run (the `gpu-free` marker on Spark_06 was
absent); serving proof PASS (queue74, below); 16 s gate PASS (queue75); both defaults flipped 2026-11-05.**

## The conflict

Only `h3_tp2/ulysses_exact.py::exchange_back` conflicted: quant-fuse added the unpack hook (`set_unpack_hook`, its
`out` site replaces the reverse unpack copy with the fused unpack + quant kernel), copy5 added `defer=True` (mode 2:
no unpack at all, a stride-0 placeholder, the out-projection quant reads the receive buffer). Resolved with both:
`defer` first; the hook only on the non-deferring path (mode 1). A refused deferral in mode 2 makes copy 5, as copy5
built it. `h3_tp2/plugin.py` and `scripts/queue-lib.sh` merged cleanly (sections 9b4 and 6e, the STRIP list from
quant-fuse; copy5 did not touch queue-lib, `ULYSSES_EXACT` was already stripped).

## Combination rule

| `H3_TP2_ULYSSES_EXACT` | `H3_TP2_FP8_QUANT_FUSE` | out-projection input | fused sites |
|---|---|---|---|
| `1` (default) | `1` | fused unpack + quant (`fp8_quant_fuse.unpack_hook`) | qkv, out, fc1, fc2 (unchanged) |
| `2` | off | `ulysses_outq` quant from the receive buffer | none (unchanged) |
| `2` | `1` | `ulysses_outq` | qkv, fc1, fc2; out skipped automatically |
| `2` | `qkv,fc1,fc2` | `ulysses_outq` | qkv, fc1, fc2 (same as `1`) |
| `2` | a list with `out` | `ulysses_outq` | the list without out; out ignored |

Head journal lines (`fp8_quant_fuse.install`, one line, at install):

    h3_tp2 fp8 quant fuse: site out skipped automatically: H3_TP2_ULYSSES_EXACT=2 makes no reverse unpack copy to fuse into: the out-projection quant reads the a2a receive buffer there (h3_tp2/ulysses_outq.py); the other sites are unaffected
    h3_tp2 fp8 quant fuse: site out requested but ignored: H3_TP2_ULYSSES_EXACT=2 ...   (when out was named)
    h3_tp2: EXPERIMENT FP8 quant fuse requested=1, installed sites: qkv, fc1, fc2

If ulysses_outq disables itself (self-check mismatch, an ineligible layer), that layer gets copy 5 and the stock
quant; the fuse's out site does not take over.

Placeholders never cross: the fuse's `QuantFP8.forward` wrapper raises on an ulysses_outq placeholder (stride 0 and
the pending deferral's pointer or a one-element storage, `ulysses_outq.is_placeholder`); `_OutQuant` raises on the
fuse's pending stash placeholder (`fp8_quant_fuse.pending_placeholder`). Every guard, fingerprint, self-check and
fallback of both branches is unchanged.

## Checks (Spark_06 CPU, `CUDA_VISIBLE_DEVICES=`, scratch copy `/tmp/levers`, `~/h3-long/.venv`)

    PYTHONPATH=/tmp/levers:/tmp/levers/tests python -m pytest tests -q

- `tests/test_levers_combo.py` (new, 13): the install table above with its log lines (exactly one out line, no
  unpack hook in mode 2); the whole DiT block (fake layers, eager mirrors) in mode 2 + fuse bit-identical to the stock
  path on two calls (qkv/fc1/fc2 fused, out deferred, outq self-check on the first call); outq disabled -> copy 5,
  still bit-identical; both cross-placeholder guards raise and nothing is computed or consumed; the deferring path
  never calls the unpack hook.
- Both levers' suites: `test_fp8_quant_fuse` 26, `test_ulysses_outq` 14, `test_ulysses_exact` 15,
  `test_ulysses_nocopy` 13, `test_fp8_consumer_fuse` 48, `test_fp8_cutlass_routing` 12: all pass.
- Whole `tests/`: 477 passed, 57 failed, 8 skipped; the 57 failures are the same tests as on main 50b5a9d
  (continuation, render-duration, text-file and dialogue-geometry tests, e.g. `plugin.py` NameError `_lock_classes`),
  none in the lever suites. **Do not run the whole `tests/` on a serving host:** some of those render tests' stub
  `curl` is not the one `scripts/render.sh` runs, so they POST to the real API at 127.0.0.1:18093 (on Spark_06 they
  reached the live head, mostly input errors, one started a 60 s continuation). Run the lever suites by name. One fix: `test_fp8_quant_fuse.py` failed 10 tests after `test_fp8_consumer_fuse.py` in
  the same session (also on the quant-fuse branch alone: a consumer-fuse install left `fp8_consumers._state["sites"]`
  set, so fc1 was skipped); its fixture now clears and restores that state.
- `install()` on the real vLLM-Omni v0.30.0 modules (CPU, source checks on, level cutlass): mode 2 + `1` -> qkv, fc1,
  fc2 with the line above; `qkv,fc1,fc2` -> the same, no out line; `out,qkv` -> qkv, `requested but ignored`; mode 1 +
  `1` -> all four and the hook; `ulysses_outq.check_sources` passes after the fuse install.

## Serving proof: PASS (queue74, 2026-11-05 03:51-05:34)

Result (full table in `results/20261104-quant-fuse/README.md`, "Serving proof"): A defaults `4d6d69e4aad0` (21.24
s/it); B `H3_TP2_FP8_QUANT_FUSE=1` alone `4d6d69e4aad0` three times, latents bit-equal to A (21.75 / 21.02 / 21.87
s/it), `installed sites: qkv, out, fc1, fc2`, four self-checks bit-identical on both ranks; C both levers
`4d6d69e4aad0` twice, latents bit-equal to A (20.47 / 19.82 s/it), `site out skipped automatically ...`, `installed
sites: qkv, fc1, fc2`, `Ulysses exact exchange installed (mode 2)`, three fuse self-checks and copy5's self-check
bit-identical, nothing `NOT bit-identical` or `materialised`; E defaults restored, probe PASS, no lever line left in
the env. Speed is inside the boot-to-boot spread (see the quant-fuse README); the digests are the proof.

## Default gate: PASS (queue75, 2026-11-05) and the flip

Boot with `H3_TP2_ULYSSES_EXACT=2` + `H3_TP2_FP8_QUANT_FUSE=1` (probe `bf6e85687a6a`), the 16 s 1344x768 FL2VA
continuation (seed 1101, the queue63/69 gate) twice: digests `200e38721c9e 4138eba5a964`, framemd5 `e46129c05cc6` both
times, the same frames as every default boot since 2026-11-03. gen.log:

    Q75-START 05:35:07
    Q75-BOOT rc 0 PROBE PASS: first 8.05 (bf6e85687a6a), warm 7.58 s/step (reference 8.65, limit 9.342) latents bf6e85687a6a 05:48:06 05:48:06
    Q75-CONT-1: 20/20 [27:53<00:00, 80.68s/it] 20/20 [27:53<00:00, 83.70s/it] 20/20 [08:36<00:00, 25.74s/it] 20/20 [08:36<00:00, 25.80s/it]  digests 200e38721c9e 4138eba5a964  framemd5: e46129c05cc6 phase diffuse 1674.1 s phase diffuse 516.1 s phase decode 86.5 s 
    Q75 PASS 1
    Q75-CONT-2: 20/20 [28:05<00:00, 83.83s/it] 20/20 [28:05<00:00, 84.26s/it] 20/20 [08:56<00:00, 26.71s/it] 20/20 [08:56<00:00, 26.80s/it]  digests 200e38721c9e 4138eba5a964  framemd5: e46129c05cc6 phase diffuse 1685.4 s phase diffuse 536.1 s phase decode 86.3 s 
    Q75 PASS 2
    Q75-BOOT-E (defaults restored) rc 0 health 200 07:19:44
    Q75-DONE 07:19:44

Both defaults flipped in `h3_tp2/plugin.py` (`H3_TP2_ULYSSES_EXACT` `1` -> `2`, `H3_TP2_FP8_QUANT_FUSE` `0` -> `1`);
`=1` / `=0` restore the previous defaults. The fuse's install line is now `INFO h3_tp2: FP8 quant fuse requested=1,
installed sites: qkv, fc1, fc2` (the out site skipped automatically under mode 2).

The plan as run: as in both levers' READMEs (`restart_head_gated`, queue-lib on the `levers` tree): A defaults; B
`export H3_TP2_ULYSSES_EXACT=2` + `export H3_TP2_FP8_QUANT_FUSE=1` on both ranks: boot probe `bf6e85687a6a`,
calibration digests `4d6d69e4aad0`, latents max abs diff 0 against A. Journal: the out line above, `installed sites:
qkv, fc1, fc2`, three fuse `self-check ... bit-identical` lines (qkv, fc1, fc2), `copy5: self-check ...
bit-identical`, no `NOT bit-identical`, no `materialised`. Expected per step at 1344 (M = 18880): ~0.12 s (copy 5) +
~0.43 s (qkv, fc1, fc2 of the fuse); the fuse's out site alone was ~0.12 s, so the pair saves about what `QUANT_FUSE=1`
alone saves, plus the receive-buffer quant's small edge.

## First serving boot on the flipped defaults: PASS (queue77, 2026-11-05 08:45-09:06, main eda87aa)

A gated FL2VA boot with a clean env (no lever line; the stale `H3_TP2_FP8_QUANT_FUSE=1` line the queue75 restore had left
was removed by hand at 07:24): probe `bf6e85687a6a` (8.17 first / 7.54 warm s/step), the 5 s 1344x768 calibration
`4d6d69e4aad0` at 19.60 s/it (diffuse 392.1 s, decode 27.2 s; the fastest calibration on record, the two-lever boot of
queue74 gave 20.47 / 19.82, the defaults boot 21.24). Both ranks log the new defaults from the code alone:
`Ulysses exact exchange installed (mode 2)`, `FP8 quant fuse requested=1, installed sites: qkv, fc1, fc2`, the out site
skipped automatically, the three fuse self-checks and the copy5 self-check `bit-identical`, `deferred=2001 quant_recv=2000
selfcheck_pass=1`; nothing `NOT bit-identical`, nothing `materialised`. The head keeps serving this boot.
