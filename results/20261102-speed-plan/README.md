# Lossless speed plan for the H3 pair (2026-11-02, from a read-only code and profile review)

Scope: MiniMax H3 on two DGX Sparks (GB10), vLLM-Omni v0.30.0 at `a8576ccb7`, our plugin, `results/` profiles and the
head journal of 2026-11-02. Nothing here is implemented yet; each item names the first hours of work and the proof.

## Where the time goes (1344x768, 277 frames = 82 tokens, M = 42848 rows per rank, S = 85 696, 28 heads x 128, 50 layers)

| part | arithmetic | per step | share |
|---|---|---|---|
| attention | 4·S²·128·28·50 = 5.26e15 FLOP at ~88 TFLOPS (cuDNN measured) | 60 s | 70 % |
| GEMMs | 1.65e15 FLOP at 164 TFLOPS (fp32-output GEMM) 10.1 s + scale pass 814 GB at ~235 GB/s 3.5 s | 13.6 s | 16 % |
| all-to-all (SP2) | 61 GB at 19.5 GB/s | 3.2 s | 4 % |
| copies, quant, misc | | ~3 s | |
| total | | ~80 s on a good boot | 86 s observed |

At 960x544 the same model gives 15.7 + 7.0 + 1.6 + 1.5 ≈ 26 s/step, matching the good boots (514-535 s per 20-step
window); the 566-619 s windows of 2026-11-02 08:00 / 09:04 were slow boots, not a different workload.

Corrections to our earlier picture: the "NVlabs fusions" are mostly already in this stack (QKV is one GEMM
`QKVParallelLinear` N=21504, `minimax_h3_transformer.py:430`; AdaLN modulate + gate-add are fused,
`rms_norm_indexed_scale_shift` / `indexed_gate_rms_norm_scale_shift` :940-983; RoPE is fused with qk-norm
`fused_qk_norm_rope` :670; SwiGLU is our level 1). The missing one is per-token FP8 quant fused into the producing
kernel (~1 %). The real 10-14 % per-GPU gap to the single-GPU port: our two-pass fp32 round trip (5-9 %) and the SP2
all-to-alls with their permute copies (5-8 %), which a single GPU does not pay. Encoders are 0.4-4 s per request today.

## Ranked by gain per engineering day

1. **Host-memory hygiene and a boot speed gate** (0.5 day; bad boots cost +10-20 %: 619 vs 514-562 s today, 120 vs 86
   s/step on 10-30). `H3_TP2_EMPTY_CACHE_BEFORE_DECODE=1` (`plugin.py:203`) hands the denoise pool back to the host
   on every request, so the next request reallocates from fragmented memory: a hypothesis to test. Work: sudoers rule
   + compaction as a pre-start step on both hosts (agent too); an automatic warm-up request at boot (the first request
   runs 6-7 % slow); a probe gate (attention at 43k rows within 5 % of 311 ms, else restart). Proof: 3 boots x 2
   requests, s/step per window, `buddyinfo`, the latents digest. Lossless by construction.
2. **Batched VAE tile decode on by default (`H3_TP2_VAE_STACK_TILES=2`)** (0.5-1 day): decode is 148 s per 30 s at
   960x544, -35 % = -52 s (-2.8 % of the render), ~-90 s on a 30 s 1344 render. Today's equivalence is 45 dB (below
   the 60 dB bar), so: offline A/B on saved latents with `allow_fp16_reduced_precision_reduction=False` and cuDNN
   deterministic; if > 60 dB or bit-exact, no eye check needed. The `=2` path has never run on a GPU: CUDA test first.
3. **All-to-all overlap with a high-priority comm stream** (1-2 days; 3-4 % at 1344, 5-6 % at 960). The earlier
   overlap was bit-identical but slower because NCCL queued behind cuDNN; its stream has default priority
   (`ulysses_overlap.py:90`). Try priority -1 + `NCCL_MAX_CTAS=2-4`. Separately `NCCL_NET_GDR_LEVEL=0` (`env.sh:25`)
   forces bounce-buffer copies with SM kernels: run `alltoall_perf` at 150-310 MB with GDR/C2C on. Fallback: an RDMA
   all-to-all without SM kernels (RoCEnante proxy). Proof: latents digest.
4. **Fused per-row/per-column FP8 GEMM epilogue** (2-4 days; -4.6 s/step = 5-6 % at 1344, ~9 % at 960): removes the
   fp32 output write, the scale pass (`fp8_gemm.py:62`) and the 4.9 GB fc1 temporary. Probe cuBLASLt outer-vector
   scaling on sm121, then a CUTLASS sm120 tile sweep with the vLLM epilogue order (scale_b, scale_a, bias as fma),
   SwiGLU in the epilogue (gate/up columns interleaved). Stock and two-pass already give identical latents, so a
   kernel without split-K is likely bit-exact: check bitwise on the 4 shapes, then the digest.
5. **Remove the Ulysses permute copies, one combined qkv all-to-all** (2 days; 1.3-2 %): four 615 MB copies per layer
   (`ulysses.py:143`, `:206`, the split view of v). Write the scale pass / `fused_qk_norm_rope` straight into a
   [dest, M, 3, 28, 128] send buffer; cuDNN reads strided q/k/v; the out-projection quant reads the returned tensor.
   Pure data movement: bit-identical. (`--ulysses-a2a-permute` will not help: NCCL load-store, single machine only.)
6. **Fuse per-token quant into the producing kernels** (2 days; 1-1.5 %): the two modulation outputs, the
   post-all-to-all output, the SwiGLU output (level 2; find why level 2 diverged in 1 of 2 runs first).
7. **Small per-step waste** (0.5 day; < 0.2 % at 1344, ~0.5 % at 960; all bit-identical): boolean-mask indexing forces
   host syncs several times per step (`denoise_loop.py:255`, `:413`, `transformer:1491`), `torch.unique` (`:208`), the
   cancellation sync (`:455`); the RoPE table is rebuilt every step on CUDA (only the NPU path caches it,
   `transformer:1611-1620`); the text token refiner reruns every step on ~602 constant rows (`:1515`); the AdaLN
   projections reread ~13 GB of FP8 weights per step (~57 ms). Fix: precomputed index tensors, cached refiner output
   and RoPE table per branch.

## Outside the lossless bar (owner's call)

- cuDNN 9.27: -3.5 % denoising measured (~9 % faster attention kernel, ~-6 % per step at 1344) but a different sample
  (SSIM 0.85): the same class of change as the two-pass GEMM at 960 Ref2VA that the owner accepted.
- FlashInfer FA2: -4.7 % on the kernel, ~7 GiB per rank of workspace.

## Not worth doing

CUDA graphs (the GPU is the bottleneck; ~1250 launches per step cost milliseconds; "Command Buffer Full" in the profile
means the launch queue is full), regional `torch.compile` (cannot fuse across Triton / cuDNN / NCCL kernels, ≤ 1 %,
changes fp association), lossless attention savings (cuDNN is within ~5 % of the GB10 flash ceiling). Rank 1's GEMMs
were 15 % slower in the 10-29 profile: recheck after item 1 before any uneven sequence split.

## Measured 2026-11-02 12:40-13:28: item 1, the empty-cache knob and boot-to-boot variance (queue47)

Five 5 s 1344x768 T2VA calibrations (seed 0, 20 steps; reference `cmp-t2va-1344` 23.2 s/step), all with the same
latents digest `4d6d69e4aad0`:

| boot | `H3_TP2_EMPTY_CACHE_BEFORE_DECODE` | request | s/step | diffuse | decode | peak | wall |
|---|---|---|---|---|---|---|---|
| 11:42 (end of queue45) | 1 | 1st after two 30 s renders | 22.92 | 458.4 s | 42.0 s | 63.0 GB | 502 s |
| 11:42 | 1 | 2nd | 22.78 | 455.7 s | 42.0 s | 63.1 GB | 499 s |
| 12:57 (fresh) | 0 | 1st | 21.44 | 428.8 s | 43.3 s | 63.2 GB | 477 s |
| 12:57 | 0 | 2nd | 21.24 | 424.9 s | 41.9 s | 63.2 GB | 468 s |
| 12:57 | 0 | 3rd | 21.21 | ~424 s | 42.0 s | 63.2 GB | 468 s |

The knob is inert: with `=1` the plugin logs "upstream releases the stage cache before decode itself; empty_cache
patch skipped" (vLLM-Omni v0.30 does the release itself), with `=0` the code path is not entered; the decode time and
the peak are the same either way. What the table does show is **boot-to-boot variance of 7 %** between two boots 75
minutes apart on the same hosts, same code, same request (the 10-30 anomaly was 40 %). The second and third requests
on a boot are not slower than the first (no per-request degradation). So item 1 stands as "boot hygiene + a boot
speed gate", not as an env change: every boot should be probed (a 5 s calibration or a 43k-row attention probe)
before a long render, and a slow boot restarted. The README's "free cached pages before decode" lever is
superseded by upstream and marked so.

## Boot gate in use (queue49, 15:05-15:42): the host degrades with restarts, compaction brings it back

`scripts/boot-probe.sh` (5 s 960x576 T2VA, seed 0, first request of the boot) after each restart of the same code:

| boot | host state before | probe | latents digest |
|---|---|---|---|
| 15:05 (8th restart of the day) | 06: 1.7 GB free, 77 GB page cache, orders 8+ `55 13 20 16 2 0 0`; 05 no order-10+ blocks | 9.17 s/step (SLOW at the 5 % limit of the first version) | edea0733 |
| 15:16 | same, one more restart | **9.80** | 5244e25b |
| 15:31, after `drop_caches` + `compact_memory` on both hosts with the head stopped | 06: 125 GB free, orders 9-15 `21106 15244 10431 5439 2083 687 302` | **9.14** PASS (limit 9.5) | d76aabd8 |

Warm reference 8.65 (second request of a good boot); the first request carries 4-7 % of warm-up, hence the 10 %
tolerance. Two findings: (1) the hosts degrade over a day of restarts (page cache full, no large free blocks) and each
boot into that state is slower than the last; a compaction with the services down restores the fast state, so it
belongs before every boot that precedes a long render (sudo from the workstation for now); (2) the first request of
a boot gives a different latents digest on every boot (three boots, three digests at 960x576), while requests within
one boot repeat the digest exactly (queue47: 2 + 3 requests). Lossless A/B therefore means same-boot digests; the
2026-10-29 "bit-stable across boots at 960" held for the boots it compared and does not generalise.

## Measured 2026-11-02 18:21-19:20: items 3 and 5 (queue50), 5 s 1344x768 T2VA seed 0, one boot per arm

| arm | s/step | latents vs stock |
|---|---|---|
| S1 stock (the 60 s render's boot, warm) | 21.06 | - |
| N `H3_TP2_ULYSSES_NOCOPY=1` | 21.22 | **rel-l2 0.199, every element differs** (video and audio) |
| S2 stock, fresh boot | 21.32 | **bit-equal to S1** |
| P `H3_TP2_A2A_OVERLAP=2` + `H3_TP2_A2A_STREAM_PRIORITY=-1` + `NCCL_MAX_CTAS=4` | 21.27 | **bit-equal to S2** |

- **Item 5 (no-copy exchange): closed.** The GPU result is wrong (a 20 % relative difference is a bug, not reordering:
  cuDNN's reading of the strided packed-QKV views or the per-destination rope launches), and even so there was no
  speedup; the copies it removes are overlapped or cheap at this cell. The code stays behind `=force` for a future
  debug session with per-layer q/k/v comparison.
- **Item 3 (comm stream priority): no effect** at 1344 5 s: bit-equal and 21.27 vs 21.32 s/step. The all-to-all is
  not the bottleneck here; the overlap's earlier "slower" result stands. Closed unless a profile shows the a2a waiting.
- **Cross-boot reproducibility at 1344:** two stock boots gave bit-equal latents (and queue47's five requests on two
  boots all `4d6d69e4`), so the per-boot digest variance seen at 960x576 (three boots, three digests) is specific to
  that cell; at 1344 same-request digests can be compared across boots.
- Boot-to-boot speed on compacted hosts: 21.06 / 21.32 / 21.27 (1.2 % spread) against 22.9 on the 11:42 boot and 120
  s/step on 10-30: the compaction + probe gate is the lever that matters.

Remaining: item 2 (batched VAE decode default-on, offline A/B next), item 4 (fused FP8 epilogue, 2-4 days), items 6-7.
