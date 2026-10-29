# One DiT denoising step under torch.profiler (`H3_TP2_PROFILE_STEP=3`), 5 s 960x576, both ranks

vLLM-Omni v0.30.0, Ref2VA, 20 steps, seed 1101, the third `MiniMaxH3DiTModel.forward` of each process profiled
(CUDA activities, key_averages by self CUDA time). Self CUDA time per kernel family, rank 0 (rank 1 in brackets):

| Kernel family | Self CUDA | Share of the step |
|---|---|---|
| FP8 GEMMs (`cutlass_scaled_mm`, sm120 CUTLASS) | 12.71 s (12.48 s) | 47.4 % (47.0 %) |
| Attention (cuDNN SDPA, kernel `sm80_flash_fprop_wmma_f16`) | 10.48 s (10.58 s) | 39.1 % (39.8 %) |
| Ulysses all-to-all (NCCL SendRecv) | 1.64 s (1.54 s) | 6.1 % (5.8 %) |
| Copies (`aten::copy_` / contiguous) | 0.59 s | 2.2 % |
| FP8 activation quant (`triton_red_fused_..._abs_clamp_max_mul_reciprocal`) | 0.41 s | 1.5 % |
| `silu_and_mul` | 0.33 s | 1.2 % |
| fused qk-norm + RoPE | 0.24 s | 0.9 % |
| modulation / gating kernels (`_indexed_gate*`, `_rms_norm_indexed*`) | 0.37 s | 1.4 % |

Total ~26.8 s of kernel time for the step (the profiler also books 6.1 s of "Command Buffer Full", the host waiting on
a full launch queue, which is not GPU work). What this says:

- 87 % of a step is two kernel families, FP8 GEMM and attention; everything else is 13 %. Fusing the small kernels
  (quant into modulation, qk-norm) can win at most a few percent.
- The GEMMs run at 54-61 TFLOPS (`20261028-fp8-gemm-bench/`); CUTLASS and cuBLASLt rowwise are equal there.
- The attention kernel cuDNN picks on the GB10 is an sm80-generation flash kernel (`wmma_f16`), not a Blackwell-native
  one; the cuDNN 9.27 A/B that follows checks whether a newer library picks a different kernel on sm121.
- The all-to-all is 6 %, consistent with the 2.9 % per-step figure measured earlier by CUDA events (that measurement
  excluded the kernel-side SendRecv time).

Files: `~/h3-tp2-logs/profile-head-3891913.txt` on spark-06 and `profile-worker-3633819.txt` on spark-05.

## A continuation window step (277 frames, 960x576, Ref2VA, reference at 2048 px), rank 0, step 5 of window 1

| Kernel family | Self CUDA | Share | vs the 124-frame step |
|---|---|---|---|
| Attention (cuDNN) | 42.64 s | 61.5 % | 4.1x |
| FP8 GEMMs | 21.11 s | 30.4 % | 1.7x |
| Ulysses all-to-all | 2.16 s | 3.1 % | 1.3x |
| Copies | 1.12 s | 1.6 % | |
| FP8 activation quant | 0.68 s | 1.0 % | |
| `silu_and_mul` | 0.56 s | 0.8 % | |

About 69 s of kernel time per step per rank (window 1 of the 15 s run took 1356 s for 20 steps). With 2.2x the frames
the attention time grows 4.1x and dominates the step, so for long windows the two-pass GEMM saves roughly 20 % of a
step rather than 25-30 %, and the attention kernel, already at the GB10's flash ceiling, is the wall. Per frame, a
window of W frames costs about a*W + b in the denoising loop: smaller windows are cheaper per frame (192 instead of 277
would be roughly a quarter cheaper) at the price of more hand-overs, which is where the quality problems are. The
window-2 profile (the +41 % of guide windows) is queued separately.

Files: `~/h3-tp2-logs/profile-head-4088226.txt` on spark-06.

## Window 2 of a 15 s continuation (107 frames), rank 0, step 5 (profiled 2026-10-29 02:11)

Meant to explain the +1..+51 % cost of windows after the first, but a 15 s clip's second window is only 107 frames,
so this profile is a small-window profile, not an equal-size comparison (an equal-size one, from a 30 s clip, is
queued). What it does show for a 107-frame window: GEMM 13.1 s (46 %), attention 11.2 s (39 %), all-to-all 2.17 s
(7.6 %), and 3.94 s (13.9 %) booked as "Command Buffer Full", the host waiting on a full launch queue, which is a much
larger share than in the big-window profile: small windows are closer to launch-bound. The variance of the later
windows across today's runs (chain13 +1 %, plain +10 %, chain6 +41 %, chain8 +51 % against window 1 of the same run)
does not match any fixed structural cost (the guide adds 2.5 % rows); it looks time-varying, host-side.
