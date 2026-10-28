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
