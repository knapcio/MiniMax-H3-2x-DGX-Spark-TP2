# Attention kernel microbenchmark, spark-06 (GB10, sm_121), 2026-10-27 08:08

`scripts/attn-bench.py 129024 245952` in the recipe venv (torch 2.11.0, cuDNN 9.19, vLLM's bundled FlashAttention-2,
FlashInfer 0.6.14). One Ulysses rank's shapes: B=1, 28 local heads, head dim 128, bf16, non-causal, min of 3 runs.

| S (tokens) | cuDNN SDPA (recipe) | vLLM FlashAttention-2 | torch SDPA flash | FlashInfer |
|---|---|---|---|---|
| 129024 | 2698.7 ms | 2774.9 ms (+2.8 %) | 2889.7 ms (+7.1 %) | JIT needs `ninja`, not built |
| 245952 | 9779.3 ms | 10080.0 ms (+3.1 %) | 10128.8 ms (+3.6 %) | same |

Max abs difference between the FA2 / torch-flash outputs and cuDNN: 1.2e-4 (bf16 rounding; all three are exact
attention, different reduction order). FlashAttention-3 reports unsupported on this capability; the FA4 CuTe package is
not installed. cuDNN stays the backend.
