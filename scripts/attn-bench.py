"""Attention kernel microbench with H3 per-rank Ulysses shapes (B=1, H=28 local heads, D=128, bf16, non-causal).
cuDNN SDPA (the recipe's backend) vs vLLM's bundled FlashAttention-2 vs FlashInfer vs torch default SDPA."""
import sys, time, math, torch
from torch.nn.attention import sdpa_kernel, SDPBackend

S_list = [int(x) for x in sys.argv[1:]] or [129024]
H, D = 28, 128
dev = "cuda"
scale = 1.0 / math.sqrt(D)

def timeit(fn, iters=3):
    fn(); torch.cuda.synchronize()
    t = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); out = fn(); e.record(); torch.cuda.synchronize(); t.append(s.elapsed_time(e))
    return min(t), out

for S in S_list:
    torch.manual_seed(0)
    q = torch.randn(1, S, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, S, H, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, S, H, D, device=dev, dtype=torch.bfloat16)
    print(f"S={S} H={H} D={D} bf16; q/k/v/out {4*q.numel()*2/1e9:.1f} GB", flush=True)
    res = {}
    def cudnn():
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            return torch.nn.functional.scaled_dot_product_attention(q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3), scale=scale).permute(0,2,1,3)
    try:
        ms, ref = timeit(cudnn); res["cudnn"] = ms; print(f"  cudnn      {ms:9.1f} ms", flush=True)
    except Exception as e:
        print("  cudnn failed", type(e).__name__, str(e)[:120]); ref = None
    def flash_sdpa():
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            return torch.nn.functional.scaled_dot_product_attention(q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3), scale=scale).permute(0,2,1,3)
    try:
        ms, o = timeit(flash_sdpa); print(f"  torch-flash{ms:9.1f} ms  maxdiff vs cudnn {(o.float()-ref.float()).abs().max().item() if ref is not None else float('nan'):.3e}", flush=True)
    except Exception as e:
        print("  torch-flash failed", type(e).__name__, str(e)[:120])
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        cu = torch.tensor([0, S], device=dev, dtype=torch.int32)
        qf, kf, vf = q[0], k[0], v[0]
        def fa2():
            o = flash_attn_varlen_func(qf, kf, vf, cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=S, max_seqlen_k=S, causal=False, softmax_scale=scale, fa_version=2)
            return o[0] if isinstance(o, tuple) else o
        ms, o = timeit(fa2); print(f"  vllm-fa2   {ms:9.1f} ms  maxdiff vs cudnn {(o.float()-ref[0].float()).abs().max().item() if ref is not None else float('nan'):.3e}", flush=True)
    except Exception as e:
        print("  vllm-fa2 failed", type(e).__name__, str(e)[:160])
    try:
        import flashinfer
        qf, kf, vf = q[0], k[0], v[0]
        def fi():
            return flashinfer.single_prefill_with_kv_cache(qf, kf, vf, causal=False, sm_scale=scale)
        ms, o = timeit(fi); print(f"  flashinfer {ms:9.1f} ms  maxdiff vs cudnn {(o.float()-ref[0].float()).abs().max().item() if ref is not None else float('nan'):.3e}", flush=True)
    except Exception as e:
        print("  flashinfer failed", type(e).__name__, str(e)[:160])
    del q, k, v, ref; torch.cuda.empty_cache()
