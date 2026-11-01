# Isolated H3 attention probe

Device: NVIDIA GB10; PyTorch: 2.13.0+cu130; cuDNN: 92000.

The long sequence length is inferred, not measured at the serving attention call. These synthetic timings do not establish whole-render speed or output quality. FMHA-v2 uses FP16 QK reduction.

| Sequence | Q/K scale | cuDNN median ms | Backend | Median ms | cuDNN/backend | Nonfinite pairs | Relative L2 vs cuDNN |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 512 | 1 | 0.095 | flashinfer_fa2 | 0.080 | 1.183 | 0 | 0.00133863 |
| 512 | 1 | 0.095 | flashinfer_fmha_v2_fp16_qk | 0.130 | 0.730 | 0 | 6.32889e-05 |
| 512 | 4 | 0.101 | flashinfer_fa2 | 0.083 | 1.208 | 0 | 0.000587524 |
| 512 | 4 | 0.101 | flashinfer_fmha_v2_fp16_qk | 0.127 | 0.791 | 0 | 5.04093e-05 |
| 512 | 16 | 0.099 | flashinfer_fa2 | 0.087 | 1.139 | 0 | 0.000158944 |
| 512 | 16 | 0.099 | flashinfer_fmha_v2_fp16_qk | 0.128 | 0.772 | 0 | 3.92611e-05 |
| 43392 | 1 | 311.057 | flashinfer_fa2 | 296.455 | 1.049 | 0 | 0.00067712 |
| 43392 | 1 | 311.057 | flashinfer_fmha_v2_fp16_qk | 499.106 | 0.623 | 0 | 0.000203915 |

A ratio above 1 means that backend was faster in this isolated sample. Consult the JSON for all timing samples and FP32 reference comparisons; nonfinite pairs include either input to the comparison. No serving default has been changed.
