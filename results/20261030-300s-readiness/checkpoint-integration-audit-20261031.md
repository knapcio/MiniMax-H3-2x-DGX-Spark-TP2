# Checkpoint integration audit, 2026-10-31

This is an audit of files on disk, not a claim about the code already loaded into a running process. At 23:13 CET
the following files had the same SHA-256 on Spark_05 and Spark_06:

| File | SHA-256 |
| --- | --- |
| `~/h3-tp2/h3_tp2/plugin.py` | `78cff682ea92aae3544410ab2cc1b5f1114d22ca236c16bc61ad717d073459d1` |
| `~/h3-tp2/h3_tp2/cont_match.py` | `af7b41f260ef22a2640f6e6e42f775be06c903312818d5b36c934cc40b2d5852` |
| `~/h3-long/vllm-omni/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3.py` | `fda01a7ced807952bf80c3f9370d9dc0b4d77f960485b00de883abd974c952f0` |
| `~/h3-long/vllm-omni/vllm_omni/diffusion/models/minimax_h3/continuation.py` | `66c0b4495e53c53840c6789a154141b43bf738b5e86aab9165f8e9e80835bc0d` |

The checked vLLM-Omni checkout on 06 reported commit `a8576ccb725c4e21cd13c3eb5f9a546b21149d2b`, with neither of
the two named pipeline files reported modified. The local `plugin.py` matches the two node files; the local
`cont_match.py` differs because it adds experimental snapshot/restore methods and revises a comment. Those methods
are not on the checked node files and are not called by the current serving loop. No files were installed or reloaded.

## Integration boundaries

* The patched loop in `plugin.py` constructs `wrap_diffuse`, plans all windows, derives the per-window seed, media
  offset and conditioning, samples a window, then updates cumulative video/audio before starting the next window.
  A candidate checkpoint belongs **after** both cumulative AV tensors and the wrapper state have been updated, and
  only after all ranks report a durable save. A resume must restore both before it skips exactly the completed windows.
* The patched loop currently receives denoising kwargs, window size, overlap and optional text conditioning. The
  pipeline's `forward` owns output configuration, including height, width, encoding settings and output count, outside
  that call. A complete identity cannot be inferred from the loop arguments alone: that configuration must be passed
  explicitly and consistently from request preparation to the checkpoint session.
* The pipeline calls `check_request_cancellation()` before and after the whole continuation call. Cancellation during
  a checkpoint, rank failure, missing peer and restart must be handled without a rank waiting indefinitely or a
  partially saved window being mistaken for a common checkpoint. Existing Gloo/CPU tests cover some failure cases;
  they do not establish the behavior of a real GPU process group or cancellation.
* The pipeline iterates output seeds and decodes each output after its continuation. The experimental session now
  accepts only `num_outputs=1`, because earlier outputs are not part of its checkpoint. Extending this requires
  preserving and authenticating those outputs and their order, rather than simply allowing a larger count.
* `_initial_noise` creates CPU generators from the supplied seed for video and audio on each diffusion call. The
  continuation's window seed step and retry policy are also material; storing a seed alone is not a complete runtime
  identity. Numerical nondeterminism remains possible even when the same inputs and seeds are restored.
* A real resume also needs durable request/job metadata and immutable access to the original inputs across a server
  restart, plus a deployment identity covering the weights, code, relevant configuration and runtime on **both**
  ranks. The current async job registry and prototype session do not provide that workflow. If recomputed prompt or
  media encodings differ, the strict tensor fingerprint should reject the resume rather than silently mix results.

Before any deployment, exercise the actual continuation loop with a fault after an intermediate window, then use an
idle pair for a controlled small real-model run and process restart. Compare uninterrupted and resumed AV, test
partial/corrupt files, different inputs, cancellation and peer loss, and measure memory/disk costs. A synthetic
CPU-only match is not a real-model recovery test. The active 300 s render has no checkpoints and cannot resume.
