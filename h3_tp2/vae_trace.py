"""Optional diagnostics for the VAE's temporal overlap; no output adjustments."""

from contextlib import contextmanager
import logging

import torch

logger = logging.getLogger("vllm_omni.h3_tp2.vae_trace")


def _sampled_luma(raw, transform):
    """Approximate per-frame decoded Y on a spatial sample, before MP4 encoding."""
    sample = raw.detach()[..., ::8, ::8].float()
    mean = torch.as_tensor(transform.mean, device=sample.device, dtype=sample.dtype).view(1, 3, 1, 1, 1)
    std = torch.as_tensor(transform.std, device=sample.device, dtype=sample.dtype).view(1, 3, 1, 1, 1)
    channels = ((sample - mean) / std).clamp(0.0, 1.0).mean(dim=(0, 3, 4))
    return (16.0 + 219.0 * (0.299 * channels[0] + 0.587 * channels[1] + 0.114 * channels[2])).cpu().tolist()


@contextmanager
def trace_temporal_blends(model):
    """Log old/new/output overlap luma for each temporal blend, then restore the model's original method.

    Intended for a single controlled request. Spatial tile blends are ignored; the estimates are spatial samples,
    not the final encoded frames. Diagnostic errors do not interrupt decoding.
    """
    try:
        original = model.blend
        was_instance_attribute = "blend" in vars(model)
        previous_attribute = vars(model).get("blend")
    except Exception as exc:  # noqa: BLE001
        logger.warning("h3_tp2 VAE temporal blend tracing unavailable (%s)", exc)
        yield
        return
    count = 0

    def traced(a, b, blend_extent, dim):
        nonlocal count
        out = original(a, b, blend_extent, dim)
        if dim == -3 and getattr(a, "ndim", None) == 5 and getattr(b, "ndim", None) == 5:
            count += 1
            try:
                extent = min(a.shape[dim], b.shape[dim], blend_extent)
                if extent <= 0:
                    return out
                transform = model.processor.transform_rev
                before = _sampled_luma(a[:, :, -extent:], transform)
                incoming = _sampled_luma(b[:, :, :extent], transform)
                combined = _sampled_luma(out[:, :, :extent], transform)
                start = count * int(model.clip_length)
                logger.info("h3_tp2 VAE temporal blend %d approx-frame %d: old %s new %s blended %s",
                            count, start, [round(v, 2) for v in before],
                            [round(v, 2) for v in incoming], [round(v, 2) for v in combined])
            except Exception as exc:  # noqa: BLE001
                logger.warning("h3_tp2 VAE temporal blend trace skipped (%s)", exc)
        return out

    try:
        model.blend = traced
    except Exception as exc:  # noqa: BLE001
        logger.warning("h3_tp2 VAE temporal blend tracing unavailable (%s)", exc)
        yield
        return
    try:
        yield
    finally:
        try:
            if was_instance_attribute:
                model.blend = previous_attribute
            else:
                delattr(model, "blend")
        except Exception as exc:  # noqa: BLE001
            logger.warning("h3_tp2 VAE temporal blend trace restore failed (%s)", exc)
