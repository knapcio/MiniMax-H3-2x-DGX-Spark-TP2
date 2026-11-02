#!/usr/bin/env python3
"""Decode saved H3 latents (H3_TP2_SAVE_LATENTS) with the released chunk loop and with left-context variants, on one GPU.

    ~/h3-long/.venv/bin/python scripts/vae_decode_offline.py --latents latents-spark-06-....pt --height 544 --width 960 \
        --variants 0,2,4 --out /tmp/vae-offline --mp4

Per variant: per-frame luma (Y, 16..235 scale, full frame), the periodic-drop screen (period 17, >= 2 YAVG, 3 periods),
mean |dY| between neighbouring frames, PSNR against variant 0, and the decode time.  ``--mp4`` writes a video-only MP4
per variant (libx264 crf 12) for the eye.  The decode runs under FP16 autocast like the serving pipeline.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch


def luma_series(frames01: torch.Tensor) -> list[float]:
    # frames01: [1, 3, T, H, W] in 0..1 (processor revert)
    r, g, b = frames01[0, 0], frames01[0, 1], frames01[0, 2]
    y = 16.0 + 219.0 * (0.299 * r + 0.587 * g + 0.114 * b)
    return y.mean(dim=(1, 2)).cpu().tolist()


def periodic_drops(means, period=17, threshold=2.0, minimum=3):
    groups = []
    for phase in range(period):
        matches = [(i, means[i] - means[i - 1]) for i in range(1, len(means))
                   if i % period == phase and means[i] - means[i - 1] <= -threshold]
        run = []
        for m in matches:
            if run and m[0] != run[-1][0] + period:
                if len(run) >= minimum:
                    groups.append(run)
                run = []
            run.append(m)
        if len(run) >= minimum:
            groups.append(run)
    return sorted(groups, key=lambda run: run[0][0])


def write_mp4(frames01: torch.Tensor, path: str, fps: int = 24) -> None:
    u8 = (frames01[0].clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).permute(1, 2, 3, 0).contiguous()  # T,H,W,3
    t, h, w, _ = u8.shape
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(0, t, 48):
        p.stdin.write(u8[i : i + 48].cpu().numpy().tobytes())
    p.stdin.close()
    p.wait()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--vae", default=os.path.expanduser("~/h3-long/weights/FL2VA/video_vae"))
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--variants", default="0,2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mp4", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=0, help="decode only the first n latent tokens (0 = all)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
    from vllm_omni.diffusion.models.minimax_h3.temporal_chunks import decode_temporal_chunks
    from h3_tp2.vae_context import decode_temporal_chunks_ctx

    obj = torch.load(args.latents, map_location="cpu")
    video = obj[0] if isinstance(obj, (tuple, list)) else obj
    if isinstance(video, dict):
        video = next(v for v in video.values() if isinstance(v, torch.Tensor) and v.ndim == 5)
    assert video.ndim == 5, video.shape
    if args.max_tokens:
        video = video[:, :, : args.max_tokens]
    print("video latent", tuple(video.shape), video.dtype, flush=True)

    dev = torch.device("cuda")
    t0 = time.time()
    vae = MiniMaxH3VideoVAE(args.vae, device=dev, decode_only=True, trust_remote_code=True)
    vae.model.eval()
    print(f"VAE loaded in {time.time() - t0:.1f} s; token_drop {vae.model.token_drop} chunk {vae.model.tokens_chunk_size} "
          f"overlap {vae.model.token_overlap} frame_overlap {vae.model.frame_overlap} pre_pad {vae.model.frame_pre_padding}", flush=True)
    latent = video.to(dev, dtype=torch.float32)
    denorm = vae._denormalize_latent(latent)

    results = {}
    ref = None
    for v in [int(x) for x in args.variants.split(",")]:
        torch.cuda.synchronize(); t1 = time.time()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            if v == 0:
                dec = decode_temporal_chunks(vae.model, denorm, None)
            else:
                dec = decode_temporal_chunks_ctx(vae.model, denorm, None, left_context=v)
        torch.cuda.synchronize(); dt = time.time() - t1
        frames = vae.model.processor.revert_tensor(dec.float())
        frames = vae._normalize_decoded_frames(frames)[..., : args.height, : args.width].clamp(0, 1)
        y = luma_series(frames)
        d = [abs(y[i] - y[i - 1]) for i in range(1, len(y))]
        drops = periodic_drops(y)
        r = {"frames": frames.shape[2], "decode_s": round(dt, 1), "mean_abs_dY": round(sum(d) / max(1, len(d)), 3),
             "max_abs_dY": round(max(d) if d else 0.0, 2), "luma_mean": round(sum(y) / len(y), 2),
             "periodic_drops": [{"first_frame": run[0][0], "count": len(run), "median_step": round(sorted(s for _, s in run)[len(run) // 2], 2)} for run in drops]}
        if ref is None:
            ref = frames
        else:
            mse = torch.mean((frames - ref) ** 2).item()
            r["psnr_vs_0"] = round(10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item(), 2)
            r["mean_abs_dluma_vs_0"] = round(sum(abs(a - b) for a, b in zip(y, luma_series(ref))) / len(y), 3)
        results[f"ctx{v}"] = r
        with open(os.path.join(args.out, f"luma-ctx{v}.json"), "w") as f:
            json.dump(y, f)
        if args.mp4:
            write_mp4(frames, os.path.join(args.out, f"decode-ctx{v}.mp4"))
        print(f"ctx{v}: {json.dumps(r)}", flush=True)
        del dec, frames
        torch.cuda.empty_cache()
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
