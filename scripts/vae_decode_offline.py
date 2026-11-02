#!/usr/bin/env python3
"""Decode saved H3 latents (H3_TP2_SAVE_LATENTS) with the released chunk loop and with left-context variants, on one GPU.

A continuation request saves one file per WINDOW (the diffuse hook fires per window); serving decodes the stitched
clip, so a window's 70 tokens stand in for it here (the chunk geometry is the same).  Run it with the head idle or
stopped (a few GB of GPU memory).

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
    ap.add_argument("--geometry", action="store_true", help="also decode tokens 0..6 and 0..8 raw and compare the frames of tokens 0..4")
    ap.add_argument("--shift", type=int, default=0, help="drop the first n latent tokens before decoding: moves the chunk grid by 4n frames (decoder-grid vs content test)")
    ap.add_argument("--stack-tiles", type=int, default=0, help="install h3_tp2.vae_tile_scope with this tile cap before loading the VAE (batched tile decode A/B; 0 = stock)")
    ap.add_argument("--deterministic", action="store_true", help="cudnn deterministic + no reduced-precision fp16 reductions")
    ap.add_argument("--save-frames", default="", help="save the ctx-0 decoded frames as uint8 [3,T,H,W] to this .pt (for A/B between runs)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
    from vllm_omni.diffusion.models.minimax_h3.temporal_chunks import decode_temporal_chunks
    from h3_tp2.vae_context import install_on_model
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if args.stack_tiles:
        from h3_tp2 import vae_tile_scope
        vae_tile_scope.install(MiniMaxH3VideoVAE, cap=args.stack_tiles)
        print(f"vae_tile_scope installed, cap {args.stack_tiles}", flush=True)

    obj = torch.load(args.latents, map_location="cpu")
    video = obj[0] if isinstance(obj, (tuple, list)) else obj
    if isinstance(video, dict):
        video = next(v for v in video.values() if isinstance(v, torch.Tensor) and v.ndim == 5)
    assert video.ndim == 5, video.shape
    if args.max_tokens:
        video = video[:, :, : args.max_tokens]
    if args.shift:
        video = video[:, :, args.shift :]
        print(f"shift: dropped the first {args.shift} tokens; drops bound to the chunk grid stay at multiples of 17 in this series, "
              f"drops bound to the content move by {4 * args.shift} frames", flush=True)
    print("video latent", tuple(video.shape), video.dtype, flush=True)

    dev = torch.device("cuda")
    t0 = time.time()
    vae = MiniMaxH3VideoVAE(args.vae, device=dev, decode_only=True, trust_remote_code=True)
    vae.model.eval()
    print(f"VAE loaded in {time.time() - t0:.1f} s; token_drop {vae.model.token_drop} chunk {vae.model.tokens_chunk_size} "
          f"overlap {vae.model.token_overlap} frame_overlap {vae.model.frame_overlap} pre_pad {vae.model.frame_pre_padding}", flush=True)
    latent = video.to(dev, dtype=torch.float32)
    denorm = vae._denormalize_latent(latent)

    def to_u8(frames):
        return (frames[0].clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu()   # [3, T, H, W] on the host

    # warm-up (cuBLAS/cuDNN autotune, allocator): a short raw decode, discarded
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        vae.model._adaptive_decode(denorm[:, :, :7])
    torch.cuda.synchronize()

    if args.geometry:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            d7 = vae.model.processor.revert_tensor(vae.model._adaptive_decode(denorm[:, :, :7]).float())[:, :, 3:20]
            d9 = vae.model.processor.revert_tensor(vae.model._adaptive_decode(denorm[:, :, :9]).float())[:, :, 3:20]
            d11 = vae.model.processor.revert_tensor(vae.model._adaptive_decode(denorm[:, :, :11]).float())[:, :, 3:20]
        for name, d in (("9_vs_7", d9), ("11_vs_7", d11)):
            mse = torch.mean((d - d7) ** 2).item()
            dy = sum(abs(a - b) for a, b in zip(luma_series(d), luma_series(d7))) / 17
            print(f"geometry {name}: frames of tokens 0..4 PSNR {10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item():.2f} dB, mean |dY| {dy:.3f}", flush=True)
        del d7, d9, d11
        torch.cuda.empty_cache()

    variants = [int(x) for x in args.variants.split(",")]
    if variants[0] != 0:
        variants.insert(0, 0)
    results = {}
    ref_u8 = None
    ref_y = None
    for v in variants:
        restore = install_on_model(vae.model, v)
        torch.cuda.synchronize(); t1 = time.time()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            dec = decode_temporal_chunks(vae.model, denorm, None)
        torch.cuda.synchronize(); dt = time.time() - t1
        st = dict(getattr(vae.model, "_h3_tp2_ctx_stats", {}))
        restore()
        frames = vae.model.processor.revert_tensor(dec.float())
        del dec
        frames = vae._normalize_decoded_frames(frames)[..., : args.height, : args.width].clamp(0, 1)
        y = luma_series(frames)
        u8 = to_u8(frames)
        if args.mp4:
            write_mp4(frames, os.path.join(args.out, f"decode-ctx{v}.mp4"))
        del frames
        torch.cuda.empty_cache()
        d = [abs(y[i] - y[i - 1]) for i in range(1, len(y))]
        drops = periodic_drops(y)
        r = {"frames": int(u8.shape[1]), "decode_s": round(dt, 1), "chunks_with_ctx": st.get("ctx", 0), "mean_abs_dY": round(sum(d) / max(1, len(d)), 3),
             "max_abs_dY": round(max(d) if d else 0.0, 2), "luma_mean": round(sum(y) / len(y), 2),
             "periodic_drops": [{"first_frame": run[0][0], "count": len(run), "median_step": round(sorted(s for _, s in run)[len(run) // 2], 2)} for run in drops]}
        if ref_u8 is None:
            ref_u8, ref_y = u8, y
            if args.save_frames:
                torch.save(u8, args.save_frames)
        else:
            mse = torch.mean((u8.float() - ref_u8.float()) ** 2).item() / 255.0 ** 2
            r["psnr_vs_0"] = round(10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item(), 2)
            r["mean_abs_dluma_vs_0"] = round(sum(abs(a - b) for a, b in zip(y, ref_y)) / len(y), 3)
        results[f"ctx{v}"] = r
        with open(os.path.join(args.out, f"luma-ctx{v}.json"), "w") as f:
            json.dump(y, f)
        print(f"ctx{v}: {json.dumps(r)}", flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
