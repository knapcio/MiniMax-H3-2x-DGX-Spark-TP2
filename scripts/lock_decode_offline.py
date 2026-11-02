#!/usr/bin/env python3
"""Does the look-lock's latent affine create the 17-frame flicker?  Decode one saved window with the lock replayed.

    python scripts/lock_decode_offline.py --ref latents-win1.pt --window latents-win3.pt --overlap-tokens 6 \
        --height 544 --width 960 --out dir --mp4

Variants (upstream chunk loop, FP16 autocast like serving, no left context):
  raw        the window's diffusion output as saved (before the hand-over wrapper)
  lock       h3_tp2.cont_match.lock_to_reference with the serving defaults (strength 1.0, ramp 0.25, mean+std)
  lock_mean  the same with std_strength 0 (mean-only lock)
  gain_only  the same with the mean shift removed (std gain only)
Reports per variant: mean luma, mean |dY|, the period-17 drop screen, PSNR vs raw.
"""

import argparse
import json
import os
import sys
import time

import torch


def luma_series(frames01):
    r, g, b = frames01[0, 0], frames01[0, 1], frames01[0, 2]
    y = 16.0 + 219.0 * (0.299 * r + 0.587 * g + 0.114 * b)
    return y.mean(dim=(1, 2)).cpu().tolist()


def periodic_drops(means, period=17, threshold=1.0, minimum=3):
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


def write_mp4(frames01, path, fps=24):
    import subprocess
    u8 = (frames01[0].clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).permute(1, 2, 3, 0).contiguous()
    t, h, w, _ = u8.shape
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                          "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for i in range(0, t, 48):
        p.stdin.write(u8[i : i + 48].cpu().numpy().tobytes())
    p.stdin.close(); p.wait()


def load_video(path):
    obj = torch.load(path, map_location="cpu")
    v = obj[0] if isinstance(obj, (tuple, list)) else obj
    assert v.ndim == 5, v.shape
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="window-1 latents (the look-lock reference)")
    ap.add_argument("--window", required=True, help="a later window's latents")
    ap.add_argument("--overlap-tokens", type=int, default=6)
    ap.add_argument("--vae", default=os.path.expanduser("~/h3-long/weights/FL2VA/video_vae"))
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mp4", action="store_true")
    ap.add_argument("--variants", default="raw,lock,lock_mean,gain_only")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
    from vllm_omni.diffusion.models.minimax_h3.temporal_chunks import decode_temporal_chunks
    from h3_tp2 import cont_match as cm

    ref = load_video(args.ref)
    win = load_video(args.window)
    print("ref", tuple(ref.shape), "window", tuple(win.shape), flush=True)
    dev = torch.device("cuda")
    ref_stats = cm._stats(ref.to(dev))
    m_ref, s_ref = ref_stats
    ov = args.overlap_tokens
    win_d = win.to(dev)
    free = win_d[:, :, ov:].float()
    m, sd = cm._stats(free)
    print(f"reference mean {m_ref.mean().item():.4f} std {s_ref.mean().item():.4f}; window free mean {m.mean().item():.4f} std {sd.mean().item():.4f}; "
          f"per-channel gain range {(s_ref / sd).min().item():.3f}..{(s_ref / sd).max().item():.3f}, |shift| max {(m_ref - m).abs().max().item():.4f}", flush=True)

    def variant(name):
        if name == "raw":
            return win_d
        if name == "lock":
            return cm.lock_to_reference(ref_stats, win_d, ov, 1.0, index=3, ramp=0.25, std_strength=None)
        if name == "lock_mean":
            return cm.lock_to_reference(ref_stats, win_d, ov, 1.0, index=3, ramp=0.25, std_strength=0.0)
        if name == "gain_only":
            # same gain as "lock", mean untouched: lock toward (m, s_ref) instead of (m_ref, s_ref)
            return cm.lock_to_reference((m, s_ref), win_d, ov, 1.0, index=3, ramp=0.25, std_strength=None)
        raise SystemExit(name)

    t0 = time.time()
    vae = MiniMaxH3VideoVAE(args.vae, device=dev, decode_only=True, trust_remote_code=True)
    vae.model.eval()
    print(f"VAE loaded in {time.time() - t0:.1f} s", flush=True)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        vae.model._adaptive_decode(vae._denormalize_latent(win_d[:, :, :7].float()))
    torch.cuda.synchronize()

    results, ref_u8, ref_y = {}, None, None
    for name in args.variants.split(","):
        lat = variant(name).to(torch.float32)
        torch.cuda.synchronize(); t1 = time.time()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            dec = decode_temporal_chunks(vae.model, vae._denormalize_latent(lat), None)
        torch.cuda.synchronize(); dt = time.time() - t1
        frames = vae._normalize_decoded_frames(vae.model.processor.revert_tensor(dec.float()))[..., : args.height, : args.width].clamp(0, 1)
        del dec
        y = luma_series(frames)
        u8 = (frames[0].clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu()
        if args.mp4:
            write_mp4(frames, os.path.join(args.out, f"decode-{name}.mp4"))
        del frames; torch.cuda.empty_cache()
        d = [abs(y[i] - y[i - 1]) for i in range(1, len(y))]
        r = {"decode_s": round(dt, 1), "luma_mean": round(sum(y) / len(y), 2), "mean_abs_dY": round(sum(d) / len(d), 3), "max_abs_dY": round(max(d), 2),
             "drops_ge1": [{"first_frame": run[0][0], "count": len(run), "median_step": round(sorted(s for _, s in run)[len(run) // 2], 2)} for run in periodic_drops(y, threshold=1.0)],
             "drops_ge2": [{"first_frame": run[0][0], "count": len(run), "median_step": round(sorted(s for _, s in run)[len(run) // 2], 2)} for run in periodic_drops(y, threshold=2.0)]}
        # the luma step at every 17-frame boundary (frame 17k vs 17k-1), median over the window
        steps = [y[k] - y[k - 1] for k in range(17, len(y), 17)]
        r["boundary_step_median"] = round(sorted(steps)[len(steps) // 2], 2) if steps else None
        r["boundary_step_min"] = round(min(steps), 2) if steps else None
        if ref_u8 is None:
            ref_u8, ref_y = u8, y
        else:
            mse = torch.mean((u8.float() - ref_u8.float()) ** 2).item() / 255.0 ** 2
            r["psnr_vs_raw"] = round(10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item(), 2)
        results[name] = r
        with open(os.path.join(args.out, f"luma-{name}.json"), "w") as f:
            json.dump(y, f)
        print(f"{name}: {json.dumps(r)}", flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
