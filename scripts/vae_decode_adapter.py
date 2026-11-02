#!/usr/bin/env python3
"""Decode saved H3 latents through the serving adapter path (``MiniMaxH3VideoVAE.decode_latent``: denormalise, the
streaming uint8 decode_temporal fork, crop), optionally with the batched tile decode (``h3_tp2.vae_tile_scope``) and
deterministic flags, on one GPU. Saves the uint8 frames for run-to-run comparison (scripts/frames_cmp.py) and times it.

    python scripts/vae_decode_adapter.py --latents L.pt --height 544 --width 960 --save-frames out.pt [--stack-tiles 4] [--deterministic]
"""
import argparse, os, sys, time
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--vae", default=os.path.expanduser("~/h3-long/weights/FL2VA/video_vae"))
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--save-frames", required=True)
    ap.add_argument("--stack-tiles", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--repeat", type=int, default=1, help="decode this many times (timing; the first is a warm-up when > 1)")
    args = ap.parse_args()
    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE
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
    dev = torch.device("cuda")
    vae = MiniMaxH3VideoVAE(args.vae, device=dev, decode_only=True, trust_remote_code=True)
    vae.model.eval()
    latent = video.to(dev)
    times = []
    for r in range(max(1, args.repeat)):
        torch.cuda.synchronize(); t0 = time.time()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            frames = vae.decode_latent(latent)
        torch.cuda.synchronize(); times.append(time.time() - t0)
        frames = frames[..., : args.height, : args.width]
        u8 = frames[0].to(torch.uint8).cpu() if frames.dtype == torch.uint8 else (frames[0].float().clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu()
    torch.save(u8, args.save_frames)
    y = (16.0 + 219.0 * (0.299 * u8[0].float() + 0.587 * u8[1].float() + 0.114 * u8[2].float()) / 255.0).mean(dim=(1, 2))
    print(f"frames {tuple(u8.shape)} dtype-in {frames.dtype} decode_s {[round(t, 1) for t in times]} luma_mean {y.mean().item():.2f} "
          f"peak_alloc_GB {torch.cuda.max_memory_allocated() / 1e9:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
