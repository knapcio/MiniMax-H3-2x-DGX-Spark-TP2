# The 17-frame luma flicker and the temporal chunks of the VAE decoder (2026-11-02)

## What flickers

Long clips show a luma step every 17 frames at a fixed phase: -6 YAVG per step in the 5 s-history painting clip
(13 drops, frames 357..561, the owner saw "a little flicker at 15-22 s"), the last 5 s of the 60 s seed-step clip, and
weaker regional changes (lower-left, ~1.3 gray levels) in the 960x544 Koninck renders (`scripts/luma_flicker.py`,
2026-10-30/31). Post-processing copies (brightness repair) still flickered for the owner.

## Where the period comes from (read from the checkpoint and vLLM-Omni v0.30.0)

`video_vae/config.json`: `vae_clip_length` 17, `vae_token_drop` 3, `vae_ratio_t` 4 ->
`tokens_chunk_size` 5, `token_overlap` 2, `frame_pre_padding` 3, `frame_overlap` 5 (`klvae.py` `setup_forward`).
The decoder (a ViT3D, bidirectional over the clip, 4 frames per token by a pure reshape, no causal mask) is run on
chunks of **5 + 2 tokens**: chunk k = tokens `[5k, 5k+7)`. Its first 20 decoded frames minus the 3 pre-padding frames
are block k's 17 frames; the last 8 frames minus 3 are the 5-frame look-ahead that gets **linearly blended** with the
first 5 frames of block k+1, which chunk k+1 decodes from a fresh start (no past tokens). Where the fresh start
decodes brighter or darker than the contextual look-ahead, every block begins with a luma ramp/step: period 17.
Both serving paths do this: plain requests (`decode_latent` -> `decode_base` -> `decode_temporal`, patched by
vLLM-Omni into a streaming uint8 fork, `vae_temporal.py`) and `preencode_mp4` requests (`chunked_decode.py` ->
`temporal_chunks.decode_temporal_chunks`). A full-length decode is not a reference: the decoder was trained and is
served on 17-frame clips.

## The candidate fix (opt-in, lossless-direction): real past for every chunk

`h3_tp2/vae_context.py`, `H3_TP2_VAE_LEFT_CONTEXT=<tokens>`: wraps `model._adaptive_decode`; when the clip is a
temporal view `latent[:, :, a:b]` with `a > 0`, it decodes `latent[:, :, a-ctx:b]` and drops the first `4*ctx`
frames. Chunk starts, block boundaries, the blend, padding and the frame count stay the released ones; the env is
forwarded to rank 1 so both ranks decode the same shapes. Cost per chunk after the first: about x1.33 (ctx 2) and
x1.69 (ctx 4) of the decoder (linear layers x1.29/x1.57, attention x1.65/x2.46); decode is ~10 % of a render.

Independent review (Opus, 2026-11-02 08:25) of the first version: frame indexing correct (4 frames/token verified in
`vae_vit.py`, pre-padding belongs to the chunk's first token), `ctx=0` identical to upstream; **blocker** found and
fixed: the first version only patched the `preencode_mp4` loop, not the plain-request streaming path (hence the
`_adaptive_decode` wrapper). **Caveat kept:** the decoder's temporal RoPE is normalised by the clip length
(`func.py`: `(arange(0.5, T)/T)*2-1`), so a 9-token clip is a temporal geometry the released loop never uses;
`scripts/vae_decode_offline.py --geometry` decodes tokens 0..6 vs 0..8 vs 0..10 and compares the frames of tokens
0..4 to measure that effect alone.

## Measurements

`H3_TP2_SAVE_LATENTS` keeps the diffusion output per window; `scripts/vae_decode_offline.py` decodes the same latents
with ctx 0 / 2 / 4 on one Spark (FP16 autocast like serving) and reports per-frame luma, the period-17 screen, PSNR
against ctx 0 and the decode time. Baseline render: `koninck30-960x544-base` (queue42, 30 s Koninck 960x544 FL2VA
continuation, seed 1101, defaults, blend trace on).

### Baseline `koninck30-960x544-base` (08:07-08:34, 1656 s)

Hand-overs invisible (luma step -0.0 / -0.1, no cut). Period-17 screen on the served MP4: 4 drops, frames 663..714,
median -2.08 YAVG (window 3). Served per-frame luma around the first one: 126.6 126.7 | **124.7** 125.3 126.1 126.5
126.7 126.9 127.0 ... 127.3 | **125.2** 125.6 ...: every block starts 2 Y dark and recovers over ~5 frames.
Blend trace at those boundaries: the previous chunk's look-ahead and the fresh chunk agree within 0.1-0.3 Y and
**both** show the ramp (old 123.65 -> 126.31, new 123.88 -> 126.37 over the 5 overlap frames): the dark frames are a
property of the latent tokens at the block start, not of which chunk decodes them.

### Left context: closed (negative)

Offline decode of the window-3 latents (67 tokens -> 226 frames, FP16 autocast, same chunk loop):

| variant | decode | mean abs dY | max abs dY | period-17 drops (>= 2 YAVG) | PSNR vs ctx 0 |
|---|---|---|---|---|---|
| ctx 0 (upstream) | 90.9 s | 0.124 | 0.78 | none | - |
| ctx 2 | 104.4 s | 1.798 | 7.94 | 12 x -7.36 and 12 x -4.31 | 29.6 dB |
| ctx 4 | 90.5 s | 8.376 | 55.6 | 12 x -52.5 (and more) | 23.2 dB |
| geometry: tokens 0..8 vs 0..6, frames of tokens 0..4 | | | | PSNR 32.0 dB, mean dY 1.39 | |
| geometry: tokens 0..10 vs 0..6 | | | | PSNR 29.7 dB, mean dY 2.10 | |

The review's caveat is the result: the decoder's temporal RoPE is normalised by the clip length, a 9- or 11-token clip
is a geometry the decoder cannot handle, and the extra past makes every block start a step of -7 (ctx 2) or -52 Y
(ctx 4). `H3_TP2_VAE_LEFT_CONTEXT` stays in the tree as a documented negative; do not enable it.

### The new lead: the look-lock's latent affine

The same window-3 latents decoded offline **as saved** (before the hand-over wrapper) show no period-17 drops at all
(mean abs dY 0.12; the boundary dip at frame 153 is -0.7 and recovers in 3 frames), while the served clip of the same
window dips -2.0 at the same frames and sits 3 Y lower overall. What the served decode saw and the saved latents did
not: the look-lock (per-channel affine toward window 1's mean/std, here gain 0.957 on the std, a mean shift) and the
pinned prefix. The block-start tokens (the encoder's padded clip-start token, one real frame + 3 padding frames per
17-frame clip) have statistics unlike the other four tokens of a block; a uniform per-channel affine moves them off
their manifold differently, which the decoder renders as a darker first frame per block. The dip size tracks the lock
correction: strongest in the 5 s-history clip (-6 YAVG, a 10-luma pull-back), -2 at 960x544 (4 % std gain), below
threshold in the default 1344 clips (gain ~1.00). `scripts/lock_decode_offline.py` replays the lock on the saved
latents (raw / lock / mean-only / gain-only) and decodes each: results below.

(lock replay results follow)

