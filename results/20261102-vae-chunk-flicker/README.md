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

### Lock replay on the saved window-3 latents (`scripts/lock_decode_offline.py`, 08:43-09:03)

Reference = window 1's latents (whole window, as in serving), overlap 7 tokens, serving defaults (strength 1.0, ramp
0.25), upstream chunk loop, FP16 autocast. Boundary step = luma(frame 17k) - luma(frame 17k-1), median over the window.

| latents decoded | mean luma | boundary step median / min | non-boundary mean abs dY | PSNR vs raw |
|---|---|---|---|---|
| raw (as saved, no lock) | 124.49 | -0.66 / -0.78 | 0.093 | - |
| lock, serving default (mean + std) | 122.01 | **-2.52 / -2.77** | 0.192 | 36.7 dB |
| lock, mean only (`CONT_LOCK_STD=0`) | 121.76 | -1.23 / -1.49 | 0.120 | 37.2 dB |
| lock, std gain only | 124.68 | -1.72 / -1.89 | 0.238 | 41.5 dB |
| **lock, token-class-aware** (`CONT_LOCK_CLASSES=1`) | 122.25 | **-0.83 / -1.01** | 0.097 | - |
| lock, class-aware, mean only | 122.01 | -0.76 / -1.03 | 0.094 | - |

The served baseline dips -2.0 at the same frames: the replay reproduces the flicker, and both halves of the affine
contribute (mean shift and std gain, roughly additive). The decoder's own dip is -0.66 (raw).

**Fix:** `h3_tp2/cont_match.py` `class_stats` / `lock_to_reference(dict)`: the block-start tokens (local index = 0
mod 5; a continuation window shifts by 75 tokens, so the local phase is the stitched phase, and the 7-token overlap
confirms: 7056 of 82656 pinned rows = 7 of 82 tokens) and the other tokens are matched to the reference's statistics
of the same class. The brightness correction is kept (122.25 vs 122.05) and the block boundary returns to the raw
decoder's -0.8 with raw smoothness. Serving control: `koninck30-960x544-lockclasses` (queue43, same request, only
`H3_TP2_CONT_LOCK_CLASSES=1`), results below.

### Serving control `koninck30-960x544-lockclasses` (queue43, 09:10-09:41, 1852 s incl. first-request warm-up)

Same request and seed as the baseline; only `H3_TP2_CONT_LOCK_CLASSES=1`. Boundary step = luma(17k) - luma(17k-1)
on the served MP4 (`boundary_steps.py`), per window:

| served clip | window 1 (no lock) | window 2 | window 3 | period-17 screen |
|---|---|---|---|---|
| baseline (mixture affine) | -0.30 / -0.42 | -0.12 / -0.53 | **-1.94 / -2.09** | 4 drops >= 2 YAVG (27.6-29.8 s) |
| class-aware lock | -0.30 / -0.43 | -0.31 / -0.79 | **-0.53 / -0.74** | none >= 1 YAVG |

(median / min). Light per second identical to the baseline (116 .. 127: the brightness correction is intact), seams
+0.0 / -0.1 luma, no cut. The lock lines show the new path (`classes A/B`), the std of window 3 pulled 0.6525 ->
0.6425 as before.

The 10-30 painting clips at 1344x768 (mixture affine), same statistic: default 30 s window 1 -0.31, windows 2-3
**-0.80 / -1.66**; the 5 s-history clip windows 2-3 **-6.07 / -6.21** (the owner's "little flicker at 15-22 s");
the 60 s windows 2-6 **-1.62 / -3.38**. Window 1 is always -0.3: the untouched decoder.

**Default since 2026-11-02 09:50:** `H3_TP2_CONT_LOCK_CLASSES=1` (`0` restores the mixture affine). The lock's
purpose (hold the first window's look against the per-window drift) is unchanged; only the fit is per token class.

### 1344x768 control `koninck30-1344-lockclasses` (queue44, 09:45-11:07, 4926 s): only half the gain

Seams invisible (-0.5 / -0.4 luma, no cut), light per second 115..125 | 122 118 117 119 .. 130 | 124 121 123.
Boundary steps: window 1 -0.27 / -0.36, window 2 median -0.69 min **-1.85**, window 3 -0.60 / -1.78 (the 10-30 clip
with the mixture affine: -0.80 / -1.66 and -0.65 / -1.49). The period-17 screen at >= 1 YAVG still finds 4 + 3 drops,
all in the first three or four blocks after each hand-over; later blocks sit at ~-0.6.

Per-frame luma after seam 1 (frames 270..345) is a **staircase**: flat 124.4 over frames 277-288, -1.3 at 289, flat
122.0, -1.4 at 306, flat 120.0, -1.8 at 323, flat 117.6, -1.9 at 340, 115.5 ... The 10-30 clip shows the same stairs
(123.0 -> 121.1 -> 119.4 -> 117.7 -> 116.1). This is the look-lock's RAMP (the correction grows linearly over the first
quarter of the free tokens, here 19 tokens = 64 frames) pulling the window down by ~9 Y from the hand-over level
(125) toward the reference mean (~121, the whole of window 1 including its dark start), rendered by the decoder as one
brightness per 17-frame block: a linear per-token shift in latent space comes out as steps at block starts, because
a block's brightness follows its first (padded, key-frame-like) token. The class-aware fit removes the extra dip that
the mixture affine added on top (hence the better medians) but cannot remove stairs that are the correction itself.

So at 1344 the remaining flicker is the size of the post-seam pull-down, not the fit. Two consequences, both being
measured: (1) reference = the tail of window 1 (`H3_TP2_CONT_LOCK_REF=tail`, queue45 at 960x544): the target is then
the level the clip is handed over at, the pull-down shrinks from ~9 Y to ~0-2 Y and so do the stairs; (2) the ramp
length only spreads the same total over more blocks (replayed offline on the 1344 window-2 latents: ramp 0.25 / 1.0 /
none, tail reference, classes).

### Owner's eye (11:25)

960x544 baseline vs class-aware lock, same seed: "chyba na obu nie miga" (no flicker seen on either; the -2.0 Y steps
of the baseline are below the eye at this size, the -6 Y of the 5 s-history clip were not). 1344 class-aware: "wyglada
ok raczej". What the owner did see: the cloud shadows race across the plain in both 960x544 clips (our README prompt
asks for "their shadows glide over the plain"); codex's human-pace comparison (prompt without strong shadows, figures
"walk naturally at a relaxed, unhurried pace") was "chyba git". So the perceptual threshold for the block steps lies
between 2 and 6 YAVG, the fix stays (it removes a real artifact and costs nothing), and the next work is the prompt:
queue46 renders the no-shadow human-pace wording on the clean stack for the owner to compare.
