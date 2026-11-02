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

(results follow)
