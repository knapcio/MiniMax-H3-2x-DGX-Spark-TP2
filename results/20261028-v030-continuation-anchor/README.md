# Opening-frames anchor (`H3_TP2_ANCHOR_FRAMES=22`), plain 30 s 960x576: rejected (three cuts, +25 % time)

The first window's opening 22 frames (7 latent frames, 3780 packed rows) were given to windows 2 and 3 as a Ref2VA
video reference placed before the latent guide, to carry the clip's opening look into every hand-over. Same request
as the deterministic plain clip (`20261027-v030-continuation-30s/`: Ref2VA, 20 steps, seed 1101, bench prompt, one
prompt, generated audio, reference image at 2048 px). Window 1 is bit-identical to the plain clip by construction (the
metrics agree to the second for 0-11 s).

| | plain | anchor 22 |
|---|---|---|
| Wall clock | 3615 / 3591 s | 4501 s (+25 %; windows 1302 / 1403 / 1616 s) |
| Peak | 71.1 GB | 71.9 GB |
| Cuts (scene > 0.1) | none | 12.5 s (0.25), 17.6 s (0.50), 20.5 s (0.21): three hard cuts inside windows 2 and 3 |
| Light per second | 103 -> 52 -> 91 (slow swing) | 58 at 18 s (a cut to a dark shot), 108 at the end |
| Motion per window | 3-7 / 3-4 / 1.3-4 | 3-7 / 4-7.6 / 2.4-5 (kept) |
| Detail per window | 46-65 / 32-63 / 19-37 | 46-65 / 33-80 / 62-221 (over-sharpened at the end) |
| Audio mean per 10 s | -27.2 / -30.5 / -34.0 dB | -27.2 / -30.5 / -29.6 dB |

With the opening frames in the condition set the model treats them as a reference to cut back to: the clip jumps to a
new shot three times. Motion and audio level hold better than in the plain clip and the texture does not blur, but
the cuts make it unusable, and the extra rows cost a quarter of the time. Rejected; the same-shaped tail-anchor
control (`H3_TP2_ANCHOR_SOURCE=tail`) stays queued only as a control. Clip in `~/Downloads/h3-30s-anchor/`.

## The same-shaped control: the previous window's tail as the extra reference (`H3_TP2_ANCHOR_SOURCE=tail`)

| | plain | anchor 22 (opening frames) | anchor 22 (previous tail) |
|---|---|---|---|
| Wall clock | 3615 / 3591 s | 4501 s | 3950 s (+10 %; windows 1173 / 1424 / 1177 s) |
| Cuts (scene > 0.1) | none | 12.5 s, 17.6 s, 20.5 s | **17.6 s (0.35)**: the same frame 423 as the opening anchor's second cut |
| Light | slow swing | dip to 58 at 18 s | dip to 67 at 18 s |
| Detail at the end | 19-37 | 62-221 | 82-203 (over-sharpened) |
| Audio at 20-30 s | -34.0 dB | -29.6 dB | -30.0 dB |

The control cuts too, on one of the same frames, although its extra reference carries nothing the latent guide does
not already carry. So the cuts come from the presence of an extra video-reference block in the sequence (its rows and
their RoPE positions), not from the opening frames' content, and the detail and audio changes come with it. Both anchor
variants are closed: an extra video reference in a continuation window makes the model cut.
