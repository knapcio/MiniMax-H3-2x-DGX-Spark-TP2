# 60 s with H3's own first-window audio looped as the driving track: audio flat, but two cuts and runaway motion

The "no user track" fallback for the driving-soundtrack recipe: the first 11.5 s of the plain clip's generated audio
(`assets/h3loop60.wav`, looped to 60 s) as `audio_mode: lock_source`, with `H3_TP2_REF_SHORT_EDGE=1024` and
`CONT_PROMPTS=auto`, 277-frame windows, Ref2VA, 20 steps, seed 1101, 960x576. One sample.

| | plain 60 s | prompt + music track 60 s (`-60s-best`) | H3 loop as track (this) |
|---|---|---|---|
| Wall clock | 7986 s | 7313 s | 6982.9 s (windows 864 / 1308 / 1305 / 1307 / 1309 / 562 s) |
| Peak | 69.4 GB | 69 GB | 67.5 GB |
| Audio mean per 10 s | -27 .. -45 dB | flat (track) | -27.0 .. -27.8 dB (the loop) |
| Cuts (scene > 0.1) | 23.1 s, 28.0 s (dissolves) | none | **34.5 s (0.26), 46.0 s (0.30)** |
| Motion per second | 7 -> 0.8 (decays) | 6 -> 1.1 -> 4.5 | **4.2 -> 21 (runs away)**: 5 in window 1, 10 in window 3, 18 in window 5, 21 in window 6 |
| Detail per second | 65 -> 9 (blurs) | 83 -> 6 (blurs) | 27 -> 309 (over-sharpened, streaky) |
| Seams | | smooth | smooth (5.9 / 10.1 / 14.3 / 18.3 / 21.1 against the same levels before) |
| Light | | | 83-111, steady |

The loop keeps the audio level, as any driving track does, but the video does not settle the way it did with the
music track: the camera speed roughly doubles every two windows and the texture over-sharpens with it, and the model
cuts twice. So cut-freeness is not a property of `lock_source` as such (one music-track sample had none, this
ambient-loop sample has two), and this fallback is not a recipe for users without a track. Which audio makes H3
hold a shot is open; with one sample per variant nothing more can be said. Clip in `~/Downloads/h3-60s-h3audio/`.
