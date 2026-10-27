# 60 s by continuation (Ref2VA, vLLM-Omni v0.30.0, two Sparks), 960x576, 20 steps, seed 1101, 6 windows

Same request as the 30 s clip with `duration: 60`. Windows [0,277) [255,532) [510,787) [765,1042) [1020,1297)
[1275,1450); 1450 frames, 60.4 s.

| Metric | Value |
|---|---|
| Wall clock | 7985.6 s (133 min; 2.2 min per second of video) |
| Denoising per window | 1251 / 1385 / 1388 / 1393 / 1392 / 821 (shorter last window) s; VAE decode ~300 s |
| Peak (X-Peak-Memory-MB) | 69.4 GB |
| Min free, head / worker | 40.4 / 45.0 GB (cumulative latents of 6 windows cost nothing visible) |
| framemd5 | d6126a2c |

Determinism: windows 1 and 2 are the same frames as in the 30 s clip (identical per-second motion signature up to
22 s), so the continuation is deterministic for identical windows.

Drift over the windows, the central finding:

| Window | Motion (mean luma change per frame) | Audio mean level |
|---|---|---|
| 1 (0-11 s) | 6.2 falling to 3.9 | -27.2 dB (0-10 s) |
| 2 (11-22 s) | 3.4 to 3.9 | -30.5 dB |
| 3 (22-33 s) | 4.5 then 2.5 to 3.4 (two generated transitions: scene score 0.39 at 23.2 s, 0.16 at 28.0 s) | -32.8 dB |
| 4 (33-43 s) | 3.0 to 2.1 | -37.1 dB |
| 5 (43-54 s) | 2.1 to 0.85 (almost static by 53 s) | -42.9 dB |
| 6 (54-60 s) | 1.0, then a jump to 3.4 at 56 s (another transition) | -44.8 dB |

Seams at 277 / 532 / 787 / 1042 / 1297: inter-frame difference 4.44 / 5.57 / 3.45 / 2.34 / 0.94 against 3.96 / 3.62 /
3.03 / 1.90 / 0.87 just before them, i.e. a small motion bump at each hand-over, no cut. Picture stays sharp and
consistent in colour to the end (frames at 0 / 11 / 22 / 33 / 43 / 54 / 60 s inspected).

Conclusion: plain continuation with a fixed prompt loses motion and audio energy window by window (audio down 18 dB
over 60 s, motion down 7x) and inserts shot changes it was not asked for. The remedies upstream provides are
per-window prompts (`continuation_prompts`) and a driving soundtrack (`audio_mode=lock_source`); both are next.
