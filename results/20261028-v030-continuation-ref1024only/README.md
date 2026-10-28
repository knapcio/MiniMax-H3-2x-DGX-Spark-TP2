# Reference at 1024 px alone, plain 30 s 960x576 continuation: -24 % time, same decay as the plain clip

Same request as the deterministic plain clip (`20261027-v030-continuation-30s/`: Ref2VA, 20 steps, seed 1101, one
prompt, generated audio, 277-frame windows) with only `H3_TP2_REF_SHORT_EDGE=1024` (reference image resized to a 1024
short edge instead of upstream's 2048). This isolates the reference size from the other two parts of the recipe.

| | plain (ref 2048) | ref 1024 only |
|---|---|---|
| Wall clock | 3615 / 3591 s | 2735.4 s (-24 %; windows 868 / 962 / ~740 s) |
| Peak | 71.1 GB | 68.0 GB |
| Cuts (scene > 0.1) | none | none |
| Seams 277 / 532 (mean before) | 4.98 (4.16) / 6.20 (3.46) | 4.42 (4.35) / 4.31 (3.83): smoother |
| Motion per window | 3-7 / 3-4 / 1.3-4 | 3-8 / 3-5.5 / 2.1-3.4 (less decay) |
| Detail per window | 46-65 / 32-63 / 19-37 | 18-50 / 19-49 / 23-33 |
| Audio mean per 10 s | -27.2 / -30.5 / -34.0 dB | -22.8 / -26.5 / -34.7 dB (same -12 dB fade) |
| Light | 103 -> 52 -> 91 | 105 -> 67 -> 107 |

The reference size is a speed lever (-24 % here, the same as the whole recipe's -24 % at this size, so the other two
parts cost nothing) and nothing else: the audio fades exactly as before and the detail ends at the same level (the
first window simply starts with a different, softer look from the smaller reference). The texture-holding seen in the
earlier recipe clip therefore came from the soundtrack and the per-window prompts, or from that sample, not from the
reference size. `H3_TP2_REF_SHORT_EDGE=1024` stays recommended for continuation on speed grounds, with the usual
caveat that a reference whose fine detail must carry into the clip should keep 2048.
