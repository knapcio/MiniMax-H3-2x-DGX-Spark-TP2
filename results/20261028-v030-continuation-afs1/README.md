# audio_flow_shift 1 with per-window prompts, 30 s 960x576: rejected (the generated audio fades more, not less)

Idea from review round 6: a lower audio flow shift (1 instead of the recipe's 3.0) might keep the generated audio
level across windows. Same request as the per-window-prompt clip (`20261027-v030-continuation-winprompts/`, Ref2VA,
20 steps, seed 1101, `CONT_PROMPTS=auto`, no soundtrack, reference at 2048 px), only `audio_flow_shift: 1`.

| | per-window prompts, afs 3.0 | per-window prompts, afs 1 |
|---|---|---|
| Wall clock | 3656 s | 4559.5 s (windows 1516 / 1270 / 1581 s; window 1 is the first request after a boot) |
| Peak / min free head, worker | 69 GB | 71.2 GB / 43.2, 44.2 GB |
| Audio mean per 10 s | -27 / -31 / -35 dB (-8 dB) | -24.9 / -30.1 / -35.8 dB (-11 dB) |
| Cuts | none | none (max scene score 0.059) |
| Seams 277 / 532 | smooth | smooth (3.96 vs 3.74 before; 4.15 vs 3.03) |
| Motion per window | 2-6.6 / 2-5.2 / 1.3-8 | 3.6-6.4 / 3.2-4.1 / 2.9-5.3 |
| Detail per window | 11-44 / 5-39 / 12-79 | 40-114 / 50-71 / 42-83 |
| Light per second | not measured | 71 -> 100 (0-5 s), 100 -> 61 (10-23 s), 61 -> 110 (23-30 s): a slow swing |

The audio fade is larger with the lower shift, so the knob goes the wrong way; `audio_flow_shift` stays at 3.0.
Motion and detail hold better than in the afs-3.0 clip, but that clip is one sample too and the light swings by a
third across the windows here. Time: 4560 s against 3656 s with the same frame plan; window 3 (226 frames) took
1581 s against 743-1022 s in the other 960x576 runs, unexplained (the request is otherwise identical).
