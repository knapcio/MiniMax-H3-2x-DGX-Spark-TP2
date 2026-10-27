# 30 s continuation with local-time windows (`H3_TP2_WINDOW_TEMPORAL_OFFSET=0`), 960x576, single-take prompt

Hypothesis (idea round): upstream shifts every media row of window k by start_frame*40/24 RoPE units while the text
and reference rows stay fixed, so their pull weakens window by window. Test: force the offset to 0 for every window;
same request as `20261027-v030-continuation-prompt` otherwise.

| | upstream offset (0 / 425 / 850) | offset 0 everywhere |
|---|---|---|
| Wall clock | 4519.7 s | 3663.6 s (the same prompt: the +25 % was sustained-load drift, not the prompt) |
| Scene cuts | none | none (max 0.06) |
| Seams 1 / 2 (mean before) | 4.87 (4.75) / 1.84 (1.45) | 4.51 (4.75) / 1.42 (1.61) |
| Apparent motion, window 1 -> 3 | 6.1 -> 2.6 -> 1.3 | 6.1 -> 2.4 -> 1.x: the same decay |
| Audio mean per 10 s | -21.2 / -27.9 / -35.3 dB | -21.2 / -28.3 / -37.3 dB: the same fade |

Rejected: the positional shift is not what drives the decay. Window 1 is identical by construction; windows 2-3
differ slightly but decay the same way. Switch stays off.
