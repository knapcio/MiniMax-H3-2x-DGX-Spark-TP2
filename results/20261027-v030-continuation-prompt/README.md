# 30 s continuation with a prompt that asks for one continuous take (960x576, 20 steps, seed 1101, overlap 22)

Prompt = the bench prompt plus: "One single continuous take from a steady camera gliding slowly forward at constant
speed, no cuts, no transitions, no dissolves, the same shot from the first frame to the last."

| | bench prompt | single-take prompt |
|---|---|---|
| Wall clock | 3615.0 s | 4519.7 s (+25 %: the longer prompt adds text tokens to every window) |
| Denoising per window | 1151 / 1272 / 1020 s | 1413 / 1574 / 1291 s |
| Peak / min free head | 71.1 / 42.3 GB | 69.0 / 37.0 GB |
| Scene cuts / dissolves | one dissolve at 23 s | none (max scene score 0.06) |
| Seam 1 inter-frame diff (mean before) | 4.44 (3.96) | 4.87 (4.75): smooth |
| Seam 2 inter-frame diff (mean before) | 6.33 (3.62): visible hand-over | 1.84 (1.45): smooth |
| Motion per frame, start -> 22 s | 6.2 -> 3.6 | 6.1 -> 1.3 |
| Audio mean per 10 s | -27.2 / -30.5 / -34.0 dB | -21.2 / -27.9 / -35.3 dB |

The wording removes the generated shot change and the motion jump at the seams: the clip is one continuous flight
over the meadow toward the misty valley, consistent to the last frame (frames at 0 / 8 / 17 / 21 / 25 / 30 s
inspected). It does not stop the decay: motion still dies down window by window and the audio fades 14 dB over 30 s.
Those two are the targets of the next runs: per-window prompts that re-state the motion, and a driving soundtrack
(`audio_mode=lock_source`).
