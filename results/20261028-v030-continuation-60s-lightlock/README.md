# Light-lock wording in every window prompt, 60 s window 345, generated audio: the same two cuts at the same times

Same request as `20261028-v030-continuation-60s-win345-prompts/` (chain6) with one sentence appended to every generated
window prompt: "The light, the sun and the sky stay exactly as they are the whole time." (`CONT_PROMPTS_EXTRA`).

| | chain6 (per-window prompts) | + light lock |
|---|---|---|
| Wall clock | 8775.8 s (windows 1472 / 2077 / 2064 / 2068 / 732 s) | 7503.0 s (1594 / 1616 / 1604 / 1610 / 737 s) |
| Peak | 72.3 GB | 74.0 GB |
| Cuts (scene > 0.1) | 32.0 s (0.22), 44.3 s (0.23) | **32.0 s (0.21), 44.3 s (0.23)**: the same frames |
| Light per second | 100-111 | 101-111 (steady in both) |
| Audio mean per 10 s | -25.6 .. -22.4 dB (flat) | -22.0 .. -26.8 dB (flat) |
| Motion per second | 1.3 -> 4.3 | 1.4 -> 4.5 |
| Detail / coherence at the end | 162-242 / 0.62-0.69 (over-sharpened) | 168-255 / 0.60-0.69 (same) |
| Seams | smooth | smooth |

The wording changed nothing that matters: the two cuts land on exactly the same frames (769 and 1064), which means
they are not a reaction to the prompt but to the sampling itself (every window restarts from the same seed, so the
noise that produces the cut is the same in both runs). Light was steady in both clips anyway. The time difference
(-14 %) is the window-2..4 variance seen all day (chain6's middle windows were unusually slow), not the prompt.

What the day's 60 s clips say about cuts: the two clips with a driving soundtrack (`lock_source`: the music track,
`20261028-v030-continuation-60s-best/`) have no cuts; the two with generated audio (chain6 and this one) have two each,
at the same frames. The joint audio-video model appears to cut when its own generated audio calls for it; a steady
given track removes the trigger. chain8 (H3's own first-window audio looped as the driving track) tests whether that
also holds without a user-supplied track. Clip in `~/Downloads/h3-60s-lightlock/`.
