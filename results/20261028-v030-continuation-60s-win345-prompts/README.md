# 60 s with window 345 + per-window prompts, generated audio, 960x576: audio and detail hold, two cuts

Ref2VA, 20 steps, seed 1101, `CONT_PROMPTS=auto`, `continuation_window_frames: 345` (upstream's maximum, 5 windows
instead of 6), overlap 22, reference at 2048 px, no soundtrack (H3 generates the audio). One sample. Compare with
the plain 60 s clip (277-frame windows, one prompt) and the 60 s prompt + soundtrack clip.

| | plain 60 s (277) | prompt + soundtrack 60 s (277) | window 345 + per-window prompts (this) |
|---|---|---|---|
| Wall clock | 7986 s | 7313 s | 8775.8 s (windows 1472 / 2077 / 2064 / 2068 / 732 s) |
| Peak / min free head, worker | 69.4 GB | 69 GB | 72.3 GB / 38.3, 42.0 GB |
| Audio mean per 10 s | -27 .. -45 dB (-18 dB) | flat (the track) | -25.6 / -26.6 / -27.2 / -26.4 / -23.5 / -22.4 dB: no fade |
| Motion per second | 7 -> 0.8 (decays) | 6 -> 1.1 -> 4.5 -> 3 | 1.3-1.7 for 25 s, then rising to 3.6-4.3: no decay, a slow camera |
| Detail (grass) per second | 65 -> 9-33 | 83 -> 6 (blurred) | 78-155 in the first 15 s, 157-242 in the last 15 s: no blur, but over-sharpened and streaky at the end |
| Light per second | | | 100-111: steady |
| Cuts (scene > 0.1) | 23.1 s, 28.0 s (dissolves) | none | 32.0 s (0.22) and 44.3 s (0.23): hard cuts inside windows 3 and 4 |
| Seams (345 / 668 / 991 / 1314) | | | all smooth (1.38, 2.09, 3.06, 3.78 against 1.41, 2.03, 2.92, 4.08 before) |

The owner's viewing (before the metrics): fog rolls in unnaturally fast at 7 s, transitions at 21 s, 32 s and 45 s,
weaker quality towards the end. Checked frame by frame: a fog bank appears over the field between 6 and 9 s (window
1, nothing in the prompt asks for it); at 21-22 s the sun dissolves away behind cloud (a crossfade inside window 2,
missed by the scene score, visible as 14 against 7-9 in the new `1s-change` line); 32.0 s and 44.3 s are hard cuts
(sun back, sun gone); from about 50 s the sky turns heavy, over-contrasty and streaky. The grass crops at 5 / 30 /
50 / 60 s keep individual blades, so the detail did not blur, but the rising detail and coherence numbers in the last
20 s (157-242, 0.62-0.69) are the over-sharpened, streaky look, not a good sign: the drift here goes towards too much
texture and contrast instead of towards blur. All four state changes are inside windows, not at a hand-over. So with
upstream's largest window and one prompt per window the generated audio does not fade and the texture does not blur
over 60 s, which neither the plain nor the prompt + soundtrack clip managed; the defects are the model's own state
changes (fog, sun in and out: one dissolve and two cuts) and an over-contrasty sky in the last 10 s. Not a usable
60 s clip either. Cost: windows with a guide (2-4) took 2064-2077 s against 1472 s for the first window of
the same size, +41 % (clocks and temperature were sampled every 30 s: 2106-2190 MHz, 61-77 C, no throttling); the
cause is being profiled (chain12).

Clip on spark-06: `~/h3-tp2-results/v030-cont60-win345-prompts/`; on the Mac: `~/Downloads/h3-60s-win345-prompts/`.
