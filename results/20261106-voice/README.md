# Voice naturalness: where the "robotic" voices come from (2026-11-06)

The owner hears the generated voices as robotic, "like AI", in the Polish doll clips and in the park clips. This directory
holds the measurements that located the loss. All renders on the two-Spark TP2 stack, 960x544, 20 steps unless noted.

**Answer (2026-11-07).** A single window's voice is as natural as the model gets (section 1). The loss is along a chain of
windows: every window hard-pinned the previous window's last 0.9 s of audio for the whole sampling and inherited its
texture, 3.87 -> 2.93 UTMOS over six windows (section 9). Since 21:35 the pin is released after 30 % of the denoise
steps by default (section 10): 3.60 average over windows 2-6 against 3.26, the same speakers, and the quietest
hand-overs of every variant in dialogue and in continuous ambience. Slips of the tongue in Polish are the model's own
and the same with or without a pin. The opt-in speech guard re-samples the window that slipped: a third of the wrong
characters are left on three 64.6 s Polish chains for 17-48 % more render time, and on a 118 s chain 13 of 50 for
75 % more; the voice score moved by -0.13 to +0.08 UTMOS over the four chains (section 15).

## Two predictors

- **DNSMOS** (`speechmos`, OVRL / SIG / BAK / P.808): signal quality against noise and distortion. It barely moves when a
  voice turns flat or synthetic.
- **UTMOS22** (`tarepan/SpeechMOS`, trained on human naturalness ratings of synthetic speech): the one that moves with the
  complaint. Natural recordings score about 4.0-4.4, good TTS 3.8-4.2, poor TTS below 3.

`scripts/voice_naturalness.py` prints both per continuation window, plus the pitch spread (pyin F0, semitones).

## 1. The model's voice in a single window is fine

A 11 s T2VA bookshop monologue (one 26-word English line, seed 2701; queue99) and the same line in Polish. Every lever
that could plausibly change the voice was tried on the same seed:

| arm | change | DNSMOS OVRL / SIG | UTMOS | WER | render |
|---|---|---|---|---|---|
| A-en | baseline, 20 steps, audio_flow_shift 3.0 | 3.44 / 3.71 | 4.34 | 0.00 | 547 s |
| B-en | 40 steps | 3.56 / 3.78 | 4.36 | 0.04 | 988 s |
| C15-en | audio_flow_shift 1.5 | 3.46 / 3.69 | 4.35 | 0.04 | 511 s |
| C50-en | audio_flow_shift 5.0 | 3.53 / 3.75 | 4.35 | 0.04 | 508 s |
| D-en | a voice description in the prompt ("warm, natural and unhurried, with the small breaths, pauses and changing intonation of a real conversation, recorded close with a clean studio microphone") | 3.54 / 3.76 | 4.36 | 0.04 | 546 s |
| A-pl | baseline, Polish | 3.39 / 3.65 | 4.14 | 0.05 | 515 s |
| D-pl | voice description, Polish | 3.38 / 3.61 | 3.82 | 0.09 | 512 s |

Nothing here is outside the noise (the WER differences are "ten" vs "10" and one Polish ending): the five English arms
land at UTMOS 4.34-4.36, the level of natural recordings. The pitch spread is 2.0-4.6 semitones, a normal conversational
range. Steps, the audio shift and a voice description are not the lever; the single-window voice is not the problem.

## 2. The loss is in the continuation windows

The same predictor per 277-frame window on four stitched clips (window k starts at k x 10.6 s):

| clip | windows | UTMOS window 1 | UTMOS windows 2+ (mean / min) | DNSMOS SIG window 1 / rest |
|---|---|---|---|---|
| tree-seam S0 (22 s, park, two speakers) | 2 | 3.40 | 2.77 / 2.77 | 3.22 / 3.23 |
| tree-seam S2 (22 s, park, cards + guard) | 2 | 3.33 | 3.13 / 3.13 | 3.43 / 3.26 |
| doll 60 s PL5 (Polish dialogue) | 6 | 3.65 | 3.02 / 2.73 | 3.56 / 3.53 |
| doll 118 s pl120v2 (Polish dialogue) | 11 | 3.64 | 2.89 / 2.61 | 3.67 / 3.40 |

Every continuation window scores 0.4-1.0 below window 1 while the signal-quality score stays nearly flat: the later
windows lose naturalness, not fidelity. In the 118 s clip windows 4, 6, 9 and 10 sit at 2.6-2.7. Listening cuts of
window 1 and windows 9-10 of that clip were given to the owner.

Two things happen to the audio at every hand-over in the default recipe: the previous window's tail rows are pinned as
target rows (`H3_TP2_CONT_EDIT=2`, audio rows included) and the audio look-lock pulls every later window's audio latents
toward the first window's tail statistics (`H3_TP2_CONT_LOCK_AUDIO=1`, a per-dimension affine with the gain clamped to
0.5-2.0). The lock's log lines show how much it moved the audio: x1.37 on the latent std in the tree windows 2, only
x1.00-1.04 in the doll 118 s clip, so the lock alone cannot explain the doll clip.

## 3. The A/B (queue101)

A 22 s two-window version of the bookshop monologue (two lines per window, seed 2701, FL2VA) rendered with the default
hand-over and with the default without the audio lock (`H3_TP2_CONT_LOCK_AUDIO=0`), scored per window; plus two more
seeds of the single window for the seed spread of the predictor. Upstream's hand-over (`H3_TP2_CONT_EDIT=0 CONT_RETRY=0
CONT_LOCK=off`) did not render on the FL2VA head ("continuation requires Ref2VA request execution"): it is in queue103
on Ref2VA.

| clip | UTMOS window 1 | UTMOS window 2 | voiced fraction w1 / w2 | DNSMOS SIG w1 / w2 | WER | render |
|---|---|---|---|---|---|---|
| default hand-over (audio lock on) | 4.34 | 4.05 | 0.48 / 0.24 | 3.72 / 3.61 | 0.00 | 1158 s |
| audio lock off | 4.34 | 4.10 | 0.48 / 0.24 | 3.72 / 3.60 | 0.00 | 1034 s |
| single window, seed 2702 | 4.40 | - | 0.48 | 3.71 | 0.08 | 511 s |
| single window, seed 2703 | 4.45 | - | 0.51 | 3.82 | 0.04 | 512 s |

- **The audio lock is not the cause.** It moved the audio latent std of window 2 by 1 % (0.404 -> 0.400, reference
  0.390) and the UTMOS by 0.05. Window 1 is identical in both clips (same seed, deterministic).
- **The hand-over costs 0.25-0.3 here, not 0.4-1.0**: window 2 of the monologue scores 4.05-4.10, well above the 2.6-3.1
  of the park and doll clips in section 2. The seed spread of the single window is 0.11 (4.34-4.45).
- Window 2 is half as voiced as window 1 (0.24 against 0.48: the second pair of lines is shorter and the model leaves
  pauses). UTMOS rates speech, and a window with more silence scores lower on its own, so part of the 0.25-0.3 may be
  the text, not the hand-over. Queue103 renders the window-2 text alone as a single window (two seeds) to separate the
  two, and the 22 s monologue on Ref2VA with our hand-over against upstream's.
- The 118 s doll clip (Polish, 11 windows, 2.6-2.7 in four windows) is the clip the owner heard. Its drop is larger
  than anything in this English probe; the Polish single window already starts lower (4.14), and the drop may grow with
  the window count or with the pinned tail of a Polish line. Measured next on a 60 s Polish clip once the controls are in.

## 4. Controls (queue103)

The window-2 text alone as a single window (`voice/prompt_cont2_en.txt`, 11 s, FL2VA, two seeds), and the 22 s
monologue on Ref2VA (reference = the first frame of the default clip, seed 2701) with our hand-over against upstream's
(`H3_TP2_CONT_EDIT=0 CONT_RETRY=0 CONT_LOCK=off`).

| clip | UTMOS w1 | UTMOS w2 | voiced w1 / w2 | median F0 w1 / w2 (Hz) | DNSMOS SIG w1 / w2 | WER | render |
|---|---|---|---|---|---|---|---|
| window-2 text alone, seed 2702 (FL2VA) | 4.25 | - | 0.56 | 153 | 3.64 | 0.00 | 522 s |
| window-2 text alone, seed 2703 (FL2VA) | 4.26 | - | 0.38 | 160 | 3.69 | 0.00 | 536 s |
| Ref2VA, default hand-over | 4.40 | 4.31 | 0.54 / 0.71 | 156 / 134 | 3.41 / 3.56 | 0.02 | 1174 s |
| Ref2VA, upstream hand-over | 4.40 | 4.28 | 0.57 / 0.81 | 155 / 103 | 3.41 / 3.53 | 0.00 | 1263 s |

(For comparison from section 3: the FL2VA default hand-over is 4.34 / 4.05, median F0 155 / 169 Hz.)

(The "Ref2VA, default hand-over" boot carried `H3_TP2_CONT_LOCK_ROBUST=1`, left in the head's environment by an earlier
queue: its video look-lock was the robust variant. The flag does not touch the audio path, so the voice numbers are
those of the default hand-over.)

- **About half of the FL2VA drop is the text.** The second pair of lines scores 4.25-4.26 when it is a window of its
  own, against 4.40-4.45 for the first pair: 0.17 lower with no hand-over at all. The hand-over itself costs the other
  0.15-0.2 on FL2VA (4.25 -> 4.05-4.10).
- **On Ref2VA the hand-over costs 0.09 with our loop and 0.12 with upstream's**, inside two seed spreads of the
  predictor (0.01-0.05 per text). Our edits to the hand-over (re-sample, cut check, look-lock, audio lock) do not make
  the voice less natural than upstream's on this clip.
- **The pitch moves at the hand-over, more with upstream's loop**: the median F0 of the same speaker goes 156 -> 134 Hz
  (-2.6 semitones) with ours and 155 -> 103 Hz (-7 semitones) with upstream's; single windows of this speaker sit at
  153-168 Hz. UTMOS does not rate this (both windows sound natural on their own); a listener hears a different voice.
  The per-window median F0 is in the tool's output for that reason.
- Not explained by this probe: the 118 s Polish doll clip (2.6-2.7 in four of eleven windows). The English two-window
  monologue never goes below 4.05. The next voice test is a 60 s Polish clip, per window, with the speaker's pitch
  tracked along the windows.

## 5. The long Polish clips, per spoken segment

A window's score is the score of whoever speaks in it and of how much silence it holds, so the two long doll clips were
re-scored per transcribed segment (`scripts/voice_naturalness.py --segments`, segments from `scripts/speech_qa.py`),
averaged per window:

| clip | segments | UTMOS per window (mean of its segments) |
|---|---|---|
| doll 118 s pl120v2 (three lines per window, seed 3401) | 39 | 3.79, 3.70, 3.66, **3.01**, 3.26, 3.20, 3.40, 3.39, 3.04, 3.31, 3.31 |
| doll 120 s (one line per window, seed 3301) | 21 | 4.18, 3.36, 3.04, 3.70, 3.74, 3.04, 3.38, 3.13, 3.83, 3.27, 3.42, 3.01 |

- In the 118 s clip the first three windows hold 3.7-3.8 and every window from the fourth on sits at 3.0-3.4; DNSMOS
  SIG follows (3.4 -> 2.9-3.2) and the transcript turns wrong in windows 8-10 ("Napirzę go szczerze", "Piękne słaba są
  dla witryn"), so it is audible distortion, not only a flatter delivery.
- **Both speakers lose the same amount**: the man (median F0 about 100 Hz) 3.66 -> 3.28, the woman (about 200 Hz)
  3.78 -> 3.22 between windows 1-3 and 4-11. The speakers' pitch stays put along the clip.
- The audio latent statistics do not drift: the raw free-step std of windows 2-11 is 0.463-0.499 against the reference
  0.456, and the audio lock moves it by 1-4 %.
- The 120 s clip has no trend after window 1: its windows scatter between 3.0 and 3.8.

What this does not tell is whether the chain of hand-overs or the lines themselves cost the points (a fresh Polish
window already scored 3.82-4.14 in section 1). Queue106 renders the lines of windows 4, 9 and 10 of the 118 s clip each
as a fresh single window with the seed the window had in the chain, and the first six windows with the default
hand-over and with upstream's (the fresh windows became queue110, section 7; the six-window chains queue111).

## 6. The audio lock on two more clips (queue107, queue109: replays, no diffusion)

The audio look-lock is a per-(channel, dimension) affine toward the statistics of window 1's last quarter. Two replays
of saved window latents, each decoded with the lock and without it:

| clip | UTMOS with the audio lock | without | what the lock did to the latents |
|---|---|---|---|
| park seam scene, window 2 (per window) | 3.14 | **3.47** | std 0.349 -> 0.255 (reference 0.210): per-dimension gains 0.18-0.95, mean 0.61 |
| doll 118 s, windows 4-11 (mean of the segments) | 3.14 | 3.23 | std moves 1-4 %; per-dimension gains 0.62-1.48 |
| doll 118 s, all 11 windows (per window) | 3.64 3.22 3.21 2.64 2.89 2.67 3.04 2.95 2.59 2.67 2.98 | 3.64 3.28 3.27 2.78 2.95 2.77 3.03 2.87 2.58 2.11 3.02 | |

- **In the park scene the lock costs a third of a point.** Window 1 ends on the quiet moment before the woman comes
  back, so the reference is a tail without speech and the lock turns the speech of window 2 down, dimension by
  dimension. Measured against the whole first window the same latents would need gains of 0.67-1.35 (mean 0.91).
- **In the doll clip the lock is not the cause**: with it off the windows from the fourth on score the same within
  0.1 (one window, the tenth, is worse without it). The loss from window 4 on is in the sampled latents, which leaves
  the chain of hand-overs or the lines themselves (queue110: the lines of windows 4, 9 and 10 as fresh single windows).
- The replay with the lock reproduces the clip's own audio (windows 1-8 score exactly as the original render).

The audio lock was added for one reason: the ambience of a 60 s painting clip climbed 23 dB over its last two windows
(2026-10-30). A lock that only holds the level, measured on the quiet steps, would keep that and leave speech alone;
it is designed once the no-lock arm of the painting clip (queue104) has reproduced the climb on today's code.

## 7. The lines are fine, the chain is not (queue110: fresh single windows)

The lines of windows 4, 9 and 10 of the 118 s clip, each rendered as a single 11.5 s window on the same boot settings
(Ref2VA, the doll reference at 1024 px, the seed that window had in the chain), scored per segment like the clip:

| lines of window | in the 118 s chain | as a fresh single window | the fresh window's segments |
|---|---|---|---|
| 4 | 3.01 | **3.77** | 3.88, 3.82, 3.73, 3.66 |
| 9 | 3.04 | **3.66** | 3.70, 3.77, 3.50 |
| 10 | 3.31 | **3.79** | 3.82, 3.67, 3.88 |

- The same Polish lines, the same speakers and seeds score 3.66-3.79 alone, the level of windows 1-3 of the chain
  (3.66-3.79). The 0.4-0.7 the chain loses from window 4 on is the chain's doing, not the lines'.
- The transcript of the fresh windows is right where the chain's was garbled ("Napiszę go szczerze" against "Napirzę go
  szczerze" in window 10).

Two mechanisms fit, and queue111 renders the first six windows of the scene three times to separate them:

1. **Position.** Upstream (and our loop) shifts every media row of window k by its start frame in RoPE units
   (start_frame * 40 / 24): window 4 begins at 1275 units, window 6 at 2125, while a 15 s training clip spans 600. The
   reference image and the text stay at the origin. Arm `toff64` keeps every window on local time (`H3_TP2_ID_LOCK=toff0`,
   the switch of `h3_tp2/identity_lock.py`; the older `H3_TP2_WINDOW_TEMPORAL_OFFSET` wraps upstream's loop, which the
   edited loop replaces, so it is a no-op on the current stack).
2. **Inheritance.** Every window pins the previous window's last 37 audio steps (74 of its 924 rows) and samples the
   rest to match them; a slightly worse tail breeds a worse window. Arm `noapin64` samples the audio of every window free
   (`H3_TP2_CONT_EDIT_AUDIO=0`, new, diagnostic only: the soundtrack is cut at every hand-over; the video prefix stays
   pinned), so a window's voice can only be as good as a fresh one.

Arm `def64` is the control (default hand-over; its windows 1-3 must match the 118 s clip).

## 8. The audio lock: default off, and why a level-only lock was not built (queue104 latents)

The 60 s painting clip rendered with no lock at all on today's code (queue104 K-off) has an ambience between -39.6 and
-49.1 dB per 10 s; with the audio lock 1-4 dB less in the loud windows. The 23 dB climb that made the lock (2026-10-30) did
not come back. The lock's cost is a third of a point of speech naturalness when window 1 ends quiet (section 6).

The replacement in mind was a scalar level lock measured on the quiet steps, with a deadband, so that speech is never
re-coloured. On the saved raw audio latents of three chains (`/tmp/level_lock_proto.py`: per-step RMS over (channel, dim)
after removing each dimension's mean, floor = the 10-30 % quantile of the free steps, reference = the whole first window):

| chain | floor of window 1 | floors of the later windows against it |
|---|---|---|
| painting, 6 windows, no lock | 0.273 | -1.0, -0.9, -0.2, +0.2, +0.6 dB |
| park seam scene, 2 windows | 0.214 | +1.0 dB |
| doll 118 s, 11 windows | 0.347 | -0.2 .. +2.8 dB |

Every window of every chain sits within 3 dB of its reference, while the decoded painting ambience moves 10 dB between
windows (-49.1 in window 1, -39.6 at 40 s). The latent floor, like the latent std, is nearly constant and does not carry
the loudness the ear hears: a latent-level lock cannot hold the level, and the old lock held 1-4 dB of it only through
its per-dimension gains, the part that hurts speech. So: `H3_TP2_CONT_LOCK_AUDIO=0` is the default since 2026-11-07,
`1` re-enables the old lock, and a loudness problem in a delivered clip is a job for the mix (`mix_v2.sh` already
normalises the bed).

## 9. Inheritance it is: the pinned audio tail degrades the chain (queue111, three 64.6 s chains)

The first six windows of the Polish doll scene (Ref2VA, reference at 1024 px, seed 3401, the segment prompts of
`voice/extra_pl6.json`), rendered three times on the new defaults (robust video lock, audio lock off); UTMOS per
window is the mean over the spoken segments that start in it (`scripts/voice_naturalness.py --segments`):

| window | 1 | 2 | 3 | 4 | 5 | 6 | render |
|---|---|---|---|---|---|---|---|
| `def64` control (default hand-over) | 3.87 | 3.53 | 3.48 | 3.06 | 3.31 | 2.93 | 3654 s |
| `toff64` local RoPE time per window | 3.88 | 3.55 | 3.49 | 3.13 | 3.22 | 3.07 | 3654 s |
| `noapin64` audio prefix not pinned | 3.89 | **3.86** | **3.72** | **3.75** | 3.18 | **3.68** | 3654 s |

- **Position is not it.** Local time per window changes the render (different framemd5) and nothing about the voice:
  every window within 0.1 of the control.
- **Inheritance is.** Without the pinned audio tail every window scores like a fresh one (section 7: 3.66-3.79); the
  control already loses 0.3 in window 2 and 0.9 by window 6. Window 5 of `noapin64` is two one-word lines ("Proszę."
  2.0, "Dobrze." 2.6) that UTMOS rates low in any clip; its sentences score 3.4-4.1.
- **The speakers stay themselves without the pin.** WavLM x-vector cosine of each window's segments to the window-1
  segments of the same speaker (`scripts/speaker_drift.py`, speakers split by pitch):

  | | w2 | w3 | w4 | w5 | w6 |
  |---|---|---|---|---|---|
  | `def64` low voice / high voice | 0.97 / 0.96 | 0.94 / 0.98 | - / 0.82 | 0.93 / 0.94 | 0.86 / 0.85 |
  | `noapin64` low voice / high voice | 0.96 / 0.97 | 0.97 / 0.95 | 0.94 / 0.96 | 0.92 / 0.75 | 0.95 / 0.95 |

  The voices are carried by the text and the scene, not by the pinned tail; the pinned chain drifts further by window 6.
- **The cut is not salient in dialogue.** Log-mel distance across the 250 ms on each side of the five hand-overs
  (`scripts/audio_seams.py`) against the same distance on the clip's own 250 ms grid: `noapin64` seams 15.6, 24.6, 21.1,
  20.7, 7.5 with a grid median of 15.7 and a 95th percentile of 36.3; level jumps -4.8 to +4.4 dB. The control's seams
  12.0-27.8 against a grid median of 19.1. In speech with pauses the free window starts where the old one stopped.

So the hard pin of the previous tail (37 audio latent steps, 0.9 s) is a copy of a copy: each window matches a slightly
worse tail and gets worse itself. The remedy has to keep continuity at the cut without handing the whole tail down.
`H3_TP2_CONT_EDIT_AUDIO` therefore takes soft values (`h3_tp2/cont_match.py:audio_pin_mask`; the upstream latent edit
lerps model input, velocity and the restored row by the mask value, so a soft mask is a partial condition, not a hack):

| value | what later windows do with the 37-row audio prefix |
|---|---|
| `1` (default) | hard pin of the previous tail |
| `ramp` | mask 0 at the seam rising linearly to 1 at the end of the prefix: continuity at the cut, free after |
| `rows:N` | hard pin of the first N rows only (N=12 is 0.3 s) |
| `0.5` | constant soft mask on the whole prefix |
| `0` | no audio pin (this section's `noapin64`) |

queue113 renders the doll chain with `ramp` and `rows:12` and the 60 s painting (FL2VA, continuous ambience, where a
cut would be heard) on the default and with `ramp`, scoring voice, speakers and seams the same way.

Clips: `voicepl-def64`, `voicepl-toff64`, `voicepl-noapin64` (64.6 s each, on the Mac as
`~/Downloads/h3-prod-smoke/voice/lalka-chain64-*.mp4`).

## 10. The soft hand-overs measured, and the default: pin released after 30 % of the steps (queue113-115)

Same doll chain as section 9 (six 277-frame windows, 64.6 s, seed 3401, Ref2VA), scored per window with UTMOS22 on the
spoken segments, WavLM x-vector cosine per speaker against window 1, and the log-mel seam distance at the five hand-overs
against the clip's own 250 ms grid.

| arm | audio prefix | per-window UTMOS | windows 2-6 mean | min |
|---|---|---|---|---|
| `def64` (section 9) | hard pin | 3.87 3.53 3.48 3.06 3.31 2.93 | 3.26 | 2.93 |
| `noapin64` (section 9) | none | 3.89 3.86 3.72 3.75 3.18 3.68 | 3.64 | 3.18 |
| `ramp64` | 0 at the seam rising to 1 | 3.88 3.47 3.48 3.40 3.65 3.57 | 3.51 | 3.40 |
| `rows12-64` | first 12 rows pinned | 3.88 3.54 3.65 3.22 3.32 3.27 | 3.40 | 3.22 |
| `steps30r-64` | hard pin for the first 30 % of the steps, free after | 3.87 3.70 3.60 3.26 3.70 3.75 | **3.60** | 3.26 |
| `steps50-64` | hard pin for the first 50 % of the steps, free after | 3.88 3.56 3.36 3.00 3.07 2.96 | 3.19 | 2.96 |

Speaker cosines (low voice / high voice, windows 2-6): `ramp64` 0.98/0.97, -/0.70, 0.94/0.81, 0.92/0.95, 0.96/0.98;
`rows12-64` 0.91/0.98, 0.94/0.77, 0.93/0.95, 0.92/0.92, 0.89/0.98; `steps30r-64` 0.94/0.97, 0.95/0.95, 0.95/0.94,
0.93/0.84, 0.97/0.98; `steps50-64` 0.96/0.96, 0.95/0.98, 0.96/0.87, 0.94/0.95, 0.91/0.85. All hold the speakers as well
as the control does; the step release at 30 % is the only arm with every cosine at 0.84 or above.

Seams (log-mel distance at the five hand-overs, then the clip's grid median and 95th percentile): `ramp64` 15.6, 28.5,
23.4, 31.4, 11.4 against 19.3 / 45.1; `rows12-64` 15.6, 29.3, 25.1, 25.2, 11.0 against 19.2 / 43.9; `noapin64` 15.6,
24.6, 21.1, 20.7, 7.5 against 15.7 / 36.3; `steps30r-64` 15.6, 16.8, 19.9, 25.2, 9.9 against 17.2 / 39.9; `steps50-64`
15.6, 17.5, 24.7, 26.0, 14.5 against 19.4 / 47.4. Neither spatial soft mask makes the cut quieter than no pin at all;
the step release at 30 % does.

Reading: the ramp recovers half of the loss (3.51 against 3.26 and 3.64) and `rows:12` almost none, so the loss is not in
the first rows after the cut: any pinned rows hand the previous window's audio texture down for the whole sampling, and
the free sampler restores the voice as soon as the prefix is free. Splitting by time instead of by row does what the
spatial masks cannot: the first 30 % of the steps (6 of 20) with the pinned tail fix what sounds, who speaks and how
loud, the remaining 70 % sample the whole window free and the voice comes back to the no-pin level (3.60 against 3.64)
with the quietest hand-overs of any arm. Holding the pin to 50 % of the steps (10 of 20) already loses what the whole-run
pin loses (3.19): the texture that degrades along the chain is set in the middle steps.

**The painting does not need the pin either.** The 60 s FL2VA Koninck painting (continuous room ambience, where a cut
would be heard) on the four hand-overs:

| arm | seam distance at the five hand-overs | grid median / p95 | light, coherence, flicker |
|---|---|---|---|
| `K-def` (hard pin) | 2.4 4.0 4.3 7.1 5.2 | 4.4 / 7.2 | baseline |
| `K-ramp` | 2.4 3.6 4.6 3.5 4.1 | 4.4 / 7.1 | identical curves, no flicker |
| `K-noapin` | 2.4 3.6 4.5 7.8 4.0 | 4.1 / 8.3 | identical curves, no flicker |
| `K-steps30` | 2.4 3.3 4.3 5.0 4.4 | 4.5 / 7.5 | identical curves, no flicker |

Every seam of every arm sits inside the clip's own grid: the ambience hands over without a pin as well as with one. The
video is untouched by the audio pin (the framemd5 differs because the joint model samples video and audio together, the
light and coherence curves are the same to the second digit).

**The default since 2026-11-07 21:35: `H3_TP2_CONT_EDIT_AUDIO_STEPS=0.3`.** `h3_tp2/audio_pin_steps.py` wraps the denoise
loop with an edit that is the real pinned edit before the release step and the free sampler's arithmetic from it on
(the first build patched the step API's request state, which a render never touches, so queue114's `steps30-64` was bit
for bit the control and is not listed). `1` restores the whole-run pin, `H3_TP2_CONT_EDIT_AUDIO=0` drops the audio pin
altogether (the same voice, louder cuts in dialogue). Deliberate change of the audio output, not output-preserving;
the video latents are not touched (the joint model samples both, so the frames differ too). queue116 rendered the doll
chain on the defaults alone: bit for bit `steps30r-64` (framemd5 652da51e); the per-window line "audio pin released at
step 6 of 20" is written by the rank that holds the audio rows in the Ulysses split, rank 1 (`~/h3-tp2-logs/worker-rank1-*.log`
on spark-05), not by the head's journal. `tests/test_audio_pin_steps_loop.py` runs the gate inside upstream's verbatim
loop over 20 steps: identical to the stock loop through step 5, different from step 6 on, different from no pin.

The video side of the release (2026-11-08, `clip-metrics.sh` and `face_drift.py` on the three doll chains). The
model samples video and audio jointly, so freeing the audio rows changes the video digest too (`def64` ea7560b2,
`steps30r-64` 652da51e, `noapin64` bbd22ba9; window 1 identical in all three). The pictures are the same clip: motion,
detail and coherence per second within a unit of each other, the five video seams 0.33-0.80 of the local motion (hard
pin) vs 0.34-0.58 (released) vs 0.43-1.10 (no pin), luma steps at the seams <= 0.2 in every arm, no cut, no
17-frame flicker; faces vs the reference (embedding match, every second) 0.813 / 0.707 (hard pin), 0.818 / 0.700
(released), 0.823 / 0.701 (no pin) for the two figures, minima 0.46 / 0.29 / 0.44 for the second figure from one
turned-head sample each. Nothing the default changes in the picture is measurable.

Pronunciation is not the hand-over's doing. `scripts/script_wer.py` scores every window's transcript against that
window's `<d>` lines: the six chains land at 0.03-0.07 word error rate over the clip (re-scored 2026-11-08 with the
aligned tool and the respelling table, see below; the prompt's phonetic `morze` / `pieniondze` are not errors when
`może` / `pieniądze` is heard) with the same words slipping in every arm
(`cierpliwy` -> `cierplowy` in six of six, `czynsz` -> `czynż`, a fused `dziewczyna pisze` -> `dziewczy napisze`, the
last window worst everywhere, window 1 clean everywhere). Those are the model's Polish `sz` clusters and word boundaries
(`docs/DIALOGUE_CLIPS.md`), the same with or without a pin. The default (`default-64`, = `steps30r-64`) scores
0.047 over the clip, the hard pin (`def64`) 0.047, no pin 0.060, ramp 0.067, rows:12 0.033, 50 % release 0.027: one
seed, one script, a spread of 0.04 across arms that differ only in the hand-over, so no arm is better at the words.

(2026-11-08) The first version of `script_wer.py` put each Whisper segment into the window that held its midpoint;
a segment that fused the last line of one window with the first line of the next charged the whole fusion to one
window and the other window for the missing words (the shop clip read 0.13 and 0.11 in windows 2 and 3 that way,
against 0.0 and 0.0 after the fix). The tool now aligns the whole transcript to the whole script and charges each
error to the window of the script word it touches, so the per-window errors add up to the clip's own word error rate
(`clip` in the output, equal to `speech_qa.py` on the same transcript).

Clips: `voicepl-ramp64`, `voicepl-rows12-64`, `voicepl-steps30r-64`, `voicepl-steps50-64` (64.6 s doll chains),
`koninck60-K-def`, `koninck60-K-ramp`, `koninck60-K-noapin`, `koninck60-K-steps30` (60 s paintings), on the Mac as
`~/Downloads/h3-prod-smoke/voice/lalka-chain64-*.mp4` and `~/Downloads/h3-prod-smoke/painting/koninck60-K-*.mp4`.
queue116 renders the doll chain on the defaults alone (must reproduce `steps30r-64` bit for bit) and the public shop
scene on today's defaults.

## 11. Sampling levers do not move Polish pronunciation (queue116 part 3, 2026-11-08 02:07-05:35)

Three Polish windows from the doll script (`voice/pl_w{4,9,10}.txt`, three lines each), two seeds each, rendered as
single 11.5 s 960x544 Ref2VA windows from the generic reference 2208 under three settings: P20 = the default 20 steps,
P30 = 30 steps, S2 = `audio_flow_shift` 2.0 (default 3.0). Whole-window word error rate (faster-whisper small, Polish)
and UTMOS22 per window.

| script, seed | P20 WER / UTMOS | P30 WER / UTMOS | S2 WER / UTMOS | the slip |
|---|---|---|---|---|
| w4, 3600 | 0.042 / 3.02 | 0.042 / 2.99 | 0.042 / 3.07 | `wyjścia` -> `wyjście` |
| w4, 3700 | 0.083 / 3.16 | 0.083 / 3.03 | 0.125 / 3.09 | `dlatego` -> `dla tego`; S2 adds a stutter `ty tylko` |
| w9, 3600 | 0.000 / 3.28 | 0.000 / 3.26 | 0.000 / 3.36 | clean |
| w9, 3700 | 0.043 / 3.31 | 0.043 / 3.41 | 0.000 / 3.25 | `dziewczyna` -> `dzieczyna` |
| w10, 3600 | 0.048 / 3.39 | 0.048 / 3.36 | 0.048 / 3.42 | `długo` -> `dłongo` (nasal) |
| w10, 3700 | 0.048 / 3.39 | 0.048 / 3.34 | 0.048 / 3.39 | `długo` -> `dłongo` |
| mean | 0.044 / 3.26 | 0.044 / 3.23 | 0.044 / 3.26 | |
| render per window | 584 s | 857 s (1.47x) | 584 s | |

Reading: the transcripts of the three settings are the same words, slip for slip, at every seed (the two S2 deviations
are one slip less and one stutter more, a wash); 30 steps costs 1.47x the time and buys nothing, a lower audio flow
shift buys nothing. UTMOS within 0.1. Pronunciation is decided by the text and the seed, not by the sampler: the
nasal vowel (`długo`) and the word boundary (`dlatego`) are the known Polish weaknesses, and the prompt-side phonetic
respelling (`docs/DIALOGUE_CLIPS.md`) remains the only lever that moved it (0.082 -> 0.035 on the 60 s scene). Nothing
changes in the defaults. Note the per-window WER here is the single-window word error rate; the same windows inside the
64.6 s chains scored 0.0-0.35 per window with the last window worst, so the chain adds slips of its own on top of the
text's (section 10) but no hand-over arm removed them either.

Clips: `pron-{P20,P30,S2}-w{4,9,10}-s{3600,3700}` on Spark_06 (`~/h3-tp2-results/`), transcripts `voice/seg_pron-*.json`.

## 12. The chain adds a few slips the fresh windows do not make (queue117, 2026-11-08 06:22-07:43)

Windows 2-6 of the 64.6 s doll chain (`voicepl-default-64`) rendered as fresh single 11.5 s windows from the same
reference with the chain's own seeds (3401 + k - 1, the same initial noise) and the same per-window prompt, so the only
difference is the hand-over context (the pinned 22-frame video tail, the audio prefix released at 30 %). Scored with
the respelling table (`voice/respell_pl.txt`: `morze może`, `pieniondze pieniądze`, `porzondnie porządnie`), so only
words the model got wrong count.

| window | chain WER (slips) | fresh WER (slips) | UTMOS chain / fresh |
|---|---|---|---|
| 2 | 0.000 | 0.038 (`dziewczyna` -> `dziewcza`) | 3.70 / 3.23 |
| 3 | 0.080 (`i` -> `jej`, `porządnie` -> `pożąd`) | 0.000 | 3.60 / 3.48 |
| 4 | 0.042 (`wyjścia` -> `wyjście`) | 0.000 | 3.26 / 3.37 |
| 5 | 0.000 | 0.000 | 3.70 / 3.48 |
| 6 | 0.200 (`cierpliwy` -> `cierplawy`, `czynsz` -> `czyż`, `dziewczyna nie` fused) | 0.050 (`cierpliwy` -> `cierplowy`) | 3.75 / 3.21 |
| windows 2-6 | 7 slips in 121 words (0.058) | 2 slips in 121 words (0.017) | 3.60 / 3.35 |

Reading, one seed: the words that slip in every render (`cierpliwy`) slip here too; on top of them the chain adds
five slips the fresh windows do not make, four of them in the last window (the pattern of section 10: the last window
worst in every arm). The voice, on the other hand, is not worse in the chain: with the released pin the chain's
windows score 3.60 UTMOS against 3.35 for the same lines rendered fresh (the chain inherits the first window's clear
voice; a fresh window draws its own). So after the 2026-11-07 default the chain costs words, not timbre, and only a
little: about 4 words in 120 on this script. What in the hand-over context causes the slips is open (the pinned video
tail, the 30 % audio prefix, or simply more speech in the model's context); the sampler levers are ruled out
(section 11). Candidates for a later test, each a 64.6 s chain per arm: a shorter audio prefix together with the step
release (`rows:12` scored 0.033 here, the lowest hard-pin variant, one seed), and the text lever of giving the last
window one line less. The respelling table is a generic feature of the tool now (`--respell`); without it the prompt's
phonetic spellings inflated every Polish number in sections 10-11 by 0.02-0.03.

Re-counted by characters (2026-11-08 10:00, `script_wer.py` now prints a per-window character error rate with the word
boundaries removed): of the chain's 7 word errors, 2 are Whisper word boundaries (`dziewczyna nie` written as one word
in window 6) and 5 are real slips (`jej`, `pożąd`, `wyjście`, `cierplawy`, `czyż`), chain character error rate 0.013
against 0.005 for the two fresh slips (`dziewcza`, `cierplowy`). The conclusion stands, the count is 5 extra words, not
7; `dziewczy napisze` in other arms is the same boundary effect and no slip at all.

Clips: `fresh-pl6-w{2..6}` on Spark_06 (`~/h3-tp2-results/`), transcripts `voice/seg_fresh-pl6-w*.json`.

## 13. Local RoPE time (`toff0`) halves the slips and lets the voice drift again (queue118, 2026-11-08 08:37-15:14)

Upstream shifts every media row of window k by its start frame in RoPE units (window 6 of a 64.6 s clip sits 2125
units from the reference image and the text, a 15 s training clip spans about 600). `H3_TP2_ID_LOCK=toff0`
(`h3_tp2/identity_lock.py`) keeps every window at local time 0 instead. Under the hard audio pin it had made 3 slips
where the default made 7 (section 10, one seed), so queue118 renders the doll chain pl6 (64.6 s 960x544 20 steps, the
released pin of the 2026-11-07 default) default against `toff0` at three seeds. Slips are counted by the character error
rate without word boundaries (`script_wer.py --respell`, see section 12), the voice by UTMOS of the spoken lines, the
same lines of the two arms paired by text (`scripts/voice_pairs.py`).

| seed | arm | real slips (words) | CER | clip WER | UTMOS windows 2-6 | paired UTMOS, B - A | window 6 paired | render s |
|---|---|---|---|---|---|---|---|---|
| 3401 | default (`voicepl-default-64`) | 5: `jej`, `pożąd`, `wyjście`, `cierplawy`, `czyż` | 0.013 | 0.047 | 3.60 | | | 3585 |
| 3401 | toff0 | 1: `sklepie winiem` | 0.005 | 0.047 | 3.35 | -0.19 (A better 12 of 17) | -0.59 | 3633 |
| 3501 | default | 6: `z`, `obsować`, `czekanie`, `cierplowy`, `czynż`, `mieniące` | 0.025 | 0.073 | 3.68 | | | 3645 |
| 3501 | toff0 | 4: `pieniądzie`, `pożądnie`, `cierplowy`, `czyż` | 0.009 | 0.027 | 3.54 | -0.14 (A better 11 of 17) | -0.70 | 3649 |
| 3601 | default | 6: `pieniądzie`, `po rządzie`, `obspoweć`, `cierplowy`, `czynż`, `pełnie` | 0.014 | 0.047 | 3.65 | | | 3570 |
| 3601 | toff0 | 4: `porządźnie`, `czynż`, `czyż`, `pełe` | 0.009 | 0.040 | 3.54 | -0.11 (A better 12 of 15) | -0.36 | 3593 |

Reading, three seeds, all in the same direction: `toff0` cuts the real slips of windows 2-6 from 5-6 to 1-4 (character
error rate 0.005-0.009 against 0.013-0.025; the words that slip under both arms are the `sz` clusters of section 10,
`cierpliwy` and `czynsz`) and costs the voice 0.11-0.19 UTMOS per paired line, the loss growing along the chain to
-0.36 .. -0.70 in the last window. Two default seeds differ by +0.06 per line and +0.02 in the last window
(`voice_pairs.py default-64 def-s3501`), so the drift is the arm's, not the seed's: with local time the voice
degrades window by window as it did under the hard pin (section 9), and the released audio pin does not rescue it
(`toff0` on the released pin scores like `toff64` on the hard pin, 3.07-3.12 in window 6). Speakers hold under both
arms (cosine to window 1 at 0.91-0.98; the `toff-s3601` high voice reads 0.87 already in window 1 because one of the
man's window-1 lines landed on the high side of the 150 Hz pitch split and mixed into the woman's reference, so its
0.85-0.89 is a reference artefact, not a drift), seams sit inside the clip's own grid spread, render time
is equal. The digests differ per arm and per seed, as they must.

So the RoPE time is a real lever on the words but not a free one: a window at time 0 is sampled as the start of a clip,
and its voice, pinned to a tail that was spoken "later", drifts. `toff0` stays opt-in (fewer slips in a dialogue where
the words matter more than the last window's timbre). The middle ground is `H3_TP2_ID_LOCK=toffcap` (new, same module):
every later window sits at window 2's RoPE time (425 units, inside a training clip's span, not at an opening);
queue119 renders it at the same three seeds.

Clips: `voicepl-{def,toff}-s{3501,3601}`, `voicepl-toff-s3401` on Spark_06 (`~/h3-tp2-results/`), transcripts and
scores `voice/{seg,segscore,win}_voicepl-*.json`, queue `queue118-toff.sh`.

## 14. Capping the RoPE shift sits exactly between (`toffcap`, queue119, 2026-11-08 15:15-19:10)

`H3_TP2_ID_LOCK=toffcap` (new in `h3_tp2/identity_lock.py`) keeps every window from the third on at window 2's RoPE
time (425 units, `H3_TP2_ID_LOCK_TOFF_CAP`) instead of upstream's growing shift (section 13's default) or zero
(`toff0`). Windows 1 and 2 are therefore identical to the default render of the same seed (the paired lines differ by
0.00), and the arm changes only windows 3-6. Same chain, same three seeds, same scoring as section 13.

| seed | arm | real slips (words) | CER | UTMOS windows 2-6 | paired UTMOS, B - A | window 6 paired | render s |
|---|---|---|---|---|---|---|---|
| 3401 | default | 5 | 0.013 | 3.60 | | | 3585 |
| 3401 | toffcap | 3: `winiem`, `cierplawy`, `czyż` | 0.0065 | 3.47 | -0.10 (A better 9 of 19) | -0.48 | 3702 |
| 3401 | toff0 | 1 | 0.005 | 3.35 | -0.19 | -0.59 | 3633 |
| 3501 | default | 6 | 0.025 | 3.68 | | | 3645 |
| 3501 | toffcap | 2: `cierplowy`, `czyż` (+ an inserted `z`) | 0.0065 | 3.58 | -0.07 (A better 9 of 19) | -0.27 | 4028 |
| 3501 | toff0 | 4 | 0.009 | 3.54 | -0.14 | -0.70 | 3649 |
| 3601 | default | 6 | 0.014 | 3.65 | | | 3570 |
| 3601 | toffcap | 3: `porządznie`, `czyż`, `pelłe` | 0.0079 | 3.54 | -0.105 (A better 13 of 15) | -0.29 | 3515 |
| 3601 | toff0 | 4 | 0.009 | 3.54 | -0.11 | -0.36 | 3593 |

Reading, three seeds: the cap lands between the two on every number. Slips 2-3 per chain (default 5-6, `toff0` 1-4),
character error rate 0.0065-0.0079 (0.013-0.025 and 0.005-0.009), voice -0.07 .. -0.10 per paired line (`toff0`
-0.11 .. -0.19), last window -0.27 .. -0.48 (`toff0` -0.36 .. -0.70). The loss again shows up two windows after the
first window the arm changes (windows 3-4 at 0.00 .. -0.13, windows 5-6 lower), so it is the same cumulative drift as
under `toff0`, started two windows later, not a function of how far the time was moved: once a window's prefix is placed
at a RoPE time other than the one it was generated at, the hand-over inherits a slightly worse voice and the chain
compounds it. The words, on the other hand, improve with how close to the text and the reference the window sits.

So there is no setting of the RoPE time that gives the default's voice with fewer slips; the two modes are a dial
between words and voice and both stay opt-in (`toffcap` when a dialogue's words matter and the voice of the last
windows a little less; `toff0` when the words matter most). The lever that does not touch the dial is the speech guard
(section 15): judge each window's words after sampling and re-sample the window that slipped.

Clips: `voicepl-cap-s{3401,3501,3601}` on Spark_06, scores `voice/*_voicepl-cap-*.json`, queue `queue119-cap.sh`.

## 15. The speech guard: re-sample the window that slipped (queue121-125, 2026-11-09 00:53-20:29)

`H3_TP2_SPEECH_GUARD=<max CER>` (`h3_tp2/speech_guard.py`, `docs/DIALOGUE_CLIPS.md`): after a window is sampled its
audio is decoded, transcribed on the CPU and compared with the `<d>` lines of that window's prompt as a character error
rate without word boundaries; above the limit the window is sampled again with another seed (two retries, the best
attempt is kept). It does not touch the sampling itself, so a window that passes is the default render bit for bit:
windows 1-2 of the guarded clips below have the video and audio hashes of the plain clips of the same seed.

**Which limit (dry run on finished clips).** `scripts/speech_guard_dryrun.py` on the 18 chains of sections 9-14
(108 windows, the `small` judge):

| share of windows above the limit | 0.005 | 0.01 | 0.02 |
|---|---|---|---|
| Polish, 90 windows | 40 % | 22 % | 11 % |
| English, 18 windows | 28 % | 11 % | 6 % |

A window holds 100-135 script characters, so 0.01 is "at most one wrong character", 0.005 is "none" (and pays for
Whisper's own mishearings), 0.02 lets a one-word slip through. Two judges: on 48 windows scored by both `small` and
`medium`, each alone puts 11 above 0.01 and both agree on 8, so three of every eleven re-samples a single judge orders
are its own hearing (`H3_TP2_SPEECH_MODEL=small,medium`: the lowest rate counts, `medium` is asked only when `small`
objects).

**Live, three seeds** (the 64.6 s Polish chain of section 13, 6 windows, 764 script characters, limit 0.01, one judge,
against the default render of the same seed):

| seed | arm | re-sampled | wrong characters (both judges) | whole-clip CER | UTMOS windows 2-6 | paired UTMOS, B - A | render s |
|---|---|---|---|---|---|---|---|
| 3401 | default | | 2 | 0.0131 | 3.60 | | 3585 |
| 3401 | guard | w3: 0.037 > 0.000 | 1 | 0.0026 | 3.49 | -0.09 (A better 9 of 19) | 4207 |
| 3501 | default | | 13 | 0.0249 | 3.68 | | 3645 |
| 3501 | guard | w4: 0.046 > 0.046 > 0.030 (kept), w6: 0.017 > 0.009 | 5 | 0.0131 | 3.52 | -0.13 (A better 10 of 19) | 5289 |
| 3601 | default | | 6 | 0.0144 | 3.65 | | 3570 |
| 3601 | guard | w3: 0.015 > 0.052 > 0.000, w4: 0.038 > 0.000 | 1 | 0.0013 | 3.55 | -0.11 (A better 14 of 17) | 6571 (slow host, see below) |

"Wrong characters" is the guard's own method on the finished clip with two judges (`dry2_*.json`); the whole-clip rate
is `script_wer.py` on one transcription of the whole clip, an independent reading.

Reading:

- **Words.** 21 wrong characters in the three default clips, 7 in the guarded ones; whole-clip character error rate
  0.0175 -> 0.0057 on average. What is left is `cierplowy` for `cierpliwy` in window 6 of all three (one character, under
  the limit) and window 4 of seed 3501.
- **A word the model slips on at every seed is not fixed by re-sampling.** Window 4 of seed 3501 said `zaprośmy ją`
  wrong in all three attempts (the guard's summary line now names such words: `still above 0.01 after every attempt:
  w4 (zaprośmy ją)`). Two samplings spent, nothing gained: rewrite or respell the line.
- **Two retries are the right number.** Three windows failed their first two attempts (one of them in the void
  queue120, whose verdicts were right); two of the three passed on the third (0.015 > 0.052 > 0.000 and
  0.046 > 0.053 > 0.000), so stopping after one retry would have kept two slips.
- **Cost.** One extra sampling of one window per re-sample (about 9.3 min at 960x544 and 20 steps): 1, 3 and 3 extra
  samplings on a 6-window chain, +17 % and +45 % render time on seeds 3401 and 3501. Seed 3601 ran its nine samplings
  at 34.7 s per step instead of 27.8 (the host was in a slow state for that whole render, the rate is unrelated to the
  guard: the second chain of queue119 did the same for three windows, 4028 s in section 14); at the normal rate it is
  about 5300 s, +48 %.
- **Voice.** UTMOS falls by 0.09-0.13 per paired line, and the loss sits in the windows after the first re-sampled
  one, not in the re-sampled window itself:

  | seed | first re-sampled | paired UTMOS per window, guard - default (w1 .. w6) | paired DNSMOS SIG, clip |
  |---|---|---|---|
  | 3401 | w3 | 0.00, 0.00, +0.04, +0.03, -0.37, -0.27 | +0.07 |
  | 3501 | w4 | -0.03, 0.00, 0.00, -0.40, -0.24, -0.27 | -0.01 |
  | 3601 | w3 | -0.09, -0.05, +0.12, -0.32, -0.07, -0.14 | -0.05 |

  Windows 1-2 are the same audio in both arms, so their -0.03 .. -0.09 is the noise of pairing two or three lines per
  window. The three default seeds agree within 0.1 in windows 5-6 (3.70-3.81), the three guarded clips sit at
  3.32-3.66 there. Speaker similarity to window 1 stays within 0.03 of the default clips (the second voice of seed 3601
  is scored against another reference line and cannot be compared). The same shape appeared when the RoPE time was
  changed (sections 13-14).
- **A re-sampled window as such costs nothing (control, queue125).** Seed 3601 again with the limit at 0.5 (nothing is
  rejected) and window 3 re-sampled anyway, with no selection: `H3_TP2_CONT_RETRY_FORCE=3` samples window 3, drops the
  result and samples it again with the first retry seed (11522; 7 samplings, 4323 s). A second render in a new head
  process samples window 3 once with that seed (`H3_TP2_CONT_RETRY_SEED_SHIFT=3:7919`; 6 samplings, 3592 s). The two
  files are byte for byte the same, so the retry path leaves nothing behind: a re-sampled window is the window that
  seed gives on a first sampling. Against the default render of seed 3601:

  | clip | window 3 | paired UTMOS per window, clip - default (w1 .. w6) | paired UTMOS, clip | UTMOS windows 2-6 | paired DNSMOS SIG, clip |
  |---|---|---|---|---|---|
  | re-sampled, no selection | seed 11522, CER 0.052 (`piędzie` for `pieniądze`) | -0.09, -0.02, 0.00, -0.03, +0.03, +0.04 | -0.01 (default better 9 of 18) | 3.65 (default 3.65) | -0.05 |
  | guard, limit 0.01 (table above) | seed 19441, CER 0.000, then w4 re-sampled | -0.09, -0.05, +0.12, -0.32, -0.07, -0.14 | -0.11 | 3.55 | -0.05 |

  The chain behind an arbitrary other window 3 keeps the default voice (windows 5-6: 3.76 and 3.74, default 3.73 and
  3.71). So the 0.09-0.13 of the guarded clips is not what re-sampling does. What is left is either the attempt the
  guard keeps (the cleanly spoken one) or chain-to-chain spread: windows 5-6 of the 15 Polish chains of sections 9-14
  range from 3.02 to 3.79, and the four default-setting chains measured so far (three seeds and this control) all sit
  at 3.72-3.79. In those 15 chains a window with a slip scores 0.06 UTMOS above a clean one once chain and window
  means are removed (r = 0.12 over 75 windows, inside the noise), and the window after a slip does not differ (-0.02).
  Two more controls (queue126, 2026-11-11, seeds 3401 and 3501, window 3 sampled once at the second retry seed, no
  selection): paired UTMOS against the plain render -0.12 (windows 3-5 at -0.22, -0.29, -0.27, window 6 +0.02) and
  +0.00 (window 3 +0.20, then -0.09, -0.05). So a window re-rolled with another seed can change the voice of the
  windows after it by a quarter point either way: that is the chain's seed-to-seed spread, and the guarded clips'
  -0.1 sits inside it. The guard has no cost of its own; what it costs or gains per clip is a draw.

- **A 118 s chain, 11 windows (queue124).** The doll scene with the 120 s script, seed 3701, plain and then with the
  guard at 0.01, two judges (`small,medium`), same revision and defaults:

  | arm | window samplings | render | wrong characters of 1302, finished clip, two judges | per window (w1 .. w11) | whole-clip WER (`small`) | UTMOS windows 2-11 (w11) |
  |---|---|---|---|---|---|---|
  | plain | 11 | 6566 s | 50 | 0, 0, 0, 2, 0, 0, 1, 9, 10, 4, 24 | 0.104 | 3.44 (2.94) |
  | guard 0.01 | 19 | 11 519 s (+75 %) | 13 | 0, 0, 0, 0, 1, 0, 1, 0, 9, 2, 0 | 0.042 | 3.53 (3.41) |

  The plain chain falls apart at the end (window 8 `sklepjestzbytewnie` for `w sklepie zupełnie`, window 11 loses most
  of a line): the slips of a long chain are not single letters. The guard re-sampled five windows (w4 0.015 > 0.000;
  w7 0.010 three times; w8 0.253 > 0.000; w10 0.020, 0.020, 0.051; w11 0.161 > 0.021 > 0.000), eight extra samplings.
  Two words stay wrong at every seed (`klientków` for `klientów`, `dłongo` for `długo`) and cost four samplings for
  nothing: they belong in the prompt's phonetic spelling, which is what the dry run is for.
  Paired UTMOS guard - plain: clip +0.08 (guard better on 15 lines, plain on 9, of 33), per window from w4:
  +0.12, +0.12, +0.05, +0.35, 0.00, -0.13, -0.03, +0.47; DNSMOS SIG +0.05. So this chain gained voice where the three
  64.6 s chains lost about 0.1: four chains from -0.13 to +0.08, which reads as chain-to-chain spread and not as a
  cost of the guard.
  Nine of the 13 characters left are the two words `Nie słucha` that open window 9. The guard, hearing window 9 alone,
  has them (`Nie sucha`, one character); in the finished clip they are missing in both arms. The line starts inside the
  0.92 s at the start of the window that repeat the previous window's tail, and the clip keeps the previous window's
  version of them. One window in the 370 continuation windows scored so far. Since 2026-11-09 the guard listens to
  the part of a window the clip keeps (`cont_match.heard_audio`), so such a window is re-sampled; not yet run on the
  GPU.

**English, one clip** (the 60 s 1344x768 showcase chain, seed 2304, limit 0.01): one re-sample, window 5
(0.019 > 0.000). Wrong characters by the guard's method 5 -> 1 of 713 (the one left is `Novak` for `Nowak`, a spelling
of the judge: put names in the respelling table). Paired UTMOS +0.02 (windows 5-6 +0.03, +0.05), so no voice loss
here. The whole-clip rate reads 0.0084 -> 0.0168, and that is the transcriber, not the clip: transcribing the whole
clip, `small` writes a `Still cannot.` between the last two lines of window 6 that the same model does not hear when
it is given window 6 alone (there it returns the script word for word, and `medium` one different word). A single
whole-clip transcription moves by more than the guard does on a clip this clean; the per-window reading with two
judges is the steadier one.

**Recommendation.** Opt-in, `H3_TP2_SPEECH_GUARD=0.01`, with `H3_TP2_SPEECH_MODEL=small,medium` and a respelling table
for names and phonetic prompt spellings. Run the dry run on a clip you already have first: it names the words, and a
word that is wrong there in every clip belongs in the prompt, not in the guard. Expect about one re-sample per five
Polish windows and one per nine English ones.

queue120 (2026-11-08 19:11-00:30) is void: the respelling table existed on the head only, rank 1 ran without the guard
and the ranks hung on mismatched collectives (fixed: the guard is configured from the forwarded environment only).

Clips: `voicepl-guard01b-s3401`, `voicepl-guard01c-s{3501,3601}`, `xclip60guard` on Spark_06; scores
`voice/dry2_*.json`, `voice/segscore_*.json`; queues `queue121-xguard.sh`, `queue122-guard.sh`, `queue123-guard.sh`.
Control: `voicepl-force3-s3601`, `voicepl-shift3-s3601` (`queue125-retry.sh`, rev 779102a).
118 s: `lalka118-plain-s3701`, `lalka118-guard-s3701`, `voice/dry2_lalka118-*.json` (`queue124-long.sh`, rev 779102a).

## 16. How early a slip can be heard (queue127, 2026-11-11 00:15-03:02)

`H3_TP2_SPEECH_EARLY=4,6,8,10,12,14,16,20` decodes the denoising loop's clean estimate of the audio at those steps and
transcribes it (observe only). Two 64.6 s chains, seeds 3501 and 3601, guard limit 0.5 (nothing re-sampled). Both clips
are byte for byte the plain renders of their seeds (video and audio), so the tap is lossless. The journal logged only
some of the checks (a logging gap, the transcripts of all 96 are on disk), so the table was scored offline with the
guard's own method from the saved transcripts. Character error rate of the window's lines at step S of 20 (the audio
ladder with shift 3 still has 75 % noise at step 10, 56 % at step 14, 43 % at step 16):

| seed | window | s4 | s6 | s8 | s10 | s12 | s14 | s16 | s20 = final |
|---|---|---|---|---|---|---|---|---|---|
| 3501 | 1 | 0.907 | 0.791 | 0.217 | 0.108 | 0.000 | 0.000 | 0.008 | 0.000 |
| 3501 | 2 | 0.879 | 0.452 | 0.121 | 0.032 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3501 | 3 | 0.659 | 0.385 | 0.148 | 0.030 | 0.074 | 0.037 | 0.037 | 0.007 |
| 3501 | 4 | 0.651 | 0.659 | 0.167 | 0.091 | 0.045 | 0.045 | 0.045 | **0.045** |
| 3501 | 5 | 1.000 | 0.682 | 0.372 | 0.209 | 0.093 | 0.039 | 0.031 | **0.046** |
| 3501 | 6 | 0.722 | 0.426 | 0.365 | 0.183 | 0.035 | 0.026 | 0.017 | 0.017 |
| 3601 | 1 | 1.000 | 0.589 | 0.240 | 0.140 | 0.015 | 0.000 | 0.000 | 0.000 |
| 3601 | 2 | 0.782 | 0.419 | 0.065 | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3601 | 3 | 0.644 | 0.385 | 0.215 | 0.037 | 0.000 | 0.015 | 0.015 | 0.015 |
| 3601 | 4 | 0.735 | 0.561 | 0.235 | 0.091 | 0.045 | 0.015 | 0.015 | **0.038** |
| 3601 | 5 | 0.736 | 0.612 | 0.388 | 0.108 | 0.031 | 0.000 | 0.000 | 0.000 |
| 3601 | 6 | 0.774 | 0.504 | 0.296 | 0.096 | 0.000 | 0.017 | 0.017 | 0.017 |

- The words are there from step 10-12 on; the fine verdict settles only around step 14-16, and a window's rate still
  moves by 0.01-0.02 between step 16 and the end (window 5 of 3501 up, window 4 of 3601 up).
- An early stop at step 14 with the limit at 0.03 would have caught the two clear failures of seed 3501 (0.045,
  0.039) and raised a false alarm on window 3 (0.037, final 0.007); at step 16 the same. A stop before step 12 is
  noise. So the saving is 20-30 % of a failed attempt, bought with a false alarm now and then that costs a whole one.
  `H3_TP2_SPEECH_EARLY_STOP` stays an experiment that is not worth enabling; the measurement switch stays for the
  record.
- Cost of the measurement: 3-4 s per check (decode + `small`), about 25 s per window at eight steps.

Clips `voicepl-early-s3501`, `voicepl-early-s3601`; transcripts `voice/early_voicepl-early-s*/early-w*-a*-s*/`;
queue `queue127-early.sh`. The second chain of this queue is also where the slow render state was caught and fixed
(`results/20261111-slow-state-compaction/`).

## 17. The voice anchor, first run (queue142, 2026-11-11 14:03-15:32)

`H3_TP2_VOICE_ANCHOR=5`: windows 2-6 of the 64.6 s doll chain (seed 3501) receive the first 5 s of window 1's audio
latent as a Ref2VA audio reference (400 rows, placed at the window's origin). Against the plain render of the seed:

| | plain | voice anchor |
|---|---|---|
| speaker similarity to window 1, low voice, w2 .. w6 | 0.976 0.962 0.975 0.940 0.953 | 0.967 0.981 0.975 0.974 0.945 |
| high voice, w2 .. w6 | 0.952 0.959 0.936 0.939 0.956 | 0.925 0.943 0.928 0.944 0.946 |
| paired UTMOS, clip (per window) | | +0.01 (+0.03 +0.08 +0.17 0.00 -0.12 -0.15) |
| wrong characters, whole clip (per window) | 0.025 (0 0 0.007 0.046 0.039 0.061) | 0.009 (0 0 0.015 0 0 0.043) |
| echoed lines | | none: every word heard is a script word |
| pace | 555-584 s | 557-562 s (+400 reference rows) |

- At 60 s the voices of this seed do not drift, so there is nothing for the anchor to hold: the speaker numbers move
  within their noise in both directions. The clip is not worse (UTMOS neutral, no echo, no seam effect), and two of
  the three windows that slipped in the plain render came out clean, which may be the anchor or the seed's draw.
- The question the anchor was built for is the second minute (speaker similarity 0.75-0.94 by window 11 in the 118 s
  chains). That run is the next step if the voice of long clips becomes the priority; one 118 s chain at 3-4 h.
  (2026-11-12: run as queue145 arm 2; the 0.75-0.94 of the 118 s chains was largely the pitch split handing the man's
  lines to the woman, and with speakers assigned by voice the anchor showed no effect on one seed;
  `results/20261109-long-chain-identity/` section 8.)

Clip `voicepl-anchor5-s3501`; queue `queue142-voice-anchor.sh`.

## 18. A word the model slips at every seed: respell it (queue146, 2026-11-12 08:32-10:48)

Shot 4 of the comedy clip has `chrupnięcie`; at seed 5004 the model said `chrupiącie` in two of three attempts
(queue144). Three spellings of the word in the prompt, four seeds each (5004, 5104, 5204, 5304), 7.5 s at 960x544, the
guard at 0.5 so it judges and logs every take and never re-samples (small + medium judges). The respelling table maps
B back to the real word for the score; C is two tokens, so saying it right costs one space of CER.

| spelling in the prompt | the target word heard right | what was heard instead | line CER per seed | mean |
|---|---|---|---|---|
| A `chrupnięcie` (control) | 0 of 4 | chrupiącie, chrupiąćłe, chrupiąziec, chrupiącie | 0.035 0.070 0.070 0.035 | 0.053 |
| B `chrupnjęcie` | 2 of 4 | chrupniesie, chrupiesię | 0.000 0.035 0.053 0.070 | 0.039 |
| C `chrup nięcie` | 2 of 4 | niecię, niecię | 0.088 0.000 0.088 0.070 | 0.061 |

- The control never says the word: the `-pnię-` cluster collapses to `-pią-` at every seed. Both respellings get it
  right half the time, and the misses move closer to the word (`chrupniesie` keeps the n). With the guard's two retries
  that is 0 % against 87 % odds of a clean take.
- Spelling C costs elsewhere: `wjeżdżamy`, clean in all four control takes, slipped in three of the four C takes
  (`wieszczamy`, `wyjeżdżamy`, `wjeżdżdżamy`). A split word changes the rhythm of the line around it; B does not (one
  slip of `wjeżdżamy` in four).
- The picture is the seed's, not the spelling's: the three takes of a seed share the composition (seed 5204 ignored
  the close-up in all three), so the respelling is free on the image side.
- Verdict: when a word fails at every seed, respell it inside the word (B's kind), map it back in the respelling table
  (`H3_TP2_SPEECH_RESPELL`), and let the guard re-sample; do not split it. Per word, one short screen like this one
  (four seeds, the guard at 0.5) says which spelling the model can say.

Clips `rs{A,B,C}-s{5004,5104,5204,5304}`; queue `queue146-respell.sh`.
