# Polish audio-lock: two 10 s samples and a separate soundbed experiment

On 2026-11-01 the TP2 FL2VA service received a film-still image and a short synthetic Polish Piper voice as
`audio_reference`, with `audio_mode=lock_source`. These are two 960x544, 24 fps, 20-step single-window samples,
not tests of a 300 s continuous generation or of native H3 speech. The voices are synthetic and are not recordings
or clones of the actors.

| Sample | Exact video SHA-256 | Speech and waveform screen |
| --- | --- | --- |
| Initial 10 s test | `32c35a638fde890690cee2c23f6a9f7fd32eaddb0bd602e9ee30ec581c5fe6a4` | Whisper-small recovered the intended line (WER 0.0); three overlapping 3 s speech portions correlated 0.975–0.977 with the source, with 0 s estimated presentation offset. |
| Five-minute assembly, shot 1 of 30 | `46d5f9671cde1b6de7194ae2b6bb9ea6f035895e6f3fa0b9102f9a150261e6cb` | Whisper-small recovered the intended line (WER 0.0); waveform correlations were 0.979–0.987 and estimated presentation offsets 0 s. All 40 sampled frames had a face; a montage showed mouth motion during the line and a calmer mouth afterward. |

The first assembly shot took 487.8 s on the checked pair, plus export/verification, with a 16 ms audio gap at the
exact 10 s boundary. The initial five-minute plan consisted of 30 independently rendered 10 s shots; the owner
rejected the resulting 20 s preview because adjacent scenes had no continuity. That plan was stopped after shot 8
and replaced by the linked story below. These sparse waveform measurements do not prove that the mouth is accurately
synchronized, the performance is natural, the actor's likeness is maintained or the audio is perceptually clean.

The first assembly shot's driving audio was padded with silence. Its decoded output was almost silent during seconds
7–10 (RMS about 0.00003 at 48 kHz), and 74 of 479,232 decoded mono samples exceeded 1.0 in magnitude; lossy
reconstruction can overshoot, so this alone does not prove audible clipping. As a separate local experiment, the
shot was mixed at 0.70 gain with the 10 s H3-generated background from an earlier animation of the same market
still. The background had no recognized speech in one automatic screen; it has not been approved by listening.
With the background at gain 0.20, the output's decoded peak was 0.781, no 48 kHz mono sample exceeded 1.0, and
RMS in seconds 7–9 was 0.00395. The output has all 240 decoded video frames identical to its source; Whisper-small
again recognized the intended words (WER 0.0). Its SHA-256 is
`c385308a3e804c346a1dd99b79b2ad65eaf9372c178e54d87ab3903feb486c0a`.

A gain-0.40 background variant also preserved the video, but the same ASR made one word error (WER 0.071); that
single ASR difference is not a listening verdict. Neither variant is promoted. A scene-matched background for a
market is not automatically appropriate for an interior or garden, and the voice, noise, transitions and lip sync
need human/audio review before a full-film mix.

`scripts/compare_driving_audio.py` is an optional diagnostic for sparse waveform alignment, per-second levels and
file digests. It accounts for available stream timestamps and tests an offset WAV-to-MP4 fixture. It does not
change serving or infer audio/visual synchronization.

The optional SyncNet screen was later extended to sample an interval starting inside a clip. Its first implementation
of the artificial 200 ms control actually produced only 128 ms of delay: after a mid-file trim, ffmpeg's `adelay`
created some silence packets without timestamps, and the next trim discarded part of them. The screen correctly
flagged the failed control. Resetting timestamps from the sample count before the final trim yields a verified
200 ms; a regression fixture compares the decoded PCM samples, and the real first shot's 1–5 s interval then produced
-2 frames versus -7 for the delayed control. The earlier start-at-zero controls were separately checked by waveform
correlation and did have the intended 200 ms delay. These measurements remain exploratory, not a lip-sync verdict.

When looking for a garden soundbed, a separate 12–14.8 s extraction from the earlier 15 s *native-audio* Izabela
sample (`2a233a5e7586b03ac831090c7f1e709668efec90ab6fb03f04e56e08f07dc96b`) produced speech-like, incoherent
Whisper output under both automatic-language and forced-Polish settings. The earlier whole-clip forced-Polish
transcript had WER 0.0 for the requested sentence and did not report that tail. ASR can hallucinate or miss speech,
so this is a tail requiring listening, not proof of an extra line or its language. The tail was not used as an
ambient bed. This observation also illustrates why a clean whole-clip transcript alone cannot certify silent pauses.

An experimental 16 s, two-window FL2VA request was staged with distinct Polish lines at 0.3 s and 12 s in a single
synthetic driving track, plus a different prompt for each window. It is guarded behind the film, the existing quality
and speed comparisons, and Turbo artifact staging; a held predecessor or explicit movie interruption stops it. The
source track's automatic transcript contains both lines, but **no real-model result exists yet**. A Mac collector is
prepared to check each line, source-waveform timing, face samples and the planned handover when the result appears.

The separate `scripts/plan_dialogue_windows.py` produces explicit continuation prompts from time-stamped dialogue.
It assigns a line once to a window's newly generated frames and rejects one that crosses a handover. CPU fixtures cover
10, 15, 30, 60 and 300 s plans and invalid or overlapping entries. It is not connected to serving and cannot guarantee
when or whether a model speaks; the linked story below uses it to prepare its per-window request prompts.

## Linked two-person story: first 30 s (2026-11-01 20:15)

The replacement 300 s plan is one original Polish conversation between Wokulski and Izabela in a Warsaw shop,
split into 30/60/60/60/60/30 s requests. Each later request takes the actual final frame of the preceding clip as
its first-frame reference. The opening reference was generated from supplied actor/still references; voices are
synthetic Piper voices, not the actors' voices. Frame inheritance is not a guarantee of identity, scene or story
continuity. The remaining 270 s is in progress and is **not yet a completed film**.

The first 30 s FL2VA request (960x544, 20 steps, three windows, elapsed 1600.3 s) produced 720 exact frames,
SHA-256 `c50da4341c432b4b446915d205ba891810f63e97946fd795800fcefd33b771d0`; its copied AAC ends 16 ms early.
The first and second latent seam ratios were 1.00 and 0.98. In a sampled contact sheet both characters and their
shop remain present, and close samples show mouth motion on the intended speaker while the other listens. The
separate positional face tracks detected both faces in 60/60 sampled frames. This is not a full playback review.

Whisper-small gave WER 0.033 (one word difference); the three timed speech portions correlated 0.9848, 0.9663
and 0.9639 with the driving audio at zero measured offset. Speaker-specific SyncNet screens returned -1, 0 and -2
frames at 25 fps, and their artificial 200 ms controls recovered the expected shift in all three. The screens
are fallible and neither they nor still frames prove natural acting, actor identity or accurate lip synchronization.

The live journal confirmed the old ambient audio look-lock also modified later locked-source latents (window 2
std 0.3812 to 0.4004), though the speech screens above remained close to the source. `c81f1b6` fixes that interaction
in the repository with a CPU regression; it is not deployed or GPU-validated. The subsequent 60 s block started
with the previous clip's final-frame SHA-256 `0d41c72f9ee892310d3f5c105db4089f7f76e71e059a8fff3c215477f382f85b`.

On the first 30 s clip, stereo decoded sample peaks were below 1.0, but the two identical channels produced 489 mono
downmix samples above 1.0 at 48 kHz; this is a potential downmix issue, not proof of audible clipping. Per-line stereo
RMS was 0.124 / 0.190 / 0.074. A separate `balance_dialogue_audio.py` experiment applies bounded per-line gains and
preserves all 720 decoded video frames. Its line RMS values in the mono screen were 0.127 / 0.127 / 0.117, with a
mono sample peak of 0.917; FFmpeg measured stereo true peak -3.8 dBFS versus -1.2 before. Line waveform correlations
to the original were at least 0.9999 at zero measured shift, and Whisper recognized the intended words in this run.
The experimental copy has not been approved by listening and is not automatically applied to the film.

The first submission failed immediately with HTTP 400: curl `-F` treated a semicolon in text and JSON as form
metadata and truncated the fields. `2827d04` changed text fields to `--form-string`; an actual multipart receiver
and the subsequent live H3 request verified the fix. A previously run camera-default comparison had the same prompt
truncation, so it is not a valid matched baseline. Corrected comparisons are queued after this film.

Dialogue pacing still needs review. On the first clip, `screen_dialogue_pauses.py` detected 17.435 s of audio
below -45 dBFS in intervals at least 0.8 s long (audio duration 29.984 s); the longest interval was 7.389 s.
This measures near-silence, not conversational pauses over background sound or whether the pacing works. The first
clip has not been assessed in full playback, and these long gaps may make the conversation feel slow. The screen
is also scheduled for subsequent blocks so the result is available without changing their audio.

An independent timestamp audit found that comparing decoded sample arrays can miss an offset or an internal PTS
gap: raw sample decoding discards timestamps, and an MP4 packet duration can stretch over a gap. The first clip's
decoded AAC frames and all six source MP3s have continuous frame timestamps. `check_timed_audio.py` now validates
decoded-frame PTS against sample counts, as well as the absolute offset and waveform correlations. On Spark_06 an
isolated future-gate candidate passed the real first block and rejected both a 1 s initial offset and a 1 s internal
gap. The immutable gate of the **already running** film was not replaced; the separate collector now applies the
stronger screen to each result. This remains an audio timing screen, not proof of natural speech or lip sync.

The second linked block completed at 21:09 (60 s, SHA-256
`a63aa0af4bdfdee13d154440192e1e9625c080e578dc0754c720f7436d135ca5`); the third started at 21:09.
The join screen measured a 3.12/255 mean RGB difference at the boundary. Sampled frames show both people in the
same shop and clothing, and two positional face tracks were detected in 120/120 samples each; these checks do not
establish full-playback continuity or actor likeness. Whisper's WER was 0.074 for both the source and output, with
the same four word differences, so those particular recognition differences cannot be attributed to the render.
Every timed line passed the separate frame-PTS and waveform screen, but the final line was close to its threshold:
2.5 s sample correlation 0.669, and output RMS 0.0389 versus source 0.1159, about 9.5 dB lower. This is a measured
level difference, not a listening verdict.

An optional copy of block 2 replaces the rendered soundtrack with its exact synthetic driving speech. All 1,440
decoded video frames and timestamps were preserved; all six timed speech checks passed, with waveform correlations
at least 0.99999 at zero measured offset, and the last line's RMS was 0.1158 versus source 0.1159. Its audio and lip
synchronization have not been judged in playback, so the original remains the main result. The collector prepares
separate source-speech comparisons for later blocks and, if all six pass, an optional 300 s copy. A six-part
synthetic assembly preserved 288 video frames and passed all six waveform/PTS checks; the real 300 s assembly is
still pending.

A separate `lalka-list-w-sklepie-PODGLAD-90s-EXPERIMENT.mp4` joins the original first two clips into a single
90 s preview, SHA-256 `e661487d55ced60355a0076e2eb00234e1f7fa6b05a7186caecec1a4be552701`. It was re-encoded
at CRF 19, contains 2,160 frames at 24 fps and fully decodes. Sparse sampled frames differed from the respective
source clips by 0.71–0.79/255 mean RGB; median sampled audio correlations were 1.00000 and 0.99997 with zero
measured offset. The sample screen verifies neither every frame nor perceived continuity or speech quality. This
is an interim preview, not the requested 300 s movie.

The third linked block completed at 22:03 (60 s, SHA-256
`4efdcfb349e10815171ef59da0eca7828bd49ea6767cec746229f640ef92a608`); the fourth started at 22:03.
Its six timed speech intervals passed the independent waveform and decoded-frame PTS screen; the second join's
sampled boundary difference was 3.46/255. Sampled images retain both people and the shop, while the dialogue and
camera remain slow and largely static. An independent 90 s review measured seven of eight planned inter-line gaps
over six seconds and found few changes of action. This is evidence for testing pacing and staging, not a verdict
on full-playback quality.

The original first three clips also form `lalka-list-w-sklepie-PODGLAD-150s-EXPERIMENT.mp4`, SHA-256
`271db549c2c3c843c094099ae3d1bccb6ca383af7604f4aeb8fab5fe25723cb8`. It contains 3,600 frames at 24 fps,
was re-encoded at CRF 19, fully decodes, and its decoded audio PTS are continuous from zero. Sampled maximum RGB
differences from the three source clips were 0.79 / 0.79 / 0.88 on a 255 scale; median audio correlations were
1.00000 / 0.99997 / 0.99997 at zero measured shifts. These samples do not establish acting, lip sync or
perceived continuity. The preview is not the requested 300 s movie.

As an exploratory negative control on block 3, each of its six dialogue intervals was also evaluated against
the *other* position-tracked face using the same audio, time interval and SyncNet weights. The intended-position
crop had a higher model confidence in all six cases; the artificial 200 ms delay control responded as expected
for six intended crops and no other-position crops. This is consistent with the model distinguishing the two
positions in these samples, but the crops are not identity recognition, confidence is not a probability, and
short synthetic speech or pose can bias either score. It does not establish lip sync, acting or listening quality.
The per-line results are saved in the local film's `analiza/fragment-03-syncnet-porownanie.json`.

The waveform screen is insensitive to pure attenuation. In block 3, a separate first-channel comparison found
the six output/source speech-interval RMS differences +0.28 / -4.91 / -7.76 / -6.96 / -7.97 / **-17.98 dB**;
the last line's RMS was 0.0143 versus 0.1135 in the driving audio. The separate source-voice copy measured within
0.01 dB of the source on those intervals. These are RMS comparisons, not listening or lip-sync verdicts, and do
not establish the cause. An optional numeric speech-level screen is documented in `docs/SPEECH_LEVEL_SCREEN.md`;
it has no default quality threshold and does not alter existing film gates.

Block 4 completed at 22:56 (60 s, SHA-256
`58e1a464e9880f333f8ffe4af0e2595479b8d4ca176d4752d5b30443779f0a61`); block 5 started at 22:57.
Its dialogue passed the independent waveform and frame-PTS screen, Whisper WER was 0.066, and the sampled join
difference was 3.5/255 RGB. The first-channel output/source interval levels were +0.28 / -5.11 / -4.35 / -0.54 /
+0.01 / -11.55 dB. The original and a separate source-voice copy were collected. These measurements and sampled
frames do not establish playback quality, actor likeness or lip synchronization.

The final dialogue line in each of blocks 2–4 was also compared in consecutive 0.5 s slices at 8 kHz mono, with a
±120 ms waveform search and decoded audio PTS checked first. The strongest correlations occurred within 0.125 ms of
zero offset in every eligible slice (source RMS at least 0.005): the fading is not explained by a large time shift.
For block 2 the fitted gain starts at -3.1 dB and is -16.9 to -22.0 dB in later slices; block 3 starts at -9.0 dB and
then ranges -24.6 to -28.9 dB; block 4 starts at -3.7 dB and later ranges -18.5 to -21.3 dB. Per-slice reports,
input hashes and the reproducer are saved with the local film analysis. This is a waveform measurement, not an
audible-quality or causal verdict. The still-running head was started at 12:09, before the source-audio look-lock
bypass (`c81f1b6` at 19:38); its journal confirms that the audio look-lock ran in the later windows. The bypass is a
candidate fix, but has not yet been compared on the model, so the cause and its effect remain unproven.

The first four blocks were also published as a separate **210 s preview** in the local film directory
(`lalka-list-w-sklepie-PODGLAD-210s-EXPERIMENT.mp4`, SHA-256
`6404c30848fec8474b62b356b96fe1097acf528c0b4ec5c0ef46d11b07b45249`). It is not the requested 300 s result.
The preview was re-encoded at CRF 19, fully decodes, and has 5,040 video frames with continuous 24 fps video PTS
and continuous decoded audio PTS. Sampled RGB differences from its four sources were at most 0.885/255; median
audio correlations were at least 0.99997 at zero sampled shift. These checks do not establish acting or quality.
Both 210 s previews also yielded exactly 6,720,000 decoded audio samples at 32 kHz from first PTS 0 to 210 s;
this was checked independently of the container's declared duration.
The five original collected blocks were checked separately before final assembly: blocks 2–5 each decoded
60.000 s of audio, and block 1 decoded 29.984 s (16 ms short of its 30 s video interval). All five fall within
the chosen 50 ms endpoint tolerance. This matters because assembly can pad a short audio track with silence.
In a visual contact sheet sampled at 0, 30, 60, 90, 120, 150 and 180 s, the same two foreground characters,
shop and letter remain recognizable; Wokulski still holds the letter in each sampled frame. This supports visual
continuity at those instants but also shows a very static staging; it does not establish that the action, acting,
actor likeness, or lip synchronization is convincing between samples. The staged-action experiment remains pending.
The full 300 s dialogue plan contains 30 lines and approximately 278 words; the scheduled line intervals cover
120.38 s, with a median positive gap of 6.76 s between lines. These are planned intervals, not measured
silence or a judgment of natural delivery. The common instruction also asks for unchanged physical positions,
which may contribute to the static staging; that cause needs a controlled comparison.

A second 210 s preview uses the synthetic driving voices in place of the entire model soundtrack (therefore it
does not include the generated ambience). Its SHA-256 is
`4a555a763f0bc8be30ffd649a921e293c371933b9a26002331474afe8ef126f7`; its encoded video packets match the
original 210 s preview exactly and it fully decodes with continuous video/audio PTS. After extracting its block
3 and 4 audio, the final lines measured -0.009 / -0.009 dB RMS against their respective sources, with waveform
correlation 0.99999 and zero measured shift. The original preview remains intact. Neither copy has a listening,
lip-sync or actor-likeness verdict; the alternate is an explicitly labeled comparison, not a replacement.

Block 5 completed at 23:50 (60 s, SHA-256
`0d278d2716786e638904ed344701725f2b172cd63799132b5ff8975261454ba1`), and the last 30 s block started at 23:51
from its verified final frame. The fifth block passed the independent waveform and decoded-frame PTS screen;
its first-channel output/source interval levels were +0.24 / -1.41 / -2.40 / -3.33 / -2.47 / **-12.52 dB**.
Whisper WER was 0.018, and the sampled 4→5 boundary differed by 3.6/255 RGB. Sampled frames retain the shop and
main pair, but show limited action; neither these samples nor ASR establish acting, natural speech or lip sync.

## The complete 300 s experiment (2026-11-02)

The sixth 30 s block completed at 00:18 from block 5's verified final frame. Its SHA-256 is
`559e3768d8424bd678f375f244a368d99abd8fb082db380293b08ac3fecefee2`. Whisper recognized the final Polish
line, with WER 0.037 for the block; this is not a listening judgment. Its three first-channel output/source
interval levels were +0.37 / -3.75 / **-7.23 dB**. The 5→6 sampled frame boundary was 3.925/255 RGB, with the
same shop and foreground pair visible on both sides. Decoded audio in this original block ended 16 ms before its
30 s video interval, within the explicitly selected 50 ms tolerance.

The assembled original is `lalka-list-w-sklepie-300s-EXPERIMENT.mp4`, SHA-256
`2dea192688247f38b13d41ef4ea5b0afddca4bd6a3fee0093a60779a6efee1a8`. It has six linked blocks with exact
durations 30/60/60/60/60/30 s. All 7,200 video-frame PTS are exactly 24 fps from 0 to 300 s; audio decoded
to 9,600,000 samples at 32 kHz from 0 to 300 s. The original blocks were checked separately because assembly
may pad audio. Three sampled frames per block differ from their source by at most 1.021/255 RGB, and the lowest
median sampled audio correlation is 0.99995 at zero measured shift. The five sampled block boundaries range
from 3.12 to 3.93/255 RGB. None of those checks replaces full playback or proves continuity of action.

Two separate, explicitly experimental 300 s copies use the same encoded video packets and timestamps as the
original. The source-voice copy SHA-256 is `40caa3df484890396a3da8e3ba544853d742c59bfde72269cc3d76b74ee938e0`;
it replaces the whole model soundtrack with the synthetic driving track and therefore omits model-generated
ambience. The balanced-voice copy SHA-256 is `c7a3698d573ce9f546fbc6ed30a6a7a627f160ad9a4350779830831a3ae8d08f`;
its source is the experimentally gain-adjusted model audio. Both have 9,600,000 decoded audio samples. The
30 planned speech intervals in each alternative pass their corresponding waveform/timing comparison, but
perceived volume, artifacts and lip sync still require listening and viewing.

In ten still frames sampled every 30 s from the completed film, the main pair and shop remain recognizable and
the man continues to hold the letter. This is substantially more continuous than disconnected scenes, but the
staging is very static; the sampled frames cannot verify action between them, performer likeness or acting.
The measured speech fading and lack of action remain open problems; no full-playback quality verdict is claimed.
