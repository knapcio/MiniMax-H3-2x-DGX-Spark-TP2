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
