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
exact 10 s boundary. The planned five-minute film consists of 30 independently rendered 10 s shots; production was
still in progress at this checkpoint. These sparse waveform measurements do not prove that the mouth is accurately
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
