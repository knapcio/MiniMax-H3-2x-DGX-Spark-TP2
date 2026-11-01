# Comparing speech levels with the driving track

`scripts/screen_speech_levels.py` measures the first-channel RMS of each timed
source/output interval at the same sample rate. It checks decoded audio PTS
continuity and maps both inputs to their presentation origins. It does not
modify audio or video. For example:

```sh
python scripts/screen_speech_levels.py source.mp3 render.mp4 \
  --intervals dialogue.json --report speech-levels.json
```

The dialogue input is a list of objects with `start`, `end` and optional
`speaker`, or an object containing that list as `dialogue`. The default status
is `MEASURED`: there is no calibrated level threshold. For an explicit,
experiment-specific check, `--min-relative-db=-12` marks an interval as failing
when output RMS is more than 12 dB below its source; `--max-relative-db` can
also bound amplification. These are caller-selected numerical thresholds, not
universal speech quality grades. A source interval with negligible energy is
`INCOMPLETE`; a silent output cannot pass a selected minimum. Exit codes are
0 for measured/pass, 1 for a selected threshold failure, and 2 for incomplete
or invalid input. Existing reports are never overwritten.

The tool takes private, hash-bound snapshots before probing or decoding (up to
8 GiB per input), so changing an original during analysis cannot silently
change the data being measured. It checks original and snapshot hashes again
afterward. Audio decoding is capped at 20 million samples per input; lower
`--sample-rate` or split intervals for longer material. Full PTS probing is
also bounded: ffprobe output is limited on disk and reports over 16 MiB are
rejected before JSON is loaded. It does not establish waveform identity, visual
timing, lip sync, perceptual loudness or naturalness. Channel layout, silence,
mixing and speech movements within an interval influence the ratios. Use the
existing waveform/PTS screen and full playback separately.

## Example from the in-progress Lalka film

For its third 60 s block, first-channel output/source RMS differences across
the six scheduled lines were approximately +0.28, -4.91, -7.76, -6.96, -7.97
and **-17.98 dB**. The waveform/PTS screen passed, illustrating why correlation
alone does not measure level. A separate source-voice copy measured within
0.01 dB of the source on these intervals; neither version has been evaluated
by listening, and the copy has not replaced the original. `-12 dB` would flag
the last line in this example, but is not asserted as a general threshold.

These results use existing clips and CPU decoding only. They do not prove the
cause of the attenuation or the effect of any model change.
