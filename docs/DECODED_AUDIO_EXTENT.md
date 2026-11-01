# Checking decoded audio coverage

Container duration and continuous audio timestamps alone do not prove that a
render contains audio through its final video frame. `check_decoded_audio_extent.py`
uses decoded audio frames and sample counts to measure both endpoints.

First verify the video's presentation interval independently. Supply that
interval explicitly; this tool does not determine it from container metadata.
For a video verified to span 0–210 seconds, for example:

```sh
python3 scripts/check_decoded_audio_extent.py render.mp4 \
  --expected-start 0 --expected-end 210 --max-missing 0.05 \
  --report audio-extent.json
```

Without `--max-missing`, the result is `MEASURED` rather than a pass/fail.
With it, `PASS` means that neither endpoint is missing more than the
caller-selected tolerance; `FAIL` exits 1. Invalid or discontinuous decoded
timestamps and operational errors exit 2. The report is written only to a
new path. Extra decoder priming or padding is reported in the endpoints and
does not count as missing audio.

The tool snapshots and hashes the input, verifies the original and snapshot,
and bounds its ffprobe frame report. It checks coverage and timestamp
continuity, not intelligibility, relative speech level, speaker identity,
lip synchronization, or the perceived quality of the soundtrack. For timed
speech-level checks see [SPEECH_LEVEL_SCREEN.md](SPEECH_LEVEL_SCREEN.md).
