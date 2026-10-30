# Exact-duration copies without changing retained frames (2026-10-30)

MiniMax H3 rounds requested frame counts up to its `17n+5` grid. For example, the 30 s painting sample has 736 frames
at 24 fps (30.6667 s of video), and the 16 s sample has 396 frames (16.5 s). These are valid native model outputs,
but they are not exactly the requested lengths.

`python3 scripts/exact_duration.py input.mp4 output.mp4 30` creates a **new** copy of a 24 fps H3 MP4, trimmed to the
nearest whole video frame. It losslessly re-encodes the retained YUV frames and copies only complete AAC packets whose
end does not exceed the video end; this avoids re-encoding audio, which can finish up to one AAC packet early. It
compares every retained decoded video frame, audio packet hashes and timestamps, stream properties and container
duration before atomically publishing the output. The original is never overwritten. Non-24 fps, unsupported pixel
formats or non-square pixels, non-AAC audio, nonzero stream starts and missing frame/packet information are rejected. Seconds that are not
exact multiples of 1/24 are rounded to the nearest frame; the result cannot represent arbitrary sub-frame durations.

The following separate files were created under `~/Downloads/h3-prod-smoke/` and verified:

| Input | Copy | Video | Copied audio ends before video by |
|---|---|---|---|
| 10 s FL2VA smoke | `d10_1344x768_s20_seed1101_fl2va_EXACT10s.mp4` | 240 frames, 10.000 s | 16 ms |
| 15 s FL2VA smoke (earlier source image, not Koninck) | `d15_1344x768_s20_seed1101_fl2va_EXACT15s.mp4` | 360 frames, 15.000 s | 24 ms |
| 16 s Koninck | `d16_koninck_painting_EXACT16s.mp4` | 384 frames, 16.000 s | 0 ms |
| 30 s Koninck | `d30_koninck_painting_EXACT30s.mp4` | 720 frames, 30.000 s | 16 ms |
| owner-reviewed regional deflicker experiment | `d30_koninck_history5s_regional_EXACT30s_EXPERIMENT.mp4` | 720 frames, 30.000 s | 16 ms |
| 60 s Koninck, original | `d60_koninck_painting_EXACT60s.mp4` | 1440 frames, 60.000 s | 0 ms |
| 60 s Koninck, regional deflicker experiment | `d60_koninck_painting_regional_EXACT60s_EXPERIMENT.mp4` | 1440 frames, 60.000 s | 0 ms |
| 60 s Koninck, broader regional experiment | `d60_koninck_allregional_EXACT60s_EXPERIMENT.mp4` | 1440 frames, 60.000 s | 0 ms |

In each copy, every retained decoded frame and every retained AAC packet including its timestamp matches the source;
only the end is removed. This does not make model motion or flicker better. The audio may already have been shorter
in another input; the tool does not synthesize or pad missing sound. FFmpeg and ffprobe with libx264 are required.
Five local tests cover frame rounding, audio and silent inputs, output protection and failure before publication.
