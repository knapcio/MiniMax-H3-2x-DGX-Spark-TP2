#!/bin/bash
# Media acceptance (after joeynyc/MiniMax-H3-2x-DGX-Spark scripts/verify-output.sh, Apache-2.0).
# Usage: verify-output.sh <file.mp4> [expected_duration_s] [expected_w] [expected_h]
# Checks: one video + one audio stream, full decode, duration/size within tolerance, audio present and not silent,
# and prints a framemd5 digest of the decoded frames (MP4 bytes differ between identical renders; decoded frames do not).
# Exit 1 on any failure.
set -uo pipefail
f=$1; edur=${2:-}; ew=${3:-}; eh=${4:-}
rc=0
probe=$(ffprobe -v error -show_entries stream=codec_type,codec_name,width,height,r_frame_rate,nb_frames:format=duration -of compact=p=0 "$f" 2>&1) || { echo "probe: FAIL"; echo "$probe"; exit 1; }
echo "$probe" | tr '\n' ' '; echo
nv=$(grep -c 'codec_type=video' <<< "$probe"); na=$(grep -c 'codec_type=audio' <<< "$probe")
[ "$nv" = 1 ] && [ "$na" = 1 ] || { echo "streams: FAIL (video=$nv audio=$na)"; rc=1; }
if [ -n "$ew" ]; then
  gw=$(grep -o 'width=[0-9]*' <<< "$probe" | head -1 | cut -d= -f2); gh=$(grep -o 'height=[0-9]*' <<< "$probe" | head -1 | cut -d= -f2)
  [ "$gw" = "$ew" ] && [ "$gh" = "$eh" ] || { echo "size: FAIL (${gw}x${gh}, want ${ew}x${eh})"; rc=1; }
fi
if [ -n "$edur" ]; then
  d=$(grep -o 'duration=[0-9.]*' <<< "$probe" | head -1 | cut -d= -f2)
  awk -v d="$d" -v e="$edur" 'BEGIN{ if (d+0 < e-0.6 || d+0 > e+0.6) exit 1 }' || { echo "duration: FAIL ($d s, want $edur s)"; rc=1; }
fi
ffmpeg -v error -xerror -i "$f" -f null - && echo "decode: OK" || { echo "decode: FAIL"; rc=1; }
vol=$(ffmpeg -hide_banner -i "$f" -map 0:a:0 -af volumedetect -f null - 2>&1 | grep -E 'mean_volume|max_volume' | sed 's/.*\] //' | tr '\n' ' ')
echo "$vol"
mean=$(grep -o 'mean_volume: [-0-9.]*' <<< "$vol" | awk '{print $2}')
if [ -z "$mean" ]; then echo "audio: FAIL (no volume stats)"; rc=1
elif awk -v m="$mean" 'BEGIN{exit !(m+0 < -70)}'; then echo "audio: FAIL (silent, mean $mean dB)"; rc=1
else echo "audio: OK (mean $mean dB)"; fi
md=$(ffmpeg -v error -xerror -i "$f" -map 0:v:0 -f framemd5 - | grep -v '^#' | md5sum | awk '{print $1}') || { echo "framemd5: FAIL"; rc=1; }
echo "framemd5: $md"
[ $rc = 0 ] && echo "verify: OK" || echo "verify: FAIL"
exit $rc
