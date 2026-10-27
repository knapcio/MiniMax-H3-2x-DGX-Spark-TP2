#!/bin/bash
# Same-seed quality comparison of two renders (e.g. one-Spark vs two-Spark): decoded-frame digests, per-frame
# SSIM/PSNR of the video streams, and audio peak/mean levels. Usage: compare-outputs.sh <a.mp4> <b.mp4>
# "framemd5 identical" means the decoded frames match bit for bit; otherwise SSIM/PSNR say how far apart they are.
set -uo pipefail
a=$1; b=$2
md() { ffmpeg -v error -i "$1" -map 0:v:0 -f framemd5 - | grep -v '^#' | md5sum | awk '{print $1}'; }
ma=$(md "$a"); mb=$(md "$b")
echo "framemd5 a: $ma"; echo "framemd5 b: $mb"
[ "$ma" = "$mb" ] && echo "framemd5: identical" || echo "framemd5: differ"
ssim=$(ffmpeg -v info -nostats -i "$a" -i "$b" -lavfi "[0:v][1:v]ssim" -f null - 2>&1 | grep -o 'All:[0-9.]*' | head -1)
psnr=$(ffmpeg -v info -nostats -i "$a" -i "$b" -lavfi "[0:v][1:v]psnr" -f null - 2>&1 | grep -o 'average:[0-9.inf]*' | head -1)
echo "video SSIM ${ssim:-?}  PSNR ${psnr:-?}"
for f in "$a" "$b"; do
  v=$(ffmpeg -hide_banner -i "$f" -map 0:a:0 -af volumedetect -f null - 2>&1 | grep -E 'mean_volume|max_volume' | sed 's/.*\] //' | tr '\n' ' ')
  echo "audio $(basename "$f"): $v"
done
