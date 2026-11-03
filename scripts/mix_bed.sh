#!/bin/bash
# mix_bed.sh <dialogue.mp4> <bed.(mp4|wav)> <out.mp4> [bed_lufs=-38] : normalises the bed to an integrated loudness
# (the H3 beds come out anywhere between -25 and -45 LUFS, so a fixed dB offset is meaningless), loops it under the
# dialogue and ducks it under speech (sidechain, ratio 4), video stream copied unchanged. Dialogue clips sit at ~-24 LUFS.
# loudnorm only accepts TP >= -9, so the bed is normalised to -23 LUFS and then lowered by (-23 - target) dB.
d=$1; b=$2; o=$3; I=${4:--38}
off=$(( I + 23 ))
bed=$(mktemp --suffix=.wav)
ffmpeg -v error -y -i "$b" -vn -af "loudnorm=I=-23:TP=-3:LRA=18,volume=${off}dB,aresample=48000" -ar 48000 "$bed"
ffmpeg -v error -y -i "$d" -stream_loop -1 -i "$bed" -filter_complex \
 "[1:a]afade=t=in:d=1.5[bed];[0:a]aresample=48000,asplit[v1][v2];[bed][v2]sidechaincompress=threshold=0.02:ratio=4:attack=40:release=600[duck];[v1][duck]amix=inputs=2:duration=first:normalize=0[a]" \
 -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 160k -shortest "$o"
rm -f "$bed"
