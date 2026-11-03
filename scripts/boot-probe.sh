#!/bin/bash
# Boot speed gate (speed plan 2026-11-02, item 1). Boots of the same code on the same hosts differ by 7-40 % in
# denoising speed (2026-10-30: 120 vs 86 s/step; 2026-11-02: 22.9 vs 21.2 s/step at 1344, identical latents digests).
# Run this right after the head reports healthy and before any long render:
#   bash scripts/boot-probe.sh [reference_s_per_step] [tolerance_fraction]
# It renders a 5 s 960x576 T2VA calibration (seed 0, 20 steps; ~8.6 s/step on a good boot of the long stack with the
# two-pass GEMM and fused SwiGLU), reads the step rate from the head journal and prints PROBE PASS|SLOW|FAIL with the
# numbers. Exit 0 = pass, 2 = slow (restart the head and probe again), 1 = the probe did not run.
REF=${1:-8.65}; TOL=${2:-0.08}
# Two probes: the FIRST request of a boot is special (4-7 % slower and numerically different: its latents digest
# changes per boot, later requests repeat bf6e8568 across boots at 960x576), and it does not predict the steady state
# (queue52/53, 2026-11-02: a boot probed 9.19 first and ran 10.5-10.7 afterwards; the next boot 9.35 then 8.96/8.99).
# The verdict is on the SECOND (warm) request against the warm reference; the first is reported.
PROBES=${PROBES:-2}
# The probe sets its own PROMPT / task / size: a caller's request variables (queue60b exported H3_PROMPT_FILE before its
# gated boot, 2026-11-03 19:57, and render.sh refused "choose H3_PROMPT_FILE or PROMPT, not both") must not leak in.
unset H3_PROMPT_FILE H3_EXTRA_JSON_FILE PROMPT CONT_PROMPTS CONT_PROMPTS_EXTRA AUDIO_URL EXTRA_JSON
OUTROOT=${H3_PROBE_ROOT:-$HOME/h3-tp2-results}
NAME=boot-probe-$(date +%Y%m%d-%H%M%S)
health() { curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:18093/health; }
[ "$(health)" = 200 ] || { echo "PROBE FAIL: head health $(health)"; exit 1; }
T0=$(date "+%Y-%m-%d %H:%M:%S")
grep Normal /proc/buddyinfo | awk '{printf "buddyinfo order9+ (head): "; for(i=14;i<=NF;i++) printf "%s ", $i; print ""}'
unset CONT_PROMPTS CONT_PROMPTS_EXTRA AUDIO_URL EXTRA_JSON; export TASK=t2va
export PROMPT="A cinematic shot of a forest stream at golden hour, leaves swaying in a light breeze, sunlight flickering through the canopy, gentle water sounds."
# The Ref2VA partition rejects T2VA ("checkpoint partition 'ref2va' supports ['ref2va']", 2026-11-03), so on that
# partition the probe is a 5 s ref2va request with a fixed reference image; its rate reference (PROBE_REF_R2V) is
# separate and, when unset, the probe only reports (PASS with the numbers) instead of judging.
IMG=""
if grep -q "H3_MODEL=.*Ref2VA" ~/.h3-tp2.env 2>/dev/null; then
  export TASK=ref2va; IMG=${H3_PROBE_IMAGE:-$HOME/h3-tp2-results/koninck.png}
  [ -f "$IMG" ] || { echo "PROBE FAIL: Ref2VA partition but no probe image at $IMG"; exit 1; }
  if [ -n "${PROBE_REF_R2V:-}" ]; then REF=$PROBE_REF_R2V; else REF=0; fi
fi
FIRST=""
for k in $(seq 1 $PROBES); do
  Tk=$(date "+%Y-%m-%d %H:%M:%S")
  OUT=$OUTROOT/$NAME-$k bash ~/h3-tp2/scripts/bench-matrix.sh "5:960:576:20:0" ${IMG:+"$IMG"} > $OUTROOT/$NAME-$k.log 2>&1 < /dev/null
  RATE=$(journalctl --user -u h3-tp2-head.service -a --since "$Tk" --no-pager -o short | tr "\r" "\n" | grep -o -E "20/20 \[[0-9:]+<00:00, +[0-9.]+s/it\]" | tail -1 | grep -o -E "[0-9.]+s/it" | tr -d "s/it")
  DIG=$(journalctl --user -u h3-tp2-head.service --since "$Tk" --no-pager | grep -o "h3_tp2 latents sha256 [0-9a-f]\{12\}" | tail -1 | awk '{print $4}')
  [ $k -lt $PROBES ] && { FIRST="first $RATE ($DIG), "; echo "probe $k: $RATE s/step latents $DIG"; }
done
[ -n "$RATE" ] || { echo "PROBE FAIL: no step rate in the journal (see $OUTROOT/$NAME.log)"; exit 1; }
[ "$REF" = 0 ] && { echo "PROBE PASS (report only, $TASK, no reference rate): ${FIRST}warm $RATE s/step latents $DIG $(date +%H:%M:%S)"; exit 0; }
LIMIT=$(python3 -c "print(round($REF*(1+$TOL),3))")
if python3 -c "import sys; sys.exit(0 if $RATE <= $LIMIT else 2)"; then
  echo "PROBE PASS: ${FIRST}warm $RATE s/step (reference $REF, limit $LIMIT) latents $DIG $(date +%H:%M:%S)"; exit 0
else
  echo "PROBE SLOW: ${FIRST}warm $RATE s/step (reference $REF, limit $LIMIT) latents $DIG $(date +%H:%M:%S)"; exit 2
fi
