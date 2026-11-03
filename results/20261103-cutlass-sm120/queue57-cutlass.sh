#!/bin/bash
# queue57-cutlass (2026-11-03): serving proof of the opt-in H3_TP2_FP8_GEMM=cutlass level (CuTe-DSL sm120 FP8 GEMM for the
# DiT qkv / out / fc1), same-boot-style A/B against today's defaults; serving ends on the defaults.
#   A: defaults (new code), gated boot, 5 s 1344x768 T2VA calibration (seed 0, 20 steps): digest 4d6d69e4, s/step
#   B: + H3_TP2_FP8_GEMM=cutlass, gated boot, the same calibration twice: digest unchanged, s/step from the second
#   C: defaults again (gated boot)
# Stops early (straight to C) if a boot does not come up healthy or the digest differs.
. ~/h3-tp2/scripts/queue-lib.sh
DEFAULT_ENV="export H3_TP2_SAVE_LATENTS=/home/knapcio/h3-tp2-results/latents
export H3_TP2_WINDOW_SEED_STEP=1"
CUT_ENV="$DEFAULT_ENV
export H3_TP2_FP8_GEMM=cutlass"
WANT=4d6d69e4aad0
unset CONT_PROMPTS CONT_PROMPTS_EXTRA AUDIO_URL EXTRA_JSON; export TASK=t2va
export PROMPT="A cinematic shot of a forest stream at golden hour, leaves swaying in a light breeze, sunlight flickering through the canopy, gentle water sounds."
calib() {   # calib <name>: one calibration, prints "<name>: <s/it> <digest>" and sets DIG / RATE
  local T; T=$(date "+%Y-%m-%d %H:%M:%S"); run "$1" 5 "5:1344:768:20:0"
  RATE=$(journalctl --user -u h3-tp2-head.service -a --since "$T" --no-pager -o short | tr "\r" "\n" | grep -o -E "20/20 \[[0-9:]+<00:00, +[0-9.]+s/it\]" | tail -1)
  DIG=$(journalctl --user -u h3-tp2-head.service --since "$T" --no-pager | grep -o -E "latents sha256 [0-9a-f]{12}" | tail -1 | awk '{print $3}')
  echo "CALIB $1: $RATE latents $DIG $(journalctl --user -u h3-tp2-head.service --since "$T" --no-pager | grep -o -E "phase (diffuse|decode)[_a-z0-9]* [0-9.]+ s" | tr "\n" " ")" | tee -a gen.log
}
levels() {  # the FP8 lines of the running boot
  journalctl --user -u h3-tp2-head.service --since "-25min" --no-pager | grep -o -E "FP8 GEMM (cutlass|two-pass)[^\"]{0,170}|consumer fuse requested=[^\"]{0,60}|computes qkv, out in bf16[^\"]{0,60}|first DSL GEMM[^\"]{0,60}|FP8 GEMM requested=[^\"]{0,40}|cutlass level unavailable[^\"]{0,200}" | sort | uniq -c | head -12
}
# the pair is busy with the lalka60 render (queue56-lalka-a.sh) until it appends LALKA-A-DONE: no restart before that
for i in $(seq 1 600); do grep -q "LALKA-A-DONE" gen.log 2>/dev/null && break; sleep 30; done
grep -q "LALKA-A-DONE" gen.log || { echo "Q57 gave up waiting for LALKA-A-DONE $(date +%H:%M:%S)"; exit 1; }
sleep 10
echo "Q57-CUTLASS-START $(date +%H:%M:%S)" | tee -a gen.log
rm -f ~/h3-tp2-results/.head-rev            # the synced tree is new code at the same git rev: force the first restart
restart_head_gated "$DEFAULT_ENV"; rcA=$?
echo "Q57-BOOT-A rc $rcA $(date +%H:%M:%S)" | tee -a gen.log; levels
OK=1
if [ "$(health)" = 200 ]; then
  calib q57-A-default; DA=$DIG
  [ "$DA" = "$WANT" ] || { echo "Q57 STOP: default digest $DA != $WANT" | tee -a gen.log; OK=0; }
else echo "Q57 STOP: head not healthy after boot A" | tee -a gen.log; OK=0; fi
if [ $OK = 1 ]; then
  restart_head_gated "$CUT_ENV"; rcB=$?
  echo "Q57-BOOT-B rc $rcB $(date +%H:%M:%S)" | tee -a gen.log; levels
  if [ "$(health)" = 200 ]; then
    calib q57-B-cutlass-1; D1=$DIG
    if [ "$D1" = "$WANT" ]; then calib q57-B-cutlass-2; D2=$DIG
      [ "$D2" = "$WANT" ] || echo "Q57 STOP: cutlass digest (second) $D2 != $WANT" | tee -a gen.log
    else echo "Q57 STOP: cutlass digest $D1 != $WANT" | tee -a gen.log; fi
    journalctl --user -u h3-tp2-head.service --since "-30min" --no-pager | grep -o -E "dispatch so far[^\"]{0,160}" | tail -2
  else echo "Q57 STOP: head not healthy after boot B" | tee -a gen.log; fi
fi
restart_head_gated "$DEFAULT_ENV"; rcC=$?
echo "Q57-BOOT-C (defaults restored) rc $rcC health $(health) $(date +%H:%M:%S)" | tee -a gen.log; levels
echo "Q57-DONE $(date +%H:%M:%S)" | tee -a gen.log
echo "CUTLASS-PROOF-DONE $(date +%H:%M:%S)" >> gen.log   # the next render queue waits for this marker
