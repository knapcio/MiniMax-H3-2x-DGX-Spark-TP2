# The slow render state: proactive compaction of the unified memory (2026-11-11)

**2026-11-11 03:50-07:35, a false alarm on the way:** about 90 minutes after the setting was persisted, both Sparks
stopped answering on the LAN at the same minute and the setting was suspected and withdrawn. The previous boot's
journal, read after the owner's power cycle at 07:28, shows both hosts losing their Wi-Fi link to the same access
point at 03:45:46-48 (`wlP9s9: Connection to AP ... lost`, then `ssid-not-found`, no lease) and never reconnecting,
while the hosts kept rendering all night: queue126 finished at 05:41 with every window at 547-551 s, the fastest
chains recorded on the pair, and queue139 rendered its six shots after it. The `NVRM ... NV_ERR_NO_MEMORY` lines in
that journal appear every 20-60 minutes all day, before and after the setting, and are not a crash signature. The
setting is exonerated and re-applied (see the end of this file); the lesson about the Sparks' Wi-Fi LAN is in the
workspace notes.

Since 2026-11-08 some renders on the two-Spark TP2 stack ran about 25 % slower per step than others of the same size,
whole chains or the rest of a process from some window on, at normal GPU power and clock, with nothing in the request
to explain it (`h3_tp2/pace.py` was written to log it; `results/20261106-voice/` section 15). This directory holds the
catch and the fix.

## The catch (queue127, 2026-11-11 00:15-02:16, spark-06)

queue127 renders two 64.6 s chains with the same environment, and the queue library's restart shortcut kept the second
chain in the first chain's head process. The host monitor (`hostmon.sh`, one line a minute: CPU ticks of kcompactd0,
kswapd0, the UVM thread, GPU power and clock, memory counters) and the catcher (`slowcatch.sh`, which fires on the pace
warning) were running.

| phase | time | kcompactd0 ticks per minute | window pace |
|---|---|---|---|
| chain 1, windows 1-6 | 00:36-01:32 | 0-425 (mostly 0 until 01:09, then 170-425) | 593, 587, 590, 600, 610, 620 s |
| scoring of chain 1 on the CPU (whisper, UTMOS, faces), then the next request | 01:33-01:39 | **1350-1846** | (no render) |
| chain 2, window 1 | 01:46-01:57 | **1063-2135** | **710 s, 21 % slower** (pace warning, `slowcatch-1111-0156.log`) |
| `sysctl -w vm.compaction_proactiveness=0` | 01:58:49 | | |
| chain 2, window 2 (started 01:57, the first 1.5 min still under the storm) | 01:57-02:08 | 1372, then **0** from 01:59 on | 612 s |
| chain 2, windows 3-6 | 02:08-02:47 | 0 | **578, 579, 577, 572 s**, all faster than any window of chain 1 |
| queue126, two 64.6 s chains under the setting (03:03-05:41, same head process) | | 0 | **547-551 s every window**, the fastest chains recorded on the pair (reference 555-584) |

GPU power stayed at 82-86 W and the clock at 2138-2184 MHz throughout; `pgmigrate_success` moved by 826 000 pages
in the six scoring minutes and by only 8 000 during the slow window, `compact_daemon_wake` did not move: kcompactd
was scanning (proactive compaction, `vm.compaction_proactiveness=20`), not migrating much, and that scanning alone cost
the render a fifth of its speed. With proactive compaction off the thread went silent within the minute and every later
window was faster than any window of chain 1 (572-579 s against 587-620).

Earlier observations fit: the first slow chains (queue119 chain 2, queue122 chain 3, queue123 chain 2) all followed a
scoring phase in the same queue; the 10-30 slow render had the UVM thread busy; and a CPU scoring job run deliberately
on the rendering head on 2026-11-09 halved the GPU power (that is a second, different mechanism: CPU load takes the
shared power budget; `results/20261109-long-chain-identity/` section 3).

## The fix

`vm.compaction_proactiveness = 0` on both Sparks (`/etc/sysctl.d/99-h3-compaction.conf`, applied 2026-11-11 02:16).
On-demand compaction (for high-order allocations) still runs; what stops is the background scanning that the kernel
starts when fragmentation crosses its threshold, which on a 128 GB unified-memory host with a 60 GB GPU pool is most of
the time after any memory churn. The explicit compaction before every boot (`/usr/local/sbin/h3-compact`, with the head
stopped) stays: a fragmented host at boot time was a separate, measured cost (`results/20261102-speed-plan/`).

Watch next: the pace lines of the following renders (queue126, queue139) should show no window more than a few percent
off the fastest of its size, and `hostmon` should show kcompactd0 near zero during sampling. If a slow window appears
with kcompactd silent, the cause is something else and this entry gets amended.

Files: `slowcatch-1111-0156.log`, `hostmon-spark-06.log` (2026-11-11 00:00-02:20), the head journal's `h3_tp2 pace`
lines, on Spark_06.
