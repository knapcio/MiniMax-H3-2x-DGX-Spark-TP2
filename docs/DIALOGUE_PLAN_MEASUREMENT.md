# Measuring a dialogue plan before rendering

`scripts/measure_dialogue_plan.py` summarizes the scheduled intervals from a
JSON list with `start`, `end`, `text`, and an optional `speaker`, the same
shape used by the dialogue-window planner. For example:

```sh
python3 scripts/measure_dialogue_plan.py --seconds 300 \
  --dialogue dialogue.json --report dialogue-pacing.json
```

The report contains approximate Unicode word counts, the sum of planned
utterance durations, all positive intervals without a planned line, and the
median of positive between-line gaps. It rejects overlapping or out-of-range
lines rather than silently combining them. Its input is limited to 16 MiB and
the output must be a new file. For several requests, place each request's
dialogue on one global timeline before measuring the full film.

The window planner (`scripts/plan_dialogue_windows.py`) adds a per-window
count: the words each window must speak in its *new* content and that density
in words per second. It warns on stderr and in the report when a window is
above `--max-words-per-second` (default 2.75, `0` disables). The default sits
between the two observations of the 2026-11-05 Polish pacing arm
(`results/20261105-polish-pacing/`): three Polish lines per 10.6-11.5 s window
were all spoken at 2.26-2.64 words/s of new content, while the 6 s tail
window, three short lines at 2.85 words/s, dropped half a line. One arm, one
language: treat the number as a first estimate, not a measured limit. The last window of a clip that is not a whole number of
windows is the usual offender; give it fewer lines or round the duration.

This is a description of a plan, not a quality score. A line's time slot may
include pauses; an unscheduled interval may contain music or other sound.
The model might not follow either the text or its timing, and word segmentation
is approximate. Assess the generated soundtrack, pacing, acting and lip sync
independently.
