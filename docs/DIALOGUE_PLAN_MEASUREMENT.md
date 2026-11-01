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

This is a description of a plan, not a quality score. A line's time slot may
include pauses; an unscheduled interval may contain music or other sound.
The model might not follow either the text or its timing, and word segmentation
is approximate. Assess the generated soundtrack, pacing, acting and lip sync
independently.
