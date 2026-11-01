# Planning connected story beats over H3 continuation windows

`scripts/plan_story_windows.py` prepares an explicit prompt for each window of
**one** continuation request. Put stable character, place and style information
in a UTF-8 file; put ordered, nonoverlapping timed actions in a JSON array:

```json
[
  {"start": 0, "end": 60, "action": "Wokulski and Izabela examine the letter in the shop."},
  {"start": 60, "end": 180, "action": "They discuss what happened and decide to invite its writer."},
  {"start": 180, "end": 300, "action": "They make the arrangements and leave together."}
]
```

For example:

```sh
python scripts/plan_story_windows.py --seconds 300 \
  --continuity-file continuity.txt --beats beats.json \
  --window 277 --overlap 22 --report story-plan.json --extra-output story-extra.json
```

Each prompt repeats the stable context and includes only beats intersecting
the window's **new** frames. A beat that began earlier is marked as continuing.
Windows with more than one beat are flagged, because a text prompt cannot make
the model perform each beat at an exact frame or in the specified order. Empty
intervals receive a neutral continuation direction. Unlike the server's
default auto-prompt wording, this planner does not impose a locked camera,
one shot or a ban on cuts. Both window geometry fields are included in the
generated `extra_params`.
The requested duration is rounded to a whole number of 24 fps frames; a beat
extending beyond that actual endpoint is rejected instead of silently dropped.

These are directions, not evidence of a connected film. The model can still
change faces, locations, events or ordering, or cut within a window. Prompt
text does not deliver timed dialogue; use a separate audio source and verify
speech and synchronization. This tool has CPU planning tests only; its
storytelling quality has not been tested on the GPU.

## Planning separate requests on one global timeline

For a 300 s film rendered as six requests, `plan_segmented_story.py` maps the
same global beats to each request's **own** continuation geometry:

```sh
python scripts/plan_segmented_story.py --seconds 300 \
  --continuity-file continuity.txt --beats beats.json \
  --segment-frames 720 1440 1440 1440 1440 720 \
  --window 277 --overlap 22 --output segmented-plan.json
```

Frame counts are at 24 fps, must each be at least 96, and must add up to the
requested duration rounded to frames. Each segment includes its zero-based,
end-exclusive global range, local windows, beat indices and its own
`extra_params`. A beat spanning a request boundary is carried into the next
request's prompts as an action already underway. The default reference **plan**
calls for the last frame of the previous segment's verified exact-duration
output, with its expected frame count and zero-based frame index recorded.
For example, a 30 s segment has 720 output frames and references frame 719;
its internal aligned generation may contain 736 frames, so taking the last
frame of an untrimmed native result would refer to the wrong point in time.
`--reference explicit` instead leaves later reference choices to the caller.
The first request uses its normal initial inputs, which can depend on the task.

This tool does not run a request or extract a frame: a controller must trim and
verify the actual completed output to the planned 24 fps frame count, extract
the planned frame, pass it in the next request's task-specific field, and
supply audio trimmed and offset for that request.
The arrays and source ranges are planning information, not one server request;
send each segment's `extra_params` separately with the matching duration and
inputs. A last-frame reference constrains an initial image, not the subsequent
motion, identity or events. The frame boundaries are exact in the plan; the
directions remain approximate. CPU tests cover the mapping, not storytelling
quality, speech, reference extraction or model execution.

For example, extract just the first request's parameters into a private file:

```sh
python3 - <<'PY'
import json, os
from pathlib import Path
plan = json.loads(Path('segmented-plan.json').read_text(encoding='utf-8'))
params = plan['segments'][0]['extra_params']
fd = os.open('segment-01-extra.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w', encoding='utf-8') as f:
    json.dump(params, f, ensure_ascii=False)
PY
```

Use that file with `H3_EXTRA_JSON_FILE` for that request, or with
`inspect_prompt_tokens.py --plan segment-01-extra.json` (plus its required
tokenizer and report options). The tokenizer inspector accepts a single
request's parameters, not the entire segmented plan. Repeat for each segment.

`extra_params_json_bytes_default_python` is the serialized planner output size
before a client merges task and duration. It is not a token count or an exact
server-side field size. Long duplicated prompts can exceed the operating
system's argument limit or the server's multipart field limit. Use a client
that sends large text from files, and set any server field limit only to match
the server's verified configuration. `scripts/inspect_prompt_tokens.py`
counts text tokens in the resulting window list if the local model tokenizer
is available; that count does not establish a safe multimodal context size.
