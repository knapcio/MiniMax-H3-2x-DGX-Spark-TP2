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
storytelling quality has not been tested on the GPU. Splitting a film into
multiple separate requests additionally requires carrying reference frames and
mapping the global timeline into each request; this planner does not do that.

`extra_params_json_bytes_default_python` is the serialized planner output size
before a client merges task and duration. It is not a token count or an exact
server-side field size. Long duplicated prompts can exceed the operating
system's argument limit or the server's multipart field limit. Use a client
that sends large text from files, and set any server field limit only to match
the server's verified configuration. `scripts/inspect_prompt_tokens.py`
counts text tokens in the resulting window list if the local model tokenizer
is available; that count does not establish a safe multimodal context size.
