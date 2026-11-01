# File-backed prompt and extra_params transport

`scripts/render.sh` accepts `H3_PROMPT_FILE` and `H3_EXTRA_JSON_FILE` for UTF-8
text files. This avoids putting large prompt text or JSON into executable
arguments or exported environment values. The original positional prompt and
`EXTRA_JSON` interface remains supported for smaller inputs.

```bash
H3_PROMPT_FILE=/path/to/prompt.txt \
H3_EXTRA_JSON_FILE=/path/to/extra.json \
CONT_PROMPTS=auto \
bash scripts/render.sh result.mp4 300 960 544 20 3101 '' /path/to/reference.png first
```

The seventh positional argument is an empty placeholder when using a prompt
file. Supplying both a nonempty positional prompt and `H3_PROMPT_FILE` is an
error. Likewise, choose either nonempty `EXTRA_JSON` or `H3_EXTRA_JSON_FILE`.
Each file option can also be used independently with the other legacy input.
Do not use `"$(cat prompt.txt)"` or `EXTRA_JSON="$(cat extra.json)"` for large
inputs: the caller can fail before `render.sh` starts.

### Matrix wrapper

The updated `scripts/bench-matrix.sh` recognizes `H3_PROMPT_FILE` and passes
an empty seventh argument to `render.sh`; it does not read the file into
`PROMPT`. Clear an inherited `PROMPT` and `EXTRA_JSON` before using the file
options. In a separately authorized benchmark window, the invocation is:

```bash
unset PROMPT EXTRA_JSON
H3_PROMPT_FILE=/path/to/prompt.txt \
H3_EXTRA_JSON_FILE=/path/to/extra.json \
CONT_PROMPTS=auto \
bash scripts/bench-matrix.sh '300:960:544:20:3101' /path/to/reference.png
```

Without a prompt file, the wrapper keeps its existing `PROMPT` or default
scene. With a prompt file and nonempty `PROMPT`, it fails before SSH,
memwatch or rendering. `PROMPT=''` also works with the updated wrapper.
**Old pinned copies do not support this mode**: `${PROMPT:-default}` restores
their nonempty default even after `PROMPT=''`, causing a conflict with the
new render client. Do not mix that old wrapper with the new file interface.
CPU tests exercise the real wrapper with inert render/SSH/memwatch stubs.

The helper `scripts/render_text_fields.py` is required alongside `render.sh`.
It takes paths and scalar options only. It reads the inputs, merges the same
task/duration/sampler fields as before, and generates automatic continuation
prompts when requested. Explicit `continuation_prompts` still take precedence.
Legacy text is written with bash's builtin `printf`; generated JSON is never
captured in a shell variable, executable argument or environment entry.

Both modes create a private `h3-text.*` temporary directory. curl reads
`prompt=<file` and `extra_params=<file` as **text fields**, not uploaded files.
Semicolons, UTF-8, CRLF and trailing newlines in the prompt survive verbatim.
JSON is parsed, merged and serialized again, preserving its string values
rather than its original whitespace layout. Image references remain `@file`
uploads. Paths themselves are quoted for curl's form grammar, including
spaces, semicolons, commas, double quotes and backslashes.

`timing.json` reads the same private snapshots used by curl. Its existing
`prompt` and `extra_params` values remain available; the added `text_fields`
object records each transmitted field's byte count and SHA-256. Changing an
input file after submission cannot change that timing record. Temporary text
and inline-audio files are removed when the script exits normally or handles
an ordinary failure; an uncatchable process kill may leave its private folder.

### Future bundles must include the helper

This change adds a required file dependency, not a service dependency.
Every future bundle adopting the new `render.sh` must copy
`scripts/render_text_fields.py` beside it and include both files in its
checksum manifest and verification/snapshot closure. A bundle adopting the
matrix file interface must also include the updated `bench-matrix.sh`.
Test that complete bundle with local HTTP before any separately authorized
deployment. Do not update a pinned active bundle in place.

The inspected workspace paths needing care on a **future** revision are:

- `diagnostics/h3-lalka-20261101/coherent-300/prepare_followup_retry.py`
  explicitly copies and pins only `scripts/render.sh` when replacing its
  client. Adopting this revision requires adding the helper copy/hash and
  regenerating the new bundle/queue pins; its present pinned old client is
  unchanged and needs no helper.
- `diagnostics/h3-lalka-20261101/pacing-ab-queue/backend.py` executes verified
  render bytes with `$0` pointing into its private snapshot. A future bundle
  using the new client must include/hash the helper in `assets-manifest.json`
  and the snapshot, because the shell resolves it relative to `$0`. Any
  switch to large file inputs also requires that backend to pass the empty
  prompt placeholder and file paths rather than its current prompt argv and
  `EXTRA_JSON` environment. None of those candidate/pinned files was edited.
- `scripts/sync-to-nodes.sh` copies the whole repository, so the new helper
  is included by its existing rule. That deployment command was not run.

Existing coherent-film staging and historical retry bundles remain unchanged;
their older self-contained render scripts continue to use their original pins.

## Limits

`H3_TEXT_MAX_FIELD_BYTES` defaults to **1048576 bytes per field**, matching the
unmodified server multipart limit. The check applies to each input file and
to each final serialized field, including the generated prompt list. UTF-8
bytes count, not characters. An oversized generated list is rejected before
curl runs. The allowed client setting is 1..67108864 bytes; raising it does
not change the server. Use a larger value only after the server's matching
limit has been configured and verified separately. Inline audio retains its
existing `H3_AUDIO_MAX_FIELD_BYTES` setting.

These are **transport limits, not supported model token counts**. The file
route does not establish safe text-encoder memory, instruction following or
long-video quality. Server-generated continuation prompts and text-plus-image
conditioning can have additional costs not represented by the multipart size.

## Reproduction and CPU verification

On a local Linux VM with 4096-byte pages, a 131072-byte argument failed with
`errno=7 / E2BIG`. A 19200-byte input prompt expanded into 29 prompts also
reproduced the bug in the **old actual render.sh**: curl could not start, then
the timing Python invocation failed with `Argument list too long`. The old
script returned 3 without contacting a server. A generated 574342-byte JSON
also failed when supplied directly as one argument or exported value. Exact
limits depend on the OS/page size; those are measured values, not a universal
maximum for every host.

The new tests use a real local HTTP multipart receiver and curl. They exercise
both fields above 128 KiB, 29 automatically generated prompts above 4 MiB,
literal Unicode/punctuation/newlines, source-file changes before timing,
image upload semantics, conflicting inputs, invalid UTF-8 and byte limits.
Hashes received over HTTP match the timing metadata. Python/curl exec wrappers
also verify every argument and environment value remains below 64 KiB on the
file route. The original small positional/env path and its 29-window expansion
are tested as well.

```bash
python3 -m unittest discover -s tests -p 'test_render*.py' -v
python3 -m unittest discover -s tests -p 'test_plan_dialogue*.py' -v
```

The enclosing workspace retains the Linux reproduction and source hashes at
`diagnostics/h3-text-file-transport-20261101/`. Testing uses only CPU and local
HTTP; it neither invokes H3 nor changes running services or queued jobs.
