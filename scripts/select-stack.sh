#!/bin/bash
# Point the systemd units (head, agent, memguards) at a runtime stack by editing ~/.h3-tp2.env on THIS host.
#   select-stack.sh base                 -> $HOME/h3-base-20260916 (vLLM-Omni e1aa6eae, FL2VA)
#   select-stack.sh long [FL2VA|Ref2VA]  -> $HOME/h3-long (vLLM-Omni v0.30.0), partition default Ref2VA
# Then restart: worker `systemctl --user restart h3-tp2-agent.service`, head `systemctl --user restart h3-tp2-head.service`.
set -euo pipefail
stack=${1:?base|long}; part=${2:-Ref2VA}
case "$stack" in base|long) ;; *) echo "unknown stack $stack (base|long)" >&2; exit 2 ;; esac
case "$part" in FL2VA|Ref2VA) ;; *) echo "unknown partition $part (FL2VA|Ref2VA)" >&2; exit 2 ;; esac
f="$HOME/.h3-tp2.env"; touch "$f"
tmp=$(mktemp "$f.XXXXXX")
grep -v -E '^(export )?H3_(ROOT|VENV|MODEL)=' "$f" > "$tmp" || true
if [ "$stack" = long ]; then
  printf 'H3_ROOT=%s\nH3_VENV=%s\nH3_MODEL=%s\n' "$HOME/h3-long" "$HOME/h3-long/.venv" "$HOME/h3-long/weights/$part" >> "$tmp"
fi
mv -f "$tmp" "$f"   # atomic: a restart never sees a half-edited file
echo "== $(hostname): $f"; cat "$f"
