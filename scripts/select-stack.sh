#!/bin/bash
# Point the systemd units (head, agent, memguards) at a runtime stack by editing ~/.h3-tp2.env on THIS host.
#   select-stack.sh base                 -> $HOME/h3-base-20260916 (vLLM-Omni e1aa6eae, FL2VA)
#   select-stack.sh long [FL2VA|Ref2VA]  -> $HOME/h3-long (vLLM-Omni v0.30.0), partition default Ref2VA
# Then restart: worker `systemctl --user restart h3-tp2-agent.service`, head `systemctl --user restart h3-tp2-head.service`.
set -euo pipefail
stack=${1:?base|long}; part=${2:-Ref2VA}
f="$HOME/.h3-tp2.env"; touch "$f"
sed -i '/^H3_ROOT=/d; /^H3_VENV=/d; /^H3_MODEL=/d' "$f"
case "$stack" in
  base) ;;
  long) printf 'H3_ROOT=%s\nH3_VENV=%s\nH3_MODEL=%s\n' "$HOME/h3-long" "$HOME/h3-long/.venv" "$HOME/h3-long/weights/$part" >> "$f" ;;
  *) echo "unknown stack $stack" >&2; exit 2 ;;
esac
echo "== $(hostname): $f"; cat "$f"
