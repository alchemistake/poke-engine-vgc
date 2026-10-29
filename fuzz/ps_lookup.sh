#!/usr/bin/env bash
# Print pokemon-showdown's implementation of a move / ability / condition /
# sim function — the spec to copy. Champions-mod overrides are printed first.
#
#   fuzz/ps_lookup.sh focuspunch          # move, ability, item or condition id
#   fuzz/ps_lookup.sh resolveAction       # a sim function (from a PS call stack)
set -u
ID="${1:?usage: fuzz/ps_lookup.sh <id | simFunction>}"
PS="$(cd "$(dirname "$0")/../../pokemon-showdown" && pwd)"
found=0

# Data entries: `\t<id>: {` up to the matching closing `\t},`.
for f in data/mods/champions/moves.ts data/mods/champions/abilities.ts data/mods/champions/conditions.ts \
         data/mods/champions/items.ts data/mods/champions/scripts.ts \
         data/moves.ts data/abilities.ts data/conditions.ts data/items.ts; do
  out=$(awk -v id="$ID" '
    $0 ~ "^\t" id ": \\{" { p=1 }
    p { print FNR": "$0 }
    p && $0 ~ /^\t},?$/ { exit }' "$PS/$f")
  if [ -n "$out" ]; then
    echo "===== $f ====="
    echo "$out"
    found=1
  fi
done

# Sim / mod-script functions: `\t<name>(` up to the closing `\t}`.
for f in sim/battle.ts sim/battle-actions.ts sim/battle-queue.ts sim/pokemon.ts sim/side.ts \
         data/mods/champions/scripts.ts; do
  out=$(awk -v fn="$ID" '
    $0 ~ "^\t+" fn "\\(" { p=1 }
    p { print FNR": "$0 }
    p && $0 ~ /^\t}$/ { exit }' "$PS/$f")
  if [ -n "$out" ]; then
    echo "===== $f (function) ====="
    echo "$out"
    found=1
  fi
done

if [ "$found" = 0 ]; then
  echo "not found as an id or function. Try: grep -rn \"$ID\" $PS/data $PS/sim"
  exit 1
fi
