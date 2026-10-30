#!/usr/bin/env bash
# Speed test: time the 100 benchmark games (fuzz seeds 0-99) on a commit and
# append "<sha>,<microseconds>" to bench/SPEED.csv.
#
#   bench/run.sh            current working tree (sha gets "-dirty" if modified)
#   bench/run.sh <rev>      any commit, built in a temporary worktree with this
#                           tree's bench files copied in
#   NO_RECORD=1 bench/run.sh ...   print only, do not append
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPS="${REPS:-20}"
FEATURES=(--release --features champions --no-default-features --bin bench-doubles)

if [[ $# -ge 1 ]]; then
  SHA="$(git -C "$ROOT" rev-parse --short=10 "$1")"
  WT="$(mktemp -d)/wt"
  trap 'git -C "$ROOT" worktree remove --force "$WT" >/dev/null 2>&1 || true' EXIT
  git -C "$ROOT" worktree add --detach "$WT" "$SHA" >/dev/null 2>&1
  mkdir -p "$WT/bench"
  cp "$ROOT/bench/bench_doubles.rs" "$WT/bench/"
  if ! grep -q 'name = "bench-doubles"' "$WT/Cargo.toml"; then
    printf '\n[[bin]]\nname = "bench-doubles"\npath = "bench/bench_doubles.rs"\n' >> "$WT/Cargo.toml"
  fi
  DIR="$WT"
  # share the build cache between worktrees
  export CARGO_TARGET_DIR="$ROOT/target/bench-wt"
  BIN="$CARGO_TARGET_DIR/release/bench-doubles"
else
  SHA="$(git -C "$ROOT" rev-parse --short=10 HEAD)"
  [[ -z "$(git -C "$ROOT" status --porcelain -- src)" ]] || SHA="$SHA-dirty"
  DIR="$ROOT"
  BIN="$ROOT/target/release/bench-doubles"
fi

(cd "$DIR" && cargo build -q "${FEATURES[@]}" 2>/dev/null) || (cd "$DIR" && cargo build "${FEATURES[@]}")
US="$("$BIN" "$ROOT/bench/games.json" "$REPS")"
echo "$SHA,$US"
if [[ -z "${NO_RECORD:-}" ]]; then
  echo "$SHA,$US" >> "$ROOT/bench/SPEED.csv"
fi
