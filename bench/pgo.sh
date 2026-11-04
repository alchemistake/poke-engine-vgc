#!/usr/bin/env bash
# Optional profile-guided build of the engine binaries for THIS machine: run
# the benchmark games under an instrumented build, then rebuild with the
# profile. Not used by bench/run.sh (SPEED.csv numbers stay plain release).
#
#   bench/pgo.sh            -> target/pgo/release/{poke-engine,bench-doubles}
#
# Needs `rustup component add llvm-tools-preview`.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="$(rustc -vV | sed -n 's/^host: //p')"
PROFDATA="$(rustc --print sysroot)/lib/rustlib/$HOST/bin/llvm-profdata"
DATA="$ROOT/target/pgo-data"
FEATURES=(--release --features champions --no-default-features)
rm -rf "$DATA" && mkdir -p "$DATA"
cd "$ROOT"
RUSTFLAGS="-Cprofile-generate=$DATA" cargo build -q "${FEATURES[@]}" --target-dir target/pgo-gen
target/pgo-gen/release/bench-doubles bench/games.json 5 >/dev/null 2>&1
target/pgo-gen/release/bench-doubles bench/games.json 5 --sim >/dev/null 2>&1
"$PROFDATA" merge -o "$DATA/merged.profdata" "$DATA"/*.profraw
RUSTFLAGS="-Cprofile-use=$DATA/merged.profdata" cargo build -q "${FEATURES[@]}" --target-dir target/pgo
echo "built target/pgo/release/ (profile: $DATA/merged.profdata)"
