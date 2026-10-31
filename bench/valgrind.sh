#!/usr/bin/env bash
# Run valgrind (memcheck, callgrind, cachegrind, dhat) on the benchmark in a
# Linux container (valgrind does not run on macOS arm64). Needs Docker.
# Output: target/vg/ (cg_self.txt, cg_incl.txt, cache.txt, dhat.out, *.log).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
docker run --rm -v "$ROOT":/src -w /src -e CARGO_TARGET_DIR=/src/target/linux rust:1.81-bookworm bash -c '
set -e
apt-get update -qq >/dev/null && apt-get install -y -qq valgrind >/dev/null 2>&1
cargo build -q --release --features champions --no-default-features --bin bench-doubles
mkdir -p target/vg && cd target/vg
B=/src/target/linux/release/bench-doubles
G=/src/bench/games.json
valgrind --tool=memcheck $B $G 1 > /dev/null 2> memcheck.log
valgrind --tool=callgrind --callgrind-out-file=callgrind.out $B $G 1 2> callgrind.log
callgrind_annotate --inclusive=no callgrind.out > cg_self.txt
callgrind_annotate --inclusive=yes callgrind.out > cg_incl.txt
valgrind --tool=cachegrind --cache-sim=yes --branch-sim=yes --cachegrind-out-file=cachegrind.out $B $G 1 2> cachegrind.log
cg_annotate cachegrind.out > cache.txt
valgrind --tool=dhat --dhat-out-file=dhat.out $B $G 1 2> dhat.log
grep -E "ERROR SUMMARY|total heap usage" memcheck.log
grep -E "Total:" dhat.log
'
