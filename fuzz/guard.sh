#!/usr/bin/env bash
# Guard rails for the fuzz-parity work. Run from anywhere:
#
#   fuzz/guard.sh            # all checks; fails (exit 1) on any violation
#   fuzz/guard.sh --update   # same, and raise a score file if its score went up
#
# The git pre-commit hook (fuzz/install-hooks.sh) runs it, so a commit is
# refused when a check fails. Checks:
#   1. pokemon-showdown is unmodified (it is the spec).
#   2. Protected files are unchanged (fuzz harness, generated data) unless
#      ALLOW_HARNESS_EDIT=1 (only the user sets this). 2b: the generated
#      event-handler table matches fuzz/gen_event_handlers.js.
#   3. src/choices.rs is not bulk-rewritten.
#   4. No build artifacts / fuzz output are staged.
#   5. The champions build succeeds and `cargo test` is green.
#   6. Score (consecutive passing battles from seed 0) did not drop, for each
#      fuzzer mode: champions (main) -> fuzz/SCORE, random -> fuzz/SCORE_RANDOM.
set -u
cd "$(dirname "$0")/.."           # poke-engine/
ROOT="$(pwd)"
PS_DIR="$ROOT/../pokemon-showdown"
FAIL=0
fail() { echo "GUARD FAIL: $*"; FAIL=1; }
ok() { echo "guard ok: $*"; }

# 1. pokemon-showdown untouched.
# The pre-commit hook exports GIT_DIR (a worktree's): clear it for the PS repo.
if [ -n "$(env -u GIT_DIR -u GIT_INDEX_FILE -u GIT_WORK_TREE git -C "$PS_DIR" status --porcelain 2>/dev/null)" ]; then
  fail "pokemon-showdown has local changes. It is the spec: revert them (git -C ../pokemon-showdown checkout -- . ; git -C ../pokemon-showdown clean -fd)."
else
  ok "pokemon-showdown unmodified"
fi

# 2. Protected files. Compared against HEAD (staged + unstaged).
PROTECTED=(
  fuzz/fuzz.js fuzz/fuzz_purpose.js fuzz/teams.js fuzz/ps_trace.js fuzz/ps_lookup.sh
  fuzz/guard.sh fuzz/install-hooks.sh fuzz/SCORE fuzz/SCORE_RANDOM
  src/doubles/species_types.rs src/doubles/species_weights.rs src/doubles/move_targets.rs
  src/doubles/event_handlers.rs fuzz/gen_event_handlers.js
  src/doubles/ability_flags.rs fuzz/gen_ability_flags.js
)
if [ "${ALLOW_HARNESS_EDIT:-0}" != "1" ]; then
  for f in "${PROTECTED[@]}"; do
    if git cat-file -e "HEAD:$f" 2>/dev/null && ! git diff --quiet HEAD -- "$f"; then
      # The score files may only go up through `guard.sh --update`.
      if [ "$f" = "fuzz/SCORE" ] || [ "$f" = "fuzz/SCORE_RANDOM" ]; then continue; fi
      fail "protected file changed: $f (harness / generated data). If the harness is wrong, notify the user and stop."
    fi
  done
  ok "protected files unchanged"
fi

# 2b. The generated event-handler table matches its generator (PS data).
if [ -f fuzz/gen_event_handlers.js ] && ! node fuzz/gen_event_handlers.js | cmp -s - src/doubles/event_handlers.rs; then
  fail "src/doubles/event_handlers.rs is not what fuzz/gen_event_handlers.js generates. Never edit it by hand."
else
  ok "event handler table up to date"
fi
if [ -f fuzz/gen_ability_flags.js ] && ! node fuzz/gen_ability_flags.js | cmp -s - src/doubles/ability_flags.rs; then
  fail "src/doubles/ability_flags.rs is not what fuzz/gen_ability_flags.js generates. Never edit it by hand."
else
  ok "ability flag table up to date"
fi

# 3. No bulk rewrite of the shared move table.
DEL=$(git diff --numstat HEAD -- src/choices.rs | awk '{print $2}')
if [ -n "$DEL" ] && [ "$DEL" -gt 200 ]; then
  fail "src/choices.rs: $DEL lines deleted vs HEAD. Never regenerate or bulk-rewrite it; add champions changes in apply_champions_modifiers."
else
  ok "src/choices.rs not bulk-rewritten"
fi

# 4. Nothing generated is staged.
BAD=$(git diff --cached --name-only | grep -E '^(target/|fuzz/work/)|\.log$' || true)
if [ -n "$BAD" ]; then
  fail "generated files are staged (unstage them: git restore --staged <file>):"; echo "$BAD"
else
  ok "no generated files staged"
fi

# 5. Build + tests (champions build last, the fuzzer uses this binary).
if ! cargo build --release --features champions --no-default-features >/tmp/guard_build.log 2>&1; then
  fail "champions build failed (see /tmp/guard_build.log)"
fi
TESTS=$(cargo test --release --features champions --no-default-features 2>&1 | grep -E '^test result' |
  awk '{p+=$4; f+=$6} END {print p" "f}')
if [ "$(echo "$TESTS" | awk '{print $2}')" != "0" ] || [ -z "$TESTS" ]; then
  fail "cargo test failed (passed/failed: $TESTS). Run: cargo test --release --features champions --no-default-features"
else
  ok "cargo test green ($(echo "$TESTS" | awk '{print $1}') passed)"
fi

# 6. Score, per fuzzer mode (one at a time: each sweep uses every core).
TARGET=1000000
for MODE in champions random; do
  if [ "$MODE" = "champions" ]; then FILE=fuzz/SCORE; else FILE=fuzz/SCORE_RANDOM; fi
  MARG=" --mode $MODE"
  OUT=$(node fuzz/fuzz.js --seed 0 --count $TARGET --mode $MODE 2>&1)
  if [ -z "$OUT" ]; then
    SCORE=$TARGET
  else
    # The first failing seed (from the report's `reproduce:` line) is the score.
    SCORE=$(echo "$OUT" | grep -oE 'reproduce: node fuzz/fuzz.js --seed [0-9]+' | head -1 | grep -oE '[0-9]+$')
    SCORE=${SCORE:-0}
  fi
  BEST=$(cat $FILE 2>/dev/null || echo 0)
  echo "score ($MODE): $SCORE (consecutive passing battles from seed 0; baseline $FILE = $BEST)"
  if [ "$SCORE" -lt "$BEST" ]; then
    fail "$MODE score dropped from $BEST to $SCORE: seed $SCORE regressed. Reproduce: node fuzz/fuzz.js --seed $SCORE --count 1$MARG"
  elif [ "$SCORE" -gt "$BEST" ] && [ "${1:-}" = "--update" ]; then
    echo "$SCORE" > $FILE
    ok "$FILE raised to $SCORE"
  fi
done

if [ "$FAIL" = "1" ]; then
  echo "GUARD: FAILED"
  exit 1
fi
echo "GUARD: PASSED"
