# Speed loop

How to make the doubles engine faster without losing accuracy against
pokemon-showdown (PS). The work is single-core only: no threads, no
parallelism. The goal is more battles per core, so we can run many processes
later.

Each iteration is: **measure → profile → change one thing → check accuracy →
commit → record**.

## Rules

- **Accuracy floor: 99.9 %** of fuzz battles must match PS (census over seeds
  0–199,999). The current engine is at 99.96 % (71 failing seeds). A speed-up
  may spend at most the difference, and only on purpose.
- **Keep the PRNG draw order.** The engine and PS share one PRNG. One extra or
  missing draw changes every later roll and fails the whole battle. Most
  speed-ups must keep the handler order and the draws the same.
- **Keep debug output working.** `PE_DEBUG_*` / `PE_SLOT` switches
  (`src/doubles/debug.rs`) and the fuzz trace (`rng.trace`) must still print
  the same output. Build debug strings only when the switch is on.
- **One change per commit**, so you can roll back exactly the change that
  made the engine slower or less accurate.
- **Only the champions Reg M-C format counts.** Build with
  `--features champions --no-default-features`. Singles and gen 1–3 builds do
  not need to compile.
- **Guard during speed work.** `fuzz/guard.sh` runs as a pre-commit hook. In
  the speed phase the maintainer may approve `git commit --no-verify` and
  changes to protected files (`src/choices.rs`, generated tables). Run the
  accuracy check yourself in that case.

## 1. Measure

The benchmark plays a fixed set of 100 games (fuzz seeds 0–99, 1,446 turns,
choices recorded once from PS) and prints the best of N runs in microseconds.
Only the games are timed; loading `bench/games.json` is not.

```bash
NO_RECORD=1 bench/run.sh           # working tree, print only
bench/run.sh                        # append "<sha>,<us>" to bench/SPEED.csv
bench/run.sh <rev>                  # any older commit, in a temporary worktree
REPS=40 bench/run.sh                # more runs = less noise
```

Noise is about 3 % on a laptop. Run a change 2–3 times before you trust a gain
smaller than that:

```bash
for i in 1 2 3; do ./target/release/bench-doubles bench/games.json 30 2>/dev/null; done
```

Compare with PS on the same games (one time per machine):

```bash
node bench/ps_bench.js 7
```

Report results as a multiple of PS: `PS microseconds / engine microseconds`,
for example `x112.2`.

Do not regenerate `bench/games.json` (`node bench/gen_games.js 0 100`) between
commits you compare. Numbers in `SPEED.csv` are only comparable on the same
game set.

## 2. Profile

Profile before you change code. Guesses about hot spots were wrong more than
once in this loop.

**CPU time (macOS / Linux), samply:**

```bash
cargo install samply   # once
cargo build --release --features champions --no-default-features --bin bench-doubles
samply record --save-only --unstable-presymbolicate -o prof.json.gz -r 4000 \
    ./target/release/bench-doubles bench/games.json 200
python3 bench/profile_top.py prof.json.gz                   # self / inclusive %
python3 bench/profile_top.py prof.json.gz --callers malloc   # who calls a function
```

**Valgrind (memcheck, callgrind, cachegrind, DHAT)** in Docker, because
valgrind does not run on macOS arm64:

```bash
bench/valgrind.sh
python3 bench/dhat_sites.py target/vg/dhat.out   # allocations per source line
```

Results go to `target/vg/` (`cg_self.txt`, `cg_incl.txt`, `cache.txt`,
`dhat.out`).

**Count calls** when you need to know how often something runs: add a
temporary `static` `AtomicU64` counter, print it from `bench/bench_doubles.rs`,
and remove it before you commit.

## 3. Change one thing

Write down the change and the expected gain before you start. Typical
changes that worked are listed under [What worked](#what-worked).

## 4. Check accuracy

Quick check for every speed-up (about 1 s):

```bash
cargo build --release --features champions --no-default-features
node fuzz/fuzz.js --seed 0 --count 500
cargo test -q --no-default-features --features champions
```

Full census when a change touches battle state or mechanics (for example
moving state from a thread-local into `DoublesState`), or before you finish a
series of changes (about 70 s on 10 cores):

```bash
node fuzz/census.js 0 200000 10 > fails-new.txt
wc -l < fails-new.txt               # must stay <= 200 (99.9 %)
diff fails-old.txt fails-new.txt    # which seeds broke or got fixed
```

To check that debug output did not change, run a batch with
`PE_DEBUG_EVENTS=1` (or another switch) on the old and the new binary and
`diff` stderr.

## 5. Commit and record

```bash
git add <files>
git commit                          # one change, with the reason and the check you ran
bench/run.sh                        # measures the clean HEAD, appends to SPEED.csv
git add bench/SPEED.csv
git commit -m "📝 Record speed of <change>"
```

Then decide:

- **Faster and accurate:** keep it and go to step 1.
- **Faster but less accurate:** fix the accuracy problem in the next commit,
  then measure again. If the fixed version is slower than before the
  speed-up, roll back or plan a different fix.
- **Not faster:** revert it, even when it looks cleaner. Write down why in the
  table below.

## What worked

| change | 100 games | vs PS |
|--------|-----------|-------|
| baseline | 118.6 ms | x14.6 |
| read `PE_DEBUG_*` env vars once; RNG draw log and context strings only when tracing; LTO + `codegen-units = 1` | 89.3 ms | x19.4 |
| event handler sorts: `Event` enum, handler tables resolved once, effects as bit masks (was string matching and a `String` per effect) | 44.6 ms | x38.8 |
| move target types by `Choices` index (was a string match on the lowercased move name); cache species / move name lookups | 26.2 ms | x66.1 |
| static `&str` volatile ids; stack (`ArrayVec` / `SmallVec`) lists in sorts; Mold Breaker state in `DoublesState` instead of a thread-local | 19.1 ms | x90.6 |
| snapshot built from static strings; fewer `Choice` clones | 16.8 ms | x103.1 |
| event sorts check only the conditions the event has handlers for | 16.4 ms | x105.8 |
| per-action scratch lists on the stack; move actions instead of cloning them | 15.4 ms | x112.2 |

## What did not work

- **`Arc` for `Choice::secondaries`** (to make `Choice` clones cheap): no
  measurable gain. The clone cost is the size of the struct, not the
  secondaries list.
- **A 66-arm `match` inside a loop** to read conditions: 2x slower than
  straight-line `if` checks that build a bit mask.
- **Guarding each straight-line check with a "wanted" bit:** slower than
  reading all conditions; the extra branches cost more than the loads they
  skip.
- **Filtering `damaged_by` in place** instead of collecting a new `Vec`: no
  measurable gain (kept because it is not slower).

## Lessons

- **Strings in hot paths were the largest cost.** Look for `to_string()`,
  `to_lowercase()`, `format!` and string `match` tables on every action or
  event. Resolve names to enums or indices once.
- **Most event sorts do nothing.** About 42K event sorts per 100 games, but
  only about 94 have more than one handler, and only those can draw. Make the
  "no handler" case cheap.
- **Thread-locals are not free on macOS.** Each access is a function call
  (`tlv_get_addr`); in this loop they cost about 8 %. Keep battle state in
  `DoublesState`.
- **Allocations:** from 1.66 M to about 148 K per 100 games. The rest is mostly
  fuzz-only output (per-turn snapshot, move-order strings).
- The benchmark measures the fuzz path, which builds a snapshot of every
  turn. An engine path for search without snapshots would be faster again.
