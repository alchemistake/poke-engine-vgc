# Speed test

Fixed set of 100 games: fuzz seeds 0–99 (1446 turns), choices recorded once
from pokemon-showdown. Single core, no parallelism.

To make the engine faster, follow [SPEED_LOOP.md](SPEED_LOOP.md).

| file | what |
|------|------|
| `SPEED.csv` | `<sha>,<microseconds for 100 games>` — one line per measured commit |
| `games.json` | engine input (fuzz `--batch` format), fixed |
| `ps_games.json` | the same games as PS `choose()` calls, for `ps_bench.js` |
| `gen_games.js` | regenerates both files (`node bench/gen_games.js 0 100`) |
| `bench_doubles.rs` | `bench-doubles` binary: best of N runs, JSON parse not timed |
| `run.sh` | build + run + append to `SPEED.csv` |
| `ps_bench.js` | pokemon-showdown replay of the same games (best of N) |
| `profile_top.py` | summarize a samply profile (self / inclusive %, callers) |
| `valgrind.sh` | memcheck, callgrind, cachegrind, DHAT in Docker |
| `dhat_sites.py` | DHAT allocations per engine source line |

```bash
bench/run.sh            # working tree ("-dirty" if src/ is modified)
bench/run.sh <rev>      # any commit, in a temporary worktree
NO_RECORD=1 bench/run.sh
node bench/ps_bench.js 7
```

Do not regenerate `games.json` between commits you compare: the numbers in
`SPEED.csv` are only comparable on the same game set.

## Baseline (84aaa8e, Apple M1 Max)

| simulator | 100 games | per turn |
|-----------|-----------|----------|
| pokemon-showdown (Node 24, replay, no snapshots) | 1,731,183 us | 1.20 ms |
| poke-engine doubles (release) | 118,557 us | 82 us |

## Current (7cb52cd, Apple M1 Max)

| simulator | 100 games | per turn |
|-----------|-----------|----------|
| pokemon-showdown (Node, replay, no snapshots) | 915,252 us | 633 us |
| poke-engine doubles, plain release, fuzz-output mode (`bench/run.sh`) | 7,513 us | 5.2 us |
| poke-engine doubles, PGO, `--sim` (`make bench`) | 5,680 us | 3.9 us |
| poke-engine doubles, plain release, `--sim` (`make bench_plain`) | about 6,560 us | 4.5 us |

## Profile of the baseline (valgrind 3.19 in Docker, linux/arm64, 1 run)

- memcheck: 0 errors, no definite leaks.
- 1,145 M instructions; 1,661,837 heap allocations (~1,150 per turn), 55 MB.
- I1 miss 3.7 %, D1 miss 0.9 %, branch mispredict 5.0 % — not memory bound.
- Self cost: malloc/free/realloc ~27 %; string compare (`handlers(&str)`,
  `ps_move_target(&str)`, `bcmp`) ~20 %; `to_lowercase` / `String::push` /
  `format!` ~8 %; `std::env::var` per RNG draw / event ~8 % (samply, macOS).
- Inclusive: event tie sorts (`run_event_ties_on_effect`) ~58 %, of which
  building `Vec<(String, HandlerKind)>` in `pokemon_effects` and the debug
  `Vec<String>` are most.
