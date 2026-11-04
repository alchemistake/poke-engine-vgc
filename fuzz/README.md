# Fuzz Tester — poke-engine vs pokemon-showdown (VGC Reg M-C doubles)

Runs the same random battle in **both** simulators, **one turn at a time**, and
exits immediately at the first divergent turn.

- **pokemon-showdown** (`../pokemon-showdown`, source of truth, unmodified):
  a real `Battle` object for `gen9championsvgc2026regmc`, driven in-process
  from Node.
- **poke-engine** (`..`): `doubles-fuzz-battle` subcommand, which runs the
  doubles engine with *explicit* per-turn controller choices.

## Usage

```shell
# build the engine first (champions feature)
cd .. && cargo build --release --features champions --no-default-features

# run the fuzzer
node fuzz/fuzz.js --seed <testerSeed> --count <numTests> [--mode random|champions]
```

### Modes

| mode | teams | score file |
|---|---|---|
| `random` (`fuzz.js` default, secondary) | every species, move, ability and item, past gens included (`teams.js` WIDE_POOL) | `fuzz/SCORE_RANDOM` |
| `champions` (main; `census.js` default) | only sets the format's validator accepts (Reg M-C legal species, moves, abilities, items) | `fuzz/SCORE` |

Each mode has its own seed space, so seed N is a different battle in each
mode. It also has its own PS cache (`fuzz/work/ps-cache`,
`fuzz/work/ps-cache-champions`) and default workdir (`fuzz/work`,
`fuzz/work/champions`). `ps_trace.js` takes `--mode` too; `census.js`
reads `FUZZ_MODE`. `fuzz/guard.sh` checks both scores.

Options:

| flag        | default                              | meaning                                  |
|-------------|--------------------------------------|------------------------------------------|
| `--seed`    | (required)                           | tester seed; makes the whole run reproducible |
| `--count`   | (required)                           | number of tests to run                   |
| `--workdir` | `poke-engine/fuzz/work`              | where per-test artifacts are written     |
| `--engine`  | `poke-engine/target/release/poke-engine` | path to the engine binary           |
| `--verbose` | off                                    | print progress and full labeled per-turn state for passing turns, and the full report on failure |

## How it works

1. A **controller PRNG** (PS `Gen5RNG`, one per test, seeded from
   `tester seed + test index`) drives everything *outside* the battle. So test
   #k of `--seed S --count N` is exactly `--seed S+k-1 --count 1`:
   - the **battle seed** (4 × 16-bit words) — fed identically to both simulators,
   - the two **random teams**, always legal for the format (pokemon-showdown
     `TeamValidator`: Species Clause, Item Clause): random species (every
     team-legal species and forme of the format's own species table,
     `Dex.forFormat(FORMAT)`), 66 random stat points (0–32 per stat, champions
     semantics), random nature, random ability *within the species' pool*, up
     to 4 random distinct moves *from the species' move pool*
     (`getMovePool`), random gender, and a random legal held item (1 in 20
     holds none, never one item twice per team). **Mega formes** (-Mega,
     -Mega-X/Y/Z) are picks of their own: the sheet has the base species
     holding that Mega Stone; no other pick ever holds a Mega Stone.
   - team members are printed with both the raw `stat points: (…)` and the
     effective `stats (level 50, after stat points + nature): (hp: …, atk: …, …)`
     so the expected stat block is visible up front.
   - **team preview**: random 4 of 6, random order (first two are the leads).
   - **in-game actions**: each turn, each active picks a uniformly random
     legal move and random target (1 in 8 single-target moves aim at the
     ally), Mega Evolves half of the time it is offered (one per side), and 1
     in 12 actions is a voluntary switch instead; switch requests pick a
     random legal replacement; fainted slots without a replacement send
     `pass`.
   - the controller seed mixes in the mode's `SEED_SPACES` entry (fuzz.js): a harness change
     that makes different battles gets a new one, so seed N is never an old
     harness's battle.
2. The PS battle is driven turn by turn; after every turn a **full-state
   snapshot** is taken (forme, HP/maxHP, status, faints, active slot, stat
   boosts, held item, current ability, PP per move slot for all battle-team
   Pokémon; volatiles of active Pokémon; side conditions; weather, terrain
   and pseudo-weather with their turns left; move execution order).
3. **Stepwise comparison**: first the **initial state (turn 0)** — right after
   team preview, before any move — is compared (stat calc / max HP, species
   data, leads, starting conditions), so basic setup errors fail before
   anything is simulated. Then after each PS turn, poke-engine is re-run on
   the same choices *so far* (`stop_after: <turn>` in `input.json`, so a
   partial run is not an error) and its state for that turn is compared
   field-by-field (after normalizing cosmetic vocabulary: status ids,
   weather/terrain names, move-name spacing, forme suffixes, slot numbers in
   the move order). **Any difference exits immediately** — the rest of the
   battle is never simulated, so the report only contains the failing turn.
   **Passing tests print nothing** (exit code 0); `--verbose` prints progress
   (`turn N: OK`) and the full labeled per-turn state. The final winner is compared after the last turn of
   a passing battle.

   Note: PS reorders `side.pokemon` when Pokémon switch in, so snapshots are
   taken in stable original-team order on both sides.

Only timestamps / internal ids are ignored (no gameplay field is ignored).

## On mismatch (early exit)

The fuzzer stops at the first divergent turn. The console gets a compact
summary: the `reproduce:` command, battle seed, the 4-Pokémon battle teams,
the differing fields, only the differing lines of the failing turn, and that
turn's choices. The full labeled report is saved to `work/test-N/report.txt`
(and printed instead with `--verbose`):

- tester seed, test index, battle seed,
- both full teams with labeled fields (`ability: …`, `gender: …`, `nature: …`,
  `stat points: (hp: …, atk: …, …)`, the effective
  `stats (level 50, after stat points + nature): (…)`, numbered
  `moves: 1: …, 2: …`), plus the 4-Pokémon battle team picked in team preview,
- `RESULT:` — `STATE MISMATCH at turn 0 (initial state, before any moves were
  simulated)` / `STATE MISMATCH at turn K — simulation stopped here (early
  exit)`, with a machine-readable list of every differing field
  (`path: PS=… PE=…`), or `ENGINE ERROR` / `PS ERROR` with the originating
  crash message,
- **the failing turn in detail**: state of both simulators with every value
  labeled (`hp: 135/181`, `status: paralyze`, `active: slot 0`,
  `stat_change: (spd: +2)`), `<< MISMATCH` markers on differing lines,
  **`turn events`** — what happened to each Pokémon during that turn, per
  simulator (`Happiny: took 27 damage (162 -> 135) | Magearna: status:
  none -> paralyze | Copperajah: bench -> slot 0`), and the move execution
  order with real names,
- the complete choice history up to the failing turn with labeled entries
  (`A slot 0: move: Magearna-Original uses move 2 (Imprison)  target: n/a`,
  `B slot 1: switch: in position 4 (Cresselia)`, `pass: fainted, no replacement`).

Artifacts kept for reproduction: `input.json` (engine input for the last run,
with `stop_after`), `ps.json` (PS snapshots + choices so far), `pe.json`
(engine output for the last run).

Exit codes: `0` all tests passed, `1` mismatch (or simulator error),
`2` bad arguments.

## Engine changes required by the fuzzer

- `src/doubles/state.rs` — `FuzzAction` / `FuzzTurn` input types (including
  `stop_after` for stepwise mode), full-state snapshot types (`FullSlot`,
  `FuzzTurnState`, `FuzzBattleResult`), `fuzz_snapshot()`.
- `src/doubles/battle.rs` — `run_battle_fuzz()`: battle loop driven by
  explicit per-turn choices (move slot + signed target, switch position,
  pass) instead of the old auto move-0 policy; stops cleanly after
  `stop_after` turns.
- `src/io.rs` — `doubles-fuzz-battle` CLI subcommand.

## Tracing one PS turn

```shell
node fuzz/ps_trace.js --seed <S> --turn <T>
```

Replays the battle of `--seed S --count 1` and prints, for turn T, PS's
protocol lines, every PRNG draw (with its purpose, see `fuzz_purpose.js`),
every modifier with the effect that applied it, and every event that changed a
value. Use it when the PRNG draws line up but a number differs.

## Legal-choice audit

```shell
node fuzz/legal_audit.js --seed <S> --count <N>
```

Plays the same battles and, at every move request, compares the engine's
legal choices (`src/doubles/legal.rs`, the `a_options` / `b_options` of each
turn snapshot) with PS's: the offered moves, their PP, which are disabled
(hidden Imprison ones included) and whether the pokemon is trapped.
Mismatches are grouped by kind with examples. It does not change the fuzz
score; the comparator ignores those snapshot fields.

## Surface audit (`surface_audit.js`)

```shell
node fuzz/surface_audit.js [--json]
```

Static check, no battle run: every species, in-battle forme (Mega picks,
battle-only formes), ability, move and held item the generator can produce,
against what the engine knows from its source (`species_data`, the
`Choices` / `Abilities` / `Items` enums, `ps_move_target`) and whether the
doubles engine references it at all. Run it before a fuzz loop: each entry
it lists fails battles for a known reason. Exit code 1 when anything is
missing.

## Speed

- Fast path: battles are checked in chunks of 8 — each whole PS battle is
  played (without PRNG call-stack tracing), then ONE engine call
  (`doubles-fuzz-battle --batch`) runs all of them, and every turn and the
  winner are compared. Only a failing battle is re-run stepwise (one engine
  run per turn, with tracing) to build the report, so reports are unchanged.
- `--jobs N` (default: all CPU cores) splits the seeds across worker
  processes. The parent starts them before loading pokemon-showdown, stops
  workers that can no longer find an earlier failure, and re-runs the lowest
  failing seed for the report. `--jobs 1` runs sequentially; `--verbose`
  disables both.
- PS cache: the PS side of a battle depends only on the seed, the harness
  (`fuzz.js`, `teams.js`) and the pokemon-showdown build — never on the
  engine — so the fast path stores each PS battle in
  `fuzz/work/ps-cache/<key>/<seed>.json` and later sweeps only re-run the
  engine. The key hashes `fuzz.js`, `teams.js` and the PS build files
  (`dist/`); a harness change or PS rebuild starts a new cache (old ones are
  deleted). Cached battles run in engine batches of 128. Failure reports always run PS live.
  `--no-cache` disables it.
- Engine-side compare: a cache file is one entry of the engine's
  `doubles-fuzz-battle --batch --check` input, `{input, expect, winner}`,
  where `expect[n]` is a 64-bit FNV-1a hash of PS's normalized state after
  turn n (`psCanonical`). The engine replays the battle, hashes its own state
  the same way (`fuzz_canonical_hash` in `src/doubles/state.rs`, which mirrors
  `normalizePETurn`) and returns only the first bad turn. Change
  `normalizePETurn` / `normalizePSState` and `fuzz_canonical` together; a
  mismatch between them shows as "fast path failed, stepwise re-run passed".
- Fill the cache ahead of time (all PS battles up to a seed, failures listed):
  `node fuzz/census.js 0 100000`.
- Typical (18 cores): 100 battles ~0.6 s cold, ~0.12 s cached; a
  1000-battle sweep that fails at seed 100 ~0.3 s cached.
