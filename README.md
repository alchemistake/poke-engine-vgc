# Poke Engine - Doubles

This is a port of Pokemon Showdown to Rust based on [pmariglia/poke-engine](https://github.com/pmariglia/poke-engine). It only supports Champions Reg MC

> [!CAUTION]
> This is a slop port, i didn't read the code, i didn't extensively test it. It is not reflecting my or pmariglia's coding skills.
> It is meant to be a starting point for you own pokemon AI's engine.

Tests show around x250 speed up against unmodified pokemon showdown.

It has been developed using generation based fuzzer. It should have 99.96% accuracy against PS.

Only doubles on the Champions format is supported. Singles, search (expectiminimax, iterative deepening, MCTS), instruction generation and gen 1-9 are not part of this project.

## Links

- [Python bindings](poke-engine-py)
- [Fuzzer](fuzz/README.md)
- [CHANGELOG](CHANGELOG.md)

## Building

Install Rust / Cargo, then build with the `champions` feature:

```shell
make champions
# same as:
cargo build --release --features champions --no-default-features
```

This produces `./target/release/poke-engine`.

## Command line

The binary has three doubles subcommands. Each reads a JSON file.

| Subcommand | Input | Output |
|------------|-------|--------|
| `doubles-battle --input <file> [--output <file>]` | Teams, seed, leads | Battle result as JSON (winner, turns, HP per slot, PRNG draws) |
| `doubles-replay --input <file> [--output <file>]` | Teams plus ground-truth hits and heals extracted from a Showdown battle | Per-turn replay result |
| `doubles-fuzz-battle --input <file> [--output <file>] [--batch]` | Teams, seed, leads and explicit per-turn choices | Per-turn full-state snapshots and legal options |

`--batch` takes a JSON array of battles and returns an array of results in the same order. A battle that panics returns `{"error": "panic: ..."}` instead of stopping the batch.

Without `--output`, the result is printed to stdout.

```shell
./target/release/poke-engine doubles-battle --input battle.json
```

## Python bindings

The bindings live in [poke-engine-py](poke-engine-py). They are built with [maturin](https://www.maturin.rs/) and need the `champions` feature.

### Building

```shell
make dev_doubles
```

This creates `venv/` and runs `maturin develop --no-default-features --features champions` inside it. Activate it with `. venv/bin/activate` before using `poke_engine`.

### How it is wired

- `poke-engine-py/src/doubles.rs` defines the PyO3 classes and functions (`DoublesPokemon`, `DoublesAction`, `DoublesBattle`, ...). Each one wraps a type from `poke_engine::doubles`.
- `poke-engine-py/src/lib.rs` adds the module only when the crate is built with `champions` (`#[cfg(feature = "champions")] mod doubles;`). A build without it has no doubles API.
- Engine panics are caught and re-raised as Python `RuntimeError`. Invalid input raises `ValueError`.
- `poke-engine-py/python/poke_engine/__init__.py` re-exports everything from the compiled module. `poke_engine.pyi` holds the type stubs.

### API

- `DoublesPokemon(species, moves, ability, item, nature, evs, level, ivs, gender, name)`: one team member. Stats come from the species base stats and the Champions stat points in `evs`. Teams have 2 to 6 members.
- `DoublesAction.move(slot, target=1, struggle=False)`, `DoublesAction.switch(position)`, `DoublesAction.pass_()`: one action for one active slot. Targets use Showdown locations: `1`/`2` is a foe slot, `-1`/`-2` is an own slot.
- `DoublesTurnChoices(a, b, switches=[])`: both sides' actions for one turn, plus answers to any mid-turn switch requests.
- `run_doubles_battle(team_a, team_b, seed, leads_a, leads_b)`: plays a whole battle with the engine's built-in move policy.
- `run_doubles_battle_with_choices(team_a, team_b, seed, choices, ..., stop_after=None, trace=False)`: plays a battle from explicit choices and returns a snapshot per turn.
- `DoublesBattle(team_a, team_b, seed, leads_a, leads_b, trace=False)`: plays one turn at a time. Use `choose(a, b)` for a turn, `legal_options(side)` / `legal_actions(side)` / `legal_joint_actions(side)` for legal choices, `switch(side, slot, position)` to answer a pending switch request, and `copy()` for search.

### Example

```python
from poke_engine import (
    DoublesAction,
    DoublesBattle,
    DoublesPokemon,
    run_doubles_battle,
)

team = lambda *species: [DoublesPokemon(s, ["tackle", "protect"]) for s in species]
team_a = team("Incineroar", "Rillaboom", "Pikachu", "Garchomp")
team_b = team("Amoonguss", "Kingambit", "Dragonite", "Sneasler")
seed = (1, 2, 3, 4)

# Whole battle, built-in policy
print(run_doubles_battle(team_a, team_b, seed).winner)

# One turn at a time
battle = DoublesBattle(team_a, team_b, seed)
state = battle.choose(
    [DoublesAction.move(0, 1), DoublesAction.move(0, 2)],
    [DoublesAction.move(0, 1), DoublesAction.move(0, 2)],
)
print(battle.turn, battle.waiting_for)  # 2 turn
```

## Verifying

### Tests

```shell
cargo test --no-default-features --features champions
pytest --rootdir=poke-engine-py/python poke-engine-py/python/tests/test_doubles.py
```

`test_doubles.py` skips itself when `poke_engine` was built without `champions`. It covers battle determinism, per-turn snapshots, Protect, and the switch flow of `DoublesBattle`.

### Binding check

After `make dev_doubles`, confirm the bindings load:

```shell
python -c "import poke_engine; print(poke_engine.run_doubles_battle)"
```

If this fails with `AttributeError`, the module was built without `champions`. Rebuild with `make dev_doubles`.

### Fuzzer

The fuzzer plays the same random battle in pokemon-showdown (`../pokemon-showdown`, unmodified, built so that `dist/` exists) and in this engine, one turn at a time. It stops at the first turn where the states differ.

Build the engine first, then run:

```shell
make champions
node fuzz/fuzz.js --seed <S> --count <N>
```

- Test `k` of `--seed S --count N` is the same as `--seed S+k-1 --count 1`. Use that to reproduce a failure.
- Passing tests print nothing. Exit code `0` means all passed, `1` means a mismatch or simulator error, `2` means bad arguments.
- On a mismatch, the console shows the differing fields and the failing turn. The full report is in `fuzz/work/test-N/report.txt`.
- `--verbose` prints every turn and disables the parallel and cache paths.
- `--jobs N` sets the worker count (default: all cores). `--no-cache` disables the cache of pokemon-showdown battles.

To trace one turn of a Showdown battle (PRNG draws and modifiers):

```shell
node fuzz/ps_trace.js --seed <S> --turn <T>
```

To check the engine's legal choices against Showdown's move requests:

```shell
node fuzz/legal_audit.js --seed <S> --count <N>
```

The full option list and report format are in [fuzz/README.md](fuzz/README.md).

`fuzz/guard.sh` runs the checks used before commits: Showdown is unmodified, the champions build and `cargo test` pass, and the fuzz score has not dropped below `fuzz/SCORE`.

### Speed

`bench/run.sh` builds and times the fixed set of 100 fuzz games and records the result in `bench/SPEED.csv`. See [bench/README.md](bench/README.md).

## Engine input

Teams are described by `DoublesPokemon` in Python or by the JSON that `doubles-battle` reads. The type definitions are in [src/doubles/state.rs](src/doubles/state.rs) and the battle loop is in [src/doubles/battle.rs](src/doubles/battle.rs).
