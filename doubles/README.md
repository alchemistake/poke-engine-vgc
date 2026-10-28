# Doubles Engine (gen9championsvgc2026regmc)

A self-contained **doubles** (2-active-per-side) battle engine for the
`gen9championsvgc2026regmc` format, built to match [pokemon-showdown](https://github.com/smogon/pokemon-showdown)
exactly.

It lives in `src/doubles/` and is **separate from the existing singles
`State`/`Side` model**, so the singles engine and its test suite are untouched.
It reuses the shared value types (`Pokemon`, `Choices`, `MOVES`, `Abilities`,
`Items`) and the Gen 9 data tables.

## Status

**Core mechanics verified correct against pokemon-showdown:**
- Stat calculation (level 50, champions EV "stat points" → effective EV, 31 IV) — exact match.
- Damage formula (`getDamage` + `modifyDamage` + `randomizer` + `modify` + `trunc`) — exact match (base damage values match PS move-for-move).
- Gen 9 type chart (via the shared `type_effectiveness_modifier`).
- Gen5RNG LCG PRNG — byte-for-byte match with PS's `Gen5RNG`.
- Move ordering by priority/speed with PS's `speedSort` tie-shuffle.
- Targeting (foe location 1 = leftmost active), retargeting on faint.
- Faints and end-of-turn switch-ins.

**Battle-loop validation (replay test):** `harness/ps_extract.js` drives a real PS
battle (auto-choices via `default`) and hooks `pokemon.damage` / `pokemon.heal` to
record ground-truth per-hit damage and heals. `poke-engine doubles-replay` replays
those events through the engine's battle loop (faints, switch-ins, retargeting,
winner determination). This validates a **complete end-to-end battle** against PS.

**Random-team parity:** the replay test works on full random teams
(`gen9championsvgc2026regmc`, generated via PS's `Teams.generate`), covering
abilities, items, spread/partner/self/foeSide moves, status, heals (berries,
recovery), and forme changes (Decidueye-Hisui, Castform formes, Furfrou costumes).
Verified: **300+ random battles, 100% winner match**, with a balanced A/B
winner distribution (so the test is not trivially one-sided).

> Note: this format uses **4-pokemon teams** ("bring 6, pick 4" — the doubles
> `Picked Team Size = Auto` rule), not 6.

## Comparison harness

- `harness/genrand.js` — generate two random 4-pokemon teams for the format.
- `harness/ps_extract.js` — drive a real PS battle and emit ground-truth hits/heals.
- `harness/batch_test.sh` / `batch_test2.sh` — run N random battles and count
  winner matches.

```shell
node harness/genrand.js gen9championsvgc2026regmc 11,22,33,44 a.json b.json
node harness/ps_extract.js 11,22,33,44 a.json b.json ext.json
./target/release/poke-engine doubles-replay --input harness/ext.json

# run 100 random battles
harness/batch_test.sh 100
```

There is also a self-driven comparison (`harness/ps_driver.js` + `doubles-battle`) that
validates the PRNG draw ordering; it matches PS exactly through the first few turns of
a controlled battle, with the remaining gap being PS's event-system draw consumption
for mid-turn faints/switches (deferred).

## Layout

- `src/doubles/rng.rs` — `Gen5Rng`, a faithful Gen 5 LCG PRNG (with a draw log).
- `src/doubles/state.rs` — `DoublesState` / `DoublesSide` / `ActiveSlot`, team
  setup, stat calc, serde input/output types.
- `src/doubles/species_types.rs` — auto-generated species → (base stats, types)
  table from PS's Gen 9 Pokedex (1481 species).
- `src/doubles/damage.rs` — damage formula matching PS exactly.
- `src/doubles/battle.rs` — the battle driver (turn loop, ordering, switches).
- `src/io.rs` — `doubles-battle` CLI subcommand.

## Using it

Build (champions feature):
```shell
cargo build --release --features champions --no-default-features
```

Run a battle from a JSON input file (see `harness/out.json` for the format):
```shell
./target/release/poke-engine doubles-battle --input harness/out.json --output harness/rust_out.json
```

Run the replay test:
```shell
./target/release/poke-engine doubles-replay --input harness/ext.json
```
