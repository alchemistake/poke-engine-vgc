# Task: make poke-engine's doubles engine match pokemon-showdown exactly

Working directory: `/Users/alchemistake/Code/DubiousDiscSim`.

- **pokemon-showdown** (`pokemon-showdown/`) is the **spec**. Read it freely, **never
  modify it**. Built simulator: `pokemon-showdown/dist/sim`.
- **poke-engine** (`poke-engine/`) is a Rust engine; its doubles engine
  (`src/doubles/`) simulates `gen9championsvgc2026regmc` (2v2, bring 6 pick 4,
  level 50, stat points instead of EVs). **All fixes go here.**
- A fuzzer runs the same random battle in both simulators, one turn at a time,
  and stops at the first turn whose state differs.

All commands below run from `poke-engine/`.

---

## 1. The workflow (follow it exactly)

**Score** = number of consecutive passing battles from seed 0 (seeds
0..N-1 all pass). The baseline is in `fuzz/SCORE`. Your job is to raise it.

**Setup, once per session:**
```shell
cd poke-engine
git checkout fuzz-parity            # work branch; never commit to main
fuzz/install-hooks.sh               # pre-commit hook runs fuzz/guard.sh
cargo build --release --features champions --no-default-features
```

**The cycle:**
1. **Find the first failure.**
   `node fuzz/fuzz.js --seed 0 --count 100`
   No output and exit 0 means all 100 pass: stop and report.
   Otherwise copy the `reproduce:` line (e.g. `--seed 2 --count 1`). That
   seed is the score.
2. **Fix only that battle.** Run only its `reproduce:` command while you work;
   do not re-run the sweep. Read the failure (section 2) and fix it (sections
   3–5). After each engine change:
   `cargo build --release --features champions --no-default-features` and run
   `reproduce:` again. Each change must move the failure later (a later turn or
   a later PRNG draw). If it moves earlier, undo the change.
3. **When the battle passes, run the guard:** `fuzz/guard.sh --update`.
   It checks the build, `cargo test`, the protected files, and that the
   score did not drop; it raises `fuzz/SCORE`. If it prints `GUARD: FAILED`,
   fix what it says before anything else.
4. **Commit** (section 6 lists the files): one commit per fixed battle,
   message `fuzz: seed <N> passes (<mechanic>)`. The body lists, for each
   mechanic, the PS file and function you copied it from. The hook re-runs
   the guard.
5. Go back to step 1.

**Stop and ask the user** (use the notify skill) when:
- the report says `PS ERROR`, or the choices in the report do not match the
  PS battle log (harness bug: never edit `fuzz/fuzz.js` and co.). A
  difference in PRNG draws or damage is **never** a harness bug: both
  simulators share one PRNG stream, so it is always an engine gap. Before
  you call something a harness bug, quote the PS code and the report line
  that prove the harness recorded something PS did not do;
- a fix needs a protected file (guard check 2);
- you are stuck on one battle for 3 attempts that do not move the failure
  later — say what you tried in your report.

If the engine binary "does not know the 'doubles-fuzz-battle' subcommand", it
is a stale build: rebuild. All feature builds share
`target/release/poke-engine`; if you build another feature (`gen9`, `gen4`, …)
**rebuild champions before fuzzing**, or the fuzzer runs the wrong
generation's data. gen1–3 builds are out of scope (ignore them).

**Progress lives in git and the score, not in notes.** To see where the
work stands, run the sweep (step 1). Known unimplemented mechanics are
`// TODO(PS parity):` comments in the code: `grep -rn "TODO(PS parity)" src/doubles src/genx/mechanics.rs`.
Check them first: your failure may already be described there.

---

## 2. Reading a failure

Passing runs print nothing. A failure prints, top to bottom:

| section | what it tells you |
|---|---|
| `reproduce:` | the single-battle command for this failure |
| `team A` / `team B` | the 4 battling Pokémon: species, ability, moves |
| `RESULT:` | `STATE MISMATCH at turn N` + every differing field (`path: PS=… PE=…`), or `ENGINE ERROR` / `PS ERROR` |
| `DIAGNOSIS:` | **read this first** (see below) |
| `differences in detail` | only the differing lines of the failing turn, plus per-simulator `turn events` (`took 39 damage`, `status: none -> burn`) |
| `PS battle log` | what pokemon-showdown actually did that turn (`\|move\|`, `\|-damage\|`, `\|-miss\|`, `\|-immune\|`, `\|-activate\|…ability: X`, `\|-fail\|`) |
| `choices sent` | the controller's choices for that turn, including forced switches |

The full report (all stats, full state, every turn's choices) is in
`fuzz/work/test-1/report.txt`. Engine input and output are in `input.json` and `pe.json` in the same folder.

**DIAGNOSIS** compares the PRNG draws. Both simulators share one PRNG stream, so
the n-th draw has the same *value* on both sides. A bug shows up as a draw used
for a different *purpose*. Every draw is tagged with a purpose (`accuracy`, `crit`,
`damage roll`, `secondary`, `self boost`, `multihit`, `retarget`, `status before move`,
`sleep duration`, `status before move`, `ability`, `sample`, `fickle beam`, `speed tie (move order |
residual | eachEvent | switch-in | queue insert)`). It says one of:

- **"PRNG misaligned at draw #N"**: the engine is missing a draw, or makes an
  extra one, at that point. Every later random number is shifted, so the
  state diff may be only a side effect. **Fix that draw first** (section 3).
  The table shows both sides' draws around #N, the PS call stack, and a `hint:`.
- **"PRNG draws line up"**: RNG is fine; a mechanic or number is wrong
  (section 4).

---

## 2b. Look it up in PS — never guess (mandatory)

Every engine change copies PS code you have **read**. Do not infer a
mechanic from damage numbers, from the move's name, or from memory of the
games, and never write "this build seems to …". The steps:

1. `node fuzz/ps_trace.js --seed <S> --turn <T>`: what PS did, which draws,
   which modifiers (with the ability / item / condition that applied them).
2. `fuzz/ps_lookup.sh <id>` for every move / ability / condition involved
   (e.g. `focuspunch`, `punkrock`, `slp`): prints the champions override
   first, then the generic entry.
3. `fuzz/ps_lookup.sh <function>` for every PS function in the draw's call
   stack (e.g. `resolveAction`, `hitStepAccuracy`, `getRandomTarget`) and
   read where the draw or the value comes from.
4. Write the engine code to do the same thing in the same order, and put
   the PS location in a comment: `// PS: data/moves.ts focuspunch.priorityChargeCallback`.

Example of what goes wrong without this: seed 8 was reported as a
"queue-build speed-tie draw count difference". `ps_trace` plus
`ps_lookup.sh resolveAction` showed the real cause in minutes: Focus Punch's
`priorityChargeMove` action, which makes a `getRandomTarget` draw.

## 3. Fixing PRNG misalignment

Find the acting Pokémon/move at draw #N in the table and decide why PS made
(or skipped) that draw. These are the rules PS follows:

**When a move draws:**
1. `accuracy`: one `random(100)` per move whose accuracy is a number, even
   100. Moves with `accuracy: true` never roll (engine: `always_hit`).
2. No draw at all when the move stops before the accuracy step:
   - an `onTry` / `onTryMove` failure (Spit Up with no Stockpile, Snore while
     awake, Steel Roller with no terrain, …);
   - the target is semi-invulnerable (Dig, Fly, … on their charge turn);
   - Protect;
   - a type immunity or an ability immunity (Levitate, Flash Fire, …);
   - a charge turn of a two-turn move;
   - a failed status check before the move (sleep, freeze, full paralysis).
3. `crit`, then `damage roll`, in `getDamage` for each hit. Fixed-damage and
   OHKO moves draw neither. Fickle Beam draws its 30% roll between them.
4. `secondary`: one `random(100)` per entry in the move's `secondaries`, even
   when `chance` is 100 or missing, and even if the hit KO'd the target.
   Effects done by `onHit` code (Tri Attack, Dire Claw) then draw again
   (`sample`).
5. `self boost`: one roll for a move's `self.boosts` (`selfDrops`).
6. `multihit`: one `sample` for 2–5-hit moves, right after accuracy.
7. `retarget`: if the chosen foe fainted, PS picks a random foe (only when 2
   are left).

**Speed sorts:** each sort makes one `random` per tie group member beyond the first.
- `speed tie (move order)`: the turn's action order.
- `speed tie (eachEvent)`: `eachEvent('Update')` speed-sorts all actives
  after **every action** (`runAction`) and after **every hit**
  (`hitStepMoveHitLoop`). Any two actives with equal speed draw here.
- `speed tie (residual)`: the end-of-turn handler sort. Every status,
  volatile **with a `duration`** (twoturnmove, dig, protect, stall, flinch,
  helpinghand, mustrecharge, …), ability with `onResidual`, side condition and
  weather is a handler. Same order + priority + speed + subOrder means a tie.
- `speed tie (switch-in)`: `runSwitch` sorts all actives after a switch-in.

**Other draws:** `status before move` (champions: full paralysis 1/8, freeze
thaw 1/4, both per move), `sleep duration` (`sample([2,3,3])` when sleep is
set), contact abilities (`onDamagingHit`: Static, Flame Body, …).

**How to fix:**
- Make the engine draw in the same place and order as PS, using the same
  method (`random(n)`, `random_chance(a, b)`, `random_range`).
- Tag every new draw with its purpose so the diagnosis stays accurate:
  `rng.label("secondary").random(100)`. The labels must match `psDrawPurpose`
  in `fuzz/fuzz_purpose.js`.
- Re-run. The misaligned draw number must move later. If it moved earlier,
  you added a draw PS doesn't make.

---

## 4. Fixing mechanics and numbers (PRNG lines up)

1. Read the `PS battle log`: `|-activate|`, `|-ability|`, `|-fail|`, `|-immune|`,
   `|-crit|`, `|-supereffective|` lines usually name the mechanic.
2. If a number is off (damage, base power, stat), trace PS:
   ```shell
   node fuzz/ps_trace.js --seed <S> --turn <T>
   ```
   It prints the turn's protocol lines, every PRNG draw, and every modifier
   with its source, e.g.:
   ```
   modifier: Tough Claws (Pinsir) on BasePower x1.3
   event BasePower: 3 -> 4  (target Pinsir, source Lycanroc, Hard Press)
   ```
   A modifier the engine doesn't apply is the bug. Don't guess the formula.
   Abilities can change hands mid-battle (Wandering Spirit, Skill Swap), so
   check the holder named in the trace, not the team sheet.
3. Read the PS code for that effect and mirror it:
   - **Champions mod first:** `pokemon-showdown/data/mods/champions/` (`moves.ts`,
     `abilities.ts`, `conditions.ts`, `scripts.ts`). If an entry is there,
     it overrides the generic one.
   - **Otherwise the generic code:**
     - `data/moves.ts`: move data, `basePowerCallback`, `onTry`, `condition`;
     - `data/abilities.ts`;
     - `data/conditions.ts`: statuses, weather, twoturnmove, …;
     - `sim/battle-actions.ts`: the move pipeline, damage, switching;
     - `sim/battle.ts`: turn loop, `runAction`, `fieldEvent`, `speedSort`, `modify` / `chainModify`.
   - **Quick data check:**
     ```shell
     node -e "const {Dex}=require('../pokemon-showdown/dist/sim');console.log(Dex.mod('champions').moves.get('hardpress'))"
     ```

**PS integer math** (copy it exactly; never use float multipliers):
- `modify(v, m)` = `trunc((trunc(v * trunc(m*4096)) + 2047) / 4096)`.
- Chained modifiers (`chainModify`) combine as `M = (M * m + 2048) >> 12`, starting at
  4096, and are applied once with `modify`. Base-power handlers run in
  `onBasePowerPriority` order. Engine: `base_power_with_modifiers`.
- `basePowerCallback` formulas (Hard Press, Reversal, …) live in
  `dynamic_base_power`.

**Mechanics that were wrong before and how they work now** (keep them that way):
- **Pivots:** Volt Switch / U-turn / Flip Turn / Teleport switch the user out
  *mid-turn*, right after the move. The switch-in comes from the turn's
  `switches` list (`pivot: true`).
- **Two-turn moves:** the firing turn reuses the charge turn's target
  (`charge_target`).
- **Switch-out:** clears boosts and resets the ability to the base ability
  (`switch_in`).
- **Fainting:** clears status (`faint_pokemon`). The battle ends as soon as a
  side is out of Pokémon; remaining queued actions do not run.
- **Status immunity by type:** Fire can't be burned, Electric can't be paralyzed,
  Ice can't be frozen, Poison/Steel can't be poisoned (`status_type_immune`).

---

## 5. Where engine code goes

**`src/doubles/battle.rs`**
- `run_battle_fuzz`: builds the actions from the choices, sorts them
  (`speed_sort`), runs `execute_action` for each, performs pivot switches,
  then calls `end_of_turn`.
- `execute_action`: follows PS's order. Put a new check at the matching step:
  1. PP;
  2. recharge / flinch / sleep / paralysis;
  3. target and retarget;
  4. charge turn;
  5. `onTry` failures (Stockpile family, Steel Roller);
  6. semi-invulnerable;
  7. Protect;
  8. type immunity;
  9. accuracy;
  10. multihit;
  11. per hit: damage → drain → contact abilities → self boost → `boost` / heal /
      status / volatile / side condition / weather → secondaries → DamagingHit
      abilities;
  12. recoil, self-destruct, recharge, pivot flag.
- `end_of_turn`: the residual handlers (`collect_residual_handlers`
  → `sort_residual_handlers` → run), then faint replacements (`switch_in`),
  then duration ticks.

**Other files**
- `src/doubles/damage.rs`: `calculate_damage` (crit → Fickle Beam → base damage →
  modifyDamage, matching champions `modifyDamage`) and `roll_accuracy`.
- `src/doubles/state.rs`: `ActiveSlot`, which holds per-slot volatiles and resets on
  switch-out; team-member state (HP, status, ability) persists.

**Adding a volatile:**
1. Add a field to `ActiveSlot` and initialize it in `ActiveSlot::empty`.
2. If PS gives it a `duration` or `onResidual`, add it to
   `collect_residual_handlers` with PS's sort keys:
   - order: `on…ResidualOrder`, or `NO_ORDER` if it has none;
   - subOrder: the explicit `on…ResidualSubOrder`, otherwise Status 0, Condition 2, side condition 4,
     weather/field 5, Ability 7, Item 8;
   - speed: the holder's speed.

**`src/choices.rs` (move data):** this is upstream poke-engine's table, with the doubles fields
(`pp`, `multihit`, `self_boosts`, `charge_setup`, `crit_ratio`, `always_hit`,
`Ally` targets, …) added on top. It is shared with the singles engine.
- **Never regenerate or bulk-rewrite it.**
- Champions value differences go in `apply_champions_modifiers` (same file).
- Upstream conventions the doubles engine already handles:
  - a self-drop like Close Combat's is in `boost` (target `User`) *and* in
    `self_boosts`; doubles uses `self_boosts` only;
  - flinch is `Effect::VolatileStatus(FLINCH)`;
  - an ally move's `User` boost goes to the ally.
- When doubles needs PS behaviour that upstream approximates for singles (Tri
  Attack, Fickle Beam), special-case it by move id in `execute_action`
  instead of changing the table.

**Reuse singles code through `src/genx/mechanics.rs`.** The singles engine
(`src/genx/`) already implements most mechanics, but you cannot call it from
doubles directly: it works on the singles `State` (one active per side) and
branches on probabilities instead of drawing random numbers. Shared
mechanics live in `src/genx/mechanics.rs` (in doubles code:
`crate::engine::mechanics`): plain functions that take a `Pokemon`, a move,
or a small context struct, and follow PS exactly. Both engines call them.
Already shared: `immune_to_status` (+ `StatusContext`), `weight_hg`,
`grass_knot_power`, `heavy_slam_power`, `electro_ball_power`,
`gyro_ball_power`, `reversal_power`, `target_hp_power`,
`stat_drop_reaction` (Defiant / Competitive), `THAWS_USER`.

When the fuzzer needs a mechanic:
1. Search `src/genx/` for it (`grep -n ABILITYNAME src/genx/*.rs`).
2. If singles has it, **move** its logic into a function in `mechanics.rs`
   that takes plain values (no `State`, no `SideReference`). Make it PS-exact.
   Change the singles code to call it (keep the singles call site's
   behaviour; `cargo test` must stay green). Then call it from doubles.
3. If singles approximates PS (floats, probability branches), the shared
   function holds the PS-exact version; singles converts its inputs or, if
   tests would change, keeps its own code and you note it in a comment.
4. If singles does not have it, still write it in `mechanics.rs` when it is a
   pure rule (a formula, a table, an immunity check), so singles can use it
   later. Only engine-specific flow (queues, slots, PRNG draw order) stays in
   `src/doubles/`.
5. Plain helpers on shared types are already usable from doubles:
   `Pokemon::is_grounded`, `Pokemon::has_type`,
   `abilities::ignores_target_ability`, `damage_calc::type_effectiveness_modifier`,
   and the champions constants in `generate_instructions.rs`
   (`FULLY_PARALYZED_CHANCE`, `THAW_CHANCE`, …).

---

## 6. Rules, guard rails and commits

**Never:**
- modify `pokemon-showdown/` (guard check 1);
- edit protected files without the user: `fuzz/fuzz.js`, `fuzz/fuzz_purpose.js`,
  `fuzz/teams.js`, `fuzz/ps_trace.js`, `fuzz/ps_lookup.sh`, `fuzz/guard.sh`, `fuzz/install-hooks.sh`,
  and the generated tables `src/doubles/species_types.rs`,
  `src/doubles/species_weights.rs`, `src/doubles/move_targets.rs` (guard check 2);
- regenerate or bulk-rewrite `src/choices.rs` (guard check 3);
- lower or hand-edit `fuzz/SCORE` (only `fuzz/guard.sh --update` raises it;
  only the user may reset it, e.g. after a fuzzer change that widens coverage);
- commit with `--no-verify`, commit to `main`, force-push, or rewrite history;
- change the comparison in `fuzz/fuzz.js` to hide a difference.

**Always:**
- mirror PS exactly, including draw positions; simplest faithful version
  first, `// TODO(PS parity): …` where you stop short;
- keep `cargo test --release --features champions --no-default-features` green
  (singles tests use `src/choices.rs` and `src/genx/` too);
- keep the legacy `doubles-battle` / `doubles-replay` subcommands working.

**What gets committed** (`poke-engine` repo, branch `fuzz-parity`):

| commit | never commit |
|---|---|
| `src/**/*.rs` (engine code) | `target/` (build output) |
| `fuzz/*.md` (prompt / README only), `fuzz/SCORE` | `fuzz/work/` (per-run artifacts; in `.gitignore`) |
| `fuzz/*.js`, `fuzz/*.sh` (only with user approval) | `*.log`, scratch scripts, `/tmp` files |
| `Cargo.toml`, `Cargo.lock` (only if a dependency changed) | anything in `pokemon-showdown/` |
| `doubles/` docs | |

Stage files by name (`git add src/doubles/battle.rs fuzz/SCORE …`),
never `git add -A` / `git add .`. Check `git status` before committing.

Facts the harness already handles:
- battle teams are the 4 picked in team preview;
- `evs` are 0–20 stat points (`stat_points_to_effective_ev`);
- a fainted slot with no replacement is `pass`;
- snapshots use stable battle-team order.


## 7. Debug tools

| tool | use |
|---|---|
| `node fuzz/ps_trace.js --seed S --turn T` | PS's protocol lines, draws, modifiers and value changes for one turn |
| `fuzz/ps_lookup.sh <id or function>` | PS source of a move / ability / condition / item (champions override first) or a sim function |
| `PE_DEBUG_DMG=1 PE_DEBUG_RNG=1 ./target/release/poke-engine doubles-fuzz-battle --input fuzz/work/test-1/input.json --output /tmp/pe.json` | engine trace on stderr: each action, damage calc inputs, every draw |
| `fuzz/guard.sh [--update]` | all guard rails + score; run before every commit |
| `fuzz/work/test-1/pe.json` → `prng_draw_ctx` | engine purpose of every draw (`"crit\|turn 2: B1 Weavile ICESHARD"`) |
| `node fuzz/fuzz.js … --verbose` | full report and per-turn state, including passing turns |

## 8. Report back

For each fix: the mechanic, how PS implements it (file + function), the
engine change, the fuzzer result before and after, and the next failing
battle.
