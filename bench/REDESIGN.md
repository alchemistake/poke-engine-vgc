# Redesign plan: from x152 to x650

Goal: 100 benchmark games (seeds 0–99) in about **2,660 us** (x650 of PS),
single core. Accuracy must stay at **≥ 99.9 %** (200K census). Every step
follows [SPEED_LOOP.md](SPEED_LOOP.md): one change, 500-game fuzz check,
commit, `bench/run.sh`.

## Where we are (2026-10-31, 80eccb4, Apple M1 Max)

| run | 100 games | per turn | vs PS |
|-----|-----------|----------|-------|
| pokemon-showdown (`node bench/ps_bench.js 5`) | 1,729,972 us | 1.20 ms | x1 |
| engine, fuzz output (`bench/run.sh`) | 11,358 us | 7.9 us | x152 |
| engine, `--sim` (no snapshots) | 10,386 us | 7.2 us | x167 |
| **target** | **2,660 us** | **1.8 us** | **x650** |

We need about **4.3x** more. Small changes give 3–8 % each now; the profile
is flat (no function above 12 % self).

### Valgrind (3.19, Docker linux/arm64, 1 run incl. JSON load)

- memcheck: 0 errors.
- 161 M instructions (≈ 97 K per turn, after the JSON load).
- **I1 miss 5.0 %** (high: the code does not fit the instruction cache).
- **Branch mispredict 7.5 %.** D1 miss 1.7 % (memory is not the problem).
- 72 K heap allocations, of which ≈ 25 K are JSON load; the rest is
  per-game setup (`build_pokemon`: `to_id`, `parse_*`), `order_entry`,
  per-action `Vec`s and snapshots.

### Code shape (the real cost)

| function | lines | arm64 insns | stack frame |
|----------|-------|-------------|-------------|
| `execute_action_inner` | 7,176 | 19,991 | 3.5 KB |
| `run_battle_fuzz` | ~1,380 | 10,279 | 3.9 KB |
| `end_of_turn` | ~1,270 | 9,203 | 1.4 KB |
| `compute_hit_damage` | ~2,880 | 2,952 | 2.8 KB |

Data: `DoublesState` 11.4 KB, `Pokemon` 680 B (×12), `ActiveSlot` 336 B,
`Choice` 144 B with a heap `Vec` (cloned from a `LazyLock<HashMap>` with
SipHash on each move use).

So most time goes to **control flow**, not to math: giant functions with
hundreds of `if move == X` checks, I-cache misses, mispredicts, and the
event system re-deriving "who has a handler" on each event.

## Strategy

The PS-parity rule (same PRNG draws, same order) stays. We do not rewrite
the engine at once; we replace its core layers one at a time, each behind
the 500-game fuzz check. Estimates are per-step gains on the current time;
they do not add, they multiply.

### 1. Static data tables, no per-use clone (est. 8–12 %)

- `MOVES: LazyLock<HashMap<Choices, Choice>>` → `static MOVES: [Choice; N]`
  (generated, `const`), index by `Choices as usize`. No hash, no `Once`.
- Actions hold `&'static Choice` plus a small `MoveMods` (bp, type,
  category, target, flags) for changes made during the move. No 144-byte
  clone, no heap `secondaries`.
- Same for species / ability / item lookups that still go through
  `to_id` / string parsing.

### 2. Event dispatch by precomputed handler masks (est. 15–25 %)

Today each event collects handlers per slot (`cached_conds`,
`Collector::push_pokemon`, `field_mask`, `sync_vol_order`: ≈ 15 % self
together, `run_event_ties_on_effect` 15.6 % incl.).

- Give each active pokemon, side and the field a `handlers: u128` mask =
  the union of the event bits of its ability, item, status and volatiles.
- Update the mask only in the setters that change one of those (set/clear
  volatile, change item/ability/status, switch). Make the fields private so
  no code path forgets.
- An event first does `if (mask_all & EVENT) == 0 { return }`. Only the
  ≈ 94 sorts per 100 games with more than one handler build a list.

### 3. Speed cache (est. 8–12 %)

`effective_speed` + `sorted_actives` + `resort_queue` + `update_speed` ≈
15 %. Compute the 4 effective speeds once at each PS `updateSpeed()` point,
keep them in the state, and sort 4 entries with a fixed network. Recompute
only when a speed input changes (boost, paralysis, item, ability, Tailwind,
Trick Room, weather for Swift Swim/Chlorophyll …) via a dirty bit.

### 4. Split the giant functions by move behaviour (est. 20–35 %)

This is the largest change and the one that attacks the I-cache and
mispredict numbers.

- Generate a `MoveBehaviour` bit set per move (has custom onTry, onHit,
  basePower callback, multi-hit, charge, protect-like, switch-out, …) into
  the static table from step 1.
- `execute_action_inner` becomes a short common path: PP, BeforeMove, target,
  accuracy, damage, secondaries. Each move-specific block moves to its own
  `#[cold] #[inline(never)] fn` and is called only when its bit is set.
- Same for `end_of_turn` (one function per residual kind, skipped by a
  "something is active" mask) and `run_battle_fuzz` (turn loop vs. fuzz
  bookkeeping).
- Expected: common-path code a few KB instead of ≈ 160 KB, smaller stack
  frames, far fewer unpredictable branches.

### 5. Compact battle state (est. 5–10 %; more for future search)

- `Pokemon` 680 B → ≈ 128 B: keep static data (base stats, types, weight,
  move data) as indices into static tables; keep only mutable data.
- `ActiveSlot` `bool` fields → one `u64` flags word; `Option<String>` gender
  → enum; `Vec` fields in `DoublesState` → `ArrayVec` (no malloc).
- Per-game setup from a pre-built team template (memcpy), not from strings.
  `build_pokemon` is 6.4 % today.

### 6. Fuzz output out of the hot path (est. 5–8 % on the fuzz metric)

`record` checks are runtime `if`s today. Make the recorder a generic
parameter (`R: Recorder`, `NoRecord` = zero cost), move `order_entry` /
snapshot building behind it.

## Expected result

| after step | estimate (us) | vs PS |
|------------|---------------|-------|
| now | 11,358 | x152 |
| 1 static tables | ≈ 10,200 | x170 |
| 2 handler masks | ≈ 8,200 | x210 |
| 3 speed cache | ≈ 7,300 | x237 |
| 4 split functions | ≈ 5,100 | x340 |
| 5 compact state | ≈ 4,600 | x375 |
| 6 recorder generic | ≈ 4,300 | x400 |

Realistic outcome of this plan: **x350–x450**. x650 needs a further step
we can only size after step 4:

### 7. (If needed) Hard-coded fast path for the common turn

Most turns are: 4 actives use damaging moves with no special behaviour, no
field change. After steps 1–6, add a specialised path that handles a turn
where every action and every active has no "special" bit set, without the
generic event machinery, and falls back to the general path the moment any
bit is set. This is how very fast engines (for example `@pkmn/engine`) get
their numbers: no event system on the common path. The fuzz check tells us
at once if the fast path diverges.

## Order of work

Do 1 → 2 → 3 first: low risk, each a few hundred lines, and they make 4
simpler (4 needs the static behaviour bits from 1). Measure after each.
Run the full census after 2, 4 and 5 (they touch state and event order).

## Result of the first pass (2026-10-31, branch `perf/redesign`)

Accuracy: the 200K census fails on the **same 71 seeds** as 80eccb4
(99.96 %); 500-game fuzz passes; `cargo test` passes (2 singles-engine tests
ignored, see below).

| build | 100 games | vs PS |
|-------|-----------|-------|
| 80eccb4 (start) | 11,358 us | x152 |
| 723c08c, fuzz output (`bench/run.sh`) | 7,744 us | **x223** |
| 723c08c, `--sim` | 7,576 us | x228 |
| 723c08c, PGO (`bench/pgo.sh`, optional, per machine) | 6,849 us | x253 |

What each step became:

1. **Static tables:** done. `MOVES` is indexed by `Choices`; secondaries are
   `'static` slices; `Choice` clones are a plain copy; `Move` holds only the
   id (`Pokemon` 680 B → ~100 B); target types are an enum.
2. **Event masks:** done. Per-slot effect bits are cached; the union of the
   four slots is cached per side version (`DoublesSide::ver`); side
   conditions and field are tested by precomputed masks.
3. **Speed cache:** done in part. Speeds are computed once per refresh and
   reused by the queue re-sort; damage computes speeds only for Electro Ball
   / Gyro Ball; the untied active order is reused per side version.
4. **Split giant functions:** not done. A line profile shows
   `execute_action_inner` time spread evenly (~0.3 % per 100 lines over
   ~6,000 lines), no hot region. The engine runs at about 3.5 instructions
   per cycle, so it is instruction-bound, not cache-bound. PGO (which does
   the hot/cold layout this step aimed at) gives 10 %, the upper bound for it.
5. **Compact state:** done in part (`Move`, setup without string parsing,
   FxHash maps). Setup of 100 games: 713 → ~450 us.
6. **Recorder:** move-order entries are a compact type formatted on output.
   This also fixed `--sim`, which skipped them and played a different
   battle (1,445 vs 1,446 turns).

Also: mimalloc in the binaries (−6 %), thread-locals moved into the state.

### Why x650 is not in reach with this design

The remaining time is the PS-parity pipeline itself: per action the engine
runs ~8 speed computations, ~5 target resolutions, ~3.5 `eachEvent('Update')`
sorts, ~3.5 faint resolutions, ~8 event sorts, each a scan of the 4 slots,
plus ~6,000 lines of mechanic checks. Each is already 10–70 cycles. The last
ten changes gave 0–3 % each.

x650 needs ~2.9x fewer instructions per turn. That means a second engine
core, not more tuning of this one:

- a **fast path for plain turns** (no active volatile, no special move or
  ability bit, no field effect): hard-coded order, damage and residual steps,
  with no event machinery; fall back to the full pipeline the moment a bit is
  set (plan step 7);
- **incremental state** for everything the pipeline re-derives after each
  action (speeds, sorts, faint checks), driven by explicit dirty bits set in
  the few mutators, instead of re-scanning after every action.

Both need the fuzzer to guard every divergence and are multi-week work.
