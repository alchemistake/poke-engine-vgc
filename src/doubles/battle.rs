//! The doubles battle driver.
//!
//! Drives a full doubles battle turn by turn, consuming the shared `Gen5Rng`
//! in the exact order pokemon-showdown does, so damage rolls / criticals /
//! accuracy / speed-tie shuffles all line up with a pokemon-showdown battle run
//! on the same seed and the same move choices.

use crate::doubles::move_targets::Tgt;
use crate::doubles::state::OrderEntry;
use crate::choices::{ChargeSetup, Choice, Choices, Effect, MoveCategory, MoveTarget, StatBoosts, MOVES};
use crate::doubles::damage::{calculate_damage, roll_accuracy, roll_accuracy_boosted, roll_accuracy_boosted_labeled, DamageCtx, DamageOutcome};
use crate::doubles::rng::Gen5Rng;
use crate::engine::abilities::Abilities;
use crate::engine::state::PokemonVolatileStatus;
use crate::doubles::state::{
    ActiveSlot, BattleResult, DoublesSide, FuzzAction, FuzzBattleResult, FuzzTurn, MoveResult,
    ReplayMove, ReplayTurnHp, Side, SlotHp, SwitchResult, TeamBoosts, TurnEntry, TurnHp,
    DoublesState, fuzz_snapshot, multiply_boost,
};
use crate::state::{
    PokemonBoostableStat, PokemonIndex, PokemonMoveIndex, PokemonSideCondition, PokemonStatus, PokemonType,
    StateTerrain, StateTrickRoom, StateWeather,
};
use crate::engine::state::Weather;
use crate::engine::generate_instructions::{FULLY_PARALYZED_CHANCE, THAW_CHANCE};

/// One queued action (an on-field Pokémon using a move).
#[derive(Debug, Clone)]
struct Action {
    side: Side,
    slot: usize,
    pokemon_index: PokemonIndex,
    choice: Choice,
    priority: i8,
    speed: i16,
    /// Explicit target slot on the foe side (fuzz mode); 0 when unused.
    /// Negative values (signed PS ally locations) are clamped to 0.
    target: usize,
    /// The signed PS target location the choice used (1/2 foe slot,
    /// -1/-2 own slot, 0 none), for `ps_get_target`.
    target_loc: i8,
    /// PS `fractionalPriority` in tenths (Quick Draw +1, Stall -1, ...).
    fractional: i8,
    /// PS `queue.prioritizeAction` (`order = 3`): sorts before every move.
    prioritized: bool,
    /// A locked move that is not one of the user's own (Copycat -> Dive):
    /// no PP is deducted (PS `deductPP` finds no matching move slot).
    no_pp: bool,
    /// The user's queued action is the `recharge` pseudo-move (no target
    /// type). Kept on the action so it survives the user fainting.
    recharge: bool,
    /// The ability / Grassy Glide priority boost folded into `priority` (PS
    /// `getActionSpeed` `ModifyPriority`). Re-sorts recompute it with the
    /// current ability (Skill Swap / Trace can give Prankster mid-turn).
    boost: i8,
    /// PS `action.originalTarget` (`resolveAction`: `getAtLoc(targetLoc)` when
    /// the action is queued), as (side, team index). Read by `tracksTarget`.
    orig_target: Option<(Side, PokemonIndex)>,
}

/// PS sim/battle-queue.ts resolveAction: `originalTarget = pokemon.getAtLoc(targetLoc)`.
fn orig_target_at(state: &DoublesState, side: Side, loc: i8) -> Option<(Side, PokemonIndex)> {
    if loc == 0 {
        return None;
    }
    let (ts, tsl) = if loc > 0 { (side.other(), (loc - 1) as usize) } else { (side, (-loc - 1) as usize) };
    if tsl >= 2 {
        return None;
    }
    state.side(ts).active()[tsl].pokemon_index.map(|i| (ts, i))
}

impl Action {
    fn key(&self) -> (i8, i16) {
        (self.priority, self.speed)
    }
}

/// pokemon-showdown's `prng.shuffle(items, start, end)`.
fn shuffle<T>(items: &mut [T], start: usize, end: usize, rng: &mut Gen5Rng, label: &'static str) {
    let mut i = start;
    while i < end - 1 {
        let next_index = rng.label(label).random_range(i, end);
        if i != next_index {
            items.swap(i, next_index);
        }
        i += 1;
    }
}

/// pokemon-showdown's `speedSort` with `comparePriority`.
///
/// `compare(a, b) > 0` means *a comes after b*. For moves this reduces to
/// (b.priority - a.priority) || (b.speed - a.speed).
fn speed_sort(actions: &mut [Action], rng: &mut Gen5Rng) {
    speed_sort_by(actions, rng, |a: &Action, b: &Action| -> i32 {
        if a.prioritized != b.prioritized {
            return if a.prioritized { -1 } else { 1 };
        }
        let dp = (b.priority - a.priority) as i32;
        if dp != 0 {
            return dp;
        }
        let df = (b.fractional - a.fractional) as i32;
        if df != 0 {
            return df;
        }
        (b.speed - a.speed) as i32
    });
}

/// PS `speedSort` (selection sort; each group of ties is shuffled), generic over
/// the items. `compare(a, b) > 0` means `a` comes after `b`.
fn speed_sort_by<T>(items: &mut [T], rng: &mut Gen5Rng, compare: impl Fn(&T, &T) -> i32) {
    if items.len() < 2 {
        return;
    }
    let mut sorted = 0usize;
    let mut next_indexes: smallvec::SmallVec<[usize; 8]> = smallvec::SmallVec::new();
    while sorted + 1 < items.len() {
        next_indexes.clear();
        next_indexes.push(sorted);
        for i in (sorted + 1)..items.len() {
            let delta = compare(&items[next_indexes[0]], &items[i]);
            if delta < 0 {
                continue;
            } else if delta > 0 {
                next_indexes.clear();
                next_indexes.push(i);
            } else {
                next_indexes.push(i);
            }
        }
        for (i, &index) in next_indexes.iter().enumerate() {
            if index != sorted + i {
                items.swap(sorted + i, index);
            }
        }
        if next_indexes.len() > 1 {
            shuffle(items, sorted, sorted + next_indexes.len(), rng, "speed tie (move order)");
        }
        sorted += next_indexes.len();
    }
}

/// One entry of the turn's queue at `commitChoices`: a move, or one of the
/// actions PS's `resolveAction` queues ahead of it.
#[derive(Clone, Copy)]
struct QueueSlot {
    /// Index into the turn's actions; `None` for a pseudo-action.
    action: Option<usize>,
    /// PS `action.order`: 3 prioritized, 5 `beforeTurnMove`, 107
    /// `priorityChargeMove`, 200 a move.
    order: i32,
    priority: i32,
    fractional: i32,
    speed: i32,
}

/// PS `comparePriority` for queued actions (order, priority, speed).
fn queue_slot_compare(a: &QueueSlot, b: &QueueSlot) -> i32 {
    if a.order != b.order {
        return a.order - b.order;
    }
    let dp = b.priority - a.priority;
    if dp != 0 {
        return dp;
    }
    let df = b.fractional - a.fractional;
    if df != 0 {
        return df;
    }
    b.speed - a.speed
}

/// The `commitChoices` queue sort (PS `battle.ts` `commitChoices` -> `queue.sort`).
/// PS `resolveAction` queues a `beforeTurnMove` (Counter / Mirror Coat, order
/// 5) and a `priorityChargeMove` (Focus Punch / Beak Blast, order 107) ahead of
/// the move. They sort with the moves, so their swaps in the selection sort
/// reorder the speed-tied moves before the tie shuffle runs.
fn commit_queue_sort(actions: &mut Vec<Action>, state: &DoublesState, rng: &mut Gen5Rng) {
    let mut slots: smallvec::SmallVec<[QueueSlot; 8]> = smallvec::SmallVec::new();
    for (i, a) in actions.iter().enumerate() {
        let live = !state.side(a.side).active()[a.slot].recharging;
        let pseudo_order = match a.choice.move_id {
            Choices::COUNTER | Choices::MIRRORCOAT if live => Some(5),
            Choices::FOCUSPUNCH | Choices::BEAKBLAST | Choices::CHILLYRECEPTION if live => Some(107),
            _ => None,
        };
        if let Some(order) = pseudo_order {
            slots.push(QueueSlot { action: None, order, priority: 0, fractional: 0, speed: a.speed as i32 });
        }
        slots.push(QueueSlot {
            action: Some(i),
            order: if a.prioritized { 3 } else { 200 },
            priority: a.priority as i32,
            fractional: a.fractional as i32,
            speed: a.speed as i32,
        });
    }
    speed_sort_by(&mut slots, rng, queue_slot_compare);
    let mut taken: smallvec::SmallVec<[Option<Action>; 4]> = actions.drain(..).map(Some).collect();
    actions.extend(slots.iter().filter_map(|s| s.action).map(|i| taken[i].take().unwrap()));
}

/// Move names as `Choices` prints them (`{}`), indexed by `Choices`.
static MOVE_NAMES: std::sync::LazyLock<Vec<&'static str>> = std::sync::LazyLock::new(|| {
    (0..=u16::MAX)
        .map(Choices::from)
        .enumerate()
        .take_while(|&(i, m)| m as usize == i)
        .map(|(_, m)| &*Box::leak(m.to_string().into_boxed_str()))
        .collect()
});

/// A move-order entry (fuzz snapshot); formatted only when serialized.
fn order_entry(side: Side, slot: usize, species: &'static str, m: Choices) -> OrderEntry {
    OrderEntry { side, slot: slot as u8, species, mv: m }
}

fn side_tag(side: Side) -> &'static str {
    match side {
        Side::One => "A",
        Side::Two => "B",
    }
}

/// Effective speed for ordering, matching pokemon-showdown's `calculateStat`
/// chain: base speed -> stat boosts -> paralysis halving. Trick Room negates
/// the result (champions `getActionSpeed` override), which is equivalent to
/// reversing the speed comparison.
pub fn effective_speed(state: &DoublesState, side: Side, slot: usize) -> Option<i16> {
    effective_speed_in(state, side, slot, effective_weather(state))
}

/// `effective_speed` with the effective weather already known (callers that
/// compute several speeds read it once).
fn effective_speed_in(state: &DoublesState, side: Side, slot: usize, weather: Weather) -> Option<i16> {
    let s = state.side(side);
    let idx = s.active()[slot].pokemon_index?;
    let p = s.team_member(idx);
    let boosted = multiply_boost(s.team_boosts[idx as usize].spe, p.speed);
    // PS `getStat('spe')`: `runEvent('ModifySpe')` chains the modifiers (tailwind
    // x2, speed abilities, Choice Scarf), `finalModify` applies them once; then
    // paralysis (priority -101) halves, unless Quick Feet.
    // TODO(PS parity): Unburden, Slow Start, Iron Ball, Quick Powder, ...
    // `weather` is PS `effectiveWeather`: none under Cloud Nine / Air Lock.
    let mut m: i64 = 4096;
    let mut chain = |x: i64| m = (m * x + 2048) >> 12;
    if s.side_conditions.tailwind > 0 {
        chain(8192);
    }
    let doubled = match p.ability {
        Abilities::SWIFTSWIM => matches!(weather, Weather::RAIN | Weather::HEAVYRAIN),
        Abilities::CHLOROPHYLL => matches!(weather, Weather::SUN | Weather::HARSHSUN),
        Abilities::SANDRUSH => weather == Weather::SAND,
        Abilities::SLUSHRUSH => matches!(weather, Weather::HAIL | Weather::SNOW),
        Abilities::SURGESURFER => state.terrain.terrain_type == crate::engine::state::Terrain::ELECTRICTERRAIN,
        _ => false,
    };
    if doubled {
        chain(8192);
    }
    if p.ability == Abilities::QUICKFEET && p.status != PokemonStatus::NONE {
        chain(6144);
    }
    if p.item == crate::engine::items::Items::CHOICESCARF {
        chain(6144);
    }
    let mut speed = ((boosted as i64 * m + 2047) / 4096) as i16;
    if p.status == PokemonStatus::PARALYZE && p.ability != Abilities::QUICKFEET {
        speed = speed / 2;
    }
    if state.trick_room.active {
        Some(-speed)
    } else {
        Some(speed)
    }
}

/// PS `removeLinkedVolatiles` when `(side, idx)` leaves the field (faint or
/// `clearVolatile` on switch-out): every slot whose `trapped` link it holds
/// loses `trapped` (Block / Mean Look / Jaw Lock). No draw.
fn release_trapped_by(state: &mut DoublesState, side: Side, idx: PokemonIndex) {
    for s in [Side::One, Side::Two] {
        for sl in 0..2 {
            if state.side(s).active()[sl].trapper != Some((side, idx)) {
                continue;
            }
            let a = state.side_mut(s).active_slot_mut(sl);
            if a.trapper == Some((side, idx)) {
                a.trapped = false;
                a.trapper = None;
            }
        }
    }
}

/// Remove a pokemon from its active slot, clearing per-pokemon state the way
/// pokemon-showdown does on faint (boosts cleared; the slot stays empty until
/// the end-of-turn switch).
fn faint_pokemon(state: &mut DoublesState, side: Side, slot: usize) {
    if let Some(idx) = state.side(side).active()[slot].pokemon_index {
        state.unresolved_faints.push((side, slot, idx));
        state.faint_log.push(side);
        let disabled = state.side(side).active()[slot].disabled;
        state.faint_disabled.push((side, idx, disabled));
        release_trapped_by(state, side, idx);
    }
    // Destiny Bond `onAfterFaint`: the last attacker also faints.
    // PS data/conditions.ts destinybond.onAfterFaint.
    let db = state.side(side).active()[slot].destinybond && state.in_move;
    let la = state.side(side).active()[slot].last_attacker;
    // PS `onFaint`: `target.isAlly(source)` returns (an ally's or the user's own
    // damage does not take the attacker down).
    if db {
        if let Some((att_side, att_idx)) = la.filter(|&(a, _)| a != side) {
            if let Some(att_slot) = (0..2).find(|&sl| {
                state.side(att_side).active()[sl].pokemon_index == Some(att_idx)
            }) {
                if state.side(att_side).team_member(att_idx).hp > 0 {
                    let lock = state.side(att_side).active()[att_slot].locked_move;
                    state.db_alive_pending.push((att_side, att_idx, lock));
                    state.side_mut(att_side).team_member_mut(att_idx).hp = 0;
                    faint_pokemon(state, att_side, att_slot);
                }
            }
        }
    }    let s = state.side_mut(side);
    if let Some(idx) = s.active()[slot].pokemon_index {
        s.clear_member_boosts(idx);
        if let Some(t) = s.active_slot_mut(slot).roost_orig.take() {
            s.team_member_mut(idx).types = t;
        }
        if let Some(t) = s.active()[slot].type_change {
            s.team_member_mut(idx).types = t;
        }
        if s.active()[slot].added_grass {
            s.team_member_mut(idx).types.1 = PokemonType::TYPELESS;
        }
        // The status stays until PS `checkFainted` (end of turn, only for a
        // side that can still switch) sets it to 'fnt'; see `check_fainted`.
        // PS keeps the fainted pokemon in the slot until its replacement
        // switches in; remember it (plus its stall slot condition) so the
        // Residual sort still lists its handlers.
        s.fainted_in_slot[slot] = Some((idx, s.active()[slot].stall));
        *s.active_slot_mut(slot) = ActiveSlot::empty();
    }
}

/// PS `faintMessages`: resolve the pending faints that `pick` selects. Until
/// then a fainted pokemon keeps its cached speed (it still counts in
/// `eachEvent` sorts). `clearVolatile` -> `setSpecies` then resets
/// `pokemon.speed` to the raw stat, which later sorts read.
fn resolve_faints(state: &mut DoublesState, rng: &mut Gen5Rng, pick: impl Fn(Side, usize) -> bool) {
    let faints = std::mem::take(&mut state.unresolved_faints);
    let mut resolved = Vec::new();
    let keep: Vec<(Side, usize, PokemonIndex)> = faints.iter().copied().filter(|&(s, sl, _)| !pick(s, sl)).collect();
    let picked: Vec<(Side, usize, PokemonIndex)> = faints.iter().copied().filter(|&(s, sl, _)| pick(s, sl)).collect();
    for (k, &(side, slot, idx)) in picked.iter().enumerate() {
        // PS `resolveFaints` -> `singleEvent('End', ability)`: a Cloud Nine / Air
        // Lock holder's `onEnd` runs `eachEvent('WeatherChange')` while the holder
        // and the faints still queued behind it count as actives (`getAllActive`).
        // Only outside a move: a move's KO already drew this sort after its hit.
        if !state.in_move
            && matches!(state.side(side).team_member(idx).ability, Abilities::CLOUDNINE | Abilities::AIRLOCK)
        {
            let mut view: Vec<(Side, usize, PokemonIndex)> = picked[k..].to_vec();
            view.extend(keep.iter().copied());
            state.unresolved_faints = view;
            each_event_update(state, rng);
            state.unresolved_faints = Vec::new();
        }
        let s = state.side_mut(side);
        // PS `clearVolatile` -> `setSpecies(baseSpecies)` restores the base
        // `storedStats`, so a Speed Swap (or Guard / Power Split) is undone
        // when the pokemon faints.
        restore_stored_stats(s, idx);
        s.speed_cache[idx as usize] = s.team_member(idx).speed;
        resolved.push((side, idx));
    }
    // Still-queued faints of other slots remain unresolved.
    state.unresolved_faints = keep;
    state.db_alive_pending.retain(|e| !resolved.contains(&(e.0, e.1)));
    let batch = resolved.len();
    // PS `Faint` event: `onAllyFaint` of the fainted pokemon's active allies.
    for &(side, idx) in &resolved {
        ally_faint_copy(state, rng, side, idx);
    }
    // PS faintMessages -> `clearVolatile`: the ability goes back to the base
    // one (an Entrainment / Skill Swap copy is lost; a Revival Blessing
    // revive comes back with its own). Runs after the Faint event, which
    // Receiver reads. PS sim/battle.ts faintMessages, sim/pokemon.ts clearVolatile.
    for &(side, idx) in &resolved {
        let p = state.side_mut(side).team_member_mut(idx);
        p.ability = p.base_ability;
    }
    // PS `resolveFaints`: `checkWin` runs before `AfterFaint`, so a batch that
    // ends the battle boosts nobody. Otherwise `onSourceAfterFaint` runs once
    // with `length` = the batch size (PS sim/battle.ts resolveFaints,
    // data/abilities.ts moxie / chillingneigh / grimneigh).
    if batch > 0 {
        if let Some((side, slot, idx, _)) = state.pending_after_faint.take() {
            if state.battle_is_over() == 0.0
                && state.side(side).active()[slot].pokemon_index == Some(idx)
                && state.side(side).slot_pokemon(slot).map(|p| p.hp > 0).unwrap_or(false)
            {
                // PS `runEvent('SourceAfterFaint')` reads the attacker's ability
                // when the faint is resolved: a Skill Swap / Wandering Spirit
                // during the same hit has already changed it.
                let ab = state.side(side).team_member(idx).ability;
                let n = batch as i8;
                let b = match ab {
                    Abilities::MOXIE | Abilities::CHILLINGNEIGH => StatBoosts { attack: n, ..Default::default() },
                    Abilities::GRIMNEIGH => StatBoosts { special_attack: n, ..Default::default() },
                    _ => StatBoosts::default(),
                };
                apply_boosts(state, side, idx, &b, false);
            }
        }
    }
}

/// Receiver / Power of Alchemy `onAllyFaint` (PS data/abilities.ts receiver,
/// powerofalchemy): an active ally with hp copies the fainted pokemon's
/// ability through `setAbility`, then its Start effects run. Abilities with
/// the `noreceiver` flag (the same set as `notrace`) cannot be copied.
fn ally_faint_copy(state: &mut DoublesState, rng: &mut Gen5Rng, side: Side, fainted_idx: PokemonIndex) {
    let mut ability = real_ability(state, side, fainted_idx);
    // Receiver reads `target.getAbility()`, which ignores Gastro Acid's
    // suppression; the engine blanks a Gastro Acid'd ability to NONE, so take
    // the stored one back (PS data/abilities.ts receiver, data/moves.ts gastroacid).
    // TODO(PS parity): an ability changed (Skill Swap, ...) before Gastro Acid.
    if ability == Abilities::NONE {
        ability = state.side(side).team_member(fainted_idx).base_ability;
    }
    if has_notrace(ability) || ability == Abilities::NOABILITY {
        return;
    }
    for slot in 0..2 {
        let Some(hidx) = state.side(side).active()[slot].pokemon_index else { continue };
        if hidx == fainted_idx {
            continue;
        }
        let holder = state.side(side).team_member(hidx);
        if holder.hp <= 0 || !matches!(holder.ability, Abilities::RECEIVER | Abilities::POWEROFALCHEMY) {
            continue;
        }
        if crate::engine::mechanics::cant_suppress(ability) {
            continue;
        }
        state.side_mut(side).team_member_mut(hidx).ability = ability;
        ability_update_cures(state, side, slot);
        // The Start event's `eachEvent('WeatherChange' / 'TerrainChange')` (Cloud
        // Nine, Drought, ...) is a sort of all actives, as for a switch-in.
        if apply_on_switch_abilities(state, side, slot, rng) {
            let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
        }
    }
}

/// PS `battle.directDamage` (Belly Drum, Clangorous Soul, Curse, Shed Tail,
/// Struggle recoil): like `apply_damage`, but it never sets `hurtThisTurn`
/// (only `spreadDamage` does, sim/battle.ts), so Assurance does not double.
fn apply_direct_damage(state: &mut DoublesState, side: Side, slot: usize, amount: i16) -> bool {
    let hurt = state.side(side).active()[slot].hurt_this_turn;
    let hurt_hp = state.side(side).active()[slot].hurt_hp;
    let fainted = apply_damage(state, side, slot, amount);
    state.side_mut(side).slot_scratch_mut(slot).hurt_this_turn = hurt;
    state.side_mut(side).slot_scratch_mut(slot).hurt_hp = hurt_hp;
    fainted
}

/// Apply damage to a specific slot; returns true if the pokemon fainted.
fn apply_damage(state: &mut DoublesState, side: Side, slot: usize, amount: i16) -> bool {
    if amount <= 0 {
        return false;
    }
    let s = state.side_mut(side);
    if let Some(idx) = s.active()[slot].pokemon_index {
        let p = s.team_member_mut(idx);
        p.hp -= amount;
        let left = p.hp.max(0);
        s.slot_scratch_mut(slot).hurt_this_turn = true;
        s.slot_scratch_mut(slot).hurt_hp = left;
        let p = s.team_member_mut(idx);
        if p.hp <= 0 {
            p.hp = 0;
            faint_pokemon(state, side, slot);
            return true;
        }
    }
    false
}

fn hp_snapshot(state: &DoublesState) -> TurnHp {
    let snap = |side: &crate::doubles::state::DoublesSide| {
        (0..2)
            .map(|slot| {
                side.slot_pokemon(slot).map(|p| SlotHp {
                    name: p.id.to_string(),
                    hp: p.hp,
                    maxhp: p.maxhp,
                    fainted: p.hp <= 0,
                })
            })
            .collect()
    };
    TurnHp {
        a: snap(&state.side_one),
        b: snap(&state.side_two),
    }
}

/// After a mid-turn switch-in, pokemon-showdown re-sorts ALL active Pokémon by
/// speed (`runSwitch` -> `speedSort(allActive)`), shuffling speed ties. This
/// consumes PRNG draws even though it does not change the move order, so we must
/// replicate it to keep the draw sequence aligned.
fn resort_all_active(state: &DoublesState, rng: &mut Gen5Rng) {
    // PS sim/battle-actions.ts runSwitch: `speedSort(getAllActive(true))`.
    // The active slots are fixed by switch-in order; only the draws matter here.
    let _ = sorted_actives(state, rng, "speed tie (switch-in)");
}

/// Switch a replacement into a fainted slot at the end of the turn, and
/// replicate pokemon-showdown's `runSwitch` speed re-sort of all actives
/// (which consumes PRNG draws for speed ties).
fn process_switch(state: &mut DoublesState, rng: &mut Gen5Rng, side: Side, slot: usize) {
    {
        let s = state.side_mut(side);
        *s.active_slot_mut(slot) = ActiveSlot::empty();
        if let Some(repl) = s.find_replacement() {
            s.active_slot_mut(slot).pokemon_index = Some(repl);
        }
    }
    resort_all_active(state, rng);
}

/// Choose the target slot on the opposing side: the first alive active, else
/// the other. Mirrors the harness policy of targeting foe location 1.
fn choose_target(state: &DoublesState, defender_side: Side) -> usize {
    let s = state.side(defender_side);
    if s.slot_pokemon(0).map(|p| p.hp > 0).unwrap_or(false) {
        0
    } else if s.slot_pokemon(1).map(|p| p.hp > 0).unwrap_or(false) {
        1
    } else {
        0
    }
}

fn active_move(state: &DoublesState, side: Side, slot: usize) -> Option<Choice> {
    let s = state.side(side);
    let idx = s.active()[slot].pokemon_index?;
    let p = s.team_member(idx);
    // Policy: always use move slot 0 (matches the harness).
    let mv = &p.moves[&PokemonMoveIndex::M0];
    if mv.id == crate::choices::Choices::NONE {
        return None;
    }
    Some(mv.choice.clone())
}

/// Snapshot the full-team HP (all real team members, excluding empty slots) for
/// both sides at a given turn.
fn snapshot_full(state: &DoublesState, turn: u32) -> ReplayTurnHp {
    let snap_side = |s: &DoublesSide| -> Vec<SlotHp> {
        s.team
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                !s.species_names[*i].is_empty()
                    || s.team[*i].id != crate::pokemon::PokemonName::NONE
            })
            .map(|(i, p)| SlotHp {
                name: s.species_names[i].to_string(),
                hp: p.hp,
                maxhp: p.maxhp,
                fainted: p.hp <= 0,
            })
            .collect()
    };
    ReplayTurnHp {
        turn,
        a: snap_side(&state.side_one),
        b: snap_side(&state.side_two),
    }
}

/// Replay a battle using ground-truth per-move damage from pokemon-showdown.
///
/// This validates the engine's *battle loop* — faints, switch-ins, and winner
/// determination — without depending on the PRNG draw ordering. Damage values
/// are taken from the provided move list (PS ground truth), applied by name
/// (preferring the Pokémon currently in the target slot).
pub fn run_battle_replay(state: &mut DoublesState, moves: &[ReplayMove]) -> BattleResult {
    let to_side = |s: &str| -> Side {
        if s == "A" {
            Side::One
        } else {
            Side::Two
        }
    };

    // Ensure the attacker occupies its slot (handles a fresh switch-in), so the
    // slot occupancy stays consistent with the move sequence.
    let mut ensure_attacker = |state: &mut DoublesState, mv: &ReplayMove| {
        let att_side = to_side(&mv.att_side);
        let s = state.side_mut(att_side);
        if s.active()[mv.att_slot].pokemon_index.is_some() {
            return;
        }
        let name = mv.att.to_lowercase();
        let mut found = None;
        for pos in 0..6 {
            let idx = PokemonIndex::deserialize(&pos.to_string());
            let p = s.team[pos].clone();
            let is_active = s.active().iter().any(|sl| sl.pokemon_index == Some(idx));
            if p.hp > 0 && !is_active && p.id.to_string().to_lowercase() == name {
                found = Some(idx);
                break;
            }
        }
        if found.is_none() {
            found = s.find_replacement();
        }
        if let Some(idx) = found {
            s.active_slot_mut(mv.att_slot).pokemon_index = Some(idx);
        }
    };

    let mut replay_hp: Vec<ReplayTurnHp> = Vec::new();
    let mut last_turn: Option<usize> = None;
    for mv in moves {
        // Snapshot the previous turn's full-team HP when the turn advances.
        if let Some(t) = mv.turn {
            if last_turn != Some(t) {
                if let Some(pt) = last_turn {
                    replay_hp.push(snapshot_full(state, pt as u32));
                }
                last_turn = Some(t);
            }
        }

        let tgt_side = to_side(&mv.tgt_side);
        ensure_attacker(state, mv);

        // Handle Transform (Ditto): update the species name so subsequent hits
        // (which target the new name) line up.
        if let Some(old_name) = &mv.transform_from {
            let norm = |x: &str| x.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect::<String>();
            let want = norm(old_name);
            let s = state.side_mut(tgt_side);
            let new_name = mv.tgt.clone();
            for i in 0..6 {
                let n = norm(&s.species_names[i]);
                if !n.is_empty() && (n == want || n.starts_with(&want) || want.starts_with(&n)) {
                    s.species_names[i] = crate::doubles::state::intern(&new_name);
                    break;
                }
            }
        }

        // Identify the target: the Pokémon named `mv.tgt` (PS only ever hits an
        // alive Pokémon, so this is always the correct target). Match by species
        // name (with formes) so e.g. "Decidueye-Hisui" / "Castform-Sunny" line up.
        let target_idx = {
            let s = state.side(tgt_side);
            let norm = |x: &str| x.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect::<String>();
            let want = norm(&mv.tgt);
            let matches = |sp: &str| {
                let n = norm(sp);
                !n.is_empty() && (n.starts_with(&want) || want.starts_with(&n))
            };
            (0..6)
                .map(|pos| PokemonIndex::deserialize(&pos.to_string()))
                .find(|&idx| {
                    // Heals (incl. revives) may target a fainted Pokémon; damage
                    // and self-faints require it to be alive.
                    let alive_ok = if mv.damage < 0 { true } else { s.team[idx as usize].hp > 0 };
                    alive_ok && matches(&s.species_names[idx as usize])
                })
        };

        if let Some(tgt_idx) = target_idx {
            let s = state.side_mut(tgt_side);
            let p = s.team_member_mut(tgt_idx);
            if mv.self_faint {
                // A self-faint (Final Gambit, Self-Destruct, ...): faint the user.
                p.hp = 0;
                for sl in s.active_mut().iter_mut() {
                    if sl.pokemon_index == Some(tgt_idx) {
                        *sl = ActiveSlot::empty();
                    }
                }
            } else if mv.damage < 0 {
                // A heal: add HP (capped at maxhp).
                p.hp = (p.hp - mv.damage).min(p.maxhp);
            } else {
                p.hp -= mv.damage;
                if p.hp <= 0 {
                    p.hp = 0;
                    // clear any slot holding this pokemon
                    for sl in s.active_mut().iter_mut() {
                        if sl.pokemon_index == Some(tgt_idx) {
                            *sl = ActiveSlot::empty();
                        }
                    }
                }
            }
        }
    }

    // Snapshot the final turn.
    if let Some(pt) = last_turn {
        replay_hp.push(snapshot_full(state, pt as u32));
    }

    let winner = match state.battle_is_over() {
        1.0 => "A",
        -1.0 => "B",
        _ => "draw",
    };
    BattleResult {
        winner: winner.to_string(),
        total_turns: state.turn,
        prng_draws: Vec::new(),
        turns: Vec::new(),
        replay_hp,
    }
}

/// The Pokémon's move at the given 0-based slot.
fn move_by_index(
    p: &crate::state::Pokemon,
    i: usize,
) -> &crate::state::Move {
    match i {
        0 => &p.moves.m0,
        1 => &p.moves.m1,
        2 => &p.moves.m2,
        _ => &p.moves.m3,
    }
}

/// The champions ("dubious") mod changes some moves' category / base power
/// relative to the standard table. Return the mod's category for moves that
/// differ; `None` when the move is unchanged.
#[cfg(feature = "champions")]
fn champions_move_category(move_id: Choices) -> Option<MoveCategory> {
    match move_id {
        // featherdance: a pure Status move (basePower 0) in the mod.
        Choices::FEATHERDANCE => Some(MoveCategory::Status),
        _ => None,
    }
}

#[cfg(not(feature = "champions"))]
fn champions_move_category(_move_id: Choices) -> Option<MoveCategory> {
    None
}

/// PS `move.ignoreEvasion`: Keen Eye / Illuminate / Mind's Eye
/// (`onModifyMove`), Chip Away, Darkest Lariat, Sacred Sword. The attacker's
/// own Unaware (`unaware` `onAnyModifyBoost`, data/abilities.ts: the target's
/// evasion is zeroed when `activePokemon` holds it) also reads the target's
/// evasion as 0 in `hitStepAccuracy`.
/// The user's ability for `ignores_evasion`: Unaware works through
/// `onAnyModifyBoost`, which PS collects only from foes / allies with HP
/// (`side.allies()` / `foes()` drop `!hp`), so a user already fainted by
/// `selfdestruct: 'always'` (Explosion, Misty Explosion) loses it.
fn evasion_ability(state: &DoublesState, side: Side, idx: PokemonIndex, mv: Choices) -> Abilities {
    let p = state.side(side).team_member(idx);
    if p.ability == Abilities::UNAWARE && (p.hp <= 0 || is_selfdestruct_always(mv)) {
        Abilities::NONE
    } else {
        p.ability
    }
}

/// PS `move.selfdestruct === 'always'`: `useMoveInner` faints the user before
/// the accuracy and damage steps (data/moves.ts explosion, selfdestruct,
/// mistyexplosion).
fn is_selfdestruct_always(mv: Choices) -> bool {
    matches!(mv, Choices::EXPLOSION | Choices::MISTYEXPLOSION | Choices::SELFDESTRUCT)
}

fn ignores_evasion(ability: Abilities, mv: Choices) -> bool {
    matches!(ability, Abilities::KEENEYE | Abilities::ILLUMINATE | Abilities::MINDSEYE | Abilities::UNAWARE)
        || matches!(
            mv,
            Choices::CHIPAWAY | Choices::DARKESTLARIAT | Choices::SACREDSWORD
        )
}

/// A queue change planned by a move (Encore `onStart`): the queue indices are
/// relative to the queue the move saw (`actions[i..]`). The PRNG draws were
/// taken when it was planned; the battle loop applies the removal and insert.
#[derive(Debug, Clone)]
pub(crate) struct QueuePlan {
    remove: usize,
    insert: usize,
    action: Action,
}

/// PS `queue.changeAction` (Encore): replace the pending move of the pokemon
/// in `(side, slot)` with `mv` unless it already queues `mv`. Plans the
/// change against `queue` (draws now, as PS does inside the move).
/// PS sim/battle-queue.ts changeAction / insertChoice / resolveAction.
fn plan_queue_change(
    state: &mut DoublesState,
    rng: &mut Gen5Rng,
    queue: &[Action],
    side: Side,
    slot: usize,
    mv: Choices,
) -> Option<QueuePlan> {
    let idx = state.side(side).active()[slot].pokemon_index?;
    let j = queue.iter().position(|a| a.side == side && a.pokemon_index == idx)?;
    // A recharging pokemon's queued action is the `recharge` pseudo-move, which is
    // not the locked move: PS `changeAction` replaces it (PS sim/pokemon.ts getMoves).
    if queue[j].choice.move_id == mv && !queue[j].recharge {
        return None;
    }
    let p = state.side(side).team_member(idx);
    let mi = [&p.moves.m0, &p.moves.m1, &p.moves.m2, &p.moves.m3]
        .iter()
        .position(|m| m.id == mv)
        .unwrap_or(0);
    let mut choice = move_by_index(p, mi).choice.clone();
    choice.move_index = match mi {
        0 => crate::state::PokemonMoveIndex::M0,
        1 => crate::state::PokemonMoveIndex::M1,
        2 => crate::state::PokemonMoveIndex::M2,
        _ => crate::state::PokemonMoveIndex::M3,
    };
    if let Some(cat) = champions_move_category(choice.move_id) {
        choice.category = cat;
        choice.base_power = 0.0;
    }
    let boost = ability_priority_boost(p, &choice) + grassy_glide_boost(state, p, &choice);
    update_speed(state);
    let mut a = Action {
        side,
        slot,
        pokemon_index: idx,
        boost,
        priority: choice.priority + boost,
        choice,
        speed: effective_speed(state, side, slot).unwrap_or(0),
        target: 0,
        target_loc: 0,
        fractional: 0,
        prioritized: false,
        no_pp: false,
        recharge: false,
        orig_target: None,
    };
    // `resolveAction`: no target location -> `getRandomTarget` (a draw), then
    // `getActionSpeed` -> `getTarget`. The new action is a real `move` (not the
    // recharge pseudo-move, even when the target is recharging), so it resolves
    // once, and the target location it gets is kept for later re-sorts.
    // A move with `beforeTurnCallback` (Counter, Mirror Coat, Pursuit) first
    // queues a `beforeTurnMove` resolved with no target location (a draw of
    // its own), then the move itself draws again (PS `resolveAction`, same
    // as the turn-start path above).
    if matches!(a.choice.move_id, Choices::COUNTER | Choices::MIRRORCOAT | Choices::PURSUIT) {
        let mut charge = a.clone();
        charge.target_loc = 0;
        let _ = ps_get_target_queue_insert(state, rng, &charge);
    }
    // A move with `priorityChargeCallback` (Focus Punch, Beak Blast, Chilly
    // Reception) first queues a `priorityChargeMove` resolved with no target
    // location (PS sim/battle-queue.ts resolveAction): one draw of its own,
    // then the move draws again. The fuzzer labels that second draw "queue
    // insert" because its stack runs under insertChoice.
    if matches!(a.choice.move_id, Choices::FOCUSPUNCH | Choices::BEAKBLAST | Choices::CHILLYRECEPTION) {
        let mut charge = a.clone();
        charge.target_loc = 0;
        let _ = ps_get_target_queue_insert(state, rng, &charge);
    }
    // `resolveAction`: `runEvent('FractionalPriority')` (Quick Draw rolls here),
    // before the random target. Same rule as the turn-start path.
    {
        let p = state.side(side).team_member(idx);
        a.fractional = match crate::engine::mechanics::fractional_priority(p.ability, a.choice.category) {
            Some(crate::engine::mechanics::FractionalPriority::Fixed(f)) => f,
            Some(crate::engine::mechanics::FractionalPriority::QuickDraw) => {
                if rng.label("fractional priority").random_chance(3, 10) {
                    1
                } else {
                    0
                }
            }
            None => 0,
        };
    }
    if let Some((ts, tsl)) = ps_get_target_queue_insert(state, rng, &a) {
        a.target_loc = if ts == a.side { -(tsl as i8 + 1) } else { tsl as i8 + 1 };
        a.target = tsl;
    }
    a.orig_target = orig_target_at(state, a.side, a.target_loc);
    let _ = ps_get_target_queue_insert(state, rng, &a);
    // `insertChoice`: first / last index among the queue (the target's own
    // action already cancelled) by comparePriority (priority includes the
    // fractional priority: PS `getActionSpeed`).
    let rest: Vec<&Action> = queue.iter().enumerate().filter(|&(k, _)| k != j).map(|(_, x)| x).collect();
    let mut first: Option<usize> = None;
    let mut last: Option<usize> = None;
    for (k, cur) in rest.iter().enumerate() {
        // `comparePriority`: priority (+ fractional priority), then speed.
        let cmp = (cur.priority, cur.fractional, cur.speed).cmp(&(a.priority, a.fractional, a.speed)); // new first when cur is "less"
        // compared(new, cur) <= 0 <=> new sorts before or ties with cur
        if cmp != std::cmp::Ordering::Greater && first.is_none() {
            first = Some(k);
        }
        if cmp == std::cmp::Ordering::Less {
            last = Some(k);
            break;
        }
    }
    let insert = match first {
        None => rest.len(),
        Some(f) => {
            let l = last.unwrap_or(rest.len());
            if f == l { f } else { f + rng.label("speed tie (queue insert)").random((l - f) as u32 + 1) as usize }
        }
    };
    Some(QueuePlan { remove: j, insert, action: a })
}

/// Moves Instruct cannot repeat: PS data/moves.ts (champions mod) flags
/// `failinstruct`, `charge` or `recharge`.
const INSTRUCT_FAIL_IDS: &[Choices] = &[
    Choices::BEAKBLAST, Choices::BELCH, Choices::BLAZINGTORQUE, Choices::CELEBRATE,
    Choices::COMBATTORQUE, Choices::DYNAMAXCANNON, Choices::FREEZESHOCK, Choices::HOLDHANDS,
    Choices::ICEBURN, Choices::KINGSSHIELD, Choices::MAGICALTORQUE, Choices::METEORASSAULT,
    Choices::METRONOME, Choices::MIMIC, Choices::NOXIOUSTORQUE, Choices::OBSTRUCT,
    Choices::PRISMATICLASER, Choices::ROAROFTIME, Choices::ROLLOUT, Choices::SHADOWFORCE,
    Choices::SHELLTRAP, Choices::SKETCH, Choices::WICKEDTORQUE, Choices::ASSIST, Choices::BIDE,
    Choices::BLASTBURN, Choices::BOUNCE, Choices::CHATTER, Choices::COPYCAT, Choices::DIG,
    Choices::DIVE, Choices::ELECTROSHOT, Choices::ETERNABEAM, Choices::FLY, Choices::FOCUSPUNCH,
    Choices::FRENZYPLANT, Choices::GEOMANCY, Choices::GIGAIMPACT, Choices::HYDROCANNON,
    Choices::HYPERBEAM, Choices::ICEBALL, Choices::INSTRUCT, Choices::MEFIRST, Choices::METEORBEAM,
    Choices::MIRRORMOVE, Choices::NATUREPOWER, Choices::OUTRAGE, Choices::PETALDANCE,
    Choices::PHANTOMFORCE, Choices::RAZORWIND, Choices::ROCKWRECKER, Choices::SKULLBASH,
    Choices::SKYATTACK, Choices::SKYDROP, Choices::SLEEPTALK, Choices::SOLARBEAM, Choices::SOLARBLADE,
    Choices::STRUGGLE, Choices::THRASH, Choices::TRANSFORM, Choices::UPROAR,
];

/// The action PS `resolveAction` builds for an Instruct repeat: the target's
/// `lastMove` with its last target location (no draw: the location is set).
/// Prioritized, so it runs next (`prioritizeAction`, order 3).
fn instruct_action(
    state: &DoublesState,
    side: Side,
    slot: usize,
    idx: PokemonIndex,
    mv: Choices,
    loc: i8,
) -> Option<Action> {
    let p = state.side(side).team_member(idx);
    let mi = [&p.moves.m0, &p.moves.m1, &p.moves.m2, &p.moves.m3]
        .iter()
        .position(|m| m.id == mv)?;
    let mut choice = move_by_index(p, mi).choice.clone();
    choice.move_index = match mi {
        0 => crate::state::PokemonMoveIndex::M0,
        1 => crate::state::PokemonMoveIndex::M1,
        2 => crate::state::PokemonMoveIndex::M2,
        _ => crate::state::PokemonMoveIndex::M3,
    };
    if let Some(cat) = champions_move_category(choice.move_id) {
        choice.category = cat;
        choice.base_power = 0.0;
    }
    let boost = ability_priority_boost(p, &choice) + grassy_glide_boost(state, p, &choice);
    Some(Action {
        side,
        slot,
        pokemon_index: idx,
        boost,
        priority: choice.priority + boost,
        choice,
        speed: effective_speed(state, side, slot).unwrap_or(0),
        target: (loc.unsigned_abs() as usize).saturating_sub(1).min(1),
        target_loc: loc,
        fractional: 0,
        prioritized: true,
        no_pp: false,
        recharge: false,
        orig_target: orig_target_at(state, side, loc),
    })
}


/// Applies a planned queue change to `actions[from..]`.
/// PS `setAbility` (Skill Swap, Trace, Role Play, Entrainment, Simple Beam,
/// Worry Seed): the holder's ability effect state is new, so Protean / Libero
/// may change its type again (`protean_done`).
fn reset_protean_effect(state: &mut DoublesState, side: Side, idx: PokemonIndex) {
    for sl in 0..2 {
        if state.side(side).active()[sl].pokemon_index == Some(idx) {
            state.side_mut(side).slot_scratch_mut(sl).protean_done = false;
        }
    }
}

fn apply_queue_plan(actions: &mut Vec<Action>, from: usize, plan: QueuePlan) {
    actions.remove(from + plan.remove);
    actions.insert(from + plan.insert, plan.action);
}

/// Abilities' `ModifyAccuracy` chain pieces: (priority 0, priority -1), each a
/// 4096-based multiplier. Priority 0: Victory Star (`onAnyModifyAccuracy`, an
/// ally of the holder, x4506). Priority -1: Hustle (physical), Compound Eyes
/// (x5325), and the target's Sand Veil / Snow Cloak (x3277 in sand / snow) and
/// Tangled Feet (x0.5 while confused) unless the attacker ignores its ability.
/// PS data/abilities.ts.
/// PS `unaware` `onAnyModifyBoost` (data/abilities.ts): an Unaware target zeroes
/// the attacker's accuracy stage (`ModifyBoost` in `hitStepAccuracy`). Breakable.
fn unaware_zeroes_accuracy(
    state: &DoublesState,
    side: Side,
    idx: PokemonIndex,
    tside: Side,
    tslot: usize,
    choice: &Choice,
) -> bool {
    let attacker = state.side(side).team_member(idx);
    state.side(tside).slot_pokemon(tslot).is_some_and(|d| {
        d.ability == Abilities::UNAWARE && !crate::engine::abilities::ignores_target_ability(attacker, choice, &d.ability)
    })
}

fn accuracy_ability_mods(
    state: &DoublesState,
    side: Side,
    idx: PokemonIndex,
    tside: Side,
    tslot: usize,
    choice: &Choice,
) -> (i32, i32) {
    let chain = |m: i32, x: i32| (m * x + 2048) >> 12;
    let attacker = state.side(side).team_member(idx);
    let mut pre = 4096;
    if (0..2).any(|sl| {
        state.side(side).slot_pokemon(sl).map(|p| p.hp > 0 && p.ability == Abilities::VICTORYSTAR).unwrap_or(false)
    }) {
        pre = chain(pre, 4506);
    }
    let mut post = 4096;
    if attacker.ability == Abilities::HUSTLE && choice.category == MoveCategory::Physical {
        post = chain(post, 3277);
    }
    if attacker.ability == Abilities::COMPOUNDEYES {
        post = chain(post, 5325);
    }
    if let Some(t) = state.side(tside).slot_pokemon(tslot) {
        if !crate::engine::abilities::ignores_target_ability(attacker, choice, &t.ability) {
            let w = effective_weather(state);
            match t.ability {
                Abilities::SANDVEIL if w == Weather::SAND => post = chain(post, 3277),
                Abilities::SNOWCLOAK if matches!(w, Weather::HAIL | Weather::SNOW) => post = chain(post, 3277),
                Abilities::TANGLEDFEET if state.side(tside).active()[tslot].confusion > 0 => post = chain(post, 2048),
                _ => {}
            }
        }
    }
    (pre, post)
}

/// PS `getLastDamagedBy(true)`: the last attacker from the other side that
/// damaged `(side, idx)` — (attacker side, attacker slot, damage, turn).
fn last_damaged_by(state: &DoublesState, side: Side, idx: PokemonIndex) -> Option<(Side, usize, i16, u32, usize)> {
    state.side(side).damaged_by[idx as usize]
        .iter()
        .rev()
        .find(|e| e.0 != side)
        .copied()
}

/// Run a doubles battle driven by explicit per-turn controller choices
/// (fuzz testing). The same choices are fed to pokemon-showdown; the two
/// simulators must produce identical state.
pub fn run_battle_fuzz(
    state: &mut DoublesState,
    rng: &mut Gen5Rng,
    choices: &[FuzzTurn],
    stop_after: Option<u32>,
) -> FuzzBattleResult {
    let mut turn_states: Vec<crate::doubles::state::FuzzTurnState> = Vec::new();
    // On-switch abilities for the leads (Intimidate), in PS send-out order:
    // side A's leads first, then side B's (p1 switches in before p2).
    // PS `runSwitch` for the four leads: speed sort, SwitchIn handlers, then
    // the action's `eachEvent('Update')`.
    if rng.trace {
        rng.context = "turn 0: move order".to_string();
    }
    team_preview_draws(state, rng);
    // Illusion `onBeforeSwitchIn` for the leads (PS data/abilities.ts illusion):
    // the disguise is fixed when the holder is sent out, from the party order
    // at that time (a later switch reorders `side.pokemon`).
    for sd in [Side::One, Side::Two] {
        for sl in 0..2 {
            let s = state.side_mut(sd);
            if let Some(idx) = s.active()[sl].pokemon_index {
                if s.team_member(idx).ability == Abilities::ILLUSION {
                    s.slot_scratch_mut(sl).illusion_target = Some(illusion_disguise(s, idx));
                }
            }
        }
    }
    if rng.trace {
        rng.context = "turn 0: switch-in".to_string();
    }
    // PS `case "start"`: each lead's `switchIn` inserts a `runSwitch` action
    // into the queue (`insertChoice`), drawing on speed ties before the
    // `runSwitch` speed sort below.
    queue_insert_draws(state, rng);
    // PS `runSwitch` -> `fieldEvent('SwitchIn')`: the leads' on-switch handlers
    // (`onStart` abilities such as Sand Stream / Snow Warning) run fastest
    // first, so the slower lead's weather is the one that stays.
    let lead_order = sorted_actives(state, rng, "speed tie (switch-in)");
    for (side, slot) in lead_order {
        if apply_on_switch_abilities(state, side, slot, rng) {
            // Terrain just set (e.g. Grassy Surge): `eachEvent("TerrainChange")`.
            // Labelled switch-in like PS (`runSwitch` is on its stack); no
            // pokemon has fainted yet, so the sort is the same.
            let _ = sorted_actives(state, rng, "speed tie (switch-in)");
        }
        // Forecast `onWeatherChange` (PS data/abilities.ts forecast): a lead's
        // Drought / Drizzle changes Castform's forme during the send-out.
        update_forecast(state);
    }
    each_event_update(state, rng);
    // Forecast `onStart` / `onWeatherChange` (PS data/abilities.ts forecast):
    // a lead's Drizzle / Drought sets the weather, and a Castform lead follows it.
    update_forecast(state);
    // Turn 0: the initial state right after the leads are sent out (stat
    // calc, species data, leads). Compared before any turn is simulated.
    if state.record {
        let mut init = fuzz_snapshot(state, Vec::new());
        init.turn = 0;
        turn_states.push(init);
    }

    let limit = stop_after.unwrap_or(u32::MAX);
    let mut ci = 0usize;
    let mut error: Option<String> = None;

    loop {
        if ci as u32 >= limit {
            break; // stepwise fuzz mode: stop here, not an error
        }
        if state.battle_is_over() != 0.0 {
            break;
        }

        let turn_no = state.turn;
        if crate::doubles::debug::flags().dmg {
            eprintln!("=== TURN index {} (state.turn {})", ci + 1, turn_no);
        }
        let turn = match choices.get(ci) {
            Some(t) => t,
            None => {
                error = Some(format!(
                    "choices exhausted at turn {} but the battle is not over",
                    turn_no
                ));
                break;
            }
        };

        // Build actions and pending switches from the controller's choices.
        let mut actions: Vec<Action> = Vec::new();
        let mut pending_switches: Vec<(Side, usize, usize)> = Vec::new();
        for (side, acts) in [(Side::One, &turn.a), (Side::Two, &turn.b)] {
            for slot in 0..2 {
                let act = match acts.get(slot) {
                    Some(a) => a,
                    None => {
                        error = Some(format!(
                            "missing action for side {} slot {} at turn {}",
                            side_tag(side),
                            slot,
                            turn_no
                        ));
                        break;
                    }
                };
                let s = state.side(side);
                match act {
                    FuzzAction::Pass => {}
                    FuzzAction::Move {
                        slot: mi,
                        target,
                        struggle: struggle_flag,
                    } => {
                        // PS `side.chooseMove` (`!moves.length`): a request with no
                        // enabled move is Struggle, whatever move was sent.
                        let struggle_now = *struggle_flag
                            || s.active()[slot].pokemon_index.is_some_and(|i| no_usable_moves(state, side, slot, i));
                        let struggle = &struggle_now;
                        if let Some(idx) = s.active()[slot].pokemon_index {
                            let p = s.team_member(idx);
                            let (mv, prio) = if *struggle {
                                // Struggle (forced when no usable moves left).
                                // PS data/moves.ts struggle.onModifyMove: `move.type = '???'`
                                // (typeless: no STAB, no type immunity).
                                let mut m = MOVES.get(&crate::choices::Choices::STRUGGLE).cloned().unwrap();
                                m.move_type = PokemonType::TYPELESS;
                                (m, 0i8)
                            } else {
                                let m = move_by_index(p, (*mi).min(3));
                                let mut choice = m.choice.clone();
                                // The slot the move was chosen from (PP / `used`).
                                choice.move_index = match (*mi).min(3) {
                                    0 => crate::state::PokemonMoveIndex::M0,
                                    1 => crate::state::PokemonMoveIndex::M1,
                                    2 => crate::state::PokemonMoveIndex::M2,
                                    _ => crate::state::PokemonMoveIndex::M3,
                                };
                                if let Some(cat) = champions_move_category(choice.move_id) {
                                    choice.category = cat;
                                    choice.base_power = 0.0;
                                }
                                let boost = ability_priority_boost(p, &choice) + grassy_glide_boost(state, p, &choice);
                                // A recharging pokemon's action is the `recharge`
                                // pseudo-move (priority 0), not the chosen move.
                                let base = if s.active()[slot].recharging {
                                    0
                                } else {
                                    choice.priority + boost
                                };
                                (choice, base)
                            };
                            // A locked charge move that is not the chosen one (Dive
                            // called by Copycat): PS runs the locked move instead.
                            // Uproar lock (PS `lockedmove` from `uproar`): the user keeps
                            // using Uproar whatever was sent (no PP, priority 0).
                            let uproar_lock = s.active()[slot].uproar > 0 && mv.move_id != Choices::UPROAR && !*struggle;
                            let prio = if uproar_lock { 0 } else { prio };
                            // `lockedmove` (Petal Dance, Outrage, ...): PS `getLockedMove`
                            // replaces the chosen move with the locked one; no PP is used.
                            let locked_id = s.active()[slot].locked_move.and(s.active()[slot].locked_move_id);
                            let (mv, no_pp) = match s.active()[slot].charging {
                                _ if !uproar_lock && !*struggle && locked_id.is_some_and(|m| m != mv.move_id) => {
                                    match MOVES.get(&locked_id.unwrap()).cloned() {
                                        Some(l) => (l, true),
                                        None => (mv, false),
                                    }
                                }
                                _ if uproar_lock => match MOVES.get(&Choices::UPROAR).cloned() {
                                    Some(l) => (l, true),
                                    None => (mv, false),
                                },
                                Some((m, _)) if m != mv.move_id && !*struggle
                                    && ![&p.moves.m0, &p.moves.m1, &p.moves.m2, &p.moves.m3].iter().any(|x| x.id == m) =>
                                {
                                    match MOVES.get(&m).cloned() {
                                        Some(l) => (l, true),
                                        None => (mv, false),
                                    }
                                }
                                _ => (mv, false),
                            };
                            let charge_id = mv.move_id;
                            // PS `insertChoice` queues the resolved move: the
                            // priority comes from the move that actually runs
                            // (a locked two-turn move copied by Copycat is queued
                            // with its own priority 0, not the ignored chosen
                            // move's -6 Roar-style priority).
                            let prio = if s.active()[slot].recharging || uproar_lock {
                                0
                            } else {
                                mv.priority
                                    + ability_priority_boost(p, &mv)
                                    + grassy_glide_boost(state, p, &mv)
                            };
                            actions.push(Action {
                                side,
                                slot,
                                pokemon_index: idx,
                                boost: if s.active()[slot].recharging || uproar_lock {
                                    0
                                } else {
                                    ability_priority_boost(p, &mv) + grassy_glide_boost(state, p, &mv)
                                },
                                choice: mv,
                                priority: prio,
                                speed: effective_speed(state, side, slot).unwrap_or(0),
                                // A locked two-turn move (Dig / Meteor Beam
                                // firing turn) keeps the charge turn's target
                                // (PS side.ts `chooseMove`: lockedMoveTargetLoc).
                                target: match s.active()[slot].charging {
                                    Some((m, _)) if m == charge_id => s.active()[slot].charge_target,
                                    _ if *target > 0 => (*target).min(2) as usize - 1,
                                    _ => 0,
                                },
                                target_loc: match s.active()[slot].charging {
                                    // A forced Struggle (no usable move) has no target
                                    // location: `resolveAction` draws a random foe
                                    // (PS sim/side.ts chooseMove `!moves.length`).
                                    _ if *struggle => 0,
                                    Some((m, _)) if m == charge_id => s.active()[slot].charge_target as i8 + 1,
                                    // A locked move (Thrash, ...) reuses
                                    // `lastMoveTargetLoc` (side.ts `chooseMove`).
                                    _ if s.active()[slot].locked_move.is_some() || s.active()[slot].uproar > 0 || (s.active()[slot].recharging && matches!(p.ability, Abilities::STALWART | Abilities::PROPELLERTAIL)) => {
                                        s.active()[slot].last_target_loc
                                    }
                                    _ => *target,
                                },
                                fractional: 0,
                                prioritized: false,
                                no_pp,
                                recharge: s.active()[slot].recharging,
                                orig_target: None,
                            });
                        }
                    }
                    FuzzAction::Switch { position } => {
                        pending_switches.push((side, slot, *position));
                    }
                }
            }
        }
        if error.is_some() {
            break;
        }

        // Sort by priority/speed, shuffling ties (consumes rng draws).
        if crate::doubles::debug::flags().speed {
            for a in &actions {
                eprintln!(
                    "SPEEDDBG t{} {}{} {} prio={} speed={} (member#{} atk={} spe={})",
                    turn_no,
                    side_tag(a.side),
                    a.slot,
                    state.side(a.side).species_names[a.pokemon_index as usize],
                    a.priority,
                    a.speed,
                    a.pokemon_index as usize,
                    state.side(a.side).team_member(a.pokemon_index).attack,
                    state.side(a.side).team_member(a.pokemon_index).speed
                );
            }
        }
        // PS turn structure (sim/battle.ts `commitChoices` / `turnLoop` /
        // `runAction`): every queued move resolves its target when it is
        // added (`resolveAction` -> `getActionSpeed` -> `getTarget`), the
        // queue is sorted, then `beforeTurn` runs. Every action ends with
        // `eachEvent('Update')` and, when the next action is a move, a
        // re-sort of the queue (`getActionSpeed` again for every queued move).
        if rng.trace {
            rng.context = format!("turn {}: move order", turn_no);
        }
        // `commitChoices` -> `updateSpeed()`.
        update_speed(state);
        for a in actions.iter_mut() {
            // `resolveAction` (sim/battle-queue.ts), in PS order:
            // 1. a move with `priorityChargeCallback` (Focus Punch) queues a
            //    `priorityChargeMove` action, resolved with no target location:
            //    `getRandomTarget` (a draw).
            // `beforeTurnCallback` (Counter / Mirror Coat): a `beforeTurnMove`
            // action is queued first, resolved the same way; its action adds
            // the move's volatile.
            // A recharging pokemon's action is `recharge` (no `beforeTurnCallback`).
            if matches!(a.choice.move_id, Choices::COUNTER | Choices::MIRRORCOAT) && !state.side(a.side).active()[a.slot].recharging {
                let mut charge = a.clone();
                charge.target_loc = 0;
                let _ = ps_get_target(state, rng, &charge);
                let vol = state.side_mut(a.side).active_slot_mut(a.slot);
                if a.choice.move_id == Choices::COUNTER {
                    vol.counter_vol = Some((None, 0));
                } else {
                    vol.mirror_coat_vol = Some((None, 0));
                }
            }
            // A recharging pokemon's action is `recharge`, not the chosen move.
            // Beak Blast's `priorityChargeMove` resolves the same way (PS
            // data/moves.ts beakblast.priorityChargeCallback), in its action's turn.
            if matches!(a.choice.move_id, Choices::FOCUSPUNCH | Choices::BEAKBLAST) && !state.side(a.side).active()[a.slot].recharging {
                let mut charge = a.clone();
                charge.target_loc = 0;
                let _ = ps_get_target(state, rng, &charge);
            }
            // 2. FractionalPriority event.
            let ability = state.side(a.side).team_member(a.pokemon_index).ability;
            // PS `recharge` pseudo-move (`moveid: 'recharge'`, no `category`):
            // not Status, so Quick Draw rolls for it (sim/side.ts chooseMove).
            let category = if a.recharge { MoveCategory::Physical } else { a.choice.category };
            a.fractional = match crate::engine::mechanics::fractional_priority(ability, category) {
                Some(crate::engine::mechanics::FractionalPriority::Fixed(f)) => f,
                Some(crate::engine::mechanics::FractionalPriority::QuickDraw) => {
                    if rng.label("fractional priority").random_chance(3, 10) {
                        1
                    } else {
                        0
                    }
                }
                None => 0,
            };
            // 3. No target location: `getRandomTarget`, and PS keeps the
            //    result as the action's `targetLoc` for later re-sorts.
            //    4. `getActionSpeed` -> `getTarget`.
            //    (A locked `recharge` action already has the last move's
            //    `targetLoc`.)
            if a.target_loc == 0 && !state.side(a.side).active()[a.slot].recharging {
                if let Some((ts, tsl)) = ps_get_target(state, rng, a) {
                    a.target_loc = if ts == a.side { -(tsl as i8 + 1) } else { tsl as i8 + 1 };
                }
            }
            // `action.originalTarget = pokemon.getAtLoc(action.targetLoc)`.
            a.orig_target = orig_target_at(state, a.side, a.target_loc);
            let _ = ps_get_target(state, rng, a);
        }
        commit_queue_sort(&mut actions, state, rng);
        // `beforeTurn` action: `eachEvent('BeforeTurn')`, then the action's
        // closing `eachEvent('Update')`; each is a speed sort of the actives.
        each_event_update(state, rng);
        each_event_update(state, rng);
        // `beforeTurnMove` actions (Counter / Mirror Coat `beforeTurnCallback`):
        // each is an action of its own and ends with `eachEvent('Update')`.
        for a in actions.iter() {
            // A recharging pokemon's action is `recharge` (no `beforeTurnMove`).
            if matches!(a.choice.move_id, Choices::COUNTER | Choices::MIRRORCOAT) && !state.side(a.side).active()[a.slot].recharging {
                each_event_update(state, rng);
            }
        }
        // `priorityChargeMove` actions (order 107: after `beforeTurn`, before
        // every move): Focus Punch adds its volatile. When one is queued,
        // `beforeTurn` is not followed by a move, so its re-sort is skipped;
        // each charge action ends with `eachEvent('Update')`, the last with
        // the re-sort.
        let chargers: Vec<(Side, usize, PokemonIndex, Choices)> = actions
            .iter()
            .filter(|a| {
                matches!(a.choice.move_id, Choices::FOCUSPUNCH | Choices::BEAKBLAST | Choices::CHILLYRECEPTION)
                    && !state.side(a.side).active()[a.slot].recharging
            })
            .map(|a| (a.side, a.slot, a.pokemon_index, a.choice.move_id))
            .collect();
        for &(cs, csl, cidx, charge_id) in &chargers {
            if state.side(cs).active()[csl].pokemon_index == Some(cidx) {
                let a = state.side_mut(cs).active_slot_mut(csl);
                if charge_id == Choices::FOCUSPUNCH {
                    a.focus_punch = true;
                    a.lost_focus = false;
                } else if charge_id == Choices::CHILLYRECEPTION {
                    // PS chillyreception.priorityChargeCallback: `addVolatile`.
                    a.chilly_reception = true;
                } else {
                    // PS beakblast.priorityChargeCallback: `addVolatile('beakblast')`.
                    a.beak_blast = true;
                }
            }
            each_event_update(state, rng);
        }
        resort_queue(state, rng, &mut actions[..]);

        // Execute in order. A pivot move (Volt Switch / U-turn) switches its
        // user out right after it acts, using the controller's answer to PS's
        // mid-turn switch request (PS `runAction` -> `makeRequest('switch')`).
        let mut move_order: Vec<OrderEntry> = Vec::new();
        let mut used_switches = vec![false; turn.switches.len()];
        let mut i = 0;
        // An instructed Counter / Mirror Coat: its `beforeTurnMove` action, run
        // as the next action (see the Instruct handling below).
        let mut instruct_before_turn: Option<Action> = None;
        while i < actions.len() || instruct_before_turn.is_some() {
            if let Some(pre) = instruct_before_turn.take() {
                // PS runAction `beforeTurnMove`: `getTarget`, then the move's
                // `beforeTurnCallback` (`addVolatile('counter' / 'mirrorcoat')`);
                // the action ends like any other, and the queue is re-sorted
                // before the next move. PS sim/battle.ts runAction.
                let present = state.side(pre.side).active()[pre.slot].pokemon_index == Some(pre.pokemon_index)
                    && state.side(pre.side).team_member(pre.pokemon_index).hp > 0;
                if present && ps_get_target(state, rng, &pre).is_some() {
                    let vol = state.side_mut(pre.side).active_slot_mut(pre.slot);
                    if pre.choice.move_id == Choices::COUNTER {
                        vol.counter_vol = Some((None, 0));
                    } else {
                        vol.mirror_coat_vol = Some((None, 0));
                    }
                }
                if present {
                    each_event_update(state, rng);
                }
                if i < actions.len() {
                    resort_queue(state, rng, &mut actions[i..]);
                }
                continue;
            }
            let action = actions[i].clone();
            i += 1;
            // PS ends the battle as soon as a side has no pokemon left
            // (`checkWin` in `faintMessages`); queued actions never run.
            if state.battle_is_over() != 0.0 {
                break;
            }
            // The chosen pokemon must still be the slot's occupant (PS
            // `runAction` returns early, skipping the end-of-action steps).
            // Actions follow the pokemon: Ally Switch swaps positions.
            let mut action = action;
            match (0..2).find(|&sl| state.side(action.side).active()[sl].pokemon_index == Some(action.pokemon_index)) {
                Some(sl) => action.slot = sl,
                None => continue,
            }
            if rng.trace {
                rng.context = format!(
                    "turn {}: {}{} {} {}",
                    turn_no,
                    side_tag(action.side),
                    action.slot,
                    state.side(action.side).species_names[action.pokemon_index as usize],
                    action.choice.move_id
                );
            }
            execute_action(state, rng, &action, &mut move_order, &actions[i..]);
            // Round `onTry` (flagged in execute_action_inner, only when the user
            // got past BeforeMove): the first other queued Round is moved to the
            // front of the queue with `sourceEffect: 'round'` (BP doubled).
            // PS data/moves.ts round, sim/battle-queue.ts prioritizeAction.
            if std::mem::take(&mut state.queue_round) {
                // `i` is already past the running action (PS `queue.list`).
                // PS round.onTry: `action.move.id === 'round'`. A recharging pokemon's
                // queued action is the `recharge` pseudo-move, not Round.
                if let Some(j) = (i..actions.len()).find(|&j| actions[j].choice.move_id == Choices::ROUND && !actions[j].recharge) {
                    let mut a = actions.remove(j);
                    a.choice.base_power *= 2.0;
                    a.prioritized = true;
                    actions.insert(i, a);
                }
            }
            // Encore `onStart` -> `queue.changeAction(target, {choice: 'move', moveid})`:
            // cancel the target's queued move and insert the encored one
            // (`resolveAction` draws a random target; `insertChoice` shuffles ties).
            // A Counter / Mirror Coat insert also queues its `beforeTurnMove`
            // (order 5), which runs right after this action (see below).
            let mut before_turn_move: Option<(Side, PokemonIndex, Choices)> = None;
            if let Some(plan) = state.queue_change.take() {
                if matches!(plan.action.choice.move_id, Choices::COUNTER | Choices::MIRRORCOAT) {
                    before_turn_move = Some((plan.action.side, plan.action.pokemon_index, plan.action.choice.move_id));
                }
                apply_queue_plan(&mut actions, i, plan);
            }
            // Gravity: `queue.cancelMove(pokemon)` for pokemon whose Fly / Bounce ended.
            for (cs, csl) in std::mem::take(&mut state.queue_cancel) {
                if let Some(cidx) = state.side(cs).active()[csl].pokemon_index {
                    let mut k = i;
                    while k < actions.len() {
                        if actions[k].side == cs && actions[k].pokemon_index == cidx {
                            actions.remove(k);
                        } else {
                            k += 1;
                        }
                    }
                }
            }
            // Quash `onHit`: the target's action moves to the end (order 201).
            if let Some((ts, tsl)) = state.queue_quash.take() {
                if let Some(tidx) = state.side(ts).active()[tsl].pokemon_index {
                    if let Some(j) = (i..actions.len()).find(|&j| actions[j].side == ts && actions[j].pokemon_index == tidx) {
                        let mut a = actions.remove(j);
                        a.priority = -100;
                        actions.push(a);
                    }
                }
            }
            // After You `onHit` -> `queue.prioritizeAction(action)`.
            // PS sim/battle-queue.ts prioritizeAction.
            if let Some((ts, tsl)) = state.queue_prioritize.take() {
                if let Some(tidx) = state.side(ts).active()[tsl].pokemon_index {
                    if let Some(j) = (i..actions.len()).find(|&j| actions[j].side == ts && actions[j].pokemon_index == tidx) {
                        let mut a = actions.remove(j);
                        a.prioritized = true;
                        actions.insert(i, a);
                    }
                }
            }
            // Instruct `onHit` -> `queue.prioritizeAction(resolveAction(...))`: the
            // target's last move runs next, with its last target location.
            if let Some((ts, tsl, tidx, mv, loc)) = state.queue_instruct.take() {
                if let Some(a) = instruct_action(state, ts, tsl, tidx, mv, loc).filter(|_| matches!(mv, Choices::COUNTER | Choices::MIRRORCOAT)) {
                    // PS instruct.onHit prioritizes `resolveAction(...)[0]`: for a
                    // move with `beforeTurnCallback` that is the `beforeTurnMove`
                    // action, not the move. `resolveAction` resolves both
                    // (`getActionSpeed` -> `getTarget` each); the move is dropped.
                    // PS data/moves.ts instruct, sim/battle-queue.ts resolveAction.
                    let _ = ps_get_target(state, rng, &a);
                    let _ = ps_get_target(state, rng, &a);
                    instruct_before_turn = Some(a);
                } else if let Some(a) = instruct_action(state, ts, tsl, tidx, mv, loc) {
                    // PS `queue.resolveAction` (instruct.onHit) -> `getActionSpeed` ->
                    // `getTarget`: the instructed copy resolves its target at once.
                    let _ = ps_get_target(state, rng, &a);
                    actions.insert(i, a);
                }
            }
            // Phazing (PS `runAction`: `forceSwitchFlag` -> `dragIn`) happens
            // before faints are resolved.
            let dragged = drag_in_all(state, rng);
            // PS switchIn: `queue.cancelAction(oldActive)`: a pokemon forced out
            // by Roar / Whirlwind / Dragon Tail loses its queued action (and its
            // later `getActionSpeed` target draws). PS sim/battle-actions.ts switchIn.
            if !dragged.is_empty() {
                let mut k = i;
                while k < actions.len() {
                    if dragged.contains(&(actions[k].side, actions[k].pokemon_index)) {
                        actions.remove(k);
                    } else {
                        k += 1;
                    }
                }
            }
            resolve_faints(state, rng, |_, _| true);
            if state.battle_is_over() != 0.0 {
                break;
            }
            // End of the move's `runAction`.
            update_forecast(state);
            state.side_one.sync_sc_order();
            state.side_two.sync_sc_order();
            sync_vol_order(state);
            each_event_update(state, rng);
            // PS `runAction` case 'beforeTurnMove' (the Encore-inserted
            // Counter / Mirror Coat): `beforeTurnCallback` adds the volatile,
            // then the action's own `eachEvent('Update')`. The re-sort after
            // the previous action is skipped (next action was not a move).
            if let Some((bs, bidx, bmv)) = before_turn_move {
                if let Some(bsl) = (0..2).find(|&sl| state.side(bs).active()[sl].pokemon_index == Some(bidx)) {
                    if state.side(bs).team_member(bidx).hp > 0 {
                        let vol = state.side_mut(bs).active_slot_mut(bsl);
                        if bmv == Choices::COUNTER {
                            vol.counter_vol.get_or_insert((None, 0));
                        } else {
                            vol.mirror_coat_vol.get_or_insert((None, 0));
                        }
                        each_event_update(state, rng);
                    }
                }
            }
            let queue_start = i;
            // PS runAction tail: `runPursuitActivation` (gen >= 5: `eachEvent('Update')`)
            // for every live switching pokemon, sides then slots, before the switch
            // request; Baton Pass / Shed Tail set `skipPursuit`. Then commitChoices
            // `queue.sort()`s the new switch actions (speed order, ties shuffled)
            // and they run one by one. PS sim/battle.ts runAction / commitChoices,
            // sim/battle-actions.ts runPursuitActivation.
            let mut fs_slots: Vec<(Side, usize)> = Vec::new();
            for (side, slot) in [(Side::One, 0usize), (Side::One, 1), (Side::Two, 0), (Side::Two, 1)] {
                if !state.side(side).active()[slot].force_switch {
                    continue;
                }
                fs_slots.push((side, slot));
                let a = &state.side(side).active()[slot];
                if a.reviving || a.baton_passing || a.shed_tailing {
                    continue;
                }
                let key = if side == Side::One { "a" } else { "b" };
                let answered = turn.switches.iter().enumerate().any(|(k, sw)| {
                    !used_switches[k] && sw.pivot && sw.side == key && sw.slot == slot && sw.position.is_some()
                });
                if answered && state.side(side).slot_pokemon(slot).is_some_and(|p| p.hp > 0) {
                    each_event_update(state, rng);
                }
            }
            if fs_slots.len() >= 2 {
                update_speed(state);
                let hs: Vec<ResidualHandler> = fs_slots
                    .iter()
                    .enumerate()
                    .map(|(k, &(sd, sl))| {
                        let speed = state.side(sd).active()[sl]
                            .pokemon_index
                            .map(|w| state.side(sd).speed_cache[w as usize] as i32)
                            .unwrap_or(0);
                        ResidualHandler { order: 0, priority: 0, speed, sub_order: 0, kind: ResidualKind::Status(sd, k) }
                    })
                    .collect();
                let old = fs_slots.clone();
                fs_slots = sort_residual_handlers(hs, rng, "speed tie (move order)")
                    .into_iter()
                    .filter_map(|h| match h.kind {
                        ResidualKind::Status(_, k) => Some(old[k]),
                        _ => None,
                    })
                    .collect();
            }
            // Two or more `instaswitch` actions: each `switchIn` queues its
            // `runSwitch` behind the next instaswitch, so they all switch in
            // first and one `runSwitch` batch follows (as at the end of a turn).
            // PS sim/battle-actions.ts switchIn / runSwitch, sim/battle.ts runAction.
            let switching = fs_slots.iter().filter(|&&(sd, sl)| {
                let a = &state.side(sd).active()[sl];
                let key = if sd == Side::One { "a" } else { "b" };
                !a.reviving && turn.switches.iter().enumerate().any(|(k, sw)| {
                    !used_switches[k] && sw.pivot && sw.side == key && sw.slot == sl && sw.position.is_some()
                })
            }).count();
            let batch = switching >= 2;
            let mut batch_in: Vec<(Side, usize)> = Vec::new();
            let mut run_switch_speeds: Vec<i16> = Vec::new();
            for (side, slot) in fs_slots {
                if !state.side(side).active()[slot].force_switch {
                    continue;
                }
                state.side_mut(side).slot_scratch_mut(slot).force_switch = false;
                let key = if side == Side::One { "a" } else { "b" };
                let found = turn.switches.iter().enumerate().find(|(i, sw)| {
                    !used_switches[*i] && sw.pivot && sw.side == key && sw.slot == slot
                });
                if let Some((i, sw)) = found {
                    used_switches[i] = true;
                    if state.side(side).active()[slot].reviving {
                        state.side_mut(side).slot_scratch_mut(slot).reviving = false;
                        if let (true, Some(position)) = (sw.revive, sw.position) {
                            // The switch answer (`commitChoices` -> `updateSpeed`),
                            // then the `revivalblessing` action (PS battle.ts
                            // `runAction`): half HP, status cured.
                            update_speed(state);
                            revive_member(state, side, position);
                            each_event_update(state, rng);
                            // The revived pokemon is the fainted ally still in its
                            // active slot (`target.position < active.length`): PS
                            // queues an `instaswitch` that sends it back in there
                            // (`switchIn`: pursuit `eachEvent('Update')`, then the
                            // `runSwitch` action). PS sim/battle.ts runAction
                            // `revivalblessing` / `instaswitch`.
                            let back = (0..2).find(|&sl| {
                                state.side(side).active()[sl].pokemon_index.is_none()
                                    && state.side(side).fainted_in_slot[sl].is_some_and(|(fi, _)| fi as usize == position)
                            });
                            if let Some(sl) = back {
                                // `switchIn` sees the revived pokemon itself as the
                                // live `oldActive` and cancels its queued move
                                // (`queue.cancelAction(oldActive)`). PS sim/battle-actions.ts switchIn.
                                let fi = state.side(side).fainted_in_slot[sl].unwrap().0;
                                let mut k = queue_start;
                                while k < actions.len() {
                                    if actions[k].side == side && actions[k].pokemon_index == fi {
                                        actions.remove(k);
                                    } else {
                                        k += 1;
                                    }
                                }
                                each_event_update(state, rng);
                                switch_in(state, side, sl, position);
                                update_speed(state);
                                each_event_update(state, rng);
                                let _ = sorted_actives(state, rng, "speed tie (switch-in)");
                                apply_sticky_web(state, side, sl);
                                if switched_in_alive(state, side, sl) && apply_on_switch_abilities(state, side, sl, rng) {
                                    let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
                                }
                                each_event_update(state, rng);
                                // The `runSwitch` action is followed by a move: the
                                // gen 8+ `runAction` re-sort. PS sim/battle.ts runAction.
                                if queue_start < actions.len() {
                                    resort_queue(state, rng, &mut actions[queue_start..]);
                                }
                            }
                        }
                        continue;
                    }
                    if let Some(position) = sw.position {
                        // PS `runPursuitActivation` (sim/battle-actions.ts, called
                        // from `Battle.runAction` for a live pokemon with a
                        // `switchFlag`): `eachEvent('Update')` (gen >= 5).
                        // Baton Pass / Shed Tail set `skipPursuit` in `self.onHit`
                        // (PS data/moves.ts), so they skip it.
                        // TODO(PS parity): Shed Tail (not in the engine yet).
                        // (Its `runPursuitActivation` sort ran above, for all switchers.)
                        // The switch answer (`commitChoices` -> `updateSpeed`),
                        // the `instaswitch` action, then the `runSwitch` action.
                        update_speed(state);
                        // Baton Pass (`selfSwitch: 'copyvolatile'`): boosts and
                        // the copyable volatiles move to the replacement.
                        // Baton Pass `copyVolatileFrom`: the passable volatiles.
                        let baton_extra = if state.side(side).active()[slot].baton_passing {
                            Some(state.side(side).active()[slot].clone())
                        } else {
                            None
                        };
                        // PS gastroacid has no `noCopy`: Baton Pass copies it to the replacement.
                        let baton_gastro = state.side(side).active()[slot].baton_passing && state.side(side).active()[slot].gastro_acid;
                        let shed = state.side(side).active()[slot].shed_tailing;
                        let passed = if shed {
                            let a = &state.side(side).active()[slot];
                            Some((Default::default(), 0, a.substitute_health))
                        } else if state.side(side).active()[slot].baton_passing {
                            let a = &state.side(side).active()[slot];
                            a.pokemon_index.map(|old| {
                                (
                                    state.side(side).team_boosts[old as usize].clone(),
                                    a.confusion,
                                    a.substitute_health,
                                )
                            })
                        } else {
                            None
                        };
                        // PS `switchIn`: a live switch-out cancels the old pokemon's
                        // queued actions (`queue.cancelAction(oldActive)`).
                        let old_pi = state.side(side).active()[slot].pokemon_index;
                        switch_out_end_event(state, rng, side, slot);
                        if batch {
                            switch_in_inner(state, side, slot, position);
                        } else {
                            switch_in(state, side, slot, position);
                        }
                        // PS `insertChoice` -> `pokemon.updateSpeed()` for the incoming
                        // pokemon: its `speed` is fresh for `runSwitch`'s speedSort.
                        update_speed(state);
                        if let Some(op) = old_pi {
                            let mut k = queue_start;
                            while k < actions.len() {
                                if actions[k].side == side && actions[k].pokemon_index == op {
                                    actions.remove(k);
                                } else {
                                    k += 1;
                                }
                            }
                        }
                        // PS sim/pokemon.ts copyVolatileFrom: every volatile without
                        // `noCopy` moves to the replacement. Copied (champions pool):
                        // aquaring, allyswitch, charge, curse, dragoncheer, electrify,
                        // focusenergy, healblock, helpinghand, ingrain, leechseed,
                        // magnetrise, noretreat, octolock, partiallytrapped, perishsong,
                        // powertrick (`onCopy` swaps Atk/Def), stall, throatchop
                        // (+ confusion, substitute, gastroacid below). noCopy: attract,
                        // disable, encore, lockon, saltcure, smackdown, syrupbomb, yawn, ...
                        if let Some(old) = baton_extra {
                            let a = state.side_mut(side).active_slot_mut(slot);
                            a.cursed = old.cursed;
                            a.focus_energy = old.focus_energy;
                            a.leech_seed = old.leech_seed;
                            a.ingrain = old.ingrain;
                            a.aqua_ring = old.aqua_ring;
                            a.heal_block = old.heal_block;
                            a.partial_trap = old.partial_trap;
                            a.octolock = old.octolock;
                            a.perish = old.perish;
                            a.helping_hand_mult = old.helping_hand_mult;
                            a.charge = old.charge;
                            a.dragon_cheer = old.dragon_cheer;
                            a.magnet_rise = old.magnet_rise;
                            a.no_retreat = old.no_retreat;
                            a.throat_chop = old.throat_chop;
                            a.stall = old.stall;
                            a.electrify = old.electrify;
                            a.ally_switch = old.ally_switch;
                            // PS copyVolatileFrom adds them in the source's key order.
                            a.vol_order = old.vol_order.clone();
                            a.vol_synced = u64::MAX;
                            if old.power_trick {
                                a.power_trick = true;
                                if let Some(ni) = a.pokemon_index {
                                    let pm = state.side_mut(side).team_member_mut(ni);
                                    std::mem::swap(&mut pm.attack, &mut pm.defense);
                                }
                            }
                        }
                        // Baton Pass copies the `gastroacid` volatile (PS copyVolatileFrom):
                        // the replacement's ability is suppressed again unless it cannot be
                        // suppressed (`onCopy`). TODO(PS parity): the End event's side effects
                        // (Cloud Nine / Air Lock weather change) are not run here.
                        if baton_gastro {
                            if let Some(ni) = state.side(side).active()[slot].pokemon_index {
                                let a = state.side(side).team_member(ni).ability;
                                if !gastro_cant_suppress(a) {
                                    if a != Abilities::NONE {
                                        mold_breaker_forget(state, side, ni);
                                        state.side_mut(side).team_member_mut(ni).ability = Abilities::NONE;
                                    }
                                    state.side_mut(side).active_slot_mut(slot).gastro_acid = true;
                                    state.side_mut(side).slot_scratch_mut(slot).gastro_ability = a;
                                }
                            }
                        }
                        if let Some((boosts, confusion, sub)) = passed {
                            if let Some(new_idx) = state.side(side).active()[slot].pokemon_index {
                                state.side_mut(side).team_boosts[new_idx as usize] = boosts;
                                let a = state.side_mut(side).active_slot_mut(slot);
                                a.confusion = confusion;
                                a.substitute_health = sub;
                            }
                            // `copyVolatileFrom` runs inside `switchIn`, before the
                            // `runSwitch` insertChoice's `updateSpeed`: the passed
                            // Speed boosts are in the cached speed. PS
                            // sim/battle-actions.ts switchIn, sim/battle-queue.ts insertChoice.
                            update_speed(state);
                        }
                        if batch {
                            // `insertChoice(runSwitch)`: a tie with an already queued
                            // `runSwitch` draws its place.
                            update_speed(state);
                            if let Some(sp) = effective_speed(state, side, slot) {
                                let ties = run_switch_speeds.iter().filter(|&&x| x == sp).count();
                                if ties > 0 {
                                    let _ = rng.label("speed tie (queue insert)").random(ties as u32 + 1);
                                }
                                run_switch_speeds.push(sp);
                            }
                            batch_in.push((side, slot));
                            continue;
                        }
                        // PS `switchIn`: `copyVolatileFrom` runs before
                        // `queue.insertChoice({choice: 'runSwitch'})` -> `updateSpeed`,
                        // so the passed boosts count in the switch-in speed sorts.
                        update_speed(state);
                        each_event_update(state, rng);
                        let _ = sorted_actives(state, rng, "speed tie (switch-in)");
                        apply_sticky_web(state, side, slot);
                        // TODO(PS parity): `fieldEvent('SwitchIn')` handler ties.
                        if switched_in_alive(state, side, slot) && apply_on_switch_abilities(state, side, slot, rng) {
                            let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
                        }
                        // PS `runAction` (the `runSwitch` action): `resolveFaints`
                        // before `eachEvent('Update')`, so a hazard KO is gone.
                        resolve_faints(state, rng, |_, _| true);
                        each_event_update(state, rng);
                        // Forecast `onWeatherChange` (PS data/abilities.ts forecast):
                        // Snow Warning / Drizzle on the pivot's switch-in may have just
                        // set the weather, so Castform's type follows it now.
                        update_forecast(state);
                    }
                }
            }
            if !batch_in.is_empty() {
                // The batched `runSwitch` (see above): the last instaswitch's
                // `eachEvent('Update')`, the switch-in sort, each incoming pokemon's
                // hazards and Start in that order, then the runSwitch's eachEvent.
                each_event_update(state, rng);
                let order = sorted_actives(state, rng, "speed tie (switch-in)");
                for (sd, sl) in order {
                    if batch_in.contains(&(sd, sl)) {
                        entry_hazards(state, sd, sl);
                        apply_sticky_web(state, sd, sl);
                        refresh_mimicry_slot(state, sd, sl);
                    }
                }
                // Opportunist `onAnySwitchIn` (as `switch_in` does).
                for sd in [Side::One, Side::Two] {
                    for sl in 0..2 {
                        flush_opportunist(state, sd, sl);
                    }
                }
                for &(sd, sl) in &batch_in {
                    if !state.side(sd).slot_pokemon(sl).is_some_and(|p| p.hp > 0) {
                        continue;
                    }
                    if apply_on_switch_abilities(state, sd, sl, rng) {
                        let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
                    }
                }
                each_event_update(state, rng);
                update_forecast(state);
            }
            // PS re-sorts only when the next action is a move (an instructed
            // `beforeTurnMove` re-sorts after it runs).
            if i < actions.len() && instruct_before_turn.is_none() {
                resort_queue(state, rng, &mut actions[i..]);
            }
        }

        // Faint replacements answered at the end of the turn.
        for (i, sw) in turn.switches.iter().enumerate() {
            if used_switches[i] || sw.pivot {
                continue;
            }
            if let Some(position) = sw.position {
                let side = if sw.side == "a" { Side::One } else { Side::Two };
                pending_switches.push((side, sw.slot, position));
            }
        }

        // Pivot-style answers not used mid-turn: Emergency Exit / Wimp Out
        // switches caused by residual damage.
        let mut ee_switches: Vec<(Side, usize, usize)> = Vec::new();
        for (i, sw) in turn.switches.iter().enumerate() {
            if !used_switches[i] && sw.pivot {
                if let Some(position) = sw.position {
                    let side = if sw.side == "a" { Side::One } else { Side::Two };
                    ee_switches.push((side, sw.slot, position));
                }
            }
        }

        // End of turn: residual damage, switches (with hazards), durations.
        if rng.trace {
            rng.context = format!("turn {}: end of turn", turn_no);
        }
        // PS stops at once when a side is wiped out: no residual step.
        if state.battle_is_over() == 0.0 {
            end_of_turn(state, rng, &pending_switches, &ee_switches);
            update_forecast(state);
            state.side_one.sync_sc_order();
            state.side_two.sync_sc_order();
            sync_vol_order(state);
        }

        if state.record {
            turn_states.push(fuzz_snapshot(state, move_order));
        }
        state.turn += 1;
        ci += 1;

        if ci > 10_000 {
            error = Some("turn limit exceeded (10000)".into());
            break;
        }
    }

    let winner = match state.battle_is_over() {
        1.0 => "A",
        -1.0 => "B",
        _ => "draw",
    };
    FuzzBattleResult {
        winner: winner.to_string(),
        total_turns: state.turn,
        turns: turn_states,
        error,
        prng_draws: rng.draw_log.clone(),
        prng_draw_ctx: rng.draw_ctx.clone(),
    }
}

/// PS `move.pranksterBoosted` with the user still holding Prankster (the
/// `hitStepTryImmunity` check). Prankster's `onModifyPriority` sets it on a
/// Status move; a move called by `useMove` inherits the calling move's flag
/// (`move.pranksterBoosted = activeMove.pranksterBoosted`; every caller,
/// Copycat / Sleep Talk / Metronome, is a Status move), except a Magic Bounce
/// copy (`hasBounced`, `pranksterBoosted = false`). PS sim/battle-actions.ts
/// useMoveInner, data/abilities.ts prankster / magicbounce.
fn prankster_boosted(state: &DoublesState, side: Side, slot: usize, choice: &Choice) -> bool {
    if state.side(side).slot_pokemon(slot).map(|p| p.ability) != Some(Abilities::PRANKSTER) {
        return false;
    }
    let a = &state.side(side).active()[slot];
    if a.external_call {
        !a.bounced
    } else {
        choice.category == MoveCategory::Status
    }
}

/// pokemon-showdown `getTarget` (sim/battle.ts) for a queued move, for doubles:
/// returns the resolved target (side, slot), consuming the PRNG draw PS makes
/// when it has to pick a random target (`getRandomTarget` -> `randomFoe` ->
/// `sample`, which draws even with one candidate). PS calls it in
/// `getActionSpeed` (queue build and every re-sort) and in `runMove`.
fn ps_get_target(state: &DoublesState, rng: &mut Gen5Rng, action: &Action) -> Option<(Side, usize)> {
    ps_get_target_inner(state, rng, action, "retarget")
}

/// The pokemon in `(side, slot)` is in the charge turn of a semi-invulnerable
/// move (Dig, Dive, Fly, Bounce, Phantom Force, Shadow Force, Sky Drop) and its
/// charge was not aborted: a move aimed at it misses. PS `twoturnmove` /
/// `onInvulnerability` (data/moves.ts, data/conditions.ts).
fn ally_semi_invulnerable(state: &DoublesState, side: Side, slot: usize) -> bool {
    let a = &state.side(side).active()[slot];
    a.charging.is_some_and(|(m, _)| {
        matches!(m, Choices::DIG | Choices::DIVE | Choices::FLY | Choices::BOUNCE | Choices::PHANTOMFORCE | Choices::SHADOWFORCE | Choices::SKYDROP)
    }) && !a.charge_aborted
}

/// `ps_get_target` for a draw made under PS `insertChoice` (Encore's
/// `changeAction`): the call stack contains `insertChoice`, so PS tags the draw
/// "speed tie (queue insert)".
fn ps_get_target_queue_insert(state: &DoublesState, rng: &mut Gen5Rng, action: &Action) -> Option<(Side, usize)> {
    ps_get_target_inner(state, rng, action, "speed tie (queue insert)")
}

/// The move type `RedirectTarget` sees. PS `useMoveInner` runs the move's
/// `onModifyType` (singleEvent) and `ModifyType` handlers before `getMoveTargets`:
/// Aura Wheel is Dark for Morpeko-Hangry and Electric otherwise (data/moves.ts
/// aurawheel), and Electrify (priority -2) makes any move but Struggle Electric.
fn redirect_move_type(state: &DoublesState, action: &Action) -> PokemonType {
    if action.choice.move_id == Choices::AURAWHEEL {
        return if state.side(action.side).team_member(action.pokemon_index).id == crate::pokemon::PokemonName::MORPEKOHANGRY {
            PokemonType::DARK
        } else {
            PokemonType::ELECTRIC
        };
    }
    if state.side(action.side).active()[action.slot].electrify && action.choice.move_id != Choices::STRUGGLE {
        return PokemonType::ELECTRIC;
    }
    // Terrain Pulse / Weather Ball `onModifyType` and the -ate abilities'
    // `onModifyType` also run before `getMoveTargets`, as in `execute_action_inner`.
    let user = state.side(action.side).team_member(action.pokemon_index);
    let mut c = action.choice.clone();
    match c.move_id {
        Choices::TERRAINPULSE => {
            let grounded = slot_grounded(user, &state.side(action.side).active()[action.slot], state.gravity.active);
            if let Some(t) = crate::engine::mechanics::terrain_pulse_type(state.terrain.terrain_type, grounded) {
                c.move_type = t;
            }
        }
        Choices::WEATHERBALL => {
            let mega_sol = user.ability == Abilities::MEGASOL;
            if let Some(t) = crate::engine::mechanics::weather_ball_type(effective_weather(state), mega_sol) {
                c.move_type = t;
            }
        }
        _ => {}
    }
    crate::engine::mechanics::ate_type_change(user.ability, &c).unwrap_or(c.move_type)
}

/// PS `priorityEvent('RedirectTarget')` (sim/pokemon.ts `getMoveTargets`):
/// Lightning Rod / Storm Drain `onAnyRedirectTarget` pull an Electric / Water
/// move with a single-target type to the holder (any other active that is a
/// valid target). No draw.
// TODO(PS parity): speed order / ties between holders, Mold Breaker, tracksTarget.
fn redirect_target(state: &DoublesState, action: &Action, target: (Side, usize)) -> (Side, usize) {
    // Counter / Mirror Coat `onRedirectTarget` (priority -1): the recorded
    // attacker's slot, unless a higher-priority Follow Me / Rage Powder (1)
    // returns first (PS `priorityEvent` stops at the first result).
    let a = &state.side(action.side).active()[action.slot];
    let rec = match action.choice.move_id {
        Choices::COUNTER => a.counter_vol,
        Choices::MIRRORCOAT => a.mirror_coat_vol,
        _ => None,
    };
    let ty = move_target(action.choice.move_id);
    if !matches!(ty, Tgt::Normal | Tgt::RandomNormal | Tgt::AdjacentFoe | Tgt::Any | Tgt::Scripted) {
        if let Some((Some(t), _)) = rec {
            return t;
        }
        return target;
    }
    // Follow Me / Rage Powder `onFoeRedirectTarget` (priority 1: before the
    // abilities and before Counter / Mirror Coat's own `onRedirectTarget`,
    // priority -1; PS `priorityEvent` takes the first answer).
    let foe = action.side.other();
    for sl in 0..2 {
        let a = &state.side(foe).active()[sl];
        if a.follow_me == 0 || state.side(foe).slot_pokemon(sl).map(|p| p.hp <= 0).unwrap_or(true) {
            continue;
        }
        let powder_ok = a.follow_me == 1
            || !(crate::engine::mechanics::powder_immune(state.side(action.side).team_member(action.pokemon_index))
                || state.side(action.side).active()[action.slot].added_type == PokemonType::GRASS);
        if powder_ok {
            return (foe, sl);
        }
    }
    if let Some((Some(t), _)) = rec {
        return t;
    }
    let want = match redirect_move_type(state, action) {
        PokemonType::ELECTRIC => Abilities::LIGHTNINGROD,
        PokemonType::WATER => Abilities::STORMDRAIN,
        _ => return target,
    };
    // PS `priorityEvent` runs the holders in `compareRedirectOrder` (priority,
    // then speed, descending) and takes the first one that returns a target.
    let mut holders: Vec<(i32, Side, usize)> = Vec::new();
    for (s, sl) in [(Side::One, 0usize), (Side::One, 1), (Side::Two, 0), (Side::Two, 1)] {
        if (s, sl) == (action.side, action.slot) {
            continue;
        }
        if let Some(p) = state.side(s).slot_pokemon(sl) {
            if p.hp > 0 && p.ability == want {
                // 'adjacentFoe' only accepts foes.
                if ty == Tgt::AdjacentFoe && s == action.side {
                    continue;
                }
                holders.push((cached_speed(state, s, sl), s, sl));
            }
        }
    }
    // Stable sort: equal speeds keep the side/slot order above.
    // TODO(PS parity): equal-speed holders are ordered by `effectOrder` in PS.
    holders.sort_by(|a, b| b.0.cmp(&a.0));
    if let Some(&(_, s, sl)) = holders.first() {
        return (s, sl);
    }
    target
}

fn ps_get_target_inner(
    state: &DoublesState,
    rng: &mut Gen5Rng,
    action: &Action,
    label: &'static str,
) -> Option<(Side, usize)> {
    let (side, slot) = (action.side, action.slot);
    let a = &state.side(side).active()[slot];
    // Curse chosen while the user was not Ghost (queued as 'self', stored as the
    // user's own loc). A Ghost-type change during the turn (Trick-or-Treat) does
    // not redo the target: PS keeps the stored target, and the Ghost `onTryHit`
    // then fails or applies to the user. PS data/moves.ts curse.onModifyMove.
    if !action.recharge
        && action.choice.move_id == Choices::CURSE
        && action.target_loc == -(slot as i8 + 1)
    {
        return Some((side, slot));
    }
    // A recharging pokemon's locked action is the `recharge` pseudo-move,
    // which has no target type.
    let user_types = state.side(side).team_member(action.pokemon_index).types;
    // PS `source.hasType('Ghost')`: the current types, including an added type.
    let user_ghost = user_types.0 == PokemonType::GHOST
        || user_types.1 == PokemonType::GHOST
        || state.side(side).active()[slot].added_type == PokemonType::GHOST;
    let ty = if action.recharge {
        Tgt::None
    } else if action.choice.move_id == Choices::CURSE && !user_ghost {
        // curse.onModifyMove: `move.target = move.nonGhostTarget` ('self').
        Tgt::SelfTarget
    } else {
        move_target(action.choice.move_id)
    };
    let alive = |sd: Side, sl: usize| state.side(sd).slot_pokemon(sl).map(|p| p.hp > 0).unwrap_or(false);
    let foe = side.other();

    let random_target = |rng: &mut Gen5Rng| -> Option<(Side, usize)> {
        match ty {
            Tgt::SelfTarget | Tgt::All | Tgt::AllySide | Tgt::AllyTeam | Tgt::AdjacentAllyOrSelf => Some((side, slot)),
            Tgt::AdjacentAlly => {
                if alive(side, 1 - slot) {
                    let _ = rng.label(label).random(1);
                    Some((side, 1 - slot))
                } else {
                    None
                }
            }
            _ => {
                let foes: smallvec::SmallVec<[usize; 2]> = (0..2).filter(|&s| alive(foe, s)).collect();
                if foes.is_empty() {
                    // PS `getRandomTarget` (sim/battle.ts) with no living foe: no draw;
                    // `randomFoe() || foe.active[0]` (the slot "directly across" is
                    // the `activePerHalf > 2` triples branch only). A redirect
                    // (Lightning Rod, Follow Me) can still apply to that slot.
                    Some((foe, 0))
                } else {
                    Some((foe, foes[rng.label(label).random(foes.len() as u32) as usize]))
                }
            }
        }
    };

    let loc = action.target_loc;
    if loc == 0 {
        // `validTargetLoc(0)` is true but `getAtLoc(0)` is empty.
        return random_target(rng);
    }
    let self_loc = -(slot as i8 + 1);
    if matches!(ty, Tgt::AdjacentAlly | Tgt::Any | Tgt::Normal) && loc == self_loc && a.charging.is_none() {
        return None;
    }
    let is_self = loc == self_loc;
    let is_foe = loc > 0;
    // In doubles every other slot is adjacent.
    let valid = loc.abs() <= 2
        && match ty {
            Tgt::RandomNormal | Tgt::Scripted | Tgt::Normal => !is_self,
            Tgt::AdjacentAlly => !is_self && !is_foe,
            Tgt::AdjacentAllyOrSelf => !is_foe,
            Tgt::AdjacentFoe => is_foe,
            Tgt::Any => !is_self,
            _ => false,
        };
    if ty != Tgt::RandomNormal && valid {
        let (ts, tsl) = if is_foe { (foe, (loc - 1) as usize) } else { (side, (-loc - 1) as usize) };
        if alive(ts, tsl) {
            return Some((ts, tsl));
        }
        if !is_foe {
            // PS `getTarget`: a fainted ally is not retargeted, except that
            // `adjacentAllyOrSelf` (gen != 5) returns the user (no draw).
            if ty == Tgt::AdjacentAllyOrSelf {
                return Some((side, slot));
            }
            return Some((ts, tsl));
        }
    }
    random_target(rng)
}

/// PS `getStat('spe')`: speed with boosts and modifiers, without the Trick
/// Room negation that `effective_speed` (action speed) applies.
fn stat_speed(state: &DoublesState, side: Side, slot: usize) -> i32 {
    effective_speed(state, side, slot).map(|s| (s as i32).abs()).unwrap_or(0)
}

/// PS `battle.updateSpeed()`: refresh the cached speed of every active.
/// The cache holds PS's `pokemon.speed` (`getActionSpeed`: `10000 - stat`
/// under Trick Room, the raw stat otherwise), the same convention the other
/// cache writes use (`speed_cache[i] = raw stat`, e.g. a Mimikyu bust). Keeping
/// one convention lets a stale entry tie or order the way PS's does.
fn update_speed(state: &mut DoublesState) {
    update_speed_get(state);
}

/// `update_speed`, returning each slot's `effective_speed` (by side, slot).
fn update_speed_get(state: &mut DoublesState) -> [[Option<i16>; 2]; 2] {
    let trick_room = state.trick_room.active;
    let weather = effective_weather(state);
    let mut out = [[None; 2]; 2];
    for side in [Side::One, Side::Two] {
        for slot in 0..2 {
            // `effective_speed` is the stat negated under Trick Room.
            let eff = effective_speed_in(state, side, slot, weather);
            out[side as usize][slot] = eff;
            let sp = eff.map(|sp| if trick_room { 10000 + sp as i32 } else { sp as i32 });
            let s = state.side_mut(side);
            if let (Some(idx), Some(sp)) = (s.active()[slot].pokemon_index, sp) {
                s.speed_cache[idx as usize] = sp as i16;
            }
        }
    }
    out
}

/// PS `runEvent('ModifyDamage')` Friend Guard handlers (`onAnyModifyDamage`,
/// data/abilities.ts friendguard): one per alive active Friend Guard holder,
/// sorted by its cached speed. Returns the tie-group sizes (>= 2) in
/// descending speed order.
/// TODO(PS parity): other ModifyDamage handlers (defender / attacker abilities,
/// items) can tie with these too; fainted holders are not counted.
fn friend_guard_tie_groups(state: &DoublesState) -> [u8; 4] {
    let mut speeds: Vec<i32> = Vec::new();
    for sd in [Side::One, Side::Two] {
        for sl in 0..2 {
            if let Some(p) = state.side(sd).slot_pokemon(sl) {
                if p.hp > 0 && p.ability == Abilities::FRIENDGUARD {
                    speeds.push(cached_speed(state, sd, sl));
                }
            }
        }
    }
    speeds.sort_unstable_by(|a, b| b.cmp(a));
    let mut out = [0u8; 4];
    let mut g = 0;
    let mut i = 0;
    while i < speeds.len() {
        let mut j = i;
        while j < speeds.len() && speeds[j] == speeds[i] {
            j += 1;
        }
        if j - i > 1 && g < out.len() {
            out[g] = (j - i) as u8;
            g += 1;
        }
        i = j;
    }
    out
}

/// The cached PS `pokemon.speed` of a slot's occupant.
/// PS move target type (`ps_move_target`) per move, indexed by `Choices`
/// (built once; the generated table matches on move id strings).
static MOVE_TARGETS: std::sync::LazyLock<Vec<Tgt>> = std::sync::LazyLock::new(|| {
    (0..=u16::MAX)
        .map(Choices::from)
        .enumerate()
        .take_while(|&(i, m)| m as usize == i)
        .map(|(_, m)| Tgt::from_ps(crate::doubles::move_targets::ps_move_target(&m.to_string().to_lowercase())))
        .collect()
});

/// PS `move.target` of a move ("normal", "allAdjacentFoes", ...; "" if none).
pub(crate) fn move_target(m: Choices) -> Tgt {
    MOVE_TARGETS[m as usize]
}

pub(crate) fn cached_speed(state: &DoublesState, side: Side, slot: usize) -> i32 {
    let s = state.side(side);
    s.active()[slot].pokemon_index.map(|i| s.speed_cache[i as usize] as i32).unwrap_or(0)
}

/// PS `checkFainted` (end of turn): on a side that can still switch, fainted
/// pokemon get status 'fnt' (the fuzzer reads it as none). A side with no
/// replacement keeps the fainted pokemon's last status.
fn check_fainted(state: &mut DoublesState) {
    // PS `runAction`: `if (this.ended) return true;` comes before `checkFainted()`,
    // so a battle that ended this turn keeps the fainted pokemon's status.
    if state.battle_is_over() != 0.0 {
        return;
    }
    for side in [Side::One, Side::Two] {
        if possible_switches(state, side).is_empty() {
            continue;
        }
        let s = state.side_mut(side);
        s.ver = s.ver.wrapping_add(1);
        for i in 0..6 {
            if s.team[i].hp <= 0 && s.team[i].id != crate::pokemon::PokemonName::NONE {
                s.team[i].status = PokemonStatus::NONE;
                s.team[i].sleep_turns = 0;
                s.toxic_stage[i] = 0;
            }
        }
    }
}

/// PS `possibleSwitches`: bench pokemon (positions 2.. of `side.pokemon`)
/// that have not fainted, in PS order.
fn possible_switches(state: &DoublesState, side: Side) -> Vec<usize> {
    let s = state.side(side);
    s.ps_order[2..]
        .iter()
        .copied()
        .filter(|&t| s.team[t].hp > 0 && s.team[t].id != crate::pokemon::PokemonName::NONE)
        .collect()
}

/// PS `dragIn` for every active flagged by a phazing move: a random
/// replacement (`getRandomSwitchable` -> `sample`), switched in with
/// `isDrag`, which runs `runSwitch` at once.
/// PS `switchIn` (sim/battle-actions.ts): a live pokemon leaving the field runs
/// `singleEvent('End', ability)` before the swap. Cloud Nine / Air Lock `onEnd`
/// runs `eachEvent('WeatherChange')`, a speed sort of all actives. A fainted
/// pokemon (faint replacements) runs no End event here; its faint handles it.
fn switch_out_end_event(state: &DoublesState, rng: &mut Gen5Rng, side: Side, slot: usize) {
    let live_cloud = state
        .side(side)
        .slot_pokemon(slot)
        .map(|p| p.hp > 0 && matches!(p.ability, Abilities::CLOUDNINE | Abilities::AIRLOCK))
        .unwrap_or(false);
    if live_cloud {
        let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
    }
}

/// PS `fieldEvent('SwitchIn')` runs `resolveFaints()` after each handler and
/// skips a fainted holder's handlers: entry hazards (side conditions, subOrder
/// 4) run before the ability's Start (subOrder 7), so a pokemon KO'd by Stealth
/// Rock / Spikes never runs Intimidate & co. PS sim/battle.ts fieldEvent.
fn switched_in_alive(state: &DoublesState, side: Side, slot: usize) -> bool {
    state.side(side).slot_pokemon(slot).is_some_and(|p| p.hp > 0)
}

fn drag_in_all(state: &mut DoublesState, rng: &mut Gen5Rng) -> Vec<(Side, PokemonIndex)> {
    let mut dragged = Vec::new();
    for side in [Side::One, Side::Two] {
        for slot in 0..2 {
            if !state.side(side).active()[slot].drag_out {
                continue;
            }
            state.side_mut(side).slot_scratch_mut(slot).drag_out = false;
            if !state.side(side).slot_pokemon(slot).map(|p| p.hp > 0).unwrap_or(false) {
                continue;
            }
            let options = possible_switches(state, side);
            if options.is_empty() {
                continue;
            }
            let pick = options[rng.label("drag in").random(options.len() as u32) as usize];
            if let Some(out) = state.side(side).active()[slot].pokemon_index {
                dragged.push((side, out));
            }
            switch_out_end_event(state, rng, side, slot);
            switch_in(state, side, slot, pick);
            let _ = sorted_actives(state, rng, "speed tie (switch-in)");
            apply_sticky_web(state, side, slot);
            if switched_in_alive(state, side, slot) && apply_on_switch_abilities(state, side, slot, rng) {
                let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
            }
        }
    }
    dragged
}

/// PS roost `condition.onType`: Flying filtered out of the types (a single
/// remaining type; "???" when none is left).
fn roosted_types(t: (PokemonType, PokemonType)) -> (PokemonType, PokemonType) {
    let f = |x: PokemonType| if x == PokemonType::FLYING { PokemonType::TYPELESS } else { x };
    let nt = if t.0 == PokemonType::FLYING { (t.1, PokemonType::TYPELESS) } else { (t.0, f(t.1)) };
    if nt.0 == PokemonType::TYPELESS && nt.1 != PokemonType::TYPELESS { (nt.1, nt.0) } else { nt }
}


/// PS re-sort of the remaining queue after an action (gen 8+ `runAction`
/// tail): fresh speeds, `getTarget` for every queued move, then `speedSort`.
fn resort_queue(state: &mut DoublesState, rng: &mut Gen5Rng, queue: &mut [Action]) {
    if queue.is_empty() {
        return;
    }
    let speeds = update_speed_get(state);
    for a in queue.iter_mut() {
        // PS `getActionSpeed` (every re-sort) recomputes the priority from the
        // current ability: a Skill Swap'd Prankster boosts a queued Status move.
        // Fixed-priority actions (recharge, locked copy, After You) keep theirs.
        if !a.recharge && !a.no_pp && !a.prioritized {
            let p = state.side(a.side).team_member(a.pokemon_index);
            // A pokemon no longer active (fainted) has its ability ignored
            // (PS sim/pokemon.ts ignoringAbility: gen 5+ `!this.isActive`).
            let active = (0..2).any(|sl| state.side(a.side).active()[sl].pokemon_index == Some(a.pokemon_index));
            let ab = if active { ability_priority_boost(p, &a.choice) } else { 0 };
            let now = ab + grassy_glide_boost(state, p, &a.choice);
            a.priority += now - a.boost;
            a.boost = now;
        }
        // Actions follow the pokemon: an Ally Switch moves it to the other slot.
        if let Some(sl) = (0..2).find(|&sl| state.side(a.side).active()[sl].pokemon_index == Some(a.pokemon_index)) {
            a.slot = sl;
            if let Some(sp) = speeds[a.side as usize][a.slot] {
                a.speed = sp;
            }
        } else {
            // A fainted pokemon's queued action: PS `getActionSpeed` recomputes
            // from its stats; `clearVolatile` on faint reset its boosts.
            let sd = state.side(a.side);
            let p = sd.team_member(a.pokemon_index);
            let sp = multiply_boost(sd.team_boosts[a.pokemon_index as usize].spe, p.speed);
            a.speed = if state.trick_room.active { -sp } else { sp };
        }
        let _ = ps_get_target(state, rng, a);
    }
    speed_sort(queue, rng);
}

/// Execute a single queued move, mirroring pokemon-showdown's
/// `useMoveInner` -> `trySpreadMoveHit` -> `hitStepMoveHitLoop` ->
/// `spreadMoveHit` pipeline, consuming PRNG draws in the exact same order:
///
///   1. accuracy            -> `random(100)`      (skipped for always-hit)
///   2. multihit count      -> `random(20)`       (only for 2-5 hit moves)
///   3. per hit: crit       -> `random(critMult)`
///   4. per hit: damage     -> `random(16)`
///   5. self drops          -> `random(100)`      (only if self.boosts + chance)
///   6. per secondary       -> `random(100)`
///
/// Type-immune and fixed-damage moves make no crit/damage draws.
/// PS `flags.breakable` (data/abilities.ts, champions mod).
fn is_breakable(a: Abilities) -> bool {
    crate::doubles::ability_flags::has_flag(a, crate::doubles::ability_flags::BREAKABLE)
}

/// Mold Breaker `onModifyMove` (`move.ignoreAbility`): while the move runs,
/// every other active's breakable ability is skipped (PS sim/battle.ts
/// `suppressingAbility`, checked in `runEvent` / `isGrounded`). Modelled by
/// blanking those abilities until `mold_breaker_restore`; moves that read an
/// ability directly (Skill Swap, Role Play, ...) use `real_ability`.
// TODO(PS parity): suppression ends early if the user faints mid-move
// (`activePokemon.isActive`); a pokemon dragged in mid-move is not blanked.
fn mold_breaker_suppress(state: &mut DoublesState, action: &Action) -> bool {
    let user = state.side(action.side).team_member(action.pokemon_index).ability;
    let ignores = user == Abilities::MOLDBREAKER
        || matches!(action.choice.move_id, Choices::MOONGEISTBEAM | Choices::PHOTONGEYSER | Choices::SUNSTEELSTRIKE);
    if !ignores || action.recharge {
        return false;
    }
    let mut saved = arrayvec::ArrayVec::new();
    for sd in [Side::One, Side::Two] {
        for sl in 0..2 {
            if (sd, sl) == (action.side, action.slot) {
                continue;
            }
            let Some(i) = state.side(sd).active()[sl].pokemon_index else { continue };
            let a = state.side(sd).team_member(i).ability;
            if is_breakable(a) {
                saved.push((sd, i, a));
                state.side_mut(sd).team_member_mut(i).ability = Abilities::NONE;
            }
        }
    }
    let any = !saved.is_empty();
    state.mold_saved = saved;
    any
}

/// Undo `mold_breaker_suppress`. An ability changed during the move (Simple
/// Beam, Worry Seed, a switch-out reset, ...) is kept.
fn mold_breaker_restore(state: &mut DoublesState) {
    for (sd, i, a) in std::mem::take(&mut state.mold_saved) {
        let pm = state.side_mut(sd).team_member_mut(i);
        if pm.ability == Abilities::NONE {
            pm.ability = a;
        }
    }
}

/// The ability as PS's direct reads (`target.ability`, `getAbility()`) see it,
/// i.e. ignoring a Mold Breaker blank.
/// The `onTryHit` of a Status move that substitute blocks (champions pool:
/// electrify, entrainment, gastroacid, lockon, simplebeam, worryseed, yawn):
/// true when it returns false / null. PS data/moves.ts `<move>.onTryHit`.
#[allow(clippy::too_many_arguments)]
fn status_on_try_hit_fails(
    state: &DoublesState,
    side: Side,
    slot: usize,
    idx: PokemonIndex,
    ts: Side,
    tsl: usize,
    choice: &Choice,
    queue: &[Action],
) -> bool {
    let Some(ti) = state.side(ts).active()[tsl].pokemon_index else {
        return false;
    };
    let ta = real_ability(state, ts, ti);
    let cant = crate::engine::mechanics::cant_suppress(ta);
    match choice.move_id {
        Choices::ELECTRIFY => {
            let will_move = queue.iter().any(|q| q.side == ts && q.pokemon_index == ti);
            !will_move && state.side(ts).active()[tsl].active_turns > 0
        }
        Choices::ENTRAINMENT => {
            let sa = real_ability(state, side, idx);
            (ts == side && tsl == slot)
                || ta == sa
                || cant
                || ta == Abilities::TRUANT
                || crate::engine::mechanics::no_entrain(sa)
        }
        // (Ability Shield is not in the item pool.)
        Choices::GASTROACID => cant,
        Choices::LOCKON => state.side(side).active()[slot].lock_on.is_some(),
        Choices::SIMPLEBEAM => cant || ta == Abilities::SIMPLE || ta == Abilities::TRUANT,
        Choices::WORRYSEED => cant,
        // TODO(PS parity): `runStatusImmunity('slp')` (Insomnia, Vital Spirit, ...).
        Choices::YAWN => state.side(ts).team_member(ti).status != PokemonStatus::NONE,
        _ => false,
    }
}

#[inline]
pub(crate) fn real_ability(state: &DoublesState, side: Side, idx: PokemonIndex) -> Abilities {
    if state.mold_saved.is_empty() {
        return state.side(side).team_member(idx).ability;
    }
    state
        .mold_saved
        .iter()
        .find(|&&(s, i, _)| s == side && i == idx)
        .map(|&(_, _, a)| a)
        .unwrap_or(state.side(side).team_member(idx).ability)
}

/// PS `pokemon.getAbility()`: the held ability even while the `gastroacid`
/// volatile suppresses it (the engine stores it in `gastro_ability`).
fn held_ability(state: &DoublesState, side: Side, slot: usize, idx: PokemonIndex) -> Abilities {
    let a = &state.side(side).active()[slot];
    if a.gastro_acid && a.pokemon_index == Some(idx) {
        a.gastro_ability
    } else {
        real_ability(state, side, idx)
    }
}

/// PS `cantsuppress` abilities: Gastro Acid fails on them (onTryHit), and
/// Baton Pass's copied `gastroacid` is dropped on them (onCopy).
fn gastro_cant_suppress(a: Abilities) -> bool {
    crate::doubles::ability_flags::has_flag(a, crate::doubles::ability_flags::CANTSUPPRESS)
}

/// Drop the Mold Breaker blank of a pokemon whose ability really became
/// No Ability (Gastro Acid), so the restore keeps it suppressed.
fn mold_breaker_forget(state: &mut DoublesState, side: Side, idx: PokemonIndex) {
    state.mold_saved.retain(|&mut (s, i, _)| (s, i) != (side, idx));
}

/// Pressure `onDeductPP`: one extra PP per Pressure foe among the move's
/// `pressureTargets` (PS sim/battle-actions.ts useMoveInner,
/// sim/pokemon.ts getMoveTargets; `mustpressure` moves count every foe,
/// `foeSide` moves none).
fn pressure_deduct_pp(state: &mut DoublesState, action: &Action) {
    let pre = state.tl.pressure_extra.get();
    let extra = if pre >= 0 { pre } else { pressure_extra_count(state, action) };
    if extra == 0 {
        return;
    }
    let s = state.side_mut(action.side);
    let mv = &mut s.team_member_mut(action.pokemon_index).moves[&action.choice.move_index];
    mv.pp = (mv.pp - extra).max(0);
}

/// PS `useMoveInner` `pressureTargets` (`DeductPP` from each Pressure foe):
/// the number of foes that pay one PP. Evaluated before the hit.
fn pressure_extra_count(state: &DoublesState, action: &Action) -> i8 {
    let foe = action.side.other();
    let ty = move_target(action.choice.move_id);
    let all_foes = matches!(
        action.choice.move_id,
        Choices::IMPRISON | Choices::SPIKES | Choices::STEALTHROCK | Choices::TOXICSPIKES
    ) || matches!(ty, Tgt::AllAdjacentFoes | Tgt::AllAdjacent)
        // PS getMoveTargets: `all` targets `alliesAndSelf()` plus `foes(true)`
        // (Wonder Room, Sunny Day, ...); `allySide` / `allyTeam` no foes.
        || ty == Tgt::All;
    let single = matches!(ty, Tgt::Normal | Tgt::Any | Tgt::AdjacentFoe | Tgt::RandomNormal);
    let slots: smallvec::SmallVec<[usize; 2]> = if all_foes {
        smallvec::smallvec![0, 1]
    } else if single && state.tl.pressure_target.get().is_some() {
        // PS `pressureTargets` = the resolved target (post retarget / redirect).
        match state.tl.pressure_target.get().flatten() {
            Some((ts, tsl)) if ts == foe => smallvec::smallvec![tsl],
            _ => smallvec::SmallVec::new(),
        }
    } else if single && action.target_loc > 0 {
        smallvec::smallvec![(action.target_loc - 1) as usize]
    } else {
        smallvec::SmallVec::new()
    };
    let extra = slots
        .into_iter()
        .filter(|&sl| sl < 2)
        .filter(|&sl| state.side(foe).slot_pokemon(sl).is_some_and(|p| p.hp > 0 && p.ability == Abilities::PRESSURE))
        .count() as i8;
    extra
}

/// PS `runMove` -> `runEvent('AfterMove', pokemon)`: the handlers are the
/// user's own `onAfterMove` volatiles (lockedmove, charge, beakblast), each
/// `onAnyAfterMove` holder among the user's side and the foes (Opportunist),
/// keyed by holder speed. Two or more with equal keys shuffle (one draw per
/// swap). PS data/conditions.ts lockedmove, data/moves.ts charge / beakblast,
/// data/abilities.ts opportunist; sim/battle.ts findEventHandlers.
fn after_move_handler_ties(state: &DoublesState, rng: &mut Gen5Rng, side: Side, idx: PokemonIndex, lock_seen: bool) {
    let Some(slot) = (0..2).find(|&sl| state.side(side).active()[sl].pokemon_index == Some(idx)) else { return };
    if !state.side(side).slot_pokemon(slot).is_some_and(|p| p.hp > 0) {
        return;
    }
    let a = &state.side(side).active()[slot];
    let speed = cached_speed(state, side, slot);
    let mut hs: Vec<(i64, i32, i64)> = Vec::new();
    if lock_seen || a.locked_move.is_some() {
        hs.push((4294967296, speed, 2));
    }
    if a.charge {
        hs.push((4294967296, speed, 2));
    }
    if a.beak_blast {
        hs.push((4294967296, speed, 2));
    }
    for sd in [Side::One, Side::Two] {
        for sl in 0..2 {
            let Some(hidx) = state.side(sd).active()[sl].pokemon_index else { continue };
            if state.side(sd).slot_pokemon(sl).is_some_and(|p| p.hp > 0)
                && real_ability(state, sd, hidx) == Abilities::OPPORTUNIST
            {
                hs.push((4294967296, cached_speed(state, sd, sl), 7));
            }
        }
    }
    if hs.len() >= 2 {
        sort_handler_ties(&mut hs, rng, "speed tie (move order)");
    }
}

fn execute_action_body(
    state: &mut DoublesState,
    rng: &mut Gen5Rng,
    action: &Action,
    move_order: &mut Vec<OrderEntry>,
    queue: &[Action],
) {
    let will_act = !queue.is_empty();
    crate::doubles::damage::set_gravity(state.gravity.active);
    // Analytic: every other (unfainted) active has no pending action.
    state.tl.analytic_no_move.set(![Side::One, Side::Two].iter().any(|&sd| {
            (0..2).any(|sl| {
                let a = &state.side(sd).active()[sl];
                (sd, sl) != (action.side, action.slot)
                    && state.side(sd).slot_pokemon(sl).map(|p| p.hp > 0).unwrap_or(false)
                    && queue.iter().any(|q| q.side == sd && Some(q.pokemon_index) == a.pokemon_index)
            })
        }));
    state.tl.pressure_target.set(None);
    state.tl.pressure_extra.set(-1);
    // PS `runMove`: a locked move (lockedmove / twoturnmove firing turn / uproar)
    // runs with `sourceEffect` = the lock condition, so `useMoveInner`'s Pressure
    // block (`!sourceEffect || callerMoveForPressure`) is skipped. Externally
    // called moves keep their caller as the source.
    let pressure_skipped = {
        let a = &state.side(action.side).active()[action.slot];
        !(a.external_call || action.no_pp)
            && (a.locked_move.is_some() || a.uproar > 0 || (a.charging.is_some() && !a.charge_aborted))
    };
    let before = state.side(action.side).active()[action.slot].charging;
    let was_recharging = state.side(action.side).active()[action.slot].recharging;
    let order_len = move_order.len();
    state.side_mut(action.side).slot_scratch_mut(action.slot).move_result_this_turn = None;
    // PS destinybond `onBeforeMovePriority: -1` (and `onMoveAborted`): any move but
    // Destiny Bond removes the volatile before the move's effects, so a self-faint
    // such as Memento does not take the attacker down (PS data/conditions.ts destinybond).
    if action.choice.move_id != Choices::DESTINYBOND {
        let pre_slot = state
            .side(action.side)
            .active()
            .iter()
            .position(|a| a.pokemon_index == Some(action.pokemon_index))
            .unwrap_or(action.slot);
        if state.side(action.side).active()[pre_slot].destinybond {
            state.side_mut(action.side).active_slot_mut(pre_slot).destinybond = false;
        }
    }
    state.in_move = true;
    let mold_breaker = mold_breaker_suppress(state, action);
    state.tl.after_move_tie_drawn.set(false);
    state.tl.no_target_fail.set(false);
    state.tl.before_move_aborted.set(false);
    execute_action_inner(state, rng, action, move_order, will_act, queue);
    // PS runMove: a failed BeforeMove runs `runEvent('MoveAborted')` (charge,
    // twoturnmove, destinybond handlers: a sort). PS sim/battle-actions.ts runMove.
    // TODO(PS parity): Destiny Bond is cleared before the move here, so its
    // handler is not in this sort.
    if state.tl.before_move_aborted.replace(false) {
        if let Some(sl) = (0..2).find(|&sl| state.side(action.side).active()[sl].pokemon_index == Some(action.pokemon_index)) {
            crate::doubles::events::run_event_ties(state, rng, crate::doubles::events::Event::MoveAborted, (action.side, sl), None);
        }
    }
    if mold_breaker {
        mold_breaker_restore(state);
    }
    // `runAction` -> `eachEvent('Update')`: the abilities' `onUpdate` cures
    // (a status set through Mold Breaker, a swapped-in Limber, ...).
    for sd in [Side::One, Side::Two] {
        for sl in 0..2 {
            ability_update_cures(state, sd, sl);
        }
    }
    state.in_move = false;
    if move_order.len() > order_len && !pressure_skipped {
        pressure_deduct_pp(state, action);
    }
    // Destiny Bond `onBeforeMove` (any other move) / `onMoveAborted` (a `cant`)
    // remove the volatile. PS data/conditions.ts destinybond.
    // The user may have moved slots during its own action (Ally Switch swaps the
    // whole slot, volatiles included), so find it by pokemon index.
    let db_slot = state
        .side(action.side)
        .active()
        .iter()
        .position(|a| a.pokemon_index == Some(action.pokemon_index))
        .unwrap_or(action.slot);
    let had_db = state.side(action.side).active()[db_slot].destinybond;
    // Any other move already removed it before running (above), so a Destiny
    // Bond called by Sleep Talk / Copycat during that move keeps its volatile.
    if action.choice.move_id == Choices::DESTINYBOND && move_order.len() == order_len {
        state.side_mut(action.side).active_slot_mut(db_slot).destinybond = false;
    }
    // Destiny Bond itself: `volatileStatus: 'destinybond'` is added when the
    // move executes (PS data/moves.ts destinybond; no `beforeTurnCallback`).
    if action.choice.move_id == Choices::DESTINYBOND
        && move_order.len() > order_len
        && !had_db // onPrepareHit: `!pokemon.removeVolatile('destinybond')` fails the move
        && state.side(action.side).active()[db_slot].pokemon_index == Some(action.pokemon_index)
    {
        state.side_mut(action.side).active_slot_mut(db_slot).destinybond = true;
    }
    if action.choice.move_id == Choices::DESTINYBOND && had_db && move_order.len() > order_len {
        state.side_mut(action.side).active_slot_mut(db_slot).destinybond = false;
    }
    if move_order.len() > order_len && !std::mem::take(&mut state.copy_called) {
        state.last_move = Some(action.choice.move_id);
        // PS `pokemon.moveUsed`: `lastMove` (Encore). The user's current slot
        // (its own Ally Switch moved it).
        if let Some(sl) = (0..2).find(|&sl| state.side(action.side).active()[sl].pokemon_index == Some(action.pokemon_index)) {
            state.side_mut(action.side).slot_scratch_mut(sl).last_move = Some(action.choice.move_id);
        }
    }
    // PS runMove `runEvent('AfterMove')`: the handlers are sorted before any of
    // them runs (charge removal, lockedmove release below). Only for a move that ran.
    if move_order.len() > order_len && !state.tl.after_move_tie_drawn.get() {
        after_move_handler_ties(state, rng, action.side, action.pokemon_index, false);
    }
    // Opportunist `onAnyAfterMove` (every holder, after the move).
    if move_order.len() > order_len {
        for sd in [Side::One, Side::Two] {
            for sl in 0..2 {
                flush_opportunist(state, sd, sl);
            }
        }
    }
    // charge `onAfterMove` / `onMoveAborted`: an Electric move uses it up.
    // A recharge turn runs the typeless `recharge` pseudo-move, whatever move
    // the player picked (PS sim/side.ts chooseMove), so it keeps the charge.
    if action.choice.move_type == PokemonType::ELECTRIC
        && action.choice.move_id != Choices::CHARGE
        && !was_recharging
        && !action.recharge
    {
        state.side_mut(action.side).active_slot_mut(action.slot).charge = false;
    }
    // PS `moveThisTurnResult`: a BeforeMove abort counts as a failure
    // (`runMove`), except the recharge turn; otherwise a miss / immunity /
    // failed effect sets `false` where it happens, and the rest succeeded.
    {
        // `move_result_this_turn` is not read by `events::vol_mask`.
        let a = state.side_mut(action.side).slot_scratch_mut(action.slot);
        if a.pokemon_index.is_some() {
            if move_order.len() == order_len && !was_recharging {
                a.move_result_this_turn = Some(false);
            } else if a.move_result_this_turn.is_none() {
                a.move_result_this_turn = Some(true);
            }
        }
    }
    // lockedmove `onAfterMove` (any move the holder used, including one that
    // missed or was blocked early): removed when duration is 1; `onEnd` then
    // confuses the user unless trueDuration > 1. PS data/conditions.ts.
    // The user is found by pokemon: Ally Switch in this move may have moved it.
    let holder = (0..2)
        .find(|&sl| state.side(action.side).active()[sl].pokemon_index == Some(action.pokemon_index))
        .unwrap_or(action.slot);
    if move_order.len() > order_len && state.side(action.side).slot_pokemon(holder).is_some() {
        if let Some((1, t)) = state.side(action.side).active()[holder].locked_move {
            state.side_mut(action.side).active_slot_mut(holder).locked_move = None;
            if t <= 1 {
                try_confuse(state, rng, action.side, holder);
            }
        }
    }
    // Crash damage (Jump Kick, High Jump Kick, Axe Kick, Supercell Slam):
    // `onMoveFail` after `useMoveInner` found `moveResult` false (a miss,
    // Protect, an immunity, ...). PS data/moves.ts jumpkick / supercellslam
    // (`damage(baseMaxhp / 2)`), sim/battle-actions.ts useMoveInner.
    let no_target_fail = state.tl.no_target_fail.get();
    if move_order.len() > order_len && !no_target_fail {
        move_fail_self_damage(state, action.side, action.slot, &action.choice);
    }
}

/// `onMoveFail` self-damage, run by `useMoveInner` when `moveResult` is false
/// (a miss, Protect, an immunity, ...) for the move that was used, a called
/// one included. Crash damage (Jump Kick, High Jump Kick, Axe Kick, Supercell
/// Slam): `damage(baseMaxhp / 2)`. Steel Beam `onMoveFail` / Mind Blown,
/// Chloroblast `mindBlownRecoil`: the user loses round(maxhp / 2) (Magic Guard
/// blocks it); Steel Beam's hit recoil is its `heal` entry. PS data/moves.ts
/// jumpkick / supercellslam / steelbeam, sim/battle-actions.ts useMoveInner,
/// recoilDamage.
fn move_fail_self_damage(state: &mut DoublesState, side: Side, slot: usize, choice: &Choice) {
    if state.side(side).active()[slot].move_result_this_turn != Some(false) {
        return;
    }
    if choice.crash.is_some() {
        if let Some(p) = state.side(side).slot_pokemon(slot) {
            if p.hp > 0 {
                let dmg = (p.maxhp / 2).max(1);
                apply_damage(state, side, slot, dmg);
            }
        }
    }
    if matches!(choice.move_id, Choices::STEELBEAM | Choices::MINDBLOWN | Choices::CHLOROBLAST) {
        if let Some(p) = state.side(side).slot_pokemon(slot) {
            if p.hp > 0 && p.ability != Abilities::MAGICGUARD {
                let dmg = ((p.maxhp as f32 / 2.0).round() as i16).max(1);
                apply_damage(state, side, slot, dmg);
            }
        }
    }
}

fn execute_action(
    state: &mut DoublesState,
    rng: &mut Gen5Rng,
    action: &Action,
    move_order: &mut Vec<OrderEntry>,
    queue: &[Action],
) {
    let before = state.side(action.side).active()[action.slot].charging;
    state.copy_called = false;
    let logged_before = move_order.len();
    execute_action_body(state, rng, action, move_order, queue);
    // PS beakblast `onAfterMove`: the user's Beak Blast volatile ends with its move.
    // A failed BeforeMove (flinch, sleep, ...) skips AfterMove (PS sim/battle-actions.ts
    // runMove), so the volatile stays until its duration ends in the residual.
    // PS spitup `onAfterMove`: Stockpile ends after every use that got past
    // BeforeMove, whatever stopped the hit (miss, semi-invulnerable target,
    // immunity, Protect). `end_stockpile` is a no-op when already ended.
    if action.choice.move_id == Choices::SPITUP && move_order.len() > logged_before {
        if let Some(sl) = (0..2).find(|&sl| state.side(action.side).active()[sl].pokemon_index == Some(action.pokemon_index)) {
            end_stockpile(state, action.side, sl);
        }
    }
    if action.choice.move_id == Choices::BEAKBLAST && move_order.len() > logged_before {
        for sl in 0..2 {
            let a = state.side_mut(action.side).active_slot_mut(sl);
            if a.pokemon_index == Some(action.pokemon_index) {
                a.beak_blast = false;
            }
        }
    }
    // PS `runMove`: a failed BeforeMove runs `MoveAborted`, which removes
    // `twoturnmove` (data/conditions.ts). The charge was not consumed. The body
    // returns early for every BeforeMove failure (sleep, flinch, ...), so this
    // runs after it, not at its end.
    let c = state.side(action.side).active()[action.slot].charging;
    if let (Some(b), Some(c)) = (before, c) {
        if b.0 == c.0 && b.1 == c.1 {
            state.side_mut(action.side).active_slot_mut(action.slot).charge_aborted = true;
        }
    }
}

fn execute_action_inner(
    state: &mut DoublesState,
    rng: &mut Gen5Rng,
    action: &Action,
    move_order: &mut Vec<OrderEntry>,
    will_act: bool,
    queue: &[Action],
) {
    let side = action.side;
    let slot = action.slot;
    let idx = action.pokemon_index;
    // Glaive Rush `onBeforeMove` (priority 100): the drawback ends when the
    // holder tries to move.
    if state.side(side).active()[slot].glaive_rush {
        state.side_mut(side).active_slot_mut(slot).glaive_rush = false;
    }
    // Illusion (`onBeforeSwitchIn`): shown as the last unfainted pokemon after
    // it in `side.pokemon` until a damaging hit ends it.
    let name = {
        let sd = state.side(side);
        let mut disguise: Option<usize> = None;
        if sd.team_member(idx).ability == Abilities::ILLUSION && !sd.active()[slot].illusion_broken {
            disguise = match sd.active()[slot].illusion_target {
                Some(t) => t,
                None => illusion_disguise(sd, idx),
            };
        }
        match disguise {
            Some(t) => sd.species_names[t],
            None => sd.species_names[idx as usize],
        }
    };
    if state.side(side).active()[slot].illusion_target.is_none()
        && state.side(side).team_member(idx).ability == Abilities::ILLUSION
    {
        let d = illusion_disguise(state.side(side), idx);
        state.side_mut(side).slot_scratch_mut(slot).illusion_target = Some(d);
    }
    // The action is logged once: a spread move that bounces (Magic Bounce) logs
    // its own entry before the holder's copy runs, then keeps going.
    let outer_logged = std::cell::Cell::new(false);
    let record = state.record;
    // Always kept (cheap): the battle logic reads the entry count.
    let push_order = |move_order: &mut Vec<OrderEntry>| {
        if outer_logged.replace(true) {
            return;
        }
        if crate::doubles::debug::flags().dmg {
            eprintln!("PUSHORDER {} {}", name, action.choice.move_id);
        }
        move_order.push(order_entry(side, slot, name, action.choice.move_id))
    };

    // `useMove` called by another move (Copycat, Instruct, ...). A locked move
    // that is not one of the user's own (`no_pp`: Copycat -> Outrage, next
    // turn) is a normal `runMove` (BeforeMove runs; only `deductPP` finds no
    // slot), so it is not external. PS sim/battle-actions.ts runMove / useMove.
    let external = state.side(side).active()[slot].external_call;
    // The attacker must still be on the field.
    if state.side(side).active()[slot].pokemon_index.is_none() {
        return;
    }

    resolve_faints(state, rng, |_, _| true);

    // PS `runMove` starts by counting the move action (`activeMoveActions++`,
    // before the target and the BeforeMove checks). sim/battle-actions.ts runMove.
    // A move called from another move (Copycat, Metronome, Sleep Talk, ...) goes
    // through `useMove`, which does not count (Fake Out's `onTry` then sees 1).
    if !state.side(side).active()[slot].external_call {
        state.side_mut(side).slot_scratch_mut(slot).active_move_actions += 1;
    }
    // PS `runMove` resolves the target before the BeforeMove checks.
    // Stalwart / Propeller Tail set `tracksTarget` (`getTarget` with an active
    // `originalTarget` returns it without a draw or a redirect).
    // PS sim/battle.ts getTarget: `tracksTarget && originalTarget?.isActive`
    // returns the pokemon chosen when the action was queued, wherever it is now
    // (Ally Switch). Snipe Shot has `tracksTarget` itself (data/moves.ts).
    let tracks = matches!(
        state.side(side).team_member(action.pokemon_index).ability,
        Abilities::STALWART | Abilities::PROPELLERTAIL
    ) || action.choice.move_id == Choices::SNIPESHOT;
    // A called move (Sleep Talk, Copycat, ...) has no queued `originalTarget`.
    let tracked = if tracks && action.target_loc != 0 && !state.side(side).active()[slot].external_call {
        action.orig_target.and_then(|(ts, ti)| {
            (0..2).find(|&sl| {
                state.side(ts).active()[sl].pokemon_index == Some(ti)
                    && state.side(ts).team_member(ti).hp > 0
            })
            .map(|sl| (ts, sl))
        })
    } else {
        None
    };
    // A Magic Bounce copy of a spread move (allAdjacent) is `useMove` with its
    // target given: PS `getMoveTargets` never asks for a random target, so no draw.
    let bounce_spread_copy = state.side(side).active()[slot].bounced
        && matches!(
            move_target(action.choice.move_id),
            Tgt::AllAdjacent | Tgt::AllAdjacentFoes
        );
    // A called Metal Burst / Comeuppance (Copycat, ...) is `useMove` with no
    // target: `onModifyTarget` sets it to `getAtSlot(lastDamagedBy.slot)` before
    // `getRandomTarget`, so there is no draw when the user was ever damaged.
    // PS sim/battle-actions.ts useMoveInner (ModifyTarget); data/moves.ts metalburst.
    let scripted_called = if state.side(side).active()[slot].external_call
        && matches!(action.choice.move_id, Choices::METALBURST | Choices::COMEUPPANCE)
    {
        last_damaged_by(state, side, action.pokemon_index).map(|(s_, sl_, _, _, _)| (s_, sl_))
    } else {
        None
    };
    let resolved = match tracked {
        Some(t) => Some(t),
        None if scripted_called.is_some() => scripted_called,
        None if bounce_spread_copy && action.target_loc != 0 => {
            let loc = action.target_loc;
            Some(if loc > 0 { (side.other(), (loc - 1) as usize) } else { (side, (-loc - 1) as usize) })
        }
        // `getMoveTargets`: `priorityEvent('RedirectTarget')` after the target is chosen.
        None => ps_get_target(state, rng, action).map(|t| redirect_target(state, action, t)),
    };
    state.tl.pressure_target.set(Some(resolved));
    let pre_extra = pressure_extra_count(state, action);
    state.tl.pressure_extra.set(pre_extra);
    // PS `runMove` -> `moveUsed(move, targetLoc)` stores the action's own
    // targetLoc (before any redirect), not the pokemon it finally hit. Instruct,
    // Sleep Talk-style locks and Encore read it back. It runs after BeforeMove
    // (a flinch / sleep / paralysis keeps the old one), with `lastMove` below.
    // PS sim/battle-actions.ts runMove; sim/pokemon.ts moveUsed.
    let used_target_loc = if action.target_loc != 0 {
        Some(action.target_loc)
    } else {
        resolved.map(|(ts, tsl)| if ts == side { -(tsl as i8 + 1) } else { tsl as i8 + 1 })
    };

    // `useMove` (external: Magic Bounce, Sleep Talk, ...) skips BeforeMove, but
    // `useMoveInner` runs `runEvent('ModifyMove')`: Throat Chop (sound) and
    // Heal Block (`heal` flag) `onModifyMove` return false, so `move` is false
    // and `baseTarget !== move.target` calls `getRandomTarget(pokemon, false)`,
    // a `randomFoe` sample when a foe is up; then the move does nothing.
    // PS sim/battle-actions.ts useMoveInner, sim/battle.ts getRandomTarget,
    // data/moves.ts throatchop / healblock `condition.onModifyMove`.
    if external
        && ((state.side(side).active()[slot].throat_chop > 0 && action.choice.flags.sound)
            || (state.side(side).active()[slot].heal_block > 0 && action.choice.flags.heal))
    {
        let foes = (0..2).filter(|&sl| state.side(side.other()).slot_pokemon(sl).is_some_and(|p| p.hp > 0)).count();
        if foes > 0 {
            let _ = rng.label("retarget").random(foes as u32);
        }
        return;
    }
    if !external {
    // PS `runEvent('BeforeMove', pokemon, target)` speed-sorts every holder's
    // BeforeMove handlers before any callback runs (Heal Block + Throat Chop on
    // the user, two foes' Imprison, ...), whether or not the move is blocked.
    // PS sim/battle-actions.ts runMove. TODO(PS parity): the `|cant|` message
    // names the handler that ran first (order not modelled).
    crate::doubles::events::run_event_ties(state, rng, crate::doubles::events::Event::BeforeMove, (side, slot), resolved);
    state.tl.before_move_aborted.set(true);
    // A pokemon that must recharge (Hyper Beam / Solar Beam) cannot act.
    // No PRNG draw (PS `mustrecharge.onBeforeMove` returns false).
    if state.side(side).active()[slot].recharging {
        state.side_mut(side).active_slot_mut(slot).recharging = false;
        // PS logs `|cant|`, not `|move|`: not in the move order.
        return;
    }

    // BeforeMove handlers in PS priority order (champions conditions):
    // sleep / freeze (10), flinch (8), confusion (3), paralysis (1).
    // `sleep_turns` holds PS `statusState.time` for sleep and freeze.
    {
        let move_id = action.choice.move_id;
        let s = state.side_mut(side);
        let p = s.team_member_mut(idx);
        match p.status {
            // slp: time--; wake and act at 0. No draw. Early Bird counts
            // down twice (PS data/conditions.ts slp.onBeforeMove).
            PokemonStatus::SLEEP => {
                if p.ability == Abilities::EARLYBIRD {
                    p.sleep_turns -= 1;
                }
                p.sleep_turns -= 1;
                if p.sleep_turns <= 0 {
                    p.status = PokemonStatus::NONE;
                } else if move_id != Choices::SNORE && move_id != Choices::SLEEPTALK {
                    // Snore / Sleep Talk are `sleepUsable`.
                    // PS logs `|cant|`, not `|move|`: not in the move order.
                    return;
                }
            }
            // frz: a `defrost` move acts (and thaws); else time--, thaw at 0
            // or with THAW_CHANCE.
            PokemonStatus::FREEZE => {
                // PS champions frz.onBeforeMove: a `defrost` move acts without the
                // roll, except Burn Up from a user that lost its Fire type.
                let burnup_no_fire = move_id == Choices::BURNUP
                    && p.types.0 != PokemonType::FIRE
                    && p.types.1 != PokemonType::FIRE;
                if crate::engine::mechanics::THAWS_USER.contains(&move_id) && !burnup_no_fire {
                    // Acts while frozen; the thaw is `frz.onModifyMove` (below,
                    // after the other BeforeMove checks such as flinch).
                } else {
                    p.sleep_turns -= 1;
                    let thaw = p.sleep_turns <= 0
                        || rng
                            .label("status before move")
                            .random_chance(1, (1.0 / THAW_CHANCE).round() as u32);
                    let p = state.side_mut(side).team_member_mut(idx);
                    if thaw {
                        p.status = PokemonStatus::NONE;
                        p.sleep_turns = 0;
                    } else {
                        return;
                    }
                }
            }
            _ => {}
        }
    }

    // Flinched by a previous move's secondary: cannot act. No draw.
    // PS data/abilities.ts steadfast `onFlinch`: +1 Speed to the holder.
    if state.side(side).active()[slot].flinched {
        if let Some(idx) = state.side(side).active()[slot].pokemon_index {
            if state.side(side).team_member(idx).ability == Abilities::STEADFAST {
                apply_boosts(state, side, idx, &StatBoosts { speed: 1, ..Default::default() }, false);
            }
        }
        return;
    }

    // Disable `onBeforeMove` (priority 7): the disabled move cannot be used.
    if !external && state.side(side).active()[slot].disabled.map(|(m, _)| m) == Some(action.choice.move_id) {
        return;
    }

    // Throat Chop `onBeforeMove` (priority 6): sound moves are blocked.
    if state.side(side).active()[slot].throat_chop > 0 && action.choice.flags.sound {
        return;
    }

    // Heal Block `onBeforeMove` (priority 6, data/moves.ts healblock): moves
    // with the `heal` flag are blocked ("cant ... move: Heal Block"). No draw.
    if state.side(side).active()[slot].heal_block > 0 && action.choice.flags.heal {
        return;
    }

    // Gravity `onBeforeMove` (priority 6, data/moves.ts gravity.condition): moves
    // with the `gravity` flag are blocked ("cant ... move: Gravity"), before the
    // paralysis draw (priority 1). No draw.
    if state.gravity.active && has_gravity_flag(action.choice.move_id) {
        return;
    }

    // Taunt `onBeforeMove` (priority 5): Status moves are blocked.
    if state.side(side).active()[slot].taunt > 0 && action.choice.category == MoveCategory::Status {
        return;
    }

    // Imprison `onFoeBeforeMove` (priority 4, PS data/moves.ts imprison): a foe's
    // Imprison blocks the moves its user knows ("cant ... move: Imprison"). No draw.
    if action.choice.move_id != Choices::STRUGGLE && imprison_blocks(state, side, action.choice.move_id) {
        return;
    }

    // Confusion (data/conditions.ts `confusion.onBeforeMove`): time--, ends
    // at 0; otherwise a 33/100 chance to hit itself (40 BP typeless, own Atk
    // vs own Def, damage roll) instead of moving. The singles engine's
    // HIT_SELF_IN_CONFUSION_CHANCE is 1/3; PS rolls 33/100.
    if state.side(side).active()[slot].confusion > 0 {
        let a = state.side_mut(side).active_slot_mut(slot);
        a.confusion -= 1;
        if a.confusion > 0 && rng.label("status before move").random_chance(33, 100) {
            let dmg = confusion_damage(state, rng, side, slot, idx);
            // Disguise `onDamage` (PS data/abilities.ts disguise): the self-hit
            // is a `Move` effect, so it is absorbed and the forme busts (1/8).
            let disguised = state.side(side).team_member(idx).id == crate::pokemon::PokemonName::MIMIKYU
                && state.side(side).team_member(idx).ability == Abilities::DISGUISE;
            if disguised {
                // `onDamage` only sets `busted`; the forme change (and its 1/8
                // damage and raw cached speed) is Disguise's `onUpdate`, after
                // the next `eachEvent('Update')` sort (runAction's).
                if !state.pending_disguise.contains(&(side, slot)) {
                    state.pending_disguise.push((side, slot));
                }
            } else {
                apply_damage(state, side, slot, dmg);
            }
            return;
        }
    }

    // Attract (`attract.onBeforeMove`, priority 2): 1/2 chance to be
    // immobilized. It is dropped first when the source is no longer active
    // (`onUpdate`).
    if let Some((ss, si)) = state.side(side).active()[slot].attract {
        let src_active = (0..2).any(|sl| {
            state.side(ss).active()[sl].pokemon_index == Some(si) && state.side(ss).team_member(si).hp > 0
        });
        if !src_active {
            state.side_mut(side).active_slot_mut(slot).attract = None;
        } else if rng.label("status before move").random_chance(1, 2) {
            return;
        }
    }

    // par: full paralysis with FULLY_PARALYZED_CHANCE (1/8 in champions).
    if state.side(side).team_member(idx).status == PokemonStatus::PARALYZE
        && rng
            .label("status before move")
            .random_chance(1, (1.0 / FULLY_PARALYZED_CHANCE).round() as u32)
    {
        return;
    }

    state.tl.before_move_aborted.set(false);
    // Focus Punch `beforeMoveCallback` (after the BeforeMove checks): it
    // fails with `|cant|` if the user was hit by a damaging move this turn.
    if action.choice.move_id == Choices::FOCUSPUNCH && state.side(side).active()[slot].lost_focus {
        return;
    }
    }

    // PS `runMove` -> `deductPP` (after BeforeMove and beforeMoveCallback, and
    // skipped for a locked move). `ppData.used` is set even when the PP is 0,
    // then the move fails with `|cant|...|nopp|` (Spite, PS battle-actions.ts).
    // Struggle never checks PP.
    if !external && !action.no_pp {
        let locked = {
            let a = &state.side(side).active()[slot];
            // PS `getLockedMove`: `lockedmove` (Outrage, ...), uproar, and the
            // `twoturnmove` volatile (its `onLockMove`: the firing turn).
            a.locked_move.is_some() || a.uproar > 0 || (a.charging.is_some() && !a.charge_aborted)
        };
        if !locked {
            let s = state.side_mut(side);
            if let Some(p) = s.active()[slot].pokemon_index {
                s.moves_used[p as usize][action.choice.move_index as usize] = true;
                let mv = &mut s.team_member_mut(p).moves[&action.choice.move_index];
                // Struggle has no PP (PS `struggle` is not deducted): its
                // `move_index` is the default slot 0, which must not lose PP.
                if action.choice.move_id != Choices::STRUGGLE {
                    if mv.pp > 0 {
                        mv.pp -= 1;
                    } else {
                        return;
                    }
                }
            }
        }
    }

    // PS `useMove` -> `pokemon.moveUsed` (after `runMove`'s BeforeMove checks,
    // before `onTry`): `lastMove` is set even when the move then fails (Encore's
    // `failencore` check reads it, so a failed Copycat blocks Encore).
    if !external {
        state.side_mut(side).slot_scratch_mut(slot).last_move = Some(action.choice.move_id);
        if let Some(loc) = used_target_loc {
            state.side_mut(side).slot_scratch_mut(slot).last_target_loc = loc;
        }
    }

    // PS data/conditions.ts frz.onModifyMove: a `defrost` move thaws the user
    // once it is actually used (a flinch before it keeps the user frozen).
    if crate::engine::mechanics::THAWS_USER.contains(&action.choice.move_id) {
        let p = state.side_mut(side).team_member_mut(idx);
        if p.status == PokemonStatus::FREEZE {
            p.status = PokemonStatus::NONE;
            p.sleep_turns = 0;
        }
    }

    // PS `onModifyType` / `onModifyMove` for moves whose type and power
    // depend on the field (shared with the singles engine).
    let mut modified = action.choice.clone();
    match modified.move_id {
        Choices::TERRAINPULSE => {
            let grounded = slot_grounded(state.side(side).team_member(idx), &state.side(side).active()[slot], state.gravity.active);
            if let Some(t) = crate::engine::mechanics::terrain_pulse_type(state.terrain.terrain_type, grounded) {
                modified.move_type = t;
                modified.base_power *= 2.0;
            }
        }
        Choices::WEATHERBALL => {
            // PS data/moves.ts weatherball `onModifyType` / `onModifyMove` read
            // `pokemon.effectiveWeather()`, which is empty while Cloud Nine /
            // Air Lock suppress weather (`field.suppressingWeather()`).
            let mega_sol = state.side(side).team_member(idx).ability == Abilities::MEGASOL;
            let eff_weather = effective_weather(state);
            if let Some(t) = crate::engine::mechanics::weather_ball_type(eff_weather, mega_sol) {
                modified.move_type = t;
                modified.base_power = 100.0;
            }
        }
        _ => {}
    }
    // -ate abilities (shared rule).
    let ate_boosted = match crate::engine::mechanics::ate_type_change(
        state.side(side).team_member(idx).ability,
        &modified,
    ) {
        Some(t) => {
            modified.move_type = t;
            true
        }
        None => false,
    };
    // Stance Change `onModifyMove`: attacking moves switch Aegislash to Blade
    // forme, King's Shield back to Shield forme (stats recomputed; same HP).
    {
        let p = state.side(side).team_member(idx);
        if p.ability == Abilities::STANCECHANGE
            && matches!(p.id, crate::pokemon::PokemonName::AEGISLASH | crate::pokemon::PokemonName::AEGISLASHBLADE)
            && (choice_is_attack(&action.choice) || action.choice.move_id == Choices::KINGSSHIELD)
        {
            let to_blade = action.choice.move_id != Choices::KINGSSHIELD;
            let target_id = if to_blade {
                crate::pokemon::PokemonName::AEGISLASHBLADE
            } else {
                crate::pokemon::PokemonName::AEGISLASH
            };
            if p.id != target_id {
                let key = if to_blade { "aegislashblade" } else { "aegislash" };
                if let Some(data) = crate::doubles::species_types::species_data(key) {
                    let (e, lvl, nat) = (p.evs, p.level as u8, p.nature.clone());
                    let evs = crate::doubles::state::Evs {
                        hp: e.0 as u16,
                        atk: e.1 as u16,
                        def: e.2 as u16,
                        spa: e.3 as u16,
                        spd: e.4 as u16,
                        spe: e.5 as u16,
                    };
                    let (_, atk, def, spa, spd, _) =
                        crate::doubles::state::compute_stats(data.base_stats, &evs, lvl, nat);
                    restore_stored_stats(state.side_mut(side), idx);
                    let pm = state.side_mut(side).team_member_mut(idx);
                    pm.id = target_id;
                    pm.attack = atk;
                    pm.defense = def;
                    pm.special_attack = spa;
                    pm.special_defense = spd;
                    // `setSpecies` -> `setType(species.types)` and `addedType = ''`:
                    // a Soak / Forest's Curse type change is gone.
                    pm.types = (PokemonType::from(data.types.0), PokemonType::from(data.types.1));
                    // `formeChange` -> `setSpecies`: the cached speed is the raw stat.
                    let raw = pm.speed;
                    state.side_mut(side).speed_cache[idx as usize] = raw as _;
                    let a = state.side_mut(side).active_slot_mut(slot);
                    a.type_change = None;
                    a.added_type = PokemonType::TYPELESS;
                    a.added_grass = false;
                }
            }
        }
    }

    // Liquid Voice (`onModifyType`, priority -1): sound moves become Water,
    // with no power boost.
    if state.side(side).team_member(idx).ability == Abilities::LIQUIDVOICE && modified.flags.sound {
        modified.move_type = PokemonType::WATER;
    }
    // Thunder / Hurricane `onModifyMove`: accuracy 50 in harsh sunlight
    // (`target.effectiveWeather()`: Cloud Nine / Air Lock suppress it).
    if matches!(modified.move_id, Choices::THUNDER | Choices::HURRICANE)
        && matches!(effective_weather(state), Weather::SUN | Weather::HARSHSUN)
    {
        modified.accuracy = 50.0;
    }
    // Raging Bull `onModifyType`: the Paldean Tauros breed's type.
    if modified.move_id == Choices::RAGINGBULL {
        let sp = state.side(side).species_names[idx as usize].to_lowercase();
        if sp.contains("aqua") {
            modified.move_type = PokemonType::WATER;
        } else if sp.contains("blaze") {
            modified.move_type = PokemonType::FIRE;
        } else if sp.contains("combat") {
            modified.move_type = PokemonType::FIGHTING;
        }
    }
    // Aura Wheel `onModifyType` (PS data/moves.ts aurawheel): Dark while the user
    // is Morpeko-Hangry, else Electric.
    if modified.move_id == Choices::AURAWHEEL {
        modified.move_type = if state.side(side).team_member(idx).id == crate::pokemon::PokemonName::MORPEKOHANGRY {
            PokemonType::DARK
        } else {
            PokemonType::ELECTRIC
        };
    }
    // Electrify `onModifyType` (priority -2): any move but Struggle becomes Electric.
    if state.side(side).active()[slot].electrify && modified.move_id != Choices::STRUGGLE {
        modified.move_type = PokemonType::ELECTRIC;
    }
    let choice = &modified;
    let mut target_side = side.other();

    // Resolve the target slot. Self / ally / field moves need no FOE target.
    let is_self = choice.target == MoveTarget::User;
    let is_ally = choice.target == MoveTarget::Ally;
    // Field moves: status moves that only set weather / terrain / a side
    // condition. (Ceaseless Edge / Stone Axe also carry a side condition but
    // are attacks.)
    // The upstream table has no `weather_id` for Rain Dance / Sandstorm.
    let weather_id = match choice.move_id {
        Choices::RAINDANCE => 1,
        Choices::SANDSTORM => 3,
        Choices::SNOWSCAPE | Choices::CHILLYRECEPTION => 5,
        _ => choice.weather_id,
    };
    let is_field = weather_id != 0
        || choice.terrain_id != 0
        || choice.pseudo_weather_id != 0
        || (choice.side_condition.is_some() && choice.category == MoveCategory::Status);

    // The slot an `Ally` effect hits: the one adjacent ally.
    let ally_slot = 1 - slot;

    let mut target_slot = 0usize;
    let mut has_target = true;
    if !is_self && !is_ally && !is_field {
        // The target `ps_get_target` resolved (with PS's retarget draw).
        // TODO(PS parity): spread moves (allAdjacentFoes / allAdjacent) hit
        // every foe; the engine hits one.
        // Metal Burst / Comeuppance `onModifyTarget`: the last foe that damaged
        // the user (`getAtSlot(lastDamagedBy.slot)`).
        // A fainted last attacker is not a target: the move retargets (PS
        // `getRandomTarget` draws in `getMoveTargets`, then `scripted` misses).
        let scripted = if matches!(choice.move_id, Choices::METALBURST | Choices::COMEUPPANCE) {
            // `getMoveTargets` then runs `RedirectTarget` (Follow Me / Rage
            // Powder, Lightning Rod) on it (PS sim/pokemon.ts).
            last_damaged_by(state, side, idx)
                .map(|(s_, sl_, _, _, _)| (s_, sl_))
                .filter(|&(s_, sl_)| switched_in_alive(state, s_, sl_))
                .map(|t| redirect_target(state, action, t))
        } else {
            None
        };
        match scripted.or(resolved) {
            Some((ts, sl)) if switched_in_alive(state, ts, sl) => {
                target_side = ts;
                target_slot = sl;
            }
            _ => {
                // PS `getTarget`: a fainted ally chosen by an adjacentAlly move is
                // the target itself (no retarget). The hit loop then breaks with no
                // `eachEvent`, and the move fails.
                match resolved {
                    Some((ts, sl)) if ts == side => {
                        target_side = ts;
                        target_slot = sl;
                    }
                    _ => target_slot = action.target.min(1),
                }
                // An allAdjacent move (Earthquake) with no living foe still hits
                // the living ally: PS `getMoveTargets` lists the adjacent allies.
                // PS data/moves.ts earthquake (target allAdjacent).
                let all_adjacent_ally = move_target(choice.move_id) == Tgt::AllAdjacent
                    && switched_in_alive(state, side, 1 - slot);
                // A `futuremove` (Future Sight) keeps its fainted target: it
                // targets the slot (PS sim/pokemon.ts getMoveTargets).
                if crate::engine::mechanics::is_future_move(choice.move_id) {
                    if let Some((ts, sl)) = resolved {
                        target_side = ts;
                        target_slot = sl;
                    }
                } else if !switched_in_alive(state, target_side, target_slot) && !all_adjacent_ally {
                    has_target = false;
                }
            }
        }
        // A charge move still starts its charge with no living target: PS
        // `onTryMove` (`-prepare`, `addVolatile('twoturnmove')`) runs before the
        // target checks, and `getRandomTarget` keeps the fainted foe
        // (`randomFoe() || foe.active[0]`). PS data/moves.ts fly.onTryMove.
        // The firing turn of a charge move (Fly / Dig after their charge) with
        // no living target: PS `getMoveTargets` returns no targets, so
        // `useMoveInner` prints `-fail` before any accuracy roll (battle-actions.ts
        // `!targets.length`). The charge ends as on any firing turn.
        let firing_charge = matches!(state.side(side).active()[slot].charging, Some((m, 0)) if m == choice.move_id);
        // A charge move whose charge the weather skips (Solar Beam in sun, ...)
        // fires at once: `onTryMove` returns without charging, then the
        // `!targets.length` fail (PS data/moves.ts solarbeam.onTryMove).
        let charge_skipped = {
            let w = effective_weather(state);
            match choice.charge_setup {
                ChargeSetup::NotSun { .. } => w == Weather::SUN,
                ChargeSetup::OnlyRain { .. } => matches!(w, Weather::RAIN | Weather::HEAVYRAIN),
                _ => false,
            }
        };
        if !has_target && (matches!(choice.charge_setup, ChargeSetup::None) || firing_charge || charge_skipped) {
            if firing_charge {
                state.side_mut(side).active_slot_mut(slot).charging = None;
                state.side_mut(side).slot_scratch_mut(slot).twoturn_fired = true;
            }
            // PS useMoveInner `!targets.length`: `-fail` and `return false`, so
            // `moveThisTurnResult` is false (Stomping Tantrum's doubled BP reads it).
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            state.tl.no_target_fail.set(true);
            modify_move_field_tie(state, rng);
        modify_move_user_tie(state, rng, side, slot);
            // `selfdestruct: 'always'` faints the user before the target check
            // (PS sim/battle-actions.ts useMoveInner), even with no target left.
            if matches!(choice.move_id, Choices::EXPLOSION | Choices::MISTYEXPLOSION | Choices::SELFDESTRUCT) {
                selfdestruct_user_faint(state, rng, side, slot);
            }
            push_order(move_order);
            return;
        }
    }

    // An adjacentAlly move whose chosen location is now the user (Ally Switch
    // swapped the user into it) or has no adjacent ally: PS `getTarget` returns
    // null (`targetLoc === selfLoc`, `getRandomTarget` with no ally), and
    // `useMoveInner` prints `[notarget]` and fails. No draws.
    // PS sim/battle.ts getTarget; sim/battle-actions.ts useMoveInner (`!target`).
    // A fainted ally (already fainted this turn) is no target either:
    // `getMoveTargets` returns none for `target.fainted` (sim/pokemon.ts).
    // `useMoveInner` returns false: `moveThisTurnResult` is false.
    let ally_gone = matches!(resolved, Some((rs, rl)) if rs == side && rl != slot && !switched_in_alive(state, rs, rl));
    if is_ally && !is_self && !is_field && (resolved.is_none() || ally_gone) {
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
        state.tl.no_target_fail.set(true);
        modify_move_field_tie(state, rng);
        modify_move_user_tie(state, rng, side, slot);
        push_order(move_order);
        return;
    }
    // An adjacentAlly move (Coaching, Helping Hand, ...) whose ally fainted
    // earlier this turn: `getTarget` returns the fainted ally (no retarget),
    // `getMoveTargets` lists no target, and `useMoveInner` prints `-fail` and
    // returns false, so `moveThisTurnResult` is false (Stomping Tantrum).
    // PS sim/battle.ts getTarget; sim/pokemon.ts getMoveTargets; sim/battle-actions.ts useMoveInner.
    if is_ally && !is_self && !is_field {
        if let Some((ts, sl)) = resolved {
            if ts == side && !switched_in_alive(state, ts, sl) {
                state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
                state.tl.no_target_fail.set(true);
                modify_move_field_tie(state, rng);
                modify_move_user_tie(state, rng, side, slot);
                push_order(move_order);
                return;
            }
        }
    }

    // Expanding Force `onModifyMove` (PS data/moves.ts expandingforce): in Psychic
    // Terrain a grounded user hits all adjacent foes. PS useMoveInner then calls
    // `getRandomTarget` once per `baseTarget !== move.target` check: two foe draws.
    if expanding_force_spreads(state, side, slot, choice) {
        for _ in 0..2 {
            let foes: Vec<usize> = (0..2).filter(|&sl| state.side(side.other()).slot_pokemon(sl).is_some_and(|p| p.hp > 0)).collect();
            if !foes.is_empty() {
                let _ = rng.label("retarget").random(foes.len() as u32);
            }
        }
    }

    // Long Reach `onModifyMove` (PS data/abilities.ts longreach): the user's
    // moves lose the `contact` flag, so Fluffy, Tough Claws, Protect's contact
    // penalty and the contact abilities of the target all see a non-contact move.
    let long_reach_choice;
    let choice = if choice.flags.contact && state.side(side).team_member(idx).ability == Abilities::LONGREACH {
        long_reach_choice = {
            let mut c = choice.clone();
            c.flags.contact = false;
            c
        };
        &long_reach_choice
    } else {
        choice
    };

    // Shell Side Arm `onModifyMove` (PS data/moves.ts shellsidearm): the user's
    // Atk vs SpA and the target's Def vs SpD, all with boosts but no ability or
    // item modifiers (`getStat(stat, false, true)`). Physical wins when strictly
    // higher; a tie draws `randomChance(1, 2)`. Runs after the target is chosen
    // and before any other move step, as in PS `useMoveInner`.
    let shell_side_arm_choice;
    let choice = if choice.move_id == Choices::SHELLSIDEARM
        && !is_self
        && !is_ally
        && !is_field
        && state.side(target_side).slot_pokemon(target_slot).is_some()
    {
        let a = state.side(side).team_member(idx);
        let ab = state.side(side).team_boosts[idx as usize].clone();
        let d = state.side(target_side).slot_pokemon(target_slot).unwrap();
        let db = state.side(target_side).slot_boosts(target_slot).cloned().unwrap_or_default();
        let wonder = state.wonder_room.active;
        let atk = crate::engine::mechanics::boost_stat(ab.atk, a.attack).max(1) as i64;
        let spa = crate::engine::mechanics::boost_stat(ab.spa, a.special_attack).max(1) as i64;
        // PS sim/pokemon.ts getStat(unmodified) under Wonder Room: the stored
        // stat stays (`storedStats.def`), only the boost is read from the other
        // defensive stat ("Download ignores Wonder Room's effect").
        let def = if wonder {
            crate::engine::mechanics::boost_stat(db.spd, d.defense)
        } else {
            crate::engine::mechanics::boost_stat(db.def, d.defense)
        }
        .max(1) as i64;
        let spd = if wonder {
            crate::engine::mechanics::boost_stat(db.def, d.special_defense)
        } else {
            crate::engine::mechanics::boost_stat(db.spd, d.special_defense)
        }
        .max(1) as i64;
        let s1 = (2 * a.level as i64) / 5 + 2;
        let physical = (s1 * 90 * atk) / def / 50;
        let special = (s1 * 90 * spa) / spd / 50;
        if physical > special || (physical == special && rng.label("other: onModifyMove").random_chance(1, 2)) {
            shell_side_arm_choice = {
                let mut c = choice.clone();
                c.category = MoveCategory::Physical;
                c.flags.contact = true;
                c
            };
            &shell_side_arm_choice
        } else {
            choice
        }
    } else {
        choice
    };

    // `runEvent('ModifyMove')` in `useMoveInner` (after the move's own
    // `singleEvent` handlers, before TryMove): Gravity and Wonder Room tie.
    modify_move_field_tie(state, rng);
    modify_move_user_tie(state, rng, side, slot);

    // Two-turn charge moves (Dig, Meteor Beam, ...): mirrors PS
    // `onTryMove` + `addVolatile('twoturnmove')` (data/moves.ts). The first
    // use(s) only charge: no accuracy roll, no damage, no secondary, but the
    // charge-turn self boost applies (Meteor Beam +1 SpA, Skull Bash +1 Def).
    // The move fires when the same move is chosen again after the last
    // charge turn (PS locks the move via `twoturnmove.onLockMove`).
    {
        // PS `attacker.effectiveWeather()` (Cloud Nine / Air Lock suppress it).
        let eff_weather = effective_weather(state);
        let needs_charge = match choice.charge_setup {
            ChargeSetup::None => false,
            ChargeSetup::Always { .. } => true,
            ChargeSetup::NotSun { .. } => eff_weather != Weather::SUN,
            ChargeSetup::OnlyRain { .. } => !matches!(eff_weather, Weather::RAIN | Weather::HEAVYRAIN),
            ChargeSetup::OnlySnow { .. } => matches!(eff_weather, Weather::HAIL | Weather::SNOW),
        };
        // PS data/moves.ts electroshot `onTryMove`: `boost({spa: 1})` runs on the
        // charge turn and on the rain-skip turn, before the weather check (the
        // firing turn returns early via `removeVolatile`). The table has no
        // `charge_boosts` for it, so it is special-cased here.
        let charge_boost: Option<StatBoosts> = if choice.move_id == Choices::ELECTROSHOT {
            Some(StatBoosts { special_attack: 1, ..Default::default() })
        } else {
            choice.charge_boosts.clone()
        };
        let charged = match state.side(side).active()[slot].charging {
            Some(_) => state.side_mut(side).active_slot_mut(slot).charging.take(),
            None => None,
        };
        match charged {
            Some((m, turns_left)) if m == choice.move_id && turns_left > 0 => {
                // Still charging (Geomancy's second charge turn).
                state.side_mut(side).active_slot_mut(slot).charging = Some((m, turns_left - 1));
                push_order(move_order);
                return;
            }
            Some((_, _)) => {
                // turns_left == 0 (fire this turn) or a different move was
                // chosen while charging (charge wasted; PS charge volatiles
                // expire at end of turn).
                state.side_mut(side).slot_scratch_mut(slot).twoturn_fired = true;
            }
            None if needs_charge => {
                // Start charging.
                if let Some(b) = &charge_boost {
                    apply_boosts(state, side, idx, b, false);
                }
                let turns = match choice.charge_setup {
                    ChargeSetup::Always { turns }
                    | ChargeSetup::NotSun { turns }
                    | ChargeSetup::OnlyRain { turns }
                    | ChargeSetup::OnlySnow { turns } => turns,
                    ChargeSetup::None => 0,
                };
                // `attacker.addVolatile('twoturnmove')`, whose `onStart` runs
                // `attacker.addVolatile(move.id)`: two TryAddVolatile sorts.
                // PS data/moves.ts (dig, solarbeam, ...) onTryMove,
                // data/conditions.ts twoturnmove.onStart.
                try_add_volatile_ties(state, rng, side, slot);
                try_add_volatile_ties(state, rng, side, slot);
                state.side_mut(side).active_slot_mut(slot).charging =
                    Some((choice.move_id, turns.saturating_sub(1)));
                // `twoturnmove.onStart` runs `runEvent('PrepareHit')` on the charge
                // turn (Protean / Libero). PS data/conditions.ts twoturnmove.
                prepare_hit(state, side, slot, idx, choice);
                // A Copycat-called charge move has no chosen target: keep the
                // randomly resolved one (PS `lockedMoveTargetLoc`).
                state.side_mut(side).slot_scratch_mut(slot).charge_target = match resolved {
                    Some((ts, tsl)) if ts != side && action.target_loc == 0 => tsl,
                    _ => action.target,
                };
                push_order(move_order);
                return;
            }
            None => {
                // Weather skips the charge (Electro Shot in rain): the
                // charge-turn boost still applies.
                if let Some(b) = &charge_boost {
                    apply_boosts(state, side, idx, b, false);
                }
            }
        }
    }

    // Round `onTry` (PS data/moves.ts round): never fails; the battle loop
    // queues the next Round with BP doubled after this action.
    if choice.move_id == Choices::ROUND {
        state.queue_round = true;
    }

    // PS `runEvent('TryMove', pokemon, target)` (useMoveInner): after the move's
    // own `onTryMove` singleEvent (Double Shock / Burn Up fail there, and the
    // runEvent is skipped), before every `onTry` / `onTryHit` below.
    let try_move_fails = matches!(choice.move_id, Choices::DOUBLESHOCK | Choices::BURNUP) && {
        let t = state.side(side).team_member(idx).types;
        let need = if choice.move_id == Choices::DOUBLESHOCK { PokemonType::ELECTRIC } else { PokemonType::FIRE };
        t.0 != need && t.1 != need
    };
    if !try_move_fails {
        crate::doubles::events::run_event_ties(state, rng, crate::doubles::events::Event::TryMove, (side, slot), Some((target_side, target_slot)));
    }

    // Stockpile family (PS `onTry` / `stockpile` condition). No PRNG draws.
    match choice.move_id {
        Choices::STOCKPILE => {
            let a = &state.side(side).active()[slot];
            if a.stockpile < 3 {
                // `onTry` passed: `trySpreadMoveHit` runs PrepareHit (Protean /
                // Libero) before the hit. PS sim/battle-actions.ts trySpreadMoveHit.
                prepare_hit(state, side, slot, idx, choice);
                let before = state.side(side).team_boosts[idx as usize].clone();
                apply_boosts(
                    state,
                    side,
                    idx,
                    &StatBoosts {
                        defense: 1,
                        special_defense: 1,
                        ..Default::default()
                    },
                    false,
                );
                let after = state.side(side).team_boosts[idx as usize].clone();
                let a = state.side_mut(side).active_slot_mut(slot);
                a.stockpile += 1;
                a.stockpile_def += after.def - before.def;
                a.stockpile_spd += after.spd - before.spd;
                // PS `hitStepMoveHitLoop`: `eachEvent('Update')` after the hit,
                // and once more after the loop.
                each_event_update(state, rng);
                each_event_update(state, rng);
            } else {
                // `onTry` returns false at 3 layers: `-fail`, moveThisTurnResult false.
                state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            }
            push_order(move_order);
            return;
        }
        // Snore / Sleep Talk `onTry`: fails unless the user is asleep (or has Comatose).
        Choices::SNORE | Choices::SLEEPTALK
            if state.side(side).team_member(idx).status != PokemonStatus::SLEEP
                && state.side(side).team_member(idx).ability != Abilities::COMATOSE =>
        {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
        // Double Shock / Burn Up `onTryMove`: fail unless the user still has
        // the move's type (no draw). PS data/moves.ts doubleshock / burnup.
        // `return null` (not false), so `moveThisTurnResult` is null and
        // Stomping Tantrum's `=== false` doubling stays off: the result is left
        // unset here (the end-of-action step records it as not false).
        Choices::DOUBLESHOCK | Choices::BURNUP
            if {
                let t = state.side(side).team_member(idx).types;
                let need = if choice.move_id == Choices::DOUBLESHOCK { PokemonType::ELECTRIC } else { PokemonType::FIRE };
                t.0 != need && t.1 != need
            } =>
        {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = None;
            push_order(move_order);
            return;
        }
        // Healing Wish / Lunar Dance `onTryHit`: fail when nobody can switch in
        // (`!canSwitch(source.side)`). PS data/moves.ts healingwish.
        Choices::HEALINGWISH | Choices::LUNARDANCE if possible_switches(state, side).is_empty() => {
            push_order(move_order);
            return;
        }
        // Sucker Punch / Thunderclap `onTry`: fails unless the target still has
        // a queued attacking move (`queue.willMove(target)`) and is not
        // recharging. PS data/moves.ts suckerpunch / thunderclap. `willMove`
        // matches the pokemon, not its slot (an Ally Switch may have moved it).
        Choices::SUCKERPUNCH | Choices::THUNDERCLAP
            if !is_self
                && !is_field
                && (state.side(target_side).active()[target_slot].recharging
                    || {
                        let tidx = state.side(target_side).active()[target_slot].pokemon_index;
                        !queue.iter().any(|a| {
                            a.side == target_side
                                && Some(a.pokemon_index) == tidx
                                && a.choice.category != MoveCategory::Status
                        })
                    }) =>
        {
            // PS `onTry` failure (`-fail`): `moveThisTurnResult = false`, which
            // Stomping Tantrum / Temper Flare read next turn (`moveLastTurnResult`).
            state.side_mut(action.side).slot_scratch_mut(action.slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
        // Last Resort `onTry`: every other move slot must have been used.
        // PS data/moves.ts lastresort.
        Choices::LASTRESORT
            if {
                let sd = state.side(side);
                let p = sd.team_member(idx);
                let mut count = 0;
                let mut has = false;
                let mut ok = true;
                for (mi, m) in [&p.moves.m0, &p.moves.m1, &p.moves.m2, &p.moves.m3].into_iter().enumerate() {
                    if m.id == Choices::NONE {
                        continue;
                    }
                    count += 1;
                    if m.id == Choices::LASTRESORT {
                        has = true;
                    } else if !sd.moves_used[idx as usize][mi] {
                        ok = false;
                    }
                }
                !(count >= 2 && has && ok)
            } =>
        {
            // PS `onTry` failure (`-fail`): `moveThisTurnResult = false`.
            state.side_mut(action.side).slot_scratch_mut(action.slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
        Choices::SPITUP | Choices::SWALLOW if state.side(side).active()[slot].stockpile == 0 => {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
        Choices::SWALLOW => {
            prepare_hit(state, side, slot, idx, choice);
            let layers = state.side(side).active()[slot].stockpile;
            let s = state.side_mut(side);
            let p = s.team_member_mut(idx);
            // PS data/moves.ts swallow.onHit: `heal(modify(maxhp, [0.25, 0.5, 1][layers - 1]))`.
            let fraction = match layers {
                1 => 0.25,
                2 => 0.5,
                _ => 1.0,
            };
            let heal = crate::doubles::damage::modify(p.maxhp as i64, fraction, 1.0) as i16;
            p.hp = (p.hp + heal).min(p.maxhp);
            end_stockpile(state, side, slot);
            each_event_update(state, rng);
            each_event_update(state, rng);
            push_order(move_order);
            return;
        }
        _ => {}
    }

    // Damp `onAnyTryMove` (data/abilities.ts damp): an active Damp holder fails
    // explosion / mindblown / mistyexplosion / selfdestruct (`cant`). No draw.
    if damp_blocks_explosion(state, side, idx, choice) {
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
        push_order(move_order);
        return;
    }

    // Steel Roller (gen 9): fails when no terrain is active (PS `onTry`).
    // No PRNG draw; PP is already spent.
    if choice.requires_terrain
        && state.terrain.terrain_type == crate::engine::state::Terrain::NONE
    {
        // `-fail`: `moveThisTurnResult = false` (Stomping Tantrum doubles next turn).
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
        push_order(move_order);
        return;
    }

    // Ally Switch `onPrepareHit`: `addVolatile('allyswitch')`; a restart rolls
    // `randomChance(1, counter)` and fails (dropping the volatile) on a miss.
    if choice.move_id == Choices::ALLYSWITCH {
        let a = state.side_mut(side).active_slot_mut(slot);
        match a.ally_switch {
            None => a.ally_switch = Some((3, 2)),
            Some((c, _)) => {
                if !rng.label("volatile duration").random_chance(1, c.max(1) as u32) {
                    state.side_mut(side).active_slot_mut(slot).ally_switch = None;
                    // A failed PrepareHit is a `-fail`: `moveThisTurnResult = false`.
                    state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
                    push_order(move_order);
                    return;
                }
                state.side_mut(side).active_slot_mut(slot).ally_switch = Some((if c < 729 { c * 3 } else { c }, 2));
            }
        }
    }

    // Quick Guard `onTry`: `!!this.queue.willAct()` (PS data/moves.ts quickguard).
    if matches!(choice.move_id, Choices::QUICKGUARD | Choices::WIDEGUARD) && !will_act {
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
        push_order(move_order);
        return;
    }

    // Stalling moves (Protect / Detect / Endure) `onPrepareHit`:
    // `!!this.queue.willAct() && this.runEvent('StallMove', pokemon)`; the
    // `stall` volatile rolls `randomChance(1, counter)` and is dropped on a
    // failure (data/conditions.ts stall.onStallMove).
    if matches!(
        choice.volatile_status.as_ref().map(|v| v.volatile_status),
        Some(
            PokemonVolatileStatus::PROTECT
                | PokemonVolatileStatus::ENDURE
                | PokemonVolatileStatus::KINGSSHIELD
                | PokemonVolatileStatus::SPIKYSHIELD
                | PokemonVolatileStatus::BANEFULBUNKER
                | PokemonVolatileStatus::SILKTRAP
                | PokemonVolatileStatus::BURNINGBULWARK
        )
    ) {
        let mut ok = will_act;
        if ok {
            if let Some((counter, _)) = state.side(side).active()[slot].stall {
                ok = rng.label("stall").random_chance(1, counter.max(1) as u32);
                if !ok {
                    state.side_mut(side).active_slot_mut(slot).stall = None;
                }
            }
        }
        if !ok {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
    }

    // Counter / Mirror Coat `onTry`: fail unless the volatile recorded a hit.
    if matches!(choice.move_id, Choices::COUNTER | Choices::MIRRORCOAT) {
        let a = &state.side(side).active()[slot];
        let rec = if choice.move_id == Choices::COUNTER { a.counter_vol } else { a.mirror_coat_vol };
        // PS sim/pokemon.ts getMoveTargets: `onRedirectTarget` returns the recorded
        // attacker's slot (`getAtSlot`). A fainted pokemon there leaves no target:
        // `useMoveInner` emits `-fail` with no accuracy roll.
        let redirect_fainted = matches!(rec, Some((Some((ts, tsl)), _)) if !switched_in_alive(state, ts, tsl));
        if !matches!(rec, Some((Some(_), _))) || redirect_fainted {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
    }

    // Metal Burst / Comeuppance `onTry`: fail unless a foe damaged the user
    // this turn (`getLastDamagedBy(true)?.thisTurn`). No draw.
    if matches!(choice.move_id, Choices::METALBURST | Choices::COMEUPPANCE) {
        let last = last_damaged_by(state, side, idx);
        if let Some((ls, lsl, _, _, _)) = last {
            if !switched_in_alive(state, ls, lsl) {
                let foes: Vec<usize> = (0..2).filter(|&s| switched_in_alive(state, side.other(), s)).collect();
                if !foes.is_empty() {
                    let _ = rng.label("retarget").random(foes.len() as u32);
                }
            }
        }
        if !matches!(last, Some((_, _, _, t, _)) if t == state.turn) {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
    }

    // Rest `onTry`: fails when asleep, at full HP, or with Insomnia / Vital
    // Spirit / Comatose (no draw).
    if choice.move_id == Choices::REST {
        let p = state.side(side).team_member(idx);
        if p.status == PokemonStatus::SLEEP
            || p.hp >= p.maxhp
            || matches!(p.ability, Abilities::COMATOSE | Abilities::INSOMNIA | Abilities::VITALSPIRIT)
        {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
    }

    // Fling `onPrepareHit`: fails without an item (no draw). PS runs the move's own
    // `PrepareHit` before the user's ability, so a failed Fling also stops Libero /
    // Protean below (sim/battle-actions.ts `singleEvent('PrepareHit', move)` &&
    // `runEvent('PrepareHit', pokemon)`).
    // TODO(PS parity): item base power table, item removal, item effects.
    if choice.move_id == Choices::FLING
        && state.side(side).team_member(idx).item == crate::engine::items::Items::NONE
    {
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
        push_order(move_order);
        return;
    }

    // Payback `basePowerCallback`: doubled unless the target is newly switched
    // in or still has a queued move (`queue.willMove(target)`).
    state.side_mut(side).slot_scratch_mut(slot).bp_double = false;
    // Assurance `basePowerCallback`: doubled if the target was hurt earlier this turn.
    if choice.move_id == Choices::ASSURANCE && !is_self && !is_field {
        state.side_mut(side).slot_scratch_mut(slot).bp_double = state.side(target_side).active()[target_slot].hurt_this_turn;
    }
    if choice.move_id == Choices::PAYBACK && !is_self && !is_field {
        let tidx = state.side(target_side).active()[target_slot].pokemon_index;
        let will_move = queue.iter().any(|a| a.side == target_side && Some(a.pokemon_index) == tidx);
        let newly = state.side(target_side).active()[target_slot].newly_switched;
        state.side_mut(side).slot_scratch_mut(slot).bp_double = !will_move && !newly;
    }

    // Future Sight / Doom Desire `onTry`: queue the delayed hit on the target's
    // slot (fails if one is pending). NOT_FAIL: no accuracy roll now.
    if crate::engine::mechanics::is_future_move(choice.move_id) && !is_self && !is_field {
        if state.side(target_side).future_move[target_slot].is_some() {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
        } else {
            state.side_mut(target_side).future_move[target_slot] = Some((choice.move_id, side, idx, 0));
        }
        push_order(move_order);
        return;
    }

    // Upper Hand `onTry`: the target must still have a queued priority attack
    // (`queue.willMove(target)`).
    if choice.move_id == Choices::UPPERHAND && !is_self && !is_field {
        let tidx = state.side(target_side).active()[target_slot].pokemon_index;
        let ok = queue
            .iter()
            .find(|a| a.side == target_side && Some(a.pokemon_index) == tidx)
            // PS: `action.choice === 'move'` (a `recharge` action is not one) and
            // `move.priority` is the modified priority (`ModifyPriority`: Prankster,
            // Grassy Glide, ...), set on the move in `resolveAction`.
            .map(|a| !a.recharge && a.priority > 0 && a.choice.category != MoveCategory::Status)
            .unwrap_or(false);
        if !ok {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
    }

    // No Retreat `onTry`: fails if the volatile is already up (no draw).
    if choice.move_id == Choices::NORETREAT && state.side(side).active()[slot].no_retreat {
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
        push_order(move_order);
        return;
    }

    // Stuff Cheeks `onTry`: fails unless the user holds a berry (PS
    // data/moves.ts stuffcheeks: `source.getItem().isBerry`). No draw.
    // TODO(PS parity): the +2 Defense and the berry being eaten (`onHit`).
    if choice.move_id == Choices::STUFFCHEEKS
        && !state.side(side).team_member(idx).item.to_string().ends_with("BERRY")
    {
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
        push_order(move_order);
        return;
    }

    // Belch `onTry`: needs `ateBerry` (berries are never eaten here).
    // TODO(PS parity): `ateBerry` once berries are modeled.
    if choice.move_id == Choices::BELCH {
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
        push_order(move_order);
        return;
    }

    // PS `runEvent('PrepareHit')` (after every `onTry` failure above, before the
    // hit steps; a failed Upper Hand / No Retreat / Stuff Cheeks / Belch changes
    // nothing). PS sim/battle-actions.ts `Try` then `PrepareHit`.
    prepare_hit(state, side, slot, idx, choice);

    // Spread moves (allAdjacentFoes / allAdjacent): PS `getMoveTargets` lists
    // the (alive) allies first, then the foes. Each target is filtered like
    // the single-target pre-hit checks (no draws), then rolls its own accuracy.
    // TODO(PS parity): invulnerability exceptions (Earthquake vs Dig, ...).
    let move_ty = if expanding_force_spreads(state, side, slot, choice) {
        Tgt::AllAdjacentFoes
    } else {
        move_target(choice.move_id)
    };
    // PS `smartTarget` (Dragon Darts, multihit 2): `getSmartTargets` pairs the
    // chosen foe with its adjacent ally, chosen first. Both are accuracy-rolled
    // up front; hit N then hits `targets[N - 1]` only (`hitStepMoveHitLoop`).
    // No spread damage modifier (`spreadHit` is not set for smartTarget).
    let smart_target = choice.move_id == Choices::DRAGONDARTS
        && !is_self
        && !is_ally
        && !is_field
        && switched_in_alive(state, target_side, target_slot)
        && switched_in_alive(state, target_side, 1 - target_slot);
    let is_spread = (matches!(move_ty, Tgt::AllAdjacentFoes | Tgt::AllAdjacent) || smart_target) && !is_self && !is_field;
    let mut spread_list: Vec<(Side, usize)> = Vec::new();
    // PS `trySpreadMoveHit`: `spreadHit` is set from the original target list.
    let mut spread_count = 0usize;
    // Set when the smartTarget pair survives the accuracy step (PS clears
    // `smartTarget` on any failure, and the hits then go to the whole list).
    let mut smart_hits = false;
    // Flash Fire `onTryHit` sets `move.accuracy = true` for the rest of the move.
    let mut flash_fire_used = false;
    if is_spread {
        let mut cands: Vec<(Side, usize)> = Vec::new();
        if smart_target {
            cands.push((target_side, target_slot));
            cands.push((target_side, 1 - target_slot));
        } else {
            if move_ty == Tgt::AllAdjacent && switched_in_alive(state, side, 1 - slot) {
                cands.push((side, 1 - slot));
            }
            for sl in 0..2 {
                if switched_in_alive(state, side.other(), sl) {
                    cands.push((side.other(), sl));
                }
            }
        }
        spread_count = if smart_target { 1 } else { cands.len() };
        let damaging = matches!(choice.category, MoveCategory::Physical | MoveCategory::Special);
        // Targets dropped by Protect / Wide Guard: their `onTryHit` returns
        // NOT_FAIL, which is not a failure (`atLeastOneFailure` stays false).
        let n_cands = cands.len();
        let mut not_fail_skips = 0usize;
        for (ts, tsl) in cands {
            // PS checkMoveBypassesProtect: only moves with the `protect` flag are blocked.
            if state.side(ts).active()[tsl].protected && protect_blocks_move(choice, state.side(ts).active()[tsl].protect_kind) {
                // Each protected target still runs the variant's contact penalty.
                if choice.flags.contact {
                    protect_contact_penalty(state, rng, side, slot, idx, ts, tsl);
                }
                not_fail_skips += 1;
                continue;
            }
            // Wide Guard `onTryHit` (spread moves with the protect flag).
            // PS data/moves.ts wideguard.
            if state.side(ts).side_conditions.wide_guard > 0 && choice.flags.protect {
                not_fail_skips += 1;
                continue;
            }
            let d = state.side(ts).slot_pokemon(tsl).unwrap().clone();
            // PS sim/battle-actions.ts hitStepTryImmunity (gen >= 7): a Prankster-boosted
            // Status move skips each Dark foe on its own (no draw); the other
            // targets still get their accuracy roll.
            if ts != side
                && prankster_boosted(state, side, slot, choice)
                && (d.types.0 == PokemonType::DARK || d.types.1 == PokemonType::DARK)
            {
                continue;
            }
            // Psychic Terrain `onTryHit` per target: a priority move skips a
            // grounded foe (no accuracy roll). PS trySpreadMoveHit.
            if ts != side
                && action.priority > 0
                && state.terrain.terrain_type == crate::engine::state::Terrain::PSYCHICTERRAIN
                && slot_grounded(&d, &state.side(ts).active()[tsl], state.gravity.active)
                && !ally_semi_invulnerable(state, ts, tsl)
            {
                continue;
            }
            // Semi-invulnerable target, minus the moves that still hit it
            // (Earthquake vs Dig, ...; PS `onInvulnerability`).
            if let Some((m, _)) = state.side(ts).active()[tsl].charging.filter(|_| !state.side(ts).active()[tsl].charge_aborted) {
                let hits_anyway = match m {
                    Choices::DIG => matches!(choice.move_id, Choices::EARTHQUAKE | Choices::MAGNITUDE),
                    Choices::FLY | Choices::BOUNCE => matches!(
                        choice.move_id,
                        Choices::GUST
                            | Choices::TWISTER
                            | Choices::SKYUPPERCUT
                            | Choices::THUNDER
                            | Choices::HURRICANE
                            | Choices::SMACKDOWN
                            | Choices::THOUSANDARROWS
                    ),
                    Choices::DIVE => matches!(choice.move_id, Choices::SURF | Choices::WHIRLPOOL),
                    _ => false,
                };
                // No Guard on either side: `onAnyInvulnerability` returns 0, so the
                // move hits (PS data/abilities.ts noguard).
                // `selfdestruct: 'always'` faints the user before the hits, so the
                // user's own No Guard no longer applies (PS: the fainted holder is
                // not an active source; ps_trace seed 13015 turn 2, Golurk
                // Self-Destruct misses its ally Scolipede with no draw).
                let user_no_guard = state.side(side).team_member(idx).ability == Abilities::NOGUARD
                    && !matches!(choice.move_id, Choices::EXPLOSION | Choices::MISTYEXPLOSION | Choices::SELFDESTRUCT);
                // Lock-On `onSourceInvulnerability`: the user's locked foe is hit.
                let no_guard = user_no_guard || d.ability == Abilities::NOGUARD || lock_on_hits(state, side, slot, ts, tsl);
                if matches!(
                    m,
                    Choices::DIG | Choices::DIVE | Choices::FLY | Choices::BOUNCE | Choices::PHANTOMFORCE | Choices::SHADOWFORCE | Choices::SKYDROP
                ) && !hits_anyway
                    && !no_guard
                {
                    continue;
                }
            }
            // Magic Bounce `onTryHit` on a spread move (PS `hitStepTryHitEvent`,
            // data/abilities.ts magicbounce): a reflectable move that reaches the
            // holder is used back by the holder on its user, before this move's
            // accuracy rolls. The holder drops out of this move's targets. The
            // copy does not bounce again (its user's `bounced` flag, PS hasBounced).
            // TODO(PS parity): semi-invulnerable holder (handled above for the
            // charge turn only).
            if choice.flags.reflectable
                && (ts, tsl) != (side, slot)
                && !state.side(side).active()[slot].bounced
                && d.ability == Abilities::MAGICBOUNCE
                && !crate::engine::abilities::ignores_target_ability(state.side(side).team_member(idx), choice, &d.ability)
            {
                if let Some(bidx) = state.side(ts).active()[tsl].pokemon_index {
                    push_order(move_order);
                    let mut sub = action.clone();
                    sub.side = ts;
                    sub.slot = tsl;
                    sub.pokemon_index = bidx;
                    sub.target = slot;
                    sub.target_loc = slot as i8 + 1;
                    state.side_mut(ts).slot_scratch_mut(tsl).external_call = true;
                    state.side_mut(ts).slot_scratch_mut(tsl).bounced = true;
                    execute_action_inner(state, rng, &sub, move_order, will_act, queue);
                    state.side_mut(ts).slot_scratch_mut(tsl).external_call = false;
                    state.side_mut(ts).slot_scratch_mut(tsl).bounced = false;
                }
                continue;
            }
            // Telepathy `onTryHit`: an ally's damaging move skips the holder.
            if ts == side
                && (ts, tsl) != (side, slot)
                && damaging
                && d.ability == Abilities::TELEPATHY
                && !crate::engine::abilities::ignores_target_ability(state.side(side).team_member(idx), choice, &d.ability)
            {
                continue;
            }
            if (ts, tsl) != (side, slot) && try_absorb(state, ts, tsl, choice.move_type) {
                if d.ability == Abilities::FLASHFIRE && choice.move_type == PokemonType::FIRE {
                    flash_fire_used = true;
                }
                continue;
            }
            if (d.ability == Abilities::SOUNDPROOF && choice.flags.sound)
                || (d.ability == Abilities::BULLETPROOF && choice.flags.bullet)
            {
                continue;
            }
            // Good as Gold: the holder is immune to status / non-damaging moves.
            if d.ability == Abilities::GOODASGOLD
                && choice.category == MoveCategory::Status
                && !crate::engine::abilities::ignores_target_ability(
                    state.side(side).team_member(idx),
                    choice,
                    &d.ability,
                )
            {
                continue;
            }
            // PS sim/pokemon.ts runImmunity (`hitStepTypeImmunity`, after the TryHit
            // handlers such as Earth Eater): a Ground move needs a grounded target
            // (Magnet Rise, Levitate, ...). No draw.
            if choice.move_type == PokemonType::GROUND
                && choice.move_id != Choices::THOUSANDARROWS
                && !slot_grounded(&d, &state.side(ts).active()[tsl], state.gravity.active)
            {
                continue;
            }
            // Powder moves (PS `hitStepTryImmunity`, 'natural powder immunity'):
            // a Grass / Overcoat target is skipped before the accuracy roll.
            if choice.flags.powder
                && (ts, tsl) != (side, slot)
                && (crate::engine::mechanics::powder_immune(&d)
                    || state.side(ts).active()[tsl].added_type == PokemonType::GRASS)
            {
                continue;
            }
            if damaging
                && crate::doubles::damage::type_mod_with_added(
                    state.side(side).team_member(idx),
                    &choice.move_type,
                    &crate::doubles::damage::smackdown_ground_view(&choice.move_type, &d, state.side(ts).active()[tsl].smackdown),
                    state.side(ts).active()[tsl].added_type,
                )
                .is_none()
            {
                continue;
            }
            spread_list.push((ts, tsl));
        }
        if spread_list.is_empty() {
            // PS useMoveInner: `selfdestruct: 'always'` faints the user before
            // the target step, so it faints even when every adjacent foe is
            // skipped (a semi-invulnerable Dig holder is `-miss`ed in PS).
            if matches!(choice.move_id, Choices::EXPLOSION | Choices::MISTYEXPLOSION | Choices::SELFDESTRUCT) {
                selfdestruct_user_faint(state, rng, side, slot);
            }
            // Every target skipped (immunity, absorb, miss): the step results are
            // `false`, so `trySpreadMoveHit` returns false and the move failed
            // (Stomping Tantrum doubles). PS sim/battle-actions.ts trySpreadMoveHit
            // (`moveResult = !!targets.length`, atLeastOneFailure). When every
            // target was dropped by Protect / Wide Guard (NOT_FAIL) there is no
            // failure: `moveThisTurnResult = null`, so it is not a failed move.
            if not_fail_skips < n_cands {
                state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            }
            push_order(move_order);
            return;
        }
        target_side = spread_list[0].0;
        target_slot = spread_list[0].1;
    }

    // Semi-invulnerable target (Dig / Fly / Bounce / Dive / Phantom Force /
    // Shadow Force charge turn): the move misses with no PRNG draw (PS
    // `hitStepInvulnerabilityEvent`, the move volatiles' `onInvulnerability`
    // in data/moves.ts). No Guard on either side lets it hit
    // (`noguard.onAnyInvulnerability`).
    // TODO(PS parity): Earthquake x2 vs Dig, Surf x2 vs Dive, Gust x2 vs Fly
    // (`onSourceModifyDamage` / `onSourceBasePower`); ally-targeted moves.
    if crate::doubles::debug::flags().dmg {
        eprintln!(
            "INVULNCHK {} -> slot {} charging={:?}",
            choice.move_id, target_slot, state.side(target_side).active()[target_slot].charging
        );
    }
    // Ally-targeted moves (Dragon Cheer, Coaching, ...) miss a semi-invulnerable ally.
    // `adjacentAllyOrSelf` moves (Acupressure, Milk Drink) that picked the ally
    // are ally-targeted too (PS `hitStepInvulnerabilityEvent` for the ally).
    let ally_or_self_picked_ally = is_self
        && move_target(choice.move_id)
            == Tgt::AdjacentAllyOrSelf
        && matches!(resolved, Some((rs, rl)) if rs == side && rl != slot);
    if (is_ally && !is_self || ally_or_self_picked_ally) && !is_field && choice.move_id != Choices::HELPINGHAND {
        // PS data/moves.ts noguard `onAnyInvulnerability` (priority 1) returns 0
        // when the user or the target has No Guard: the hit lands, and the
        // dig / fly handlers never run. Coaching from a No Guard user hits a Dig ally.
        let ally_no_guard = state.side(side).team_member(idx).ability == Abilities::NOGUARD
            || state.side(side).slot_pokemon(ally_slot).is_some_and(|p| p.ability == Abilities::NOGUARD);
        if ally_semi_invulnerable(state, side, ally_slot) && !ally_no_guard {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
    }
    // Acupressure that picked the ally: Good as Gold `onTryHit` (status move,
    // target !== source) makes the ally immune before any draw. PS
    // data/abilities.ts goodasgold.onTryHit; the ally's `-immune` line is printed.
    if ally_or_self_picked_ally
        && state.side(side).slot_pokemon(ally_slot).is_some_and(|d| {
            d.ability == Abilities::GOODASGOLD
                && !crate::engine::abilities::ignores_target_ability(
                    state.side(side).team_member(idx),
                    choice,
                    &d.ability,
                )
        })
    {
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
        push_order(move_order);
        return;
    }
    let toxic_hits = crate::engine::mechanics::toxic_always_hits(choice.move_id, state.side(side).team_member(idx));
    if !is_self && !is_ally && !is_field && !toxic_hits {
        // An aborted charge (confusion self-hit, ...) already lost its `twoturnmove`
        // volatile: no semi-invulnerability (the residual handler lingers).
        if semi_invulnerable_blocks(state, side, idx, choice.move_id, target_side, target_slot) {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
    }


    // Psychic Terrain `onTryHit`: priority moves aimed at a grounded foe fail.
    // PS data/moves.ts psychicterrain.condition (spread moves: per target above).
    // `target.isSemiInvulnerable()` returns early (a move that still reaches it).
    if !is_self
        && !is_field
        && action.priority > 0
        && target_side != side
        && state.terrain.terrain_type == crate::engine::state::Terrain::PSYCHICTERRAIN
        && state.side(target_side).slot_pokemon(target_slot).is_some_and(|d| slot_grounded(d, &state.side(target_side).active()[target_slot], state.gravity.active))
        && !ally_semi_invulnerable(state, target_side, target_slot)
    {
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
        push_order(move_order);
        return;
    }

    // Armor Tail / Queenly Majesty / Dazzling `onFoeTryMove`: priority moves
    // aimed at the holder's side fail (no draw). PS data/abilities.ts armortail.
    // TODO(PS parity): `target: all` moves.
    if !is_self
        && !is_field
        && action.priority > 0
        && target_side != side
        && (0..2).any(|sl| {
            state.side(target_side).slot_pokemon(sl).is_some_and(|p| {
                p.hp > 0
                    && matches!(p.ability, Abilities::ARMORTAIL | Abilities::QUEENLYMAJESTY | Abilities::DAZZLING)
                    && !crate::engine::abilities::ignores_target_ability(
                        state.side(side).team_member(idx),
                        choice,
                        &p.ability,
                    )
            })
        })
    {
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
        push_order(move_order);
        return;
    }

    // Quick Guard `onTryHit` (priority 4, before Protect's): blocks moves with
    // positive priority that Protect would block. PS data/moves.ts quickguard.
    // TODO(PS parity): Outrage counter reset, Feint-style bypasses.
    if !is_self
        && !is_field
        && action.priority > 0
        && choice.flags.protect
        && state.side(target_side).side_conditions.quick_guard > 0
    {
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
        push_order(move_order);
        return;
    }

    // Protect: incoming non-priority moves fail before the accuracy roll.
    // (Charge turns happen before this check: Protect does not block a
    // charging Dig, only its hitting turn.)
    if !is_self
        && !is_field
        && state.side(target_side).active()[target_slot].protected
        && protect_blocks_move(choice, state.side(target_side).active()[target_slot].protect_kind)
    {
        if choice.flags.contact {
            protect_contact_penalty(state, rng, side, slot, idx, target_side, target_slot);
        }
        // Protect's `onTryHit` returns NOT_FAIL: `moveThisTurnResult` becomes
        // null (not false), but `moveResult` is false so `onMoveFail` still
        // runs (crash damage). PS sim/battle-actions.ts trySpreadMoveHit.
        if choice.crash.is_some() {
            if let Some(p) = state.side(side).slot_pokemon(slot) {
                if p.hp > 0 {
                    let dmg = (p.maxhp / 2).max(1);
                    apply_damage(state, side, slot, dmg);
                }
            }
        }
        // Steel Beam `onMoveFail` (PS data/moves.ts steelbeam): a blocked move
        // is a failed move, so the user loses round(maxhp / 2) (Magic Guard
        // blocks it). Mind Blown's `onMoveFail` is the same, but it is allAdjacent
        // and takes the spread path. TODO(PS parity): spread Mind Blown.
        if choice.move_id == Choices::STEELBEAM {
            if let Some(p) = state.side(side).slot_pokemon(slot) {
                if p.hp > 0 && p.ability != Abilities::MAGICGUARD {
                    let dmg = ((p.maxhp as f32 / 2.0).round() as i16).max(1);
                    apply_damage(state, side, slot, dmg);
                }
            }
        }
        push_order(move_order);
        return;
    }

    // Magic Bounce `onTryHit` (priority 1): a reflectable move aimed at the
    // holder is used back on its user (no accuracy roll for the original).
    // PS data/abilities.ts magicbounce. Single-target moves only; side moves
    // are handled with the side condition below.
    // A semi-invulnerable holder does not bounce (`target.isSemiInvulnerable()`
    // returns early), so a move that still reaches it (Poison-type Toxic) lands.
    // TODO(PS parity): Mold Breaker (breakable).
    let holder_semi_invulnerable = state.side(target_side).active()[target_slot]
        .charging
        .filter(|_| !state.side(target_side).active()[target_slot].charge_aborted)
        .is_some_and(|(m, _)| {
            matches!(
                m,
                Choices::DIG
                    | Choices::DIVE
                    | Choices::FLY
                    | Choices::BOUNCE
                    | Choices::PHANTOMFORCE
                    | Choices::SHADOWFORCE
                    | Choices::SKYDROP
            )
        });
    if choice.flags.reflectable
        && !is_self
        && !is_field
        && !is_spread
        && !holder_semi_invulnerable
        && (target_side, target_slot) != (side, slot)
        && !state.side(side).active()[slot].bounced
        && state
            .side(target_side)
            .slot_pokemon(target_slot)
            .is_some_and(|d| {
                d.ability == Abilities::MAGICBOUNCE
                    && !crate::engine::abilities::ignores_target_ability(
                        state.side(side).team_member(idx),
                        choice,
                        &d.ability,
                    )
            })
    {
        if let Some(bidx) = state.side(target_side).active()[target_slot].pokemon_index {
            push_order(move_order);
            let mut sub = action.clone();
            sub.side = target_side;
            sub.slot = target_slot;
            sub.pokemon_index = bidx;
            sub.target = slot;
            sub.target_loc = slot as i8 + 1;
            sub.priority = action.priority;
            state.side_mut(target_side).slot_scratch_mut(target_slot).external_call = true;
            state.side_mut(target_side).slot_scratch_mut(target_slot).bounced = true;
            execute_action_inner(state, rng, &sub, move_order, will_act, queue);
            state.side_mut(target_side).slot_scratch_mut(target_slot).external_call = false;
            state.side_mut(target_side).slot_scratch_mut(target_slot).bounced = false;
            // `onTryHit` returned null, which `hitStepTryHitEvent` turns into
            // `false`: the user's move failed (Stomping Tantrum doubles).
            // PS sim/battle-actions.ts hitStepTryHitEvent / useMove.
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            return;
        }
    }

    // Oblivious `onTryHit`: immune to Attract / Taunt (before the accuracy roll).
    if matches!(choice.move_id, Choices::TAUNT | Choices::ATTRACT)
        && !is_self
        && !is_field
        && (target_side, target_slot) != (side, slot)
        && state.side(target_side).slot_pokemon(target_slot).is_some_and(|d| {
            d.ability == Abilities::OBLIVIOUS
                && !crate::engine::abilities::ignores_target_ability(
                    state.side(side).team_member(idx),
                    choice,
                    &d.ability,
                )
        })
    {
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
        push_order(move_order);
        return;
    }

    // Ability absorbs (PS `onTryHit`, `hitStepTryHitEvent`: after Protect,
    // before type immunity and accuracy; no draws).
    if !is_self && !is_field && (side, slot) != (target_side, target_slot) && !is_spread {
        if try_absorb(state, target_side, target_slot, choice.move_type) {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
    }

    // Thunder Wave is the one Status move with `ignoreImmunity: false`: the
    // type-immunity step (before accuracy, no draw) fails it on Ground types.
    // PS data/moves.ts thunderwave.
    if choice.move_id == Choices::THUNDERWAVE
        && !is_self
        && !is_field
        && (target_side, target_slot) != (side, slot)
        && state
            .side(target_side)
            .slot_pokemon(target_slot)
            .is_some_and(|d| crate::doubles::damage::type_mod(&choice.move_type, d).is_none())
    {
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
        push_order(move_order);
        return;
    }

    // Type immunity is checked before the accuracy roll (PS moveSteps order:
    // Invulnerability -> TryHit -> TypeImmunity -> Accuracy). An immune
    // target consumes no PRNG draws at all.
    // TODO(PS parity): move-specific immunities (PS `hitStepTryImmunity`).
    let speeds = if matches!(choice.move_id, Choices::ELECTROBALL | Choices::GYROBALL) {
        (stat_speed(state, side, slot), stat_speed(state, target_side, target_slot))
    } else {
        (0, 0)
    };
    let effective_power = if dynamic_base_power(
        choice,
        state.side(side).team_member(idx),
        state.side(target_side).slot_pokemon(target_slot),
        speeds,
        false,
        false,
        side_total_fainted(state, side),
        1,
    ) > 0.0
        // Spit Up's `basePowerCallback` is 100 per Stockpile layer (onTry needs one),
        // so it is an attack with power for the immunity step (PS hitStepTryImmunity).
        || (choice.move_id == Choices::SPITUP && state.side(side).active()[slot].stockpile > 0)
        // `damageCallback` moves (Endeavor, ...) are attacks: type immunity applies.
        || (!is_self
            && !is_field
            && state.side(target_side).slot_pokemon(target_slot).is_some_and(|d| {
                choice.category != MoveCategory::Status
                    && choice.move_id.fixed_damage(state.side(side).team_member(idx), d).is_some()
            }))
    {
        1.0
    } else {
        choice.base_power
    };
    // Soundproof / Bulletproof `onTryHit`: immune to sound / ball moves (no draw).
    // An ally-targeted move (Dragon Cheer, ...) checks the ally, not the foe slot.
    let (imm_side, imm_slot) = if is_ally { (side, ally_slot) } else { (target_side, target_slot) };
    if !is_self && !is_field && (side, slot) != (imm_side, imm_slot) {
        let immune = state
            .side(imm_side)
            .slot_pokemon(imm_slot)
            .map(|d| {
                (d.ability == Abilities::SOUNDPROOF && choice.flags.sound)
                    || (d.ability == Abilities::BULLETPROOF && choice.flags.bullet)
            })
            .unwrap_or(false);
        if immune && !is_spread {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
    }

    // Destiny Bond `onPrepareHit` (PS data/moves.ts destinybond): `!removeVolatile`
    // fails the move when the volatile is already up. PS `useMoveInner` returns
    // before accuracy and the hit loop, so the only draw left is runAction's
    // `eachEvent`. The volatile itself is cleared by the caller.
    if choice.move_id == Choices::DESTINYBOND
        && state.side(side).active()[slot].destinybond
        && state.side(side).active()[slot].pokemon_index == Some(idx)
    {
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
        push_order(move_order);
        return;
    }

    // Helping Hand is `target: adjacentAlly` in PS (the table stores it as User),
    // so the Good as Gold `onTryHit` below never sees the ally. Check it here:
    // PS data/abilities.ts goodasgold.onTryHit (Status move, target !== source).
    if choice.move_id == Choices::HELPINGHAND {
        // PS data/moves.ts helpinghand.onTryHit: fails unless the ally is newly
        // switched in or still has a queued move (`queue.willMove(target)`).
        let ally_idx = state.side(side).active()[ally_slot].pokemon_index;
        let ally_ok = state.side(side).active()[ally_slot].newly_switched
            || queue.iter().any(|q| q.side == side && Some(q.pokemon_index) == ally_idx);
        if !ally_ok {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
        let immune = state
            .side(side)
            .slot_pokemon(ally_slot)
            .map(|d| {
                d.hp > 0
                    && d.ability == Abilities::GOODASGOLD
                    && !crate::engine::abilities::ignores_target_ability(
                        state.side(side).team_member(idx),
                        choice,
                        &d.ability,
                    )
            })
            .unwrap_or(false);
        if immune {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
    }

    // Good as Gold `onTryHit`: immune to status / non-damaging moves (no draw).
    // Breakable (Mold Breaker / Terabreak / ...). Spread moves filter the
    // holder out of the target list instead (handled in the spread block).
    // An `Ally` move (Dragon Cheer, ...) aims at the adjacent ally, not the foe
    // default `target_side` (PS `target` is the chosen ally).
    let (gag_side, gag_slot) = if is_ally { (side, ally_slot) } else { (target_side, target_slot) };
    if !is_self && !is_field && !is_spread && (side, slot) != (gag_side, gag_slot) {
        let immune = state
            .side(gag_side)
            .slot_pokemon(gag_slot)
            .map(|d| {
                d.ability == Abilities::GOODASGOLD
                    && choice.category == MoveCategory::Status
                    && !crate::engine::abilities::ignores_target_ability(
                        state.side(side).team_member(idx),
                        choice,
                        &d.ability,
                    )
            })
            .unwrap_or(false);
        if immune {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
    }

    // Poltergeist `onTryHit`: fails if the target is not holding an item.
    if choice.move_id == Choices::POLTERGEIST
        && !is_self
        && !is_field
        && (side, slot) != (target_side, target_slot)
        && state
            .side(target_side)
            .slot_pokemon(target_slot)
            .map(|d| d.item == crate::engine::items::Items::NONE)
            .unwrap_or(true)
    {
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
        push_order(move_order);
        return;
    }

    // Yawn `onTryHit`: a target with a status fails in `hitStepTryHitEvent`,
    // before the hit loop (no hit-loop `eachEvent` sorts). PS data/moves.ts yawn.
    if choice.move_id == Choices::YAWN
        && !is_self
        && !is_field
        && state
            .side(target_side)
            .slot_pokemon(target_slot)
            .is_some_and(|d| d.hp > 0 && d.status != PokemonStatus::NONE)
    {
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
        push_order(move_order);
        return;
    }

    if matches!(choice.category, MoveCategory::Physical | MoveCategory::Special)
        && (effective_power > 0.0
            || choice.ohko
            || matches!(
                choice.move_id,
                Choices::COUNTER
                    | Choices::MIRRORCOAT
                    | Choices::METALBURST
                    | Choices::COMEUPPANCE
                    | Choices::SEISMICTOSS
                    | Choices::NIGHTSHADE
                    | Choices::SUPERFANG
                    | Choices::NATURESMADNESS
                    | Choices::RUINATION
                    | Choices::FINALGAMBIT
            ))
        && !is_self
        && !is_field
    {
        if let Some(def) = state.side(target_side).slot_pokemon(target_slot) {
            // PS sim/pokemon.ts runImmunity: a Ground move needs a grounded
            // target (`isGrounded`: Magnet Rise, Levitate, Air Balloon, …).
            // Thousand Arrows ignores this (`ignoreImmunity`). No draw.
            let ground_immune = choice.move_type == PokemonType::GROUND
                && choice.move_id != Choices::THOUSANDARROWS
                && !slot_grounded(def, &state.side(target_side).active()[target_slot], state.gravity.active);
            let added = state.side(target_side).active()[target_slot].added_type;
            if ground_immune
                || crate::doubles::damage::type_mod_with_added(
                    state.side(side).team_member(idx),
                    &choice.move_type,
                    &crate::doubles::damage::smackdown_ground_view(&choice.move_type, def, state.side(target_side).active()[target_slot].smackdown),
                    added,
                )
                .is_none()
            {
                state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
                // Spit Up's `onAfterMove` still runs after an immune hit (PS removes
                // the Stockpile volatile and its stat drops): `end_stockpile`.
                if choice.move_id == Choices::SPITUP {
                    end_stockpile(state, side, slot);
                }
                push_order(move_order);
                return;
            }
        }
    }

    // Prankster-boosted Status moves: Dark foes are immune (PS
    // sim/battle-actions.ts `hitStepTryImmunity`, gen >= 7). No draw. PS
    // skips `isAlly` targets, so a move aimed at the user's own side (Wonder
    // Room's `all` target resolves to the user) is not affected.
    if prankster_boosted(state, side, slot, choice)
        && !is_self
        && !is_ally
        && !is_field
        && !is_spread
        && target_side != side
        && state
            .side(target_side)
            .slot_pokemon(target_slot)
            .map(|d| d.types.0 == PokemonType::DARK || d.types.1 == PokemonType::DARK)
            .unwrap_or(false)
    {
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
        push_order(move_order);
        return;
    }

    // Endeavor `onTryImmunity`: immune unless the user has less HP (no draw).
    if choice.move_id == Choices::ENDEAVOR && !is_self && !is_field {
        let user_hp = state.side(side).team_member(idx).hp;
        let target_hp = state.side(target_side).slot_pokemon(target_slot).map(|p| p.hp).unwrap_or(0);
        if user_hp >= target_hp {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
    }

    // Worry Seed `onTryImmunity` / `onTryHit`: fails (before the accuracy roll) on
    // Truant / Insomnia holders and on abilities that can't be suppressed.
    if choice.move_id == Choices::WORRYSEED && !is_self && !is_field {
        // `onTryImmunity`: Truant / Insomnia fail before the accuracy roll; the
        // `cantsuppress` check is `onTryHit`, after it (see the onHit block).
        let bad = state.side(target_side).active()[target_slot]
            .pokemon_index
            .map(|t| matches!(real_ability(state, target_side, t), Abilities::TRUANT | Abilities::INSOMNIA))
            .unwrap_or(false);
        if bad {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
    }

    // Trick / Switcheroo `onTryImmunity`: `!target.hasAbility('stickyhold')`, so a
    // Sticky Hold target fails before the accuracy roll (no draw).
    // PS data/moves.ts trick / switcheroo.
    if matches!(choice.move_id, Choices::TRICK | Choices::SWITCHEROO) && !is_self && !is_field {
        let sticky = state
            .side(target_side)
            .active()[target_slot]
            .pokemon_index
            .map(|t| real_ability(state, target_side, t) == Abilities::STICKYHOLD)
            .unwrap_or(false);
        if sticky {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
    }

    // Powder moves: PS `hitStepTryImmunity` ('natural powder immunity' for
    // Grass types) and Overcoat `onTryHit`. No draw.
    // TODO(PS parity): Mold Breaker bypass of Overcoat.
    if choice.flags.powder && !is_self && !is_field && (target_side, target_slot) != (side, slot) {
        let immune = state
            .side(target_side)
            .slot_pokemon(target_slot)
            .map(|p| crate::engine::mechanics::powder_immune(p))
            .unwrap_or(false)
            // `hasType('Grass')` includes Forest's Curse's `addedType` (kept in
            // the slot for a dual-typed target). PS sim/pokemon.ts hasType.
            || state.side(target_side).active()[target_slot].added_type == PokemonType::GRASS;
        if immune {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
    }

    // Leech Seed `onTryImmunity`: Grass targets are immune (no draw).
    // PS data/moves.ts leechseed.
    if choice.move_id == Choices::LEECHSEED && !is_self && !is_field {
        // `target.hasType('Grass')` includes a type added by Forest's Curse
        // (`addedType`, kept in the slot for a dual-typed target).
        let grass = state
            .side(target_side)
            .slot_pokemon(target_slot)
            .map(|p| p.has_type(&PokemonType::GRASS))
            .unwrap_or(false)
            || state.side(target_side).active()[target_slot].added_type == PokemonType::GRASS;
        if grass {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
    }

    // Fake Out / First Impression `onTry` (data/moves.ts): fails after the first
    // move action since switch-in (`activeMoveActions > 1`). No draw.
    if matches!(choice.move_id, Choices::FAKEOUT | Choices::FIRSTIMPRESSION)
        && state.side(side).active()[slot].active_move_actions > 1
    {
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
        push_order(move_order);
        return;
    }

    // Attract: PS `onTryImmunity` succeeds only for male -> female or
    // female -> male; Oblivious blocks it in TryHit. Both happen before the
    // accuracy roll (no draw).
    // TODO(PS parity): Aroma Veil, already-infatuated targets.
    if choice.move_id == Choices::ATTRACT && !is_self && !is_ally && !is_field {
        let att_gender = state.side(side).genders[idx as usize].clone();
        let def = state.side(target_side).active()[target_slot].pokemon_index;
        let def_gender = def.and_then(|di| state.side(target_side).genders[di as usize].clone());
        let oblivious = def
            .map(|di| state.side(target_side).team_member(di).ability == Abilities::OBLIVIOUS)
            .unwrap_or(false);
        let opposite = matches!(
            (att_gender.as_deref(), def_gender.as_deref()),
            (Some("M"), Some("F")) | (Some("F"), Some("M"))
        );
        if !opposite || oblivious {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
    }

    // Weather `onModifyAccuracy`: Thunder always hits in rain, Fire Blast in
    // sun (accuracy becomes `true`: no roll at all).
    // PS data/moves.ts thunder / hurricane (rain: true, sun: 50),
    // bleakwindstorm / sandsearstorm / wildboltstorm (rain: true),
    // blizzard (hail / snow: true).
    let eff_weather = effective_weather(state);
    let weather_guaranteed = matches!(
        (choice.move_id, eff_weather),
        (
            Choices::THUNDER
                | Choices::HURRICANE
                | Choices::BLEAKWINDSTORM
                | Choices::SANDSEARSTORM
                | Choices::WILDBOLTSTORM,
            Weather::RAIN | Weather::HEAVYRAIN
        ) | (Choices::BLIZZARD, Weather::HAIL | Weather::SNOW)
    );

    // Octolock `onTryImmunity`: Ghost targets are immune (PS typechart `trapped`
    // immunity; only Octolock checks it, the Bind family does not). No draw.
    let traps = choice.volatile_status.as_ref().is_some_and(|v| v.volatile_status == PokemonVolatileStatus::OCTOLOCK);
    if traps && !is_self && !is_field && (side, slot) != (target_side, target_slot) {
        let ghost = state
            .side(target_side)
            .slot_pokemon(target_slot)
            .map(|p| p.has_type(&PokemonType::GHOST))
            .unwrap_or(false);
        if ghost {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
    }

    // Accuracy roll (drawn even at 100%; skipped for always-hit / 0.0).
    if crate::doubles::debug::flags().dmg {
        eprintln!("REACHED_ACC {} {} target_slot={} has_target={}", name, choice.move_id, target_slot, has_target);
    }
    if !is_spread
        && !choice.always_hit
        && !choice.ohko
        && move_ty != Tgt::All
        && !toxic_hits
        && !weather_guaranteed
        && roll_accuracy_of(choice) > 0.0
    {
        accuracy_event_ties(state, rng, side, slot, target_side, target_slot);
    }
    // PS `hitStepAccuracy` sets `battle.activeTarget` to each target in turn
    // (all of them, hit or miss): the last one is still set when the first
    // hit's Substitute `onTryPrimaryHit` runs.
    let mut ps_active_target: Option<(Side, usize)> = spread_list.last().copied();
    if is_spread {
        // PS `hitStepAccuracy`: one roll per target, in target order.
        if !choice.always_hit && !toxic_hits && !weather_guaranteed && !flash_fire_used && roll_accuracy_of(choice) > 0.0 {
            let ab = state.side(side).slot_boosts(slot).map(|b| b.acc).unwrap_or(0);
            // PS useMoveInner: `selfdestruct: 'always'` faints the user before the
            // hits, and `allies()` / `foes()` skip 0-HP pokemon, so the user's own
            // No Guard (`onAnyAccuracy`) no longer applies.
            let user_no_guard = state.side(side).team_member(idx).ability == Abilities::NOGUARD
                && !matches!(choice.move_id, Choices::EXPLOSION | Choices::MISTYEXPLOSION | Choices::SELFDESTRUCT);
            let ignore_eva = ignores_evasion(evasion_ability(state, side, idx, choice.move_id), choice.move_id);
            let st = &*state;
            spread_list.retain(|&(ts, tsl)| {
                accuracy_event_ties(st, rng, side, slot, ts, tsl);
                // Glaive Rush `onAccuracy`: moves against the holder always hit.
                if st.side(ts).active()[tsl].glaive_rush {
                    return true;
                }
                // Lock-On `onSourceAccuracy`: the user's locked foe always gets hit.
                if lock_on_hits(st, side, slot, ts, tsl) {
                    return true;
                }
                // Minimize `onAccuracy`: minimize-flagged moves always hit it.
                if crate::engine::mechanics::minimize_move(choice.move_id) && st.side(ts).active()[tsl].minimize {
                    return true;
                }
                // No Guard `onAnyAccuracy`: no roll when either side has it.
                if user_no_guard
                    || st.side(ts).slot_pokemon(tsl).map(|p| p.ability == Abilities::NOGUARD).unwrap_or(false)
                {
                    return true;
                }
                let eb = if ignore_eva {
                    0
                } else {
                    st.side(ts).slot_boosts(tsl).map(|b| b.eva).unwrap_or(0)
                };
                let ab = if unaware_zeroes_accuracy(st, side, idx, ts, tsl, choice) { 0 } else { ab };
                roll_accuracy_boosted(roll_accuracy_of(choice), ab, eb, accuracy_ability_mods(st, side, idx, ts, tsl, choice), rng)
            });
        }
        smart_hits = smart_target && spread_list.len() == 2;
        if spread_list.is_empty() {
            // Every target missed: PS `hitStepAccuracy` leaves `atLeastOneFailure`
            // set, so `moveThisTurnResult` is `false` (Stomping Tantrum reads it).
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            // `selfdestruct: 'always'` faints the user before the accuracy rolls
            // (PS useMoveInner; ifHit moves faint only when they hit).
            if matches!(choice.move_id, Choices::EXPLOSION | Choices::MISTYEXPLOSION | Choices::SELFDESTRUCT) {
                selfdestruct_user_faint(state, rng, side, slot);
            }
            push_order(move_order);
            return;
        }
        target_side = spread_list[0].0;
        target_slot = spread_list[0].1;
    } else if !choice.always_hit
        && !choice.ohko
        && move_ty != Tgt::All
        && !toxic_hits
        && !lock_on_hits(state, side, slot, target_side, target_slot)
        && !state.side(target_side).active()[target_slot].glaive_rush
        && !(crate::engine::mechanics::minimize_move(choice.move_id)
            && state.side(target_side).active()[target_slot].minimize)
        && !weather_guaranteed
        && roll_accuracy_of(choice) > 0.0
        && !(state.side(side).team_member(idx).ability == Abilities::NOGUARD
            && !matches!(choice.move_id, Choices::EXPLOSION | Choices::MISTYEXPLOSION | Choices::SELFDESTRUCT)
            || state
                .side(target_side)
                .slot_pokemon(target_slot)
                .map(|p| p.ability == Abilities::NOGUARD)
                .unwrap_or(false))
    {
        let ab = if unaware_zeroes_accuracy(state, side, idx, target_side, target_slot, choice) {
            0
        } else {
            state.side(side).slot_boosts(slot).map(|b| b.acc).unwrap_or(0)
        };
        let eb = if ignores_evasion(evasion_ability(state, side, idx, choice.move_id), choice.move_id) {
            0
        } else {
            state.side(target_side).slot_boosts(target_slot).map(|b| b.eva).unwrap_or(0)
        };
        let mods = accuracy_ability_mods(state, side, idx, target_side, target_slot, choice);
        if !roll_accuracy_boosted(roll_accuracy_of(choice), ab, eb, mods, rng) {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            if matches!(choice.move_id, Choices::EXPLOSION | Choices::MISTYEXPLOSION | Choices::SELFDESTRUCT) {
                selfdestruct_user_faint(state, rng, side, slot);
            }
            // Spit Up's `onAfterMove` still runs after a miss (PS runMove runs
            // `AfterMove` after `useMove`): the Stockpile volatile ends.
            if choice.move_id == Choices::SPITUP {
                end_stockpile(state, side, slot);
            }
            push_order(move_order);
            return;
        }
    }

    // `hitStepBreakProtect` (PS sim/battle-actions.ts, step 5: after TryHit /
    // Protect, type immunity, TryImmunity and accuracy): `breaksProtect` moves
    // remove the target's protection volatiles (and its `stall` counter). A
    // Protect that blocks the move has already returned above.
    // The target's side also loses its guards (PS `target.side.removeSideCondition`
    // for craftyshield, matblock, quickguard, wideguard; gen >= 6 for allies too).
    if matches!(
        choice.move_id,
        Choices::FEINT | Choices::PHANTOMFORCE | Choices::SHADOWFORCE | Choices::HYPERSPACEFURY | Choices::HYPERSPACEHOLE
    ) && !is_self
        && !is_field
    {
        // PS hitStepBreakProtect: `broke` when any protection volatile or guard
        // side condition was removed; then (gen >= 6) `delete target.volatiles['stall']`.
        let sc = &mut state.side_mut(target_side).side_conditions;
        // `broke`: a guard side condition was removed (it also clears `stall`).
        let broke_side = sc.crafty_shield > 0 || sc.mat_block > 0 || sc.quick_guard > 0 || sc.wide_guard > 0;
        sc.crafty_shield = 0;
        sc.mat_block = 0;
        sc.quick_guard = 0;
        sc.wide_guard = 0;
        // PS: `if (broke) { ...; if (gen >= 6) delete target.volatiles['stall']; }`.
        let t = state.side_mut(target_side).active_slot_mut(target_slot);
        if t.protected || broke_side {
            t.protected = false;
            t.stall = None;
        }
    }

    // Uproar `onTryHit` (move-level, so after the immunity and accuracy steps):
    // cures sleep on every active pokemon.
    if choice.move_id == Choices::UPROAR {
        for sd in [Side::One, Side::Two] {
            for sl in 0..2 {
                if let Some(ti) = state.side(sd).active()[sl].pokemon_index {
                    let pm = state.side_mut(sd).team_member_mut(ti);
                    if pm.status == PokemonStatus::SLEEP {
                        pm.status = PokemonStatus::NONE;
                    }
                }
            }
        }
    }

    // OHKO moves (`hitStepAccuracy`): immune when the target out-levels the user
    // (or has the Ice type for Sheer Cold); otherwise accuracy 30 (20 for a
    // non-Ice Sheer Cold user) + the level difference, bypassing every
    // accuracy modifier. PS sim/battle-actions.ts.
    // Sturdy `onTryHit`: OHKO moves are blocked before the accuracy roll.
    if choice.ohko
        && !is_self
        && !is_field
        && !is_spread
        && (side, slot) != (target_side, target_slot)
        && state.side(target_side).slot_pokemon(target_slot).is_some_and(|d| {
            d.ability == Abilities::STURDY
                && !crate::engine::abilities::ignores_target_ability(
                    state.side(side).team_member(idx),
                    choice,
                    &d.ability,
                )
        })
    {
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
        push_order(move_order);
        return;
    }
    if choice.ohko
        && !is_self
        && !is_field
        && !is_spread
        && (side, slot) != (target_side, target_slot)
        // PS `target.isSemiInvulnerable()`: a non-semi-invulnerable charge (Solar
        // Beam, Razor Wind, ...) still rolls.
        && !ally_semi_invulnerable(state, target_side, target_slot)
    {
        let (alevel, atypes) = {
            let a = state.side(side).team_member(idx);
            (a.level as i32, a.types)
        };
        let (dlevel, dtypes) = {
            let d = state.side(target_side).slot_pokemon(target_slot).unwrap();
            (d.level as i32, d.types)
        };
        let ice = |t: (PokemonType, PokemonType)| t.0 == PokemonType::ICE || t.1 == PokemonType::ICE;
        let sheer = choice.move_id == Choices::SHEERCOLD;
        if alevel < dlevel || (sheer && ice(dtypes)) {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
        let mut acc = if sheer && !ice(atypes) { 20 } else { 30 };
        acc += alevel - dlevel;
        let no_guard = state.side(side).team_member(idx).ability == Abilities::NOGUARD
            || state.side(target_side).slot_pokemon(target_slot).is_some_and(|d| d.ability == Abilities::NOGUARD);
        if !no_guard && !rng.label("accuracy").random_chance(acc.max(0) as u32, 100) {
            state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
            push_order(move_order);
            return;
        }
    }

    // Brick Break / Psychic Fangs / Raging Bull `onTryHit`: shatter the target
    // side's screens. PS runs it in `moveHit` (sim/battle-actions.ts), after the
    // type-immunity and accuracy steps (an immune or missed target keeps them),
    // and before the substitute check (so it shatters through a Substitute).
    // PS data/moves.ts brickbreak / psychicfangs. TODO(PS parity): Raging Bull's
    // type-dependent variants.
    if matches!(choice.move_id, Choices::BRICKBREAK | Choices::PSYCHICFANGS | Choices::RAGINGBULL)
        && !is_self
        && !is_field
    {
        let sc = &mut state.side_mut(target_side).side_conditions;
        sc.reflect = 0;
        sc.light_screen = 0;
        sc.aurora_veil = 0;
    }

    // Substitute `onTryPrimaryHit` (after the accuracy step): a Status move
    // aimed at another pokemon's substitute fails (getDamage -> undefined)
    // unless it is a sound move or the user has Infiltrator. A spread move
    // checks each target in the hit loop instead.
    if choice.category == MoveCategory::Status
        && !is_self
        && !is_field
        && !is_spread
        && (target_side, target_slot) != (side, slot)
        && state.side(target_side).active()[target_slot].substitute_health > 0
        && !bypasses_substitute(choice)
        && state.side(side).team_member(idx).ability != Abilities::INFILTRATOR
    {
        // PS `spreadMoveHit`: the move's own `TryHit` runs before
        // `tryPrimaryHitEvent`. A failing `onTryHit` returns `[false]`, so the
        // hit loop breaks before its `eachEvent('Update')`.
        if !status_on_try_hit_fails(state, side, slot, idx, target_side, target_slot, choice, queue) {
            // `onTryPrimaryHit` returns null (not false): the hit loop still runs its
            // `eachEvent('Update')` (once in the loop, once after it).
            each_event_update(state, rng);
            each_event_update(state, rng);
        }
        push_order(move_order);
        return;
    }

    // `adjacentAllyOrSelf` moves (Acupressure, Milk Drink, ...: the engine's move
    // target is User, so the block above skips them): an ally holding a
    // Substitute fails them in `TryPrimaryHit` (status moves have no damage:
    // `getDamage` is falsy), unless the move bypasses it or the user is an
    // Infiltrator. The user itself is not blocked.
    if move_target(choice.move_id) == Tgt::AdjacentAllyOrSelf
        && choice.category == MoveCategory::Status
        && !bypasses_substitute(choice)
        && state.side(side).team_member(idx).ability != Abilities::INFILTRATOR
    {
        if let Some((ts, tsl)) = resolved {
            if ts == side && tsl != slot && state.side(ts).active()[tsl].substitute_health > 0 {
                each_event_update(state, rng);
                each_event_update(state, rng);
                push_order(move_order);
                return;
            }
        }
    }

    // Multi-hit count.
    // Beat Up `onModifyMove`: one hit per team member that is the user or is
    // unfainted and status-free (`side.pokemon` order); each hit's base power
    // is 5 + baseAtk/10 of that member.
    let mut beat_up_hits = 0u32;
    state.side_mut(side).slot_scratch_mut(slot).beat_up_queue.clear();
    state.side_mut(side).slot_scratch_mut(slot).bp_fixed = 0.0;
    if choice.move_id == Choices::BEATUP {
        let sd = state.side(side);
        let mut q: Vec<f32> = Vec::new();
        for &t in sd.ps_order.iter() {
            let p = &sd.team[t];
            if p.id == crate::pokemon::PokemonName::NONE || sd.species_names[t].is_empty() {
                continue;
            }
            if t == idx as usize || (p.hp > 0 && p.status == PokemonStatus::NONE) {
                let sid = crate::doubles::state::to_id(&sd.species_names[t]);
                let atk = crate::doubles::species_types::species_data(&sid)
                    .map(|d| d.base_stats.1)
                    .unwrap_or(100);
                q.push((5 + atk / 10) as f32);
            }
        }
        beat_up_hits = q.len() as u32;
        state.side_mut(side).slot_scratch_mut(slot).beat_up_queue = q;
    }
    let hit_count: u32 = match choice.multihit {
        _ if beat_up_hits > 0 => beat_up_hits,
        0 => 1,
        // Skill Link `onModifyMove` (PS data/abilities.ts): the 2-5 range is 5, no draw.
        20 if state.side(side).team_member(idx).ability == Abilities::SKILLLINK => 5,
        20 => {
            // 35-35-15-15 out of 100 for 2-3-4-5 hits (PS sample of 20).
            const TABLE: [u32; 20] = [2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5];
            let n = TABLE[rng.label("multihit").random(20) as usize];
            // Loaded Dice (PS scripts.ts hitStepMoveHitLoop): 4-5 hits instead of 2-3.
            if n < 4 && state.side(side).team_member(idx).item == crate::engine::items::Items::LOADEDDICE {
                5 - rng.label("multihit").random(2)
            } else {
                n
            }
        }
        n => n as u32,
    };

    let is_attack = matches!(choice.category, MoveCategory::Physical | MoveCategory::Special);
    let mut total_damage: i16 = 0;
    // A hit that did nothing ends PS's hit loop with no `eachEvent`.
    // TODO(PS parity): other failing status effects (boosts at +6, ...).
    let mut move_failed = false;
    // PS `onHit` returning NOT_FAIL (Heal Pulse at full HP): the move's result is
    // `null`, not `false`, so Stomping Tantrum's `moveLastTurnResult === false` stays off.
    let mut not_fail_result = false;
    let mut heal_fail_event = false;
    let mut hits_done = 0u32;
    // Some hit of a spread attack landed on a target (PS `damage` array).
    let mut any_hit_landed = false;
    // Hits that landed on a Substitute only: PS's closing `eachEvent` is skipped
    // (`damage` holds no number).
    let mut sub_hit = false;
    let mut real_hit = false;
    // PS `move.hitTargets` (targets that were not immune / missed), for Magician's
    // `speedSort(hitTargets)`: the target's side and team index.
    let mut magician_targets: smallvec::SmallVec<[(Side, PokemonIndex); 4]> = smallvec::SmallVec::new();
    // Cloud Nine / Air Lock `onEnd` (faint): the ability ending triggers
    // `eachEvent('WeatherChange')` — a speed sort. Set when such a holder
    // faints this move; consumed after the per-hit eachEvent.
    let mut weather_suppress_fainted = false;
    let mut weather_suppress_holder: Option<(Side, usize)> = None;
    // Targets that took a hit this move (PS `targetsCopy` of `afterMoveSecondaryEvent`).
    let mut after_hit_targets: smallvec::SmallVec<[(Side, usize); 4]> = smallvec::SmallVec::new();
    // Targets that got the `sparklingaria` volatile (PS data/moves.ts sparklingaria).
    let mut sparkling_marked: Vec<(Side, usize)> = Vec::new();
    let mut drop_landed = false;
    // Parting Shot aimed at a Mirror Armor holder (PS `onHit`: `success` is false
    // when the drops bounced, but `!target.hasAbility('mirrorarmor')` keeps selfSwitch).
    let mut parting_shot_held = false;
    // A spread boost-only status move: a target that changed nothing is only a
    // per-target failure. The move fails when no target changed (see HITEND below).
    let mut spread_boost_noop = false;
    // A spread Status move blocked by a target's Substitute: that target's
    // `damage` is null (TryPrimaryHit), not false, so the hit loop does not
    // count as failed. PS sim/battle-actions.ts spreadMoveHit / hitStepMoveHitLoop.
    let mut spread_sub_null = false;
    // Spread status move whose volatile (confusion) failed on some targets: PS
    // `spreadMoveHit` returns `false` per target, and the move fails only when
    // every target gave `false`. Consumed after the hit loop's targets.
    let mut spread_volatile_fail = false;
    let mut spread_volatile_ok = false;
    let mut extra_order: Vec<OrderEntry> = Vec::new();
    // Moves a Copycat / Sleep Talk `onHit` ran (`useMove`) inside the hit loop.
    let mut copy_order: Vec<OrderEntry> = Vec::new();
    let mut berserk_before: Vec<(Side, usize, i16)> = Vec::new();
    let mut ee_before: smallvec::SmallVec<[(Side, usize, i16); 4]> = smallvec::SmallVec::new();

    state.side_mut(side).slot_scratch_mut(slot).hit_number = 1;
    let faint_mark = state.faint_log.len();
    // The user's HP before its hits, for its own Emergency Exit (contact
    // damage, Life Orb-style self damage, recoil: see after the recoil).
    let user_hp_before = state.side(side).slot_pokemon(slot).map(|p| p.hp).unwrap_or(0);
    // PS hitStepMoveHitLoop `isSleepUsable`: the move (Snore, Sleep Talk) or
    // its source (a move called by Sleep Talk, the only call an asleep user
    // makes) is usable while asleep.
    let sleep_usable = matches!(choice.move_id, Choices::SNORE | Choices::SLEEPTALK)
        || (state.side(side).active()[slot].external_call
            && state.side(side).team_member(idx).status == PokemonStatus::SLEEP);
    'hit_loop: for hit_idx in 0..hit_count {
      // PS hitStepMoveHitLoop: `if (hit > 1 && pokemon.status === 'slp' &&
      // !isSleepUsable) break;` (the user fell asleep mid-move: Effect Spore).
      if hit_idx > 0 && !sleep_usable && state.side(side).team_member(idx).status == PokemonStatus::SLEEP {
          break 'hit_loop;
      }
      state.side_mut(side).slot_scratch_mut(slot).hit_number = hit_idx as u8 + 1;
      if choice.move_id == Choices::BEATUP {
          let a = state.side_mut(side).active_slot_mut(slot);
          if !a.beat_up_queue.is_empty() {
              a.bp_fixed = a.beat_up_queue.remove(0);
          }
      }
      let hit_targets: Vec<(Side, usize)> = if smart_hits {
          vec![spread_list[hit_idx as usize]]
      } else if is_spread {
          spread_list.clone()
      } else {
          vec![(target_side, target_slot)]
      };
      if smart_hits {
          (target_side, target_slot) = hit_targets[0];
      }
      // PS hitStepMoveHitLoop: `if (targets.every(target => !target?.hp)) break;`
      // at the top of every hit after the first (all multi-hit moves).
      if hit_idx > 0 && !is_self && !is_field && !hit_targets.iter().any(|&(ts, tsl)| switched_in_alive(state, ts, tsl)) {
          break 'hit_loop;
      }
      // `multiaccuracy` (Triple Axel / Triple Kick / Population Bomb): hits after the first roll
      // accuracy again; a miss ends the loop. PS sim/battle-actions.ts
      // hitStepMoveHitLoop. The loop top checks `targets.every(!hp)` before the
      // roll, so a fainted target ends it here with no draw.
      // TODO(PS parity): ModifyAccuracy / Accuracy events (abilities, items).
      if hit_idx > 0 && matches!(choice.move_id, Choices::TRIPLEAXEL | Choices::TRIPLEKICK | Choices::POPULATIONBOMB) {
          if !is_self && !is_field && !hit_targets.iter().any(|&(ts, tsl)| switched_in_alive(state, ts, tsl)) {
              break 'hit_loop;
          }
          let ab = if unaware_zeroes_accuracy(state, side, idx, target_side, target_slot, choice) {
              0
          } else {
              state.side(side).slot_boosts(slot).map(|b| b.acc).unwrap_or(0)
          };
          let eb = if ignores_evasion(evasion_ability(state, side, idx, choice.move_id), choice.move_id) {
              0
          } else {
              state.side(target_side).slot_boosts(target_slot).map(|b| b.eva).unwrap_or(0)
          };
          let mods = accuracy_ability_mods(state, side, idx, target_side, target_slot, choice);
          // No Guard `onAnyAccuracy` (user or target) and Glaive Rush / Lock-On
          // make the `Accuracy` event return true: no roll for the later hits
          // either (PS champions scripts.ts hitStepMoveHitLoop -> runEvent('Accuracy')).
          let no_guard_hit = state.side(side).team_member(idx).ability == Abilities::NOGUARD
              || state.side(target_side).slot_pokemon(target_slot).map(|p| p.ability == Abilities::NOGUARD).unwrap_or(false)
              || state.side(target_side).active()[target_slot].glaive_rush
              || lock_on_hits(state, side, slot, target_side, target_slot);
          if !no_guard_hit && !crate::doubles::damage::roll_multiaccuracy(choice.accuracy, ab, eb, mods, rng) {
              break;
          }
      }
      // PS `getSpreadDamage`: every target's crit + damage roll comes before
      // any per-target effect.
      let pre_damage: smallvec::SmallVec<[(i16, bool); 4]> = if is_spread && is_attack {
          let sp = spread_count > 1;
          // A substitute's `onTryPrimaryHit` computes its damage first
          // (before `getSpreadDamage` covers the other targets).
          let bypass = bypasses_substitute(choice)
              || state.side(side).team_member(idx).ability == Abilities::INFILTRATOR;
          let mut order: Vec<usize> = (0..hit_targets.len()).collect();
          order.sort_by_key(|&i| {
              let (ts, tsl) = hit_targets[i];
              let has_sub = !bypass && state.side(ts).active()[tsl].substitute_health > 0;
              (!has_sub) as u8
          });
          let mut out: smallvec::SmallVec<[(i16, bool); 4]> = smallvec::smallvec![(0i16, false); hit_targets.len()];
          for i in order {
              let (ts, tsl) = hit_targets[i];
              // PS getSpreadDamage skips a target that is gone (`!target`).
              if state.side(ts).slot_pokemon(tsl).is_none() {
                  continue;
              }
              let has_sub = !bypass && state.side(ts).active()[tsl].substitute_health > 0;
              // PS substitute.onTryPrimaryHit -> `getDamage` with the stale
              // `activeTarget`; `getSpreadDamage` then sets it per target.
              // data/abilities.ts unaware.onAnyModifyBoost reads it.
              state.tl.sub_active_target.set(if has_sub { ps_active_target } else { None });
              out[i] = compute_hit_damage(state, rng, side, slot, idx, ts, tsl, choice, ate_boosted, sp);
              state.tl.sub_active_target.set(None);
              if !has_sub {
                  ps_active_target = Some((ts, tsl));
              }
          }
          // PS `spreadDamage`: every target's `Damage` event (Damp) runs after
          // all the rolls and before any secondary.
          for (i, &(d, _)) in out.iter().enumerate() {
              if d > 0 {
                  crate::doubles::events::run_event_ties(state, rng, crate::doubles::events::Event::Damage, hit_targets[i], Some((side, slot)));
              }
          }
          out
      } else {
          smallvec::SmallVec::new()
      };
      let mut hit_damage_by: smallvec::SmallVec<[Option<i16>; 4]> = smallvec::smallvec![None; hit_targets.len()];
      // A target took something this hit (damage, or a Substitute hit): PS
      // `hitStepMoveHitLoop` skips `eachEvent` when every `damage[i]` is false.
      let mut hit_landed = false;
      let mut hit_pi_by: smallvec::SmallVec<[_; 4]> = smallvec::smallvec![None; hit_targets.len()];
      for (ti, &(hts, htsl)) in hit_targets.iter().enumerate() {
        target_side = hts;
        target_slot = htsl;
        // PS data/moves.ts batonpass.onHit: with no replacement (`canSwitch`
        // false) it is `-fail` (NOT_FAIL), so the hit did nothing and
        // hitStepMoveHitLoop breaks before its `eachEvent('Update')`.
        if choice.move_id == Choices::BATONPASS && state.side(side).find_replacement().is_none() {
            move_failed = true;
        }
        // Substitute `onTryPrimaryHit` for a spread Status move: this target
        // is skipped (the accuracy roll was already drawn). It returns `null`,
        // not `false`, so the hit still "did something" for the hit loop's
        // `moveDamage.some(val => val !== false)` and its `eachEvent('Update')`
        // (PS data/moves.ts substitute.onTryPrimaryHit, sim/battle-actions.ts
        // hitStepMoveHitLoop).
        if is_spread
            && choice.category == MoveCategory::Status
            && (hts, htsl) != (side, slot)
            && state.side(hts).active()[htsl].substitute_health > 0
            && !bypasses_substitute(choice)
            && state.side(side).team_member(idx).ability != Abilities::INFILTRATOR
        {
            // `onTryPrimaryHit` returns null (not false): the hit loop still
            // counts it (`damage[i] = 0`), so the move does not fail.
            spread_sub_null = true;
            spread_volatile_ok = true;
            drop_landed = true;
            continue;
        }
        // The target as it was when the hit started (a KO clears the slot).
        let hit_target_pi = state.side(hts).active()[htsl].pokemon_index;
        // PS keeps a KO'd target's volatiles until the faint is processed after
        // the move, so the Beak Blast `onHit` still sees the volatile.
        let hit_target_beak = state.side(hts).active()[htsl].beak_blast;
        // Damage this hit dealt (PS `damagedTargets`), for DamagingHit.
        let mut hit_damage: Option<i16> = None;
        if crate::doubles::debug::flags().dmg {
            eprintln!(
                "HITLOOP {} is_attack={} is_self={} is_field={} has_target={} target_alive={}",
                choice.move_id,
                is_attack,
                is_self,
                is_field,
                has_target,
                switched_in_alive(state, target_side, target_slot)
            );
        }
        // Ally-targeted moves (Coaching, ...) need the ALLY alive, not the
        // default foe slot 0.
        let target_ok = if is_ally {
            switched_in_alive(state, side, ally_slot)
        } else {
            switched_in_alive(state, target_side, target_slot)
        };
        if !is_self && !is_field && !target_ok {
            // PS `hitStepMoveHitLoop`: `if (targets.every(target => !target?.hp)) break;`
            // A fainted target ends the hit loop (no further hits, no further
            // `eachEvent('Update')`); for a spread move only the remaining
            // (alive) targets would still be processed.
            let mut any_alive = false;
            // An ally-targeted move's target is the ally (`target_ok` above),
            // not the default foe slot in `hit_targets`.
            let check_targets: Vec<(Side, usize)> = if is_ally { vec![(side, ally_slot)] } else { hit_targets.clone() };
            for &(ts, tsl) in check_targets.iter() {
                if switched_in_alive(state, ts, tsl) {
                    any_alive = true;
                    break;
                }
            }
            if !any_alive {
                break 'hit_loop;
            }
            continue;
        }

        // Pollen Puff on an ally (`onTryHit`: basePower 0, so `getDamage`
        // draws nothing; `onHit`: heal floor(baseMaxhp / 2)). The user's Heal
        // Block stops it (`onTryMove`). PS data/moves.ts pollenpuff.
        // TODO(PS parity): the Heal Block `onTryMove` fail comes before accuracy.
        let pollen_ally = choice.move_id == Choices::POLLENPUFF && target_side == side && has_target;
        if pollen_ally {
            if state.side(side).active()[slot].heal_block > 0 {
                move_failed = true;
            } else if state.side(target_side).active()[target_slot].heal_block == 0 {
                if let Some(tidx) = state.side(target_side).active()[target_slot].pokemon_index {
                    let pm = state.side_mut(target_side).team_member_mut(tidx);
                    if pm.hp > 0 {
                        pm.hp = (pm.hp + pm.maxhp / 2).min(pm.maxhp);
                    }
                }
            }
        }
        if is_attack && !is_self && !is_field && has_target && !pollen_ally {
            let (damage, was_crit) = pre_damage
                .get(ti)
                .copied()
                .unwrap_or_else(|| compute_hit_damage(state, rng, side, slot, idx, target_side, target_slot, choice, ate_boosted, false));
            // Substitute `onTryPrimaryHit`: a damaging move hits the substitute
            // instead (sound moves and Infiltrator bypass it). No further
            // effects on that target (`HIT_SUBSTITUTE`).
            // TODO(PS parity): AfterSubDamage events.
            let sub_hp = state.side(target_side).active()[target_slot].substitute_health;
            if sub_hp > 0
                && (target_side, target_slot) != (side, slot)
                && !bypasses_substitute(choice)
                && state.side(side).team_member(idx).ability != Abilities::INFILTRATOR
            {
                sub_hit = true;
                after_hit_targets.push((target_side, target_slot));
                hit_landed = true;
                if let Some(tpi) = hit_target_pi {
                    if !magician_targets.contains(&(hts, tpi)) {
                        magician_targets.push((hts, tpi));
                    }
                }
                let d = damage.min(sub_hp);
                state.side_mut(target_side).active_slot_mut(target_slot).substitute_health = sub_hp - d;
                total_damage = total_damage.saturating_add(d);
                // Ceaseless Edge / Stone Axe `onAfterSubDamage`: the hazard is set
                // on the foe side when the user is alive and not Sheer Force
                // (PS data/moves.ts ceaselessedge / stoneaxe; the substitute
                // condition's `onTryPrimaryHit` runs AfterSubDamage).
                // TODO(PS parity): Rapid Spin / Mortal Spin / Ice Spinner / Steel
                // Roller `onAfterSubDamage`.
                if matches!(choice.move_id, Choices::CEASELESSEDGE | Choices::STONEAXE)
                    && state.side(side).team_member(idx).ability != Abilities::SHEERFORCE
                    && state.side(side).team_member(idx).hp > 0
                {
                    let hz = if choice.move_id == Choices::CEASELESSEDGE {
                        PokemonSideCondition::Spikes
                    } else {
                        PokemonSideCondition::Stealthrock
                    };
                    apply_side_condition(state, side.other(), hz);
                }
                // PS substitute.onTryPrimaryHit: `this.heal(ceil(damage * drain), source,
                // target, 'drain')`, so the Substitute holder's Liquid Ooze
                // (`onSourceTryHeal`) turns it into damage, as on a normal hit.
                let ooze = state.side(target_side).slot_pokemon(target_slot).is_some_and(|p| p.ability == Abilities::LIQUIDOOZE);
                if let (Some(frac), true) = (choice.drain, ooze) {
                    let heal = ((d as f32) * frac).ceil() as i16;
                    if heal > 0 {
                        apply_indirect_damage(state, side, slot, heal);
                    }
                } else if let Some(frac) = choice.drain {
                    let heal = ((d as f32) * frac).ceil() as i16;
                    // Liquid Ooze (the Substitute holder's) `onSourceTryHeal`
                    // turns this heal into damage too (PS substitute
                    // `onTryPrimaryHit` -> `heal(..., 'drain')`, data/abilities.ts liquidooze).
                    let ooze = state
                        .side(target_side)
                        .slot_pokemon(target_slot)
                        .is_some_and(|p| p.ability == Abilities::LIQUIDOOZE);
                    if ooze {
                        if heal > 0 {
                            apply_indirect_damage(state, side, slot, heal);
                        }
                    } else if heal > 0 && state.side(side).active()[slot].heal_block == 0 {
                        let s = state.side_mut(side);
                        let pm = s.team_member_mut(idx);
                        pm.hp = (pm.hp + heal).min(pm.maxhp);
                    }
                }
                // `selfDrops` (Close Combat, ...) still runs for a hit on the
                // Substitute: its roll is drawn and the user's drop applies.
                if let (true, Some(b)) = (ti == 0, &choice.self_boosts) {
                    let roll = rng.label("self boost").random(100) as u32;
                    let apply = if choice.self_boost_chance > 0.0 {
                        roll < choice.self_boost_chance as u32
                    } else {
                        true
                    };
                    if apply {
                        apply_boosts(state, side, idx, b, false);
                    }
                }
                // `secondaries` still rolls once per entry (the target is
                // `null`, not `false`), but nothing is applied.
                let sheer = state.side(side).team_member(idx).ability == Abilities::SHEERFORCE && has_secondaries(choice);
                let rolls = if sheer { 0 } else { match choice.move_id {
                    Choices::TRIATTACK
                    | Choices::DIRECLAW
                    | Choices::SALTCURE
                    | Choices::SPIRITSHACKLE
                    | Choices::THROATCHOP
                    | Choices::EERIESPELL
                    | Choices::PSYCHICNOISE
                    | Choices::CEASELESSEDGE
                    | Choices::STONEAXE => 1,
                    Choices::FICKLEBEAM => 0,
                    _ => choice.secondaries.as_ref().map(|v| v.len()).unwrap_or(0),
                } };
                // `self.volatileStatus: 'lockedmove'` (Outrage, Thrash, ...) still
                // applies when the hit lands on a Substitute (PS spreadMoveHit
                // `self` step); onStart draws trueDuration, onRestart resets.
                if ti == 0
                    && choice
                        .volatile_status
                        .as_ref()
                        .is_some_and(|v| v.volatile_status == PokemonVolatileStatus::LOCKEDMOVE && v.target == MoveTarget::User)
                {
                    state.side_mut(side).slot_scratch_mut(slot).locked_move_id = Some(choice.move_id);
                    match state.side(side).active()[slot].locked_move {
                        None => {
                            let t = rng.label("volatile duration").random_range(2, 4) as u8;
                            state.side_mut(side).active_slot_mut(slot).locked_move = Some((2, t));
                        }
                        Some((d, t)) => {
                            state.side_mut(side).active_slot_mut(slot).locked_move =
                                if t >= 2 { Some((2, t)) } else { Some((d, t)) };
                        }
                    }
                }
                // Glaive Rush `self.volatileStatus` applies on a Substitute hit too
                // (the target is `null`, not `false`, in PS selfDrops).
                if choice.move_id == Choices::GLAIVERUSH && ti == 0 && state.side(side).team_member(idx).hp > 0 {
                    state.side_mut(side).active_slot_mut(slot).glaive_rush = true;
                }
                for k in 0..rolls {
                    let r = rng.label("secondary").random(100) as f32;
                    // A secondary with `self` (Flame Charge's +1 Speed) still
                    // applies to the user when the hit lands on a Substitute.
                    if !matches!(
                        choice.move_id,
                        Choices::TRIATTACK | Choices::DIRECLAW | Choices::SALTCURE | Choices::SPIRITSHACKLE
                            | Choices::THROATCHOP | Choices::EERIESPELL | Choices::PSYCHICNOISE
                            | Choices::CEASELESSEDGE | Choices::STONEAXE
                    ) {
                        if let Some(sec) = choice.secondaries.as_ref().and_then(|v| v.get(k)) {
                            if sec.target == MoveTarget::User && (sec.chance <= 0.0 || r < sec.chance) {
                                if let Effect::Boost(b) = &sec.effect {
                                    apply_boosts(state, side, idx, b, false);
                                }
                            }
                        }
                    }
                }
                // Stench `onModifyMove` pushes its 10% flinch after the move's own
                // entries, and the rolls run on a Substitute hit too (PS
                // data/abilities.ts stench; the flinch itself lands on nothing).
                // Shield Dust (the target's ability) filters it out first.
                let sub_shield_dust = state.side(target_side).active()[target_slot].pokemon_index.is_some_and(|tidx| {
                    state.side(target_side).team_member(tidx).ability == Abilities::SHIELDDUST
                        && !crate::engine::abilities::ignores_target_ability(state.side(side).team_member(idx), choice, &Abilities::SHIELDDUST)
                });
                let has_flinch = choice
                    .secondaries
                    .as_ref()
                    .is_some_and(|ss| ss.iter().any(|e| matches!(e.effect, Effect::Flinch | Effect::VolatileStatus(PokemonVolatileStatus::FLINCH))));
                if state.side(side).team_member(idx).ability == Abilities::STENCH
                    && choice.category != MoveCategory::Status
                    && !sheer
                    && !sub_shield_dust
                    && !has_flinch
                {
                    let _ = rng.label("secondary").random_chance(10, 100);
                }
                continue;
            }
            real_hit = true;
            if let Some(tpi) = hit_target_pi {
                if !magician_targets.contains(&(hts, tpi)) {
                    magician_targets.push((hts, tpi));
                }
            }
            // Disguise `onDamage` (priority 1): the first damaging hit is absorbed,
            // then `onUpdate` busts the forme and deals 1/8 max HP.
            let damage = {
                let busts = state.side(target_side).slot_pokemon(target_slot).map(|d| {
                    d.id == crate::pokemon::PokemonName::MIMIKYU && d.ability == Abilities::DISGUISE
                }).unwrap_or(false)
                    && (side, slot) != (target_side, target_slot)
                    && choice.category != MoveCategory::Status;
                if busts {
                    // The forme change runs in the next `eachEvent('Update')`
                    // (`each_event_update` after its sort).
                    if !state.pending_disguise.contains(&(target_side, target_slot)) {
                        state.pending_disguise.push((target_side, target_slot));
                    }
                    0
                } else {
                    damage
                }
            };
            // Endure `onDamage` (priority -10): a move that would KO leaves 1 HP.
            let damage = match state.side(target_side).slot_pokemon(target_slot) {
                Some(d) if state.side(target_side).active()[target_slot].endure && damage >= d.hp && d.hp > 0 => d.hp - 1,
                // Sturdy `onDamage` (priority -30): a full-HP holder survives a KO at 1 HP.
                Some(d) if d.ability == Abilities::STURDY && d.hp == d.maxhp && damage >= d.hp && d.hp > 0 => d.hp - 1,
                _ => damage,
            };
            // PS damage dealt is capped by the target's remaining HP (drain,
            // recoil and Counter use the damage actually dealt).
            let damage = match state.side(target_side).slot_pokemon(target_slot) {
                Some(d) => damage.min(d.hp.max(0)),
                None => damage,
            };
            if damage > 0 {
                // PS `runEvent('Damage')`: Damp's `onAnyDamage` (spread attacks: above).
                if !(is_spread && is_attack) {
                    crate::doubles::events::run_event_ties(state, rng, crate::doubles::events::Event::Damage, (target_side, target_slot), Some((side, slot)));
                }
                damaged_by_move(state, target_side, target_slot);
                // PS `gotAttacked`: record the attacker in `attackedBy`
                // (Avalanche's base power check).
                let turn_now = state.turn;
                if let Some(tidx) = state.side(target_side).active()[target_slot].pokemon_index {
                    state
                        .side_mut(target_side)
                        .team_member_mut(tidx)
                        .attacked_by
                        .push((attacker_key(side, idx as u8), turn_now));
                    // PS `gotAttacked`: `attackedBy` entry with damage / slot
                    // (Metal Burst / Comeuppance `getLastDamagedBy`).
                    state.side_mut(target_side).damaged_by[tidx as usize].push((side, slot, damage, turn_now, idx as usize));
                    // Toxic Debris `onDamagingHit`: a physical hit on the
                    // holder adds a Toxic Spikes layer to the attacker's side
                    // (no third layer).
                    if state.side(target_side).team_member(tidx).ability == Abilities::TOXICDEBRIS
                        && choice.category == MoveCategory::Physical
                    {
                        let spikes_side = if target_side == side {
                            side.other()
                        } else {
                            side
                        };
                        if state.side(spikes_side).side_conditions.toxic_spikes < 2 {
                            state
                                .side_mut(spikes_side)
                                .side_conditions
                                .toxic_spikes += 1;
                        }
                    }
                }
            }
            // Anger Point `onHit`: a critical hit maxes Attack.
            if was_crit {
                if let Some(didx) = state.side(target_side).active()[target_slot].pokemon_index {
                    if state.side(target_side).team_member(didx).ability == Abilities::ANGERPOINT
                        && state.side(target_side).team_member(didx).hp > damage
                    {
                        // PS `boost({atk: 12})` -> `statsRaisedThisTurn` when it
                        // applies (sim/battle.ts boost: success and a positive entry).
                        let sd = state.side_mut(target_side);
                        if sd.team_boosts[didx as usize].atk < 6 {
                            sd.team_boosts[didx as usize].atk = 6;
                            sd.slot_scratch_mut(target_slot).stats_raised_this_turn = true;
                        }
                    }
                }
            }
            if damage > 0 {
                // Track the last attacker for Destiny Bond `onAfterFaint`.
                state.side_mut(target_side).slot_scratch_mut(target_slot).last_attacker = Some((side, idx));
                // Read the ability before `apply_damage` clears the slot, to
                // replicate Cloud Nine / Air Lock `onEnd` -> eachEvent.
                let target_ability = state
                    .side(target_side)
                    .slot_pokemon(target_slot)
                    .map(|p| p.ability);
                if apply_damage(state, target_side, target_slot, damage) {
                    if matches!(
                        target_ability,
                        Some(Abilities::CLOUDNINE) | Some(Abilities::AIRLOCK)
                    ) {
                        weather_suppress_fainted = true;
                        weather_suppress_holder.get_or_insert((target_side, target_slot));
                    }
                // Target fainted. `onSourceAfterFaint` (Moxie, Chilling / Grim
                // Neigh) is queued for the batch's `resolve_faints`, which boosts
                // the attacker once by the batch size (or not at all when the
                // batch ends the battle). PS also fires it for a fainted ally.
                // The attacker's ability is read when the batch resolves (a
                // Receiver may copy Moxie from an ally fainting in the same batch).
                if (target_side, target_slot) != (side, slot) {
                    let ab = state.side(side).team_member(idx).ability;
                    state.pending_after_faint = Some((side, slot, idx, ab));
                }
                }
            }
            total_damage = total_damage.saturating_add(damage);
            hit_damage = Some(damage);
            after_hit_targets.push((target_side, target_slot));
            hit_landed = true;
            // Berserk `onAfterMoveSecondary`: remember the HP before this move's
            // first hit on the holder. Allies count too (a spread move such as
            // Earthquake): PS only skips `source === target`.
            if damage > 0 && (target_side, target_slot) != (side, slot) {
                if let Some(d) = state.side(target_side).slot_pokemon(target_slot) {
                    if d.ability == Abilities::BERSERK
                        && !berserk_before.iter().any(|&(s2, l2, _)| s2 == target_side && l2 == target_slot)
                    {
                        berserk_before.push((target_side, target_slot, d.hp + damage));
                    }
                }
            }
            // Emergency Exit / Wimp Out: remember the HP before the move's first
            // damaging hit on each target; the check runs once the move is over
            // (PS `tryMoveHit`, after every hit and DamagingHit, so a Skill Swap
            // during the hit can hand the ability over first).
            if damage > 0 && (target_side, target_slot) != (side, slot) {
                if let Some(d) = state.side(target_side).slot_pokemon(target_slot) {
                    if !ee_before.iter().any(|&(s2, l2, _)| s2 == target_side && l2 == target_slot) {
                        ee_before.push((target_side, target_slot, d.hp + damage));
                    }
                }
            }
            // Counter / Mirror Coat condition `onDamagingHit`.
            if target_side != side && damage > 0 {
                // Updates an existing Counter / Mirror Coat volatile in place
                // (its `vol_mask` bit does not change).
                let t = state.side_mut(target_side).slot_scratch_mut(target_slot);
                let slot_ref = Some((side, slot));
                match choice.category {
                    MoveCategory::Physical => {
                        if let Some(c) = t.counter_vol.as_mut() {
                            *c = (slot_ref, damage.saturating_mul(2));
                        }
                    }
                    MoveCategory::Special => {
                        if let Some(c) = t.mirror_coat_vol.as_mut() {
                            *c = (slot_ref, damage.saturating_mul(2));
                        }
                    }
                    _ => {}
                }
            }

            // Drain: heal the attacker for a fraction of the damage dealt.
            if let Some(frac) = choice.drain {
                let heal = ((damage as f32) * frac).round() as i16;
                // Liquid Ooze (on the drained foe) `onSourceTryHeal`: the drainer
                // takes the amount instead, even at full HP and under Heal Block
                // (PS data/abilities.ts liquidooze; the Heal event still runs).
                // The holder is read from the hit's target index: a target that
                // this hit KO'd has already left its slot (PS keeps its ability).
                let ooze = match hit_target_pi {
                    Some(t) => state.side(target_side).team_member(t).ability == Abilities::LIQUIDOOZE,
                    None => false,
                };
                if ooze {
                    // The Ooze damage is an Ability effect: Magic Guard's
                    // `onDamage` returns false for it (PS data/abilities.ts magicguard).
                    if heal > 0 {
                        apply_indirect_damage(state, side, slot, heal);
                    }
                } else if heal > 0 && state.side(side).active()[slot].heal_block == 0 {
                    let s = state.side_mut(side);
                    if let Some(p) = s.active()[slot].pokemon_index {
                        let pm = s.team_member_mut(p);
                        pm.hp = (pm.hp + heal).min(pm.maxhp);
                    }
                }
            }

        }

        // `selfDrops` -> `moveHit(source, source, move.self)`: a recharge move's
        // `self.volatileStatus: 'mustrecharge'` is `addVolatile`d here, so its
        // TryAddVolatile handlers are sorted now (the flag itself is set after
        // the move). PS sim/battle-actions.ts selfDrops, sim/pokemon.ts addVolatile.
        if ti == 0 && choice.flags.recharge && !state.side(side).active()[slot].recharging {
            try_add_volatile_ties(state, rng, side, slot);
        }
        // Self boosts (Swords Dance, ...). PS `selfDrops` consumes the
        // `random(100)` roll even when there is no chance gate.
        if let (true, Some(b)) = (ti == 0, &choice.self_boosts) {
            let roll = rng.label("self boost").random(100) as u32;
            let apply = if choice.self_boost_chance > 0.0 {
                roll < choice.self_boost_chance as u32
            } else {
                true
            };
            if apply {
                apply_boosts(state, side, idx, b, false);
            }
        }

        // Target boosts (Scary Face, Nasty Plot on foe, etc.) — no draw.
        // Upstream also stores PS `self.boosts` drops (Close Combat, ...) in
        // `boost`; those are applied (with their roll) via `self_boosts`.
        if let Some(b) = choice.boost.as_ref().filter(|_| choice.self_boosts.is_none()) {
            // The upstream table has no evasion stages: Double Team +1,
            // Minimize +2, Sweet Scent -2 (data/moves.ts `boosts`).
            let mut b = b.clone();
            match choice.move_id {
                Choices::DOUBLETEAM => b.boosts.evasion = 1,
                Choices::MINIMIZE => b.boosts.evasion = 2,
                Choices::SWEETSCENT => b.boosts.evasion = -2,
                // PS growth.onModifyMove: +2 Atk / SpA in harsh sunlight.
                Choices::GROWTH
                    if matches!(effective_weather(state), Weather::SUN | Weather::HARSHSUN) =>
                {
                    b.boosts.attack = 2;
                    b.boosts.special_attack = 2;
                }
                _ => {}
            }
            let b = &b;
            // Multi-hit moves (Scale Shot) apply a user stat change once, after
            // the last hit (`selfBoost`).
            // Clanging Scales is PS `selfBoost` too: applied once after the whole move.
            let defer_user_boost = (choice.multihit != 0 || choice.move_id == Choices::CLANGINGSCALES)
                && is_attack
                && b.target == MoveTarget::User;
            // Curse: a Ghost user pays HP instead of changing stats (`onTryHit`).
            let ghost_curse = choice.move_id == Choices::CURSE && {
                let t = state.side(side).team_member(idx).types;
                t.0 == PokemonType::GHOST
                    || t.1 == PokemonType::GHOST
                    || state.side(side).active()[slot].added_type == PokemonType::GHOST
            };
            // Ally-targeted moves (Coaching, Aromatic Mist, ...) keep upstream's
            // singles-era `User` boost target; in doubles they hit the ally.
            let (ts, tslot) = if is_ally {
                (side, ally_slot)
            } else if b.target == MoveTarget::User || is_self {
                (side, slot)
            } else if b.target == MoveTarget::Ally {
                (side, ally_slot)
            } else {
                (target_side, target_slot)
            };
            if move_ty == Tgt::Allies {
                // Howl & co. (`target: 'allies'`): the user and its ally, in
                // slot order.
                for sl in 0..2 {
                    if switched_in_alive(state, side, sl) {
                        // Soundproof `onTryHit` on a sound move: the ally is
                        // immune (`-immune [from] ability: Soundproof`), no boost.
                        // PS sim/battle-actions.ts tryMoveHit, data/abilities.ts soundproof.
                        if sl != slot
                            && choice.flags.sound
                            && state.side(side).slot_pokemon(sl).is_some_and(|p| p.ability == Abilities::SOUNDPROOF)
                        {
                            continue;
                        }
                        // Good as Gold `onTryHit` (status move, target != source): the
                        // ally is immune (`-immune [from] ability: Good as Gold`), no boost.
                        // PS data/abilities.ts goodasgold, sim/battle-actions.ts tryMoveHit.
                        if sl != slot
                            && choice.category == MoveCategory::Status
                            && state.side(side).slot_pokemon(sl).is_some_and(|p| p.ability == Abilities::GOODASGOLD)
                            && !crate::engine::abilities::ignores_target_ability(
                                state.side(side).team_member(idx),
                                choice,
                                &Abilities::GOODASGOLD,
                            )
                        {
                            continue;
                        }
                        // A semi-invulnerable ally misses (PS `hitStepInvulnerabilityEvent`
                        // for each target): no boost for it.
                        // No Guard on the user or the ally (`onAnyInvulnerability`
                        // returns 0) lets it land. PS data/abilities.ts noguard.
                        let no_guard = state.side(side).team_member(idx).ability == Abilities::NOGUARD
                            || state.side(side).slot_pokemon(sl).is_some_and(|p| p.ability == Abilities::NOGUARD);
                        if sl != slot && ally_semi_invulnerable(state, side, sl) && !no_guard {
                            continue;
                        }
                        if let Some(tidx) = state.side(side).active()[sl].pokemon_index {
                            apply_boosts(state, side, tidx, &b.boosts, false);
                        }
                    }
                }
            } else if ghost_curse {
                // no stat change
            } else if defer_user_boost {
                // Multi-hit and Clanging Scales (PS `selfBoost`): applied after
                // the hit loop's faints are resolved (see below).
            } else if let Some(tidx) = state.side(ts).active()[tslot].pokemon_index {
                let before = state.side(ts).team_boosts[tidx as usize].clone();
                let boosts = if (ts, tslot) != (side, slot) { mirror_armor_bounce(state, side, slot, ts, tidx, &b.boosts) } else { b.boosts.clone() };
                if choice.move_id == Choices::PARTINGSHOT && state.side(ts).team_member(tidx).ability == Abilities::MIRRORARMOR {
                    parting_shot_held = true;
                }
                let landed = apply_boosts_from(state, ts, tidx, &boosts, ts != side, (ts, tslot) != (side, slot));
                let after = &state.side(ts).team_boosts[tidx as usize];
                if (before.atk, before.def, before.spa, before.spd, before.spe, before.acc, before.eva)
                    != (after.atk, after.def, after.spa, after.spd, after.spe, after.acc, after.eva)
                {
                    drop_landed = true;
                } else if !landed && choice.category == MoveCategory::Status
                    && choice.heal.is_none()
                    && choice.status.is_none()
                    // Curse here is the non-Ghost self-boost (its volatile is Ghost-only).
                    && (choice.volatile_status.is_none()
                        || choice.move_id == Choices::CURSE
                        // Confusion (Swagger, Flatter, ...): `addVolatile` returns false
                        // while the target is already confused (no `onRestart`).
                        || (choice.volatile_status.as_ref().is_some_and(|v| v.volatile_status == PokemonVolatileStatus::CONFUSION)
                            && state.side(ts).active()[tslot].confusion > 0)
                        // Minimize: `addVolatile('minimize')` returns null (its
                        // `onRestart`) while the volatile is already present.
                        || (choice.move_id == Choices::MINIMIZE
                            && state.side(ts).active()[tslot].minimize))
                    && choice.side_condition.is_none()
                    // Defog's table boost is empty: its `onHit` (-1 evasion) decides.
                    && choice.move_id != Choices::DEFOG
                {
                    // A boost-only status move that changed nothing (stat already
                    // at +/-6) fails: the hit loop ends without an `eachEvent`.
                    // PS `spreadMoveHit`: for a spread move this is one target's
                    // `false`; the move fails only when every target gave false.
                    if is_spread {
                        spread_boost_noop = true;
                    } else {
                        move_failed = true;
                    }
                }
            }
        }

        // Healing (Moonlight, Synthesis, etc.) — no draw.
        // Life Dew: `target: allies`, `heal: [1, 4]` for the user and its ally
        // (maxhp / 4, rounded); fails only if nobody could be healed.
        // PS data/moves.ts lifedew, sim/battle.ts heal.
        if choice.move_id == Choices::LIFEDEW && ti == 0 {
            let mut healed = false;
            // Good as Gold `onTryHit` (status move, target != source) makes an
            // ally immune: no heal for it (PS prints `-immune` and skips it).
            let user_breaks_gold = crate::engine::abilities::ignores_target_ability(
                state.side(side).team_member(idx),
                choice,
                &Abilities::GOODASGOLD,
            );
            for sl in [slot, ally_slot] {
                // A semi-invulnerable ally misses (PS `hitStepInvulnerabilityEvent`
                // prints `-miss` and skips it: no heal for it, no draw).
                // PS data/conditions.ts dive / fly / dig `onInvulnerability`.
                let ally_misses = sl != slot && semi_invulnerable_blocks(state, side, idx, choice.move_id, side, sl);
                // Heal Block `onTryHeal` returns false for this heal: that target
                // is skipped (PS battle-actions.ts 'heal interrupted'). PS
                // data/moves.ts healblock.condition.onTryHeal.
                if state.side(side).active()[sl].heal_block > 0 {
                    continue;
                }
                if let Some(tidx) = state.side(side).active()[sl].pokemon_index {
                    let pm = state.side_mut(side).team_member_mut(tidx);
                    if sl != slot && pm.hp > 0 && pm.ability == Abilities::GOODASGOLD && !user_breaks_gold {
                        continue;
                    }
                    if ally_misses {
                        continue;
                    }
                    // PS data/abilities.ts waterabsorb / dryskin `onTryHit`: an ally
                    // with them takes the Water move as a heal of `maxhp / 4`
                    // (truncated) and the move does not hit it, so Life Dew's own
                    // heal (rounded) does not apply to that ally. Not a success.
                    // TODO(PS parity): Mold Breaker on the user.
                    if sl != slot && pm.hp > 0 && matches!(pm.ability, Abilities::WATERABSORB | Abilities::DRYSKIN) {
                        if pm.hp < pm.maxhp {
                            pm.hp = (pm.hp + (pm.maxhp / 4).max(1)).min(pm.maxhp);
                        }
                        continue;
                    }
                    if pm.hp > 0 && pm.hp < pm.maxhp {
                        pm.hp = (pm.hp + ((pm.maxhp as f32 / 4.0).round() as i16).max(1)).min(pm.maxhp);
                        healed = true;
                    }
                }
            }
            if !healed {
                move_failed = true;
            }
        }
        // Heal Bell (`target: allyTeam`, onHit once with the user as target): every
        // party member but the user is skipped when it has Soundproof or Good as
        // Gold (`-immune`); the rest, the user included, are cured. Fails when
        // nobody was cured. Bench members count too. PS data/moves.ts healbell,
        // sim/pokemon.ts cureStatus.
        // TODO(PS parity): Mold Breaker / Gastro Acid suppression of the ally's ability.
        if choice.move_id == Choices::HEALBELL && ti == 0 {
            let mut cured = false;
            for t in crate::state::pokemon_index_iter() {
                let (hp, status, ability) = {
                    let p = state.side(side).team_member(t);
                    (p.hp, p.status, p.ability)
                };
                if hp <= 0 || status == PokemonStatus::NONE {
                    continue;
                }
                if t != idx
                    && matches!(ability, Abilities::SOUNDPROOF | Abilities::GOODASGOLD)
                {
                    continue;
                }
                state.side_mut(side).team_member_mut(t).status = PokemonStatus::NONE;
                cured = true;
            }
            if !cured {
                move_failed = true;
            }
        }

        if let Some(h) = choice.heal.as_ref().filter(|_| !matches!(choice.move_id, Choices::LIFEDEW | Choices::STEELBEAM)) {
            // Milk Drink (champions `target: adjacentAllyOrSelf`): heals the
            // chosen target, the user or its ally. PS data/mods/champions/moves.ts.
            let (ts, tslot) = if choice.move_id == Choices::MILKDRINK {
                match resolved {
                    Some((rs, rl)) if rs == side => (rs, rl),
                    _ => (side, slot),
                }
            } else if h.target == MoveTarget::User || is_self {
                (side, slot)
            } else if h.target == MoveTarget::Ally {
                (side, ally_slot)
            } else {
                (target_side, target_slot)
            };
            // PS `pokemon.effectiveWeather()`: none under Cloud Nine / Air Lock.
            let weather = effective_weather(state);
            if let Some(tidx) = state.side(ts).active()[tslot].pokemon_index {
                let s = state.side_mut(ts);
                let pm = s.team_member_mut(tidx);
                // Moonlight family (data/moves.ts `onHit`): 0.5, sun 0.667,
                // rain / sand / hail / snow 0.25, via `modify(maxhp, factor)`.
                let amount = if matches!(
                    choice.move_id,
                    Choices::SYNTHESIS | Choices::MORNINGSUN | Choices::MOONLIGHT
                ) {
                    let factor: f64 = match weather {
                        Weather::SUN | Weather::HARSHSUN => 0.667,
                        Weather::RAIN | Weather::HEAVYRAIN | Weather::SAND | Weather::HAIL | Weather::SNOW => 0.25,
                        _ => 0.5,
                    };
                    let m = (factor * 4096.0).trunc() as i64;
                    ((pm.maxhp as i64 * m + 2047) / 4096) as i16
                } else {
                    (pm.maxhp as f32 * h.amount).round() as i16
                };
                if amount < 0 {
                    // Memento style self-KO (-1.0) or Steel Beam / Mind Blown
                    // recoil (`mindBlownRecoil`: round(maxhp / 2)).
                    let hp = if h.amount <= -1.0 { pm.hp } else { -amount };
                    let _ = pm;
                    apply_damage(state, ts, tslot, hp);
                } else {
                    // PS `heal` returns false (the move fails) at full HP.
                    if pm.hp >= pm.maxhp {
                        move_failed = true;
                        // Moonlight family `onHit` returns NOT_FAIL on a failed heal.
                        heal_fail_event = matches!(
                            choice.move_id,
                            Choices::SYNTHESIS | Choices::MORNINGSUN | Choices::MOONLIGHT
                        );
                        // PS sim/battle-actions.ts hitStepMoveHitLoop: NOT_FAIL leaves
                        // `moveThisTurnResult` null, not false (Stomping Tantrum / Temper Flare).
                        if heal_fail_event {
                            not_fail_result = true;
                        }
                    }
                    pm.hp = (pm.hp + amount).min(pm.maxhp);
                }
            }
        }

        // PS `spreadMoveHit` returns a 0 damage entry (NOT_FAIL), so the hit
        // runs `eachEvent('Update')` twice: inside `spreadMoveHit` and after it
        // in `hitStepMoveHitLoop` (sim/battle-actions.ts).
        if heal_fail_event {
            each_event_update(state, rng);
            each_event_update(state, rng);
            heal_fail_event = false;
        }

        // Reflect Type `onHit` (PS data/moves.ts reflecttype): the user
        // becomes the target's apparent types (type-change included).
        // `setType` refuses on Arceus / Silvally (`species.num 493 / 773`).
        // TODO(PS parity): Tera, substitutes in front of the target.
        if choice.move_id == Choices::REFLECTTYPE && !is_self {
            let user_locked = matches!(
                state.side(side).team_member(idx).id,
                crate::pokemon::PokemonName::ARCEUS
                    | crate::pokemon::PokemonName::ARCEUSBUG
                    | crate::pokemon::PokemonName::ARCEUSDARK
                    | crate::pokemon::PokemonName::ARCEUSDRAGON
                    | crate::pokemon::PokemonName::ARCEUSELECTRIC
                    | crate::pokemon::PokemonName::ARCEUSFAIRY
                    | crate::pokemon::PokemonName::ARCEUSFIGHTING
                    | crate::pokemon::PokemonName::ARCEUSFIRE
                    | crate::pokemon::PokemonName::ARCEUSFLYING
                    | crate::pokemon::PokemonName::ARCEUSGHOST
                    | crate::pokemon::PokemonName::ARCEUSGRASS
                    | crate::pokemon::PokemonName::ARCEUSGROUND
                    | crate::pokemon::PokemonName::ARCEUSICE
                    | crate::pokemon::PokemonName::ARCEUSPOISON
                    | crate::pokemon::PokemonName::ARCEUSPSYCHIC
                    | crate::pokemon::PokemonName::ARCEUSROCK
                    | crate::pokemon::PokemonName::ARCEUSSTEEL
                    | crate::pokemon::PokemonName::ARCEUSWATER
                    | crate::pokemon::PokemonName::SILVALLY
                    | crate::pokemon::PokemonName::SILVALLYBUG
                    | crate::pokemon::PokemonName::SILVALLYDARK
                    | crate::pokemon::PokemonName::SILVALLYDRAGON
                    | crate::pokemon::PokemonName::SILVALLYELECTRIC
                    | crate::pokemon::PokemonName::SILVALLYFAIRY
                    | crate::pokemon::PokemonName::SILVALLYFIGHTING
                    | crate::pokemon::PokemonName::SILVALLYFIRE
                    | crate::pokemon::PokemonName::SILVALLYFLYING
                    | crate::pokemon::PokemonName::SILVALLYGHOST
                    | crate::pokemon::PokemonName::SILVALLYGRASS
                    | crate::pokemon::PokemonName::SILVALLYGROUND
                    | crate::pokemon::PokemonName::SILVALLYICE
                    | crate::pokemon::PokemonName::SILVALLYPOISON
                    | crate::pokemon::PokemonName::SILVALLYPSYCHIC
                    | crate::pokemon::PokemonName::SILVALLYROCK
                    | crate::pokemon::PokemonName::SILVALLYSTEEL
                    | crate::pokemon::PokemonName::SILVALLYWATER
            );
            if user_locked {
                move_failed = true;
            } else {
                // `target.getTypes(true)`: the current types. Libero / Soak /
                // Magic Powder write them to the team member and keep the
                // original in `type_change` (restore data), so read the member.
                let (t0, t1) = state
                    .side(target_side)
                    .slot_pokemon(target_slot)
                    .map(|p| p.types)
                    .unwrap_or((PokemonType::TYPELESS, PokemonType::TYPELESS));
                // PS `getTypes(true).filter(type => type !== '???')`: the engine
                // stores both `???` (Burn Up, Double Shock) and "no second type"
                // as TYPELESS. No type left: Normal if the target has an added
                // type, else the move fails. `source.addedType = target.addedType`.
                let kept: Vec<PokemonType> = [t0, t1].iter().copied().filter(|t| *t != PokemonType::TYPELESS).collect();
                let target_added = state.side(target_side).active()[target_slot].added_type;
                let new_types = match kept.as_slice() {
                    [] if target_added != PokemonType::TYPELESS => Some((PokemonType::NORMAL, PokemonType::TYPELESS)),
                    [] => None,
                    [a] => Some((*a, PokemonType::TYPELESS)),
                    [a, b, ..] => Some((*a, *b)),
                };
                if let Some(new_types) = new_types {
                    let orig = state.side(side).team_member(idx).types;
                    if orig != new_types {
                        let s = state.side_mut(side);
                        if s.active()[slot].type_change.is_none() {
                            s.slot_scratch_mut(slot).type_change = Some(orig);
                        }
                        s.team_member_mut(idx).types = new_types;
                    }
                    state.side_mut(side).slot_scratch_mut(slot).added_type = target_added;
                } else {
                    move_failed = true;
                }
            }
        }

        // PS `moveHit` forceSwitch step: a Status forceSwitch move (Roar,
        // Whirlwind) gives `hitResult = canSwitch(target.side)`. With no living
        // reserve on the target's side the move does nothing (`-fail`), so the
        // hit loop runs no `eachEvent('Update')`. PS sim/battle-actions.ts moveHit.
        if choice.force_switch
            && choice.category == MoveCategory::Status
            && has_target
            && !is_self
            && !is_field
            && possible_switches(state, target_side).is_empty()
        {
            move_failed = true;
            continue;
        }

        // Aroma Veil `onAllyTryAddVolatile`: the holder and its ally can't gain
        // Attract / Disable / Encore / Heal Block / Taunt / Torment from others
        // (after the accuracy roll). PS data/abilities.ts aromaveil.
        // TODO(PS parity): Mold Breaker.
        let veiled = match choice.move_id {
            Choices::ATTRACT => Some(AromaVeiled::Attract),
            Choices::DISABLE => Some(AromaVeiled::Disable),
            Choices::ENCORE => Some(AromaVeiled::Encore),
            Choices::HEALBLOCK => Some(AromaVeiled::HealBlock),
            Choices::TAUNT => Some(AromaVeiled::Taunt),
            Choices::TORMENT => Some(AromaVeiled::Torment),
            _ => None,
        };
        if let Some(v) = veiled.filter(|_| ti == 0 && (target_side, target_slot) != (side, slot))
            .filter(|&v| aroma_veil_blocks(state, target_side, v, Some((side, idx, choice))))
        {
            let _ = v;
            move_failed = true;
            continue;
        }

        // Status moves (Thunder Wave, Will-O-Wisp, Toxic, Hypnosis, ...).
        if let Some(st) = &choice.status {
            let (ts, tslot) = if st.target == MoveTarget::User || is_self {
                (side, slot)
            } else if st.target == MoveTarget::Ally {
                (side, ally_slot)
            } else {
                (target_side, target_slot)
            };
            if let Some(tidx) = state.side(ts).active()[tslot].pokemon_index {
                // PS `trySetStatus` -> `setStatus(this.status || status)`: a target
                // that already has a status re-applies its own one, which fails
                // (`-fail` / `[still]`), whatever the move's status. No draw.
                let already = state.side(ts).team_member(tidx).status != PokemonStatus::NONE;
                // Already in a status, or immune: the move fails.
                // PS `substitute.onTryPrimaryHit`: `bypasssub` moves (Sing, and the
                // list in `bypasses_substitute`) skip the Substitute.
                let sub_blocks = !bypasses_substitute(choice);
                if !already {
                    set_status_event_tie(state, rng, ts, tslot, st.status, ts != side, sub_blocks);
                }
                if already || status_immune_ex(state, ts, tslot, st.status, ts != side, side, slot, sub_blocks, true, false) {
                    move_failed = true;
                } else {
                    set_status(state, rng, ts, tidx, st.status);
                    // Synchronize copies non-sleep/freeze status back to the
                    // move user.
                    maybe_synchronize(state, ts, tidx, side, idx, st.status);
                }
            }
        }

        // Volatile status (Yawn, Leech Seed, Substitute, ...). No draw.
        if let Some(v) = &choice.volatile_status {
            let (ts, tslot) = if v.target == MoveTarget::User || is_self {
                (side, slot)
            } else if v.target == MoveTarget::Ally {
                (side, ally_slot)
            } else {
                (target_side, target_slot)
            };
            match v.volatile_status {
                PokemonVolatileStatus::PROTECT
                | PokemonVolatileStatus::KINGSSHIELD
                | PokemonVolatileStatus::SPIKYSHIELD
                | PokemonVolatileStatus::BANEFULBUNKER
                | PokemonVolatileStatus::SILKTRAP
                | PokemonVolatileStatus::BURNINGBULWARK => {
                    // `addVolatile(protect)` -> `TryAddVolatile` (no draw if already up).
                    if !state.side(ts).active()[tslot].protected {
                        try_add_volatile_ties(state, rng, ts, tslot);
                    }
                    state.side_mut(ts).active_slot_mut(tslot).protected = true;
                    state.side_mut(ts).active_slot_mut(tslot).protect_kind = match v.volatile_status {
                        PokemonVolatileStatus::KINGSSHIELD => 1,
                        PokemonVolatileStatus::SPIKYSHIELD => 2,
                        PokemonVolatileStatus::BANEFULBUNKER => 3,
                        PokemonVolatileStatus::SILKTRAP => 4,
                        PokemonVolatileStatus::BURNINGBULWARK => 5,
                        _ => 0,
                    };

                    // onHit: `addVolatile('stall')` (start: counter 3; restart: x3).
                    // A restart (counter up) runs `Restart`, not `TryAddVolatile`.
                    if state.side(ts).active()[tslot].stall.is_none() {
                        try_add_volatile_ties(state, rng, ts, tslot);
                    }
                    let a = state.side_mut(ts).active_slot_mut(tslot);
                    a.stall = match a.stall {
                        None => Some((3, 2)),
                        Some((c, _)) => Some((if c < 729 { c * 3 } else { c }, 2)),
                    };
                }
                PokemonVolatileStatus::SMACKDOWN => {
                    // PS data/moves.ts smackdown.condition.onStart / onRestart.
                    // A Fly / Bounce charge on the holder ends first: `removeVolatile`
                    // of fly / bounce + twoturnmove, `queue.cancelMove`, and it
                    // counts as applying (both onStart and onRestart).
                    // TODO(PS parity): Telekinesis is not modelled.
                    if let Some(tidx) = state.side(ts).active()[tslot].pokemon_index {
                        let charge_flight = matches!(
                            state.side(ts).active()[tslot].charging,
                            Some((Choices::FLY | Choices::BOUNCE, _))
                        );
                        if charge_flight {
                            let sd = state.side_mut(ts);
                            sd.active_slot_mut(tslot).charging = None;
                            sd.active_slot_mut(tslot).charge_aborted = false;
                            state.queue_cancel.push((ts, tslot));
                        }
                        let sd = state.side(ts);
                        let p = sd.team_member(tidx);
                        let a = &sd.active()[tslot];
                        if a.smackdown {
                            // onRestart: no further effect.
                        } else {
                            let mut applies = p.has_type(&PokemonType::FLYING)
                                || matches!(p.ability, Abilities::LEVITATE | Abilities::EELEVATE)
                                || charge_flight;
                            if p.item == crate::engine::items::Items::IRONBALL
                                || a.ingrain
                                || state.gravity.active
                            {
                                applies = charge_flight;
                            }
                            if a.magnet_rise > 0 {
                                applies = true;
                            }
                            if applies {
                                let sd = state.side_mut(ts);
                                sd.active_slot_mut(tslot).magnet_rise = 0;
                                sd.active_slot_mut(tslot).smackdown = true;
                            }
                        }
                    }
                }
                PokemonVolatileStatus::MAGNETRISE => {
                    // PS data/moves.ts magnetrise.onTry: fails with Ingrain or
                    // Gravity (`cant`). TODO(PS parity): Smack Down.
                    // PS `addVolatile`: an active volatile (no `onRestart`) makes it
                    // return false, so the status move fails with no `eachEvent`.
                    if state.side(ts).active()[tslot].ingrain || state.gravity.active || state.side(ts).active()[tslot].magnet_rise > 0 {
                        move_failed = true;
                    } else {
                        try_add_volatile_ties(state, rng, ts, tslot);
                        state.side_mut(ts).active_slot_mut(tslot).magnet_rise = 5;
                    }
                }
                PokemonVolatileStatus::CONFUSION => {
                    // Safeguard `onTryAddVolatile` (PS data/moves.ts safeguard): `-activate`
                    // and the add returns null, so a status move with no boost "did nothing"
                    // (no hit-loop eachEvent), as with Misty Terrain below.
                    if safeguard_blocks_confusion(state, side, slot, idx, ts, tslot) {
                        if choice.category == MoveCategory::Status && choice.boost.is_none() {
                            if is_spread {
                                spread_volatile_fail = true;
                            } else {
                                move_failed = true;
                            }
                        }
                    } else {
                        // `addVolatile` fails (already confused, Own Tempo): a status
                        // move's `-fail` ends the hit loop with no `eachEvent`.
                        let own_tempo = state.side(ts).slot_pokemon(tslot).map(|p| p.ability == Abilities::OWNTEMPO).unwrap_or(false);
                        // Misty Terrain `onTryAddVolatile` (PS data/conditions.ts) also
                        // makes the add fail: `-activate` and "did nothing" (no eachEvent).
                        let misty_blocked = misty_terrain_blocks_confusion(state, ts, tslot);
                        // Swagger / Flatter keep their boost, so they do not fail.
                        let add_fails = choice.category == MoveCategory::Status
                            && choice.boost.is_none()
                            && (own_tempo || misty_blocked || state.side(ts).active()[tslot].confusion > 0);
                        if is_spread && choice.category == MoveCategory::Status && choice.boost.is_none() {
                            if add_fails {
                                spread_volatile_fail = true;
                            } else {
                                spread_volatile_ok = true;
                            }
                        } else if add_fails {
                            move_failed = true;
                        }
                        try_confuse(state, rng, ts, tslot)
                    }
                }
                PokemonVolatileStatus::SUBSTITUTE => {
                    // PS substitute `onTryHit` (fails if already up or hp <=
                    // maxhp/4), `onHit` (directDamage maxhp/4), `onStart`
                    // (sub hp = floor(maxhp/4); ends partial trapping).
                    let (hp, maxhp) = state
                        .side(ts)
                        .slot_pokemon(tslot)
                        .map(|p| (p.hp, p.maxhp))
                        .unwrap_or((0, 1));
                    if state.side(ts).active()[tslot].substitute_health > 0
                        || (hp as i32) * 4 <= maxhp as i32
                        || maxhp == 1
                    {
                        move_failed = true;
                    } else if let Some(tidx) = state.side(ts).active()[tslot].pokemon_index {
                        // `addVolatile('substitute')` (before `onHit`'s HP cost).
                        try_add_volatile_ties(state, rng, ts, tslot);
                        let cost = (maxhp / 4).max(1);
                        // PS substitute `onHit` directDamage: the cost is a damage
                        // number for the holder, so `spreadMoveHit`'s EmergencyExit
                        // (champions scripts.ts, `targetHPBeforeDamage` = hp + cost)
                        // runs for the user too.
                        if !ee_before.iter().any(|&(s2, l2, _)| s2 == ts && l2 == tslot) {
                            ee_before.push((ts, tslot, hp + cost));
                        }
                        state.side_mut(ts).team_member_mut(tidx).hp -= cost;
                        let a = state.side_mut(ts).active_slot_mut(tslot);
                        a.substitute_health = maxhp / 4;
                        a.partial_trap = None;
                    }
                }
                PokemonVolatileStatus::FOCUSENERGY => {
                    // focusenergy.onStart: fails if the target has Dragon Cheer.
                    // A failed `addVolatile` adds nothing (PS data/moves.ts
                    // focusenergy.condition.onStart).
                    // `TryAddVolatile` runs before `onStart` (a new volatile only).
                    if !state.side(ts).active()[tslot].focus_energy {
                        try_add_volatile_ties(state, rng, ts, tslot);
                    }
                    if state.side(ts).active()[tslot].focus_energy
                        || state.side(ts).active()[tslot].dragon_cheer.is_some()
                    {
                        move_failed = true;
                    } else {
                        state.side_mut(ts).active_slot_mut(tslot).focus_energy = true;
                    }
                }
                PokemonVolatileStatus::INGRAIN => {
                    // `addVolatile` fails if already present.
                    if state.side(ts).active()[tslot].ingrain {
                        move_failed = true;
                    } else {
                        try_add_volatile_ties(state, rng, ts, tslot);
                    }
                    state.side_mut(ts).active_slot_mut(tslot).ingrain = true;
                }
                PokemonVolatileStatus::AQUARING => {
                    // `addVolatile` fails if already present.
                    if state.side(ts).active()[tslot].aqua_ring {
                        move_failed = true;
                    } else {
                        try_add_volatile_ties(state, rng, ts, tslot);
                    }
                    state.side_mut(ts).active_slot_mut(tslot).aqua_ring = true;
                }
                PokemonVolatileStatus::FOLLOWME | PokemonVolatileStatus::RAGEPOWDER => {
                    // `addVolatile('followme')` (no `onRestart`).
                    if state.side(ts).active()[tslot].follow_me == 0 {
                        try_add_volatile_ties(state, rng, ts, tslot);
                    }
                    state.side_mut(ts).active_slot_mut(tslot).follow_me =
                        if v.volatile_status == PokemonVolatileStatus::RAGEPOWDER { 2 } else { 1 };
                }
                PokemonVolatileStatus::ENDURE => {
                    // `addVolatile(endure)` -> `TryAddVolatile`: Insomnia and a Flower
                    // Veil ally tie at subOrder 7 and draw (Worry Seed makes Insomnia).
                    if !state.side(ts).active()[tslot].endure {
                        try_add_volatile_ties(state, rng, ts, tslot);
                    }
                    state.side_mut(ts).active_slot_mut(tslot).endure = true;

                    // onHit: `addVolatile('stall')` (start: counter 3; restart: x3).
                    if state.side(ts).active()[tslot].stall.is_none() {
                        try_add_volatile_ties(state, rng, ts, tslot);
                    }
                    let a = state.side_mut(ts).active_slot_mut(tslot);
                    a.stall = match a.stall {
                        None => Some((3, 2)),
                        Some((c, _)) => Some((if c < 729 { c * 3 } else { c }, 2)),
                    };
                }
                PokemonVolatileStatus::LOCKEDMOVE => {
                    // PS `addVolatile` returns false for a fainted pokemon (`!this.hp`),
                    // so a user that fainted this move (Destiny Bond) takes no lock
                    // and draws no duration (sim/pokemon.ts addVolatile). A Destiny
                    // Bond KO is resolved only after the move, so the user is still
                    // alive here (`db_alive_pending`, with its lock at the KO); its
                    // slot is already empty, so only the draw and no state change.
                    let alive_idx = slot_index_or_unresolved(state, ts, tslot);
                    let db_lock = alive_idx.and_then(|ti_idx| {
                        state.db_alive_pending.iter().find(|e| e.0 == ts && e.1 == ti_idx).map(|e| e.2)
                    });
                    let alive = match alive_idx {
                        Some(ti_idx) => state.side(ts).team_member(ti_idx).hp > 0 || db_lock.is_some(),
                        None => false,
                    };
                    if alive {
                        let in_slot = state.side(ts).active()[tslot].pokemon_index.is_some();
                        if in_slot && ts == side && tslot == slot {
                            state.side_mut(ts).slot_scratch_mut(tslot).locked_move_id = Some(choice.move_id);
                        }
                        // PS lockedmove `onStart`: trueDuration = random(2, 4);
                        // `onRestart`: duration back to 2 while trueDuration >= 2.
                        let current = if in_slot { state.side(ts).active()[tslot].locked_move } else { db_lock.flatten() };
                        let next = match current {
                            None => Some((2, rng.label("volatile duration").random_range(2, 4) as u8)),
                            Some((d, t)) => {
                                if t >= 2 { Some((2, t)) } else { Some((d, t)) }
                            }
                        };
                        if in_slot {
                            state.side_mut(ts).active_slot_mut(tslot).locked_move = next;
                        }
                    }
                }
                PokemonVolatileStatus::YAWN => {
                    // PS yawn `onTryHit`: fails on any status; `addVolatile`
                    // fails if already yawning (see `yawn_volatile_blocked`).
                    let has_status = state
                        .side(ts)
                        .slot_pokemon(tslot)
                        .map(|p| p.status != PokemonStatus::NONE)
                        .unwrap_or(true);
                    // A new `yawn` volatile sorts its TryAddVolatile handlers
                    // before they can block it.
                    if !has_status && state.side(ts).active()[tslot].yawn == 0 {
                        try_add_volatile_ties(state, rng, ts, tslot);
                    }
                    if has_status
                        || state.side(ts).active()[tslot].yawn > 0
                        || yawn_volatile_blocked(state, ts, tslot, side, slot)
                    {
                        move_failed = true;
                    } else {
                        let a = state.side_mut(ts).active_slot_mut(tslot);
                        a.yawn = 2;
                        a.yawn_source = Some((side, idx));
                    }
                }
                PokemonVolatileStatus::PARTIALLYTRAPPED => {
                    try_partial_trap(state, rng, ts, tslot, side, idx)
                }
                PokemonVolatileStatus::TRAPPED => {
                    // PS `addVolatile('trapped')` returns false while present, and
                    // Ghost types are immune (data/typechart.ts `trapped`, checked by
                    // `runStatusImmunity`): the move fails (Block / Mean Look `onHit`).
                    let a = &state.side(ts).active()[tslot];
                    let ghost = state.side(ts).slot_pokemon(tslot).is_some_and(|p| p.has_type(&PokemonType::GHOST));
                    if a.trapped || ghost || !state.side(ts).slot_pokemon(tslot).map(|p| p.hp > 0).unwrap_or(false) {
                        move_failed = true;
                    } else {
                        // PS `addVolatile('trapped', source, move, 'trapper')`: the
                        // user holds the link (released when the user leaves).
                        try_add_volatile_ties(state, rng, ts, tslot);
                        let a = state.side_mut(ts).active_slot_mut(tslot);
                        a.trapped = true;
                        a.trapper = Some((side, idx));
                    }
                }
                PokemonVolatileStatus::OCTOLOCK => {
                    // PS `addVolatile`: returns false while present (no
                    // `onRestart`), so the status move fails (no hit-loop
                    // `eachEvent`); a new one runs the TryAddVolatile ties.
                    if state.side(ts).active()[tslot].octolock.is_some() {
                        move_failed = true;
                    } else if state.side(ts).slot_pokemon(tslot).map(|p| p.hp > 0).unwrap_or(false) {
                        try_add_volatile_ties(state, rng, ts, tslot);
                        state.side_mut(ts).active_slot_mut(tslot).octolock = Some((side, idx));
                    }
                }
                PokemonVolatileStatus::HEALBLOCK => {
                    // data/moves.ts healblock.condition: duration 5 (2 from
                    // Psychic Noise, 7 with Persistent); no refresh while active.
                    // Psychic Noise's Heal Block is a `secondary`: Sheer Force
                    // deletes it and the target's Shield Dust filters it out
                    // (PS data/abilities.ts sheerforce / shielddust).
                    let secondary_removed = choice.move_id == Choices::PSYCHICNOISE
                        && (state.side(side).team_member(idx).ability == Abilities::SHEERFORCE
                            || state.side(ts).slot_pokemon(tslot).is_some_and(|p| {
                                p.ability == Abilities::SHIELDDUST
                                    && !crate::engine::abilities::ignores_target_ability(
                                        state.side(side).team_member(idx),
                                        choice,
                                        &p.ability,
                                    )
                            }));
                    // A new volatile sorts its TryAddVolatile handlers (Aroma Veil).
                    if !secondary_removed && state.side(ts).active()[tslot].heal_block == 0 {
                        try_add_volatile_ties(state, rng, ts, tslot);
                    }
                    if secondary_removed {
                    } else if aroma_veil_blocks(state, ts, AromaVeiled::HealBlock, Some((side, idx, choice))) {
                        // Blocked (`null`): no effect, and Psychic Noise's hit still counts.
                        if choice.move_id != Choices::PSYCHICNOISE {
                            move_failed = true;
                        }
                    } else if state.side(ts).active()[tslot].heal_block > 0 {
                        // `onRestart`: Psychic Noise returns early (no fail).
                        // PS data/moves.ts healblock.condition.onRestart.
                        if choice.move_id != Choices::PSYCHICNOISE {
                            move_failed = true;
                        }
                    } else {
                        let persistent = state.side(side).team_member(idx).ability == Abilities::PERSISTENT;
                        state.side_mut(ts).active_slot_mut(tslot).heal_block = if choice.move_id == Choices::PSYCHICNOISE {
                            2
                        } else if persistent {
                            7
                        } else {
                            5
                        };
                    }
                }
                PokemonVolatileStatus::MINIMIZE => {
                    // data/moves.ts minimize.volatileStatus; no restart fail.
                    if !state.side(ts).active()[tslot].minimize {
                        try_add_volatile_ties(state, rng, ts, tslot);
                    }
                    state.side_mut(ts).active_slot_mut(tslot).minimize = true;
                }
                other => {
                    // TODO(PS parity): model remaining volatiles (Leech Seed,
                    // Substitute, Tar Shot, ...).
                    let _ = other;
                }
            }
        }

        // Helping Hand: x1.5 base power on the ally's next move this turn
        // (PS `helpinghand.condition.onBasePower`; stacks by x1.5 on
        // restart). The fuzzer only aims it at the one adjacent ally.
        // Skill Swap (`onHit` -> `skillSwap(source, target)`).
        if choice.move_id == Choices::SKILLSWAP {
            let (ts, tsl) = if is_ally { (side, ally_slot) } else { (target_side, target_slot) };
            if !skill_swap(state, side, slot, ts, tsl, rng) {
                move_failed = true;
            } else {
                ability_update_cures(state, side, slot);
                ability_update_cures(state, ts, tsl);
            }
        }

        // Entrainment (`onTryHit` + `onHit` -> `target.setAbility(source.ability)`).
        // PS data/moves.ts entrainment, sim/pokemon.ts setAbility. No draws.
        if choice.move_id == Choices::ENTRAINMENT {
            let (ts, tsl) = if is_ally { (side, ally_slot) } else { (target_side, target_slot) };
            let src_ability = state.side(side).team_member(idx).ability;
            let target = state.side(ts).active()[tsl].pokemon_index.map(|ti| {
                let mut tm = state.side(ts).team_member(ti).clone();
                tm.ability = real_ability(state, ts, ti);
                (ti, tm)
            });
            match target {
                Some((ti, tm))
                    if !(ts == side && tsl == slot)
                        && tm.hp > 0
                        && tm.ability != src_ability
                        && !crate::engine::mechanics::cant_suppress(tm.ability)
                        && tm.ability != Abilities::TRUANT
                        && !crate::engine::mechanics::no_entrain(src_ability)
                        && !crate::engine::mechanics::cant_suppress(src_ability) =>
                {
                    // `setAbility`: the old ability's End first (Cloud Nine / Air
                    // Lock `onEnd` -> `eachEvent('WeatherChange')`). PS sim/pokemon.ts setAbility.
                    if matches!(tm.ability, Abilities::CLOUDNINE | Abilities::AIRLOCK) {
                        let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
                    }
                    state.side_mut(ts).team_member_mut(ti).ability = src_ability;
                    reset_protean_effect(state, ts, ti);
                    ability_update_cures(state, ts, tsl);
                    // The new ability's Start event (Intimidate, Drought, ...); its
                    // weather / terrain / Cloud Nine `eachEvent` is a sort.
                    if apply_on_switch_abilities(state, ts, tsl, rng) {
                        let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
                    }
                }
                _ => move_failed = true,
            }
        }

        // Psych Up (`onHit`): copy the target's stat stages onto the user.
        // PS data/moves.ts psychup.onHit: the crit-stage volatiles are removed
        // from the user, then copied from the target (Focus Energy here).
        // TODO(PS parity): Laser Focus is not copied yet.
        if choice.move_id == Choices::PSYCHUP {
            if let Some(tidx) = state.side(target_side).active()[target_slot].pokemon_index {
                let copied = state.side(target_side).team_boosts[tidx as usize].clone();
                state.side_mut(side).team_boosts[idx as usize] = copied;
            }
            let target_fe = state.side(target_side).active()[target_slot].focus_energy;
            state.side_mut(side).active_slot_mut(slot).focus_energy = target_fe;
            // Dragon Cheer is removed and copied the same way, with the target's
            // `hasDragonType`.
            let target_dc = state.side(target_side).active()[target_slot].dragon_cheer;
            state.side_mut(side).active_slot_mut(slot).dragon_cheer = target_dc;
        }

        // Uproar self volatile (`addVolatile('uproar')`, duration 3).
        if choice.move_id == Choices::UPROAR && ti == 0 && state.side(side).active()[slot].uproar == 0 {
            state.side_mut(side).active_slot_mut(slot).uproar = 3;
        }

        if choice.move_id == Choices::NORETREAT && ti == 0 {
            state.side_mut(side).active_slot_mut(slot).no_retreat = true;
        }

        if choice.move_id == Choices::CHARGE && ti == 0 {
            state.side_mut(side).active_slot_mut(slot).charge = true;
        }

        // Ghost-type Curse `onTryHit` / `onHit`: fails on an already cursed target,
        // else the user loses half its max HP and the target is cursed.
        if choice.move_id == Choices::CURSE && ti == 0 {
            let t = state.side(side).team_member(idx).types;
            if t.0 == PokemonType::GHOST
                || t.1 == PokemonType::GHOST
                || state.side(side).active()[slot].added_type == PokemonType::GHOST
            {
                if state.side(target_side).active()[target_slot].cursed {
                    move_failed = true;
                } else {
                    let half = state.side(side).team_member(idx).maxhp / 2;
                    apply_direct_damage(state, side, slot, half.max(1));
                    // Champions curse.onHit: aimed at the user or an ally (a
                    // 'self' Curse chosen before the user became Ghost, e.g. by
                    // Protean), `getRandomTarget(source, 'Curse')` picks a foe:
                    // `randomFoe()` (a draw when any foe is up) or `foe.active[0]`.
                    // PS data/mods/champions/moves.ts curse.onHit, sim/battle.ts getRandomTarget.
                    let (cs, csl) = if target_side == side {
                        let foes: Vec<usize> = (0..2)
                            .filter(|&k| state.side(side.other()).slot_pokemon(k).is_some_and(|p| p.hp > 0))
                            .collect();
                        if foes.is_empty() {
                            (side.other(), 0)
                        } else {
                            (side.other(), foes[rng.label("retarget").random(foes.len() as u32) as usize])
                        }
                    } else {
                        (target_side, target_slot)
                    };
                    if state.side(cs).slot_pokemon(csl).is_some_and(|p| p.hp > 0) {
                        state.side_mut(cs).active_slot_mut(csl).cursed = true;
                    }
                }
            }
        }

        // Clangorous Soul (PS data/moves.ts clangoroussoul, champions accuracy
        // true): `onTry` fails at or below 33% HP (`hp <= maxhp * 33 / 100`, a
        // float compare); `onTryHit` +1 to every stat (`boost` fails when no stat
        // can rise); `onHit` the user takes `floor(maxhp * 33 / 100)`.
        if choice.move_id == Choices::CLANGOROUSSOUL && ti == 0 {
            let (hp, maxhp) = {
                let p = state.side(side).team_member(idx);
                (p.hp as i32, p.maxhp as i32)
            };
            let b = state.side(side).slot_boosts(slot).cloned().unwrap_or_default();
            let can_rise = [b.atk, b.def, b.spa, b.spd, b.spe].iter().any(|&x| x < 6);
            if maxhp == 1 || hp * 100 <= maxhp * 33 || !can_rise {
                move_failed = true;
            } else {
                apply_boosts(
                    state,
                    side,
                    idx,
                    &StatBoosts { attack: 1, defense: 1, special_attack: 1, special_defense: 1, speed: 1, ..Default::default() },
                    false,
                );
                apply_direct_damage(state, side, slot, (maxhp * 33 / 100) as i16);
            }
        }

        // Heal Pulse `onHit`: heals the target by ceil(maxhp/2) (x0.75 with
        // Mega Launcher); fails at full HP.
        if choice.move_id == Choices::HEALPULSE && ti == 0 {
            let mega = state.side(side).team_member(idx).ability == Abilities::MEGALAUNCHER;
            // Heal Block (PS data/moves.ts healblock.onTryHeal) returns
            // false for any heal but Pollen Puff's: `battle.heal` fails
            // (`-fail|heal`) before the target is healed.
            let blocked = state.side(target_side).active()[target_slot].heal_block > 0;
            if let Some(tidx) = state.side(target_side).active()[target_slot].pokemon_index {
                let pm = state.side_mut(target_side).team_member_mut(tidx);
                let amount = if mega {
                    ((pm.maxhp as i64 * 3072 + 2047) / 4096) as i16
                } else {
                    (pm.maxhp + 1) / 2
                };
                // Heal Block (PS data/moves.ts healblock.onTryHeal) returns
                // false for any heal but Pollen Puff's: `battle.heal` fails
                // (`-fail|heal`) before the target is healed.
                let full = pm.hp >= pm.maxhp || blocked;
                if !full {
                    pm.hp = (pm.hp + amount).min(pm.maxhp);
                }
                if full {
                    // PS `onHit` returns NOT_FAIL on a failed heal (`-fail`): the
                    // hit still runs its two `eachEvent('Update')`s (spreadMoveHit,
                    // then hitStepMoveHitLoop). sim/battle-actions.ts.
                    move_failed = true;
                    not_fail_result = true;
                    each_event_update(state, rng);
                    each_event_update(state, rng);
                }
            }
        }

        // Imprison (PS data/moves.ts imprison, `addVolatile('imprison')` on the
        // user): fails if the user already has it, else it lands (`-start`) and
        // the move runs its hit loop.
        if choice.move_id == Choices::IMPRISON && ti == 0 && is_self && !move_failed {
            if state.side(side).active()[slot].imprison {
                move_failed = true;
            } else {
                state.side_mut(side).active_slot_mut(slot).imprison = true;
            }
        }
        // Spite `onHit` (PS data/moves.ts spite): the target's last move loses
        // 4 PP (`deductPP(move, 4)`, `ppData.used` set). No last move, or no PP
        // left, fails silently. No draw.
        if choice.move_id == Choices::SPITE && ti == 0 && !move_failed {
            // PS `onHit` returns false (the move fails: `hitStepMoveHitLoop`
            // returns on its first hit, no `eachEvent`) when there is no last
            // move or `deductPP` returns 0 (no PP left, move not in the moveset).
            let mut spite_ok = false;
            let tgt = &state.side(target_side).active()[target_slot];
            if let (Some(lm), Some(tidx)) = (tgt.last_move, tgt.pokemon_index) {
                let ts = state.side_mut(target_side);
                let p = ts.team_member_mut(tidx);
                let found = (0..4usize).find(|&k| {
                    let mi = match k {
                        0 => crate::state::PokemonMoveIndex::M0,
                        1 => crate::state::PokemonMoveIndex::M1,
                        2 => crate::state::PokemonMoveIndex::M2,
                        _ => crate::state::PokemonMoveIndex::M3,
                    };
                    p.moves[&mi].id == lm
                });
                if let Some(k) = found {
                    let mi = match k {
                        0 => crate::state::PokemonMoveIndex::M0,
                        1 => crate::state::PokemonMoveIndex::M1,
                        2 => crate::state::PokemonMoveIndex::M2,
                        _ => crate::state::PokemonMoveIndex::M3,
                    };
                    let mv = &mut p.moves[&mi];
                    if mv.pp > 0 {
                        mv.pp = (mv.pp - 4).max(0);
                        spite_ok = true;
                    }
                    ts.moves_used[tidx as usize][k] = true;
                }
            }
            if !spite_ok {
                move_failed = true;
            }
        }
        // Defog `onHit`: -1 evasion (unless behind a Substitute), clear the
        // target side's screens and both sides' hazards, clear the terrain.
        if choice.move_id == Choices::DEFOG && ti == 0 {
            let mut success = false;
            // PS data/moves.ts defog.onHit: `!target.volatiles['substitute'] || move.infiltrates`.
            let infiltrates = state.side(side).team_member(idx).ability == Abilities::INFILTRATOR;
            if state.side(target_side).active()[target_slot].substitute_health <= 0 || infiltrates {
                if let Some(tidx) = state.side(target_side).active()[target_slot].pokemon_index {
                    let before = state.side(target_side).team_boosts[tidx as usize].eva;
                    // Mirror Armor `onTryBoost` bounces the drop onto the user
                    // (the move's `this.boost` goes through TryBoost).
                    let drop = StatBoosts { evasion: -1, ..Default::default() };
                    let drop = if (target_side, target_slot) != (side, slot) {
                        mirror_armor_bounce(state, side, slot, target_side, tidx, &drop)
                    } else {
                        drop
                    };
                    apply_boosts_from(state, target_side, tidx, &drop, target_side != side, true);
                    success = state.side(target_side).team_boosts[tidx as usize].eva != before;
                }
            }
            let sc = &mut state.side_mut(target_side).side_conditions;
            sc.reflect = 0;
            sc.light_screen = 0;
            sc.aurora_veil = 0;
            sc.safeguard = 0;
            sc.mist = 0;
            for sd in [target_side, side] {
                let sc = &mut state.side_mut(sd).side_conditions;
                if sc.spikes > 0 || sc.toxic_spikes > 0 || sc.stealth_rock > 0 || sc.sticky_web > 0 {
                    success = true;
                }
                sc.spikes = 0;
                sc.toxic_spikes = 0;
                sc.stealth_rock = 0;
                sc.sticky_web = 0;
            }
            if state.terrain.terrain_type != crate::engine::state::Terrain::NONE {
                state.terrain = crate::state::StateTerrain {
                    terrain_type: crate::engine::state::Terrain::NONE,
                    turns_remaining: 0,
                };
                // `clearTerrain` -> `eachEvent('TerrainChange')`.
                refresh_mimicry(state);
                let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
            }
            if !success {
                move_failed = true;
            }
        }

        // Acupressure `onHit`: +2 to a random stat below +6 (`sample`).
        if choice.move_id == Choices::ACUPRESSURE && ti == 0 {
            let (ts, tsl) = match resolved {
                Some((rs, rl)) if rs == side => (rs, rl),
                _ => (side, slot),
            };
            if let Some(tidx) = state.side(ts).active()[tsl].pokemon_index {
                let tb = state.side(ts).team_boosts[tidx as usize].clone();
                let vals = [tb.atk, tb.def, tb.spa, tb.spd, tb.spe, tb.acc, tb.eva];
                let stats: Vec<usize> = (0..7).filter(|&n| vals[n] < 6).collect();
                if stats.is_empty() {
                    move_failed = true;
                } else {
                    let pick = stats[rng.label("multihit").random(stats.len() as u32) as usize];
                    let mut b = StatBoosts::default();
                    match pick {
                        0 => b.attack = 2,
                        1 => b.defense = 2,
                        2 => b.special_attack = 2,
                        3 => b.special_defense = 2,
                        4 => b.speed = 2,
                        5 => b.accuracy = 2,
                        _ => b.evasion = 2,
                    }
                    apply_boosts(state, ts, tidx, &b, false);
                }
            }
        }

        // Role Play `onTryHit` / `onHit` (PS data/moves.ts roleplay): fails when
        // both share an ability, otherwise the user takes the target's ability
        // (`setAbility`, no draw; the switch-out reset restores the base one).
        // `onTryHit`: a target with the failroleplay flag, or a user whose
        // ability can't be suppressed, fails; `setAbility` refuses a
        // cantsuppress target ability (PS data/moves.ts roleplay, data/abilities.ts).
        if choice.move_id == Choices::ROLEPLAY && ti == 0 {
            let user_idx = state.side(side).active()[slot].pokemon_index;
            let target_ability = state.side(target_side).active()[target_slot]
                .pokemon_index
                .map(|t| held_ability(state, target_side, target_slot, t));
            let blocked = match (user_idx, target_ability) {
                (Some(u), Some(ta)) => {
                    crate::engine::mechanics::fail_role_play(ta)
                        || crate::engine::mechanics::cant_suppress(ta)
                        || crate::engine::mechanics::cant_suppress(real_ability(state, side, u))
                }
                _ => false,
            };
            match (user_idx, target_ability) {
                (Some(u), Some(ta)) if !blocked && state.side(side).team_member(u).ability != ta => {
                    // A user under Gastro Acid has No Ability here (the engine's
                    // model of the `gastroacid` volatile). PS `setAbility` still
                    // sets the copy, but `singleEvent('Start')` is suppressed
                    // (`suppressingAbility`) and the suppression lasts until the
                    // user switches out, so the copy is not applied here.
                    if state.side(side).team_member(u).ability != Abilities::NONE {
                        // `setAbility`: `singleEvent('End', oldAbility)` first; a Cloud
                        // Nine / Air Lock `onEnd` runs `eachEvent('WeatherChange')`.
                        // PS sim/pokemon.ts setAbility, data/abilities.ts cloudnine.
                        if matches!(real_ability(state, side, u), Abilities::CLOUDNINE | Abilities::AIRLOCK) {
                            let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
                        }
                        state.side_mut(side).team_member_mut(u).ability = ta;
                        reset_protean_effect(state, side, u);
                        // PS `setAbility` runs the new ability's Start event
                        // (Sand Stream, Drought, ...), as Skill Swap does: its
                        // `eachEvent('WeatherChange')` & co. is a sort of all actives.
                        if apply_on_switch_abilities(state, side, slot, rng) {
                            let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
                        }
                    }
                }
                _ => move_failed = true,
            }
        }

        // Taunt: `addVolatile('taunt')`, duration 3 (+1 when the target already
        // acted this turn after its first turn out).
        if choice.move_id == Choices::TAUNT && ti == 0 {
            let tidx = state.side(target_side).active()[target_slot].pokemon_index;
            if state.side(target_side).active()[target_slot].taunt > 0 {
                move_failed = true;
            } else {
                let will_move = queue.iter().any(|a| a.side == target_side && Some(a.pokemon_index) == tidx);
                let a = state.side_mut(target_side).active_slot_mut(target_slot);
                a.taunt = 3 + (a.active_turns > 0 && !will_move) as u8;
            }
        }

        // Strength Sap `onHit`: heal by the target's boosted Atk, then -1 Atk.
        if choice.move_id == Choices::STRENGTHSAP && ti == 0 {
            let tb = state.side(target_side).slot_boosts(target_slot).cloned().unwrap_or_default();
            if tb.atk == -6 {
                move_failed = true;
            } else if let Some(tp) = state.side(target_side).slot_pokemon(target_slot) {
                let atk = multiply_boost(tb.atk, tp.attack).max(1);
                let ooze = tp.ability == Abilities::LIQUIDOOZE;
                let tidx = state.side(target_side).active()[target_slot].pokemon_index.unwrap();
                // Mirror Armor bounces the drop back onto the user (`boost`
                // from another pokemon; PS data/abilities.ts mirrorarmor).
                let drop = StatBoosts { attack: -1, ..Default::default() };
                let drop = if (target_side, target_slot) != (side, slot) {
                    mirror_armor_bounce(state, side, slot, target_side, tidx, &drop)
                } else {
                    drop
                };
                apply_boosts_from(state, target_side, tidx, &drop, target_side != side, true);
                if ooze {
                    // Liquid Ooze `onSourceTryHeal` (`strengthsap` is on its list): the
                    // heal event still runs and the user takes the amount as damage,
                    // even at full HP. PS data/abilities.ts liquidooze, sim/battle.ts heal.
                    // Magic Guard blocks it too (see apply_indirect_damage).
                    apply_indirect_damage(state, side, slot, atk);
                } else {
                    let pm = state.side_mut(side).team_member_mut(idx);
                    pm.hp = (pm.hp + atk).min(pm.maxhp);
                }
            }
        }

        // Perish Song `onHitField`: every active pokemon without the volatile
        // (and not immune via Soundproof) gets it with duration 4.
        if choice.move_id == Choices::PERISHSONG && ti == 0 {
            let mut result = false;
            for sd in [Side::One, Side::Two] {
                for sl in 0..2 {
                    let Some(p) = state.side(sd).slot_pokemon(sl) else { continue };
                    if p.hp <= 0 {
                        continue;
                    }
                    // PS `runEvent('Invulnerability')` first: a pokemon in the
                    // semi-invulnerable turn of Fly / Dig / ... is missed.
                    if semi_invulnerable_blocks(state, side, idx, choice.move_id, sd, sl) {
                        result = true;
                        continue;
                    }
                    // PS `runEvent('TryHit')` per pokemon: Good as Gold's
                    // `onTryHit` returns null for a Status move from another pokemon
                    // (Mold Breaker ignores it), which still counts as a result.
                    // Soundproof and the absorbing abilities (Dry Skin / Water
                    // Absorb on a Liquid Voice Perish Song) are `onTryHit` too,
                    // all with `target !== source`. PS data/abilities.ts.
                    let other_breakable = (sd, sl) != (side, slot)
                        && !crate::engine::abilities::ignores_target_ability(
                            state.side(side).team_member(idx),
                            choice,
                            &p.ability,
                        );
                    let gag_blocks = other_breakable
                        && matches!(p.ability, Abilities::GOODASGOLD | Abilities::SOUNDPROOF);
                    if gag_blocks || (other_breakable && try_absorb(state, sd, sl, choice.move_type)) {
                        result = true;
                    } else if state.side(sd).active()[sl].perish == 0 {
                        state.side_mut(sd).active_slot_mut(sl).perish = 4;
                        result = true;
                    }
                }
            }
            if !result {
                move_failed = true;
            }
        }

        // Attract: `addVolatile('attract')` (fails if already infatuated).
        if choice.move_id == Choices::ATTRACT && ti == 0 {
            if state.side(target_side).active()[target_slot].attract.is_some() {
                move_failed = true;
            } else {
                state.side_mut(target_side).active_slot_mut(target_slot).attract = Some((side, idx));
            }
        }

        // Roost `self.volatileStatus`: Flying is filtered out of the user's types
        // (`condition.onType`) until the end of the turn. PS data/moves.ts roost.
        if choice.move_id == Choices::ROOST && ti == 0 && !move_failed {
            let t = state.side(side).team_member(idx).types;
            if (t.0 == PokemonType::FLYING || t.1 == PokemonType::FLYING)
                && state.side(side).active()[slot].roost_orig.is_none()
            {
                let nt = roosted_types(t);
                state.side_mut(side).team_member_mut(idx).types = nt;
                state.side_mut(side).active_slot_mut(slot).roost_orig = Some(t);
            }
        }

        // Encore (champions override, data/mods/champions/moves.ts): `onStart`
        // fails without a usable `lastMove` (or for `failencore` moves / 0 PP);
        // otherwise the target's queued move is replaced (`changeAction`), or
        // the duration grows by one if it has no move left this turn.
        if choice.move_id == Choices::ENCORE && ti == 0 && !move_failed {
            let t = &state.side(target_side).active()[target_slot];
            let tidx = t.pokemon_index;
            let last = t.last_move;
            let fail_ids = [
                Choices::BLAZINGTORQUE, Choices::COMBATTORQUE, Choices::DYNAMAXCANNON, Choices::ENCORE,
                Choices::MAGICALTORQUE, Choices::METRONOME, Choices::MIMIC, Choices::NOXIOUSTORQUE,
                Choices::SKETCH, Choices::WICKEDTORQUE, Choices::ASSIST, Choices::COPYCAT, Choices::MEFIRST,
                Choices::MIRRORMOVE, Choices::NATUREPOWER, Choices::SLEEPTALK, Choices::STRUGGLE,
                Choices::TRANSFORM,
            ];
            let usable = match (tidx, last) {
                (Some(ti_), Some(m)) if !fail_ids.contains(&m) => {
                    let p = state.side(target_side).team_member(ti_);
                    [&p.moves.m0, &p.moves.m1, &p.moves.m2, &p.moves.m3]
                        .iter()
                        .any(|mv| mv.id == m && mv.pp > 0)
                }
                _ => false,
            };
            if t.encore.is_some() || !usable {
                move_failed = true;
            } else {
                let m = last.unwrap();
                let tidx = tidx.unwrap();
                let will_move = queue
                    .iter()
                    .any(|q| q.side == target_side && q.pokemon_index == tidx);
                state.side_mut(target_side).active_slot_mut(target_slot).encore =
                    Some((m, if will_move { 3 } else { 4 }));
                if will_move {
                    state.queue_change = plan_queue_change(state, rng, queue, target_side, target_slot, m);
                }
            }
        }

        // Torment (PS data/moves.ts torment, `addVolatile('torment')`): fails
        // if the target already has it. The disable itself is in
        // `end_turn_disable_moves` and the choices come from PS's requests.
        if choice.move_id == Choices::TORMENT && ti == 0 && !move_failed {
            if state.side(target_side).active()[target_slot].torment {
                move_failed = true;
            } else {
                state.side_mut(target_side).active_slot_mut(target_slot).torment = true;
            }
        }

        // Forest's Curse (PS data/moves.ts forestscurse.onHit): fails on a Grass
        // type, else `addType('Grass')` (a dual-typed target gets it as `added_type`).
        if choice.move_id == Choices::FORESTSCURSE && ti == 0 && !move_failed {
            let tidx = state.side(target_side).active()[target_slot].pokemon_index;
            match tidx {
                Some(tidx) => {
                    let t = state.side(target_side).team_member(tidx).types;
                    if t.0 == PokemonType::GRASS || t.1 == PokemonType::GRASS {
                        move_failed = true;
                    } else if t.1 == PokemonType::TYPELESS {
                        state.side_mut(target_side).team_member_mut(tidx).types.1 = PokemonType::GRASS;
                        state.side_mut(target_side).slot_scratch_mut(target_slot).added_grass = true;
                    } else {
                        state.side_mut(target_side).slot_scratch_mut(target_slot).added_type = PokemonType::GRASS;
                    }
                }
                None => move_failed = true,
            }
        }

        // Trick-or-Treat `onHit` (PS data/moves.ts trickortreat): fails if the
        // target has Ghost (`hasType`, counting `addedType`), else `addType('Ghost')`.
        // TODO(PS parity): Curse Glitch (a target in slot 1 with a queued Curse).
        if choice.move_id == Choices::TRICKORTREAT && ti == 0 && !move_failed {
            let tidx = state.side(target_side).active()[target_slot].pokemon_index;
            match tidx {
                Some(tidx) => {
                    let t = state.side(target_side).team_member(tidx).types;
                    let has_ghost = t.0 == PokemonType::GHOST
                        || t.1 == PokemonType::GHOST
                        || state.side(target_side).active()[target_slot].added_type == PokemonType::GHOST;
                    if has_ghost {
                        move_failed = true;
                    } else {
                        state.side_mut(target_side).slot_scratch_mut(target_slot).added_type = PokemonType::GHOST;
                    }
                }
                None => move_failed = true,
            }
        }

        // Quash (PS data/moves.ts quash.onHit): the target's queued action gets
        // `order = 201` and moves after every other move; fails without one.
        if choice.move_id == Choices::QUASH && ti == 0 {
            let tidx = state.side(target_side).active()[target_slot].pokemon_index;
            let will_move = tidx.is_some()
                && queue.iter().any(|q| q.side == target_side && Some(q.pokemon_index) == tidx);
            if will_move {
                state.queue_quash = Some((target_side, target_slot));
            } else {
                move_failed = true;
            }
        }
        // After You (PS data/moves.ts afteryou.onHit): fails when the target
        // has no queued action; otherwise it moves next.
        if choice.move_id == Choices::AFTERYOU && ti == 0 {
            let tidx = state.side(target_side).active()[target_slot].pokemon_index;
            let will_move = tidx.is_some()
                && queue.iter().any(|q| q.side == target_side && Some(q.pokemon_index) == tidx);
            if will_move {
                state.queue_prioritize = Some((target_side, target_slot));
            } else {
                move_failed = true;
            }
        }

        // Instruct (PS data/moves.ts instruct.onHit): fails without a last move
        // (`lastMove`), when that move has `failinstruct` / charge / recharge, or
        // when its PP slot is empty. Otherwise the target's last move is queued
        // again at the front of the queue, with the target's last target location.
        // TODO(PS parity): the beakblast / focuspunch / shelltrap volatiles on the
        // target (not modeled; focuspunch is checked through `focus_punch`).
        if choice.move_id == Choices::INSTRUCT && ti == 0 && !move_failed {
            let t = &state.side(target_side).active()[target_slot];
            let tidx = t.pokemon_index;
            let last = t.last_move;
            let loc = t.last_target_loc;
            let focus = t.focus_punch;
            let repeat = match (tidx, last) {
                (Some(ti_), Some(m)) if !INSTRUCT_FAIL_IDS.contains(&m) && !focus => {
                    let p = state.side(target_side).team_member(ti_);
                    let pp_ok = [&p.moves.m0, &p.moves.m1, &p.moves.m2, &p.moves.m3]
                        .iter()
                        .find(|mv| mv.id == m)
                        .map_or(true, |mv| mv.pp > 0);
                    if pp_ok { Some((ti_, m)) } else { None }
                }
                _ => None,
            };
            match repeat {
                Some((ti_, m)) => state.queue_instruct = Some((target_side, target_slot, ti_, m, loc)),
                None => move_failed = true,
            }
        }

        // Magnetic Flux (PS data/moves.ts magneticflux.onHitSide, target allySide):
        // each living active ally of the user, the user included, with Plus or
        // Minus gets +1 Def and +1 SpD (`boost`, the user as the source). Fails
        // with no such ally or when no stat changes.
        if choice.move_id == Choices::MAGNETICFLUX && ti == 0 && !move_failed {
            let mut did = false;
            for sl in 0..2 {
                let Some(aidx) = state.side(side).active()[sl].pokemon_index else { continue };
                let m = state.side(side).team_member(aidx);
                if m.hp <= 0 || !matches!(m.ability, Abilities::PLUS | Abilities::MINUS) {
                    continue;
                }
                let before = (state.side(side).team_boosts[aidx as usize].def, state.side(side).team_boosts[aidx as usize].spd);
                apply_boosts_from(
                    state,
                    side,
                    aidx,
                    &StatBoosts { defense: 1, special_defense: 1, ..Default::default() },
                    false,
                    sl != slot,
                );
                let tb = &state.side(side).team_boosts[aidx as usize];
                did |= (tb.def, tb.spd) != before;
            }
            if !did {
                move_failed = true;
            }
        }

        // Disable `condition.onStart`: fails if already disabled or the last
        // move is out of PP; the duration drops by one when the target still
        // has a queued action. PS data/moves.ts disable.
        if choice.move_id == Choices::DISABLE && ti == 0 {
            let t = &state.side(target_side).active()[target_slot];
            let tidx = t.pokemon_index;
            let last = t.last_move;
            let no_pp = match (tidx, last) {
                (Some(ti_), Some(m)) => {
                    let p = state.side(target_side).team_member(ti_);
                    [&p.moves.m0, &p.moves.m1, &p.moves.m2, &p.moves.m3]
                        .iter()
                        .any(|mv| mv.id == m && mv.pp <= 0)
                }
                _ => false,
            };
            // PS data/moves.ts disable `onTryHit` (Struggle / no last move) runs
            // in `spreadMoveHit`, after the accuracy roll, like `onStart`.
            let struggle = last == Some(Choices::STRUGGLE);
            if t.disabled.is_some() || last.is_none() || no_pp || struggle {
                move_failed = true;
            } else {
                let will_move = queue
                    .iter()
                    .any(|q| q.side == target_side && Some(q.pokemon_index) == tidx);
                state.side_mut(target_side).active_slot_mut(target_slot).disabled =
                    Some((last.unwrap(), if will_move { 4 } else { 5 }));
            }
        }

        // Dragon Cheer: `addVolatile('dragoncheer')` on the ally; `onStart`
        // fails if the target has Focus Energy (or already has the volatile).
        if choice.move_id == Choices::DRAGONCHEER && ti == 0 {
            let a = &state.side(side).active()[ally_slot];
            if a.pokemon_index.is_none() || a.focus_energy || a.dragon_cheer.is_some() {
                move_failed = true;
            } else {
                let tidx = a.pokemon_index.unwrap();
                let t = state.side(side).team_member(tidx).types;
                let dragon = t.0 == PokemonType::DRAGON || t.1 == PokemonType::DRAGON;
                state.side_mut(side).active_slot_mut(ally_slot).dragon_cheer = Some(dragon);
            }
        }

        // Guard Split / Power Split `onHit`: both pokemon get the floored average of
        // the stored Def & SpD (Atk & SpA). PS data/moves.ts guardsplit / powersplit.
        if matches!(choice.move_id, Choices::GUARDSPLIT | Choices::POWERSPLIT)
            && ti == 0
            && !move_failed
            && (target_side, target_slot) != (side, slot)
        {
            if let Some(tidx) = state.side(target_side).active()[target_slot].pokemon_index {
                let guard = choice.move_id == Choices::GUARDSPLIT;
                let snap = |p: &crate::state::Pokemon| [p.attack, p.defense, p.special_attack, p.special_defense];
                let (ts_orig, us_orig) = (
                    snap(state.side(target_side).team_member(tidx)),
                    snap(state.side(side).team_member(idx)),
                );
                if state.side(target_side).saved_stats[tidx as usize].is_none() {
                    state.side_mut(target_side).saved_stats[tidx as usize] = Some(ts_orig);
                }
                if state.side(side).saved_stats[idx as usize].is_none() {
                    state.side_mut(side).saved_stats[idx as usize] = Some(us_orig);
                }
                let (a, b) = if guard { (1, 3) } else { (0, 2) };
                let new_a = (ts_orig[a] + us_orig[a]) / 2;
                let new_b = (ts_orig[b] + us_orig[b]) / 2;
                for (sd, ix) in [(target_side, tidx), (side, idx)] {
                    let p = state.side_mut(sd).team_member_mut(ix);
                    if guard {
                        p.defense = new_a;
                        p.special_defense = new_b;
                    } else {
                        p.attack = new_a;
                        p.special_attack = new_b;
                    }
                }
            }
        }

        // Worry Seed `onHit`: the target's ability becomes Insomnia (cures sleep).
        // Trick / Switcheroo `onHit`: fail when neither pokemon holds an item
        // (items are not exchanged otherwise yet). PS data/moves.ts trick.
        // TODO(PS parity): item swap, Sticky Hold, Klutz-style restrictions.
        if matches!(choice.move_id, Choices::TRICK | Choices::SWITCHEROO) && ti == 0 && !move_failed {
            let mine = state.side(side).team_member(idx).item;
            let theirs = state
                .side(target_side)
                .slot_pokemon(target_slot)
                .map(|d| d.item)
                .unwrap_or(crate::engine::items::Items::NONE);
            if mine == crate::engine::items::Items::NONE && theirs == crate::engine::items::Items::NONE {
                move_failed = true;
            }
        }
        // Gastro Acid: `volatileStatus: 'gastroacid'` suppresses the target's
        // ability until it switches out (modelled as No Ability, which the
        // switch-out reset undoes). Fails on unsuppressable abilities.
        // PS data/moves.ts gastroacid. TODO(PS parity): Ability Shield.
        // PS addVolatile fails when the `gastroacid` volatile already exists.
        if choice.move_id == Choices::GASTROACID && ti == 0 && !move_failed
            && state.side(target_side).active()[target_slot].gastro_acid
        {
            move_failed = true;
        }
        if choice.move_id == Choices::GASTROACID && ti == 0 && !move_failed {
            if let Some(tidx) = state.side(target_side).active()[target_slot].pokemon_index {
                let a = real_ability(state, target_side, tidx);
                if gastro_cant_suppress(a) {
                    move_failed = true;
                } else {
                    if a != Abilities::NONE {
                        // PS gastroacid `onStart`: `singleEvent('End', ability)`. A Cloud
                        // Nine / Air Lock `onEnd` runs `eachEvent('WeatherChange')`.
                        if matches!(a, Abilities::CLOUDNINE | Abilities::AIRLOCK) {
                            let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
                        }
                        mold_breaker_forget(state, target_side, tidx);
                        state.side_mut(target_side).team_member_mut(tidx).ability = Abilities::NONE;
                    }
                    state.side_mut(target_side).active_slot_mut(target_slot).gastro_acid = true;
                    state.side_mut(target_side).slot_scratch_mut(target_slot).gastro_ability = a;
                }
            }
        }
        // Revival Blessing `onTryHit`: fails without a fainted party member.
        // PS data/moves.ts revivalblessing.
        if choice.move_id == Choices::REVIVALBLESSING && ti == 0 && !move_failed {
            if state.side(side).team.iter().any(|p| p.hp <= 0 && p.id != crate::pokemon::PokemonName::NONE) {
                state.side_mut(side).slot_scratch_mut(slot).reviving = true;
            } else {
                move_failed = true;
            }
        }
        // Shed Tail `onTryHit` / `onHit`: needs a replacement, no existing
        // Substitute and more than ceil(maxhp / 2) HP; the user pays that much
        // and the replacement gets the substitute (floor(maxhp / 4) HP).
        // PS data/moves.ts shedtail.
        if choice.move_id == Choices::SHEDTAIL && ti == 0 && !move_failed {
            let (hp, maxhp) = {
                let p = state.side(side).team_member(idx);
                (p.hp as i32, p.maxhp as i32)
            };
            let cost = (maxhp + 1) / 2;
            if state.side(side).find_replacement().is_none()
                || state.side(side).active()[slot].substitute_health > 0
                || hp <= cost
            {
                move_failed = true;
            } else {
                apply_direct_damage(state, side, slot, cost as i16);
                let a = state.side_mut(side).active_slot_mut(slot);
                a.substitute_health = (maxhp / 4) as i16;
                a.partial_trap = None;
                a.shed_tailing = true;
            }
        }
        // Power Trick: swaps Attack and Defense; using it again swaps back
        // (`onRestart` ends the volatile). Undone on switch-out. PS data/moves.ts.
        if choice.move_id == Choices::POWERTRICK && ti == 0 && !move_failed {
            let pm = state.side_mut(side).team_member_mut(idx);
            std::mem::swap(&mut pm.attack, &mut pm.defense);
            let a = state.side_mut(side).active_slot_mut(slot);
            a.power_trick = !a.power_trick;
        }
        // Electrify: fails if the target already moved (`onTryHit`); lasts the turn.
        if choice.move_id == Choices::ELECTRIFY && ti == 0 && !move_failed {
            let tidx = state.side(target_side).active()[target_slot].pokemon_index;
            let will_move = queue.iter().any(|q| q.side == target_side && Some(q.pokemon_index) == tidx);
            if !will_move && state.side(target_side).active()[target_slot].active_turns > 0 {
                move_failed = true;
            } else {
                state.side_mut(target_side).active_slot_mut(target_slot).electrify = true;
            }
        }
        // Belly Drum `onHit`: fails at or below half HP (or Atk +6); otherwise
        // the user pays floor(maxhp / 2) and its Attack goes to +6 (`boost atk: 12`).
        // PS data/moves.ts bellydrum.
        if choice.move_id == Choices::BELLYDRUM && ti == 0 && !move_failed {
            let (hp, maxhp) = {
                let p = state.side(side).team_member(idx);
                (p.hp as i32, p.maxhp as i32)
            };
            if hp * 2 <= maxhp
                || state.side(side).team_boosts[idx as usize].atk >= 6
                || maxhp == 1
            {
                move_failed = true;
            } else {
                apply_direct_damage(state, side, slot, (maxhp / 2) as i16);
                apply_boosts(state, side, idx, &StatBoosts { attack: 12, ..Default::default() }, false);
            }
        }
        // Soak `onHit`: `setType('Water')` (fails if already pure Water).
        // PS data/moves.ts soak. The original types come back on switch-out
        // (`type_change`, as for Libero).
        // TODO(PS parity): `setType` refusals (Multitype, Arceus, Tera).
        // Magic Powder `onHit`: the same with Psychic (PS data/moves.ts magicpowder).
        let set_type = match choice.move_id {
            Choices::SOAK => Some(PokemonType::WATER),
            Choices::MAGICPOWDER => Some(PokemonType::PSYCHIC),
            _ => None,
        };
        if let (Some(new_type), 0, false) = (set_type, ti, move_failed) {
            if let Some(tidx) = state.side(target_side).active()[target_slot].pokemon_index {
                let cur = state.side(target_side).team_member(tidx).types;
                if cur == (new_type, PokemonType::TYPELESS) {
                    move_failed = true;
                } else {
                    if state.side(target_side).active()[target_slot].type_change.is_none() {
                        state.side_mut(target_side).slot_scratch_mut(target_slot).type_change = Some(cur);
                    }
                    state.side_mut(target_side).team_member_mut(tidx).types = (new_type, PokemonType::TYPELESS);
                }
            }
        }
        // Topsy-Turvy `onHit`: inverts every stat stage; fails when all are 0.
        // PS data/moves.ts topsyturvy.
        if choice.move_id == Choices::TOPSYTURVY && ti == 0 && !move_failed {
            if let Some(tidx) = state.side(target_side).active()[target_slot].pokemon_index {
                let b = &mut state.side_mut(target_side).team_boosts[tidx as usize];
                let stages = [&mut b.atk, &mut b.def, &mut b.spa, &mut b.spd, &mut b.spe, &mut b.acc, &mut b.eva];
                let mut success = false;
                for st in stages {
                    if *st != 0 {
                        *st = -*st;
                        success = true;
                    }
                }
                if !success {
                    move_failed = true;
                }
            }
        }
        // Simple Beam `onHit`: `setAbility('simple')` (PS data/moves.ts simplebeam).
        // Its `onTryHit` runs after the accuracy roll: it fails on Simple /
        // Truant / unsuppressable abilities.
        if choice.move_id == Choices::SIMPLEBEAM && ti == 0 && !move_failed {
            if let Some(tidx) = state.side(target_side).active()[target_slot].pokemon_index {
                let a = real_ability(state, target_side, tidx);
                if a == Abilities::SIMPLE
                    || a == Abilities::TRUANT
                    || matches!(
                        a,
                        Abilities::MULTITYPE
                            | Abilities::STANCECHANGE
                            | Abilities::DISGUISE
                            | Abilities::ZENMODE
                            | Abilities::SCHOOLING
                            | Abilities::SHIELDSDOWN
                            | Abilities::POWERCONSTRUCT
                            | Abilities::COMATOSE
                            | Abilities::ICEFACE
                            | Abilities::GULPMISSILE
                            | Abilities::HUNGERSWITCH
                            | Abilities::BATTLEBOND
                    )
                {
                    move_failed = true;
                } else {
                    state.side_mut(target_side).team_member_mut(tidx).ability = Abilities::SIMPLE;
                    reset_protean_effect(state, target_side, tidx);
                }
            }
        }
        if choice.move_id == Choices::WORRYSEED && ti == 0 && !move_failed {
            if let Some(tidx) = state.side(target_side).active()[target_slot].pokemon_index {
                let real = real_ability(state, target_side, tidx);
                let pm = state.side_mut(target_side).team_member_mut(tidx);
                // PS data/moves.ts worryseed.onTryHit: `flags['cantsuppress']`.
                if crate::engine::mechanics::cant_suppress(real) {
                    move_failed = true;
                } else {
                    pm.ability = Abilities::INSOMNIA;
                    reset_protean_effect(state, target_side, tidx);
                    let pm = state.side_mut(target_side).team_member_mut(tidx);
                    if pm.status == PokemonStatus::SLEEP {
                        pm.status = PokemonStatus::NONE;
                    }
                }
            }
        }

        // Clear Smog `onHit`: `target.clearBoosts()` (PS data/moves.ts clearsmog).
        if choice.move_id == Choices::CLEARSMOG && ti == 0 && !move_failed {
            if let Some(tidx) = state.side(target_side).active()[target_slot].pokemon_index {
                state.side_mut(target_side).clear_member_boosts(tidx);
            }
        }

        // Recycle `onHit`: fails without a consumed item (`lastItem`).
        // TODO(PS parity): items are not modeled, so `lastItem` is always empty.
        if choice.move_id == Choices::RECYCLE && ti == 0 {
            move_failed = true;
        }

        // Leech Seed: `addVolatile('leechseed')` (fails if already seeded).
        if choice.move_id == Choices::LEECHSEED && ti == 0 {
            if state.side(target_side).active()[target_slot].leech_seed.is_some() {
                move_failed = true;
            } else {
                state.side_mut(target_side).active_slot_mut(target_slot).leech_seed = Some((side, slot));
            }
        }

        // Rest `onHit`: `setStatus('slp')` (draws the sleep duration), then the
        // sleep counter is forced to 3 and the user heals fully.
        if choice.move_id == Choices::REST && ti == 0 {
            // `setStatus('slp')` replaces another status (only sleep itself is
            // refused, handled in `onTry`).
            let prev = state.side(side).team_member(idx).status;
            state.side_mut(side).team_member_mut(idx).status = PokemonStatus::NONE;
            // `setStatus` -> `runEvent('SetStatus')` handler ties (Sweet Veil, ...).
            set_status_event_tie(state, rng, side, slot, PokemonStatus::SLEEP, false, false);
            let immune = status_immune(state, side, slot, PokemonStatus::SLEEP, false, side, slot);
            state.side_mut(side).team_member_mut(idx).status = prev;
            if immune {
                move_failed = true;
            } else {
                set_status(state, rng, side, idx, PokemonStatus::SLEEP);
                let pm = state.side_mut(side).team_member_mut(idx);
                pm.sleep_turns = 3;
                pm.hp = pm.maxhp;
            }
        }

        // Sleep Talk `onHit` (PS data/moves.ts sleeptalk): `sample` one of the
        // user's moves without `nosleeptalk` / `charge`, then `useMove` it.
        if choice.move_id == Choices::SLEEPTALK && ti == 0 {
            const SKIP: &[&str] = &["beakblast", "belch", "blazingtorque", "celebrate", "combattorque", "dynamaxcannon", "freezeshock", "holdhands", "iceburn", "magicaltorque", "metronome", "mimic", "noxioustorque", "shadowforce", "shelltrap", "sketch", "wickedtorque", "assist", "bide", "bounce", "chatter", "copycat", "dig", "dive", "fly", "focuspunch", "geomancy", "mefirst", "mirrormove", "naturepower", "phantomforce", "razorwind", "skullbash", "skyattack", "skydrop", "sleeptalk", "solarbeam", "solarblade", "struggle", "uproar"];
            let p = state.side(side).team_member(idx);
            let cands: Vec<Choices> = [&p.moves.m0, &p.moves.m1, &p.moves.m2, &p.moves.m3]
                .iter()
                .map(|m| m.id)
                .filter(|id| *id != Choices::NONE && !SKIP.contains(&id.to_string().to_lowercase().as_str()))
                .collect();
            if cands.is_empty() {
                move_failed = true;
            } else {
                // PS runs `onHit` inside `hitStepMoveHitLoop`, so the
                // harness tags this `sample` "multihit" (fuzz_purpose.js).
                let m = cands[rng.label("multihit").random(cands.len() as u32) as usize];
                run_copied_move(state, rng, action, m, &mut copy_order, will_act, queue);
            }
        }

        // Copycat `onHit`: `useMove(lastMove)` unless it has `failcopycat`.
        if choice.move_id == Choices::COPYCAT && ti == 0 {
            const FAIL: &[&str] = &["banefulbunker", "beakblast", "belch", "behemothbash", "behemothblade", "blazingtorque", "burningbulwark", "celebrate", "combattorque", "dynamaxcannon", "holdhands", "kingsshield", "magicaltorque", "metronome", "mimic", "noxioustorque", "protect", "shelltrap", "sketch", "spikyshield", "terastarstorm", "wickedtorque", "assist", "bestow", "chatter", "circlethrow", "copycat", "counter", "covet", "destinybond", "detect", "dragontail", "endure", "feint", "focuspunch", "followme", "helpinghand", "mefirst", "mirrormove", "naturepower", "ragepowder", "roar", "snatch", "sleeptalk", "spotlight", "struggle", "switcheroo", "thief", "transform", "trick", "whirlwind", "matblock"];
            match state.last_move {
                Some(m) if !FAIL.contains(&m.to_string().to_lowercase().as_str()) => {
                    run_copied_move(state, rng, action, m, &mut copy_order, will_act, queue);
                }
                Some(_) => move_failed = true,
                None => {}
            }
        }

        // Tidy Up `onHit` (PS data/moves.ts tidyup): remove every Substitute and
        // the entry hazards of both sides (the +1 Atk / Spe is the move's boost).
        if choice.move_id == Choices::TIDYUP && ti == 0 {
            for sd in [Side::One, Side::Two] {
                for sl in 0..2 {
                    state.side_mut(sd).active_slot_mut(sl).substitute_health = 0;
                }
                let sc = &mut state.side_mut(sd).side_conditions;
                sc.spikes = 0;
                sc.toxic_spikes = 0;
                sc.stealth_rock = 0;
                sc.sticky_web = 0;
            }
        }

        // Haze `onHitField`: clear every active pokemon's stat stages.
        if choice.move_id == Choices::HAZE && ti == 0 {
            for sd in [Side::One, Side::Two] {
                for sl in 0..2 {
                    if let Some(ai) = state.side(sd).active()[sl].pokemon_index {
                        state.side_mut(sd).clear_member_boosts(ai);
                    }
                }
            }
        }

        // Guard Swap `onHit`: swap Def / SpD stat stages with the target.
        if choice.move_id == Choices::GUARDSWAP && ti == 0 {
            if let Some(tidx) = state.side(target_side).active()[target_slot].pokemon_index {
                let (d1, s1) = {
                    let sb = &state.side(side).team_boosts[idx as usize];
                    (sb.def, sb.spd)
                };
                let (d2, s2) = {
                    let sb = &state.side(target_side).team_boosts[tidx as usize];
                    (sb.def, sb.spd)
                };
                let sb = &mut state.side_mut(side).team_boosts[idx as usize];
                sb.def = d2;
                sb.spd = s2;
                let tb = &mut state.side_mut(target_side).team_boosts[tidx as usize];
                tb.def = d1;
                tb.spd = s1;
            }
        }

        // Power Swap `onHit`: swap Atk / SpA stat stages with the target.
        // PS data/moves.ts powerswap.
        if choice.move_id == Choices::POWERSWAP && ti == 0 {
            if let Some(tidx) = state.side(target_side).active()[target_slot].pokemon_index {
                let (a1, s1) = {
                    let sb = &state.side(side).team_boosts[idx as usize];
                    (sb.atk, sb.spa)
                };
                let (a2, s2) = {
                    let sb = &state.side(target_side).team_boosts[tidx as usize];
                    (sb.atk, sb.spa)
                };
                let sb = &mut state.side_mut(side).team_boosts[idx as usize];
                sb.atk = a2;
                sb.spa = s2;
                let tb = &mut state.side_mut(target_side).team_boosts[tidx as usize];
                tb.atk = a1;
                tb.spa = s1;
            }
        }

        // Wish (`slotCondition`): fails (onRestart false) while one is pending.
        if choice.move_id == Choices::WISH && ti == 0 {
            if state.side(side).wish[slot].is_some() {
                move_failed = true;
            } else {
                let maxhp = state.side(side).team_member(idx).maxhp;
                state.side_mut(side).wish[slot] = Some((maxhp / 2, 0));
            }
        }

        // Ally Switch `onHit`: swap positions with the (alive) ally.
        if choice.move_id == Choices::ALLYSWITCH && ti == 0 {
            if switched_in_alive(state, side, 1 - slot) {
                let sd = state.side_mut(side);
                sd.active_mut().swap(0, 1);
                sd.ps_order.swap(0, 1);
                // PS `swapPosition`: `runEvent('Swap', ally)` then `('Swap', user)`.
                // Slot conditions (Healing Wish / Lunar Dance) stay with their slot,
                // so the pokemon that arrives in a waiting slot is healed.
                healing_wish_on_switch_in(sd, slot);
                healing_wish_on_switch_in(sd, 1 - slot);
            } else {
                // PS data/moves.ts allyswitch.onHit returns NOT_FAIL after its
                // `-fail`: the hit still runs its two `eachEvent('Update')`s.
                move_failed = true;
                each_event_update(state, rng);
                each_event_update(state, rng);
            }
        }

        // Healing Wish / Lunar Dance `slotCondition`: `addSlotCondition` returns
        // false while the slot still holds an unused one, so the hit does nothing
        // (`damage` false: no hit-loop `eachEvent`); the user still faints
        // (`selfdestruct: 'ifHit'` is checked before). PS sim/battle-actions.ts
        // runMoveEffects, sim/side.ts addSlotCondition.
        if matches!(choice.move_id, Choices::HEALINGWISH | Choices::LUNARDANCE)
            && ti == 0
            && state.side(side).healing_wish_slot[slot]
        {
            move_failed = true;
        }

        // Lock-On `onTryHit` (fails while the user already has it) and `onHit`
        // (`source.addVolatile('lockon', target)`, duration 2). No draws.
        if choice.move_id == Choices::LOCKON && ti == 0 && !move_failed {
            if state.side(side).active()[slot].lock_on.is_some() {
                move_failed = true;
            } else if let Some(tidx) = state.side(target_side).active()[target_slot].pokemon_index {
                state.side_mut(side).active_slot_mut(slot).lock_on = Some((target_side, tidx, 2));
            }
        }

        // Pain Split (`onHit`): both end at the floored average HP, clamped
        // to each one's max HP (`sethp`).
        if choice.move_id == Choices::PAINSPLIT && !move_failed {
            if let Some(tidx) = state.side(target_side).active()[target_slot].pokemon_index {
                let thp = state.side(target_side).team_member(tidx).hp;
                let uhp = state.side(side).team_member(idx).hp;
                let avg = (((thp + uhp) / 2) as i16).max(1);
                let t = state.side_mut(target_side).team_member_mut(tidx);
                t.hp = avg.min(t.maxhp);
                let u = state.side_mut(side).team_member_mut(idx);
                u.hp = avg.min(u.maxhp);
                // Champions `spreadMoveHit` runs EmergencyExit for a single
                // target with `hurtThisTurn + move.totalDamage` (0 here) as the
                // original HP (PS data/mods/champions/scripts.ts).
                let hurt_hp = state.side(target_side).active()[target_slot].hurt_hp;
                if !ee_before.iter().any(|&(s2, l2, _)| s2 == target_side && l2 == target_slot) {
                    ee_before.push((target_side, target_slot, hurt_hp));
                }
            }
        }

        if choice.move_id == Choices::HELPINGHAND {
            // PS `addVolatile('helpinghand')`: a new volatile runs the
            // `TryAddVolatile` event (handler ties); an existing one restarts
            // (`onRestart`, no event). PS sim/pokemon.ts addVolatile.
            if state.side(side).active()[1 - slot].helping_hand_mult <= 1.0 {
                try_add_volatile_ties(state, rng, side, 1 - slot);
            }
            state.side_mut(side).active_slot_mut(1 - slot).helping_hand_mult *= 1.5;
        }

        // PS `tryMoveHit` -> `runEvent('TryHitSide', target)` for allySide /
        // foeSide moves: the Pokemon target is the user (allySide) or the chosen
        // foe (foeSide). Its Magic Bounce holders tie and shuffle before any effect.
        let ps_target = move_target(choice.move_id);
        if ps_target == Tgt::AllySide || ps_target == Tgt::FoeSide {
            let (ps, pslot) = if ps_target == Tgt::AllySide { (side, slot) } else { (target_side, target_slot) };
            crate::doubles::events::run_event_ties(state, rng, crate::doubles::events::Event::TryHitSide, (ps, pslot), Some((side, slot)));
        }

        // Side conditions (Reflect, Light Screen, Spikes, Tailwind, ...).
        // Stone Axe / Ceaseless Edge set their hazard in `onAfterHit` only when
        // `!move.hasSheerForce` (PS data/moves.ts stoneaxe, ceaselessedge): a
        // Sheer Force holder sets none.
        let hazard_sheer_force = matches!(choice.move_id, Choices::STONEAXE | Choices::CEASELESSEDGE)
            && state.side(side).team_member(idx).ability == Abilities::SHEERFORCE;
        if let Some(sc) = choice.side_condition.as_ref().filter(|_| !hazard_sheer_force) {
            let cs = if sc.target == MoveTarget::User {
                side
            } else if sc.target == MoveTarget::Ally {
                side
            } else {
                target_side
            };
            // Magic Bounce `onAllyTryHitSide`: a reflectable move aimed at the
            // holder's side is used back on the user's side instead.
            // PS `tryMoveHit` runs `TryHitSide` on the first foe (`targets[0]`,
            // a Pokemon), and the handler returns early when that foe is
            // semi-invulnerable (`target.isSemiInvulnerable()`), so no bounce.
            // TODO(PS parity): single-target reflectable moves, Mold Breaker.
            let reference_semi_invulnerable = (0..2)
                .find(|&sl| state.side(cs).active()[sl].pokemon_index.is_some())
                .is_some_and(|sl| ally_semi_invulnerable(state, cs, sl));
            let bouncer = if choice.flags.reflectable && cs != side && !reference_semi_invulnerable {
                (0..2).find(|&sl| {
                    state.side(cs).slot_pokemon(sl).map(|p| p.hp > 0 && p.ability == Abilities::MAGICBOUNCE).unwrap_or(false)
                })
            } else {
                None
            };
            if let Some(bsl) = bouncer {
                let bidx = state.side(cs).active()[bsl].pokemon_index.unwrap();
                extra_order.push(order_entry(cs, bsl, state.side(cs).species_names[bidx as usize], choice.move_id));
                if !apply_side_condition(state, side, sc.condition) {
                    move_failed = true;
                }
            } else {
                // Stone Axe / Ceaseless Edge add their hazard in `onAfterHit`:
                // an existing hazard does not fail the attack.
                if !apply_side_condition(state, cs, sc.condition) && choice.category == MoveCategory::Status {
                    move_failed = true;
                }
                // quickguard / wideguard `onHitSide`: `source.addVolatile('stall')`.
                // PS sim/battle-actions.ts runMoveEffects runs HitSide even when
                // addSideCondition failed (the guard was already up).
                if matches!(choice.move_id, Choices::QUICKGUARD | Choices::WIDEGUARD) {
                if state.side(side).active()[slot].stall.is_none() {
                    try_add_volatile_ties(state, rng, side, slot);
                }
                let a = state.side_mut(side).active_slot_mut(slot);
                a.stall = match a.stall {
                    None => Some((3, 2)),
                    Some((c, _)) => Some((if c < 729 { c * 3 } else { c }, 2)),
                };
                }
            }
        }

        // Court Change (`onHitField`): swap the listed side conditions between
        // the two sides. (PS list: mist, lightscreen, reflect, spikes,
        // safeguard, tailwind, toxicspikes, stealthrock, stickyweb,
        // auroraveil, luckychant, ...; pledges / guard / crafty shield are
        // not swapped.)
        if choice.move_id == Choices::COURTCHANGE {
            let a = state.side(side).side_conditions.clone();
            let b = state.side(side.other()).side_conditions.clone();
            state.side_mut(side).side_conditions = crate::state::SideConditions {
                mist: b.mist,
                light_screen: b.light_screen,
                reflect: b.reflect,
                spikes: b.spikes,
                safeguard: b.safeguard,
                tailwind: b.tailwind,
                toxic_spikes: b.toxic_spikes,
                stealth_rock: b.stealth_rock,
                sticky_web: b.sticky_web,
                aurora_veil: b.aurora_veil,
                lucky_chant: b.lucky_chant,
                ..a
            };
            state.side_mut(side.other()).side_conditions = crate::state::SideConditions {
                mist: a.mist,
                light_screen: a.light_screen,
                reflect: a.reflect,
                spikes: a.spikes,
                safeguard: a.safeguard,
                tailwind: a.tailwind,
                toxic_spikes: a.toxic_spikes,
                stealth_rock: a.stealth_rock,
                sticky_web: a.sticky_web,
                aurora_veil: a.aurora_veil,
                lucky_chant: a.lucky_chant,
                ..b
            };
        }

        // Weather / terrain / pseudo-weather.
        if weather_id != 0 {
            let w = match weather_id {
                1 => Weather::RAIN,
                2 => Weather::SUN,
                3 => Weather::SAND,
                4 => Weather::HAIL,
                5 => Weather::SNOW,
                _ => Weather::NONE,
            };
            // PS weather `onFieldStart` / `setWeather`: fails if already up.
            // Chilly Reception (`selfSwitch`) still counts as having done
            // something: PS `moveHit` sets `didSomething = true` for a
            // `selfSwitch` user that can switch, so the failed snow does not
            // stop the switch (no `-fail`).
            if state.weather.weather_type == w {
                if choice.move_id != Choices::CHILLYRECEPTION {
                    move_failed = true;
                }
            } else {
                state.weather = crate::state::StateWeather {
                    weather_type: w,
                    turns_remaining: 5,
                };
                // `setWeather` -> `eachEvent('WeatherChange')`.
                let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
            }
        }
        if choice.terrain_id != 0 {
            let t = match choice.terrain_id {
                1 => crate::engine::state::Terrain::ELECTRICTERRAIN,
                2 => crate::engine::state::Terrain::PSYCHICTERRAIN,
                3 => crate::engine::state::Terrain::MISTYTERRAIN,
                4 => crate::engine::state::Terrain::GRASSYTERRAIN,
                _ => crate::engine::state::Terrain::NONE,
            };
            // PS terrain `onFieldStart` / `setTerrain`: fails if already up.
            if state.terrain.terrain_type == t {
                move_failed = true;
            } else {
                state.terrain = crate::state::StateTerrain {
                    terrain_type: t,
                    turns_remaining: 5,
                };
                // `setTerrain` -> `eachEvent('TerrainChange')`.
                refresh_mimicry(state);
                let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
            }
        }
        if choice.pseudo_weather_id == 1 {
            // `onFieldRestart` removes Trick Room when used again.
            state.trick_room = if state.trick_room.active {
                crate::state::StateTrickRoom { active: false, turns_remaining: 0 }
            } else {
                crate::state::StateTrickRoom { active: true, turns_remaining: 5 }
            };
        }
        if choice.move_id == Choices::GRAVITY {
            // No `onFieldRestart`: using it again fails.
            if state.gravity.active {
                move_failed = true;
            } else {
                state.gravity = crate::state::StateTrickRoom { active: true, turns_remaining: 5 };
                crate::doubles::damage::set_gravity(true);
                // `onFieldStart`: Fly / Bounce charges end (volatile removed,
                // queued move cancelled). PS data/moves.ts gravity.condition.
                // TODO(PS parity): Sky Drop, Telekinesis, `-activate` log.
                for gs in [Side::One, Side::Two] {
                    for gsl in 0..2 {
                        let a = state.side_mut(gs).active_slot_mut(gsl);
                        if matches!(a.charging, Some((Choices::FLY | Choices::BOUNCE, _))) {
                            a.charging = None;
                            state.queue_cancel.push((gs, gsl));
                        }
                    }
                }
            }
        }
        if choice.move_id == Choices::MAGICROOM {
            // Same toggle; duration 5 (7 with Persistent).
            // TODO(PS parity): item suppression (`ignoringItem`).
            state.magic_room = if state.magic_room.active {
                crate::state::StateTrickRoom { active: false, turns_remaining: 0 }
            } else {
                crate::state::StateTrickRoom { active: true, turns_remaining: 5 }
            };
        }
        if choice.move_id == Choices::FAIRYLOCK {
            // PS data/moves.ts fairylock: `addPseudoWeather` fails while it is up.
            // Duration 2. The trap (`onTrapPokemon`) is not modeled: PS only
            // offers switches that are legal, so the fuzzer never sends one.
            // TODO(PS parity): Fairy Lock trap (`tryTrap`) on switch-out.
            if state.fairy_lock.active {
                move_failed = true;
            } else {
                state.fairy_lock = crate::state::StateTrickRoom { active: true, turns_remaining: 2 };
            }
        }
        if choice.move_id == Choices::WONDERROOM {
            state.wonder_room_seen = false;
            // Same toggle; duration 5 (7 with Persistent).
            state.wonder_room = if state.wonder_room.active {
                crate::state::StateTrickRoom { active: false, turns_remaining: 0 }
            } else {
                crate::state::StateTrickRoom { active: true, turns_remaining: 5 }
            };
        }
        if choice.pseudo_weather_id == 2 {
            state.side_mut(side).side_conditions.tailwind = 4;
        }

        // Secondaries whose PS effect is custom `onHit` code (or that the
        // upstream table approximates differently): same draws as PS.
        // Sheer Force `onModifyMove`: the secondaries are deleted (no rolls).
        let sheer_force = state.side(side).team_member(idx).ability == Abilities::SHEERFORCE && has_secondaries(choice);
        // Shield Dust `onModifySecondaries` (PS data/abilities.ts shielddust,
        // breakable): the target's non-`self` secondaries are filtered out, so
        // they make no rolls.
        let shield_dust = (side, slot) != (target_side, target_slot)
            && hit_target_pi
                .map(|ti| state.side(target_side).team_member(ti))
                .map(|p| {
                    p.ability == Abilities::SHIELDDUST
                        && !crate::engine::abilities::ignores_target_ability(
                            state.side(side).team_member(idx),
                            choice,
                            &p.ability,
                        )
                })
                .unwrap_or(false);
        let custom_secondary = if sheer_force || shield_dust { true } else { match choice.move_id {
            Choices::TRIATTACK | Choices::DIRECLAW => {
                // PS `secondary.onHit`: `this.sample([...])` -> random(3).
                let chance = if choice.move_id == Choices::TRIATTACK { 20 } else { 30 };
                if rng.label("secondary").random(100) < chance {
                    let pick = rng.label("sample").random(3) as usize;
                    let st = if choice.move_id == Choices::TRIATTACK {
                        [PokemonStatus::BURN, PokemonStatus::PARALYZE, PokemonStatus::FREEZE][pick]
                    } else {
                        [PokemonStatus::POISON, PokemonStatus::PARALYZE, PokemonStatus::SLEEP][pick]
                    };
                    // The status comes from the secondary's `onHit`, so the
                    // effect Safeguard sees is that secondary, not the move:
                    // Infiltrator does not get past it (PS data/moves.ts
                    // safeguard.onSetStatus `effect.infiltrates`).
                    if state.side(target_side).side_conditions.safeguard > 0
                        && (target_side, target_slot) != (side, slot)
                        && state.side(side).team_member(idx).ability == Abilities::INFILTRATOR
                    {
                        set_status_event_tie(state, rng, target_side, target_slot, st, target_side != side, true);
                    } else {
                        try_set_status(state, rng, target_side, target_slot, st, side, idx);
                    }
                }
                true
            }
            // Its 30% roll is the base-power draw in `calculate_damage`.
            Choices::FICKLEBEAM => true,
            Choices::SPARKLINGARIA => {
                // secondary { chance 100, volatileStatus 'sparklingaria' }:
                // `addVolatile` fails on a fainted target.
                let _ = rng.label("secondary").random(100);
                if switched_in_alive(state, target_side, target_slot) && !sparkling_marked.contains(&(target_side, target_slot)) {
                    sparkling_marked.push((target_side, target_slot));
                }
                true
            }
            Choices::SALTCURE => {
                let _ = rng.label("secondary").random(100);
                if switched_in_alive(state, target_side, target_slot) {
                    state.side_mut(target_side).active_slot_mut(target_slot).salt_cure = true;
                }
                true
            }
            Choices::THROATCHOP => {
                // secondary { chance 100, onHit: addVolatile('throatchop') } (duration 2).
                let _ = rng.label("secondary").random(100);
                if switched_in_alive(state, target_side, target_slot)
                    && state.side(target_side).active()[target_slot].substitute_health <= 0
                    && state.side(target_side).active()[target_slot].throat_chop == 0
                {
                    state.side_mut(target_side).active_slot_mut(target_slot).throat_chop = 2;
                }
                true
            }
            // TODO(PS parity): trapping (Spirit Shackle), Throat Chop's sound
            // lock, Eerie Spell's PP cut, Psychic Noise's Heal Block. The
            // roll is consumed; hazards for Ceaseless Edge / Stone Axe are
            // applied via `side_condition` above.
            Choices::SPIRITSHACKLE
            | Choices::EERIESPELL
            | Choices::PSYCHICNOISE
            | Choices::CEASELESSEDGE
            | Choices::STONEAXE => {
                let _ = rng.label("secondary").random(100);
                true
            }
            _ => false,
        } };

        // Secondary effects (flinch/burn/poison/boosts after a hit).
        // Sheer Force deleted the secondaries outright (PS data/abilities.ts
        // sheerforce.onModifyMove), so Shield Dust has no `self` entry to keep.
        if let Some(secs) = choice.secondaries.as_ref().filter(|_| !custom_secondary || (shield_dust && !sheer_force)) {
            for sec in secs.iter() {
                if shield_dust && sec.target != MoveTarget::User {
                    continue;
                }
                if !rng.label("secondary").random_chance(sec.chance as u32, 100) {
                    continue;
                }
                let (ts, tslot) = if sec.target == MoveTarget::User {
                    (side, slot)
                } else if sec.target == MoveTarget::Ally {
                    (side, ally_slot)
                } else {
                    (target_side, target_slot)
                };
                match &sec.effect {
                    Effect::Flinch => {
                        // Focus Punch's volatile blocks flinch (`onTryAddVolatile`).
                        // Inner Focus `onTryAddVolatile`: flinch is blocked.
                        // TODO(PS parity): Mold Breaker.
                        let inner_focus = state
                            .side(ts)
                            .slot_pokemon(tslot)
                            .map(|p| p.ability == Abilities::INNERFOCUS)
                            .unwrap_or(false);
                        if !state.side(ts).active()[tslot].focus_punch && !inner_focus {
                            state.side_mut(ts).active_slot_mut(tslot).flinched = true;
                        }
                    }
                    Effect::Boost(b) => {
                        if let Some(tidx) = state.side(ts).active()[tslot].pokemon_index {
                            let b = if (ts, tslot) != (side, slot) { mirror_armor_bounce(state, side, slot, ts, tidx, b) } else { b.clone() };
                            apply_boosts_from(state, ts, tidx, &b, ts != side, (ts, tslot) != (side, slot));
                        }
                    }
                    Effect::Status(st) => {
                        // PS `secondary.status` uses `trySetStatus`, which passes the
                        // current status: a statused target (a frozen one is only thawed
                        // after the secondaries) makes setStatus a no-op, with no
                        // immunity check and no SetStatus event (sim/pokemon.ts setStatus).
                        let statused = state
                            .side(ts)
                            .slot_pokemon(tslot)
                            .map(|p| p.status != PokemonStatus::NONE)
                            .unwrap_or(false);
                        if let (false, Some(tidx)) = (statused, state.side(ts).active()[tslot].pokemon_index) {
                            set_status_event_tie(state, rng, ts, tslot, *st, ts != side, true);
                            if !status_immune(state, ts, tslot, *st, ts != side, side, slot) {
                                set_status(state, rng, ts, tidx, *st);
                                // Synchronize also answers an ally's move (PS
                                // synchronize.onAfterSetStatus: `source === target` only).
                                maybe_synchronize(state, ts, tidx, side, idx, *st);
                            }
                        }
                    }
                    Effect::Heal(amount) => {
                        if let Some(tidx) = state.side(ts).active()[tslot].pokemon_index {
                            let s = state.side_mut(ts);
                            let pm = s.team_member_mut(tidx);
                            let amt = (pm.maxhp as f32 * amount).round() as i16;
                            pm.hp = (pm.hp + amt).min(pm.maxhp);
                        }
                    }
                    Effect::VolatileStatus(PokemonVolatileStatus::FLINCH) => {
                        // Focus Punch's volatile blocks flinch (`onTryAddVolatile`).
                        let inner_focus = state
                            .side(ts)
                            .slot_pokemon(tslot)
                            .map(|p| p.ability == Abilities::INNERFOCUS)
                            .unwrap_or(false);
                        if !state.side(ts).active()[tslot].focus_punch && !inner_focus {
                            state.side_mut(ts).active_slot_mut(tslot).flinched = true;
                        }
                    }
                    Effect::VolatileStatus(PokemonVolatileStatus::CONFUSION) => {
                        if !safeguard_blocks_confusion(state, side, slot, idx, ts, tslot) {
                            let min = if choice.move_id == Choices::AXEKICK { 3 } else { 2 };
                            try_confuse_from(state, rng, ts, tslot, min);
                        }
                    }
                    Effect::VolatileStatus(vv) => {
                        if *vv == PokemonVolatileStatus::PROTECT {
                            state.side_mut(ts).active_slot_mut(tslot).protected = true;
                            state.side_mut(ts).active_slot_mut(tslot).protect_kind = 0;
                        }
                        // Syrup Bomb condition: duration 4, source = the user
                        // (no refresh while present). PS data/moves.ts syrupbomb.
                        if *vv == PokemonVolatileStatus::SYRUPBOMB
                            && state.side(ts).active()[tslot].syrup_bomb.is_none()
                            && state.side(ts).slot_pokemon(tslot).map(|p| p.hp > 0).unwrap_or(false)
                        {
                            state.side_mut(ts).active_slot_mut(tslot).syrup_bomb = Some((side, idx, 4));
                        }
                        // TODO(PS parity): other secondary volatiles.
                    }
                    Effect::IfStatsRaised { status } => {
                        // The chance roll was consumed above; the effect is
                        // gated on the target having raised a stat this turn
                        // (PS `statsRaisedThisTurn`).
                        if state.side(ts).active()[tslot].stats_raised_this_turn {
                            // PS `target.trySetStatus` (Burning Jealousy): the status
                            // immunity check applies (a Fire type can't be burned).
                            if let Some(st) = status {
                                let st = *st;
                                if let Some(tidx) = state.side(ts).active()[tslot].pokemon_index {
                                    if !status_immune(state, ts, tslot, st, ts != side, side, slot) {
                                        set_status(state, rng, ts, tidx, st);
                                        maybe_synchronize(state, ts, tidx, side, idx, st);
                                    }
                                }
                            }
                            // Alluring Voice: the confusion volatile (draws its duration).
                            if status.is_none()
                                && !safeguard_blocks_confusion(state, side, slot, idx, ts, tslot)
                            {
                                try_confuse(state, rng, ts, tslot);
                            }
                        }
                    }
                    Effect::RemoveItem => {
                        // TODO(PS parity): item removal.
                    }
                }
            }
        }

        // `thawsTarget` (Scald, Scorching Sands, ...): `frz.onAfterMoveSecondary`
        // cures a frozen target after the move's secondaries. No draw.
        if crate::engine::mechanics::THAWS_TARGET.contains(&choice.move_id) && has_target {
            if let Some(tidx) = state.side(target_side).active()[target_slot].pokemon_index {
                let pm = state.side_mut(target_side).team_member_mut(tidx);
                if pm.status == PokemonStatus::FREEZE && pm.hp > 0 {
                    pm.status = PokemonStatus::NONE;
                    pm.sleep_turns = 0;
                }
            }
        }

        // Stench `onModifyMove` (priority -1): a damaging move without a flinch
        // secondary gets one (chance 10), rolled after the move's own entries.
        // PS data/abilities.ts stench.
        // TODO(PS parity): Sheer Force order, moves with custom flinch (Fake Out).
        if state.side(side).team_member(idx).ability == Abilities::STENCH
            && choice.category != MoveCategory::Status
            && !sheer_force
            && !shield_dust
            && !(side, slot).eq(&(target_side, target_slot))
            && !choice.secondaries.as_ref().is_some_and(|ss| {
                ss.iter().any(|e| matches!(e.effect, Effect::Flinch | Effect::VolatileStatus(PokemonVolatileStatus::FLINCH)))
            })
        {
            if rng.label("secondary").random_chance(10, 100) {
                let inner_focus = state
                    .side(target_side)
                    .slot_pokemon(target_slot)
                    .map(|p| p.ability == Abilities::INNERFOCUS)
                    .unwrap_or(false);
                if !state.side(target_side).active()[target_slot].focus_punch && !inner_focus {
                    state.side_mut(target_side).active_slot_mut(target_slot).flinched = true;
                }
            }
        }

        // PS `forceSwitch` step (after secondaries): Roar / Whirlwind /
        // Dragon Tail / Circle Throw flag the target to be dragged out.
        // TODO(PS parity): Ingrain, Commander, Dynamax block it too.
        if choice.force_switch && !move_failed {
            if let Some(d) = state.side(target_side).slot_pokemon(target_slot) {
                // Ingrain `onDragOut`: blocks phazing.
                let blocked = matches!(d.ability, Abilities::SUCTIONCUPS | Abilities::GUARDDOG)
                    || state.side(target_side).active()[target_slot].ingrain;
                let attacker_alive = state.side(side).slot_pokemon(slot).map(|p| p.hp > 0).unwrap_or(false);
                if d.hp > 0 && attacker_alive && !blocked && !possible_switches(state, target_side).is_empty() {
                    state.side_mut(target_side).slot_scratch_mut(target_slot).drag_out = true;
                }
            }
        }

        // Focus Punch `condition.onHit`: a damaging move breaks the focus.
        if hit_damage.is_some() && state.side(target_side).active()[target_slot].focus_punch {
            state.side_mut(target_side).slot_scratch_mut(target_slot).lost_focus = true;
        }

        // Beak Blast `condition.onHit` (PS data/moves.ts beakblast): a contact hit
        // on the charging user burns the attacker (`source.trySetStatus('brn',
        // target)`). The effect is the volatile, not a move; no draw.
        if hit_damage.is_some() && choice.flags.contact && hit_target_beak {
            if let Some(bidx) = hit_target_pi {
                let attacker_alive = state.side(side).team_member(idx).hp > 0;
                if attacker_alive {
                    try_set_status_from_ability(state, rng, side, slot, PokemonStatus::BURN, target_side, bidx);
                }
            }
        }

        // Recorded for the DamagingHit pass below (PS `spreadMoveHit` runs the
        // secondaries of every target before DamagingHit for any of them).
        hit_damage_by[ti] = hit_damage;
        hit_pi_by[ti] = hit_target_pi;
      }
      // DamagingHit pass (PS `spreadMoveHit`, after the secondaries of every target).
      // PS collects the DamagingHit handlers once, before any of them runs, so the
      // attacker's Poison Touch still fires after a target's Mummy has already
      // changed the attacker's ability (PS `runEvent` keeps the captured handler).
      let attacker_poison_touch = state.side(side).team_member(idx).ability == Abilities::POISONTOUCH;
      for (ti, &(hts, htsl)) in hit_targets.iter().enumerate() {
        target_side = hts;
        target_slot = htsl;
        let hit_damage = hit_damage_by[ti];
        let hit_target_pi = hit_pi_by[ti];
        // Double Shock / Burn Up `self.onHit`: the user loses the type
        // (`setType(types.map(t => t === X ? '???' : t))`).
        // A substitute hit counts (PS `runMoveEffects` runs `self.onHit`).
        if matches!(choice.move_id, Choices::DOUBLESHOCK | Choices::BURNUP) && ti == 0 && (hit_damage.is_some() || sub_hit) {
            let lose = if choice.move_id == Choices::DOUBLESHOCK { PokemonType::ELECTRIC } else { PokemonType::FIRE };
            if state.side(side).team_member(idx).hp > 0 {
                let orig = state.side(side).team_member(idx).types;
                let f = |x: PokemonType| if x == lose { PokemonType::TYPELESS } else { x };
                let s = state.side_mut(side);
                s.team_member_mut(idx).types = (f(orig.0), f(orig.1));
                if s.active()[slot].type_change.is_none() {
                    s.slot_scratch_mut(slot).type_change = Some(orig);
                }
            }
        }
        // Steel Roller `onHit` / `onAfterSubDamage`: `field.clearTerrain()` (no
        // user-alive check; a fainted target still counts). `onHit` runs in
        // `runMoveEffects`, before DamagingHit, so a Seed Sower target sets
        // Grassy Terrain again. PS data/moves.ts steelroller.
        if choice.move_id == Choices::STEELROLLER
            && ti == 0
            && (hit_damage.is_some() || sub_hit)
            && state.terrain.terrain_type != crate::engine::state::Terrain::NONE
        {
            state.terrain = crate::state::StateTerrain {
                terrain_type: crate::engine::state::Terrain::NONE,
                turns_remaining: 0,
            };
            refresh_mimicry(state);
            let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
        }
        // DamagingHit (after secondaries, PS `spreadMoveHit`).
        if hit_damage.is_some() {
            // Fire-type damaging move `onDamagingHit` (data/conditions.ts frz): a frozen
            // target thaws. PS runs it after the secondaries, so a secondary status on
            // the frozen target fails first. No draw. `cureStatus` returns when the
            // target has no HP left, so a target KO'd by the hit stays frozen.
            if choice.move_type == PokemonType::FIRE && choice.category != MoveCategory::Status {
                if let Some(didx) = state.side(target_side).active()[target_slot].pokemon_index {
                    let pm = state.side_mut(target_side).team_member_mut(didx);
                    if pm.status == PokemonStatus::FREEZE && pm.hp > 0 {
                        pm.status = PokemonStatus::NONE;
                        pm.sleep_turns = 0;
                    }
                }
            }
            // Glaive Rush `self.volatileStatus`.
            if choice.move_id == Choices::GLAIVERUSH && ti == 0 && state.side(side).team_member(idx).hp > 0 {
                state.side_mut(side).active_slot_mut(slot).glaive_rush = true;
            }
            contact_abilities(state, rng, side, slot, idx, target_side, target_slot, choice);
            cursed_body(state, rng, side, slot, target_side, target_slot, choice);
            // Weak Armor, Steam Engine, ... (shared with the singles engine).
            if let Some(didx) = state.side(target_side).active()[target_slot].pokemon_index {
                let ability = state.side(target_side).team_member(didx).ability;
                if let Some(b) = crate::engine::mechanics::on_damaging_hit_boosts(ability, choice) {
                    apply_boosts(state, target_side, didx, &b, false);
                }
                // Electromorphosis `onDamagingHit` / Wind Power (wind moves only):
                // `addVolatile('charge')`. PS data/abilities.ts.
                if (ability == Abilities::ELECTROMORPHOSIS
                    || (ability == Abilities::WINDPOWER && choice.flags.wind))
                    && state.side(target_side).team_member(didx).hp > 0
                {
                    state.side_mut(target_side).active_slot_mut(target_slot).charge = true;
                }
                // Stamina `onDamagingHit`: +1 Def (PS data/abilities.ts stamina).
                if ability == Abilities::STAMINA && state.side(target_side).team_member(didx).hp > 0 {
                    apply_boosts(
                        state,
                        target_side,
                        didx,
                        &StatBoosts { defense: 1, ..Default::default() },
                        false,
                    );
                }
            }
            // Sand Spit `onDamagingHit` (PS data/abilities.ts sandspit): the
            // holder sets sand even when this hit KO'd it (the hit's holder).
            if let Some(pi) = hit_target_pi {
                if state.side(target_side).team_member(pi).ability == Abilities::SANDSPIT {
                    field_setting_damaging_hit(state, rng, Abilities::SANDSPIT);
                }
            }
            // Gooey / Tangling Hair (`onDamagingHit`, also when the hit KO'd the
            // holder): a contact hit lowers the attacker's Speed by 1 stage.
            // PS data/abilities.ts gooey; Long Reach removes contact.
            let holder = state.side(target_side).active()[target_slot].pokemon_index.or_else(|| {
                state
                    .unresolved_faints
                    .iter()
                    .find(|(s, sl, _)| *s == target_side && *sl == target_slot)
                    .map(|&(_, _, i)| i)
            });
            if let Some(h) = holder {
                let ability = state.side(target_side).team_member(h).ability;
        if ability == Abilities::SEEDSOWER {
            field_setting_damaging_hit(state, rng, Abilities::SEEDSOWER);
        }
                if matches!(ability, Abilities::GOOEY | Abilities::TANGLINGHAIR)
                    && choice.flags.contact
                    && state.side(side).team_member(idx).ability != Abilities::LONGREACH
                {
                    // Mirror Armor on the attacker bounces the drop back onto the
                    // holder (PS `this.boost` goes through TryBoost).
                    let drop = StatBoosts { speed: -1, ..Default::default() };
                    let drop = mirror_armor_bounce(state, target_side, target_slot, side, idx, &drop);
                    apply_boosts_from(state, side, idx, &drop, target_side != side, true);
                }
            }
            maybe_wandering_spirit(state, side, slot, target_side, target_slot, choice, rng);
            // PS `runEvent('DamagingHit')` runs the target's `onDamagingHit` handlers
            // before the attacker's `onSourceDamagingHit` (same order and priority,
            // so the stable insertion order decides; sim/battle.ts findEventHandlers).
            // Poison Touch `onSourceDamagingHit`: a contact hit has a 3/10 chance to
            // poison the target (Shield Dust blocks). PS data/abilities.ts poisontouch.
            if attacker_poison_touch
                && choice.flags.contact
                && (side, slot) != (target_side, target_slot)
            {
                // The hit's target as it was at the hit (a KO already cleared the
                // slot, and PS still rolls for the hit and still sees its Shield
                // Dust: `target.hasAbility` on the still-active fainted target).
                let shield = hit_target_pi
                    .map(|ti| state.side(target_side).team_member(ti).ability == Abilities::SHIELDDUST)
                    .unwrap_or(false);
                if !shield && hit_target_pi.is_some() {
                    if rng.label("ability").random_chance(3, 10) {
                        try_set_status(state, rng, target_side, target_slot, PokemonStatus::POISON, side, idx);
                    }
                }
            }
        }
      }
        // PS `hitStepMoveHitLoop`: `eachEvent('Update')` after every hit that
        // did something. Field moves (`tryMoveHit`) have no hit loop; neither
        // do `target: 'all'` moves (Haze), which PS runs through `tryMoveHit`
        // (sim/battle-actions.ts useMoveInner).
        // Spread boost-only status move where no target changed (PS
        // `hitStepMoveHitLoop`: every `damage[i]` false -> `return`, no eachEvent).
        if spread_boost_noop && !drop_landed && !spread_sub_null {
            move_failed = true;
        }
        // PS `spreadMoveHit`: a spread volatile that failed on every target fails the move.
        if spread_volatile_fail && !spread_volatile_ok && !spread_sub_null {
            move_failed = true;
        }
        // PS useMoveInner: `all` / `foeSide` / `allySide` / `allyTeam` moves run
        // `tryMoveHit` (one `moveHit`, no hit loop and no `eachEvent('Update')`).
        if is_field || move_failed || matches!(move_ty, Tgt::All | Tgt::FoeSide | Tgt::AllySide | Tgt::AllyTeam) {
            break;
        }
        hits_done += 1;
        // PS `hitStepMoveHitLoop`: `if (!damage.some(val => !!val || val === 0)) return;`
        // before the `eachEvent` (a spread attack every target of which was
        // immune or missed skips it).
        any_hit_landed |= hit_landed;
        if !is_spread || !is_attack || hit_landed {
            each_event_update(state, rng);
        }
        // Cloud Nine / Air Lock `onEnd` (faint): `eachEvent('WeatherChange')`
        // right after the hit that KO'd the holder. PS data/abilities.ts.
        if weather_suppress_fainted {
            weather_suppress_fainted = false;
            // PS runs it from `faintMessages` (the holder's ability `End`), in
            // `faintQueue` order: the faints queued before the holder are already
            // processed (fainted, out of `getAllActive`), among them the user of a
            // `selfdestruct: 'always'` move (queued in useMoveInner, before the hit).
            // PS sim/battle.ts faintMessages, sim/battle-actions.ts useMoveInner.
            let saved = state.unresolved_faints.clone();
            if let Some(pos) = weather_suppress_holder
                .and_then(|h| saved.iter().position(|&(fs, fsl, _)| (fs, fsl) == h))
            {
                state.unresolved_faints = saved[pos..].to_vec();
            }
            let user_queued_first = matches!(
                choice.move_id,
                Choices::EXPLOSION | Choices::SELFDESTRUCT | Choices::MISTYEXPLOSION
            );
            let user_hp = state.side(side).team_member(idx).hp;
            if user_queued_first {
                state.side_mut(side).team_member_mut(idx).hp = 0;
            }
            each_event_update(state, rng);
            if user_queued_first {
                state.side_mut(side).team_member_mut(idx).hp = user_hp;
            }
            state.unresolved_faints = saved;
        }
        // PS `hitStepMoveHitLoop` (after the per-hit eachEvent): `if (!pokemon.hp &&
        // targets.length === 1) break;`. A user KO'd by the hit (Rough Skin, Rocky Helmet,
        // ...) takes no further hits. Spread and smart-target moves have several targets.
        if !is_spread && !smart_hits && state.side(side).team_member(idx).hp <= 0 {
            break 'hit_loop;
        }
    }

    // PS `hitStepMoveHitLoop` -> `afterMoveSecondaryEvent(targets)` (after the
    // hit loop): `runEvent('AfterMoveSecondary')` speed-sorts the handlers of
    // the hit targets. Skipped under Sheer Force (`suppressingSecondaries`).
    if any_hit_landed && is_attack && !after_hit_targets.is_empty() {
        let sheer = state.side(side).team_member(idx).ability == Abilities::SHEERFORCE && has_secondaries(choice);
        if !sheer {
            after_move_secondary_ties(state, rng, &after_hit_targets);
        }
    }

    // Sparkling Aria `onAfterMove` (PS data/moves.ts sparklingaria): a hit
    // target that got the volatile (or any, when it hit several) is cured of
    // its burn, unless the user fainted or has Sheer Force.
    // TODO(PS parity): `runEvent('AfterMove')` order against other handlers.
    if choice.move_id == Choices::SPARKLINGARIA
        && state.side(side).team_member(idx).hp > 0
        && state.side(side).team_member(idx).ability != Abilities::SHEERFORCE
    {
        let mut hit: Vec<(Side, usize)> = Vec::new();
        for &t in &after_hit_targets {
            if !hit.contains(&t) {
                hit.push(t);
            }
        }
        let many = hit.len() > 1;
        for &(ts, tsl) in &hit {
            if (ts, tsl) == (side, slot) {
                continue;
            }
            if !(many || sparkling_marked.contains(&(ts, tsl))) {
                continue;
            }
            if let Some(p) = state.side_mut(ts).slot_pokemon_mut(tsl) {
                if p.hp > 0 && p.status == PokemonStatus::BURN {
                    p.status = PokemonStatus::NONE;
                }
            }
        }
    }

    if move_failed && !not_fail_result {
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
    }

    // Scale Shot & co.: the user's stat change after all hits. PS resolves the
    // hit loop's faints first (`hitStepMoveHitLoop` -> `resolveFaints`), then
    // `useMoveInner` runs `selfBoost`, whose `boost` fails once the foe side has
    // no Pokemon left (`foePokemonLeft`).
    if (choice.multihit != 0 || choice.move_id == Choices::CLANGINGSCALES) && is_attack && hits_done > 0 {
        if let Some(b) = choice.boost.as_ref().filter(|b| b.target == MoveTarget::User) {
            let foe_left = state
                .side(side.other())
                .team
                .iter()
                .any(|m| m.id != crate::pokemon::PokemonName::NONE && m.hp > 0);
            if foe_left {
                apply_boosts(state, side, idx, &b.boosts, false);
            }
        }
    }

    // Berserk: +1 SpA when the move took the holder from above to at/below 50%.
    for &(bs, bl, before) in &berserk_before {
        if let Some(d) = state.side(bs).slot_pokemon(bl) {
            if d.hp > 0 && (d.hp as i32) * 2 <= d.maxhp as i32 && (before as i32) * 2 > d.maxhp as i32 {
                let bidx = state.side(bs).active()[bl].pokemon_index.unwrap();
                apply_boosts(state, bs, bidx, &StatBoosts { special_attack: 1, ..Default::default() }, false);
            }
        }
    }

    // Emergency Exit / Wimp Out `onEmergencyExit` (champions), PS `tryMoveHit`
    // after the hit loop: crossing 50% HP flags the holder to switch out after
    // the action (allies included: PS runs it for every damaged target).
    // PS data/mods/champions/abilities.ts emergencyexit.
    for &(ts, tsl, before) in &ee_before {
        emergency_exit_check(state, ts, tsl, before);
    }

    // Healing Wish / Lunar Dance (slot-condition moves) make their first hit-loop
    // `eachEvent('Update')` while the user still counts as active; the faint and
    // the closing `eachEvent` come after it (see the selfdestruct block).
    if hits_done == 0 && matches!(choice.move_id, Choices::HEALINGWISH | Choices::LUNARDANCE) && !move_failed {
        each_event_update(state, rng);
    }


    // Rapid Spin `onAfterHit` / `onAfterSubDamage` (not with Sheer Force): clears
    // the user's Leech Seed, its side's hazards and any binding. PS data/moves.ts.
    // Mortal Spin: the same `onAfterHit` (its psn is a plain secondary).
    if matches!(choice.move_id, Choices::RAPIDSPIN | Choices::MORTALSPIN)
        && (hits_done > 0 || sub_hit)
        && state.side(side).team_member(idx).ability != Abilities::SHEERFORCE
    {
        let sc = &mut state.side_mut(side).side_conditions;
        sc.spikes = 0;
        sc.toxic_spikes = 0;
        sc.stealth_rock = 0;
        sc.sticky_web = 0;
        let a = state.side_mut(side).active_slot_mut(slot);
        a.leech_seed = None;
        a.partial_trap = None;
    }

    // Ice Spinner `onAfterHit` / `onAfterSubDamage` (user alive):
    // `field.clearTerrain()` -> `eachEvent('TerrainChange')`. PS data/moves.ts.
    if choice.move_id == Choices::ICESPINNER
        && (hits_done > 0 || (sub_hit && state.side(side).team_member(idx).hp > 0))
        && state.terrain.terrain_type != crate::engine::state::Terrain::NONE
    {
        state.terrain = crate::state::StateTerrain {
            terrain_type: crate::engine::state::Terrain::NONE,
            turns_remaining: 0,
        };
        refresh_mimicry(state);
        let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
    }

    // Jaw Lock `onHit` (PS data/moves.ts jawlock): the user and the target both
    // gain the `trapped` volatile (no-op while present; the target must be alive).
    if choice.move_id == Choices::JAWLOCK && hits_done > 0 {
        let user_idx = state.side(side).active()[slot].pokemon_index;
        let target_pi = state.side(target_side).active()[target_slot].pokemon_index;
        let user_alive = state.side(side).slot_pokemon(slot).is_some_and(|p| p.hp > 0);
        let target_alive = state.side(target_side).slot_pokemon(target_slot).is_some_and(|p| p.hp > 0);
        // Each side gains `trapped` (no-op while present), linked to the other.
        // PS `addVolatile` with a `linkedStatus` fails when the link's source
        // has no HP (sim/pokemon.ts addVolatile): a KO'd target traps nobody.
        if let (Some(_), true) = (user_idx, user_alive && target_alive) {
            let a = state.side_mut(side).active_slot_mut(slot);
            if !a.trapped {
                a.trapped = true;
                a.trapper = target_pi.map(|ti| (target_side, ti));
            }
        }
        if let (Some(_), true) = (target_pi, target_alive && user_alive) {
            let a = state.side_mut(target_side).active_slot_mut(target_slot);
            if !a.trapped {
                a.trapped = true;
                a.trapper = user_idx.map(|ui| (side, ui));
            }
        }
    }

    // PS `resolveFaints` right after the hit loop (before the recoil).
    resolve_faints(state, rng, |_, _| true);

    // Recoil (after all hits): round(damage * fraction), min 1.
    if total_damage > 0 && choice.move_id == Choices::STRUGGLE {
        // PS applyRecoilDamage `struggleRecoil`: round(baseMaxhp / 4), min 1,
        // independent of the damage dealt.
        let maxhp = state.side(side).team_member(idx).maxhp;
        let recoil = (((maxhp as f32) / 4.0).round().max(1.0)) as i16;
        apply_direct_damage(state, side, slot, recoil);
    } else if total_damage > 0 {
        // Rock Head `onDamage` (recoil, not Struggle) and Magic Guard
        // `onDamage` (non-move damage) cancel recoil. PS data/abilities.ts.
        let no_recoil = matches!(
            state.side(side).team_member(idx).ability,
            Abilities::ROCKHEAD | Abilities::MAGICGUARD
        );
        // `mindBlownRecoil` / `chloroblastRecoil`: round(maxhp / 2) after the hits
        // (PS sim/battle-actions.ts recoilDamage).
        // `mindBlownRecoil` damages with the move's condition as effect (not
        // 'recoil'), so Rock Head does not block it; Magic Guard does.
        let magic_guard = state.side(side).team_member(idx).ability == Abilities::MAGICGUARD;
        let blocked = match choice.move_id {
            Choices::STEELBEAM | Choices::MINDBLOWN => magic_guard,
            _ => no_recoil,
        };
        if matches!(choice.move_id, Choices::STEELBEAM | Choices::MINDBLOWN | Choices::CHLOROBLAST) && !blocked {
            let maxhp = state.side(side).team_member(idx).maxhp;
            apply_damage(state, side, slot, ((maxhp as f32 / 2.0).round() as i16).max(1));
        }
        if let Some(frac) = choice.recoil {
            if !no_recoil {
                let recoil = ((total_damage as f32) * frac).round().max(1.0) as i16;
                apply_damage(state, side, slot, recoil);
                if !real_hit {
                    state.unresolved_faints.retain(|&(fs, fsl, _)| (fs, fsl) != (side, slot));
                }
            }
        }
    }

    // The user's own Emergency Exit: PS runs `runEvent('EmergencyExit', pokemon)`
    // for the user after each `spreadMoveHit` (Rough Skin / Rocky Helmet-style
    // damage) and after its recoil (sim/battle-actions.ts spreadMoveHit,
    // applyRecoilDamage).
    // TODO(PS parity): per hit (a drain heal between hits), crash damage (MoveFail).
    if hits_done > 0 {
        if let Some(sl) = (0..2).find(|&sl| state.side(side).active()[sl].pokemon_index == Some(idx)) {
            emergency_exit_check(state, side, sl, user_hp_before);
        }
    }

    // PS `useMoveInner`: `selfdestruct: 'always'` faints the user (HP 0); the
    // faint is resolved right after the hit loop, so the loop's closing
    // `eachEvent` no longer counts the user.
    if choice.selfdestruct {
        let hp = state.side(side).slot_pokemon(slot).map(|p| p.hp).unwrap_or(0);
        let before = state.faint_log.len();
        apply_damage(state, side, slot, hp);
        // PS queues the user's faint before its victims' (`faintQueue` order).
        if state.faint_log.len() > before && before >= faint_mark {
            let f = state.faint_log.pop().unwrap();
            state.faint_log.insert(faint_mark, f);
        }
        // ...and resolved right after the loop, before its closing `eachEvent`.
        resolve_faints(state, rng, |fs, fsl| (fs, fsl) == (side, slot));
        // Healing Wish / Lunar Dance: the closing `eachEvent` (PS ~1062) runs after
        // the faint is resolved, so the fainted user no longer counts in the sort.
        if matches!(choice.move_id, Choices::HEALINGWISH | Choices::LUNARDANCE) && !move_failed {
            each_event_update(state, rng);
        }
    }

    // End of PS's hit loop: one more `eachEvent('Update')` (after recoil), if a
    // target took something (`if (!damage.some(...)) return damage;`).
    if hits_done > 0 && (!is_spread || !is_attack || any_hit_landed) {
        each_event_update(state, rng);
    }

    // lockedmove `onAfterMove` (PS runMove, after useMove's hit loop): removed
    // when duration is 1; `onEnd` then confuses the user unless trueDuration > 1.
    // PS data/conditions.ts lockedmove.
    // PS runs `onAfterMove` on the pokemon, not the slot: an Ally Switch in
    // this move may have moved the user to the other slot.
    let holder = (0..2).find(|&sl| state.side(side).active()[sl].pokemon_index == Some(idx)).unwrap_or(slot);
    // PS runs the `AfterMove` sort before the lock's `onAfterMove` removes it.
    if state.side(side).active()[holder].locked_move.is_some() {
        after_move_handler_ties(state, rng, side, idx, true);
        state.tl.after_move_tie_drawn.set(true);
    }
    if let Some((1, t)) = state.side(side).active()[holder].locked_move {
        state.side_mut(side).active_slot_mut(holder).locked_move = None;
        if t <= 1 {
            try_confuse(state, rng, side, holder);
        }
    }

    // Speed Swap `onHit` (data/moves.ts speedswap): swap the two pokemon's
    // `storedStats.spe`. The originals are restored on switch-in (`setSpecies`).
    if choice.move_id == Choices::SPEEDSWAP && !move_failed && (target_side, target_slot) != (side, slot) {
        if let (Some(tidx), Some(uidx)) = (
            state.side(target_side).active()[target_slot].pokemon_index,
            state.side(side).active()[slot].pokemon_index,
        ) {
            let t_sp = state.side(target_side).team_member(tidx).speed;
            let u_sp = state.side(side).team_member(uidx).speed;
            if state.side(target_side).saved_speed[tidx as usize].is_none() {
                state.side_mut(target_side).saved_speed[tidx as usize] = Some(t_sp);
            }
            if state.side(side).saved_speed[uidx as usize].is_none() {
                state.side_mut(side).saved_speed[uidx as usize] = Some(u_sp);
            }
            state.side_mut(target_side).team_member_mut(tidx).speed = u_sp;
            state.side_mut(side).team_member_mut(uidx).speed = t_sp;
        }
    }

    // Sparkling Aria `onAfterMove` (PS data/moves.ts sparklingaria): every hit
    // target other than the user that is still active is cured of its burn when
    // the move hit several targets, or when its secondary added the
    // `sparklingaria` volatile (Shield Dust filters it out on a single target).
    if choice.move_id == Choices::SPARKLINGARIA
        && state.side(side).team_member(idx).hp > 0
        && !(state.side(side).team_member(idx).ability == Abilities::SHEERFORCE && has_secondaries(choice))
    {
        let mut hit: Vec<(Side, usize)> = Vec::new();
        for &t in &after_hit_targets {
            if !hit.contains(&t) {
                hit.push(t);
            }
        }
        let n = hit.len();
        for (ts, tsl) in hit {
            if (ts, tsl) == (side, slot) {
                continue;
            }
            let Some(ti) = state.side(ts).active()[tsl].pokemon_index else { continue };
            let tp = state.side(ts).team_member(ti);
            if tp.hp <= 0 || tp.status != PokemonStatus::BURN {
                continue;
            }
            let shield_dust = tp.ability == Abilities::SHIELDDUST
                && !crate::engine::abilities::ignores_target_ability(
                    state.side(side).team_member(idx),
                    choice,
                    &tp.ability,
                );
            if n > 1 || !shield_dust {
                state.side_mut(ts).team_member_mut(ti).status = PokemonStatus::NONE;
            }
        }
    }

    // Fell Stinger `onAfterMoveSecondarySelf`: +3 Atk to the user when the
    // move's target is down after its hits (data/moves.ts fellstinger).
    if choice.move_id == Choices::FELLSTINGER
        && hits_done > 0
        && state.side(target_side).slot_pokemon(target_slot).map(|p| p.hp <= 0).unwrap_or(true)
    {
        apply_boosts(state, side, idx, &StatBoosts { attack: 3, ..Default::default() }, false);
    }

    // Self-destruct moves faint the user after dealing damage.
    // Healing Wish / Lunar Dance: add the slot condition before the user
    // faints (PS data/moves.ts healingwish `slotCondition`).
    if matches!(choice.move_id, Choices::HEALINGWISH | Choices::LUNARDANCE) {
        state.side_mut(side).healing_wish_slot[slot] = true;
    }

    // Spit Up: `onAfterMove` removes the Stockpile volatile.
    if choice.move_id == Choices::SPITUP {
        end_stockpile(state, side, slot);
    }

    // Recharge flag (Hyper Beam, Solar Beam): cannot act next turn.
    if choice.flags.recharge {
        state.side_mut(side).active_slot_mut(slot).recharging = true;
    }

    // Pivot moves (PS `selfSwitch`, battle-actions.ts `spreadMoveHit`): a
    // move that did something flags its still-alive user to switch out when
    // a replacement exists (`canSwitch`). run_battle_fuzz performs the switch
    // right after this action, from the controller's switch answer.
    // TODO(PS parity): Parting Shot (only if a drop landed), Baton Pass
    // (copies boosts), Shed Tail (Substitute), Chilly Reception (snow).
    // PS `moveHit`: `selfSwitch` sets `didSomething` whenever the target was hit
    // (a 0-damage hit such as a Disguise bust still switches the user out).
    let self_switch = match choice.move_id {
        Choices::UTURN | Choices::VOLTSWITCH | Choices::FLIPTURN => total_damage > 0 || real_hit,
        Choices::TELEPORT => true,
        // Parting Shot: only if a stat drop landed (`selfSwitch` is deleted otherwise),
        // or the target has Mirror Armor (it never deletes `selfSwitch`).
        Choices::PARTINGSHOT => drop_landed || parting_shot_held,
        // Chilly Reception (`selfSwitch`): the snow is set and the user switches out.
        Choices::CHILLYRECEPTION => !move_failed,
        _ => false,
    };
    // Revival Blessing (`selfSwitch: true`): flags the switch request even
    // without a healthy replacement (battle-actions.ts `spreadMoveHit`).
    if choice.move_id == Choices::REVIVALBLESSING && state.side(side).active()[slot].reviving {
        state.side_mut(side).slot_scratch_mut(slot).force_switch = true;
    }
    let shed = choice.move_id == Choices::SHEDTAIL && state.side(side).active()[slot].shed_tailing;
    let baton = choice.move_id == Choices::BATONPASS;
    if baton && state.side(side).find_replacement().is_none() {
        // Baton Pass `onHit`: `-fail` without a replacement.
        state.side_mut(side).slot_scratch_mut(slot).move_result_this_turn = Some(false);
    }
    if (self_switch || baton || shed)
        && state.side(side).slot_pokemon(slot).map(|p| p.hp > 0).unwrap_or(false)
        && state.side(side).find_replacement().is_some()
    {
        state.side_mut(side).slot_scratch_mut(slot).force_switch = true;
        state.side_mut(side).slot_scratch_mut(slot).baton_passing = baton;
    }
    // PS data/abilities.ts magician `onAfterMoveSecondarySelf` (run by useMoveInner
    // after the secondaries): with no item and no switch flag, a non-Status move
    // `speedSort`s its `hitTargets` (a draw per speed tie). The item steal
    // (`takeItem`) is not modeled: the harness gives no items.
    // TODO(PS parity): Magician item steal from the hit targets.
    if choice.category != MoveCategory::Status
        && choice.move_id != Choices::FLING
        && !self_switch
        && state.side(side).team_member(idx).ability == Abilities::MAGICIAN
        && state.side(side).team_member(idx).item == crate::engine::items::Items::NONE
        && !magician_targets.is_empty()
    {
        let mut speeds: Vec<i32> = magician_targets
            .iter()
            .map(|&(ts, tpi)| state.side(ts).speed_cache[tpi as usize] as i32)
            .collect();
        speed_sort_by(&mut speeds, rng, |a: &i32, b: &i32| b - a);
    }

    push_order(move_order);
    move_order.extend(extra_order);
    move_order.extend(copy_order);
}

/// Copycat / Sleep Talk `useMove` of the copied move, run as an external call
/// from the `onHit` (PS: before the hit's `eachEvent`, so its draws come first).
fn run_copied_move(
    state: &mut DoublesState,
    rng: &mut Gen5Rng,
    action: &Action,
    m: Choices,
    copy_order: &mut Vec<OrderEntry>,
    will_act: bool,
    queue: &[Action],
) {
    let Some(mc) = MOVES.get(&m).cloned() else { return };
    let called_type = mc.move_type;
    let mut sub = action.clone();
    sub.choice = mc;
    sub.target = 0;
    sub.target_loc = 0;
    // PS `useMoveInner` (called by Copycat's `onHit`) has no chosen target, so
    // `getRandomTarget` runs on the move's base data before `onModifyMove`.
    // Curse's `onModifyMove` then sets 'self' for a non-Ghost user, but the
    // random foe was already drawn. PS data/moves.ts curse.onModifyMove.
    if m == Choices::CURSE {
        let user_types = state.side(action.side).team_member(action.pokemon_index).types;
        let user_ghost = user_types.0 == PokemonType::GHOST
            || user_types.1 == PokemonType::GHOST
            || state.side(action.side).active()[action.slot].added_type == PokemonType::GHOST;
        if !user_ghost {
            let foes: Vec<usize> = (0..2)
                .filter(|&sl| state.side(action.side.other()).slot_pokemon(sl).is_some_and(|p| p.hp > 0))
                .collect();
            if !foes.is_empty() {
                let _ = rng.label("retarget").random(foes.len() as u32);
            }
            sub.target_loc = -(action.slot as i8 + 1);
        }
    }
    state.side_mut(action.side).slot_scratch_mut(action.slot).external_call = true;
    let n = copy_order.len();
    let outer_no_target = state.tl.no_target_fail.replace(false);
    execute_action_inner(state, rng, &sub, copy_order, will_act, queue);
    let copied_no_target = state.tl.no_target_fail.replace(outer_no_target);
    // The copy may swap positions (Ally Switch): the flag travels with the
    // `ActiveSlot`, so clear it on every slot of the side, not only `action.slot`.
    for sl in 0..2 {
        state.side_mut(action.side).slot_scratch_mut(sl).external_call = false;
    }
    // PS `useMoveInner` sets `setActiveMove(copied)`, so the copied move's
    // `clearActiveMove` records `battle.lastMove` (PS battle.ts clearActiveMove).
    state.copy_called = true;
    if copy_order.len() > n {
        state.last_move = Some(m);
    }
    // The called move's own `onMoveFail` (crash damage, ...) runs inside its
    // `useMoveInner`, not in the caller's.
    // PS sim/battle-actions.ts useMoveInner, data/moves.ts jumpkick.
    if copy_order.len() > n && !copied_no_target {
        if let Some(sl) = (0..2).find(|&sl| state.side(action.side).active()[sl].pokemon_index == Some(action.pokemon_index)) {
            move_fail_self_damage(state, action.side, sl, &MOVES[&m]);
        }
    }
    // PS runMove: `if (this.battle.activeMove) move = this.battle.activeMove;`
    // makes the called move the one `AfterMove` sees, so Charge's `onAfterMove`
    // (data/moves.ts charge.condition) is used up by a called Electric move.
    if called_type == PokemonType::ELECTRIC && m != Choices::CHARGE {
        if let Some(sl) = (0..2).find(|&sl| state.side(action.side).active()[sl].pokemon_index == Some(action.pokemon_index)) {
            state.side_mut(action.side).active_slot_mut(sl).charge = false;
        }
    }
    // A called Destiny Bond adds its volatile like a direct one (PS data/moves.ts
    // destinybond `volatileStatus`), unless `onPrepareHit` failed it.
    if m == Choices::DESTINYBOND && copy_order.len() > n {
        if let Some(sl) = (0..2).find(|&sl| state.side(action.side).active()[sl].pokemon_index == Some(action.pokemon_index)) {
            if state.side(action.side).active()[sl].move_result_this_turn != Some(false) {
                state.side_mut(action.side).active_slot_mut(sl).destinybond = true;
            }
        }
    }
}

/// PS `flags.bypasssub` (champions data): the move ignores a Substitute.
fn bypasses_substitute(c: &Choice) -> bool {
    const IDS: &[&str] = &[
    "confide",
    "conversion2",
    "curse",
    "disable",
    "disarmingvoice",
    "dragoncheer",
    "echoedvoice",
    "encore",
    "growl",
    "heartswap",
    "holdhands",
    "howl",
    "hyperspacefury",
    "hyperspacehole",
    "junglehealing",
    "mimic",
    "playnice",
    "relicsong",
    "sketch",
    "supersonic",
    "afteryou",
    "alluringvoice",
    "aromaticmist",
    "attract",
    "bestow",
    "boomburst",
    "bugbuzz",
    "chatter",
    "clangingscales",
    "clangoroussoulblaze",
    "coaching",
    "defog",
    "destinybond",
    "eeriespell",
    "fairylock",
    "foresight",
    "gearup",
    "grasswhistle",
    "grudge",
    "guardswap",
    "haze",
    "healbell",
    "helpinghand",
    "hypervoice",
    "imprison",
    "instruct",
    "lifedew",
    "magneticflux",
    "mefirst",
    "metalsound",
    "miracleeye",
    "nobleroar",
    "odorsleuth",
    "overdrive",
    "partingshot",
    "perishsong",
    "powder",
    "powerswap",
    "psychicnoise",
    "psychup",
    "reflecttype",
    "roar",
    "roleplay",
    "round",
    "screech",
    "sing",
    "skillswap",
    "snarl",
    "snatch",
    "snore",
    "sparklingaria",
    "spectralthief",
    "speedswap",
    "spite",
    "taunt",
    "teatime",
    "torchsong",
    "torment",
    "uproar",
    "whirlwind",
    ];
    let id = c.move_id.to_string().to_lowercase();
    IDS.contains(&id.as_str())
}

fn choice_is_attack(c: &Choice) -> bool {
    matches!(c.category, MoveCategory::Physical | MoveCategory::Special)
}

/// Whether the move has PS `secondaries` (the custom-handled ones included).
fn has_secondaries(choice: &Choice) -> bool {
    choice.secondaries.is_some()
        || matches!(
            choice.move_id,
            Choices::TRIATTACK
                | Choices::DIRECLAW
                | Choices::SALTCURE
                | Choices::SPIRITSHACKLE
                | Choices::THROATCHOP
                | Choices::EERIESPELL
                | Choices::PSYCHICNOISE
                | Choices::CEASELESSEDGE
                | Choices::STONEAXE
        )
}

/// Base power after the PS `BasePower` event (`getDamage`): the dynamic
/// base power, then every `onBasePower` handler chained into one 4096-based
/// modifier (`chainModify`), applied once with `modify`. Handlers run in
/// priority order: Tough Claws (21) before Helping Hand (10).
/// PS `basePowerCallback` for moves whose power comes from the user's slot
/// state; 0.0 when the move has none (the caller then uses the other
/// callbacks / the move's base power). The result is the base the
/// `onBasePower` modifiers (Helping Hand, Technician, ...) act on.
/// PS data/moves.ts beatup / spitup / storedpower / powertrip.
fn slot_base_power_callback(state: &DoublesState, side: Side, slot: usize, att_slot: &ActiveSlot, choice: &Choice) -> f32 {
    match choice.move_id {
        // A Beat Up hit's power can linger in the slot after the move (a later
        // Future Sight from the same slot must not use it).
        Choices::BEATUP => att_slot.bp_fixed,
        Choices::SPITUP => 100.0 * att_slot.stockpile as f32,
        Choices::STOREDPOWER | Choices::POWERTRIP => {
            let b = state.side(side).slot_boosts(slot).cloned().unwrap_or_default();
            crate::engine::mechanics::stored_power([b.atk, b.def, b.spa, b.spd, b.spe, b.acc, b.eva])
        }
        _ => 0.0,
    }
}

fn base_power_with_modifiers(
    choice: &Choice,
    attacker: &crate::state::Pokemon,
    defender: &crate::state::Pokemon,
    helping_hand_mult: f32,
    // `onBasePower` modifiers from the user's slot (Charge, Lash Out):
    // (priority, 4096-based factor), chained with the others.
    slot_mods: &[(i32, i64)],
    speeds: (i32, i32),
    terrain: crate::engine::state::Terrain,
    defender_semi_invulnerable: bool,
    ate_boosted: bool,
    fallen: u8,
    steely_spirits: u8,
    avalanche_boosted: bool,
    total_fainted: u32,
    hit_number: u32,
    bp_doubled: bool,
    // Beat Up's per-hit base power (`bp_fixed`, 0 when unused): the base that
    // the modifiers (Technician, ...) act on, as PS `basePowerCallback` runs
    // before `onBasePower`.
    fixed_bp: f32,
    // Rivalry: 1 same gender, -1 opposite, 0 when either is genderless.
    rivalry: i8,
    // PS `isGrounded` for the attacker and the defender (Gravity, Smack Down,
    // Magnet Rise, Levitate, Air Balloon, Iron Ball): terrain and Gravity checks.
    attacker_grounded: bool,
    defender_grounded: bool,
    state: &DoublesState,
) -> f32 {
    let dynamic = dynamic_base_power(
        choice,
        attacker,
        Some(defender),
        speeds,
        avalanche_boosted,
        state.terrain.terrain_type == crate::engine::state::Terrain::ELECTRICTERRAIN && defender_grounded,
        total_fainted,
        hit_number,
    );
    let bp = if fixed_bp > 0.0 {
        fixed_bp
    } else if dynamic > 0.0 {
        dynamic
    } else {
        choice.base_power
    };
    // ragefist.basePowerCallback: `min(350, 50 + 50 * timesAttacked)`. Each
    // damaging hit taken since switching in adds an `attacked_by` entry (the
    // champions mod resets the counter in `clearVolatile`).
    // TODO(PS parity): PS also counts 0-damage hits (`typeof damage === 'number'`).
    let bp = if choice.move_id == Choices::RAGEFIST {
        (50.0 + 50.0 * attacker.attacked_by.len() as f32).min(350.0)
    } else {
        bp
    };
    // A doubling `basePowerCallback` (Payback, Assurance, Stomping Tantrum,
    // Temper Flare) returns the doubled base, which the `onBasePower`
    // modifiers then act on (PS sim/battle-actions.ts getDamage order).
    let (bp, dynamic) = if bp_doubled {
        (bp * 2.0, if dynamic > 0.0 { dynamic * 2.0 } else { dynamic })
    } else {
        (bp, dynamic)
    };
    // (onBasePowerPriority, 4096-based modifier); PS runs higher priority
    // first. Values from data/abilities.ts / data/moves.ts `onBasePower`.
    // Technician sees the base power after `basePowerCallback` (a doubled
    // Assurance / Payback is 120, not 60).
    let tech_bp = bp;
    let mut mods: Vec<(i32, i64)> = Vec::new();
    let f = &choice.flags;
    let physical = choice.category == MoveCategory::Physical;
    let special = choice.category == MoveCategory::Special;
    match attacker.ability {
        // Checks the base power after higher-priority modifiers; it has the
        // highest priority, so that is the plain base power.
        Abilities::TECHNICIAN if tech_bp <= 60.0 => mods.push((30, 6144)),
        Abilities::IRONFIST if f.punch => mods.push((23, 4915)),
        Abilities::RECKLESS if choice.recoil.is_some() || choice.crash.is_some() => mods.push((23, 4915)),
        Abilities::TOUGHCLAWS if f.contact => mods.push((21, 5325)),
        // Analytic `onBasePower`: no other active has a move left in the queue.
        Abilities::ANALYTIC if state.tl.analytic_no_move.get() => mods.push((21, 5325)),
        Abilities::STRONGJAW if f.bite => mods.push((19, 6144)),
        Abilities::MEGALAUNCHER if f.pulse => mods.push((19, 6144)),
        Abilities::SHARPNESS if f.slicing => mods.push((19, 6144)),
        Abilities::TOXICBOOST
            if physical && matches!(attacker.status, PokemonStatus::POISON | PokemonStatus::TOXIC) =>
        {
            mods.push((19, 6144))
        }
        Abilities::FLAREBOOST if special && attacker.status == PokemonStatus::BURN => mods.push((19, 6144)),
        Abilities::PUNKROCK if f.sound => mods.push((7, 5325)),
        // supremeoverlord.onBasePower (priority 21): `powMod[fallen]`.
        Abilities::SUPREMEOVERLORD if fallen > 0 => {
            mods.push((21, [4096, 4506, 4915, 5325, 5734, 6144][fallen.min(5) as usize]))
        }
        // sheerforce.onBasePower (priority 21), when it removed the secondaries.
        Abilities::SHEERFORCE if has_secondaries(choice) => mods.push((21, 5325)),
        // rivalry.onBasePower (priority 24): x1.25 same gender, x0.75 opposite.
        Abilities::RIVALRY if rivalry > 0 => mods.push((24, 5120)),
        Abilities::RIVALRY if rivalry < 0 => mods.push((24, 3072)),
        // sandforce.onBasePower (priority 21): x1.3 for Rock / Ground / Steel
        // in a sandstorm (`isWeather`: not while Cloud Nine / Air Lock suppress it).
        Abilities::SANDFORCE
            if effective_weather(state) == Weather::SAND
                && matches!(choice.move_type, PokemonType::ROCK | PokemonType::GROUND | PokemonType::STEEL) =>
        {
            mods.push((21, 5325))
        }
        _ => {}
    }
    // dryskin.onSourceBasePower (priority 17): x1.25 for Fire moves.
    if defender.ability == Abilities::DRYSKIN && choice.move_type == PokemonType::FIRE {
        mods.push((17, 5120));
    }
    if ate_boosted {
        mods.push((23, 4915));
    }
    // steelyspirit.onAllyBasePower (priority 22): x1.5 per holder (user or ally).
    if choice.move_type == PokemonType::STEEL {
        for _ in 0..steely_spirits {
            mods.push((22, 6144));
        }
    }
    // Facade (`onBasePower`): x2 when the user has a non-sleep status.
    if choice.move_id == Choices::FACADE
        && matches!(
            attacker.status,
            PokemonStatus::BURN
                | PokemonStatus::POISON
                | PokemonStatus::TOXIC
                | PokemonStatus::FREEZE
                | PokemonStatus::PARALYZE
        )
    {
        mods.push((0, 8192));
    }
    // solarbeam / solarblade `onBasePower` (default priority): x0.5 in
    // rain / sandstorm / hail / snow (`pokemon.effectiveWeather()`, so Cloud
    // Nine / Air Lock suppression and the user's Mega Sol apply).
    if matches!(choice.move_id, Choices::SOLARBEAM | Choices::SOLARBLADE) {
        let weather = effective_weather(state);
        let weakening = matches!(
            weather,
            Weather::RAIN
                | Weather::HEAVYRAIN
                | Weather::SAND
                | Weather::HAIL
                | Weather::SNOW
        );
        let suppressed = [Side::One, Side::Two]
            .iter()
            .flat_map(|s| (0..2).map(move |sl| (*s, sl)))
            .any(|(s, sl)| {
                state
                    .side(s)
                    .slot_pokemon(sl)
                    .map(|p| {
                        p.hp > 0
                            && (p.ability == Abilities::CLOUDNINE || p.ability == Abilities::AIRLOCK)
                    })
                    .unwrap_or(false)
            });
        if weakening && !suppressed && attacker.ability != Abilities::MEGASOL {
            mods.push((0, 2048));
        }
    }
    // expandingforce / mistyexplosion `onBasePower`: x1.5 in Psychic / Misty
    // Terrain for a grounded user; gravapple: x1.5 under Gravity.
    let move_boost = match choice.move_id {
        Choices::EXPANDINGFORCE => terrain == crate::engine::state::Terrain::PSYCHICTERRAIN && attacker_grounded,
        Choices::MISTYEXPLOSION => terrain == crate::engine::state::Terrain::MISTYTERRAIN && attacker_grounded,
        Choices::GRAVAPPLE => state.gravity.active,
        _ => false,
    };
    if move_boost {
        mods.push((0, 6144));
    }
    if let Some(m) = crate::engine::mechanics::terrain_base_power_modifier(terrain, choice, attacker_grounded, defender_grounded, defender_semi_invulnerable) {
        mods.push((6, m));
    }
    mods.extend_from_slice(slot_mods);
    // Helping Hand stacks one x1.5 per use (`helping_hand_mult` = 1.5^n).
    let mut hh = helping_hand_mult;
    while hh > 1.01 {
        mods.push((10, 6144));
        hh /= 1.5;
    }
    if mods.is_empty() {
        return if choice.move_id == Choices::RAGEFIST || fixed_bp > 0.0 {
            bp
        } else if bp_doubled && dynamic <= 0.0 {
            // No callback value: the doubled choice base power.
            choice.base_power * 2.0
        } else {
            dynamic
        };
    }
    mods.sort_by_key(|&(prio, _)| -prio);
    let mut chain: i64 = 4096;
    for (_, m) in mods {
        chain = (chain * m + 2048) >> 12;
    }
    let modified = ((bp as i64) * chain + 2047) / 4096;
    modified.max(1) as f32
}

/// Dynamic base power from PS `basePowerCallback` (0.0 = move has no
/// callback; use `choice.base_power`). Reversal / Flail scale with the
/// user's remaining HP; Hard Press / Crush Grip / Wring Out with the
/// target's.
/// PS `pokemon.isGrounded()` (sim/pokemon.ts), in its order: Gravity, Ingrain
/// and Smack Down ground the holder first; then Flying / Levitate / Air Balloon
/// (`Pokemon::is_grounded`) and Magnet Rise make it airborne. The one grounded
/// check for doubles: pass `state.gravity.active`, never add it at the call site.
fn slot_grounded(p: &crate::state::Pokemon, a: &ActiveSlot, gravity: bool) -> bool {
    gravity || a.ingrain || a.smackdown || (a.magnet_rise == 0 && p.is_grounded())
}

/// `slot_grounded` for the pokemon in a slot (false for an empty slot).
fn ps_grounded(state: &DoublesState, side: Side, slot: usize) -> bool {
    state
        .side(side)
        .slot_pokemon(slot)
        .is_some_and(|p| slot_grounded(p, &state.side(side).active()[slot], state.gravity.active))
}

/// Grassy Glide `onModifyPriority`: +1 in Grassy Terrain for a grounded user
/// (PS data/moves.ts grassyglide).
/// Ability `onModifyPriority`. PS data/abilities.ts prankster (+1 on Status
/// moves), galewings (+1 on Flying moves at full HP).
fn ability_priority_boost(p: &crate::state::Pokemon, choice: &Choice) -> i8 {
    match p.ability {
        Abilities::PRANKSTER => (choice.category == MoveCategory::Status) as i8,
        Abilities::GALEWINGS => (choice.move_type == PokemonType::FLYING && p.hp == p.maxhp) as i8,
        _ => 0,
    }
}

/// Expanding Force in Psychic Terrain from a grounded user: `onModifyMove` makes
/// it an `allAdjacentFoes` move (PS data/moves.ts expandingforce).
fn expanding_force_spreads(state: &DoublesState, side: Side, slot: usize, choice: &Choice) -> bool {
    choice.move_id == Choices::EXPANDINGFORCE
        && state.terrain.terrain_type == crate::engine::state::Terrain::PSYCHICTERRAIN
        && state.side(side).slot_pokemon(slot).is_some_and(|p| {
            slot_grounded(p, &state.side(side).active()[slot], state.gravity.active)
        })
}

fn grassy_glide_boost(state: &DoublesState, p: &crate::state::Pokemon, choice: &Choice) -> i8 {
    (choice.move_id == Choices::GRASSYGLIDE
        && state.terrain.terrain_type == crate::engine::state::Terrain::GRASSYTERRAIN
        // TODO(PS parity): use `slot_grounded` (Ingrain / Smack Down / Magnet Rise);
        // the callers pass no slot.
        && (p.is_grounded() || state.gravity.active)) as i8
}

/// Residual / weather / status damage: PS `Magic Guard` `onDamage` returns
/// false for any non-Move effect (data/abilities.ts magicguard).
fn apply_indirect_damage(state: &mut DoublesState, side: Side, slot: usize, dmg: i16) -> bool {
    if state.side(side).slot_pokemon(slot).map(|p| p.ability == Abilities::MAGICGUARD).unwrap_or(false) {
        return false;
    }
    apply_damage(state, side, slot, dmg)
}

/// Illusion `onBeforeSwitchIn`: the last unfainted pokemon after the holder in
/// `side.pokemon` order.
fn illusion_disguise(sd: &crate::doubles::state::DoublesSide, idx: PokemonIndex) -> Option<usize> {
    let len = sd.ps_order.iter().filter(|&&t| sd.team[t].id != crate::pokemon::PokemonName::NONE).count();
    let own = sd.ps_order.iter().position(|&t| t == idx as usize).unwrap_or(0);
    let mut i = len;
    while i > own + 1 {
        i -= 1;
        let t = sd.ps_order[i];
        if sd.team[t].hp > 0 {
            return Some(t);
        }
    }
    None
}

/// PS `field.suppressingWeather()`: an active Cloud Nine / Air Lock holder.
/// PS `field.effectiveWeather()` (also what `isWeather` reads): no weather
/// while a Cloud Nine / Air Lock holder is active. Use it for every weather
/// effect; only setting / ending the weather reads the raw weather.
fn effective_weather(state: &DoublesState) -> Weather {
    if weather_suppressed(state) {
        Weather::NONE
    } else {
        state.weather.weather_type
    }
}

fn weather_suppressed(state: &DoublesState) -> bool {
    [Side::One, Side::Two].iter().any(|&s| {
        (0..2).any(|sl| {
            state
                .side(s)
                .slot_pokemon(sl)
                .is_some_and(|p| p.hp > 0 && matches!(p.ability, Abilities::CLOUDNINE | Abilities::AIRLOCK))
        })
    })
}

/// PS `side.totalFainted`: fainted members of the battle team.
fn side_total_fainted(state: &DoublesState, side: Side) -> u32 {
    let sd = state.side(side);
    sd.team
        .iter()
        .enumerate()
        .filter(|(i, p)| p.id != crate::pokemon::PokemonName::NONE && p.hp <= 0 && !sd.species_names[*i].is_empty())
        .count() as u32
}

fn dynamic_base_power(
    choice: &Choice,
    attacker: &crate::state::Pokemon,
    defender: Option<&crate::state::Pokemon>,
    // PS `getStat('spe')` of attacker and defender (boosts + modifiers, no
    // Trick Room): Electro Ball / Gyro Ball.
    speeds: (i32, i32),
    // PS avalanche.basePowerCallback: doubled when the target damaged the
    // user this turn.
    avalanche_boosted: bool,
    // Electric Terrain is active and the target is grounded (PS
    // `target.isGrounded()`, the engine's `slot_grounded` rule): Rising Voltage.
    electric_terrain_grounded: bool,
    // PS `side.totalFainted`: Last Respects.
    total_fainted: u32,
    // PS `move.hit`: Triple Axel / Triple Kick.
    hit_number: u32,
) -> f32 {
    use crate::engine::mechanics as m;
    let (att_spe, def_spe) = speeds;
    let def_hg = |d: &crate::state::Pokemon| {
        let ignore = crate::engine::abilities::ignores_target_ability(attacker, choice, &d.ability);
        m::weight_hg(d, ignore)
    };
    match choice.move_id {
        // PS acrobatics.basePowerCallback: doubled with no item.
        Choices::ACROBATICS => {
            if attacker.item == crate::engine::items::Items::NONE {
                110.0
            } else {
                55.0
            }
        }
        // PS risingvoltage.basePowerCallback: doubled on a grounded target in
        // Electric Terrain.
        Choices::RISINGVOLTAGE => {
            if electric_terrain_grounded && defender.is_some() {
                choice.base_power * 2.0
            } else {
                choice.base_power
            }
        }
        Choices::LASTRESPECTS => m::last_respects_power(total_fainted),
        // PS tripleaxel / triplekick basePowerCallback: 20 (10) * move.hit.
        Choices::TRIPLEAXEL => 20.0 * hit_number as f32,
        Choices::TRIPLEKICK => 10.0 * hit_number as f32,
        Choices::ERUPTION | Choices::WATERSPOUT | Choices::DRAGONENERGY => m::eruption_power(attacker.hp, attacker.maxhp),
        Choices::VENOSHOCK | Choices::BARBBARRAGE => {
            if defender.is_some_and(m::venoshock_boosted) {
                choice.base_power * 2.0
            } else {
                choice.base_power
            }
        }
        Choices::HEX | Choices::INFERNALPARADE => {
            if defender.is_some_and(m::hex_boosted) {
                choice.base_power * 2.0
            } else {
                choice.base_power
            }
        }
        Choices::AVALANCHE if avalanche_boosted => choice.base_power * 2.0,
        Choices::ELECTROBALL => m::electro_ball_power(att_spe, def_spe),
        Choices::GYROBALL => m::gyro_ball_power(att_spe, def_spe),
        Choices::GRASSKNOT | Choices::LOWKICK => m::grass_knot_power(defender.map(def_hg).unwrap_or(1)),
        Choices::HEAVYSLAM | Choices::HEATCRASH => {
            m::heavy_slam_power(m::weight_hg(attacker, false), defender.map(def_hg).unwrap_or(1))
        }
        Choices::REVERSAL | Choices::FLAIL => m::reversal_power(attacker.hp, attacker.maxhp),
        Choices::HARDPRESS => defender.map(|d| m::target_hp_power(100, d.hp, d.maxhp)).unwrap_or(100.0),
        Choices::CRUSHGRIP | Choices::WRINGOUT => {
            defender.map(|d| m::target_hp_power(120, d.hp, d.maxhp)).unwrap_or(120.0)
        }
        _ => 0.0,
    }
}

/// End the Stockpile volatile (PS `stockpile.onEnd`): drop the layers and
/// take back the Def / SpD stages they added.
fn end_stockpile(state: &mut DoublesState, side: Side, slot: usize) {
    let (def, spd) = {
        let a = state.side_mut(side).active_slot_mut(slot);
        let r = (a.stockpile_def, a.stockpile_spd);
        a.stockpile = 0;
        a.stockpile_def = 0;
        a.stockpile_spd = 0;
        r
    };
    if let Some(idx) = state.side(side).active()[slot].pokemon_index {
        if def != 0 || spd != 0 {
            apply_boosts(
                state,
                side,
                idx,
                &StatBoosts {
                    defense: -def,
                    special_defense: -spd,
                    ..Default::default()
                },
                false,
            );
        }
    }
}

/// Set a non-volatile status the way PS `setStatus` -> `onStart` does
/// (champions conditions): sleep lasts `sample([2, 3, 3])` turns, freeze 3,
/// toxic restarts its counter.
fn set_status(state: &mut DoublesState, rng: &mut Gen5Rng, ts: Side, tidx: PokemonIndex, st: PokemonStatus) {
    let s = state.side_mut(ts);
    let pm = s.team_member_mut(tidx);
    pm.status = st;
    match st {
        PokemonStatus::SLEEP => {
            pm.sleep_turns = [2, 3, 3][rng.label("sleep duration").random(3) as usize];
        }
        PokemonStatus::FREEZE => pm.sleep_turns = 3,
        PokemonStatus::TOXIC => s.toxic_stage[tidx as usize] = 1,
        _ => {}
    }
}

/// PS `getConfusionDamage` (sim/battle-actions.ts): 40 BP, the user's own
/// boosted Atk and Def, no crit, one damage roll, at least 1. Wonder Room
/// swaps the Def value for SpD (`calculateStat` swaps the name; the Def stage
/// still applies).
fn confusion_damage(state: &DoublesState, rng: &mut Gen5Rng, side: Side, slot: usize, idx: PokemonIndex) -> i16 {
    let s = state.side(side);
    let p = s.team_member(idx);
    let b = s.slot_boosts(slot).cloned().unwrap_or_default();
    let atk = multiply_boost(b.atk, p.attack).max(1) as i64;
    let base_def = if state.wonder_room.active { p.special_defense } else { p.defense };
    let def = multiply_boost(b.def, base_def).max(1) as i64;
    let level = p.level as i64;
    let base = ((((2 * level / 5 + 2) * 40 * atk) / def) / 50) + 2;
    let base = base % 65536;
    let r = rng.label("damage roll").random(16) as i64;
    ((base * (100 - r)) / 100).max(1) as i16
}

/// Add the confusion volatile (PS `addVolatile('confusion')`): fails without
/// a draw if already confused or Own Tempo; otherwise draws its duration
/// `random(2, 6)`.
// TODO(PS parity): Misty Terrain, Safeguard.
/// Safeguard `onTryAddVolatile` (PS data/moves.ts safeguard): blocks a
/// confusion added by another pokemon's move, unless the move infiltrates.
fn safeguard_blocks_confusion(
    state: &DoublesState,
    side: Side,
    slot: usize,
    idx: PokemonIndex,
    ts: Side,
    tslot: usize,
) -> bool {
    state.side(ts).side_conditions.safeguard > 0
        && (side, slot) != (ts, tslot)
        && state.side(side).team_member(idx).ability != Abilities::INFILTRATOR
}

/// Misty Terrain `onTryAddVolatile` (PS data/conditions.ts): a grounded
/// pokemon can't be confused; a semi-invulnerable one can.
// PS data/moves.ts mistyterrain.onTryAddVolatile: `!target.isGrounded() ||
// target.isSemiInvulnerable()` lets the add through, so Magnet Rise (and
// Gravity / Ingrain / Smackdown as grounding) decide.
fn misty_terrain_blocks_confusion(state: &DoublesState, ts: Side, tslot: usize) -> bool {
    let Some(p) = state.side(ts).slot_pokemon(tslot) else { return false };
    let a = &state.side(ts).active()[tslot];
    state.terrain.terrain_type == crate::engine::state::Terrain::MISTYTERRAIN
        && slot_grounded(p, a, state.gravity.active)
        && !ally_semi_invulnerable(state, ts, tslot)
}

/// PS `addVolatile` -> `runEvent('TryAddVolatile', target, source)` (sim/pokemon.ts
/// addVolatile, sim/battle.ts runEvent): the handlers are speed-sorted before any
/// callback runs, so every exact tie group shuffles (one draw per swap). The
/// handler list depends on the target, not on the volatile being added:
/// - the target's own `onTryAddVolatile`: Focus Punch's volatile (condition,
///   subOrder 2) and abilities Insomnia, Inner Focus, Own Tempo, Vital Spirit,
///   Leaf Guard, Purifying Salt, Shields Down (subOrder 7, holder speed);
/// - `onAllyTryAddVolatile` of every alive active on the target's side, the
///   target included: Aroma Veil, Flower Veil, Sweet Veil (subOrder 7);
/// - Electric / Misty Terrain `onTryAddVolatile` (field, speed 0, subOrder 5);
/// - Safeguard `onTryAddVolatile` on the target's side (side, speed 0, subOrder 4).
/// PS data/abilities.ts (insomnia, innerfocus, owntempo, vitalspirit, leafguard,
/// purifyingsalt, shieldsdown, aromaveil, flowerveil, sweetveil), data/moves.ts
/// (focuspunch, electricterrain, mistyterrain, safeguard).
// TODO(PS parity): the other `addVolatile` callers (flinch, confusion, ...) do
// not draw these ties yet.
fn try_add_volatile_ties(state: &DoublesState, rng: &mut Gen5Rng, ts: Side, tslot: usize) {
    if !state.side(ts).slot_pokemon(tslot).is_some_and(|p| p.hp > 0) {
        return;
    }
    // `runStatusImmunity(status.id)` -> `runEvent('Immunity')` first (Magnet
    // Rise / Dig / ... `onImmunity` handlers). PS sim/pokemon.ts addVolatile.
    crate::doubles::events::run_event_ties(state, rng, crate::doubles::events::Event::Immunity, (ts, tslot), None);
    crate::doubles::events::run_event_ties(state, rng, crate::doubles::events::Event::TryAddVolatile, (ts, tslot), None);
}

fn try_confuse(state: &mut DoublesState, rng: &mut Gen5Rng, ts: Side, tslot: usize) {
    try_confuse_from(state, rng, ts, tslot, 2);
}

/// `try_confuse` with the source's minimum duration: `confusion.onStart` draws
/// `random(min, 6)`, and `min` is 3 when the source is Axe Kick (PS
/// data/conditions.ts confusion.onStart), 2 otherwise.
fn try_confuse_from(state: &mut DoublesState, rng: &mut Gen5Rng, ts: Side, tslot: usize, min: usize) {
    let Some(p) = state.side(ts).slot_pokemon(tslot) else { return };
    if p.hp <= 0 || state.side(ts).active()[tslot].confusion > 0 {
        return;
    }
    // `addVolatile('confusion')`: Immunity / TryAddVolatile handler ties, then
    // Own Tempo / Misty Terrain (`onTryAddVolatile`) block.
    try_add_volatile_ties(state, rng, ts, tslot);
    if state.side(ts).slot_pokemon(tslot).is_some_and(|p| p.ability == Abilities::OWNTEMPO) {
        return;
    }
    if misty_terrain_blocks_confusion(state, ts, tslot) {
        return;
    }
    let t = rng.label("volatile duration").random_range(min, 6) as u8;
    state.side_mut(ts).active_slot_mut(tslot).confusion = t;
}


/// Add the `partiallytrapped` volatile (PS data/conditions.ts): no-op if the
/// target already has it; otherwise `durationCallback` draws `random(5, 7)`.
// TODO(PS parity): Grip Claw (8 turns) and Binding Band (1/6) items.
fn try_partial_trap(
    state: &mut DoublesState,
    rng: &mut Gen5Rng,
    ts: Side,
    tslot: usize,
    src_side: Side,
    src_idx: PokemonIndex,
) {
    let Some(p) = state.side(ts).slot_pokemon(tslot) else { return };
    if p.hp <= 0 || state.side(ts).active()[tslot].partial_trap.is_some() {
        return;
    }
    let d = rng.label("volatile duration").random_range(5, 7) as u8;
    state.side_mut(ts).active_slot_mut(tslot).partial_trap = Some(crate::doubles::state::PartialTrap {
        duration: d,
        source_side: src_side,
        source_idx: src_idx,
        divisor: 8,
    });
}

/// PS `checkMoveBypassesProtect`: does the target's protection stop `choice`?
/// Every `protect` move is stopped by Protect, Spiky Shield and Baneful Bunker.
/// King's Shield (1), Silk Trap (4) and Burning Bulwark (5) call it with
/// `blockStatus = false` (data/moves.ts kingsshield / silktrap / burningbulwark),
/// so Status moves pass through them. Only the PS flow for Status moves is
/// modelled here; Mat Block and Obstruct are not in the engine yet.
fn protect_blocks_move(choice: &Choice, kind: u8) -> bool {
    choice.flags.protect && !(choice.category == MoveCategory::Status && matches!(kind, 1 | 4 | 5))
}

/// PS `trySetStatus` from a secondary `onHit`: fails on an existing status
/// or a type immunity; sleep draws its duration.
/// Contact penalty of a protect variant (PS `onTryHit` of the protect
/// condition, data/moves.ts): Spiky Shield damages the attacker by 1/8,
/// King's Shield / Obstruct / Silk Trap lower its stats, Baneful Bunker /
/// Burning Bulwark status it.
fn protect_contact_penalty(
    state: &mut DoublesState,
    rng: &mut Gen5Rng,
    side: Side,
    slot: usize,
    idx: PokemonIndex,
    target_side: Side,
    target_slot: usize,
) {
    let kind = state.side(target_side).active()[target_slot].protect_kind;
    match kind {
        1 => {
            // PS King's Shield `onTryHit`: `this.boost({atk: -1}, source, target)`,
            // so the drop comes from the protecting pokemon and Mirror Armor
            // (`onTryBoost`) bounces it back onto the protector.
            let drop = StatBoosts { attack: -1, ..Default::default() };
            let drop = mirror_armor_bounce(state, target_side, target_slot, side, idx, &drop);
            apply_boosts_from(state, side, idx, &drop, target_side != side, true);
        }
        2 => {
            // PS spikyshield.condition.onTryHit: `this.damage(...)` from a
            // Condition, so Magic Guard (`onDamage`, non-Move effect) blocks it.
            let maxhp = state.side(side).team_member(idx).maxhp;
            apply_indirect_damage(state, side, slot, (maxhp / 8).max(1));
        }
        3 | 5 => {
            if let Some(tidx) = state.side(target_side).active()[target_slot].pokemon_index {
                let st = if kind == 3 { PokemonStatus::POISON } else { PokemonStatus::BURN };
                // PS Baneful Bunker / Burning Bulwark `onTryHit`: `source.trySetStatus(st, target)`.
                // The protector's Substitute (`onTryPrimaryHit` only) does not stop it;
                // the attacker's Safeguard still does.
                try_set_status_with(state, rng, side, slot, st, target_side, tidx, false, true, false);
            }
        }
        4 => {
            apply_boosts_from(state, side, idx, &StatBoosts { speed: -1, ..Default::default() }, target_side != side, true);
        }
        _ => {}
    }
}

fn try_set_status(
    state: &mut DoublesState,
    rng: &mut Gen5Rng,
    ts: Side,
    tslot: usize,
    st: PokemonStatus,
    src_side: Side,
    src_idx: PokemonIndex,
) {
    try_set_status_with(state, rng, ts, tslot, st, src_side, src_idx, true, true, false);
}

/// PS `setStatus` -> `runEvent('SetStatus')` (sim/pokemon.ts): the target's
/// `onSetStatus` abilities and the `onAllySetStatus` abilities of its side
/// (Flower Veil, Sweet Veil, Pastel Veil) are handlers of that event. Abilities
/// are all priority 0 and subOrder 7, so holders of equal speed tie and each
/// tie group draws one shuffle. Reached only when the status is not blocked by
/// a type / `Immunity`-event check (Sun vs freeze, Magma Armor) and not blocked
/// by a foe's Substitute. Misty / Electric Terrain, Safeguard and Uproar are
/// SetStatus handlers too, but on other keys, so they do not tie here.
fn set_status_event_tie(
    state: &DoublesState,
    rng: &mut Gen5Rng,
    ts: Side,
    tslot: usize,
    st: PokemonStatus,
    from_opponent: bool,
    substitute_blocks: bool,
) {
    if st == PokemonStatus::NONE {
        return;
    }
    let Some(target) = state.side(ts).slot_pokemon(tslot) else { return };
    if substitute_blocks && from_opponent && state.side(ts).active()[tslot].substitute_health > 0 {
        return;
    }
    // `runStatusImmunity`: the type immunity (`getImmunity`), then
    // `runEvent('Immunity')` (its handler ties), whose Magma Armor / Sun
    // handlers refuse freeze. Abilities such as Insomnia, Limber or Sweet
    // Veil are SetStatus handlers: they block after that event's sort.
    // PS sim/pokemon.ts setStatus / runStatusImmunity.
    if target.hp <= 0 {
        return;
    }
    let type_immune = match st {
        PokemonStatus::BURN => target.has_type(&PokemonType::FIRE),
        PokemonStatus::PARALYZE => target.has_type(&PokemonType::ELECTRIC),
        PokemonStatus::FREEZE => target.has_type(&PokemonType::ICE),
        PokemonStatus::POISON | PokemonStatus::TOXIC => {
            target.has_type(&PokemonType::POISON) || target.has_type(&PokemonType::STEEL)
        }
        _ => false,
    };
    if type_immune {
        return;
    }
    crate::doubles::events::run_event_ties(state, rng, crate::doubles::events::Event::Immunity, (ts, tslot), None);
    if st == PokemonStatus::FREEZE
        && (target.ability == Abilities::MAGMAARMOR
            || matches!(effective_weather(state), Weather::SUN | Weather::HARSHSUN))
    {
        return;
    }
    crate::doubles::events::run_event_ties(state, rng, crate::doubles::events::Event::SetStatus, (ts, tslot), None);
}

/// Yawn's `onEnd` -> `target.trySetStatus('slp', source)`: the setStatus effect
/// is the Yawn condition, which Safeguard exempts (`effect.id === 'yawn'`), and
/// Substitute only blocks move effects. Uproar, Sweet Veil and terrain still do.
fn try_set_status_yawn(
    state: &mut DoublesState,
    rng: &mut Gen5Rng,
    ts: Side,
    tslot: usize,
    src_side: Side,
    src_idx: PokemonIndex,
) {
    try_set_status_with(state, rng, ts, tslot, PokemonStatus::SLEEP, src_side, src_idx, false, false, true);
}

/// A status from a contact ability (Effect Spore, Static, ...). PS
/// `onDamagingHit` -> `trySetStatus` with an Ability effect: Substitute only
/// blocks move effects (its `onTryHit` / `onTryPrimaryHit`), so it does not
/// stop this status. Safeguard / Misty / Uproar still do.
fn try_set_status_from_ability(
    state: &mut DoublesState,
    rng: &mut Gen5Rng,
    ts: Side,
    tslot: usize,
    st: PokemonStatus,
    src_side: Side,
    src_idx: PokemonIndex,
) {
    try_set_status_with(state, rng, ts, tslot, st, src_side, src_idx, false, true, false);
}

#[allow(clippy::too_many_arguments)]
fn try_set_status_with(
    state: &mut DoublesState,
    rng: &mut Gen5Rng,
    ts: Side,
    tslot: usize,
    st: PokemonStatus,
    src_side: Side,
    src_idx: PokemonIndex,
    substitute_blocks: bool,
    safeguard_blocks: bool,
    effect_is_yawn: bool,
) {
    let Some(tidx) = state.side(ts).active()[tslot].pokemon_index else { return };
    let src_slot = (0..2).find(|&sl| state.side(src_side).active()[sl].pokemon_index == Some(src_idx));
    set_status_event_tie(state, rng, ts, tslot, st, ts != src_side, substitute_blocks);
    if status_immune_ex(state, ts, tslot, st, ts != src_side, src_side, src_slot.unwrap_or(0), substitute_blocks, safeguard_blocks, effect_is_yawn) {
        return;
    }
    set_status(state, rng, ts, tidx, st);
    maybe_synchronize(state, ts, tidx, src_side, src_idx, st);
}

/// Whether the pokemon in `ts`/`tslot` cannot get `status`: the shared
/// `mechanics::immune_to_status` (also used by the singles engine), with the
/// field facts read from the doubles state. The format has no Sleep Clause.
fn status_immune(
    state: &DoublesState,
    ts: Side,
    tslot: usize,
    status: PokemonStatus,
    from_opponent: bool,
    src_side: Side,
    src_slot: usize,
) -> bool {
    status_immune_ex(state, ts, tslot, status, from_opponent, src_side, src_slot, true, true, false)
}

#[allow(clippy::too_many_arguments)]
fn status_immune_ex(
    state: &DoublesState,
    ts: Side,
    tslot: usize,
    status: PokemonStatus,
    from_opponent: bool,
    src_side: Side,
    src_slot: usize,
    substitute_blocks: bool,
    safeguard_blocks: bool,
    effect_is_yawn: bool,
) -> bool {
    let Some(target) = state.side(ts).slot_pokemon(tslot) else { return true };
    // Safeguard / Substitute block statuses from any other pokemon, allies
    // included (`target !== source`, PS data/moves.ts safeguard.onSetStatus).
    let from_opponent = from_opponent || (ts, tslot) != (src_side, src_slot);
    // Uproar `onAnySetStatus`: nobody can fall asleep while one is active.
    if status == PokemonStatus::SLEEP
        && [Side::One, Side::Two]
            .iter()
            .any(|&sd| (0..2).any(|sl| state.side(sd).active()[sl].uproar > 0))
    {
        return true;
    }
    // Sweet Veil `onAllySetStatus`: the holder and its ally can't fall asleep.
    // TODO(PS parity): Mold Breaker (breakable).
    if status == PokemonStatus::SLEEP
        && (0..2).any(|sl| {
            state.side(ts).slot_pokemon(sl).map(|p| p.hp > 0 && p.ability == Abilities::SWEETVEIL).unwrap_or(false)
        })
    {
        return true;
    }
    // Flower Veil `onAllySetStatus` (PS runEvent: `alliesAndSelf`, so the holder
    // covers itself too): a Grass-type target can't get a status set by another
    // pokemon (`target !== source`); a Yawn sleep is exempt (`effect.id === 'yawn'`).
    // PS data/abilities.ts flowerveil. The check covers move, secondary,
    // ability and Synchronize statuses alike.
    if !effect_is_yawn
        && target.has_type(&PokemonType::GRASS)
        && !(ts == src_side && tslot == src_slot)
        && (0..2).any(|sl| {
            state.side(ts).slot_pokemon(sl).map(|p| p.hp > 0 && p.ability == Abilities::FLOWERVEIL).unwrap_or(false)
        })
    {
        return true;
    }
    // Infiltrator (`move.infiltrates`, PS data/abilities.ts infiltrator): the
    // move skips the target's Substitute (`substitute.onTryPrimaryHit`) and,
    // against a foe, Safeguard (`safeguard.onSetStatus`).
    let infiltrates = state
        .side(src_side)
        .slot_pokemon(src_slot)
        .is_some_and(|p| p.ability == Abilities::INFILTRATOR);
    let ctx = crate::engine::mechanics::StatusContext {
        // PS sunnyday `onImmunity('frz')` checks `effectiveWeather()`, which is
        // empty while a Cloud Nine / Air Lock holder suppresses the weather.
        sun: matches!(effective_weather(state), Weather::SUN | Weather::HARSHSUN),
        misty_terrain: state.terrain.terrain_type == crate::engine::state::Terrain::MISTYTERRAIN,
        electric_terrain: state.terrain.terrain_type == crate::engine::state::Terrain::ELECTRICTERRAIN,
        safeguard: safeguard_blocks && state.side(ts).side_conditions.safeguard > 0 && !(infiltrates && ts != src_side),
        substitute: substitute_blocks && state.side(ts).active()[tslot].substitute_health > 0 && !infiltrates,
        from_opponent,
        sleep_clause: false,
        source_ability: state
            .side(src_side)
            .slot_pokemon(src_slot)
            .map(|p| p.ability)
            .unwrap_or(Abilities::NONE),
        // PS mistyterrain / electricterrain `onSetStatus`: `!target.isGrounded()
        // || target.isSemiInvulnerable()` lets the status through.
        target_grounded: ps_grounded(state, ts, tslot) && !ally_semi_invulnerable(state, ts, tslot),
    };
    crate::engine::mechanics::immune_to_status(target, &status, &ctx)
}

/// Whether Yawn's volatile cannot be added to `ts`/`tslot`. PS yawn `onTryHit`
/// is `target.status || !target.runStatusImmunity('slp')`: `runStatusImmunity`
/// only consults the `Immunity` event, so Uproar, Misty / Electric Terrain and
/// Safeguard do not stop it (they stop the sleep later, at the residual).
/// `addVolatile('yawn')` then fails for Insomnia (`onTryAddVolatile`), for a
/// Sweet Veil holder or ally (`onAllyTryAddVolatile`), and for a foe's Substitute.
/// PS data/moves.ts yawn, data/abilities.ts insomnia / sweetveil.
fn yawn_volatile_blocked(state: &DoublesState, ts: Side, tslot: usize, src_side: Side, src_slot: usize) -> bool {
    let Some(target) = state.side(ts).slot_pokemon(tslot) else { return true };
    if target.ability == Abilities::INSOMNIA {
        return true;
    }
    if (0..2).any(|sl| state.side(ts).slot_pokemon(sl).is_some_and(|p| p.hp > 0 && p.ability == Abilities::SWEETVEIL)) {
        return true;
    }
    // Flower Veil `onAllyTryAddVolatile` (PS data/abilities.ts flowerveil): a
    // Grass-type target (the holder included, `alliesAndSelf`) blocks yawn.
    if target.has_type(&PokemonType::GRASS)
        && (0..2).any(|sl| state.side(ts).slot_pokemon(sl).is_some_and(|p| p.hp > 0 && p.ability == Abilities::FLOWERVEIL))
    {
        return true;
    }
    // Electric Terrain `onTryAddVolatile` (PS data/moves.ts electricterrain):
    // a grounded target that is not semi-invulnerable blocks yawn (`return null`).
    let t = &state.side(ts).active()[tslot];
    let semi_invulnerable = t.charging.filter(|_| !t.charge_aborted).is_some_and(|(c, _)| {
        matches!(
            c,
            Choices::FLY | Choices::BOUNCE | Choices::DIVE | Choices::DIG | Choices::PHANTOMFORCE | Choices::SHADOWFORCE | Choices::SKYDROP
        )
    });
    if state.terrain.terrain_type == crate::engine::state::Terrain::ELECTRICTERRAIN
        && slot_grounded(target, t, state.gravity.active)
        && !semi_invulnerable
    {
        return true;
    }
    // Safeguard `onTryAddVolatile` (PS data/moves.ts safeguard): yawn from
    // another pokemon is blocked, unless an Infiltrator move hits a foe
    // (`effect.infiltrates && !target.isAlly(source)`). Substitute
    // `onTryPrimaryHit` lets any `move.infiltrates` through.
    let infiltrates = state.side(src_side).slot_pokemon(src_slot).is_some_and(|p| p.ability == Abilities::INFILTRATOR);
    (ts, tslot) != (src_side, src_slot)
        && ((state.side(ts).active()[tslot].substitute_health > 0 && !infiltrates)
            || (state.side(ts).side_conditions.safeguard > 0 && !(infiltrates && ts != src_side)))
}


/// One target's damage for a hit (fixed / OHKO / `calculate_damage`), without
/// applying it. Draws crit + damage roll. PS `getDamage`.
#[allow(clippy::too_many_arguments)]
fn compute_hit_damage(
    state: &DoublesState,
    rng: &mut Gen5Rng,
    side: Side,
    slot: usize,
    idx: PokemonIndex,
    target_side: Side,
    target_slot: usize,
    choice: &Choice,
    ate_boosted: bool,
    spread: bool,
) -> (i16, bool) {
    // Only Electro Ball / Gyro Ball read the speeds (`dynamic_base_power`).
    let speeds = if matches!(choice.move_id, Choices::ELECTROBALL | Choices::GYROBALL) {
        (stat_speed(state, side, slot), stat_speed(state, target_side, target_slot))
    } else {
        (0, 0)
    };
    // PS `ignoringAbility` (gen 5+): a pokemon that is not active (a Future Sight
    // source that fainted or switched out) has its ability ignored, so Levitate
    // does not keep it out of Psychic Terrain (`isGrounded`) or affect its damage.
    let attacker_ref = state.side(side).team_member(idx);
    let attacker_inactive;
    let attacker = if state.side(side).active()[slot].pokemon_index != Some(idx)
        && attacker_ref.ability != Abilities::NONE
    {
        let mut a = attacker_ref.clone();
        a.ability = Abilities::NONE;
        attacker_inactive = a;
        &attacker_inactive
    } else {
        attacker_ref
    };
    // The attacker's slot state applies only while it occupies the slot (a Future
    // Sight source may have left; the slot then belongs to another pokemon).
    let empty_slot;
    let att_slot: &ActiveSlot = if state.side(side).active()[slot].pokemon_index == Some(idx) {
        &state.side(side).active()[slot]
    } else {
        empty_slot = ActiveSlot::empty();
        &empty_slot
    };
    let defender_actual = state
        .side(target_side)
        .slot_pokemon(target_slot)
        .expect("target must be present");
    let defender_view = crate::doubles::damage::smackdown_ground_view(
        &choice.move_type,
        defender_actual,
        state.side(target_side).active()[target_slot].smackdown,
    );
    let defender: &crate::state::Pokemon = &defender_view;
    let mut damage: i16 = 0;
    let mut crit = false;
    // Fixed-damage and OHKO moves make no crit/randomizer draws.
    if choice.ohko {
        damage = defender.maxhp;
    } else if matches!(choice.move_id, Choices::METALBURST | Choices::COMEUPPANCE) {
        // damageCallback: `(lastDamagedBy.damage * 1.5) || 1`.
        damage = last_damaged_by(state, side, idx)
            .map(|(_, _, d, _, _)| ((d as f32) * 1.5) as i16)
            .filter(|&d| d > 0)
            .unwrap_or(1);
    } else if matches!(choice.move_id, Choices::COUNTER | Choices::MIRRORCOAT) {
        // damageCallback: the recorded `2 * damage` (1 if none).
        let a = &att_slot;
        let rec = if choice.move_id == Choices::COUNTER { a.counter_vol } else { a.mirror_coat_vol };
        damage = rec.map(|(_, d)| d).filter(|&d| d > 0).unwrap_or(1);
    } else if let Some(fd) = choice.move_id.fixed_damage(attacker, defender) {
        damage = fd;
    } else {
        // Boosted stats, with crit rules (crit ignores negative attack /
        // positive defense boosts) applied inside `calculate_damage`
        // after the crit roll.
        // The attacker's own boosts, by team member: a Future Sight source may
        // have left the field (its boosts were cleared, its slot is someone else's).
        let att_boost = state.side(side).team_boosts[idx as usize].clone();
        let def_boost = state
            .side(target_side)
            .slot_boosts(target_slot)
            .cloned()
            .unwrap_or_default();
        let is_physical = choice.category == MoveCategory::Physical;
        // Which stats the move uses (PS `getDamage`): physical ->
        // Atk/Def, special -> SpA/SpD, except Psyshock (SpA vs Def)
        // and Secret Sword (Atk vs SpD). Boosts follow the same stat.
        // Foul Play (`overrideOffensivePokemon: 'target'`): the
        // target's Atk stat and boosts.
        let (base_atk, atk_boost_stat) = if choice.move_id == Choices::BODYPRESS {
            // `overrideOffensiveStat: 'def'`. PS Wonder Room `onModifyMove` renames it
            // to 'spd'; `calculateStat('spd')` then reads the stored Def value, and the
            // stage is the 'spd' one. So the value is always Def, the stage follows the name.
            let stage = if state.wonder_room.active { att_boost.spd } else { att_boost.def };
            (attacker.defense, stage)
        } else if choice.move_id == Choices::FOULPLAY {
            (defender.attack, def_boost.atk)
        } else if is_physical {
            (attacker.attack, att_boost.atk)
        } else {
            (attacker.special_attack, att_boost.spa)
        };
        // Which defensive stat the move names (true = Def). Wonder Room swaps
        // the stat VALUE only: `calculateStat` swaps the name, but the stage
        // passed in is still the named stat's (`defBoosts`).
        let names_def = match choice.move_id {
            Choices::PSYSHOCK => true,
            Choices::SECRETSWORD => false,
            _ => is_physical,
        };
        let def_boost_stat = if names_def { def_boost.def } else { def_boost.spd };
        let base_def = if names_def != state.wonder_room.active {
            defender.defense
        } else {
            defender.special_defense
        };
        // Unaware `onAnyModifyBoost` (PS data/abilities.ts unaware): an
        // Unaware defender ignores the attacker's offensive boosts (breakable);
        // an Unaware attacker ignores the defender's defensive boosts.
        // Foul Play included: PS `calculateStat(..., statUser = source)` runs
        // `ModifyBoost` with the user as `pokemon`, so the Unaware target zeroes
        // the user's Atk stage even though the stat comes from the target.
        // Both branches need the hit pokemon to be `battle.activeTarget`; a
        // spread move's Substitute damage runs with a stale one (SUB_ACTIVE_TARGET).
        // Unaware is `onAnyModifyBoost`: the holder that is `activeTarget`
        // zeroes the attacker's boosts even when another pokemon is hit.
        let (unaware_holder, is_active_target) = match state.tl.sub_active_target.get() {
            Some((ats, atsl)) => (
                state.side(ats).slot_pokemon(atsl).map(|p| p.ability).unwrap_or(Abilities::NONE),
                (ats, atsl) == (target_side, target_slot),
            ),
            None => (defender.ability, true),
        };
        let atk_boost_stat = if unaware_holder == Abilities::UNAWARE
            && !crate::engine::abilities::ignores_target_ability(attacker, choice, &unaware_holder)
        {
            0
        } else {
            atk_boost_stat
        };
        // Not from a user with no HP (selfdestruct 'always' fainted it first):
        // PS `findEventHandlers` collects `onAny` handlers only from
        // `foes()` / `alliesAndSelf()`, which skip `!hp` pokemon.
        let def_boost_stat = if is_active_target
            && attacker.ability == Abilities::UNAWARE
            && attacker.hp > 0
            && !is_selfdestruct_always(choice.move_id)
        {
            0
        } else {
            def_boost_stat
        };
        // PS `move.ignoreDefensive` (data/moves.ts chipaway, darkestlariat,
        // sacredsword): `getDamage` sets `defBoosts = 0` (any sign).
        let def_boost_stat = if matches!(
            choice.move_id,
            Choices::SACREDSWORD | Choices::CHIPAWAY | Choices::DARKESTLARIAT
        ) {
            0
        } else {
            def_boost_stat
        };
        let ctx = DamageCtx {
            added_type: state.side(target_side).active()[target_slot].added_type,
            attacker_added_type: state.side(side).active()[slot].added_type,
            attack_base: base_atk,
            attack_boost: atk_boost_stat,
            defense_base: base_def,
            defense_boost: def_boost_stat,
            weather: effective_weather(state),
            // PS data/moves.ts reflect / lightscreen / auroraveil
            // `onAnyModifyDamage`: skipped when `move.infiltrates`
            // (data/abilities.ts infiltrator.onModifyMove).
            reflect: attacker.ability != Abilities::INFILTRATOR
                && (state.side(target_side).side_conditions.reflect > 0
                    || state.side(target_side).side_conditions.aurora_veil > 0),
            light_screen: attacker.ability != Abilities::INFILTRATOR
                && (state.side(target_side).side_conditions.light_screen > 0
                    || state.side(target_side).side_conditions.aurora_veil > 0),
            defender_mod: {
                let mut m = crate::engine::mechanics::defender_damage_modifier(defender, choice);
                // Semi-invulnerable `onSourceModifyDamage`: x2 for the moves that
                // still hit (dig: earthquake / magnitude, fly + bounce: gust /
                // twister, dive: surf / whirlpool). PS data/moves.ts.
                // An aborted charge (sleep, flinch, ...) lost its volatile.
                let tsa = &state.side(target_side).active()[target_slot];
                if let Some((c, _)) = tsa.charging.filter(|_| !tsa.charge_aborted) {
                    let double = match c {
                        Choices::DIG => matches!(choice.move_id, Choices::EARTHQUAKE | Choices::MAGNITUDE),
                        Choices::FLY | Choices::BOUNCE => matches!(choice.move_id, Choices::GUST | Choices::TWISTER),
                        Choices::DIVE => matches!(choice.move_id, Choices::SURF | Choices::WHIRLPOOL),
                        _ => false,
                    };
                    if double {
                        m = (m * 8192 + 2048) >> 12;
                    }
                }
                // Minimize `onSourceModifyDamage`: x2 for the stomp family.
                if state.side(target_side).active()[target_slot].minimize
                    && crate::engine::mechanics::minimize_move(choice.move_id)
                {
                    m = (m * 8192 + 2048) >> 12;
                }
                // Glaive Rush `onSourceModifyDamage`: x2 (chained).
                if state.side(target_side).active()[target_slot].glaive_rush {
                    (m * 8192 + 2048) >> 12
                } else {
                    m
                }
            },
            solid_rock: matches!(
                defender.ability,
                Abilities::FILTER | Abilities::SOLIDROCK | Abilities::PRISMARMOR
            ) && !crate::engine::abilities::ignores_target_ability(attacker, choice, &defender.ability),
            sniper: attacker.ability == Abilities::SNIPER,
            no_crit: matches!(defender.ability, Abilities::SHELLARMOR | Abilities::BATTLEARMOR)
                && !crate::engine::abilities::ignores_target_ability(attacker, choice, &defender.ability),
            guts: attacker.ability == Abilities::GUTS,
            attack_mod: {
            let m = crate::engine::mechanics::attacker_stat_modifier(attacker, choice);
            // PS flashfire `onModifyAtk`: `attacker.hasAbility('flashfire')`.
            let m = if att_slot.flash_fire && attacker.ability == Abilities::FLASHFIRE && choice.move_type == PokemonType::FIRE {
                (m * 6144 + 2048) >> 12
            } else {
                m
            };
            // Thick Fat & co. (`onSourceModifyAtk` / `SpA`, breakable) chain first.
            let src = if (side, slot) != (target_side, target_slot)
                && !crate::engine::abilities::ignores_target_ability(attacker, choice, &defender.ability)
            {
                crate::engine::mechanics::defender_source_stat_modifier(defender, choice)
            } else {
                4096
            };
            let m = if src != 4096 { (((4096 * src + 2048) >> 12) * m + 2048) >> 12 } else { m };
            // Plus / Minus `onModifySpA`: x1.5 when an active ally has Plus or
            // Minus. PS data/abilities.ts plus, minus.
            let m = if matches!(attacker.ability, Abilities::PLUS | Abilities::MINUS)
                && choice.category == MoveCategory::Special
                && state
                    .side(side)
                    .slot_pokemon(1 - slot)
                    .is_some_and(|a| a.hp > 0 && matches!(a.ability, Abilities::PLUS | Abilities::MINUS))
            {
                (m * 6144 + 2048) >> 12
            } else {
                m
            };
            // Solar Power `onModifySpA` (priority 5, exclusive with Plus / Minus).
            let sun = matches!(effective_weather(state), Weather::SUN | Weather::HARSHSUN);
            let m = (m * crate::engine::mechanics::solar_power_spa_modifier(attacker, choice, sun) + 2048) >> 12;
            // Stakeout `onModifyAtk` / `onModifySpA`: x2 against a target that
            // switched in this turn (`!defender.activeTurns`).
            if attacker.ability == Abilities::STAKEOUT
                && choice.category != MoveCategory::Status
                && state.side(target_side).active()[target_slot].active_turns == 0
            {
                (m * 8192 + 2048) >> 12
            } else {
                m
            }
        },
        // TODO(PS parity): Mold Breaker (friendguard is breakable).
        friend_guard: state
            .side(target_side)
            .slot_pokemon(1 - target_slot)
            .map(|a| a.hp > 0 && a.ability == Abilities::FRIENDGUARD)
            .unwrap_or(false),
        friend_guard_ties: friend_guard_tie_groups(state),
        screen_handlers: [Side::One, Side::Two]
            .iter()
            .map(|&sd| {
                let sc = &state.side(sd).side_conditions;
                (sc.reflect > 0) as u8 + (sc.light_screen > 0) as u8 + (sc.aurora_veil > 0) as u8
            })
            .sum(),
        // PS focusenergy +2; Super Luck `onModifyCritRatio` +1 (data/abilities.ts).
        crit_bonus: (if att_slot.focus_energy { 2 } else { 0 })
            + (match att_slot.dragon_cheer {
                Some(true) => 2,
                Some(false) => 1,
                None => 0,
            })
            + (if attacker.ability == Abilities::SUPERLUCK { 1 } else { 0 })
            // Merciless `onModifyCritRatio`: 5 (a sure crit) on a poisoned target.
            + (if attacker.ability == Abilities::MERCILESS
                && matches!(defender.status, PokemonStatus::POISON | PokemonStatus::TOXIC)
            {
                5
            } else {
                0
            }),
        disguise: defender.id == crate::pokemon::PokemonName::MIMIKYU
            && defender.ability == Abilities::DISGUISE
            && state.side(target_side).active()[target_slot].substitute_health <= 0
            && (side, slot) != (target_side, target_slot),
        // Fur Coat / Marvel Scale `onModifyDef` (applies when the move names Def).
        defense_mod: {
            let mut m: i64 = match defender.ability {
                Abilities::FURCOAT if names_def => 8192,
                Abilities::MARVELSCALE if names_def && defender.status != PokemonStatus::NONE => 6144,
                // PS data/abilities.ts grasspelt.
                Abilities::GRASSPELT
                    if names_def && state.terrain.terrain_type == crate::engine::state::Terrain::GRASSYTERRAIN =>
                {
                    6144
                }
                _ => 4096,
            };
            // Sandstorm: Rock-types x1.5 Sp. Def; Snow: Ice-types x1.5 Def
            // (`onModifySpD` / `onModifyDef` of the weather condition).
            // PS `effectiveWeather()`: none while Cloud Nine / Air Lock suppress it.
            let has = |t| defender.types.0 == t || defender.types.1 == t;
            let eff_weather = effective_weather(state);
            if eff_weather == Weather::SAND && !names_def && has(PokemonType::ROCK) {
                m = (m * 6144 + 2048) >> 12;
            }
            if eff_weather == Weather::SNOW && names_def && has(PokemonType::ICE) {
                m = (m * 6144 + 2048) >> 12;
            }
            m
        },
            // Dynamic base power (Reversal / Flail) and Helping Hand
            // (x1.5) are combined into the override.
            base_power_override: base_power_with_modifiers(
                choice,
                attacker,
                defender,
                att_slot.helping_hand_mult,
                &{
                    let mut m: Vec<(i32, i64)> = Vec::new();
                    // charge.onBasePower (priority 9): x2 for Electric moves.
                    if att_slot.charge && choice.move_type == PokemonType::ELECTRIC {
                        m.push((9, 8192));
                    }
                    // lashout.onBasePower: x2 if the user's stats were lowered this turn.
                    if choice.move_id == Choices::LASHOUT && att_slot.stats_lowered_this_turn {
                        m.push((0, 8192));
                    }
                    m
                },
                speeds,
                state.terrain.terrain_type,
                ally_semi_invulnerable(state, target_side, target_slot),
                ate_boosted,
                att_slot.fallen,
                (0..2)
                    .filter(|&sl| {
                        state.side(side).slot_pokemon(sl).map(|p| p.hp > 0 && p.ability == Abilities::STEELYSPIRIT).unwrap_or(false)
                    })
                    .count() as u8,
                // PS avalanche.basePowerCallback: doubled when the target
                // damaged the user this turn (`attackedBy` `thisTurn`).
                choice.move_id == Choices::AVALANCHE
                    && state
                        .side(target_side)
                        .active()[target_slot]
                        .pokemon_index
                        .map(|tidx| {
                            attacker
                                .attacked_by
                                .iter()
                                .any(|&(s, t)| s == attacker_key(target_side, tidx as u8) && t == state.turn)
                        })
                        .unwrap_or(false),
                side_total_fainted(state, side),
                att_slot.hit_number as u32,
                att_slot.bp_double
                    || (matches!(choice.move_id, Choices::STOMPINGTANTRUM | Choices::TEMPERFLARE)
                        && att_slot.move_result_last_turn == Some(false)),
                // `basePowerCallback`s that read the user's slot (Beat Up, Spit Up,
                // Stored Power): they run before the `onBasePower` modifiers.
                slot_base_power_callback(state, side, slot, &att_slot, choice),
                // PS `rivalry.onBasePower`: both genders must be set (`M` / `F`).
                {
                    let gendered = |g: Option<&str>| matches!(g, Some("M") | Some("F"));
                    let ag = state.side(side).genders[idx as usize].as_deref();
                    let dg = state
                        .side(target_side)
                        .active()[target_slot]
                        .pokemon_index
                        .and_then(|di| state.side(target_side).genders[di as usize].as_deref());
                    match (gendered(ag), gendered(dg)) {
                        (true, true) if ag == dg => 1,
                        (true, true) => -1,
                        _ => 0,
                    }
                },
                slot_grounded(attacker, &att_slot, state.gravity.active),
                slot_grounded(defender, &state.side(target_side).active()[target_slot], state.gravity.active),
                state,
            ),
        };
        let mut ctx = ctx;
        // PS `getDamage` `runEvent` sorts: their handler ties draw between
        // the crit roll and the damage roll. Only while the attacker is in
        // its slot (`run_event_ties` reads the slot's effects).
        // TODO(PS parity): a Future Sight source that left the field.
        let att_active = state.side(side).active()[slot].pokemon_index == Some(idx);
        let mut on_event = |ev: crate::doubles::damage::DamageEvent, rng: &mut Gen5Rng| {
            use crate::doubles::damage::DamageEvent;
            use crate::doubles::events::Event;
            if !att_active {
                return false;
            }
            let (src, tgt) = ((side, slot), (target_side, target_slot));
            match ev {
                DamageEvent::ModifyCritRatio => {
                    crate::doubles::events::run_event_ties(state, rng, crate::doubles::events::Event::ModifyCritRatio, src, Some(tgt));
                }
                DamageEvent::BasePower => {
                    // The move's own `onBasePower` (none has a priority) joins the sort.
                    let own = matches!(
                        choice.move_id,
                        Choices::GRAVAPPLE | Choices::BARBBARRAGE | Choices::EXPANDINGFORCE | Choices::FACADE
                            | Choices::FICKLEBEAM | Choices::KNOCKOFF | Choices::LASHOUT | Choices::MISTYEXPLOSION
                            | Choices::SOLARBEAM | Choices::SOLARBLADE | Choices::VENOSHOCK
                    );
                    crate::doubles::events::run_event_ties_on_effect(
                        state, rng, crate::doubles::events::Event::BasePower, src, Some(tgt), own.then_some(0),
                    );
                }
                DamageEvent::ModifyStats => {
                    let atk = if choice.category == MoveCategory::Physical { Event::ModifyAtk } else { Event::ModifySpA };
                    // `overrideDefensiveStat: 'def'` (Psyshock, Psystrike, Secret Sword).
                    let def = if choice.category == MoveCategory::Physical
                        || matches!(choice.move_id, Choices::PSYSHOCK | Choices::PSYSTRIKE | Choices::SECRETSWORD)
                    {
                        Event::ModifyDef
                    } else {
                        Event::ModifySpD
                    };
                    crate::doubles::events::run_event_ties(state, rng, atk, src, Some(tgt));
                    crate::doubles::events::run_event_ties(state, rng, def, tgt, Some(src));
                }
                DamageEvent::ModifyDamage => {
                    crate::doubles::events::run_event_ties(state, rng, crate::doubles::events::Event::ModifyDamage, src, Some(tgt));
                }
            }
            true
        };
        if let Some(outcome) =
            crate::doubles::damage::calculate_damage_with(attacker, defender, choice, rng, spread, &ctx, &mut on_event)
        {
            damage = outcome.damage;
            crit = outcome.crit;
        }
    }
    (damage, crit)
}

/// The defender's contact abilities (shared table
/// `mechanics::contact_ability`). The defender may already be at 0 HP: PS
/// runs DamagingHit before faints are resolved.
// TODO(PS parity): Protective Pads, Long Reach.
#[allow(clippy::too_many_arguments)]
fn contact_abilities(
    state: &mut DoublesState,
    rng: &mut Gen5Rng,
    side: Side,
    slot: usize,
    idx: PokemonIndex,
    target_side: Side,
    target_slot: usize,
    choice: &Choice,
) {
    use crate::engine::mechanics::{contact_ability, powder_immune, ContactAbility};
    if !choice.flags.contact {
        return;
    }
    // Long Reach: the attacker's contact moves are non-contact (PS
    // `onModifyMove` deletes the `contact` flag).
    if state.side(side).team_member(idx).ability == Abilities::LONGREACH {
        return;
    }
    let def_idx = state.side(target_side).active()[target_slot].pokemon_index.or_else(|| {
        state
            .unresolved_faints
            .iter()
            .find(|(s, sl, _)| *s == target_side && *sl == target_slot)
            .map(|&(_, _, i)| i)
    });
    let Some(def_idx) = def_idx else { return };
    let ability = state.side(target_side).team_member(def_idx).ability;
    // Aftermath `onDamagingHit`: if the holder fainted, the contact attacker
    // loses baseMaxhp / 4. Damp (any active) blocks it (`onAnyDamage`) unless
    // the attacker ignores breakable abilities. PS data/abilities.ts aftermath / damp.
    if ability == Abilities::AFTERMATH {
        if state.side(target_side).team_member(def_idx).hp <= 0 {
            let attacker = state.side(side).team_member(idx).clone();
            let damped = [Side::One, Side::Two].iter().any(|&sd| {
                (0..2).any(|sl| {
                    state.side(sd).slot_pokemon(sl).map(|p| {
                        p.hp > 0
                            && p.ability == Abilities::DAMP
                            && !crate::engine::abilities::ignores_target_ability(&attacker, choice, &p.ability)
                    }).unwrap_or(false)
                })
            });
            if !damped {
                let maxhp = attacker.maxhp;
                apply_indirect_damage(state, side, slot, (maxhp / 4).max(1));
            }
        }
        return;
    }
    // Mummy / Lingering Aroma `onDamagingHit`: the contact attacker's ability
    // becomes the holder's unless it can't be suppressed or already matches.
    // PS data/abilities.ts mummy / lingeringaroma.
    if matches!(ability, Abilities::MUMMY | Abilities::LINGERINGAROMA) {
        let a = state.side(side).team_member(idx).ability;
        // PS `sourceAbility.flags['cantsuppress']` (Zero to Hero included).
        let cantsuppress = crate::engine::mechanics::cant_suppress(a);
        if !cantsuppress && a != ability {
            state.side_mut(side).team_member_mut(idx).ability = ability;
        }
        return;
    }
    // Cute Charm `onDamagingHit`: `randomChance(3, 10)`, then the attacker gets
    // the `attract` volatile (needs opposite genders; fails if already
    // infatuated). PS data/abilities.ts cutecharm, data/conditions.ts attract.
    if ability == Abilities::CUTECHARM {
        if rng.label("ability").random_chance(3, 10) {
            let att_gender = state.side(side).genders[idx as usize].clone();
            let def_gender = state.side(target_side).genders[def_idx as usize].clone();
            let opposite = matches!(
                (att_gender.as_deref(), def_gender.as_deref()),
                (Some("M"), Some("F")) | (Some("F"), Some("M"))
            );
            // PS `addVolatile('attract')` -> `onAllyTryAddVolatile` (data/abilities.ts
            // aromaveil): an Aroma Veil on the holder's side (itself or its ally)
            // stops it. The Cute Charm roll above still happens.
            let aroma_veil = aroma_veil_blocks(state, side, AromaVeiled::Attract, None);
            if opposite
                && !aroma_veil
                && state.side(side).team_member(idx).ability != Abilities::OBLIVIOUS
                && state.side(side).active()[slot].attract.is_none()
            {
                state.side_mut(side).active_slot_mut(slot).attract = Some((target_side, def_idx));
            }
        }
        return;
    }
    match contact_ability(ability) {
        Some(ContactAbility::Status30(st)) => {
            if rng.label("ability").random_chance(3, 10) {
                try_set_status_from_ability(state, rng, side, slot, st, target_side, def_idx);
            }
        }
        Some(ContactAbility::EffectSpore) => {
            let attacker = state.side(side).team_member(idx);
            if !(powder_immune(attacker) || state.side(side).active()[slot].added_type == PokemonType::GRASS) {
                let r = rng.label("ability").random(100);
                let st = if r < 11 {
                    Some(PokemonStatus::SLEEP)
                } else if r < 21 {
                    Some(PokemonStatus::PARALYZE)
                } else if r < 30 {
                    Some(PokemonStatus::POISON)
                } else {
                    None
                };
                if let Some(st) = st {
                    try_set_status_from_ability(state, rng, side, slot, st, target_side, def_idx);
                }
            }
        }
        Some(ContactAbility::Recoil8) => {
            let maxhp = state.side(side).team_member(idx).maxhp;
            apply_indirect_damage(state, side, slot, (maxhp / 8).max(1));
        }
        None => {}
    }
}

/// The defender's absorbing abilities (`onTryHit`): the hit is cancelled and
/// the ability's effect applies (heal 1/4 for Water / Volt Absorb, Dry Skin
/// and Earth Eater; a stat stage for Lightning Rod, Storm Drain, Sap Sipper,
/// Motor Drive and Well-Baked Body; Flash Fire's volatile). No draws.
/// Returns true if the move was absorbed.
// TODO(PS parity): Flash Fire frozen check.
fn try_absorb(state: &mut DoublesState, ts: Side, tsl: usize, move_type: PokemonType) -> bool {
    let Some(d) = state.side(ts).slot_pokemon(tsl) else { return false };
    if d.hp <= 0 {
        return false;
    }
    enum A {
        Heal,
        Boost(StatBoosts),
        FlashFire,
    }
    let a = match (d.ability, move_type) {
        (Abilities::WATERABSORB | Abilities::DRYSKIN, PokemonType::WATER)
        | (Abilities::VOLTABSORB, PokemonType::ELECTRIC)
        | (Abilities::EARTHEATER, PokemonType::GROUND)
        => A::Heal,
        (Abilities::LIGHTNINGROD, PokemonType::ELECTRIC) | (Abilities::STORMDRAIN, PokemonType::WATER) => {
            A::Boost(StatBoosts { special_attack: 1, ..Default::default() })
        }
        (Abilities::SAPSIPPER, PokemonType::GRASS) => A::Boost(StatBoosts { attack: 1, ..Default::default() }),
        (Abilities::MOTORDRIVE, PokemonType::ELECTRIC) => A::Boost(StatBoosts { speed: 1, ..Default::default() }),
        (Abilities::WELLBAKEDBODY, PokemonType::FIRE) => A::Boost(StatBoosts { defense: 2, ..Default::default() }),
        (Abilities::FLASHFIRE, PokemonType::FIRE) => A::FlashFire,
        _ => return false,
    };
    let Some(tidx) = state.side(ts).active()[tsl].pokemon_index else { return false };
    match a {
        A::Heal => {
            // PS data/abilities.ts waterabsorb.onTryHit: `this.heal(...)` fails
            // under Heal Block (data/moves.ts healblock.condition.onTryHeal),
            // so the target is still immune (`-immune`) but gains no HP.
            if state.side(ts).active()[tsl].heal_block == 0 {
                let pm = state.side_mut(ts).team_member_mut(tidx);
                pm.hp = (pm.hp + (pm.maxhp / 4).max(1)).min(pm.maxhp);
            }
        }
        A::Boost(b) => apply_boosts(state, ts, tidx, &b, false),
        A::FlashFire => state.side_mut(ts).active_slot_mut(tsl).flash_fire = true,
    }
    true
}

/// `futuremove.onEnd`: the delayed Future Sight / Doom Desire hit
/// (`trySpreadMoveHit([target], source, hitMove, true)`): type immunity, an
/// accuracy roll, then crit + damage roll. Skipped when the target slot is
/// empty / fainted or the user is the target.
// TODO(PS parity): source no longer active, secondaries, Infiltrator / Normalize.
/// The duration of one of the side conditions a `SideDuration` residual handler
/// counts down (see `ResidualKind::SideDuration` for the index).
fn side_condition_duration_mut(state: &mut DoublesState, side: Side, which: u8) -> &mut i8 {
    let sc = &mut state.side_mut(side).side_conditions;
    match which {
        0 => &mut sc.reflect,
        1 => &mut sc.light_screen,
        2 => &mut sc.safeguard,
        3 => &mut sc.mist,
        4 => &mut sc.lucky_chant,
        5 => &mut sc.aurora_veil,
        6 => &mut sc.quick_guard,
        _ => &mut sc.wide_guard,
    }
}

/// PS `hitStepInvulnerabilityEvent` for a semi-invulnerable target: the move
/// misses (no draw) unless it is one of the moves that still hit that charge
/// turn (`onInvulnerability` in data/moves.ts), or a No Guard side is involved.
/// The charge must still be live (an aborted charge lost its `twoturnmove`).
fn semi_invulnerable_blocks(
    state: &DoublesState,
    src_side: Side,
    src_idx: PokemonIndex,
    mv: Choices,
    ts: Side,
    tslot: usize,
) -> bool {
    let t = &state.side(ts).active()[tslot];
    let Some((charging, _)) = t.charging.filter(|_| !t.charge_aborted) else { return false };
    let hits_anyway = match charging {
        Choices::DIG => matches!(mv, Choices::EARTHQUAKE | Choices::MAGNITUDE),
        Choices::FLY | Choices::BOUNCE => matches!(
            mv,
            Choices::GUST
                | Choices::TWISTER
                | Choices::SKYUPPERCUT
                | Choices::THUNDER
                | Choices::HURRICANE
                | Choices::SMACKDOWN
                | Choices::THOUSANDARROWS
        ),
        Choices::DIVE => matches!(mv, Choices::SURF | Choices::WHIRLPOOL),
        Choices::PHANTOMFORCE | Choices::SHADOWFORCE => false,
        _ => true, // not a semi-invulnerable charge (Meteor Beam, ...)
    };
    let no_guard = state.side(src_side).team_member(src_idx).ability == Abilities::NOGUARD
        || state
            .side(ts)
            .slot_pokemon(tslot)
            .map(|p| p.ability == Abilities::NOGUARD)
            .unwrap_or(false);
    // PS data/moves.ts lockon.condition `onSourceInvulnerability` (priority 1):
    // the holder's moves against its locked foe return 0 (hit).
    let lock_on = (0..2).any(|sl| {
        state.side(src_side).active()[sl].pokemon_index == Some(src_idx) && lock_on_hits(state, src_side, sl, ts, tslot)
    });
    !hits_anyway && !no_guard && !lock_on && mv != Choices::HELPINGHAND
}

/// The `onDamagingHit` abilities that set the field: Sand Spit
/// (`field.setWeather('sandstorm')`, fails if already up, else
/// `eachEvent('WeatherChange')`) and Seed Sower (`field.setTerrain('grassyterrain')`,
/// then `eachEvent('TerrainChange')`). PS data/abilities.ts sandspit / seedsower.
// TODO(PS parity): Smooth Rock / Terrain Extender duration, primal weather.
fn field_setting_damaging_hit(state: &mut DoublesState, rng: &mut Gen5Rng, ability: Abilities) {
    match ability {
        Abilities::SANDSPIT if state.weather.weather_type != Weather::SAND => {
            state.weather = crate::state::StateWeather {
                weather_type: Weather::SAND,
                turns_remaining: 5,
            };
            let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
        }
        Abilities::SEEDSOWER if state.terrain.terrain_type != crate::engine::state::Terrain::GRASSYTERRAIN => {
            state.terrain = crate::state::StateTerrain {
                terrain_type: crate::engine::state::Terrain::GRASSYTERRAIN,
                turns_remaining: 5,
            };
            refresh_mimicry(state);
            let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
        }
        _ => {}
    }
}

fn resolve_future_move(
    state: &mut DoublesState,
    rng: &mut Gen5Rng,
    mv: Choices,
    ss: Side,
    si: PokemonIndex,
    ts: Side,
    tslot: usize,
) {
    // PS futuremove `onEnd`: the source may have fainted or switched out; the
    // move is still resolved with its stats (`trySpreadMoveHit([target], data.source, ...)`).
    // A source that is no longer active has its boosts / volatiles cleared.
    let src_slot = (0..2).find(|&sl| state.side(ss).active()[sl].pokemon_index == Some(si));
    let sslot = src_slot.unwrap_or_else(|| {
        (0..2).find(|&sl| state.side(ss).active()[sl].pokemon_index.is_none()).unwrap_or(0)
    });
    if !state.side(ts).slot_pokemon(tslot).map(|p| p.hp > 0).unwrap_or(false)
        || (src_slot.is_some() && (ss, sslot) == (ts, tslot))
    {
        return;
    }
    let Some(mut choice) = MOVES.get(&mv).cloned() else { return };
    choice.accuracy = 100.0;
    if state.side(ts).slot_pokemon(tslot).map(|d| crate::doubles::damage::type_mod(&choice.move_type, d).is_none()).unwrap_or(true) {
        return;
    }
    // `hitStepInvulnerabilityEvent` runs before accuracy: a semi-invulnerable
    // target (Dig, Fly, ...) misses the delayed hit with no accuracy draw.
    if semi_invulnerable_blocks(state, ss, si, mv, ts, tslot) {
        return;
    }
    let ab = if src_slot.is_some() {
        state.side(ss).slot_boosts(sslot).map(|b| b.acc).unwrap_or(0)
    } else {
        0
    };
    let eb = state.side(ts).slot_boosts(tslot).map(|b| b.eva).unwrap_or(0);
    let mods = accuracy_ability_mods(state, ss, si, ts, tslot, &choice);
    // No Guard `onAnyAccuracy` (either side): no roll, the delayed hit always lands.
    // Glaive Rush's drawback (`glaiverush.onAccuracy` returns true) too.
    let no_guard = state.side(ss).team_member(si).ability == Abilities::NOGUARD
        || state.side(ts).slot_pokemon(tslot).is_some_and(|p| p.ability == Abilities::NOGUARD)
        || state.side(ts).active()[tslot].glaive_rush;
    if !no_guard && !roll_accuracy_boosted(choice.accuracy, ab, eb, mods, rng) {
        return;
    }
    // The delayed hit is its own move: the source slot's base-power doubling
    // flag (left over from its last Payback / Assurance) does not apply.
    let saved_bp_double = std::mem::replace(&mut state.side_mut(ss).slot_scratch_mut(sslot).bp_double, false);
    let (damage, _) = compute_hit_damage(state, rng, ss, sslot, si, ts, tslot, &choice, false, false);
    state.side_mut(ss).slot_scratch_mut(sslot).bp_double = saved_bp_double;
    // Disguise absorbs the hit (a `Move` effect); the bust runs in the first
    // `eachEvent('Update')` (its `onUpdate`), before the closing sort.
    let disguised = state.side(ts).slot_pokemon(tslot).is_some_and(|d| {
        d.id == crate::pokemon::PokemonName::MIMIKYU && d.ability == Abilities::DISGUISE
    }) && state.side(ts).active()[tslot].substitute_health <= 0;
    if disguised {
        each_event_update(state, rng);
        disguise_bust(state, ts, tslot);
        each_event_update(state, rng);
        return;
    }
    // Substitute `onTryPrimaryHit` on the delayed hit (PS sim/battle.ts
    // trySpreadMoveHit -> data/conditions.ts substitute): a damaging move hits
    // the Substitute unless it bypasses it or the source has Infiltrator
    // (`data.moveData.infiltrates`). The Substitute takes the damage; no HP loss.
    // PS futuremove `onEnd`: `data.source.hasAbility('infiltrator')`. A source
    // that is no longer active has its ability ignored (gen 5+ `ignoringAbility`),
    // so only an active Infiltrator bypasses the Substitute.
    let sub_hp = state.side(ts).active()[tslot].substitute_health;
    if sub_hp > 0
        && !bypasses_substitute(&choice)
        && !(src_slot.is_some() && state.side(ss).team_member(si).ability == Abilities::INFILTRATOR)
    {
        let d = damage.min(sub_hp);
        state.side_mut(ts).active_slot_mut(tslot).substitute_health = sub_hp - d;
        each_event_update(state, rng);
        each_event_update(state, rng);
        return;
    }
    let damage = match state.side(ts).slot_pokemon(tslot) {
        Some(d) => damage.min(d.hp.max(0)),
        None => damage,
    };
    // Endure / Sturdy `onDamage` apply to the delayed hit too (its effect is a
    // `Move`): a KO leaves 1 HP. PS data/moves.ts endure, data/abilities.ts sturdy.
    let damage = match state.side(ts).slot_pokemon(tslot) {
        Some(d) if state.side(ts).active()[tslot].endure && damage >= d.hp && d.hp > 0 => d.hp - 1,
        Some(d) if d.ability == Abilities::STURDY && d.hp == d.maxhp && damage >= d.hp && d.hp > 0 => d.hp - 1,
        _ => damage,
    };
    // Berserk `onAfterMoveSecondary` (after the hit's `eachEvent('Update')`):
    // the HP before the delayed hit. PS data/abilities.ts berserk,
    // sim/battle-actions.ts hitStepMoveHitLoop (afterMoveSecondaryEvent).
    let berserk_before = state
        .side(ts)
        .slot_pokemon(tslot)
        .filter(|d| damage > 0 && d.ability == Abilities::BERSERK)
        .map(|d| d.hp);
    if damage > 0 {
        apply_damage(state, ts, tslot, damage);
        damaged_by_move(state, ts, tslot);
        // PS `DamagingHit` (trySpreadMoveHit) also runs for the delayed hit: the
        // target's damaging-hit boosts (Weak Armor, Justified, ...) and Stamina.
        // PS data/abilities.ts stamina / sim/battle-actions.ts spreadMoveHit.
        if let Some(didx) = state.side(ts).active()[tslot].pokemon_index {
            let ability = state.side(ts).team_member(didx).ability;
            if let Some(b) = crate::engine::mechanics::on_damaging_hit_boosts(ability, &choice) {
                apply_boosts(state, ts, didx, &b, false);
            }
            if ability == Abilities::STAMINA && state.side(ts).team_member(didx).hp > 0 {
                apply_boosts(state, ts, didx, &StatBoosts { defense: 1, ..Default::default() }, false);
            }
            field_setting_damaging_hit(state, rng, ability);
        }
    }
    // PS trySpreadMoveHit: the hit loop's `eachEvent('Update')`, then
    // `resolveFaints` (the target's faint is printed here), then the closing
    // `eachEvent('Update')`, which no longer counts the fainted target.
    each_event_update(state, rng);
    if let Some(before) = berserk_before {
        if let Some(d) = state.side(ts).slot_pokemon(tslot) {
            if d.hp > 0 && (d.hp as i32) * 2 <= d.maxhp as i32 && (before as i32) * 2 > d.maxhp as i32 {
                let bidx = state.side(ts).active()[tslot].pokemon_index.unwrap();
                apply_boosts(state, ts, bidx, &StatBoosts { special_attack: 1, ..Default::default() }, false);
            }
        }
    }
    resolve_faints(state, rng, |_, _| true);
    each_event_update(state, rng);
}

/// PS sim/pokemon.ts setSpecies (switch-in, `formeChange`): `storedStats` are
/// rebuilt from the species' base stats and the set, which undoes a Guard /
/// Power Split (Def/SpD, Atk/SpA) and a Speed Swap. Formes whose base stats
/// differ (Aegislash, Palafin) then overwrite the recomputed stats.
fn restore_stored_stats(s: &mut crate::doubles::state::DoublesSide, idx: PokemonIndex) {
    if let Some(st) = s.saved_stats[idx as usize].take() {
        let p = s.team_member_mut(idx);
        p.attack = st[0];
        p.defense = st[1];
        p.special_attack = st[2];
        p.special_defense = st[3];
    }
    if let Some(sp) = s.saved_speed[idx as usize].take() {
        s.team_member_mut(idx).speed = sp;
    }
}

/// Disguise busts (PS data/abilities.ts disguise `onDamage` / `onUpdate`): the
/// forme changes to Busted, the holder takes 1/8 max HP, and the cached speed
/// becomes the raw stat (`formeChange` -> `setSpecies`).
fn disguise_bust(state: &mut DoublesState, side: Side, slot: usize) {
    let Some(idx) = state.side(side).active()[slot].pokemon_index else { return };
    restore_stored_stats(state.side_mut(side), idx);
    let pm = state.side_mut(side).team_member_mut(idx);
    pm.id = crate::pokemon::PokemonName::MIMIKYUBUSTED;
    let dmg = (pm.maxhp / 8).max(1);
    apply_damage(state, side, slot, dmg);
    let raw = state.side(side).team_member(idx).speed;
    state.side_mut(side).speed_cache[idx as usize] = raw as _;
}

/// The volatiles Aroma Veil's `onAllyTryAddVolatile` blocks (PS
/// data/abilities.ts aromaveil). Keyed by the volatile, whatever adds it
/// (Taunt, Psychic Noise's Heal Block, Cursed Body's Disable, Cute Charm, ...).
#[derive(Clone, Copy)]
enum AromaVeiled {
    Attract,
    Disable,
    Encore,
    HealBlock,
    Taunt,
    Torment,
}

/// Whether adding `_v` to the pokemon in `ts`/`tsl` is blocked by an active
/// Aroma Veil on its side (the holder itself or its ally). A move from a Mold
/// Breaker-style user ignores it (`breakable`). `attacker` is the move's user
/// when the volatile comes from a move.
fn aroma_veil_blocks(state: &DoublesState, ts: Side, _v: AromaVeiled, attacker: Option<(Side, PokemonIndex, &Choice)>) -> bool {
    (0..2).any(|sl| {
        state.side(ts).slot_pokemon(sl).is_some_and(|p| {
            p.hp > 0
                && p.ability == Abilities::AROMAVEIL
                && !attacker.is_some_and(|(asd, ai, c)| {
                    crate::engine::abilities::ignores_target_ability(state.side(asd).team_member(ai), c, &Abilities::AROMAVEIL)
                })
        })
    })
}

/// The user's `onPrepareHit` handlers: Libero / Protean (once per switch-in,
/// the user becomes the move's type). PS data/abilities.ts libero / protean.
/// TODO(PS parity): Protean's champions override, Tera.
fn prepare_hit(state: &mut DoublesState, side: Side, slot: usize, idx: PokemonIndex, choice: &Choice) {
    if !matches!(state.side(side).team_member(idx).ability, Abilities::LIBERO | Abilities::PROTEAN)
        || state.side(side).active()[slot].protean_done
        || choice.move_type == PokemonType::TYPELESS
        || choice.move_id == Choices::STRUGGLE
        || crate::engine::mechanics::is_future_move(choice.move_id)
        // `move.callsMove` moves (Copycat, Metronome, ...) don't trigger it: the
        // move they call does.
        || matches!(
            choice.move_id,
            Choices::ASSIST
                | Choices::COPYCAT
                | Choices::MEFIRST
                | Choices::METRONOME
                | Choices::MIRRORMOVE
                | Choices::NATUREPOWER
                | Choices::SLEEPTALK
        )
    {
        return;
    }
    let pm = state.side_mut(side).team_member_mut(idx);
    if pm.types != (choice.move_type, PokemonType::TYPELESS) {
        let orig = pm.types;
        pm.types = (choice.move_type, PokemonType::TYPELESS);
        let a = state.side_mut(side).active_slot_mut(slot);
        if a.type_change.is_none() {
            a.type_change = Some(orig);
        }
        a.protean_done = true;
    }
}

/// State changes of a pokemon that took damage from a move's hit, shared by
/// every move-hit path (normal hits, Future Sight / Doom Desire): Illusion
/// `onDamagingHit` ends the disguise. PS data/abilities.ts illusion.
fn damaged_by_move(state: &mut DoublesState, ts: Side, tsl: usize) {
    state.side_mut(ts).slot_scratch_mut(tsl).illusion_broken = true;
}

/// Emergency Exit / Wimp Out (champions `onEmergencyExit`): the holder dropped
/// from above half HP to half or less and is still alive, its side can switch,
/// and it is not already switching: it switches out after the action.
/// PS data/mods/champions/abilities.ts emergencyexit / wimpout.
fn emergency_exit_check(state: &mut DoublesState, ts: Side, tsl: usize, before: i16) {
    let Some(d) = state.side(ts).slot_pokemon(tsl) else { return };
    let (after, max) = (d.hp as i32, d.maxhp as i32);
    if matches!(d.ability, Abilities::EMERGENCYEXIT | Abilities::WIMPOUT)
        && after > 0
        && after * 2 <= max
        && (before as i32) * 2 > max
        && state.side(ts).find_replacement().is_some()
        && !state.side(ts).active()[tsl].force_switch
    {
        state.side_mut(ts).slot_scratch_mut(tsl).force_switch = true;
    }
}

/// Cursed Body (PS data/abilities.ts `onDamagingHit`): 30% to Disable the
/// attacker's move (the move just used; `disable.onStart` shortens the
/// duration by one when the holder is mid-move, so 4 more end-of-turns).
// TODO(PS parity): Disable's `cant` BeforeMove, PP check.
fn cursed_body(
    state: &mut DoublesState,
    rng: &mut Gen5Rng,
    side: Side,
    slot: usize,
    target_side: Side,
    target_slot: usize,
    choice: &Choice,
) {
    let def_idx = state.side(target_side).active()[target_slot].pokemon_index.or_else(|| {
        state
            .unresolved_faints
            .iter()
            .find(|(s, sl, _)| *s == target_side && *sl == target_slot)
            .map(|&(_, _, i)| i)
    });
    let Some(def_idx) = def_idx else { return };
    if state.side(target_side).team_member(def_idx).ability != Abilities::CURSEDBODY {
        return;
    }
    // PS `source.volatiles['disable']`: the attacker's Disable as it was before
    // any faint in this hit (the slot is already empty if the attacker fainted).
    let att_idx = slot_index_or_unresolved(state, side, slot);
    let att_disabled = state.side(side).active()[slot].disabled.or_else(|| {
        att_idx.and_then(|ai| state.faint_disabled.iter().rev().find(|e| e.0 == side && e.1 == ai).and_then(|e| e.2))
    });
    // PS `addVolatile` fails on a fainted pokemon: no Disable for a KO'd attacker.
    let att_alive = att_idx.is_some_and(|ai| state.side(side).team_member(ai).hp > 0);
    if att_disabled.is_some() || choice.move_id == Choices::STRUGGLE {
        return;
    }
    let hit = rng.label("ability").random_chance(3, 10);
    // PS `addVolatile('disable')` -> `TryAddVolatile` -> `onAllyTryAddVolatile`
    // (data/abilities.ts aromaveil): an Aroma Veil on the attacker's side (the
    // holder or its ally) interrupts it. The roll above still happens.
    let aroma_veil = aroma_veil_blocks(state, side, AromaVeiled::Disable, None);
    // Disable `condition.onStart` (PS data/moves.ts disable): the attacker's
    // move with no PP left is "Move out of PP" and the volatile does not start.
    // The move's PP is already spent when this runs.
    let out_of_pp = match slot_index_or_unresolved(state, side, slot) {
        Some(ai) => {
            let p = state.side(side).team_member(ai);
            [
                crate::state::PokemonMoveIndex::M0,
                crate::state::PokemonMoveIndex::M1,
                crate::state::PokemonMoveIndex::M2,
                crate::state::PokemonMoveIndex::M3,
            ]
            .iter()
            .any(|mi| p.moves[mi].id == choice.move_id && p.moves[mi].pp == 0)
        }
        None => false,
    };
    if hit && !aroma_veil && !out_of_pp && att_alive {
        state.side_mut(side).active_slot_mut(slot).disabled = Some((choice.move_id, 4));
    }
}

/// The pokemon in a slot, or the one that fainted there in the current hit
/// loop and is not resolved yet (`unresolved_faints`).
fn slot_index_or_unresolved(state: &DoublesState, side: Side, slot: usize) -> Option<PokemonIndex> {
    state.side(side).active()[slot].pokemon_index.or_else(|| {
        state
            .unresolved_faints
            .iter()
            .find(|(s, sl, _)| *s == side && *sl == slot)
            .map(|&(_, _, i)| i)
    })
}

/// PS `pokemon.fainted`: hp is 0 and the faint has been resolved. A pokemon
/// KO'd by the current hit still counts as alive here.
fn fainted_now(state: &DoublesState, side: Side, idx: PokemonIndex) -> bool {
    state.side(side).team_member(idx).hp <= 0
        && !state.unresolved_faints.iter().any(|&(s, _, i)| s == side && i == idx)
}

/// Wandering Spirit (PS `wanderingspirit.onDamagingHit` -> `skillSwap`,
/// sim/battle.ts): a contact move that damages the holder swaps the
/// attacker's and holder's abilities, unless either is fainted or either
/// ability has the `failskillswap` flag.
fn maybe_wandering_spirit(
    state: &mut DoublesState,
    att_side: Side,
    att_slot: usize,
    def_side: Side,
    def_slot: usize,
    choice: &Choice,
    rng: &mut Gen5Rng,
) {
    if !choice.flags.contact {
        return;
    }
    // The holder may have been KO'd by this hit (its slot is already empty).
    let Some(di) = slot_index_or_unresolved(state, def_side, def_slot) else {
        return;
    };
    if state.side(def_side).team_member(di).ability != Abilities::WANDERINGSPIRIT {
        return;
    }
    skill_swap(state, att_side, att_slot, def_side, def_slot, rng);
}

/// PS ability `onUpdate` cures after the holder's ability changed (Skill Swap,
/// Entrainment): Own Tempo ends confusion. No draws.
fn ability_update_cures(state: &mut DoublesState, side: Side, slot: usize) {
    let Some(idx) = state.side(side).active()[slot].pokemon_index else { return };
    let (ability, status, hp) = {
        let p = state.side(side).team_member(idx);
        (p.ability, p.status, p.hp)
    };
    // PS flashfire `onEnd` -> `removeVolatile('flashfire')` when the ability
    // changes or is suppressed (Skill Swap, Entrainment, Receiver copies).
    if ability != Abilities::FLASHFIRE && state.side(side).active()[slot].flash_fire {
        state.side_mut(side).active_slot_mut(slot).flash_fire = false;
    }
    if hp <= 0 {
        return;
    }
    // Status-immunity abilities `onUpdate` (data/abilities.ts limber,
    // insomnia, vitalspirit, immunity, magmaarmor, waterbubble,
    // thermalexchange): `cureStatus()`.
    let cures = match ability {
        Abilities::LIMBER => status == PokemonStatus::PARALYZE,
        Abilities::INSOMNIA | Abilities::VITALSPIRIT => status == PokemonStatus::SLEEP,
        Abilities::IMMUNITY => matches!(status, PokemonStatus::POISON | PokemonStatus::TOXIC),
        Abilities::MAGMAARMOR => status == PokemonStatus::FREEZE,
        Abilities::WATERBUBBLE | Abilities::THERMALEXCHANGE => status == PokemonStatus::BURN,
        _ => false,
    };
    if cures {
        let pm = state.side_mut(side).team_member_mut(idx);
        pm.status = PokemonStatus::NONE;
        pm.sleep_turns = 0;
    }
    match ability {
        Abilities::OWNTEMPO => {
            if state.side(side).active()[slot].confusion > 0 {
                state.side_mut(side).active_slot_mut(slot).confusion = 0
            }
        }
        // oblivious.onUpdate: removes attract and taunt.
        Abilities::OBLIVIOUS => {
            let a = state.side_mut(side).active_slot_mut(slot);
            a.attract = None;
            a.taunt = 0;
        }
        _ => {}
    }
}

/// PS `battle.skillSwap(source, target)` (sim/battle.ts): fails if either is
/// fainted or either ability has `failskillswap`; swaps the abilities, then
/// runs the new abilities' Start events (the target's first).
fn skill_swap(
    state: &mut DoublesState,
    src_side: Side,
    src_slot: usize,
    tgt_side: Side,
    tgt_slot: usize,
    rng: &mut Gen5Rng,
) -> bool {
    let (Some(si), Some(ti)) = (
        slot_index_or_unresolved(state, src_side, src_slot),
        slot_index_or_unresolved(state, tgt_side, tgt_slot),
    ) else {
        return false;
    };
    // PS skillSwap reads `getAbility()`: a Gastro Acid-suppressed ability
    // is still swapped (and the volatile keeps the new one suppressed).
    let src_ability = if state.side(src_side).active()[src_slot].gastro_acid {
        held_ability(state, src_side, src_slot, si)
    } else {
        state.side(src_side).team_member(si).ability
    };
    let tgt_ability = held_ability(state, tgt_side, tgt_slot, ti);
    // PS `fainted` is only set once faints resolve, so a holder KO'd by the
    // current hit still swaps.
    if fainted_now(state, src_side, si)
        || fainted_now(state, tgt_side, ti)
        || crate::engine::mechanics::fails_skill_swap(src_ability)
        || crate::engine::mechanics::fails_skill_swap(tgt_ability)
    {
        return false;
    }
    // PS sim/battle.ts skillSwap: `singleEvent('End', sourceAbility)` then
    // `singleEvent('End', targetAbility)` before the swap. A Cloud Nine / Air
    // Lock `onEnd` (data/abilities.ts) runs `eachEvent('WeatherChange')`, a
    // speed sort of all actives.
    for ab in [src_ability, tgt_ability] {
        if matches!(ab, Abilities::CLOUDNINE | Abilities::AIRLOCK) {
            let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
        }
    }
    state.side_mut(src_side).team_member_mut(si).ability = tgt_ability;
    state.side_mut(tgt_side).team_member_mut(ti).ability = src_ability;
    for (sd, sl, i) in [(src_side, src_slot, si), (tgt_side, tgt_slot, ti)] {
        if state.side(sd).active()[sl].gastro_acid && state.side(sd).active()[sl].pokemon_index == Some(i) {
            let new = state.side(sd).team_member(i).ability;
            state.side_mut(sd).slot_scratch_mut(sl).gastro_ability = new;
            state.side_mut(sd).team_member_mut(i).ability = Abilities::NONE;
        }
    }
    reset_protean_effect(state, src_side, si);
    reset_protean_effect(state, tgt_side, ti);
    // The new abilities' Start events (Trace's draw included) run through
    // `apply_on_switch_abilities`.
    // TODO(PS parity): the TerrainChange speed sort after a swapped-in Surge.
    // A Start event that sets weather / terrain (Sand Stream, Cloud Nine, ...)
    // is followed by PS `eachEvent('WeatherChange')` / `TerrainChange`, a speed
    // sort of all actives (the same as the switch-in loop does).
    if apply_on_switch_abilities(state, tgt_side, tgt_slot, rng) {
        let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
    }
    // Mimicry `onStart` -> `singleEvent('TerrainChange')` (PS data/abilities.ts
    // mimicry): a holder that just gained Mimicry takes the terrain's type.
    refresh_mimicry_slot(state, tgt_side, tgt_slot);
    if apply_on_switch_abilities(state, src_side, src_slot, rng) {
        let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
    }
    refresh_mimicry_slot(state, src_side, src_slot);
    true
}


/// Synchronize: if a foe's move inflicts par/brn/psn/tox on a Synchronize
/// user, the move's user gets the same status (PS `synchronize.onAfterSetStatus`,
/// data/abilities.ts:4859). Sleep / freeze are not copied; entry hazards do
/// not trigger it (the engine only calls this for move-inflicted status).
fn maybe_synchronize(
    state: &mut DoublesState,
    target_side: Side,
    target_idx: PokemonIndex,
    source_side: Side,
    source_idx: PokemonIndex,
    status: PokemonStatus,
) {
    if matches!(status, PokemonStatus::SLEEP | PokemonStatus::FREEZE) {
        return;
    }
    // PS: only `source === target` is excluded (an ally's move counts too).
    if source_side == target_side && source_idx == target_idx {
        return;
    }
    let has_sync = state.side(target_side).team_member(target_idx).ability == Abilities::SYNCHRONIZE;
    if !has_sync {
        return;
    }
    // `source.trySetStatus(status, target)`: type immunity, status-blocking
    // abilities, Safeguard / Misty Terrain (the holder is another pokemon).
    let src_slot = (0..2).find(|&sl| state.side(source_side).active()[sl].pokemon_index == Some(source_idx));
    let tgt_slot = (0..2).find(|&sl| state.side(target_side).active()[sl].pokemon_index == Some(target_idx));
    if let (Some(ss), Some(ts)) = (src_slot, tgt_slot) {
        if status_immune_ex(state, source_side, ss, status, true, target_side, ts, false, true, false) {
            return;
        }
    }
    let s = state.side_mut(source_side);
    let pm = s.team_member_mut(source_idx);
    // PS `setStatus`: a fainted pokemon can't be statused (`!this.hp`).
    if pm.status == PokemonStatus::NONE && pm.hp > 0 {
        pm.status = status;
        if status == PokemonStatus::SLEEP {
            pm.sleep_turns = 1;
        }
        if status == PokemonStatus::TOXIC {
            s.toxic_stage[source_idx as usize] = 1;
        }
    }
}

/// Apply a `StatBoosts` to a team member's stored boosts, clamped to [-6, 6].
///
/// `source_is_foe` = true when the boost was caused by a Pokémon on the
/// opposing side. If a foe lowers at least one stat of a Defiant user, its
/// Attack rises by +2 (PS `defiant.onAfterEachBoost`, data/abilities.ts:901).
/// Mirror Armor `onTryBoost` (PS data/abilities.ts mirrorarmor): a stat drop
/// from another pokemon is bounced onto its source while the holder's stat is
/// above -6; the bounced part does not apply to the holder. Returns what still
/// applies to the holder.
fn mirror_armor_bounce(
    state: &mut DoublesState,
    side: Side,
    slot: usize,
    ts: Side,
    tidx: PokemonIndex,
    b: &StatBoosts,
) -> StatBoosts {
    let holder = state.side(ts).team_member(tidx).ability;
    if holder != Abilities::MIRRORARMOR {
        return b.clone();
    }
    // The drop is removed from the holder whatever the source's HP; only the
    // bounce needs a living source (`if (source.hp) this.boost(...)`). A source
    // whose faint already emptied its slot gets nothing either.
    let sidx = state.side(side).active()[slot].pokemon_index.filter(|&i| state.side(side).team_member(i).hp > 0);
    let cur = state.side(ts).team_boosts[tidx as usize].clone();
    let now = [cur.atk, cur.def, cur.spa, cur.spd, cur.spe, cur.acc, cur.eva];
    let mut rest = [b.attack, b.defense, b.special_attack, b.special_defense, b.speed, b.accuracy, b.evasion];
    let mut bounced = [0i8; 7];
    for n in 0..7 {
        if rest[n] < 0 && now[n] > -6 {
            bounced[n] = rest[n];
            rest[n] = 0;
        }
    }
    let bounced = StatBoosts {
        attack: bounced[0],
        defense: bounced[1],
        special_attack: bounced[2],
        special_defense: bounced[3],
        speed: bounced[4],
        accuracy: bounced[5],
        evasion: bounced[6],
    };
    if let Some(sidx) = sidx {
        apply_boosts_from(state, side, sidx, &bounced, ts != side, true);
    }
    StatBoosts {
        attack: rest[0],
        defense: rest[1],
        special_attack: rest[2],
        special_defense: rest[3],
        speed: rest[4],
        accuracy: rest[5],
        evasion: rest[6],
    }
}

/// PS data/abilities.ts opportunist `onAnyAfterMove` / `onAnySwitchIn` /
/// `onResidual`: apply the boosts the holder copied from its foes, then clear
/// them. Only a living holder with the ability applies them (`onEnd` clears).
fn flush_opportunist(state: &mut DoublesState, side: Side, slot: usize) {
    let pending = state.side(side).active()[slot].opportunist_boosts;
    if pending == [0; 7] {
        return;
    }
    state.side_mut(side).slot_scratch_mut(slot).opportunist_boosts = [0; 7];
    let Some(idx) = state.side(side).active()[slot].pokemon_index else { return };
    if !state.side(side).slot_pokemon(slot).is_some_and(|p| p.hp > 0 && p.ability == Abilities::OPPORTUNIST) {
        return;
    }
    let copy = StatBoosts {
        attack: pending[0],
        defense: pending[1],
        special_attack: pending[2],
        special_defense: pending[3],
        speed: pending[4],
        accuracy: pending[5],
        evasion: pending[6],
    };
    state.tl.opportunist_busy.set(true);
    apply_boosts(state, side, idx, &copy, false);
    state.tl.opportunist_busy.set(false);
}

fn apply_boosts(
    state: &mut DoublesState,
    side: Side,
    idx: PokemonIndex,
    b: &StatBoosts,
    source_is_foe: bool,
) {
    apply_boosts_from(state, side, idx, b, source_is_foe, source_is_foe);
}

/// `apply_boosts` with the source split in two: `source_is_foe` gates the
/// Defiant / Competitive reaction (foe-only), `source_is_other` gates the
/// stat-drop blockers. PS `TryBoost` abilities (Clear Body, Hyper Cutter, ...)
/// check `source && target === source`, so an ally's drop is blocked too.
fn apply_boosts_from(
    state: &mut DoublesState,
    side: Side,
    idx: PokemonIndex,
    b: &StatBoosts,
    source_is_foe: bool,
    source_is_other: bool,
) -> bool {
    // PS `boost` (sim/battle.ts): `if (!target?.hp) return 0`. The engine
    // resolves a Destiny Bond KO of the user at once, while PS queues it after
    // the move's boosts and then clears them (`clearVolatile`). Skipping the
    // boost here leaves the same result.
    if state.side(side).team_member(idx).hp == 0 {
        return false;
    }
    // PS `boost` (sim/battle.ts): `if (this.gen > 5 && !target.side.foePokemonLeft())
    // return false`. Once the other side has no Pokemon left, nothing is boosted
    // (a self-boost after the last KO, e.g. Scale Shot's selfBoost).
    // `pokemonLeft` drops when a faint is processed (`faintMessages`), so a
    // pokemon that fainted this turn but is still unresolved still counts.
    let other = side.other();
    let left = state.side(other).team.iter().enumerate().any(|(i, m)| {
        m.id != crate::pokemon::PokemonName::NONE
            && (m.hp > 0
                || state
                    .unresolved_faints
                    .iter()
                    .any(|&(fs, _, pi)| fs == other && pi as usize == i))
    });
    if !left {
        return false;
    }
    // PS `boost`: stat by stat; `AfterEachBoost` (Defiant / Competitive, shared
    // `mechanics::stat_drop_reaction`) fires after each stat that actually
    // changed by a drop.
    let reaction = if source_is_foe {
        crate::engine::mechanics::stat_drop_reaction(state.side(side).team_member(idx).ability)
    } else {
        None
    };
    let mut requested = [
        b.attack,
        b.defense,
        b.special_attack,
        b.special_defense,
        b.speed,
        b.accuracy,
        b.evasion,
    ];
    // PS `runEvent('ChangeBoost')`: Contrary inverts, Simple doubles
    // (data/abilities.ts contrary / simple `onChangeBoost`).
    // TODO(PS parity): Mold Breaker bypass.
    let holder_ability = state.side(side).team_member(idx).ability;
    match holder_ability {
        Abilities::CONTRARY => requested.iter_mut().for_each(|r| *r = -*r),
        Abilities::SIMPLE => requested.iter_mut().for_each(|r| *r *= 2),
        _ => {}
    }
    // PS `runEvent('TryBoost')`: stat-drop blockers, only for boosts from
    // another pokemon (`if (source && target === source) return`).
    if source_is_other {
        // Flower Veil `onAllyTryBoost`: Grass types (holder and ally) can't have
        // stats lowered by others. PS data/abilities.ts flowerveil.
        let grass = {
            let m = state.side(side).team_member(idx);
            m.types.0 == PokemonType::GRASS || m.types.1 == PokemonType::GRASS
        };
        let veil = grass
            && (0..2).any(|sl| {
                state
                    .side(side)
                    .slot_pokemon(sl)
                    .is_some_and(|p| p.hp > 0 && p.ability == Abilities::FLOWERVEIL)
            });
        let blocks_all = veil
            || matches!(
                holder_ability,
                Abilities::CLEARBODY | Abilities::WHITESMOKE | Abilities::FULLMETALBODY
            );
        for (n, r) in requested.iter_mut().enumerate() {
            let blocked = blocks_all
                || (n == 0 && holder_ability == Abilities::HYPERCUTTER)
                || (n == 1 && holder_ability == Abilities::BIGPECKS)
                || (n == 5
                    && matches!(
                        holder_ability,
                        Abilities::KEENEYE | Abilities::ILLUMINATE | Abilities::MINDSEYE
                    ));
            if *r < 0 && blocked {
                *r = 0;
            }
        }
    }
    let mut lowered = false;
    // PS `boost` returns true when any requested stat changed. Defiant's
    // reaction is a separate boost and does not count (it can cancel the drop).
    let mut any_changed = false;
    let mut gained = [0i8; 7];
    for (n, &req) in requested.iter().enumerate() {
        if req == 0 {
            continue;
        }
        let s = state.side_mut(side);
        let tb = &mut s.team_boosts[idx as usize];
        let cur = match n {
            0 => &mut tb.atk,
            1 => &mut tb.def,
            2 => &mut tb.spa,
            3 => &mut tb.spd,
            4 => &mut tb.spe,
            5 => &mut tb.acc,
            _ => &mut tb.eva,
        };
        let old = *cur;
        *cur = (old + req).clamp(-6, 6);
        let changed = *cur != old;
        any_changed |= changed;
        if changed && req > 0 {
            gained[n] = *cur - old;
        }
        lowered |= changed && req < 0;
        if changed && req < 0 {
            if let Some((stat, amount)) = reaction {
                // The reaction is its own PS `boost` (effect Competitive / Defiant):
                // a raise it makes is copied by Opportunist like any other.
                match stat {
                    PokemonBoostableStat::Attack => {
                        let before = tb.atk;
                        tb.atk = (tb.atk + amount).clamp(-6, 6);
                        gained[0] += (tb.atk - before).max(0);
                    }
                    PokemonBoostableStat::SpecialAttack => {
                        let before = tb.spa;
                        tb.spa = (tb.spa + amount).clamp(-6, 6);
                        gained[2] += (tb.spa - before).max(0);
                    }
                    _ => {}
                }
            }
        }
    }
    // statsLoweredThisTurn (PS `boost`): set when a drop actually applied.
    if lowered {
        let s = state.side_mut(side);
        for sl in 0..2 {
            let a = s.slot_scratch_mut(sl);
            if a.pokemon_index == Some(idx) {
                a.stats_lowered_this_turn = true;
            }
        }
    }
    // Opportunist `onFoeAfterBoost`: each foe holder adds the positive raises to
    // its `effectState.boosts` (not for boosts it caused itself). They are
    // applied later by `flush_opportunist`. PS data/abilities.ts opportunist.
    if gained.iter().any(|&g| g > 0) && !state.tl.opportunist_busy.get() {
        let foe = side.other();
        for fsl in 0..2 {
            let holder = state.side(foe).slot_pokemon(fsl).filter(|p| p.hp > 0 && p.ability == Abilities::OPPORTUNIST);
            if holder.is_some() {
                let pending = &mut state.side_mut(foe).slot_scratch_mut(fsl).opportunist_boosts;
                for (i, &g) in gained.iter().enumerate() {
                    if g > 0 {
                        pending[i] = pending[i].saturating_add(g);
                    }
                }
            }
        }
    }
    // statsRaisedThisTurn (PS `boost`): `getCappedBoost` runs first, so a
    // positive request on a stat already at +6 does not count; the flag is set
    // when a positive boost actually applied (`gained`, see the loop above).
    let raised = gained.iter().any(|&g| g > 0);
    if raised {
        let s = state.side_mut(side);
        for sl in 0..2 {
            let a = s.slot_scratch_mut(sl);
            if a.pokemon_index == Some(idx) {
                a.stats_raised_this_turn = true;
            }
        }
    }
    any_changed
}

/// Abilities with the `notrace` flag (PS `data/abilities.ts`): Trace cannot
/// copy them.
fn has_notrace(a: Abilities) -> bool {
    crate::doubles::ability_flags::has_flag(a, crate::doubles::ability_flags::NOTRACE)
}

/// Run the Start effects of the ability of the pokemon in `side`'s
/// `active[slot]` (switch-in, Skill Swap): Trace, weather / terrain setters,
/// Curious Medicine, Intimidate, Supersweet Syrup. Returns true when it set a
/// terrain (the caller runs PS's TerrainChange speed sort).
// TODO(PS parity): other Start abilities (Download, Frisk, ...).
/// Mimicry (PS data/abilities.ts `onTerrainChange`): the holder's types follow
/// the terrain (Electric / Grass / Fairy / Psychic) and go back to the species
/// types when no terrain is up. Runs on switch-in (`onStart`) and after every
/// terrain change (`eachEvent('TerrainChange')`). Types are restored on
/// switch-out through `type_change`, like Libero / Protean.
fn refresh_mimicry(state: &mut DoublesState) {
    for side in [Side::One, Side::Two] {
        for slot in 0..2 {
            refresh_mimicry_slot(state, side, slot);
        }
    }
}

/// Mimicry for one active slot. PS `onStart` (switch-in) and `onTerrainChange`
/// (`eachEvent('TerrainChange')`, all actives) are the only callers: a switch-in
/// must not reset the types of the other actives (Magic Powder, Soak).
fn refresh_mimicry_slot(state: &mut DoublesState, side: Side, slot: usize) {
    let terrain = state.terrain.terrain_type;
    let s = state.side_mut(side);
    let Some(idx) = s.active()[slot].pokemon_index else { return };
    let p = s.team_member(idx);
    if p.ability != Abilities::MIMICRY {
        return;
    }
    let want = match crate::engine::mechanics::terrain_pulse_type(terrain, true) {
        Some(t) => (t, PokemonType::TYPELESS),
        None => p.base_types,
    };
    let cur = p.types;
    if cur == want {
        return;
    }
    if s.active()[slot].type_change.is_none() {
        s.slot_scratch_mut(slot).type_change = Some(cur);
    }
    s.team_member_mut(idx).types = want;
}

fn apply_on_switch_abilities(
    state: &mut DoublesState,
    side: Side,
    slot: usize,
    rng: &mut Gen5Rng,
) -> bool {
    // A holder KO'd by the current hit still runs its Start event (PS
    // `singleEvent('Start')` after Skill Swap ignores `fainted`), so the
    // pokemon comes from `unresolved_faints` when its slot is already empty.
    let mut ability = {
        let s = state.side(side);
        match slot_index_or_unresolved(state, side, slot) {
            Some(idx) => s.team_member(idx).ability,
            None => return false,
        }
    };
    // Supreme Overlord `onStart`: `fallen = min(side.totalFainted, 5)`.
    if ability == Abilities::SUPREMEOVERLORD {
        let sd = state.side(side);
        let fainted = sd.team.iter().enumerate().filter(|(i, p)| {
            p.id != crate::pokemon::PokemonName::NONE && p.hp <= 0 && !sd.species_names[*i].is_empty()
        }).count();
        state.side_mut(side).slot_scratch_mut(slot).fallen = fainted.min(5) as u8;
    }
    // Trace (`trace.onStart` -> `onUpdate`): copies a RANDOM adjacent foe's
    // ability, drawing `sample(possibleTargets)`. Foes with the `notrace`
    // flag or No Ability are excluded. Must happen before the copied
    // ability's own on-switch effects run.
    if ability == Abilities::TRACE {
        // `onStart`: a foe with No Ability stops Trace (`seek = false`), then
        // `singleEvent('Update')` runs `onUpdate` once. A Trace with no valid
        // foe keeps seeking: `each_event_update` retries on every Update.
        let foe = side.other();
        let blocked = foe_active_indices(state, foe)
            .iter()
            .any(|&fidx| state.side(foe).team_member(fidx).ability == Abilities::NOABILITY);
        state.side_mut(side).slot_scratch_mut(slot).trace_seek = !blocked;
        if !blocked {
            if let Some(copied) = trace_copy(state, side, slot, rng) {
                ability = copied;
            }
        }
    }
    // Forewarn `onStart` (PS data/abilities.ts forewarn): `sample` of the
    // foes' highest-power moves (one draw when any candidate exists).
    if ability == Abilities::FOREWARN {
        let foe = side.other();
        let mut count = 0u32;
        let mut warn_bp = 1.0f32;
        for f in 0..2 {
            let Some(fidx) = state.side(foe).active()[f].pokemon_index else { continue };
            let p = state.side(foe).team_member(fidx);
            if p.hp <= 0 {
                continue;
            }
            for m in [&p.moves.m0, &p.moves.m1, &p.moves.m2, &p.moves.m3] {
                if m.id == Choices::NONE {
                    continue;
                }
                let c = &m.choice;
                let mut bp = c.base_power;
                if c.ohko {
                    bp = 150.0;
                }
                if matches!(m.id, Choices::COUNTER | Choices::METALBURST | Choices::MIRRORCOAT) {
                    bp = 120.0;
                }
                if bp == 1.0 {
                    bp = 80.0;
                }
                if bp == 0.0 && c.category != MoveCategory::Status {
                    bp = 80.0;
                }
                if bp > warn_bp {
                    count = 1;
                    warn_bp = bp;
                } else if bp == warn_bp {
                    count += 1;
                }
            }
        }
        if count > 0 {
            let _ = rng.label("sample").random(count);
        }
    }
    // Weather / terrain setters (shared table with the singles engine).
    // TODO(PS parity): fails if the same weather / terrain is already up
    // (no duration reset), primal weathers, Cloud Nine.
    match crate::engine::mechanics::switch_in_field(ability) {
        Some(crate::engine::mechanics::SwitchInField::Weather(w)) => {
            if state.weather.weather_type != w {
                state.weather = StateWeather {
                    weather_type: w,
                    turns_remaining: crate::engine::abilities::WEATHER_ABILITY_TURNS,
                };
                // PS `setWeather` -> `eachEvent('WeatherChange')`: the caller sorts.
                return true;
            }
            return false;
        }
        Some(crate::engine::mechanics::SwitchInField::Terrain(t)) => {
            if state.terrain.terrain_type != t {
                state.terrain = StateTerrain {
                    terrain_type: t,
                    turns_remaining: 5,
                };
                // PS `setTerrain` -> `eachEvent("TerrainChange")`: a speed sort
                // of all actives. The caller performs it (label depends on context).
                refresh_mimicry(state);
                return true;
            }
            return false;
        }
        None => {}
    }

    // Cloud Nine / Air Lock (`onSwitchIn` -> `onStart`, PS
    // data/abilities.ts): `eachEvent('WeatherChange')` — a speed sort of all
    // actives, drawn even with no weather up. The caller performs the sort.
    if matches!(ability, Abilities::CLOUDNINE | Abilities::AIRLOCK) {
        return true;
    }

    // Curious Medicine (`curiousmedicine.onStart`): clear the adjacent ally's
    // stat changes. Doubles-only (singles has no ally).
    if ability == Abilities::CURIOUSMEDICINE {
        if let Some(ally) = state.side(side).active()[1 - slot].pokemon_index {
            state.side_mut(side).clear_member_boosts(ally);
        }
        return false;
    }
    // Screen Cleaner `onStart` (data/abilities.ts screencleaner): removes
    // Reflect, Light Screen and Aurora Veil from both sides. No draws.
    if ability == Abilities::SCREENCLEANER {
        for sd in [side, side.other()] {
            let sc = &mut state.side_mut(sd).side_conditions;
            sc.reflect = 0;
            sc.light_screen = 0;
            sc.aurora_veil = 0;
        }
        return false;
    }
    // Hospitality `onStart` (data/abilities.ts): heal each adjacent ally by
    // baseMaxhp / 4 (Heal Block blocks it). The heal is a plain integer
    // division (PS `heal` floors it: 155 / 4 -> 38).
    if ability == Abilities::HOSPITALITY {
        let ally_slot = &state.side(side).active()[1 - slot];
        if let Some(ally) = ally_slot.pokemon_index {
            let heal_blocked = ally_slot.heal_block > 0;
            let p = state.side_mut(side).team_member_mut(ally);
            if p.hp > 0 && !heal_blocked {
                p.hp = (p.hp + p.maxhp / 4).min(p.maxhp);
            }
        }
        return false;
    }
    // Intimidate (-1 Atk) / Supersweet Syrup (-1 evasion, once per battle)
    // on each adjacent foe not behind a Substitute.
    let drop = match ability {
        Abilities::INTIMIDATE => StatBoosts {
            attack: -1,
            ..Default::default()
        },
        Abilities::SUPERSWEETSYRUP => {
            let me = slot_index_or_unresolved(state, side, slot).unwrap();
            if state.side(side).syrup_triggered[me as usize] {
                return false;
            }
            state.side_mut(side).syrup_triggered[me as usize] = true;
            StatBoosts {
                evasion: -1,
                ..Default::default()
            }
        }
        _ => return false,
    };
    let foe = match side {
        Side::One => Side::Two,
        Side::Two => Side::One,
    };
    let targets: Vec<PokemonIndex> = state
        .side(foe)
        .active()
        .iter()
        .filter(|a| a.pokemon_index.is_some() && a.substitute_health <= 0)
        .filter_map(|a| a.pokemon_index)
        .collect();
    for fidx in targets {
        let before = state.side(foe).team_boosts[fidx as usize].atk;
        // Intimidate immunity (`onTryBoost`, effect.name === 'Intimidate'):
        // Own Tempo / Oblivious / Inner Focus / Scrappy (data/abilities.ts);
        // Guard Dog turns it into +1 Atk.
        if ability == Abilities::INTIMIDATE {
            match state.side(foe).team_member(fidx).ability {
                Abilities::OWNTEMPO | Abilities::OBLIVIOUS | Abilities::INNERFOCUS | Abilities::SCRAPPY => continue,
                Abilities::GUARDDOG => {
                    apply_boosts(state, foe, fidx, &StatBoosts { attack: 1, ..Default::default() }, false);
                    continue;
                }
                _ => {}
            }
        }
        // Mirror Armor `onBoost`: a drop from another pokemon is reflected onto
        // the source (unless the stat is already at -6). PS data/abilities.ts.
        // TODO(PS parity): Mold Breaker, multi-stat drops with some stats at -6.
        if state.side(foe).team_member(fidx).ability == Abilities::MIRRORARMOR
            && state.side(foe).team_boosts[fidx as usize].atk > -6
        {
            if let Some(src) = state.side(side).active()[slot].pokemon_index {
                apply_boosts(state, side, src, &drop, true);
            }
            continue;
        }
        apply_boosts(state, foe, fidx, &drop, true);
        // Rattled `onAfterBoost`: +1 Speed after an Intimidate Atk drop.
        if ability == Abilities::INTIMIDATE
            && state.side(foe).team_member(fidx).ability == Abilities::RATTLED
            && state.side(foe).team_boosts[fidx as usize].atk != before
        {
            apply_boosts(state, foe, fidx, &StatBoosts { speed: 1, ..Default::default() }, false);
        }
    }
    false
}

/// Forecast (`onUpdate`, data/abilities.ts): Castform's type follows the
/// weather (sun Fire, rain Water, hail / snow Ice, otherwise Normal).
fn update_forecast(state: &mut DoublesState) {
    let weather = effective_weather(state);
    let t = match weather {
        Weather::SUN | Weather::HARSHSUN => PokemonType::FIRE,
        Weather::RAIN | Weather::HEAVYRAIN => PokemonType::WATER,
        Weather::HAIL | Weather::SNOW => PokemonType::ICE,
        _ => PokemonType::NORMAL,
    };
    for sd in [Side::One, Side::Two] {
        for sl in 0..2 {
            if let Some(idx) = state.side(sd).active()[sl].pokemon_index {
                // PS `onWeatherChange` only runs on a weather change (or the
                // holder's switch-in): the type is not re-set while the weather
                // is unchanged, so a Soak / Magic Powder type sticks.
                if state.side(sd).active()[sl].forecast_weather == Some(weather) {
                    continue;
                }
                let pm = state.side_mut(sd).team_member_mut(idx);
                // PS data/abilities.ts forecast.onUpdate: only Castform (any forme)
                // changes type; a Forecast holder of another species (Skill Swap,
                // Role Play) does nothing.
                let is_castform = matches!(
                    pm.id,
                    crate::pokemon::PokemonName::CASTFORM
                        | crate::pokemon::PokemonName::CASTFORMSUNNY
                        | crate::pokemon::PokemonName::CASTFORMRAINY
                        | crate::pokemon::PokemonName::CASTFORMSNOWY
                );
                if pm.ability == Abilities::FORECAST && pm.hp > 0 && is_castform {
                    pm.types = (t, PokemonType::TYPELESS);
                    // PS forecast.onWeatherChange -> `formeChange` (only when the
                    // forme differs) -> `setSpecies`: the cached speed becomes the
                    // raw stat (boosts / paralysis ignored until `updateSpeed`).
                    let forme = match t {
                        PokemonType::FIRE => crate::pokemon::PokemonName::CASTFORMSUNNY,
                        PokemonType::WATER => crate::pokemon::PokemonName::CASTFORMRAINY,
                        PokemonType::ICE => crate::pokemon::PokemonName::CASTFORMSNOWY,
                        _ => crate::pokemon::PokemonName::CASTFORM,
                    };
                    let changed = pm.id != forme;
                    pm.id = forme;
                    let s = state.side_mut(sd);
                    if changed {
                        restore_stored_stats(s, idx);
                    }
                    let raw = s.team_member(idx).speed;
                    s.slot_scratch_mut(sl).forecast_weather = Some(weather);
                    if changed {
                        s.speed_cache[idx as usize] = raw as _;
                    }
                }
            }
        }
    }
}

fn apply_side_condition(state: &mut DoublesState, side: Side, c: PokemonSideCondition) -> bool {
    let snow = matches!(effective_weather(state), Weather::HAIL | Weather::SNOW);
    let s = state.side_mut(side);
    let sc = &mut s.side_conditions;
    // PS `addSideCondition` fails on an existing condition (no `onSideRestart`);
    // layered hazards fail at their maximum.
    // TODO(PS parity): Aurora Veil's champions weather rule, Court Change.
    match c {
        PokemonSideCondition::Reflect if sc.reflect == 0 => sc.reflect = 5,
        PokemonSideCondition::LightScreen if sc.light_screen == 0 => sc.light_screen = 5,
        PokemonSideCondition::AuroraVeil if sc.aurora_veil == 0 && snow => sc.aurora_veil = 5,
        PokemonSideCondition::Tailwind if sc.tailwind == 0 => sc.tailwind = 4,
        PokemonSideCondition::Safeguard if sc.safeguard == 0 => sc.safeguard = 5,
        PokemonSideCondition::LuckyChant if sc.lucky_chant == 0 => sc.lucky_chant = 5,
        PokemonSideCondition::Mist if sc.mist == 0 => sc.mist = 5,
        PokemonSideCondition::Spikes if sc.spikes < 3 => sc.spikes += 1,
        PokemonSideCondition::ToxicSpikes if sc.toxic_spikes < 2 => sc.toxic_spikes += 1,
        PokemonSideCondition::Stealthrock if sc.stealth_rock == 0 => sc.stealth_rock = 1,
        PokemonSideCondition::StickyWeb if sc.sticky_web == 0 => sc.sticky_web = 1,
        PokemonSideCondition::QuickGuard if sc.quick_guard == 0 => sc.quick_guard = 1,
        PokemonSideCondition::WideGuard if sc.wide_guard == 0 => sc.wide_guard = 1,
        PokemonSideCondition::Reflect
        | PokemonSideCondition::LightScreen
        | PokemonSideCondition::AuroraVeil
        | PokemonSideCondition::Tailwind
        | PokemonSideCondition::Safeguard
        | PokemonSideCondition::LuckyChant
        | PokemonSideCondition::Mist
        | PokemonSideCondition::Spikes
        | PokemonSideCondition::ToxicSpikes
        | PokemonSideCondition::Stealthrock
        | PokemonSideCondition::QuickGuard
        | PokemonSideCondition::WideGuard
        | PokemonSideCondition::StickyWeb => return false,
        _ => {}
    }
    true
}

/// Put battle-team member `position` into `slot`, replacing whatever is
/// there (an alive occupant is switched out: PS `switchIn` ->
/// `clearVolatile` resets its boosts), and apply entry hazards. On-switch
/// abilities are applied by the caller.
/// Revival Blessing `revivalblessing` action: the fainted party member at
/// battle-team `position` returns with floor(maxhp / 2) HP (min 1) and no status.
fn revive_member(state: &mut DoublesState, side: Side, position: usize) {
    let s = state.side_mut(side);
    s.ver = s.ver.wrapping_add(1);
    let p = &mut s.team[position];
    p.hp = (p.maxhp / 2).max(1);
    p.status = PokemonStatus::NONE;
    p.sleep_turns = 0;
    s.toxic_stage[position] = 0;
}

fn switch_in(state: &mut DoublesState, side: Side, slot: usize, position: usize) {
    switch_in_inner(state, side, slot, position);
    entry_hazards(state, side, slot);
    refresh_mimicry_slot(state, side, slot);
    // Opportunist `onAnySwitchIn` (priority -3): every holder flushes.
    for sd in [Side::One, Side::Two] {
        for sl in 0..2 {
            flush_opportunist(state, sd, sl);
        }
    }
}

/// Healing Wish / Lunar Dance `slotCondition` `onSwitchIn` -> `onSwap`: a hurt
/// or statused replacement is fully healed and cured; otherwise the condition
/// keeps waiting. PS data/moves.ts healingwish / lunardance. The slot condition
/// (speed sort subOrder 3) runs before the side-condition hazards (subOrder 4),
/// so it is called before Spikes / Stealth Rock damage in `switch_in_inner`.
/// TODO(PS parity): Lunar Dance also restores PP (and triggers on PP).
fn healing_wish_on_switch_in(s: &mut DoublesSide, slot: usize) {
    if s.healing_wish_slot[slot] {
        if let Some(idx) = s.active()[slot].pokemon_index {
            let p = s.team_member_mut(idx);
            if p.hp > 0 && (p.hp < p.maxhp || p.status != PokemonStatus::NONE) {
                p.hp = p.maxhp;
                p.status = PokemonStatus::NONE;
                p.sleep_turns = 0;
                s.toxic_stage[idx as usize] = 0;
                s.healing_wish_slot[slot] = false;
            }
        }
    }
}

fn switch_in_inner(state: &mut DoublesState, side: Side, slot: usize, position: usize) {
    // PS `switchIn` swaps the two pokemon's places in `side.pokemon`.
    {
        let order = &mut state.side_mut(side).ps_order;
        if let Some(p) = order.iter().position(|&t| t == position) {
            order.swap(slot, p);
        }
    }
    if let Some(old) = state.side(side).active()[slot].pokemon_index {
        release_trapped_by(state, side, old);
        if state.side(side).active()[slot].power_trick {
            let pm = state.side_mut(side).team_member_mut(old);
            std::mem::swap(&mut pm.attack, &mut pm.defense);
        }
        // `SwitchOut` abilities (PS data/abilities.ts): Regenerator heals
        // floor(baseMaxhp / 3); Natural Cure clears the status.
        {
            let p = state.side_mut(side).team_member_mut(old);
            if p.hp > 0 {
                match p.ability {
                    Abilities::REGENERATOR => p.hp = (p.hp + p.maxhp / 3).min(p.maxhp),
                    Abilities::NATURALCURE if p.status != PokemonStatus::NONE => {
                        p.status = PokemonStatus::NONE;
                        p.sleep_turns = 0;
                    }
                    _ => {}
                }
            }
        }
        // Zero to Hero `onSwitchOut` (PS data/abilities.ts zerotohero): a
        // Palafin leaving the field becomes Palafin-Hero for good
        // (`formeChange(..., true)`), with the Hero stats.
        if state.side(side).team_member(old).id == crate::pokemon::PokemonName::PALAFIN
            && state.side(side).team_member(old).ability == Abilities::ZEROTOHERO
        {
            if let Some(data) = crate::doubles::species_types::species_data("palafinhero") {
                let pm = state.side(side).team_member(old);
                let (e, lvl, nat) = (pm.evs, pm.level as u8, pm.nature.clone());
                let evs = crate::doubles::state::Evs {
                    hp: e.0 as u16,
                    atk: e.1 as u16,
                    def: e.2 as u16,
                    spa: e.3 as u16,
                    spd: e.4 as u16,
                    spe: e.5 as u16,
                };
                let (_, atk, def, spa, spd, _) =
                    crate::doubles::state::compute_stats(data.base_stats, &evs, lvl, nat);
                restore_stored_stats(state.side_mut(side), old);
                let pm = state.side_mut(side).team_member_mut(old);
                pm.id = crate::pokemon::PokemonName::PALAFINHERO;
                pm.attack = atk;
                pm.defense = def;
                pm.special_attack = spa;
                pm.special_defense = spd;
            }
        }
        // PS `clearVolatile`: boosts reset, ability back to the base one, and
        // `setSpecies(baseSpecies)` (a Forecast forme reverts).
        let s = state.side_mut(side);
        s.clear_member_boosts(old);
        // `setSpecies(baseSpecies)`: Guard / Power Split and Speed Swap end
        // when the pokemon leaves (PS sim/battle-actions.ts switchIn -> clearVolatile).
        restore_stored_stats(s, old);
        let p = s.team_member_mut(old);
        p.ability = p.base_ability;
        if matches!(
            p.id,
            crate::pokemon::PokemonName::CASTFORMSUNNY
                | crate::pokemon::PokemonName::CASTFORMRAINY
                | crate::pokemon::PokemonName::CASTFORMSNOWY
        ) {
            p.id = crate::pokemon::PokemonName::CASTFORM;
        }
        // Morpeko-Hangry is a `battleOnly` forme too (Hunger Switch's
        // `formeChange` is not permanent): Morpeko comes back Full Belly.
        // PS data/abilities.ts hungerswitch, sim/pokemon.ts clearVolatile.
        if p.id == crate::pokemon::PokemonName::MORPEKOHANGRY {
            p.id = crate::pokemon::PokemonName::MORPEKO;
        }
        if let Some(t) = s.active_slot_mut(slot).roost_orig.take() {
            s.team_member_mut(old).types = t;
        }
        if let Some(t) = s.active()[slot].type_change {
            s.team_member_mut(old).types = t;
        }
        if s.active()[slot].added_grass {
            s.team_member_mut(old).types.1 = PokemonType::TYPELESS;
        }
    }
    let pos = position.min(5);
    let idx = PokemonIndex::deserialize(&pos.to_string());
    let alive = {
        let p = &state.side(side).team[pos];
        p.hp > 0 && p.id != crate::pokemon::PokemonName::NONE
    };
    let s = state.side_mut(side);
    *s.active_slot_mut(slot) = ActiveSlot::empty();
    s.fainted_in_slot[slot] = None;
    if alive {
        // PS `setSpecies` (switch-in): `attackedBy = []`.
        s.team_member_mut(idx).attacked_by.clear();
        s.damaged_by[idx as usize].clear();
        // PS battle-actions.ts switchIn: `moveSlot.used = false` for every move
        // of the incoming pokemon (Last Resort's `onTry` reads it).
        s.moves_used[idx as usize] = [false; 4];
        // `setSpecies`: `storedStats` back to the base stats (Guard / Power Split).
        restore_stored_stats(s, idx);
        // PS `clearVolatile` -> `setSpecies`: `pokemon.speed = storedStats.spe`
        // (the raw stat). A drag-in never calls `updateSpeed` before its
        // `runSwitch` sort (sim/battle.ts dragIn path), so this raw speed is
        // what the switch-in speed sort reads.
        let raw_speed = s.team_member(idx).speed;
        s.speed_cache[idx as usize] = raw_speed as _;
        s.active_slot_mut(slot).pokemon_index = Some(idx);
        // PS `clearVolatile` (switch-in): `newlySwitched = true`.
        s.slot_scratch_mut(slot).newly_switched = true;
        // PS `clearVolatile` -> `setSpecies(baseSpecies)`: Aegislash-Blade is
        // `battleOnly` (data/pokedex.ts), so every switch-in (drag included)
        // returns it to Shield forme with its stats recomputed.
        if s.team_member(idx).id == crate::pokemon::PokemonName::AEGISLASHBLADE {
            if let Some(data) = crate::doubles::species_types::species_data("aegislash") {
                let p = s.team_member(idx);
                let (e, lvl, nat) = (p.evs, p.level as u8, p.nature.clone());
                let evs = crate::doubles::state::Evs {
                    hp: e.0 as u16,
                    atk: e.1 as u16,
                    def: e.2 as u16,
                    spa: e.3 as u16,
                    spd: e.4 as u16,
                    spe: e.5 as u16,
                };
                let (_, atk, def, spa, spd, _) =
                    crate::doubles::state::compute_stats(data.base_stats, &evs, lvl, nat);
                let pm = s.team_member_mut(idx);
                pm.id = crate::pokemon::PokemonName::AEGISLASH;
                pm.attack = atk;
                pm.defense = def;
                pm.special_attack = spa;
                pm.special_defense = spd;
                let raw = pm.speed;
                s.speed_cache[idx as usize] = raw as _;
            }
        }
        // Same `setSpecies(baseSpecies)`: Hunger Switch's `formeChange` is not
        // permanent, so Morpeko-Hangry comes back in as Morpeko (both formes
        // share base stats and types). PS data/abilities.ts hungerswitch.
        if s.team_member(idx).id == crate::pokemon::PokemonName::MORPEKOHANGRY {
            s.team_member_mut(idx).id = crate::pokemon::PokemonName::MORPEKO;
        }
        // PS data/conditions.ts tox.onSwitchIn: the toxic stage resets to 0
        // when a poisoned pokemon switches in (first residual is 1x again).
        // This engine's residual computes damage before incrementing, so its
        // equivalent of PS stage 0 is 1.
        if s.team_member(idx).status == PokemonStatus::TOXIC {
            s.toxic_stage[idx as usize] = 1;
        }
        s.slot_scratch_mut(slot).illusion_target = if s.team_member(idx).ability == Abilities::ILLUSION {
            Some(illusion_disguise(s, idx))
        } else {
            None
        };
    }
}

/// PS `fieldEvent('SwitchIn')` for the pokemon now in `slot`: Healing Wish
/// (slot condition, subOrder 3), then the side-condition hazards (subOrder 4:
/// Stealth Rock, Spikes, Toxic Spikes), then the pokemon faints if the
/// hazards KO it. The caller decides when this runs: `switch_in` runs it right
/// after the placement, the replacement batch runs it in speed-sort order.
fn entry_hazards(state: &mut DoublesState, side: Side, slot: usize) {
    let Some(idx) = state.side(side).active()[slot].pokemon_index else {
        return;
    };
    // Read hazard layers + incoming pokemon data before mutating.
    let (spikes, toxic_spikes, types, alive) = {
        let s = state.side(side);
        let p = s.team_member(idx);
        (
            s.side_conditions.spikes,
            s.side_conditions.toxic_spikes,
            p.types,
            p.hp > 0 && p.id != crate::pokemon::PokemonName::NONE,
        )
    };
    let gravity_on = state.gravity.active;
    // PS toxicspikes.onSwitchIn: `trySetStatus(psn / tox, pokemon.side.foe.active[0])`,
    // so the status goes through every SetStatus check with a foe as the source
    // (Flower Veil, Pastel Veil, Leaf Guard, Misty Terrain, ...). Safeguard is
    // checked below (the hazard is not a Move, so Infiltrator does not bypass it).
    let tspikes_immune = toxic_spikes > 0
        && alive
        && status_immune_ex(
            state,
            side,
            slot,
            if toxic_spikes >= 2 { PokemonStatus::TOXIC } else { PokemonStatus::POISON },
            true,
            side.other(),
            0,
            false,
            false,
            false,
        );
    let s = state.side_mut(side);
    if alive {
        healing_wish_on_switch_in(s, slot);
        let boots = s.team_member(idx).item == crate::engine::items::Items::HEAVYDUTYBOOTS;
        // Magic Guard blocks indirect (non-move) damage, hazards included
        // (PS data/abilities.ts magicguard.onDamage).
        let magic_guard = s.team_member(idx).ability == Abilities::MAGICGUARD;
        let grounded = slot_grounded(s.team_member(idx), &s.active()[slot], gravity_on);
        // Stealth Rock `onSwitchIn`: maxhp * 2^typeMod / 8 (`damage` truncates,
        // min 1). Applied before Spikes (side-condition insertion order).
        let rock_dmg = if s.side_conditions.stealth_rock > 0 && !boots && !magic_guard {
            let mult = crate::engine::damage_calc::type_effectiveness_modifier(
                &PokemonType::ROCK,
                s.team_member(idx),
            );
            ((s.team_member(idx).maxhp as f32 * mult / 8.0) as i16).max(1)
        } else {
            0
        };
        let spikes_dmg = if grounded && spikes > 0 && !boots && !magic_guard {
            let mult = [0, 3, 4, 6][spikes.min(3) as usize];
            ((s.team_member(idx).maxhp as i32 * mult / 24) as i16).max(1)
        } else {
            0
        };
        let spikes_dmg = spikes_dmg + rock_dmg;
        // Safeguard `onSetStatus` blocks the hazard's status (source is a foe).
        // Misty Terrain `onSetStatus`: grounded pokemon can't gain a status.
        // Purifying Salt `onSetStatus` (PS data/abilities.ts purifyingsalt) blocks
        // the hazard's `trySetStatus`, while the spikes are still cleared by a
        // Poison type (PS toxicspikes `onSwitchIn`).
        let purifying_salt = s.team_member(idx).ability == Abilities::PURIFYINGSALT;
        // Immunity `onSetStatus` (PS data/abilities.ts immunity) refuses psn / tox,
        // which are the only statuses the hazard sets: the spikes stay up.
        let immunity = s.team_member(idx).ability == Abilities::IMMUNITY;
        let new_status = if grounded && !tspikes_immune && toxic_spikes > 0 && s.side_conditions.safeguard == 0 && !purifying_salt && !immunity && !boots {
            let cur = s.team_member(idx).status;
            if cur == PokemonStatus::NONE {
                if types.0 == PokemonType::POISON || types.1 == PokemonType::POISON {
                    None // poison types clear the spikes (handled below)
                } else if types.0 != PokemonType::STEEL && types.1 != PokemonType::STEEL {
                    Some(if toxic_spikes >= 2 { PokemonStatus::TOXIC } else { PokemonStatus::POISON })
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        // PS toxicspikes `onSwitchIn`: a grounded Poison type removes the layers
        // whatever its status (no status check; boots and Safeguard don't matter).
        let clears_toxicspikes = grounded
            && toxic_spikes > 0
            && (types.0 == PokemonType::POISON || types.1 == PokemonType::POISON);
        // Apply hazard damage.
        if spikes_dmg > 0 {
            let p = &mut s.team_member_mut(idx);
            p.hp -= spikes_dmg;
        }
        // Apply toxic spikes status. A pokemon KO'd by Stealth Rock / Spikes
        // has fainted before the Toxic Spikes handler runs (PS `fieldEvent`
        // calls `resolveFaints` after each handler and skips fainted holders).
        let survived = s.team_member(idx).hp > 0;
        if let Some(st) = new_status.filter(|_| survived) {
            let p = &mut s.team_member_mut(idx);
            p.status = st;
            if st == PokemonStatus::TOXIC {
                s.toxic_stage[idx as usize] = 1;
            }
        }
        if clears_toxicspikes && survived {
            s.side_conditions.toxic_spikes = 0;
        }
        if s.team_member(idx).hp <= 0 {
            s.team_member_mut(idx).hp = 0;
            faint_pokemon(state, side, slot);
        }
    }
}

/// Sticky Web `onSwitchIn` (runs in the `runSwitch` action, after the
/// switch-in speed sort): boost spe -1, sourced from the foe so Clear Body /
/// Defiant react. PS data/moves.ts stickyweb.condition.onSwitchIn.
fn apply_sticky_web(state: &mut DoublesState, side: Side, slot: usize) {
    let s = state.side(side);
    let Some(p) = s.slot_pokemon(slot) else { return };
    if p.hp <= 0 || s.side_conditions.sticky_web == 0 || !slot_grounded(p, &s.active()[slot], state.gravity.active) {
        return;
    }
    if p.item == crate::engine::items::Items::HEAVYDUTYBOOTS {
        return;
    }
    let Some(idx) = s.active()[slot].pokemon_index else { return };
    apply_boosts(state, side, idx, &StatBoosts { speed: -1, ..Default::default() }, true);
}

/// `comparePriority` order value for a handler with no `on...Order`.
const NO_ORDER: i64 = 4294967296;

/// One pokemon-showdown Residual-event handler and its sort keys
/// (`resolvePriority`: order, priority, speed, subOrder).
#[derive(Debug, Clone, Copy)]
struct ResidualHandler {
    order: i64,
    priority: i32,
    speed: i32,
    sub_order: i32,
    kind: ResidualKind,
}

#[derive(Debug, Clone, Copy)]
enum ResidualKind {
    /// A handler with no modeled effect (duration countdowns, ...): it only
    /// takes part in the sort, where ties still consume PRNG draws.
    Marker,
    /// Trick Room's `onFieldResidual` duration tick (order 27, subOrder 1).
    /// At 0 it ends and the speeds flip back before any later residual step
    /// or faint-replacement switch-in reads them.
    TrickRoom,
    /// Opportunist `onResidual` (order 29): flush the copied boosts.
    OpportunistFlush(Side, usize),
    /// A side condition with a duration (reflect, light screen, safeguard, mist,
    /// lucky chant, aurora veil, quick guard, wide guard). PS `fieldEvent`
    /// drops the duration when this handler is reached, so a battle that ends
    /// earlier in the residual leaves the later ones undropped. The payload is
    /// the `side_condition_duration_mut` index.
    SideDuration(Side, u8),
    /// Weather `onFieldResidual` (sandstorm damage via `eachEvent('Weather')`).
    Weather,
    /// brn / psn / tox `onResidual`.
    Status(Side, usize),
    /// `throatchop` duration countdown (order 22).
    ThroatChop(Side, usize),
    /// `tailwind` duration countdown (order 26, subOrder 5); the end recalculates speeds.
    Tailwind(Side),
    /// `taunt` duration countdown (order 15).
    Taunt(Side, usize),
    /// `magnetrise` duration countdown (order 18, PS data/moves.ts magnetrise).
    MagnetRise(Side, usize),
    /// Healer / Shed Skin / Hydration `onResidual` (order 5, subOrder 3).
    CureStatus(Side, usize),
    /// Moody `onResidual` (order 28, subOrder 2).
    Moody(Side, usize),
    /// Harvest `onResidual` (order 28, subOrder 2): the roll only.
    Harvest(Side, usize),
    /// Hunger Switch `onResidual` (order 29): Morpeko changes forme.
    HungerSwitch(Side, usize),
    /// Ghost `curse` damage (order 12).
    Curse(Side, usize),
    LeechSeed(Side, usize),
    Roost(Side, usize),
    Encore(Side, usize),
    Ingrain(Side, usize),
    SyrupBomb(Side, usize),
    /// `octolock` `onResidual` (order 14): def/spd -1 from the binder.
    Octolock(Side, usize),
    HealBlock(Side, usize),
    /// `perishsong` countdown (order 24).
    Perish(Side, usize),
    /// futuremove slot condition (order 3): the delayed hit.
    FutureMove(Side, usize),
    /// Wish slot condition (order 4): heals on the second residual.
    Wish(Side, usize),
    /// `aquaring` `onResidual` (order 6).
    AquaRing(Side, usize),
    /// Gravity duration countdown (order 27, subOrder 2).
    Gravity,
    /// Magic Room duration countdown (order 27, subOrder 6).
    MagicRoom,
    /// Fairy Lock duration countdown (no order, subOrder 5).
    FairyLock,
    /// Wonder Room duration countdown (order 27, subOrder 5).
    WonderRoom,
    /// Terrain duration countdown (order 27, subOrder 7), for the terrain
    /// that was up when the handlers were collected.
    Terrain(crate::engine::state::Terrain),
    /// `allyswitch` duration countdown (duration 2).
    AllySwitch(Side, usize),
    /// `lockon` duration countdown (duration 2, subOrder 2).
    LockOn(Side, usize),
    /// `uproar` duration countdown (order 28, subOrder 1).
    Uproar(Side, usize),
    /// `stall` duration countdown (duration 2).
    Stall(Side, usize),
    /// `lockedmove` `onResidual` (duration 2).
    Locked(Side, usize),
    /// `disable` duration countdown (order 17).
    Disable(Side, usize),
    /// `yawn` duration end (order 23): sleep on expiry.
    Yawn(Side, usize),
    /// `partiallytrapped` `onResidual` (duration 5, order 13).
    PartialTrap(Side, usize),
    /// Speed Boost `onResidual`.
    SpeedBoost(Side, usize),
    /// Salt Cure `onResidual` (champions mod: 1/16, 1/8 vs Water / Steel).
    SaltCure(Side, usize),
    /// Grassy Terrain `onFieldResidual`: heal grounded actives by 1/16 max HP.
    GrassyHeal(Side, usize),
}

/// Keep each slot's `vol_order` in PS `pokemon.volatiles` key order: drop
/// the volatiles that ended, append the ones added since the last call.
/// Called at the end of every action (PS `runAction`), so volatiles added
/// by different actions keep the order they were added in; ones added within
/// one action follow `pokemon_effects`' fixed order.
// TODO(PS parity): order of volatiles added within the same action.
fn sync_vol_order(state: &mut DoublesState) {
    for side in [Side::One, Side::Two] {
        for slot in 0..2 {
            // Unchanged volatiles since the last sync: nothing to do.
            let mask = state.side(side).cached_conds(slot, crate::doubles::events::vol_mask);
            if mask == state.side(side).active()[slot].vol_synced {
                continue;
            }
            let mut present: [&'static str; 64] = [""; 64];
            let mut n = 0;
            crate::doubles::events::visit_pokemon_effects(state, side, slot, |e| {
                if let crate::doubles::events::Effect::Id(id, crate::doubles::event_handlers::HandlerKind::Volatile) = e {
                    present[n] = id;
                    n += 1;
                }
            });
            let present = &present[..n];
            // `vol_order` / `vol_synced` are not read by `vol_mask`.
            let a = state.side_mut(side).slot_scratch_mut(slot);
            a.vol_synced = mask;
            let order = &mut a.vol_order;
            order.retain(|o| present.contains(o));
            for &id in present {
                if !order.contains(&id) {
                    order.push(id);
                }
            }
        }
    }
}

/// Build the Residual handler list in pokemon-showdown's `fieldEvent`
/// collection order: field (weather, terrain, pseudo-weather), then per side
/// its side conditions followed by each active's status, volatiles (in
/// insertion order) and ability. Durations themselves are ticked in
/// `end_of_turn`; here only the handlers' existence and keys matter.
// TODO(PS parity): `stall` (lingers the turn after Protect), items.
fn collect_residual_handlers(state: &DoublesState) -> Vec<ResidualHandler> {
    let mut hs = Vec::new();
    let mut push = |order: i64, speed: i32, sub_order: i32, kind: ResidualKind| {
        hs.push(ResidualHandler {
            order,
            priority: 0,
            speed,
            sub_order,
            kind,
        })
    };
    // Field: weather (subOrder 5), terrain / Trick Room (order 27).
    if state.weather.weather_type != Weather::NONE {
        push(1, 0, 5, ResidualKind::Weather);
    }
    if state.terrain.terrain_type != crate::engine::state::Terrain::NONE {
        push(27, 0, 7, ResidualKind::Terrain(state.terrain.terrain_type));
    }
    // PS data/moves.ts trickroom: onFieldResidualOrder 27, SubOrder 1. Its
    // duration drops (and it ends, `-fieldend`) in this residual handler.
    if state.trick_room.active {
        push(27, 0, 1, ResidualKind::TrickRoom);
    }
    if state.wonder_room.active {
        push(27, 0, 5, ResidualKind::WonderRoom);
    }
    if state.magic_room.active {
        push(27, 0, 6, ResidualKind::MagicRoom);
    }
    if state.gravity.active {
        push(27, 0, 2, ResidualKind::Gravity);
    }
    if state.fairy_lock.active {
        // No `onFieldResidualOrder`; a field condition's default subOrder is 5.
        push(NO_ORDER, 0, 5, ResidualKind::FairyLock);
    }
    for side in [Side::One, Side::Two] {
        // Side conditions with a duration (`onSideResidualOrder` 26).
        let sc = &state.side(side).side_conditions;
        for (active, sub, which) in [
            (sc.reflect > 0, 1, 0u8),
            (sc.light_screen > 0, 2, 1),
            (sc.safeguard > 0, 3, 2),
            (sc.mist > 0, 4, 3),
            (sc.lucky_chant > 0, 6, 4),
            (sc.aurora_veil > 0, 10, 5),
        ] {
            if active {
                push(26, 0, sub, ResidualKind::SideDuration(side, which));
            }
        }
        // Tailwind (sub 5): its countdown ends in the residual, so the speeds
        // recalculate before the rest of it (PS data/conditions.ts tailwind).
        if state.side(side).side_conditions.tailwind > 0 {
            push(26, 0, 5, ResidualKind::Tailwind(side));
        }
        // quickguard (duration 1, side condition: subOrder 4, no order).
        if state.side(side).side_conditions.quick_guard > 0 {
            push(NO_ORDER, 0, 4, ResidualKind::SideDuration(side, 6));
        }
        if state.side(side).side_conditions.wide_guard > 0 {
            push(NO_ORDER, 0, 4, ResidualKind::SideDuration(side, 7));
        }
        for slot in 0..2 {
            let s = state.side(side);
            let a = &s.active()[slot];
            let (idx, speed, alive) = match a.pokemon_index {
                Some(i) => (i, cached_speed(state, side, slot), true),
                None => match s.fainted_in_slot[slot] {
                    // PS keeps the fainted pokemon in the slot (and in the
                    // Residual sort) until its replacement switches in. Its
                    // volatiles were cleared on faint; only status / ability /
                    // slot-condition / field handlers remain, all of them
                    // skipped at run time.
                    Some((fi, _)) => (fi, s.speed_cache[fi as usize] as i32, false),
                    None => continue,
                },
            };
            let p = s.team_member(idx);
            // futuremove slot condition (`onResidualOrder` 3).
            if s.future_move[slot].is_some() {
                push(3, speed, 2, ResidualKind::FutureMove(side, slot));
            }
            // Wish slot condition (`onResidualOrder` 4).
            if s.wish[slot].is_some() {
                push(4, speed, 2, ResidualKind::Wish(side, slot));
            }
            // Status (effectType 'Status': subOrder 0).
            match p.status {
                PokemonStatus::BURN => push(10, speed, 0, ResidualKind::Status(side, slot)),
                PokemonStatus::POISON | PokemonStatus::TOXIC => {
                    push(9, speed, 0, ResidualKind::Status(side, slot))
                }
                _ => {}
            }
            if !alive {
                // Abilities (subOrder 7 unless set).
                if matches!(p.ability, Abilities::HEALER | Abilities::SHEDSKIN | Abilities::HYDRATION) {
                    push(5, speed, 3, ResidualKind::CureStatus(side, slot));
                }
                // Cud Chew / Pickup `onResidual` (order 28, subOrder 2): no
                // berries or items are modeled, so only the sort slot.
                if matches!(p.ability, Abilities::CUDCHEW | Abilities::PICKUP) {
                    push(28, speed, 2, ResidualKind::Marker);
                }
                // Opportunist `onResidual` (order 29): flushes the copied boosts.
                if p.ability == Abilities::OPPORTUNIST {
                    push(29, speed, 7, ResidualKind::OpportunistFlush(side, slot));
                }
                if p.ability == Abilities::MOODY {
                    push(28, speed, 2, ResidualKind::Moody(side, slot));
                }
                if p.ability == Abilities::HARVEST {
                    push(28, speed, 2, ResidualKind::Harvest(side, slot));
                }
                if p.ability == Abilities::HUNGERSWITCH {
                    push(29, speed, 7, ResidualKind::HungerSwitch(side, slot));
                }
                if p.ability == Abilities::SPEEDBOOST {
                    push(28, speed, 2, ResidualKind::SpeedBoost(side, slot));
                }
                // Grassy Terrain heal (`onResidualOrder` 5, subOrder 2).
                if state.terrain.terrain_type == crate::engine::state::Terrain::GRASSYTERRAIN {
                    push(5, speed, 2, ResidualKind::GrassyHeal(side, slot));
                }
                continue;
            }
            // Volatiles with a duration (Conditions: subOrder 2, no order),
            // collected in PS `pokemon.volatiles` key (insertion) order: the
            // order tied handlers enter `speedSort`. PS sim/battle.ts
            // findPokemonEventHandlers (`for (const id in pokemon.volatiles)`).
            let mut vols: Vec<(&'static str, i64, i32, ResidualKind)> = Vec::new();
            // An aborted charge lost `twoturnmove` (`onMoveAborted`), whose
            // `onEnd` also removed the move's own volatile (Dig, Fly, ...).
            if let Some((m, _)) = a.charging.filter(|_| !a.charge_aborted) {
                vols.push(("twoturnmove", NO_ORDER, 2, ResidualKind::Marker)); // twoturnmove
                if matches!(
                    m,
                    Choices::DIG
                        | Choices::DIVE
                        | Choices::FLY
                        | Choices::BOUNCE
                        | Choices::PHANTOMFORCE
                        | Choices::SHADOWFORCE
                        | Choices::SKYDROP
                ) {
                    // Gone with an aborted charge: `twoturnmove.onEnd` removes the
                    // move's volatile. PS data/conditions.ts twoturnmove.
                    vols.push((crate::doubles::events::semi_invulnerable_id(m).unwrap(), NO_ORDER, 2, ResidualKind::Marker)); // the move's own condition
                }
            }
            if a.twoturn_fired && a.charging.is_none() {
                vols.push(("twoturnmove", NO_ORDER, 2, ResidualKind::Marker)); // twoturnmove (duration 2, last turn)
            }
            if a.protected || a.endure {
                vols.push((if a.protected { crate::doubles::events::PROTECT_IDS[(a.protect_kind as usize).min(5)] } else { "endure" }, NO_ORDER, 2, ResidualKind::Marker)); // protect / endure
            }
            if a.stall.is_some() {
                vols.push(("stall", NO_ORDER, 2, ResidualKind::Stall(side, slot)));
            }
            if a.flinched {
                vols.push(("flinch", NO_ORDER, 2, ResidualKind::Marker));
            }
            if a.helping_hand_mult > 1.0 {
                vols.push(("helpinghand", NO_ORDER, 2, ResidualKind::Marker));
            }
            if a.focus_punch {
                vols.push(("focuspunch", NO_ORDER, 2, ResidualKind::Marker)); // focuspunch (duration 1)
            }
            if a.chilly_reception {
                vols.push(("chillyreception", NO_ORDER, 2, ResidualKind::Marker)); // chillyreception (duration 1)
            }
            // PS data/moves.ts electrify.condition / beakblast.condition: duration 1.
            if a.electrify {
                vols.push(("electrify", NO_ORDER, 2, ResidualKind::Marker));
            }
            if a.beak_blast {
                vols.push(("beakblast", NO_ORDER, 2, ResidualKind::Marker));
            }
            if a.recharging {
                vols.push(("mustrecharge", NO_ORDER, 2, ResidualKind::Marker)); // mustrecharge
            }
            if a.salt_cure {
                vols.push(("saltcure", 13, 2, ResidualKind::SaltCure(side, slot)));
            }
            if a.partial_trap.is_some() {
                vols.push(("partiallytrapped", 13, 2, ResidualKind::PartialTrap(side, slot)));
            }
            if a.yawn > 0 {
                vols.push(("yawn", 23, 2, ResidualKind::Yawn(side, slot)));
            }
            if a.throat_chop > 0 {
                vols.push(("throatchop", 22, 2, ResidualKind::ThroatChop(side, slot)));
            }
            if a.taunt > 0 {
                vols.push(("taunt", 15, 2, ResidualKind::Taunt(side, slot)));
            }
            if a.magnet_rise > 0 {
                vols.push(("magnetrise", 18, 2, ResidualKind::MagnetRise(side, slot)));
            }
            if a.cursed {
                vols.push(("curse", 12, 2, ResidualKind::Curse(side, slot)));
            }
            if a.encore.is_some() {
                vols.push(("encore", 16, 2, ResidualKind::Encore(side, slot)));
            }
            if a.ingrain {
                vols.push(("ingrain", 7, 2, ResidualKind::Ingrain(side, slot)));
            }
            if a.syrup_bomb.is_some() {
                vols.push(("syrupbomb", 14, 2, ResidualKind::SyrupBomb(side, slot)));
            }
            if a.octolock.is_some() {
                vols.push(("octolock", 14, 2, ResidualKind::Octolock(side, slot)));
            }
            if a.heal_block > 0 {
                vols.push(("healblock", 20, 2, ResidualKind::HealBlock(side, slot)));
            }
            if a.roost_orig.is_some() {
                vols.push(("roost", 25, 2, ResidualKind::Roost(side, slot)));
            }
            if a.leech_seed.is_some() {
                vols.push(("leechseed", 8, 2, ResidualKind::LeechSeed(side, slot)));
            }
            if a.perish > 0 {
                vols.push(("perishsong", 24, 2, ResidualKind::Perish(side, slot)));
            }
            if a.follow_me > 0 {
                vols.push(("followme", NO_ORDER, 2, ResidualKind::Marker)); // followme / ragepowder (duration 1)
            }
            if a.aqua_ring {
                vols.push(("aquaring", 6, 2, ResidualKind::AquaRing(side, slot)));
            }
            if a.counter_vol.is_some() {
                vols.push(("counter", NO_ORDER, 2, ResidualKind::Marker)); // counter (duration 1)
            }
            if a.mirror_coat_vol.is_some() {
                vols.push(("mirrorcoat", NO_ORDER, 2, ResidualKind::Marker)); // mirrorcoat (duration 1)
            }
            if a.ally_switch.is_some() {
                vols.push(("allyswitch", NO_ORDER, 2, ResidualKind::AllySwitch(side, slot)));
            }
            if a.lock_on.is_some() {
                vols.push(("lockon", NO_ORDER, 2, ResidualKind::LockOn(side, slot)));
            }
            if a.uproar > 0 {
                vols.push(("uproar", 28, 1, ResidualKind::Uproar(side, slot)));
            }
            if a.locked_move.is_some() {
                vols.push(("lockedmove", NO_ORDER, 2, ResidualKind::Locked(side, slot)));
            }
            if a.disabled.is_some() {
                vols.push(("disable", 17, 2, ResidualKind::Disable(side, slot)));
            }
            let rank = |id: &str| a.vol_order.iter().position(|&o| o == id).unwrap_or(usize::MAX);
            vols.sort_by_key(|v| rank(v.0));
            for (_, order, sub, kind) in vols {
                push(order, speed, sub, kind);
            }
            // Grassy Terrain heal (`onResidualOrder` 5, subOrder 2).
            if state.terrain.terrain_type == crate::engine::state::Terrain::GRASSYTERRAIN {
                push(5, speed, 2, ResidualKind::GrassyHeal(side, slot));
            }
            // Abilities (subOrder 7 unless set).
            if matches!(p.ability, Abilities::HEALER | Abilities::SHEDSKIN | Abilities::HYDRATION) {
                push(5, speed, 3, ResidualKind::CureStatus(side, slot));
            }
            if matches!(p.ability, Abilities::CUDCHEW | Abilities::PICKUP) {
                push(28, speed, 2, ResidualKind::Marker);
            }
            if p.ability == Abilities::OPPORTUNIST {
                push(29, speed, 7, ResidualKind::OpportunistFlush(side, slot));
            }
            if p.ability == Abilities::MOODY {
                push(28, speed, 2, ResidualKind::Moody(side, slot));
            }
            if p.ability == Abilities::HARVEST {
                push(28, speed, 2, ResidualKind::Harvest(side, slot));
            }
            if p.ability == Abilities::HUNGERSWITCH {
                push(29, speed, 7, ResidualKind::HungerSwitch(side, slot));
            }
            if p.ability == Abilities::SPEEDBOOST {
                push(28, speed, 2, ResidualKind::SpeedBoost(side, slot));
            }
        }
    }
    hs
}

/// pokemon-showdown `speedSort` over Residual handlers (`comparePriority`:
/// order asc, priority desc, speed desc, subOrder asc), shuffling ties.
fn sort_residual_handlers(
    mut hs: Vec<ResidualHandler>,
    rng: &mut Gen5Rng,
    label: &'static str,
) -> Vec<ResidualHandler> {
    sort_residual_slice(&mut hs, rng, label);
    hs
}

/// `sort_residual_handlers` in place.
fn sort_residual_slice(hs: &mut [ResidualHandler], rng: &mut Gen5Rng, label: &'static str) {
    let key = |h: &ResidualHandler| (h.order, -(h.priority as i64), -(h.speed as i64), h.sub_order as i64);
    let mut sorted = 0usize;
    let mut next_indexes: smallvec::SmallVec<[usize; 16]> = smallvec::SmallVec::new();
    while sorted + 1 < hs.len() {
        next_indexes.clear();
        next_indexes.push(sorted);
        for i in (sorted + 1)..hs.len() {
            match key(&hs[i]).cmp(&key(&hs[next_indexes[0]])) {
                std::cmp::Ordering::Less => {
                    next_indexes.clear();
                    next_indexes.push(i);
                }
                std::cmp::Ordering::Equal => next_indexes.push(i),
                std::cmp::Ordering::Greater => {}
            }
        }
        for (i, &index) in next_indexes.iter().enumerate() {
            if index != sorted + i {
                hs.swap(sorted + i, index);
            }
        }
        if next_indexes.len() > 1 {
            shuffle(hs, sorted, sorted + next_indexes.len(), rng, label);
        }
        sorted += next_indexes.len();
    }
}

/// PS team preview `commitChoices`: `queue.sort()` over the 8 `team` actions
/// (priority `-index`, speed = `getActionSpeed`). The A and B actions of the
/// same pick index tie when their speeds are equal: one `random(2)` shuffle.
/// PS: sim/side.ts chooseTeam, sim/battle.ts commitChoices, speedSort.
fn team_preview_draws(state: &DoublesState, rng: &mut Gen5Rng) {
    let speed_of = |side: &crate::doubles::state::DoublesSide, i: usize| -> i64 {
        let p = &side.team[i];
        let mut s = p.speed as i64;
        if p.item == crate::engine::items::Items::CHOICESCARF {
            s = (s * 6144 + 2047) / 4096;
        }
        s
    };
    for i in 0..4 {
        if speed_of(&state.side_one, i) == speed_of(&state.side_two, i) {
            let _ = rng.label("speed tie (move order)").random(2);
        }
    }
}

/// PS initial send-out: each lead's `switchIn` inserts a `runSwitch` action
/// into the queue (`insertChoice`), which consumes a draw when the lead's
/// speed ties with one or more already-inserted leads —
/// `random(firstIndex, lastIndex + 1)`, modulus = (ties + 1). Leads are
/// inserted in send-out order: A0, A1, B0, B1. These draws come before the
/// `runSwitch` speed sort.
fn queue_insert_draws(state: &DoublesState, rng: &mut Gen5Rng) {
    let mut inserted: Vec<i16> = Vec::new();
    for (side, slot) in [(Side::One, 0usize), (Side::One, 1), (Side::Two, 0), (Side::Two, 1)] {
        let Some(speed) = effective_speed(state, side, slot) else {
            continue;
        };
        let ties = inserted.iter().filter(|&&s| s == speed).count();
        if ties > 0 {
            let _ = rng.label("speed tie (queue insert)").random(ties as u32 + 1);
        }
        inserted.push(speed);
    }
}

/// PS `useMoveInner` `selfdestruct: 'always'`: the user faints (HP 0) and the
/// faint is resolved before the move's accuracy / hit steps and its closing
/// `eachEvent`.
fn selfdestruct_user_faint(state: &mut DoublesState, rng: &mut Gen5Rng, side: Side, slot: usize) {
    let hp = state.side(side).slot_pokemon(slot).map(|p| p.hp).unwrap_or(0);
    apply_damage(state, side, slot, hp);
    resolve_faints(state, rng, |fs, fsl| (fs, fsl) == (side, slot));
}

/// PS `eachEvent('Update')` (sim/battle.ts): speed-sorts all actives, which
/// draws on speed ties. Called after every action and every hit.
fn each_event_update(state: &mut DoublesState, rng: &mut Gen5Rng) {
    let order = sorted_actives(state, rng, "speed tie (eachEvent)");
    // Disguise `onUpdate` (after the sort): a pending absorbed hit busts the forme.
    let pending = std::mem::take(&mut state.pending_disguise);
    for (s, sl) in pending {
        let is_disguised = state.side(s).slot_pokemon(sl).is_some_and(|d| {
            d.id == crate::pokemon::PokemonName::MIMIKYU && d.ability == Abilities::DISGUISE
        });
        if is_disguised {
            disguise_bust(state, s, sl);
        }
    }
    for &(s, sl) in &order {
        // Attract `onUpdate` (PS data/moves.ts attract.condition): the volatile
        // is removed once its source is no longer active (switched / dragged
        // out or fainted: `isActive` false), even if it comes back later.
        if let Some((ss, si)) = state.side(s).active()[sl].attract {
            let src_active = (0..2).any(|k| {
                state.side(ss).active()[k].pokemon_index == Some(si) && state.side(ss).team_member(si).hp > 0
            });
            if !src_active {
                state.side_mut(s).active_slot_mut(sl).attract = None;
            }
        }
    }
    for (s, sl) in order {
        // Trace `onUpdate` (PS data/abilities.ts trace): every Update event
        // while the holder still has Trace and seeks draws a random valid foe.
        if state.side(s).active()[sl].trace_seek {
            let holder_ok = match state.side(s).slot_pokemon(sl) {
                Some(p) => p.ability == Abilities::TRACE && p.hp > 0,
                None => false,
            };
            if holder_ok {
                if trace_copy(state, s, sl, rng).is_some() {
                    // PS `setAbility` runs the new ability's `Start` event (its
                    // weather / terrain / Cloud Nine `eachEvent` is a sort).
                    if apply_on_switch_abilities(state, s, sl, rng) {
                        let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
                    }
                }
            }
        }
    }
}

/// Active foe team indices that are alive (PS `side.foes()` / `adjacentFoes`).
fn foe_active_indices(state: &DoublesState, foe: Side) -> Vec<PokemonIndex> {
    (0..2)
        .filter_map(|f| state.side(foe).active()[f].pokemon_index)
        .filter(|&fidx| state.side(foe).team_member(fidx).hp > 0)
        .collect()
}

/// Trace's `onUpdate` for the holder in `side`'s `active[slot]`: `sample` over
/// the alive adjacent foes that are not `notrace` and not No Ability (one draw
/// even for a single candidate), then copies that foe's ability. Returns the
/// copied ability, or None when there is no candidate (no draw).
fn trace_copy(state: &mut DoublesState, side: Side, slot: usize, rng: &mut Gen5Rng) -> Option<Abilities> {
    let foe = side.other();
    let targets: Vec<PokemonIndex> = foe_active_indices(state, foe)
        .into_iter()
        .filter(|&fidx| {
            let a = state.side(foe).team_member(fidx).ability;
            !has_notrace(a) && a != Abilities::NOABILITY
        })
        .collect();
    if targets.is_empty() {
        return None;
    }
    let pick = targets[rng.label("trace").random(targets.len() as u32) as usize];
    let copied = state.side(foe).team_member(pick).ability;
    if let Some(idx) = state.side(side).active()[slot].pokemon_index {
        state.side_mut(side).team_member_mut(idx).ability = copied;
        reset_protean_effect(state, side, idx);
    }
    Some(copied)
}

/// PS `eachEvent` / `runSwitch`: the actives speed-sorted (speed only), ties
/// shuffled.
/// `runEvent('ModifyMove')` in `useMoveInner`: Gravity and Wonder Room both
/// have an `onModifyMove` handler with identical sort keys, so the two tie and
/// shuffle (one draw) while both are active (PS sim/battle.ts `speedSort`).
/// Wonder Room's subOrder is 2 until its handler has run once (`state.target`
/// is set by `runEvent` just before the callback); the `ModifyMove` callback
/// runs for every move while Wonder Room is up, whether or not Gravity is. So
/// the first event with both active may already tie.
/// PS `runEvent('ModifyMove')` for the user: Heal Block and Throat Chop each
/// hold an `onModifyMove` handler (priority 0, same holder and speed), so when
/// both are up they tie and one shuffle draw happens for every move the user
/// uses (data/moves.ts healblock / throatchop, sim/battle.ts runEvent).
fn modify_move_user_tie(state: &DoublesState, rng: &mut Gen5Rng, side: Side, slot: usize) {
    if state.side(side).active()[slot].heal_block > 0 && state.side(side).active()[slot].throat_chop > 0 {
        let _ = rng.label("speed tie (move order)").random(2);
    }
}

fn modify_move_field_tie(state: &mut DoublesState, rng: &mut Gen5Rng) {
    if state.wonder_room.active {
        if state.gravity.active && state.wonder_room_seen {
            let _ = rng.label("speed tie (move order)").random(2);
        }
        state.wonder_room_seen = true;
    }
}

/// Key of a pokemon in another pokemon's `attacked_by` list: side and team
/// index (PS `attackedBy` holds the Pokemon itself, so both must match).
fn attacker_key(side: Side, idx: u8) -> u8 {
    ((side as u8) << 4) | idx
}

fn sorted_actives(
    state: &DoublesState,
    rng: &mut Gen5Rng,
    label: &'static str,
) -> arrayvec::ArrayVec<(Side, usize), 4> {
    // Actives in slot order; a pokemon that fainted in the current hit loop
    // still counts until faints are resolved (`unresolved_faints`).
    let mut hs: arrayvec::ArrayVec<(Side, usize, i32), 4> = arrayvec::ArrayVec::new();
    for (s, sl) in [(Side::One, 0usize), (Side::One, 1), (Side::Two, 0), (Side::Two, 1)] {
        let speed = if state.side(s).slot_pokemon(sl).map(|p| p.hp > 0).unwrap_or(false) {
            cached_speed(state, s, sl)
        } else if let Some(&(_, _, idx)) =
            state.unresolved_faints.iter().find(|(fs, fsl, _)| *fs == s && *fsl == sl)
        {
            // A pokemon that fainted in the current hit loop still counts
            // until faints are resolved (`unresolved_faints`).
            state.side(s).speed_cache[idx as usize] as i32
        } else if label == "speed tie (switch-in)" && state.side(s).fainted_in_slot[sl].is_some() {
            // `runSwitch` sorts `getAllActive(true)`: a fainted pokemon still
            // sitting in its slot counts, with its stale cached speed.
            // Its `ModifySpe` handlers (status, boosts) no longer run, so the
            // stored speed is the raw stat.
            let (idx, _) = state.side(s).fainted_in_slot[sl].unwrap();
            state.side(s).team_member(idx).speed as i32
        } else {
            continue;
        };
        hs.push((s, sl, speed));
    }
    if crate::doubles::debug::flags().actives {
        let shown: Vec<_> = hs.iter().map(|&(s, sl, sp)| (ResidualKind::Status(s, sl), sp)).collect();
        eprintln!("ACTIVES {} {:?}", label, shown);
    }
    // No two equal speeds (the usual case): a plain sort, no draws.
    let n = hs.len();
    let tie = (0..n).any(|i| ((i + 1)..n).any(|j| hs[i].2 == hs[j].2));
    if !tie {
        for i in 1..n {
            let mut j = i;
            while j > 0 && hs[j - 1].2 < hs[j].2 {
                hs.swap(j - 1, j);
                j -= 1;
            }
        }
        return hs.into_iter().map(|(s, sl, _)| (s, sl)).collect();
    }
    // `sort_residual_slice` with every key but speed equal: fastest first,
    // ties shuffled.
    let mut sorted = 0usize;
    while sorted + 1 < hs.len() {
        let mut best = hs[sorted].2;
        let mut ties: arrayvec::ArrayVec<usize, 4> = arrayvec::ArrayVec::new();
        ties.push(sorted);
        for i in (sorted + 1)..hs.len() {
            if hs[i].2 > best {
                best = hs[i].2;
                ties.clear();
                ties.push(i);
            } else if hs[i].2 == best {
                ties.push(i);
            }
        }
        for (i, &index) in ties.iter().enumerate() {
            if index != sorted + i {
                hs.swap(sorted + i, index);
            }
        }
        if ties.len() > 1 {
            shuffle(&mut hs, sorted, sorted + ties.len(), rng, label);
        }
        sorted += ties.len();
    }
    hs.into_iter().map(|(s, sl, _)| (s, sl)).collect()
}

/// End-of-turn `DisableMove` and `TrapPokemon` (and `MaybeTrapPokemon` for a
/// pokemon not immune to `trapped`) events, per active pokemon in PS `endTurn`
/// order (sides, then slots). Only their handler-sort draws.
/// PS sim/battle.ts endTurn.
/// TODO(PS parity): `knownType` (Illusion) for MaybeTrapPokemon.
fn end_turn_disable_moves(state: &DoublesState, rng: &mut Gen5Rng) {
    use crate::doubles::events::run_event_ties;
    for (s, sl) in [(Side::One, 0usize), (Side::One, 1), (Side::Two, 0), (Side::Two, 1)] {
        let Some(idx) = state.side(s).active()[sl].pokemon_index else { continue };
        run_event_ties(state, rng, crate::doubles::events::Event::DisableMove, (s, sl), None);
        run_event_ties(state, rng, crate::doubles::events::Event::TrapPokemon, (s, sl), None);
        // Every TrapPokemon handler that calls `pokemon.tryTrap()` runs
        // `runStatusImmunity('trapped')` -> `runEvent('Immunity')` (a sort; Dig /
        // Dive + Magnet Rise tie). Ghost types stop at `getImmunity`, no event.
        // PS sim/pokemon.ts tryTrap / runStatusImmunity.
        if !state.side(s).team_member(idx).has_type(&PokemonType::GHOST) {
            for _ in 0..try_trap_calls(state, s, sl) {
                run_event_ties(state, rng, crate::doubles::events::Event::Immunity, (s, sl), None);
            }
        }
        // `getImmunity('trapped')`: Ghost types are immune.
        if state.side(s).team_member(idx).has_type(&PokemonType::GHOST) {
            run_event_ties(state, rng, crate::doubles::events::Event::MaybeTrapPokemon, (s, sl), None);
        }
    }
}

/// How many of a pokemon's end-of-turn TrapPokemon handlers call `tryTrap`:
/// the `trapped` volatile, `partiallytrapped` / Octolock while the binder is
/// active, Ingrain, Fairy Lock, and the foes' Arena Trap (grounded target),
/// Magnet Pull (Steel target) and Shadow Tag (target without it).
/// PS data/conditions.ts trapped / partiallytrapped, data/moves.ts ingrain /
/// fairylock / octolock, data/abilities.ts arenatrap / magnetpull / shadowtag.
fn try_trap_calls(state: &DoublesState, s: Side, sl: usize) -> usize {
    let a = &state.side(s).active()[sl];
    let Some(idx) = a.pokemon_index else { return 0 };
    let p = state.side(s).team_member(idx);
    let is_active = |bs: Side, bi: PokemonIndex| (0..2).any(|k| state.side(bs).active()[k].pokemon_index == Some(bi));
    let mut n = 0;
    n += a.trapped as usize;
    n += a.partial_trap.as_ref().is_some_and(|t| is_active(t.source_side, t.source_idx)) as usize;
    n += a.octolock.is_some_and(|(bs, bi)| is_active(bs, bi)) as usize;
    n += a.ingrain as usize;
    n += state.fairy_lock.active as usize;
    for k in 0..2 {
        let Some(f) = state.side(s.other()).slot_pokemon(k).filter(|f| f.hp > 0) else { continue };
        n += match f.ability {
            Abilities::ARENATRAP => slot_grounded(p, a, state.gravity.active) as usize,
            Abilities::MAGNETPULL => p.has_type(&PokemonType::STEEL) as usize,
            Abilities::SHADOWTAG => (p.ability != Abilities::SHADOWTAG) as usize,
            _ => 0,
        };
    }
    n
}

/// PS `pokemon.getMoves()` is empty (every known move has no PP, is disabled,
/// sealed by a foe's Imprison, or a Status move under Taunt), so `chooseMove`
/// makes the pokemon Struggle (sim/side.ts `!moves.length`). A locked, charging,
/// recharging or Encore'd pokemon keeps its locked flow instead.
fn no_usable_moves(state: &DoublesState, side: Side, slot: usize, idx: PokemonIndex) -> bool {
    let a = &state.side(side).active()[slot];
    if a.encore.is_some() || a.locked_move.is_some() || a.charging.is_some() || a.uproar > 0 || a.recharging {
        return false;
    }
    let p = state.side(side).team_member(idx);
    [&p.moves.m0, &p.moves.m1, &p.moves.m2, &p.moves.m3].iter().all(|m| {
        m.pp <= 0
            || m.disabled
            || a.disabled.is_some_and(|(d, _)| d == m.id)
            || imprison_blocks(state, side, m.id)
            || (a.taunt > 0 && m.choice.category == MoveCategory::Status)
            // PS data/mods/champions/moves.ts fakeout / firstimpression
            // `onDisableMove`: disabled once `activeMoveActions` is non-zero.
            || (matches!(m.id, Choices::FAKEOUT | Choices::FIRSTIMPRESSION) && a.active_move_actions > 0)
            // PS data/moves.ts gravity / healblock / throatchop / torment
            // `condition.onDisableMove`.
            || (state.gravity.active && has_gravity_flag(m.id))
            || (a.heal_block > 0 && m.choice.flags.heal)
            || (a.throat_chop > 0 && m.choice.flags.sound)
            || (a.torment && a.last_move.is_some_and(|l| l == m.id && l != Choices::STRUGGLE))
    })
}

/// PS Imprison `onFoeBeforeMove` (data/moves.ts imprison): true when an active
/// foe holds Imprison and its pokemon knows `move_id`.
fn imprison_blocks(state: &DoublesState, side: Side, move_id: Choices) -> bool {
    let other = side.other();
    (0..2).any(|sl| {
        let a = &state.side(other).active()[sl];
        a.imprison
            && a.pokemon_index.is_some_and(|i| {
                let p = state.side(other).team_member(i);
                p.hp > 0
                    && [&p.moves.m0, &p.moves.m1, &p.moves.m2, &p.moves.m3].iter().any(|m| m.id == move_id)
            })
    })
}

/// PS move flag `gravity` (champions data): Gravity blocks these moves.
/// PS data/moves.ts: splash, bounce, floatyfall, fly, flyingpress, highjumpkick,
/// jumpkick, magnetrise, skydrop, telekinesis.
fn has_gravity_flag(move_id: Choices) -> bool {
    matches!(
        move_id,
        Choices::SPLASH
            | Choices::BOUNCE
            | Choices::FLOATYFALL
            | Choices::FLY
            | Choices::FLYINGPRESS
            | Choices::HIGHJUMPKICK
            | Choices::JUMPKICK
            | Choices::MAGNETRISE
            | Choices::SKYDROP
            | Choices::TELEKINESIS
    )
}

/// PS `speedSort` over a handler list keyed by (order, speed, subOrder):
/// shuffle every exact tie group, one PRNG draw per swap.
/// PS Damp `onAnyTryMove` / `onAnyDamage` (data/abilities.ts damp): each
/// active Damp holder is one handler with the same sort keys, so holders with
/// equal speed tie and shuffle (PS `findEventHandlers` -> `speedSort`). Called
/// at `TryMove` and at each `Damage` event. Mold Breaker only suppresses the
/// handler later, so it does not change the draws.
/// PS data/moves.ts lockon `onSourceAccuracy`: the user's Lock-On on this foe
/// makes every move of the user against it hit (no accuracy roll).
fn lock_on_hits(state: &DoublesState, side: Side, slot: usize, ts: Side, tsl: usize) -> bool {
    match state.side(side).active()[slot].lock_on {
        Some((ls, lidx, _)) => ls == ts && state.side(ts).active()[tsl].pokemon_index == Some(lidx),
        None => false,
    }
}

/// Accuracy of the accuracy roll. Explosion, Mind Blown, Misty Explosion and
/// Self-Destruct are accuracy 100 in PS (data/moves.ts) but the shared move
/// table leaves it unset; the doubles roll uses PS's value.
fn roll_accuracy_of(choice: &Choice) -> f32 {
    match choice.move_id {
        Choices::EXPLOSION | Choices::MINDBLOWN | Choices::MISTYEXPLOSION | Choices::SELFDESTRUCT => 100.0,
        _ => choice.accuracy,
    }
}

/// Damp `onAnyTryMove`: any active Damp holder (not ignored by the user's
/// breakable-ability check) blocks Explosion, Mind Blown, Misty Explosion and
/// Self-Destruct before accuracy. PS data/abilities.ts damp.
fn damp_blocks_explosion(state: &DoublesState, side: Side, idx: PokemonIndex, choice: &Choice) -> bool {
    if !matches!(
        choice.move_id,
        Choices::EXPLOSION | Choices::MINDBLOWN | Choices::MISTYEXPLOSION | Choices::SELFDESTRUCT
    ) {
        return false;
    }
    let attacker = state.side(side).team_member(idx);
    [Side::One, Side::Two].iter().any(|&sd| {
        (0..2).any(|sl| {
            let Some(hidx) = state.side(sd).active()[sl].pokemon_index else { return false };
            state.side(sd).slot_pokemon(sl).is_some_and(|p| p.hp > 0)
                && real_ability(state, sd, hidx) == Abilities::DAMP
                && !crate::engine::abilities::ignores_target_ability(attacker, choice, &Abilities::DAMP)
        })
    })
}


/// PS `afterMoveSecondaryEvent` -> `runEvent('AfterMoveSecondary', targets)`:
/// `findEventHandlers` collects each hit target's `onAfterMoveSecondary`
/// handlers (its frozen status `frz`, and the Pickpocket / Berserk / Color
/// Change / Anger Shell abilities), each keyed by its holder's speed. Equal
/// keys tie and draw (`speedSort`). The handlers' effects are modelled
/// elsewhere; only the sort's draws are made here.
/// TODO(PS parity): held items (Eject Button, Kee Berry, Maranga Berry, Red Card).
fn after_move_secondary_ties(state: &mut DoublesState, rng: &mut Gen5Rng, targets: &[(Side, usize)]) {
    let mut hs: Vec<(i64, i32, i64)> = Vec::new();
    let mut seen: smallvec::SmallVec<[(Side, usize); 4]> = smallvec::SmallVec::new();
    for &(ts, tsl) in targets {
        if seen.contains(&(ts, tsl)) {
            continue;
        }
        seen.push((ts, tsl));
        let Some(idx) = slot_index_or_unresolved(state, ts, tsl) else { continue };
        // A KO'd target still counts with its stored speed until its faint resolves.
        let speed = if state.side(ts).slot_pokemon(tsl).is_some() {
            cached_speed(state, ts, tsl)
        } else {
            state.side(ts).speed_cache[idx as usize] as i32
        };
        // PS `onAfterMoveSecondary` order: NO_ORDER; subOrder Status 0, Ability 7.
        if state.side(ts).team_member(idx).status == PokemonStatus::FREEZE {
            hs.push((4294967296, speed, 0));
        }
        if matches!(
            real_ability(state, ts, idx),
            Abilities::PICKPOCKET | Abilities::BERSERK | Abilities::COLORCHANGE | Abilities::ANGERSHELL
        ) {
            hs.push((4294967296, speed, 7));
        }
    }
    if hs.len() >= 2 {
        sort_handler_ties(&mut hs, rng, "speed tie (move order)");
    }
}

/// PS `runEvent('Accuracy', target, source)` in `hitStepAccuracy`: its handlers
/// (No Guard `onAnyAccuracy`, Glaive Rush / Minimize `onAccuracy`, Lock-On
/// `onSourceAccuracy`, ...) are speed-sorted; ties draw.
/// PS sim/battle-actions.ts hitStepAccuracy.
/// TODO(PS parity): `accuracy: true` moves also run this event.
fn accuracy_event_ties(state: &DoublesState, rng: &mut Gen5Rng, side: Side, slot: usize, ts: Side, tsl: usize) {
    crate::doubles::events::run_event_ties(state, rng, crate::doubles::events::Event::Accuracy, (ts, tsl), Some((side, slot)));
}

fn sort_handler_ties(hs: &mut [(i64, i32, i64)], rng: &mut Gen5Rng, label: &'static str) {
    let key = |e: &(i64, i32, i64)| (e.0, -(e.1 as i64), e.2);
    let mut sorted = 0usize;
    while sorted + 1 < hs.len() {
        let mut next_indexes = vec![sorted];
        for i in (sorted + 1)..hs.len() {
            match key(&hs[i]).cmp(&key(&hs[next_indexes[0]])) {
                std::cmp::Ordering::Less => next_indexes = vec![i],
                std::cmp::Ordering::Equal => next_indexes.push(i),
                std::cmp::Ordering::Greater => {}
            }
        }
        for (i, &index) in next_indexes.iter().enumerate() {
            if index != sorted + i {
                hs.swap(sorted + i, index);
            }
        }
        if next_indexes.len() > 1 {
            shuffle(hs, sorted, sorted + next_indexes.len(), rng, label);
        }
        sorted += next_indexes.len();
    }
}

/// End-of-turn processing, mirroring pokemon-showdown's `update()`:
/// resolve faints (already done on damage), apply residual status/weather
/// damage to the actives in speed order, switch in replacements (with
/// hazard damage), then tick down weather / terrain / side-condition durations.
fn end_of_turn(
    state: &mut DoublesState,
    rng: &mut Gen5Rng,
    pending_in: &[(Side, usize, usize)],
    ee_switches: &[(Side, usize, usize)],
) {
    // PS `runAction`: HP of every active before the action, for `EmergencyExit`.
    let hp_before: smallvec::SmallVec<[(Side, usize, i16); 4]> = [(Side::One, 0usize), (Side::One, 1), (Side::Two, 0), (Side::Two, 1)]
        .iter()
        .filter_map(|&(sd, sl)| state.side(sd).slot_pokemon(sl).map(|p| (sd, sl, p.hp)))
        .collect();
    // Residual phase: PS `runAction('residual')` -> `fieldEvent('Residual')`
    // collects every Residual handler (field, side, per-active status /
    // volatiles / ability) and speed-sorts them with `comparePriority`,
    // shuffling exact ties (consuming PRNG draws), then runs them in order.
    // `runAction('residual')` -> `updateSpeed()` first.
    update_speed(state);
    let handlers = collect_residual_handlers(state);
    if crate::doubles::debug::flags().res { eprintln!("=== residual ({} handlers) ===", handlers.len()); for h in &handlers { let nm = match &h.kind { ResidualKind::GrassyHeal(sd, sl) | ResidualKind::ThroatChop(sd, sl) | ResidualKind::Stall(sd, sl) => state.side(*sd).active()[*sl].pokemon_index.map(|i| state.side(*sd).team_member(i).id.to_string()).unwrap_or_default(), _ => String::new() }; eprintln!("RES order={} sub={} speed={} kind={:?} [{}]", h.order, h.sub_order, h.speed, h.kind, nm); } }
    let handlers = sort_residual_handlers(handlers, rng, "speed tie (residual)");
    // A wipe by an immediate faint ends the residual at once (PS `fieldEvent`
    // stops when the battle ends). A wipe by Perish song does not: PS defers
    // that faint until after the residual action, so the rest of it still runs.
    // The next handler that is not an `end` callback runs, then its
    // `resolveFaints` ends the battle (PS sim/battle.ts fieldEvent).
    let mut perish_wipe = false;
    let mut prev_end = false;
    for h in handlers {
        if state.battle_is_over() != 0.0 && (!perish_wipe || !prev_end) {
            return;
        }
        prev_end = false;
        match h.kind {
            ResidualKind::Marker => {}
            ResidualKind::TrickRoom => {
                if state.trick_room.active && state.trick_room.turns_remaining > 0 {
                    state.trick_room.turns_remaining -= 1;
                    if state.trick_room.turns_remaining == 0 {
                        state.trick_room.active = false;
                        update_speed(state);
                        prev_end = true;
                    }
                }
            }
            ResidualKind::OpportunistFlush(side, slot) => flush_opportunist(state, side, slot),
            ResidualKind::SideDuration(side, which) => {
                let d = side_condition_duration_mut(state, side, which);
                if *d > 0 {
                    *d -= 1;
                    // The duration ran out: PS calls `end` and skips `resolveFaints`.
                    prev_end = *d == 0;
                }
            }
            ResidualKind::Weather => {
                // PS `fieldEvent`: the duration drops before the callback; at 0
                // the weather just ends (no `onFieldResidual`, no damage).
                if state.weather.turns_remaining > 0 {
                    state.weather.turns_remaining -= 1;
                    if state.weather.turns_remaining == 0 {
                        state.weather.weather_type = Weather::NONE;
                        // `clearWeather` -> `eachEvent('WeatherChange')`: Forecast
                        // changes forme now (PS data/abilities.ts forecast), so its
                        // raw cached speed is what the residual's Update sort sees.
                        let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
                        update_forecast(state);
                        prev_end = true;
                        continue;
                    }
                }
                // `onFieldResidual` -> `eachEvent('Weather')` then
                // `eachEvent('Update')`: two speed sorts of the actives.
                // PS `isWeather` is false while Cloud Nine / Air Lock is active
                // (`field.suppressingWeather()`). Sandstorm, Hail and Snowscape
                // `onFieldResidual` only call `eachEvent('Weather')` (and so its
                // `Update`) when `this.field.isWeather(...)`: no sort at all.
                // Rain, Sun, Primordial Sea, Desolate Land and Delta Stream call
                // it unconditionally, so both sorts still draw with every handler
                // suppressed (PS data/conditions.ts `onFieldResidual`).
                if weather_suppressed(state) {
                    if !matches!(state.weather.weather_type, Weather::SAND | Weather::HAIL | Weather::SNOW) {
                        let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
                        let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
                    }
                    continue;
                }
                let order = sorted_actives(state, rng, "speed tie (eachEvent)");
                for (side, slot) in order {
                    // Ability `onWeather` handlers (PS data/abilities.ts icebody /
                    // raindish / dryskin / solarpower).
                    // TODO(PS parity): Cloud Nine / Air Lock (`effectiveWeather`).
                    if let Some(p) = state.side(side).slot_pokemon(slot) {
                        let w = state.weather.weather_type;
                        let rain = matches!(w, Weather::RAIN | Weather::HEAVYRAIN);
                        let sun = matches!(w, Weather::SUN | Weather::HARSHSUN);
                        let snow = matches!(w, Weather::HAIL | Weather::SNOW);
                        let (hp, maxhp) = (p.hp, p.maxhp);
                        let ab = p.ability;
                        let heal = match ab {
                            Abilities::ICEBODY if snow => maxhp / 16,
                            Abilities::RAINDISH if rain => maxhp / 16,
                            Abilities::DRYSKIN if rain => maxhp / 8,
                            _ => 0,
                        };
                        if heal > 0 && hp > 0 && state.side(side).active()[slot].heal_block == 0 {
                            if let Some(idx) = state.side(side).active()[slot].pokemon_index {
                                let m = state.side_mut(side).team_member_mut(idx);
                                m.hp = (m.hp + heal.max(1)).min(m.maxhp);
                            }
                        } else if sun && hp > 0 && matches!(ab, Abilities::DRYSKIN | Abilities::SOLARPOWER) {
                            apply_indirect_damage(state, side, slot, (maxhp / 8).max(1));
                        }
                    }
                    if state.weather.weather_type != Weather::SAND {
                        continue;
                    }
                    let Some(idx) = state.side(side).active()[slot].pokemon_index else { continue };
                    // PS `spreadDamage` (Weather effect) -> `runStatusImmunity('sandstorm')`:
                    // the type immunity first (no event), then `runEvent('Immunity')`,
                    // whose handlers (Dig / Dive charge, Magnet Rise, Overcoat, Sand
                    // Veil / Force / Rush, ...) are speed-sorted before they refuse.
                    let types = state.side(side).team_member(idx).types;
                    if state.side(side).team_member(idx).hp <= 0
                        || [PokemonType::ROCK, PokemonType::GROUND, PokemonType::STEEL]
                            .iter()
                            .any(|t| types.0 == *t || types.1 == *t)
                    {
                        continue;
                    }
                    crate::doubles::events::run_event_ties(state, rng, crate::doubles::events::Event::Immunity, (side, slot), None);
                    // Dig / Dive charge turn: immune to sandstorm (PS
                    // data/moves.ts dig / dive `condition.onImmunity`). Fly,
                    // Bounce and Sky Drop have no such handler, so they take it.
                    // An aborted charge (BeforeMove failed, e.g. asleep) already
                    // removed `twoturnmove`, and with it `dig`'s condition (PS
                    // `twoturnmove.onEnd`): no immunity.
                    if let Some((mv, _)) = &state.side(side).active()[slot].charging {
                        if !state.side(side).active()[slot].charge_aborted
                            && matches!(mv, crate::choices::Choices::DIG | crate::choices::Choices::DIVE)
                        {
                            continue;
                        }
                    }
                    let p2 = state.side(side).team_member(idx);
                    let types = p2.types;
                    // `runStatusImmunity('sandstorm')`: types, and the abilities with
                    // `onImmunity` (Overcoat, Sand Veil / Force / Rush).
                    let immune = [PokemonType::ROCK, PokemonType::GROUND, PokemonType::STEEL]
                        .iter()
                        .any(|t| types.0 == *t || types.1 == *t)
                        || matches!(
                            p2.ability,
                            Abilities::OVERCOAT | Abilities::SANDVEIL | Abilities::SANDFORCE | Abilities::SANDRUSH
                        );
                    if !immune {
                        apply_indirect_damage(state, side, slot, (p2.maxhp / 16) as i16);
                    }
                }
                each_event_update(state, rng);
            }
            ResidualKind::Status(side, slot) => {
                let idx = match state.side(side).active()[slot].pokemon_index {
                    Some(i) => i,
                    None => continue,
                };
    let p = state.side(side).team_member(idx);
            let dmg: i16 = match p.status {
                // PS data/conditions.ts brn.onResidual: `baseMaxhp / 16`, then
                // data/abilities.ts heatproof.onDamage (`effect.id === 'brn'`):
                // half of it, before the int clamp.
                PokemonStatus::BURN if p.ability == Abilities::HEATPROOF => (p.maxhp / 32) as i16,
                PokemonStatus::BURN => (p.maxhp / 16) as i16,
                PokemonStatus::POISON => (p.maxhp / 8) as i16,
                PokemonStatus::TOXIC => {
                    let stage = state.side(side).toxic_stage[idx as usize].max(1);
                    ((p.maxhp / 16).max(1) * stage as i16) as i16
                }
                _ => 0,
            };
            // Advance the toxic stage for next turn.
            if p.status == PokemonStatus::TOXIC {
                let s = state.side_mut(side);
                let st = &mut s.toxic_stage[idx as usize];
                if *st < 15 {
                    *st += 1;
                }
            }
            // Poison Heal `onDamage` (priority 1, before Magic Guard): psn / tox
            // damage becomes a 1/8 heal (`heal` fails under Heal Block).
            if dmg > 0
                && matches!(state.side(side).team_member(idx).status, PokemonStatus::POISON | PokemonStatus::TOXIC)
                && state.side(side).team_member(idx).ability == Abilities::POISONHEAL
            {
                if state.side(side).active()[slot].heal_block == 0 {
                    let pm = state.side_mut(side).team_member_mut(idx);
                    pm.hp = (pm.hp + (pm.maxhp / 8).max(1)).min(pm.maxhp);
                }
            } else if dmg > 0 {
                apply_indirect_damage(state, side, slot, dmg);
            }
            }
            ResidualKind::ThroatChop(side, slot) => {
                let a = state.side_mut(side).active_slot_mut(slot);
                a.throat_chop = a.throat_chop.saturating_sub(1);
            }
            ResidualKind::Tailwind(sd) => {
                let sc = &mut state.side_mut(sd).side_conditions;
                if sc.tailwind > 0 {
                    sc.tailwind -= 1;
                    // PS tailwind `onSideEnd` only logs `-sideend`: the cached
                    // speeds (`pokemon.speed`) stay doubled until the next
                    // `updateSpeed` (`runAction('residual')` or `commitChoices`),
                    // so the residual's `eachEvent` sorts still use them.
                }
            }
            ResidualKind::Taunt(side, slot) => {
                let a = state.side_mut(side).active_slot_mut(slot);
                a.taunt = a.taunt.saturating_sub(1);
            }
            ResidualKind::MagnetRise(side, slot) => {
                let a = state.side_mut(side).active_slot_mut(slot);
                a.magnet_rise = a.magnet_rise.saturating_sub(1);
            }
            ResidualKind::CureStatus(side, slot) => {
                let Some(idx) = state.side(side).active()[slot].pokemon_index else { continue };
                let (ab, hp) = {
                    let p = state.side(side).team_member(idx);
                    (p.ability, p.hp)
                };
                if hp <= 0 {
                    continue;
                }
                if ab == Abilities::HYDRATION {
                    // PS hydration.onResidual: cures its status in rain
                    // (`effectiveWeather`, so not under Cloud Nine / Air Lock).
                    if state.side(side).team_member(idx).status != PokemonStatus::NONE
                        && matches!(effective_weather(state), Weather::RAIN | Weather::HEAVYRAIN)
                    {
                        state.side_mut(side).team_member_mut(idx).status = PokemonStatus::NONE;
                    }
                } else if ab == Abilities::SHEDSKIN {
                    // 33/100 to cure its own status.
                    if state.side(side).team_member(idx).status != PokemonStatus::NONE
                        && rng.label("residual roll").random_chance(33, 100)
                    {
                        state.side_mut(side).team_member_mut(idx).status = PokemonStatus::NONE;
                    }
                } else {
                    // Healer: 1/2 per statused adjacent ally (champions mod
                    // `abilities.ts` healer `onResidual`; generic PS is 3/10).
                    let al = 1 - slot;
                    if let Some(aidx) = state.side(side).active()[al].pokemon_index {
                        let a = state.side(side).team_member(aidx);
                        if a.hp > 0 && a.status != PokemonStatus::NONE && rng.label("residual roll").random_chance(1, 2) {
                            state.side_mut(side).team_member_mut(aidx).status = PokemonStatus::NONE;
                        }
                    }
                }
            }
            ResidualKind::HungerSwitch(side, slot) => {
                // PS hungerswitch.onResidual: `formeChange` Morpeko <-> Morpeko-Hangry.
                // `setSpecies` resets the cached speed to the raw stat until the
                // next `updateSpeed`.
                let Some(idx) = state.side(side).active()[slot].pokemon_index else { continue };
                let p = state.side(side).team_member(idx);
                if p.hp <= 0 {
                    continue;
                }
                let new_id = match p.id {
                    crate::pokemon::PokemonName::MORPEKO => crate::pokemon::PokemonName::MORPEKOHANGRY,
                    crate::pokemon::PokemonName::MORPEKOHANGRY => crate::pokemon::PokemonName::MORPEKO,
                    _ => continue,
                };
                let s = state.side_mut(side);
                s.team_member_mut(idx).id = new_id;
                // PS `formeChange` -> `setSpecies` -> `setType(species.types)`: a type
                // from Magic Powder / Soak is replaced by the species type (both
                // Morpeko formes are Electric / Dark, species_types.rs).
                s.team_member_mut(idx).types = (PokemonType::ELECTRIC, PokemonType::DARK);
                // PS `formeChange` -> `setSpecies`: `storedStats` are rebuilt from the
                // base stats, so an averaged Atk / SpA (Power Split) or Def / SpD
                // (Guard Split) is undone, as on switch-in.
                // PS sim/pokemon.ts setSpecies.
                restore_stored_stats(s, idx);
                let raw = s.team_member(idx).speed;
                s.speed_cache[idx as usize] = raw as _;
            }
            ResidualKind::Harvest(side, slot) => {
                // PS harvest.onResidual: `isWeather(sun) || randomChance(1, 2)`.
                // The restored berry is not modeled (items are unused).
                // TODO(PS parity): Harvest berry restore (lastItem).
                if state.side(side).slot_pokemon(slot).map(|p| p.hp <= 0).unwrap_or(true) {
                    continue;
                }
                if !matches!(effective_weather(state), Weather::SUN | Weather::HARSHSUN) {
                    let _ = rng.label("residual roll").random_chance(1, 2);
                }
            }
            ResidualKind::Moody(side, slot) => {
                // PS moody.onResidual: +2 to a random stat below +6, -1 to a
                // different random stat above -6 (`sample` each; no accuracy /
                // evasion).
                // A holder that Perish Song fainted earlier in this residual is
                // not `fainted` yet (an `end` callback skips `resolveFaints`), so
                // Moody still runs and samples; `boost` then does nothing on 0 HP.
                // PS sim/battle.ts fieldEvent / boost, data/abilities.ts moody.
                let pending = state
                    .unresolved_faints
                    .iter()
                    .find(|&&(s_, sl_, _)| s_ == side && sl_ == slot)
                    .map(|&(_, _, i)| i);
                let Some(idx) = state.side(side).active()[slot].pokemon_index.or(pending) else { continue };
                if state.side(side).team_member(idx).hp <= 0 && pending != Some(idx) {
                    continue;
                }
                let tb = state.side(side).team_boosts[idx as usize].clone();
                let vals = [tb.atk, tb.def, tb.spa, tb.spd, tb.spe];
                let ups: Vec<usize> = (0..5).filter(|&n| vals[n] < 6).collect();
                let plus = if ups.is_empty() {
                    None
                } else {
                    Some(ups[rng.label("residual roll").random(ups.len() as u32) as usize])
                };
                let downs: Vec<usize> = (0..5).filter(|&n| vals[n] > -6 && Some(n) != plus).collect();
                let minus = if downs.is_empty() {
                    None
                } else {
                    Some(downs[rng.label("residual roll").random(downs.len() as u32) as usize])
                };
                let mut b = StatBoosts::default();
                for (n, amt) in [(plus, 2i8), (minus, -1i8)] {
                    if let Some(n) = n {
                        match n {
                            0 => b.attack = amt,
                            1 => b.defense = amt,
                            2 => b.special_attack = amt,
                            3 => b.special_defense = amt,
                            _ => b.speed = amt,
                        }
                    }
                }
                if state.side(side).team_member(idx).hp > 0 {
                    apply_boosts(state, side, idx, &b, false);
                }
            }
            ResidualKind::HealBlock(side, slot) => {
                // `fieldEvent`: duration--; at 0 the volatile ends (`onEnd`).
                let a = state.side_mut(side).active_slot_mut(slot);
                a.heal_block = a.heal_block.saturating_sub(1);
            }
            ResidualKind::SyrupBomb(side, slot) => {
                // `fieldEvent`: duration--; at 0 the volatile ends. `onUpdate`
                // removes it when the source is no longer active; `onResidual`:
                // `boost({spe: -1}, pokemon, source)`.
                let Some((ss, si, d)) = state.side(side).active()[slot].syrup_bomb else { continue };
                let Some(tidx) = state.side(side).active()[slot].pokemon_index else { continue };
                let src_active = (0..2).any(|sl| state.side(ss).active()[sl].pokemon_index == Some(si));
                if !src_active || d <= 1 {
                    state.side_mut(side).active_slot_mut(slot).syrup_bomb = None;
                    continue;
                }
                state.side_mut(side).active_slot_mut(slot).syrup_bomb = Some((ss, si, d - 1));
                apply_boosts(state, side, tidx, &StatBoosts { speed: -1, ..Default::default() }, ss != side);
            }
            ResidualKind::Octolock(side, slot) => {
                // PS octolock `onResidual` (no duration): ends silently once
                // its binder is no longer active or fainted; else boost
                // def/spd -1 from the binder.
                let Some((ss, si)) = state.side(side).active()[slot].octolock else { continue };
                let Some(tidx) = state.side(side).active()[slot].pokemon_index else { continue };
                let src_active = (0..2).any(|sl| state.side(ss).active()[sl].pokemon_index == Some(si));
                if !src_active || state.side(ss).team_member(si).hp <= 0 {
                    state.side_mut(side).active_slot_mut(slot).octolock = None;
                    continue;
                }
                // The drop comes from the binder's active slot, so Mirror Armor
                // (`onTryBoost`) bounces it back onto the binder.
                let drop = StatBoosts { defense: -1, special_defense: -1, ..Default::default() };
                let drop = match (0..2).find(|&sl| state.side(ss).active()[sl].pokemon_index == Some(si)) {
                    Some(bsl) if (ss, bsl) != (side, slot) => mirror_armor_bounce(state, ss, bsl, side, tidx, &drop),
                    _ => drop,
                };
                apply_boosts(state, side, tidx, &drop, ss != side);
            }
            ResidualKind::Ingrain(side, slot) => {
                // PS ingrain `onResidual`: heal baseMaxhp / 16.
                if let Some(idx) = state.side(side).active()[slot].pokemon_index {
                    let blocked = state.side(side).active()[slot].heal_block > 0;
                    let pm = state.side_mut(side).team_member_mut(idx);
                    if pm.hp > 0 && pm.hp < pm.maxhp && !blocked {
                        pm.hp = (pm.hp + (pm.maxhp / 16).max(1)).min(pm.maxhp);
                    }
                }
            }
            ResidualKind::Encore(side, slot) => {
                // `fieldEvent`: duration--; at 0 the volatile ends. `onResidual`:
                // ends early when the encored move is out of PP.
                let Some((m, d)) = state.side(side).active()[slot].encore else { continue };
                if state.side(side).slot_pokemon(slot).map(|p| p.hp <= 0).unwrap_or(true) {
                    continue;
                }
                if d <= 1 {
                    state.side_mut(side).active_slot_mut(slot).encore = None;
                    continue;
                }
                state.side_mut(side).active_slot_mut(slot).encore = Some((m, d - 1));
                let out_of_pp = {
                    let p = state.side(side).slot_pokemon(slot).unwrap();
                    ![&p.moves.m0, &p.moves.m1, &p.moves.m2, &p.moves.m3]
                        .iter()
                        .any(|mv| mv.id == m && mv.pp > 0)
                };
                if out_of_pp {
                    state.side_mut(side).active_slot_mut(slot).encore = None;
                }
            }
            ResidualKind::Roost(side, slot) => {
                // duration 1: the volatile ends here and the types come back.
                // PS roost's `onType` only filters the current types: a `setType`
                // during the turn (Soak after Roost) replaced the base types, and
                // those stay once the filter ends.
                if let Some(t) = state.side_mut(side).active_slot_mut(slot).roost_orig.take() {
                    if let Some(idx) = state.side(side).active()[slot].pokemon_index {
                        if state.side(side).team_member(idx).types == roosted_types(t) {
                            state.side_mut(side).team_member_mut(idx).types = t;
                        } else if state.side(side).active()[slot].type_change == Some(roosted_types(t)) {
                            // The saved switch-out types were taken while roosted.
                            state.side_mut(side).slot_scratch_mut(slot).type_change = Some(t);
                        }
                    }
                }
            }
            ResidualKind::LeechSeed(side, slot) => {
                // PS data/moves.ts leechseed.condition.onResidual: damage
                // baseMaxhp/8 to the seeded pokemon, heal the source's slot
                // occupant by the damage dealt.
                let Some((ss, sslot)) = state.side(side).active()[slot].leech_seed else { continue };
                let Some(p) = state.side(side).slot_pokemon(slot) else { continue };
                if p.hp <= 0 {
                    continue;
                }
                let (hp0, maxhp, ooze_src) = (p.hp, p.maxhp, p.ability == Abilities::LIQUIDOOZE);
                let src_alive = state.side(ss).slot_pokemon(sslot).map(|q| q.hp > 0).unwrap_or(false);
                if !src_alive {
                    continue;
                }
                apply_indirect_damage(state, side, slot, (maxhp / 8).max(1));
                // After a faint the slot is empty: the target took all its HP.
                let dealt = hp0 - state.side(side).slot_pokemon(slot).map(|q| q.hp).unwrap_or(0);
                if dealt > 0 {
                    // Liquid Ooze `onSourceTryHeal`: the drainer takes the damage.
                    if ooze_src {
                        apply_indirect_damage(state, ss, sslot, dealt);
                    } else if let Some(sidx) = state.side(ss).active()[sslot].pokemon_index {
                        let blocked = state.side(ss).active()[sslot].heal_block > 0;
                        let m = state.side_mut(ss).team_member_mut(sidx);
                        if !blocked {
                            m.hp = (m.hp + dealt).min(m.maxhp);
                        }
                    }
                }
            }
            ResidualKind::Curse(side, slot) => {
                if let Some(p) = state.side(side).slot_pokemon(slot) {
                    if p.hp > 0 {
                        let q = (p.maxhp / 4).max(1);
                        apply_indirect_damage(state, side, slot, q);
                    }
                }
            }
            ResidualKind::Perish(side, slot) => {
                if state.side(side).slot_pokemon(slot).map(|p| p.hp <= 0).unwrap_or(true) {
                    continue;
                }
                let a = state.side_mut(side).active_slot_mut(slot);
                a.perish = a.perish.saturating_sub(1);
                if a.perish == 0 {
                    // `onEnd`: `target.faint()`.
                    let hp = state.side(side).slot_pokemon(slot).map(|p| p.hp).unwrap_or(0);
                    apply_damage(state, side, slot, hp);
                    if state.battle_is_over() != 0.0 {
                        perish_wipe = true;
                    }
                    // PS `fieldEvent`: an `end` callback skips `resolveFaints`, so
                    // this faint waits for the residual action's own resolve.
                    prev_end = true;
                    continue;
                }
            }
            ResidualKind::FutureMove(side, slot) => {
                let Some((mv, ss, si, seen)) = state.side(side).future_move[slot] else { continue };
                // `getOverflowedTurnCount() < endingTurn` skips the first two
                // residuals (cast turn and the next one).
                if seen < 2 {
                    state.side_mut(side).future_move[slot] = Some((mv, ss, si, seen + 1));
                    continue;
                }
                state.side_mut(side).future_move[slot] = None;
                resolve_future_move(state, rng, mv, ss, si, side, slot);
            }
            ResidualKind::Wish(side, slot) => {
                if let Some((hp, seen)) = state.side(side).wish[slot] {
                    if seen == 0 {
                        // Same turn the Wish was made: `getOverflowedTurnCount() <= startingTurn`.
                        state.side_mut(side).wish[slot] = Some((hp, 1));
                    } else {
                        state.side_mut(side).wish[slot] = None;
                        if let Some(p) = state.side(side).slot_pokemon(slot) {
                            if p.hp > 0 && state.side(side).active()[slot].heal_block == 0 {
                                let idx = state.side(side).active()[slot].pokemon_index.unwrap();
                                let pm = state.side_mut(side).team_member_mut(idx);
                                pm.hp = (pm.hp + hp).min(pm.maxhp);
                            }
                        }
                    }
                }
            }
            ResidualKind::AquaRing(side, slot) => {
                if let Some(p) = state.side(side).slot_pokemon(slot) {
                    if p.hp > 0 && state.side(side).active()[slot].heal_block == 0 {
                        let idx = state.side(side).active()[slot].pokemon_index.unwrap();
                        let pm = state.side_mut(side).team_member_mut(idx);
                        pm.hp = (pm.hp + (pm.maxhp / 16).max(1)).min(pm.maxhp);
                    }
                }
            }
            ResidualKind::Gravity => {
                if state.gravity.turns_remaining > 0 {
                    state.gravity.turns_remaining -= 1;
                    if state.gravity.turns_remaining == 0 {
                        state.gravity.active = false;
                        prev_end = true;
                    }
                }
            }
            ResidualKind::FairyLock => {
                if state.fairy_lock.turns_remaining > 0 {
                    state.fairy_lock.turns_remaining -= 1;
                    if state.fairy_lock.turns_remaining == 0 {
                        state.fairy_lock.active = false;
                        prev_end = true;
                    }
                }
            }
            ResidualKind::MagicRoom => {
                if state.magic_room.turns_remaining > 0 {
                    state.magic_room.turns_remaining -= 1;
                    if state.magic_room.turns_remaining == 0 {
                        state.magic_room.active = false;
                        prev_end = true;
                    }
                }
            }
            ResidualKind::WonderRoom => {
                if state.wonder_room.turns_remaining > 0 {
                    state.wonder_room.turns_remaining -= 1;
                    if state.wonder_room.turns_remaining == 0 {
                        state.wonder_room.active = false;
                        prev_end = true;
                    }
                }
            }
            ResidualKind::Terrain(t) => {
                // PS `fieldEvent` skips a handler whose effect state was replaced
                // (`expectedStateLocation !== handler.state`): a terrain set
                // earlier in this residual (Seed Sower on a Future Sight hit)
                // does not count down this turn. PS sim/battle.ts fieldEvent.
                if state.terrain.terrain_type != t {
                    continue;
                }
                if state.terrain.turns_remaining > 0 {
                    state.terrain.turns_remaining -= 1;
                    if state.terrain.turns_remaining == 0 {
                        state.terrain.terrain_type = crate::engine::state::Terrain::NONE;
                        // `clearTerrain` -> `eachEvent('TerrainChange')`.
                        refresh_mimicry(state);
                        let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
                        prev_end = true;
                    }
                }
            }
            ResidualKind::LockOn(side, slot) => {
                let a = state.side_mut(side).active_slot_mut(slot);
                if let Some((ts, tidx, d)) = a.lock_on {
                    a.lock_on = if d <= 1 { None } else { Some((ts, tidx, d - 1)) };
                }
            }
            ResidualKind::AllySwitch(side, slot) => {
                let a = state.side_mut(side).active_slot_mut(slot);
                if let Some((c, d)) = a.ally_switch {
                    a.ally_switch = if d <= 1 { None } else { Some((c, d - 1)) };
                }
            }
            ResidualKind::Uproar(side, slot) => {
                // PS data/moves.ts uproar.condition.onResidual: a Throat Chop volatile
                // on the holder ends Uproar (`removeVolatile`, no countdown).
                let a = state.side_mut(side).active_slot_mut(slot);
                if a.throat_chop > 0 {
                    a.uproar = 0;
                } else {
                    a.uproar = a.uproar.saturating_sub(1);
                }
            }
            ResidualKind::Stall(side, slot) => {
                let a = state.side_mut(side).active_slot_mut(slot);
                if let Some((c, d)) = a.stall {
                    a.stall = if d <= 1 { None } else { Some((c, d - 1)) };
                }
            }
            ResidualKind::Locked(side, slot) => {
                if state.side(side).slot_pokemon(slot).map(|p| p.hp <= 0).unwrap_or(true) {
                    continue;
                }
                let Some((mut d, mut t)) = state.side(side).active()[slot].locked_move else { continue };
                d -= 1;
                if d == 0 {
                    // Duration ran out: `onEnd`.
                    state.side_mut(side).active_slot_mut(slot).locked_move = None;
                    if t <= 1 {
                        try_confuse(state, rng, side, slot);
                    }
                    continue;
                }
                if state.side(side).slot_pokemon(slot).map(|p| p.status == PokemonStatus::SLEEP).unwrap_or(false) {
                    state.side_mut(side).active_slot_mut(slot).locked_move = None;
                    continue;
                }
                t = t.saturating_sub(1);
                state.side_mut(side).active_slot_mut(slot).locked_move = Some((d, t));
            }
            ResidualKind::Disable(side, slot) => {
                let a = state.side_mut(side).active_slot_mut(slot);
                if let Some((m, d)) = a.disabled {
                    a.disabled = if d <= 1 { None } else { Some((m, d - 1)) };
                }
            }
            ResidualKind::Yawn(side, slot) => {
                // PS fieldEvent: duration--; at 0 `onEnd` -> `trySetStatus('slp')`.
                if state.side(side).slot_pokemon(slot).map(|p| p.hp <= 0).unwrap_or(true) {
                    continue;
                }
                let a = state.side_mut(side).active_slot_mut(slot);
                if a.yawn == 0 {
                    continue;
                }
                a.yawn -= 1;
                if a.yawn == 0 {
                    let src = a.yawn_source.take();
                    if let Some((ss, si)) = src {
                        let already = state.side(side).slot_pokemon(slot).map(|p| p.status != PokemonStatus::NONE).unwrap_or(true);
                        if !already {
                            try_set_status_yawn(state, rng, side, slot, ss, si);
                        }
                    }
                }
            }
            ResidualKind::PartialTrap(side, slot) => {
                // PS `fieldEvent`: duration-- before the callback; at 0 the
                // volatile ends without damage. `onResidual`: ends if the
                // binder is no longer active, else damages baseMaxhp/divisor.
                let Some(mut t) = state.side(side).active()[slot].partial_trap else { continue };
                if state.side(side).slot_pokemon(slot).map(|p| p.hp <= 0).unwrap_or(true) {
                    continue;
                }
                t.duration -= 1;
                if t.duration == 0 {
                    state.side_mut(side).active_slot_mut(slot).partial_trap = None;
                    continue;
                }
                state.side_mut(side).active_slot_mut(slot).partial_trap = Some(t);
                // PS `partiallytrapped.onResidual`: the binder must be active, alive
                // and have `activeTurns` (a binder switched back in this turn, e.g.
                // by Dragon Tail, ends the trap silently).
                let src_active = (0..2).any(|sl| {
                    let a = &state.side(t.source_side).active()[sl];
                    a.pokemon_index == Some(t.source_idx)
                        && a.active_turns > 0
                        && state.side(t.source_side).team_member(t.source_idx).hp > 0
                });
                if !src_active {
                    state.side_mut(side).active_slot_mut(slot).partial_trap = None;
                    continue;
                }
                let maxhp = state.side(side).slot_pokemon(slot).map(|p| p.maxhp).unwrap_or(0);
                apply_indirect_damage(state, side, slot, (maxhp / t.divisor).max(1));
            }
            ResidualKind::SaltCure(side, slot) => {
                if let Some(p) = state.side(side).slot_pokemon(slot) {
                    let has = |t| p.types.0 == t || p.types.1 == t;
                    let div = if has(PokemonType::WATER) || has(PokemonType::STEEL) { 8 } else { 16 };
                    let dmg = (p.maxhp / div).max(1);
                    apply_indirect_damage(state, side, slot, dmg);
                }
            }
            ResidualKind::GrassyHeal(side, slot) => {
                // PS `grassyterrain.onFieldResidual`: heal grounded actives by
                // 1/16 max HP (gen 8+).
                let heal = {
                    let s = state.side(side);
                    let Some(idx) = s.active()[slot].pokemon_index else { continue };
                    let p = s.team_member(idx);
                    // PS `pokemon.isSemiInvulnerable()` (fly / bounce / dive / dig /
                    // phantomforce / shadowforce / skydrop volatiles).
                    // A charge that was aborted this turn (full paralysis) is gone.
                    let semi_invulnerable = matches!(
                        s.active()[slot].charging.filter(|_| !s.active()[slot].charge_aborted),
                        Some((
                            Choices::FLY
                            | Choices::BOUNCE
                            | Choices::DIVE
                            | Choices::DIG
                            | Choices::PHANTOMFORCE
                            | Choices::SHADOWFORCE
                            | Choices::SKYDROP,
                            _
                        ))
                    );
                    if slot_grounded(p, &s.active()[slot], state.gravity.active) && !semi_invulnerable && p.hp < p.maxhp && s.active()[slot].heal_block == 0 {
                        Some((idx, p.hp, p.maxhp))
                    } else {
                        None
                    }
                };
                if let Some((idx, hp, maxhp)) = heal {
                    let h = (maxhp / 16).max(1);
                    let s = state.side_mut(side);
                    s.team_member_mut(idx).hp = (hp + h).min(maxhp);
                }
            }
            ResidualKind::SpeedBoost(side, slot) => {
                // PS `speedboost.onResidual`, data/abilities.ts:4454. No draw.
                // Only when `activeTurns` > 0 (not on the switch-in turn).
                if state.side(side).active()[slot].active_turns == 0 {
                    continue;
                }
                if let Some(idx) = state.side(side).active()[slot].pokemon_index {
                    apply_boosts(
                        state,
                        side,
                        idx,
                        &StatBoosts {
                            speed: 1,
                            ..Default::default()
                        },
                        false,
                    );
                }
            }
        }
        // PS `fieldEvent`: `resolveFaints()` after each handler's callback, so a
        // faint from one residual is cleared before the next handler runs (a
        // Terrain end then sorts without it). PS sim/battle.ts fieldEvent.
        resolve_faints(state, rng, |_, _| true);
    }

    // Aborted charge: the move's own condition (duration 2) has now ended.
    for side in [Side::One, Side::Two] {
        for slot in 0..2 {
            if state.side(side).active()[slot].charge_aborted {
                let a = state.side_mut(side).active_slot_mut(slot);
                a.charge_aborted = false;
                a.charging = None;
            }
        }
    }

    // End of the `residual` action (faints are resolved before its eachEvent).
    resolve_faints(state, rng, |_, _| true);
    // PS `runAction`: `resolveFaints(...)` then `if (this.ended) return true;`.
    // A battle decided by the residual's (deferred Perish song) faints runs no
    // further step: no emergency exit, replacements, `checkFainted`, or endTurn.
    if state.battle_is_over() != 0.0 {
        return;
    }
    each_event_update(state, rng);
    // Emergency Exit / Wimp Out `onEmergencyExit` after the residual action:
    // crossing 50% HP switches the holder out (the answer is the pivot-style
    // choice). PS data/mods/champions/abilities.ts emergencyexit.
    let mut pending_vec: Vec<(Side, usize, usize)> = pending_in.to_vec();
    for &(sd, sl, before) in &hp_before {
        let Some(p) = state.side(sd).slot_pokemon(sl) else { continue };
        let (after, max) = (p.hp as i32, p.maxhp as i32);
        if matches!(p.ability, Abilities::EMERGENCYEXIT | Abilities::WIMPOUT)
            && after > 0
            && after * 2 <= max
            && (before as i32) * 2 > max
            && state.side(sd).find_replacement().is_some()
        {
            if let Some(&(_, _, pos)) = ee_switches.iter().find(|&&(s2, l2, _)| s2 == sd && l2 == sl) {
                pending_vec.push((sd, sl, pos));
                // PS `runAction` switches loop: `runPursuitActivation` for every live
                // pokemon with a `switchFlag` (gen >= 5: `eachEvent('Update')`).
                each_event_update(state, rng);
            }
        }
    }
    // The `instaswitch` actions are speed-sorted (by the switching pokemon's
    // speed; exact ties shuffle: PS `queue.sort`), which also fixes the order
    // the replacements enter.
    // Repeated slots are consecutive replacement rounds (a replacement that
    // fainted to entry hazards is replaced again): each round is its own
    // `instaswitch` queue + `runSwitch`, with its own eachEvent / switch-in
    // sorts (PS sim/battle.ts runAction, one request per round). The round
    // loop below keeps the block's original indentation.
    let mut rounds: Vec<Vec<(Side, usize, usize)>> = Vec::new();
    for &e in &pending_vec {
        match rounds.last_mut() {
            Some(r) if !r.iter().any(|&(s2, l2, _)| (s2, l2) == (e.0, e.1)) => r.push(e),
            _ => rounds.push(vec![e]),
        }
    }
    for round in rounds {
    let mut pending_vec = round;
    let distinct = pending_vec
        .iter()
        .enumerate()
        .all(|(i, &(a, b, _))| pending_vec.iter().skip(i + 1).all(|&(c, d, _)| (a, b) != (c, d)));
    if pending_vec.len() >= 2 && distinct {
        let mut hs: Vec<ResidualHandler> = Vec::new();
        for (i, &(sd, sl, _)) in pending_vec.iter().enumerate() {
            let sdd = state.side(sd);
            let who = sdd.active()[sl].pokemon_index.or_else(|| sdd.fainted_in_slot[sl].map(|f| f.0));
            let speed = who.map(|w| sdd.speed_cache[w as usize] as i32).unwrap_or(0);
            // A fainted pokemon's cached speed is its raw stat; PS
            // `getActionSpeed` for its switch action still applies Trick Room
            // (`10000 - speed`).
            let speed = if sdd.active()[sl].pokemon_index.is_none() && who.is_some() && state.trick_room.active {
                10000 - speed
            } else {
                speed
            };
            hs.push(ResidualHandler {
                order: 0,
                priority: 0,
                speed,
                sub_order: 0,
                kind: ResidualKind::Status(sd, i),
            });
        }
        let sorted = sort_residual_handlers(hs, rng, "speed tie (move order)");
        let old = pending_vec.clone();
        pending_vec = sorted
            .into_iter()
            .filter_map(|h| match h.kind {
                ResidualKind::Status(_, i) => Some(old[i]),
                _ => None,
            })
            .collect();
    }
    let pending_switches: &[(Side, usize, usize)] = &pending_vec;
    check_fainted(state);

    // Switch in replacements (`instaswitch` actions), applying entry hazards,
    // then one `runSwitch` action for all of them. The switch answers go
    // through `commitChoices` -> `updateSpeed()` first.
    if !pending_switches.is_empty() {
        update_speed(state);
    }
    // PS `switchIn` -> `queue.insertChoice(runSwitch)`: a tie with an already
    // queued `runSwitch` (equal speed) draws `random(firstIndex, lastIndex + 1)`.
    let mut run_switch_speeds: Vec<i16> = Vec::new();
    // One runSwitch batch (distinct slots): the entry hazards of all incoming
    // pokemon run together after the switch-in speed sort (below).
    let batch_distinct = pending_switches.iter().enumerate().all(|(i, &(a, b, _))| {
        pending_switches.iter().skip(i + 1).all(|&(c, d, _)| (a, b) != (c, d))
    });
    for (side, slot, position) in pending_switches {
        if batch_distinct {
            switch_in_inner(state, *side, *slot, *position);
        } else {
            switch_in(state, *side, *slot, *position);
        }
        // PS `runAction` -> `eachEvent('Update')` after the switch-in: the new
        // holder's `onUpdate` cures (Insomnia wakes a sleeping replacement).
        ability_update_cures(state, *side, *slot);
        update_speed(state);
        if let Some(sp) = effective_speed(state, *side, *slot) {
            let ties = run_switch_speeds.iter().filter(|&&x| x == sp).count();
            if ties > 0 {
                let _ = rng.label("speed tie (queue insert)").random(ties as u32 + 1);
            }
            run_switch_speeds.push(sp);
        }
    }
    // The order the incoming pokemon's `onStart` abilities run in. Repeated
    // slots are consecutive replacement rounds: an earlier occupant fainted to
    // the hazards (no `onStart`), so each slot runs its final occupant once.
    let mut ability_order: Vec<(Side, usize)> = Vec::new();
    for &(sd, sl, _) in pending_switches {
        if !ability_order.contains(&(sd, sl)) {
            ability_order.push((sd, sl));
        }
    }
    if !pending_switches.is_empty() {
        each_event_update(state, rng);
        let order = sorted_actives(state, rng, "speed tie (switch-in)");
        if batch_distinct {
            // PS sim/battle.ts `findPokemonEventHandlers`: an ability's `onStart`
            // is a `SwitchIn` handler, so it runs in the same speed order.
            ability_order = order
                .iter()
                .copied()
                .filter(|&(sd, sl)| pending_switches.iter().any(|&(ps, pl, _)| ps == sd && pl == sl))
                .collect();
            // PS `fieldEvent('SwitchIn')`: each incoming pokemon's handlers all
            // share its speed, so they run as one group per pokemon, fastest
            // first (`sorted_actives` order; `speedOrder` breaks speed ties).
            // Its hazards (side condition, subOrder 4) run before its ability
            // (subOrder 7), and a faster pokemon's ability before a slower
            // pokemon's hazards (two weather setters: the slower one's stays).
            for &(side, slot) in &order {
                if pending_switches.iter().any(|&(ps, pl, _)| ps == side && pl == slot) {
                    entry_hazards(state, side, slot);
                    apply_sticky_web(state, side, slot);
                    refresh_mimicry_slot(state, side, slot);
                    // `fieldEvent` runs `resolveFaints` after each handler: a
                    // replacement KO'd by the hazards skips its ability's Start.
                    if switched_in_alive(state, side, slot) && apply_on_switch_abilities(state, side, slot, rng) {
                        let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
                    }
                }
            }
            ability_order.clear();
        } else {
            for (side, slot, _) in pending_switches {
                apply_sticky_web(state, *side, *slot);
            }
        }
    }

    // On-switch abilities (Intimidate) for each newly switched-in pokemon.
    // TODO(PS parity): `fieldEvent('SwitchIn')` handler ties.
    for (side, slot) in ability_order {
        if switched_in_alive(state, side, slot) && apply_on_switch_abilities(state, side, slot, rng) {
            let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
        }
    }
    if !pending_switches.is_empty() {
        each_event_update(state, rng);
    }
    } // replacement rounds

    // PS `endTurn`: `runEvent('DisableMove', pokemon)` for every active. The
    // handler list (the pokemon's own `onDisableMove` volatiles / ability,
    // foes' `onFoeDisableMove` (Imprison), the field's Gravity) is
    // speed-sorted, shuffling ties (a PRNG draw per tie group).
    end_turn_disable_moves(state, rng);

    // Tick down durations.

    for side in [Side::One, Side::Two] {
        let s = state.side_mut(side);
        // Clear per-turn volatiles.
        for a in s.active_mut().iter_mut() {
            a.counter_vol = None;
            a.mirror_coat_vol = None;
            a.active_turns = a.active_turns.saturating_add(1);
            a.move_result_last_turn = a.move_result_this_turn;
            a.move_result_this_turn = None;
            a.flinched = false;
            a.electrify = false;
            a.hurt_this_turn = false;
            a.hurt_hp = 0;
            // PS `battle.ts` turn start: `pokemon.newlySwitched = false`.
            a.newly_switched = false;
            a.protected = false;
            a.twoturn_fired = false;
            a.endure = false;
            a.follow_me = 0;
            a.helping_hand_mult = 1.0;
            a.focus_punch = false;
            a.lost_focus = false;
            a.beak_blast = false;
            a.chilly_reception = false;
            a.stats_raised_this_turn = false;
            a.stats_lowered_this_turn = false;
            a.last_attacker = None;
        }
    }
    // PS `endTurn` (sim/battle.ts): an active pokemon forgets the attacks whose
    // source is no longer active (fainted or switched out), so a fainted
    // attacker is never a Metal Burst / Comeuppance target later. Engine: a
    // source counts as active while its slot holds a live pokemon.
    for side in [Side::One, Side::Two] {
        for sl in 0..2 {
            let Some(pidx) = state.side(side).active()[sl].pokemon_index else { continue };
            if !state.side(side).slot_pokemon(sl).map(|p| p.hp > 0).unwrap_or(false) {
                continue;
            }
            let mut kept = std::mem::take(&mut state.side_mut(side).damaged_by[pidx as usize]);
            kept.retain(|e| {
                // PS `source.isActive`: the attacker itself is on the field and alive.
                (0..2).any(|asl| {
                    state.side(e.0).active()[asl].pokemon_index.map(|x| x as usize) == Some(e.4)
                        && state.side(e.0).slot_pokemon(asl).map(|p| p.hp > 0).unwrap_or(false)
                })
            });
            state.side_mut(side).damaged_by[pidx as usize] = kept;
        }
    }
}

pub fn run_battle(state: &mut DoublesState, rng: &mut Gen5Rng) -> BattleResult {
    let mut turns: Vec<TurnEntry> = Vec::new();

    loop {
        if state.battle_is_over() != 0.0 {
            break;
        }

        let turn_no = state.turn;
        let mut order: Vec<String> = Vec::new();
        let mut move_results: Vec<MoveResult> = Vec::new();
        let mut switch_results: Vec<SwitchResult> = Vec::new();
        // Slots that fainted this turn and need a switch-in at end of turn.
        let mut pending_switches: Vec<(Side, usize)> = Vec::new();

        // Build the action list in field-position order [A0, A1, B0, B1].
        let mut actions: Vec<Action> = Vec::new();
        for (side, slots) in [(Side::One, 0..2), (Side::Two, 0..2)] {
            for slot in slots {
                let s = state.side(side);
                if let Some(idx) = s.active()[slot].pokemon_index {
                    if let Some(choice) = active_move(state, side, slot) {
                        let p = s.team_member(idx);
                        actions.push(Action {
                            side,
                            slot,
                            pokemon_index: idx,
                            choice,
                            priority: 0,
                            boost: 0,
                            speed: effective_speed(state, side, slot).unwrap_or(0),
                            target: 0,
                            target_loc: 0,
                            fractional: 0,
                            prioritized: false,
                            no_pp: false,
                            recharge: s.active()[slot].recharging,
                            orig_target: None,
                        });
                    }
                }
            }
        }

        // Sort by priority/speed, shuffling ties (consumes rng draws).
        speed_sort(&mut actions, rng);

        // Execute in order.
        let n = actions.len();
        for i in 0..n {
            let action = &actions[i];
            let attacker_name = {
                let s = state.side(action.side);
                s.team_member(action.pokemon_index).id.to_string()
            };
            // Re-check the attacker is still active & alive.
            let still_active = state
                .side(action.side)
                .active()[action.slot]
                .pokemon_index
                .is_some();
            if !still_active {
                continue;
            }
            let target_side = action.side.other();
            // "Location 1" is the foe's leftmost active. If it has fainted this
            // turn, the move retargets, and pokemon-showdown consumes PRNG draws:
            //   1 sample when the remaining moves' targets are re-resolved after the
            //   faint, and 2 more samples when this move executes.
            let loc1_fainted = !state
                .side(target_side)
                .slot_pokemon(0)
                .map(|p| p.hp > 0)
                .unwrap_or(false);
            if loc1_fainted {
                let alive_foes = state.side(target_side).active().iter().filter(|s| {
                    s.pokemon_index
                        .map(|idx| state.side(target_side).team_member(idx).hp > 0)
                        .unwrap_or(false)
                }).count() as u32;
                // Execution re-resolution (useMove): 1 sample for the retargeting move.
                rng.random(alive_foes.max(1));
            }
            let target_slot = choose_target(state, target_side);
            let target_present = state
                .side(target_side)
                .slot_pokemon(target_slot)
                .map(|p| p.hp > 0)
                .unwrap_or(false);
            let target_name = state
                .side(target_side)
                .slot_pokemon(target_slot)
                .map(|p| p.id.to_string())
                .unwrap_or_default();

            // Paralysis flinch: champions mod uses 1/8 chance to skip the move.
            let flinched = {
                let att = state.side(action.side).team_member(action.pokemon_index);
                att.status == PokemonStatus::PARALYZE && rng.random_chance(1, 8)
            };
            if flinched {
                order.push(format!("{}{}:{}", side_tag(action.side), action.slot, attacker_name));
                move_results.push(MoveResult {
                    attacker: attacker_name,
                    move_name: action.choice.move_id.to_string(),
                    target: target_name,
                    damage: 0,
                    crit: false,
                    missed: true,
                    dmg_draw: None,
                });
                continue;
            }

            // Accuracy roll.
            let hits = roll_accuracy(action.choice.accuracy, rng);
            let mut dmg_draw: Option<u32> = None;
            let (damage, crit) = if hits && target_present {
                let attacker = state.side(action.side).team_member(action.pokemon_index);
                let defender = state
                    .side(target_side)
                    .slot_pokemon(target_slot)
                    .expect("target must be present");
                let is_physical = action.choice.category == MoveCategory::Physical;
                let att_boost = state
                    .side(action.side)
                    .slot_boosts(action.slot)
                    .cloned()
                    .unwrap_or_default();
                let def_boost = state
                    .side(target_side)
                    .slot_boosts(target_slot)
                    .cloned()
                    .unwrap_or_default();
                let (attack_base, attack_boost) = if is_physical {
                    (attacker.attack, att_boost.atk)
                } else {
                    (attacker.special_attack, att_boost.spa)
                };
                let (defense_base, defense_boost) = if is_physical {
                    (defender.defense, def_boost.def)
                } else {
                    (defender.special_defense, def_boost.spd)
                };
                let ctx = DamageCtx {
                    added_type: state.side(target_side).active()[target_slot].added_type,
                    attacker_added_type: state.side(action.side).active()[action.slot].added_type,
                    attack_base,
                    attack_boost,
                    defense_base,
                    defense_boost,
                    weather: effective_weather(state),
                    reflect: false,
                    light_screen: false,
                    defender_mod: 4096,
                    solid_rock: false,
                    no_crit: false,
                    sniper: false,
                    guts: attacker.ability == Abilities::GUTS,
                    attack_mod: crate::engine::mechanics::attacker_stat_modifier(attacker, &action.choice),
                    crit_bonus: 0,
                    screen_handlers: 0,
                    friend_guard_ties: [0; 4],
                    friend_guard: false,
                    disguise: false,
                    defense_mod: 4096,
                    base_power_override: 0.0,
                };
                let outcome: DamageOutcome = calculate_damage(attacker, defender, &action.choice, rng, false, &ctx)
                    .unwrap_or(DamageOutcome { damage: 0, crit: false });
                dmg_draw = Some(rng.draw_count() as u32 - 1);
                let defender_ability = defender.ability;
                if apply_damage(state, target_side, target_slot, outcome.damage) {
                    // Re-resolve the targets of ALL remaining actions (turnLoop): each
                    // consumes 1 sample sized to that action's target side.
                    for j in (i + 1)..n {
                        let a2 = &actions[j];
                        let foe_side = a2.side.other();
                        let alive_foes = state.side(foe_side).active().iter().filter(|s| {
                            s.pokemon_index
                                .map(|idx| state.side(foe_side).team_member(idx).hp > 0)
                                .unwrap_or(false)
                        }).count() as u32;
                        rng.random(alive_foes.max(1));
                    }
                    pending_switches.push((target_side, target_slot));
                }
                // Static: 30% chance to paralyze the attacker on a physical hit.
                // PS skips the draw when the attacker already has a status
                // (can't be paralyzed anyway).
                if defender_ability == Abilities::STATIC
                    && is_physical
                    && outcome.damage > 0
                {
                    if rng.random_chance(30, 100) {
                        let att_mut = state
                            .side_mut(action.side)
                            .team_member_mut(action.pokemon_index);
                        if att_mut.status == PokemonStatus::NONE {
                            att_mut.status = PokemonStatus::PARALYZE;
                        }
                    }
                }
                (outcome.damage, outcome.crit)
            } else {
                (0, false)
            };

            order.push(format!("{}{}:{}", side_tag(action.side), action.slot, attacker_name));
            move_results.push(MoveResult {
                attacker: attacker_name,
                dmg_draw,
                move_name: action.choice.move_id.to_string(),
                target: target_name,
                damage,
                crit,
                missed: !hits,
            });
        }

        // Record any switches that happened this turn.
        // End of turn: switch in replacements for fainted slots.
        for (side, slot) in pending_switches {
            let in_name = {
                let s = state.side(side);
                s.find_replacement().map(|i| s.team_member(i).id.to_string())
            };
            process_switch(state, rng, side, slot);
            if let Some(name) = in_name {
                switch_results.push(SwitchResult {
                    side: side_tag(side).to_string(),
                    slot,
                    in_pokemon: name,
                });
            }
        }

        if crate::doubles::debug::flags().slot {
            let slotname = |s: &crate::doubles::state::DoublesSide| {
                s.active()
                    .iter()
                    .map(|sl| sl.pokemon_index.map(|i| s.team_member(i).id.to_string()).unwrap_or_else(|| "EMPTY".into()))
                    .collect::<Vec<_>>()
                    .join(",")
            };
            eprintln!("[T{}] A=[{}] B=[{}]", turn_no, slotname(&state.side_one), slotname(&state.side_two));
        }

        state.turn += 1;
        turns.push(TurnEntry {
            turn: turn_no,
            order,
            moves: move_results,
            switches: switch_results,
            hp: hp_snapshot(state),
        });
    }

    let winner = match state.battle_is_over() {
        1.0 => "A",
        -1.0 => "B",
        _ => "draw",
    };
    BattleResult {
        winner: winner.to_string(),
        total_turns: state.turn,
        prng_draws: rng.draw_log.clone(),
        turns,
        replay_hp: Vec::new(),
    }
}
