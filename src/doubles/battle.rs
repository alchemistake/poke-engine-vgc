//! The doubles battle driver.
//!
//! Drives a full doubles battle turn by turn, consuming the shared `Gen5Rng`
//! in the exact order pokemon-showdown does, so damage rolls / criticals /
//! accuracy / speed-tie shuffles all line up with a pokemon-showdown battle run
//! on the same seed and the same move choices.

use crate::choices::{ChargeSetup, Choice, Choices, Effect, MoveCategory, MoveTarget, StatBoosts, MOVES};
use crate::doubles::damage::{calculate_damage, roll_accuracy, roll_accuracy_boosted, DamageCtx, DamageOutcome};
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
    if actions.len() < 2 {
        return;
    }
    let compare = |a: &Action, b: &Action| -> i32 {
        let dp = (b.priority - a.priority) as i32;
        if dp != 0 {
            return dp;
        }
        let df = (b.fractional - a.fractional) as i32;
        if df != 0 {
            return df;
        }
        (b.speed - a.speed) as i32
    };
    let mut sorted = 0usize;
    while sorted + 1 < actions.len() {
        let mut next_indexes = vec![sorted];
        for i in (sorted + 1)..actions.len() {
            let delta = compare(&actions[next_indexes[0]], &actions[i]);
            if delta < 0 {
                continue;
            } else if delta > 0 {
                next_indexes = vec![i];
            } else {
                next_indexes.push(i);
            }
        }
        for (i, &index) in next_indexes.iter().enumerate() {
            if index != sorted + i {
                actions.swap(sorted + i, index);
            }
        }
        if next_indexes.len() > 1 {
            shuffle(actions, sorted, sorted + next_indexes.len(), rng, "speed tie (move order)");
        }
        sorted += next_indexes.len();
    }
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
    let s = state.side(side);
    let idx = s.active[slot].pokemon_index?;
    let p = s.team_member(idx);
    let boosted = multiply_boost(s.team_boosts[idx as usize].spe, p.speed);
    let speed = if p.status == PokemonStatus::PARALYZE {
        boosted / 2
    } else {
        boosted
    };
    if state.trick_room.active {
        Some(-speed)
    } else {
        Some(speed)
    }
}

/// Remove a pokemon from its active slot, clearing per-pokemon state the way
/// pokemon-showdown does on faint (boosts cleared; the slot stays empty until
/// the end-of-turn switch).
fn faint_pokemon(state: &mut DoublesState, side: Side, slot: usize) {
    if let Some(idx) = state.side(side).active[slot].pokemon_index {
        state.unresolved_faints.push((side, slot, idx));
    }
    let s = state.side_mut(side);
    if let Some(idx) = s.active[slot].pokemon_index {
        s.clear_member_boosts(idx);
        if let Some(t) = s.active[slot].type_change {
            s.team_member_mut(idx).types = t;
        }
        // The status stays until PS `checkFainted` (end of turn, only for a
        // side that can still switch) sets it to 'fnt'; see `check_fainted`.
        s.active[slot] = ActiveSlot::empty();
    }
}

/// Apply damage to a specific slot; returns true if the pokemon fainted.
fn apply_damage(state: &mut DoublesState, side: Side, slot: usize, amount: i16) -> bool {
    if amount <= 0 {
        return false;
    }
    let s = state.side_mut(side);
    if let Some(idx) = s.active[slot].pokemon_index {
        let p = s.team_member_mut(idx);
        p.hp -= amount;
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
    // The active slots are fixed by switch-in order in pokemon-showdown; the
    // speedSort only determines move order, not slot assignment. No-op here.
    let _ = (state, rng);
}

/// Switch a replacement into a fainted slot at the end of the turn, and
/// replicate pokemon-showdown's `runSwitch` speed re-sort of all actives
/// (which consumes PRNG draws for speed ties).
fn process_switch(state: &mut DoublesState, rng: &mut Gen5Rng, side: Side, slot: usize) {
    {
        let s = state.side_mut(side);
        s.active[slot] = ActiveSlot::empty();
        if let Some(repl) = s.find_replacement() {
            s.active[slot].pokemon_index = Some(repl);
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
    let idx = s.active[slot].pokemon_index?;
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
                name: s.species_names[i].clone(),
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
        if s.active[mv.att_slot].pokemon_index.is_some() {
            return;
        }
        let name = mv.att.to_lowercase();
        let mut found = None;
        for pos in 0..6 {
            let idx = PokemonIndex::deserialize(&pos.to_string());
            let p = s.team[pos].clone();
            let is_active = s.active.iter().any(|sl| sl.pokemon_index == Some(idx));
            if p.hp > 0 && !is_active && p.id.to_string().to_lowercase() == name {
                found = Some(idx);
                break;
            }
        }
        if found.is_none() {
            found = s.find_replacement();
        }
        if let Some(idx) = found {
            s.active[mv.att_slot].pokemon_index = Some(idx);
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
                    s.species_names[i] = new_name;
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
                for sl in s.active.iter_mut() {
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
                    for sl in s.active.iter_mut() {
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
    rng.context = "turn 0: switch-in".to_string();
    // PS `case "start"`: each lead's `switchIn` inserts a `runSwitch` action
    // into the queue (`insertChoice`), drawing on speed ties before the
    // `runSwitch` speed sort below.
    queue_insert_draws(state, rng);
    let _ = sorted_actives(state, rng, "speed tie (switch-in)");
    for (side, slot) in [(Side::One, 0), (Side::One, 1), (Side::Two, 0), (Side::Two, 1)] {
        if apply_on_switch_abilities(state, side, slot, rng) {
            // Terrain just set (e.g. Grassy Surge): `eachEvent("TerrainChange")`.
            let _ = sorted_actives(state, rng, "speed tie (switch-in)");
        }
    }
    each_event_update(state, rng);
    // Turn 0: the initial state right after the leads are sent out (stat
    // calc, species data, leads). Compared before any turn is simulated.
    let mut init = fuzz_snapshot(state, Vec::new());
    init.turn = 0;
    turn_states.push(init);

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
        if std::env::var("PE_DEBUG_DMG").is_ok() {
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
                        struggle,
                    } => {
                        if let Some(idx) = s.active[slot].pokemon_index {
                            let p = s.team_member(idx);
                            let (mv, prio) = if *struggle {
                                // Struggle (forced when no usable moves left).
                                let m = MOVES.get(&crate::choices::Choices::STRUGGLE).cloned().unwrap();
                                (m, 0i8)
                            } else {
                                let m = move_by_index(p, (*mi).min(3));
                                // PS prankster `onModifyPriority`: +1 on Status moves.
                                let boost = (p.ability == Abilities::PRANKSTER
                                    && m.choice.category == MoveCategory::Status)
                                    as i8;
                                // A recharging pokemon's action is the `recharge`
                                // pseudo-move (priority 0), not the chosen move.
                                let base = if s.active[slot].recharging { 0 } else { m.choice.priority + boost };
                                (m.choice.clone(), base)
                            };
                            let charge_id = mv.move_id;
                            actions.push(Action {
                                side,
                                slot,
                                pokemon_index: idx,
                                choice: mv,
                                priority: prio,
                                speed: effective_speed(state, side, slot).unwrap_or(0),
                                // A locked two-turn move (Dig / Meteor Beam
                                // firing turn) keeps the charge turn's target
                                // (PS side.ts `chooseMove`: lockedMoveTargetLoc).
                                target: match s.active[slot].charging {
                                    Some((m, _)) if m == charge_id => s.active[slot].charge_target,
                                    _ if *target > 0 => (*target).min(2) as usize - 1,
                                    _ => 0,
                                },
                                target_loc: match s.active[slot].charging {
                                    Some((m, _)) if m == charge_id => s.active[slot].charge_target as i8 + 1,
                                    // A locked move (Thrash, ...) reuses
                                    // `lastMoveTargetLoc` (side.ts `chooseMove`).
                                    _ if s.active[slot].locked_move.is_some() || s.active[slot].uproar > 0 => {
                                        s.active[slot].last_target_loc
                                    }
                                    _ => *target,
                                },
                                fractional: 0,
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
        if std::env::var("PE_DEBUG_SPEED").is_ok() {
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
        rng.context = format!("turn {}: move order", turn_no);
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
            if matches!(a.choice.move_id, Choices::COUNTER | Choices::MIRRORCOAT) {
                let mut charge = a.clone();
                charge.target_loc = 0;
                let _ = ps_get_target(state, rng, &charge);
                let vol = &mut state.side_mut(a.side).active[a.slot];
                if a.choice.move_id == Choices::COUNTER {
                    vol.counter_vol = Some((None, 0));
                } else {
                    vol.mirror_coat_vol = Some((None, 0));
                }
            }
            if a.choice.move_id == Choices::FOCUSPUNCH {
                let mut charge = a.clone();
                charge.target_loc = 0;
                let _ = ps_get_target(state, rng, &charge);
            }
            // 2. FractionalPriority event.
            let ability = state.side(a.side).team_member(a.pokemon_index).ability;
            a.fractional = match crate::engine::mechanics::fractional_priority(ability, a.choice.category) {
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
            if a.target_loc == 0 && !state.side(a.side).active[a.slot].recharging {
                if let Some((ts, tsl)) = ps_get_target(state, rng, a) {
                    a.target_loc = if ts == a.side { -(tsl as i8 + 1) } else { tsl as i8 + 1 };
                }
            }
            let _ = ps_get_target(state, rng, a);
        }
        speed_sort(&mut actions, rng);
        // `beforeTurn` action: `eachEvent('BeforeTurn')`, then the action's
        // closing `eachEvent('Update')`; each is a speed sort of the actives.
        each_event_update(state, rng);
        each_event_update(state, rng);
        // `priorityChargeMove` actions (order 107: after `beforeTurn`, before
        // every move): Focus Punch adds its volatile. When one is queued,
        // `beforeTurn` is not followed by a move, so its re-sort is skipped;
        // each charge action ends with `eachEvent('Update')`, the last with
        // the re-sort.
        let chargers: Vec<(Side, usize, PokemonIndex)> = actions
            .iter()
            .filter(|a| a.choice.move_id == Choices::FOCUSPUNCH)
            .map(|a| (a.side, a.slot, a.pokemon_index))
            .collect();
        for &(cs, csl, cidx) in &chargers {
            if state.side(cs).active[csl].pokemon_index == Some(cidx) {
                let a = &mut state.side_mut(cs).active[csl];
                a.focus_punch = true;
                a.lost_focus = false;
            }
            each_event_update(state, rng);
        }
        resort_queue(state, rng, &mut actions[..]);

        // Execute in order. A pivot move (Volt Switch / U-turn) switches its
        // user out right after it acts, using the controller's answer to PS's
        // mid-turn switch request (PS `runAction` -> `makeRequest('switch')`).
        let mut move_order: Vec<String> = Vec::new();
        let mut used_switches = vec![false; turn.switches.len()];
        let mut i = 0;
        while i < actions.len() {
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
            match (0..2).find(|&sl| state.side(action.side).active[sl].pokemon_index == Some(action.pokemon_index)) {
                Some(sl) => action.slot = sl,
                None => continue,
            }
            rng.context = format!(
                "turn {}: {}{} {} {}",
                turn_no,
                side_tag(action.side),
                action.slot,
                state.side(action.side).species_names[action.pokemon_index as usize],
                action.choice.move_id
            );
            execute_action(state, rng, &action, &mut move_order, i < actions.len());
            // Phazing (PS `runAction`: `forceSwitchFlag` -> `dragIn`) happens
            // before faints are resolved.
            drag_in_all(state, rng);
            state.unresolved_faints.clear();
            if state.battle_is_over() != 0.0 {
                break;
            }
            // End of the move's `runAction`.
            each_event_update(state, rng);
            for (side, slot) in [(Side::One, 0usize), (Side::One, 1), (Side::Two, 0), (Side::Two, 1)] {
                if !state.side(side).active[slot].force_switch {
                    continue;
                }
                state.side_mut(side).active[slot].force_switch = false;
                let key = if side == Side::One { "a" } else { "b" };
                let found = turn.switches.iter().enumerate().find(|(i, sw)| {
                    !used_switches[*i] && sw.pivot && sw.side == key && sw.slot == slot
                });
                if let Some((i, sw)) = found {
                    used_switches[i] = true;
                    if let Some(position) = sw.position {
                        // The switch answer (`commitChoices` -> `updateSpeed`),
                        // the `instaswitch` action, then the `runSwitch` action.
                        update_speed(state);
                        // Baton Pass (`selfSwitch: 'copyvolatile'`): boosts and
                        // the copyable volatiles move to the replacement.
                        let passed = if state.side(side).active[slot].baton_passing {
                            let a = &state.side(side).active[slot];
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
                        switch_in(state, side, slot, position);
                        if let Some((boosts, confusion, sub)) = passed {
                            if let Some(new_idx) = state.side(side).active[slot].pokemon_index {
                                state.side_mut(side).team_boosts[new_idx as usize] = boosts;
                                let a = &mut state.side_mut(side).active[slot];
                                a.confusion = confusion;
                                a.substitute_health = sub;
                            }
                        }
                        each_event_update(state, rng);
                        let _ = sorted_actives(state, rng, "speed tie (switch-in)");
                        // TODO(PS parity): `fieldEvent('SwitchIn')` handler ties.
                        if apply_on_switch_abilities(state, side, slot, rng) {
                            let _ = sorted_actives(state, rng, "speed tie (switch-in)");
                        }
                        each_event_update(state, rng);
                    }
                }
            }
            if i < actions.len() {
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

        // End of turn: residual damage, switches (with hazards), durations.
        rng.context = format!("turn {}: end of turn", turn_no);
        // PS stops at once when a side is wiped out: no residual step.
        if state.battle_is_over() == 0.0 {
            end_of_turn(state, rng, &pending_switches);
        }

        turn_states.push(fuzz_snapshot(state, move_order));
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

/// pokemon-showdown `getTarget` (sim/battle.ts) for a queued move, for doubles:
/// returns the resolved target (side, slot), consuming the PRNG draw PS makes
/// when it has to pick a random target (`getRandomTarget` -> `randomFoe` ->
/// `sample`, which draws even with one candidate). PS calls it in
/// `getActionSpeed` (queue build and every re-sort) and in `runMove`.
fn ps_get_target(state: &DoublesState, rng: &mut Gen5Rng, action: &Action) -> Option<(Side, usize)> {
    let t = ps_get_target_inner(state, rng, action)?;
    Some(redirect_target(state, action, t))
}

/// PS `priorityEvent('RedirectTarget')` (sim/pokemon.ts `getMoveTargets`):
/// Lightning Rod / Storm Drain `onAnyRedirectTarget` pull an Electric / Water
/// move with a single-target type to the holder (any other active that is a
/// valid target). No draw.
// TODO(PS parity): speed order / ties between holders, Mold Breaker, tracksTarget.
fn redirect_target(state: &DoublesState, action: &Action, target: (Side, usize)) -> (Side, usize) {
    // Counter / Mirror Coat `onRedirectTarget`: the recorded attacker's slot.
    let a = &state.side(action.side).active[action.slot];
    let rec = match action.choice.move_id {
        Choices::COUNTER => a.counter_vol,
        Choices::MIRRORCOAT => a.mirror_coat_vol,
        _ => None,
    };
    if let Some((Some(t), _)) = rec {
        return t;
    }
    let ty = crate::doubles::move_targets::ps_move_target(&action.choice.move_id.to_string().to_lowercase());
    if !matches!(ty, "normal" | "randomNormal" | "adjacentFoe" | "any" | "scripted") {
        return target;
    }
    let want = match action.choice.move_type {
        PokemonType::ELECTRIC => Abilities::LIGHTNINGROD,
        PokemonType::WATER => Abilities::STORMDRAIN,
        _ => return target,
    };
    for (s, sl) in [(Side::One, 0usize), (Side::One, 1), (Side::Two, 0), (Side::Two, 1)] {
        if (s, sl) == (action.side, action.slot) {
            continue;
        }
        if let Some(p) = state.side(s).slot_pokemon(sl) {
            if p.hp > 0 && p.ability == want {
                // 'adjacentFoe' only accepts foes.
                if ty == "adjacentFoe" && s == action.side {
                    continue;
                }
                return (s, sl);
            }
        }
    }
    target
}

fn ps_get_target_inner(state: &DoublesState, rng: &mut Gen5Rng, action: &Action) -> Option<(Side, usize)> {
    let (side, slot) = (action.side, action.slot);
    let a = &state.side(side).active[slot];
    // A recharging pokemon's locked action is the `recharge` pseudo-move,
    // which has no target type.
    let move_id = action.choice.move_id.to_string().to_lowercase();
    let user_types = state.side(side).team_member(action.pokemon_index).types;
    let ty = if a.recharging {
        ""
    } else if action.choice.move_id == Choices::CURSE
        && user_types.0 != PokemonType::GHOST
        && user_types.1 != PokemonType::GHOST
    {
        // curse.onModifyMove: `move.target = move.nonGhostTarget` ('self').
        "self"
    } else {
        crate::doubles::move_targets::ps_move_target(&move_id)
    };
    let alive = |sd: Side, sl: usize| state.side(sd).slot_pokemon(sl).map(|p| p.hp > 0).unwrap_or(false);
    let foe = side.other();

    let random_target = |rng: &mut Gen5Rng| -> Option<(Side, usize)> {
        match ty {
            "self" | "all" | "allySide" | "allyTeam" | "adjacentAllyOrSelf" => Some((side, slot)),
            "adjacentAlly" => {
                if alive(side, 1 - slot) {
                    let _ = rng.label("retarget").random(1);
                    Some((side, 1 - slot))
                } else {
                    None
                }
            }
            _ => {
                let foes: Vec<usize> = (0..2).filter(|&s| alive(foe, s)).collect();
                if foes.is_empty() {
                    None
                } else {
                    Some((foe, foes[rng.label("retarget").random(foes.len() as u32) as usize]))
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
    if matches!(ty, "adjacentAlly" | "any" | "normal") && loc == self_loc && a.charging.is_none() {
        return None;
    }
    let is_self = loc == self_loc;
    let is_foe = loc > 0;
    // In doubles every other slot is adjacent.
    let valid = loc.abs() <= 2
        && match ty {
            "randomNormal" | "scripted" | "normal" => !is_self,
            "adjacentAlly" => !is_self && !is_foe,
            "adjacentAllyOrSelf" => !is_foe,
            "adjacentFoe" => is_foe,
            "any" => !is_self,
            _ => false,
        };
    if ty != "randomNormal" && valid {
        let (ts, tsl) = if is_foe { (foe, (loc - 1) as usize) } else { (side, (-loc - 1) as usize) };
        if alive(ts, tsl) {
            return Some((ts, tsl));
        }
        if !is_foe {
            // Fainted ally: no retarget.
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
fn update_speed(state: &mut DoublesState) {
    for side in [Side::One, Side::Two] {
        for slot in 0..2 {
            let sp = effective_speed(state, side, slot);
            let s = state.side_mut(side);
            if let (Some(idx), Some(sp)) = (s.active[slot].pokemon_index, sp) {
                s.speed_cache[idx as usize] = sp;
            }
        }
    }
}

/// The cached PS `pokemon.speed` of a slot's occupant.
fn cached_speed(state: &DoublesState, side: Side, slot: usize) -> i32 {
    let s = state.side(side);
    s.active[slot].pokemon_index.map(|i| s.speed_cache[i as usize] as i32).unwrap_or(0)
}

/// PS `checkFainted` (end of turn): on a side that can still switch, fainted
/// pokemon get status 'fnt' (the fuzzer reads it as none). A side with no
/// replacement keeps the fainted pokemon's last status.
fn check_fainted(state: &mut DoublesState) {
    for side in [Side::One, Side::Two] {
        if possible_switches(state, side).is_empty() {
            continue;
        }
        let s = state.side_mut(side);
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
fn drag_in_all(state: &mut DoublesState, rng: &mut Gen5Rng) {
    for side in [Side::One, Side::Two] {
        for slot in 0..2 {
            if !state.side(side).active[slot].drag_out {
                continue;
            }
            state.side_mut(side).active[slot].drag_out = false;
            if !state.side(side).slot_pokemon(slot).map(|p| p.hp > 0).unwrap_or(false) {
                continue;
            }
            let options = possible_switches(state, side);
            if options.is_empty() {
                continue;
            }
            let pick = options[rng.label("drag in").random(options.len() as u32) as usize];
            switch_in(state, side, slot, pick);
            let _ = sorted_actives(state, rng, "speed tie (switch-in)");
            if apply_on_switch_abilities(state, side, slot, rng) {
                let _ = sorted_actives(state, rng, "speed tie (switch-in)");
            }
        }
    }
}

/// PS re-sort of the remaining queue after an action (gen 8+ `runAction`
/// tail): fresh speeds, `getTarget` for every queued move, then `speedSort`.
fn resort_queue(state: &mut DoublesState, rng: &mut Gen5Rng, queue: &mut [Action]) {
    if queue.is_empty() {
        return;
    }
    update_speed(state);
    for a in queue.iter_mut() {
        if state.side(a.side).active[a.slot].pokemon_index == Some(a.pokemon_index) {
            if let Some(sp) = effective_speed(state, a.side, a.slot) {
                a.speed = sp;
            }
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
fn execute_action(
    state: &mut DoublesState,
    rng: &mut Gen5Rng,
    action: &Action,
    move_order: &mut Vec<String>,
    will_act: bool,
) {
    let before = state.side(action.side).active[action.slot].charging;
    let was_recharging = state.side(action.side).active[action.slot].recharging;
    let order_len = move_order.len();
    state.side_mut(action.side).active[action.slot].move_result_this_turn = None;
    execute_action_inner(state, rng, action, move_order, will_act);
    // PS `moveThisTurnResult`: a BeforeMove abort counts as a failure
    // (`runMove`), except the recharge turn; otherwise a miss / immunity /
    // failed effect sets `false` where it happens, and the rest succeeded.
    {
        let a = &mut state.side_mut(action.side).active[action.slot];
        if a.pokemon_index.is_some() {
            if move_order.len() == order_len && !was_recharging {
                a.move_result_this_turn = Some(false);
            } else if a.move_result_this_turn.is_none() {
                a.move_result_this_turn = Some(true);
            }
        }
    }
    // PS `runMove`: a failed BeforeMove runs `MoveAborted`, which removes
    // `twoturnmove` (data/conditions.ts). The charge was not consumed.
    let a = &mut state.side_mut(action.side).active[action.slot];
    if let (Some(b), Some(c)) = (before, a.charging) {
        if b.0 == c.0 && b.1 == c.1 {
            a.charge_aborted = true;
        }
    }
}

fn execute_action_inner(
    state: &mut DoublesState,
    rng: &mut Gen5Rng,
    action: &Action,
    move_order: &mut Vec<String>,
    will_act: bool,
) {
    let side = action.side;
    let slot = action.slot;
    let idx = action.pokemon_index;
    let name = state.side(side).species_names[idx as usize].clone();
    let mut push_order = || {
        if std::env::var("PE_DEBUG_DMG").is_ok() {
            eprintln!("PUSHORDER {} {}", name, action.choice.move_id);
        }
        move_order.push(format!(
            "{}{}:{}:{}",
            side_tag(side),
            slot,
            name,
            action.choice.move_id
        ))
    };

    // Deduct PP (pokemon-showdown deducts before the move runs).
    {
        let s = state.side_mut(side);
        if let Some(p) = s.active[slot].pokemon_index {
            let mv = &mut s.team_member_mut(p).moves[&action.choice.move_index];
            if mv.pp > 0 {
                mv.pp -= 1;
            }
        }
    }

    // The attacker must still be on the field.
    if state.side(side).active[slot].pokemon_index.is_none() {
        return;
    }

    state.unresolved_faints.clear();

    // PS `runMove` resolves the target before the BeforeMove checks.
    let resolved = ps_get_target(state, rng, action);
    if let Some((ts, tsl)) = resolved {
        state.side_mut(side).active[slot].last_target_loc =
            if ts == side { -(tsl as i8 + 1) } else { tsl as i8 + 1 };
    }

    // A pokemon that must recharge (Hyper Beam / Solar Beam) cannot act.
    // No PRNG draw (PS `mustrecharge.onBeforeMove` returns false).
    if state.side(side).active[slot].recharging {
        state.side_mut(side).active[slot].recharging = false;
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
            // slp: time--; wake and act at 0. No draw.
            PokemonStatus::SLEEP => {
                p.sleep_turns -= 1;
                if p.sleep_turns <= 0 {
                    p.status = PokemonStatus::NONE;
                } else if move_id != Choices::SNORE {
                    // Snore is `sleepUsable`.
                    // TODO(PS parity): Sleep Talk (calls a random move).
                    // PS logs `|cant|`, not `|move|`: not in the move order.
                    return;
                }
            }
            // frz: a `defrost` move acts (and thaws); else time--, thaw at 0
            // or with THAW_CHANCE.
            PokemonStatus::FREEZE => {
                if crate::engine::mechanics::THAWS_USER.contains(&move_id) {
                    p.status = PokemonStatus::NONE;
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
    if state.side(side).active[slot].flinched {
        return;
    }

    // Confusion (data/conditions.ts `confusion.onBeforeMove`): time--, ends
    // at 0; otherwise a 33/100 chance to hit itself (40 BP typeless, own Atk
    // vs own Def, damage roll) instead of moving. The singles engine's
    // HIT_SELF_IN_CONFUSION_CHANCE is 1/3; PS rolls 33/100.
    if state.side(side).active[slot].confusion > 0 {
        let a = &mut state.side_mut(side).active[slot];
        a.confusion -= 1;
        if a.confusion > 0 && rng.label("status before move").random_chance(33, 100) {
            let dmg = confusion_damage(state, rng, side, slot, idx);
            apply_damage(state, side, slot, dmg);
            return;
        }
    }

    // Attract (`attract.onBeforeMove`, priority 2): 1/2 chance to be
    // immobilized. It is dropped first when the source is no longer active
    // (`onUpdate`).
    if let Some((ss, si)) = state.side(side).active[slot].attract {
        let src_active = (0..2).any(|sl| {
            state.side(ss).active[sl].pokemon_index == Some(si) && state.side(ss).team_member(si).hp > 0
        });
        if !src_active {
            state.side_mut(side).active[slot].attract = None;
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

    // Focus Punch `beforeMoveCallback` (after the BeforeMove checks): it
    // fails with `|cant|` if the user was hit by a damaging move this turn.
    if action.choice.move_id == Choices::FOCUSPUNCH && state.side(side).active[slot].lost_focus {
        return;
    }

    // PS `onModifyType` / `onModifyMove` for moves whose type and power
    // depend on the field (shared with the singles engine).
    let mut modified = action.choice.clone();
    match modified.move_id {
        Choices::TERRAINPULSE => {
            let grounded = state.side(side).team_member(idx).is_grounded();
            if let Some(t) = crate::engine::mechanics::terrain_pulse_type(state.terrain.terrain_type, grounded) {
                modified.move_type = t;
                modified.base_power *= 2.0;
            }
        }
        Choices::WEATHERBALL => {
            let mega_sol = state.side(side).team_member(idx).ability == Abilities::MEGASOL;
            if let Some(t) = crate::engine::mechanics::weather_ball_type(state.weather.weather_type, mega_sol) {
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
    // Liquid Voice (`onModifyType`, priority -1): sound moves become Water,
    // with no power boost.
    if state.side(side).team_member(idx).ability == Abilities::LIQUIDVOICE && modified.flags.sound {
        modified.move_type = PokemonType::WATER;
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
        _ => choice.weather_id,
    };
    let is_field = weather_id != 0
        || choice.terrain_id != 0
        || choice.pseudo_weather_id != 0
        || (choice.side_condition.is_some() && choice.category == MoveCategory::Status);

    // The slot an `Ally` effect hits: the one adjacent ally.
    let ally_slot = 1 - slot;

    let slot_alive = |st: &DoublesState, ts: Side, s: usize| -> bool {
        st.side(ts)
            .slot_pokemon(s)
            .map(|p| p.hp > 0)
            .unwrap_or(false)
    };

    let mut target_slot = 0usize;
    let mut has_target = true;
    if !is_self && !is_ally && !is_field {
        // The target `ps_get_target` resolved (with PS's retarget draw).
        // TODO(PS parity): spread moves (allAdjacentFoes / allAdjacent) hit
        // every foe; the engine hits one.
        match resolved {
            Some((ts, sl)) if slot_alive(state, ts, sl) => {
                target_side = ts;
                target_slot = sl;
            }
            _ => {
                target_slot = action.target.min(1);
                if !slot_alive(state, target_side, target_slot) {
                    has_target = false;
                }
            }
        }
        if !has_target {
            push_order();
            return;
        }
    }

    // Two-turn charge moves (Dig, Meteor Beam, ...): mirrors PS
    // `onTryMove` + `addVolatile('twoturnmove')` (data/moves.ts). The first
    // use(s) only charge: no accuracy roll, no damage, no secondary, but the
    // charge-turn self boost applies (Meteor Beam +1 SpA, Skull Bash +1 Def).
    // The move fires when the same move is chosen again after the last
    // charge turn (PS locks the move via `twoturnmove.onLockMove`).
    {
        let needs_charge = match choice.charge_setup {
            ChargeSetup::None => false,
            ChargeSetup::Always { .. } => true,
            ChargeSetup::NotSun { .. } => state.weather.weather_type != Weather::SUN,
            ChargeSetup::OnlyRain { .. } => state.weather.weather_type == Weather::RAIN,
            ChargeSetup::OnlySnow { .. } => matches!(
                state.weather.weather_type,
                Weather::HAIL | Weather::SNOW
            ),
        };
        let charged = state.side_mut(side).active[slot].charging.take();
        match charged {
            Some((m, turns_left)) if m == choice.move_id && turns_left > 0 => {
                // Still charging (Geomancy's second charge turn).
                state.side_mut(side).active[slot].charging = Some((m, turns_left - 1));
                push_order();
                return;
            }
            Some((_, _)) => {
                // turns_left == 0 (fire this turn) or a different move was
                // chosen while charging (charge wasted; PS charge volatiles
                // expire at end of turn).
            }
            None if needs_charge => {
                // Start charging.
                if let Some(b) = &choice.charge_boosts {
                    apply_boosts(state, side, idx, b, false);
                }
                let turns = match choice.charge_setup {
                    ChargeSetup::Always { turns }
                    | ChargeSetup::NotSun { turns }
                    | ChargeSetup::OnlyRain { turns }
                    | ChargeSetup::OnlySnow { turns } => turns,
                    ChargeSetup::None => 0,
                };
                state.side_mut(side).active[slot].charging =
                    Some((choice.move_id, turns.saturating_sub(1)));
                state.side_mut(side).active[slot].charge_target = action.target;
                push_order();
                return;
            }
            None => {}
        }
    }

    // Stockpile family (PS `onTry` / `stockpile` condition). No PRNG draws.
    match choice.move_id {
        Choices::STOCKPILE => {
            let a = &state.side(side).active[slot];
            if a.stockpile < 3 {
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
                let a = &mut state.side_mut(side).active[slot];
                a.stockpile += 1;
                a.stockpile_def += after.def - before.def;
                a.stockpile_spd += after.spd - before.spd;
            }
            push_order();
            return;
        }
        // Snore `onTry`: fails unless the user is asleep (or has Comatose).
        Choices::SNORE
            if state.side(side).team_member(idx).status != PokemonStatus::SLEEP
                && state.side(side).team_member(idx).ability != Abilities::COMATOSE =>
        {
            push_order();
            return;
        }
        Choices::SPITUP | Choices::SWALLOW if state.side(side).active[slot].stockpile == 0 => {
            push_order();
            return;
        }
        Choices::SWALLOW => {
            let layers = state.side(side).active[slot].stockpile;
            let s = state.side_mut(side);
            let p = s.team_member_mut(idx);
            let heal = match layers {
                1 => p.maxhp / 4,
                2 => p.maxhp / 2,
                _ => p.maxhp,
            };
            p.hp = (p.hp + heal).min(p.maxhp);
            end_stockpile(state, side, slot);
            push_order();
            return;
        }
        _ => {}
    }

    // Steel Roller (gen 9): fails when no terrain is active (PS `onTry`).
    // No PRNG draw; PP is already spent.
    if choice.requires_terrain
        && state.terrain.terrain_type == crate::engine::state::Terrain::NONE
    {
        push_order();
        return;
    }

    // Ally Switch `onPrepareHit`: `addVolatile('allyswitch')`; a restart rolls
    // `randomChance(1, counter)` and fails (dropping the volatile) on a miss.
    if choice.move_id == Choices::ALLYSWITCH {
        let a = &mut state.side_mut(side).active[slot];
        match a.ally_switch {
            None => a.ally_switch = Some((3, 2)),
            Some((c, _)) => {
                if !rng.label("other: onRestart").random_chance(1, c.max(1) as u32) {
                    state.side_mut(side).active[slot].ally_switch = None;
                    push_order();
                    return;
                }
                state.side_mut(side).active[slot].ally_switch = Some((if c < 729 { c * 3 } else { c }, 2));
            }
        }
    }

    // Stalling moves (Protect / Detect / Endure) `onPrepareHit`:
    // `!!this.queue.willAct() && this.runEvent('StallMove', pokemon)`; the
    // `stall` volatile rolls `randomChance(1, counter)` and is dropped on a
    // failure (data/conditions.ts stall.onStallMove).
    if matches!(
        choice.volatile_status.as_ref().map(|v| v.volatile_status),
        Some(PokemonVolatileStatus::PROTECT | PokemonVolatileStatus::ENDURE)
    ) {
        let mut ok = will_act;
        if ok {
            if let Some((counter, _)) = state.side(side).active[slot].stall {
                ok = rng.label("other: onStallMove").random_chance(1, counter.max(1) as u32);
                if !ok {
                    state.side_mut(side).active[slot].stall = None;
                }
            }
        }
        if !ok {
            push_order();
            return;
        }
    }

    // Libero / Protean (`onPrepareHit`, before the hit steps): once per
    // switch-in, become the move's type. PS data/abilities.ts libero.
    // TODO(PS parity): Protean's champions override, Tera, Terastallized.
    if matches!(
        state.side(side).team_member(idx).ability,
        Abilities::LIBERO | Abilities::PROTEAN
    ) && state.side(side).active[slot].type_change.is_none()
        && choice.move_type != PokemonType::TYPELESS
        && choice.move_id != Choices::STRUGGLE
    {
        let pm = state.side_mut(side).team_member_mut(idx);
        if pm.types != (choice.move_type, PokemonType::TYPELESS) {
            let orig = pm.types;
            pm.types = (choice.move_type, PokemonType::TYPELESS);
            state.side_mut(side).active[slot].type_change = Some(orig);
        }
    }

    // Counter / Mirror Coat `onTry`: fail unless the volatile recorded a hit.
    if matches!(choice.move_id, Choices::COUNTER | Choices::MIRRORCOAT) {
        let a = &state.side(side).active[slot];
        let rec = if choice.move_id == Choices::COUNTER { a.counter_vol } else { a.mirror_coat_vol };
        if !matches!(rec, Some((Some(_), _))) {
            state.side_mut(side).active[slot].move_result_this_turn = Some(false);
            push_order();
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
            state.side_mut(side).active[slot].move_result_this_turn = Some(false);
            push_order();
            return;
        }
    }

    // No Retreat `onTry`: fails if the volatile is already up (no draw).
    if choice.move_id == Choices::NORETREAT && state.side(side).active[slot].no_retreat {
        state.side_mut(side).active[slot].move_result_this_turn = Some(false);
        push_order();
        return;
    }

    // Belch `onTry`: needs `ateBerry` (berries are never eaten here).
    // TODO(PS parity): `ateBerry` once berries are modeled.
    if choice.move_id == Choices::BELCH {
        push_order();
        return;
    }

    // Fling `onPrepareHit`: fails without an item (no draw).
    // TODO(PS parity): item base power table, item removal, item effects.
    if choice.move_id == Choices::FLING
        && state.side(side).team_member(idx).item == crate::engine::items::Items::NONE
    {
        push_order();
        return;
    }

    // Spread moves (allAdjacentFoes / allAdjacent): PS `getMoveTargets` lists
    // the (alive) allies first, then the foes. Each target is filtered like
    // the single-target pre-hit checks (no draws), then rolls its own accuracy.
    // TODO(PS parity): invulnerability exceptions (Earthquake vs Dig, ...).
    let move_ty = crate::doubles::move_targets::ps_move_target(&choice.move_id.to_string().to_lowercase());
    let is_spread = matches!(move_ty, "allAdjacentFoes" | "allAdjacent") && !is_self && !is_field;
    let mut spread_list: Vec<(Side, usize)> = Vec::new();
    // PS `trySpreadMoveHit`: `spreadHit` is set from the original target list.
    let mut spread_count = 0usize;
    if is_spread {
        let mut cands: Vec<(Side, usize)> = Vec::new();
        if move_ty == "allAdjacent" && slot_alive(state, side, 1 - slot) {
            cands.push((side, 1 - slot));
        }
        for sl in 0..2 {
            if slot_alive(state, side.other(), sl) {
                cands.push((side.other(), sl));
            }
        }
        spread_count = cands.len();
        let damaging = matches!(choice.category, MoveCategory::Physical | MoveCategory::Special);
        for (ts, tsl) in cands {
            if state.side(ts).active[tsl].protected && choice.priority <= 0 {
                continue;
            }
            let d = state.side(ts).slot_pokemon(tsl).unwrap();
            if let Some((m, _)) = state.side(ts).active[tsl].charging {
                if matches!(
                    m,
                    Choices::DIG | Choices::DIVE | Choices::FLY | Choices::BOUNCE | Choices::PHANTOMFORCE | Choices::SHADOWFORCE | Choices::SKYDROP
                ) {
                    continue;
                }
            }
            if matches!(
                (d.ability, choice.move_type),
                (Abilities::LIGHTNINGROD, PokemonType::ELECTRIC) | (Abilities::STORMDRAIN, PokemonType::WATER)
            ) && (ts, tsl) != (side, slot)
            {
                if let Some(tidx) = state.side(ts).active[tsl].pokemon_index {
                    apply_boosts(
                        state,
                        ts,
                        tidx,
                        &StatBoosts { special_attack: 1, ..Default::default() },
                        false,
                    );
                }
                continue;
            }
            if (d.ability == Abilities::SOUNDPROOF && choice.flags.sound)
                || (d.ability == Abilities::BULLETPROOF && choice.flags.bullet)
            {
                continue;
            }
            if damaging && crate::doubles::damage::type_mod(&choice.move_type, d).is_none() {
                continue;
            }
            spread_list.push((ts, tsl));
        }
        if spread_list.is_empty() {
            push_order();
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
    if std::env::var("PE_DEBUG_DMG").is_ok() {
        eprintln!(
            "INVULNCHK {} -> slot {} charging={:?}",
            choice.move_id, target_slot, state.side(target_side).active[target_slot].charging
        );
    }
    let toxic_hits = crate::engine::mechanics::toxic_always_hits(choice.move_id, state.side(side).team_member(idx));
    if !is_self && !is_ally && !is_field && !toxic_hits {
        if let Some((charging, _)) = state.side(target_side).active[target_slot].charging {
            let hits_anyway = match charging {
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
                Choices::PHANTOMFORCE | Choices::SHADOWFORCE => false,
                _ => true, // not a semi-invulnerable charge (Meteor Beam, ...)
            };
            let no_guard = state.side(side).team_member(idx).ability == Abilities::NOGUARD
                || state
                    .side(target_side)
                    .slot_pokemon(target_slot)
                    .map(|p| p.ability == Abilities::NOGUARD)
                    .unwrap_or(false);
            if !hits_anyway && !no_guard && choice.move_id != Choices::HELPINGHAND {
                push_order();
                return;
            }
        }
    }

    // Protect: incoming non-priority moves fail before the accuracy roll.
    // (Charge turns happen before this check: Protect does not block a
    // charging Dig, only its hitting turn.)
    if !is_self
        && !is_field
        && state.side(target_side).active[target_slot].protected
        && choice.priority <= 0
    {
        push_order();
        return;
    }

    // Type immunity is checked before the accuracy roll (PS moveSteps order:
    // Invulnerability -> TryHit -> TypeImmunity -> Accuracy). An immune
    // target consumes no PRNG draws at all.
    // TODO(PS parity): move-specific immunities (PS `hitStepTryImmunity`).
    let speeds = (stat_speed(state, side, slot), stat_speed(state, target_side, target_slot));
    let effective_power = if dynamic_base_power(
        choice,
        state.side(side).team_member(idx),
        state.side(target_side).slot_pokemon(target_slot),
        speeds,
    ) > 0.0
    {
        1.0
    } else {
        choice.base_power
    };
    // Uproar `onTryHit`: cures sleep on every active pokemon.
    if choice.move_id == Choices::UPROAR {
        for sd in [Side::One, Side::Two] {
            for sl in 0..2 {
                if let Some(ti) = state.side(sd).active[sl].pokemon_index {
                    let pm = state.side_mut(sd).team_member_mut(ti);
                    if pm.status == PokemonStatus::SLEEP {
                        pm.status = PokemonStatus::NONE;
                    }
                }
            }
        }
    }

    // Brick Break / Psychic Fangs / Raging Bull `onTryHit`: shatter the
    // target side's screens before the hit (even through a Substitute).
    // TODO(PS parity): Raging Bull's type-dependent variants.
    if matches!(choice.move_id, Choices::BRICKBREAK | Choices::PSYCHICFANGS | Choices::RAGINGBULL)
        && !is_self
        && !is_field
    {
        let sc = &mut state.side_mut(target_side).side_conditions;
        sc.reflect = 0;
        sc.light_screen = 0;
        sc.aurora_veil = 0;
    }

    // Soundproof / Bulletproof `onTryHit`: immune to sound / ball moves (no draw).
    if !is_self && !is_field && (side, slot) != (target_side, target_slot) {
        let immune = state
            .side(target_side)
            .slot_pokemon(target_slot)
            .map(|d| {
                (d.ability == Abilities::SOUNDPROOF && choice.flags.sound)
                    || (d.ability == Abilities::BULLETPROOF && choice.flags.bullet)
            })
            .unwrap_or(false);
        if immune && !is_spread {
            state.side_mut(side).active[slot].move_result_this_turn = Some(false);
            push_order();
            return;
        }
    }

    // Ability absorbs (PS `onTryHit`, `hitStepTryHitEvent`: after Protect,
    // before type immunity and accuracy; no draws). Lightning Rod / Storm
    // Drain: +1 SpA instead of the hit.
    if !is_self && !is_field && (side, slot) != (target_side, target_slot) {
        let boost = state.side(target_side).slot_pokemon(target_slot).and_then(|d| {
            match (d.ability, choice.move_type) {
                (Abilities::LIGHTNINGROD, PokemonType::ELECTRIC)
                | (Abilities::STORMDRAIN, PokemonType::WATER) => d.hp > 0,
                _ => false,
            }
            .then_some(())
        });
        if boost.is_some() {
            if let Some(tidx) = state.side(target_side).active[target_slot].pokemon_index {
                apply_boosts(
                    state,
                    target_side,
                    tidx,
                    &StatBoosts { special_attack: 1, ..Default::default() },
                    false,
                );
            }
            push_order();
            return;
        }
    }

    if matches!(choice.category, MoveCategory::Physical | MoveCategory::Special)
        && (effective_power > 0.0 || matches!(choice.move_id, Choices::COUNTER | Choices::MIRRORCOAT))
        && !is_self
        && !is_field
    {
        if let Some(def) = state.side(target_side).slot_pokemon(target_slot) {
            if crate::doubles::damage::type_mod(&choice.move_type, def).is_none() {
                state.side_mut(side).active[slot].move_result_this_turn = Some(false);
                push_order();
                return;
            }
        }
    }

    // Prankster-boosted Status moves: Dark foes are immune (PS
    // sim/battle-actions.ts `hitStepTryImmunity`, gen >= 7). No draw.
    if choice.category == MoveCategory::Status
        && !is_self
        && !is_ally
        && !is_field
        && state.side(side).slot_pokemon(slot).map(|p| p.ability) == Some(Abilities::PRANKSTER)
        && state
            .side(target_side)
            .slot_pokemon(target_slot)
            .map(|d| d.types.0 == PokemonType::DARK || d.types.1 == PokemonType::DARK)
            .unwrap_or(false)
    {
        push_order();
        return;
    }

    // Attract: PS `onTryImmunity` succeeds only for male -> female or
    // female -> male; Oblivious blocks it in TryHit. Both happen before the
    // accuracy roll (no draw).
    // TODO(PS parity): Aroma Veil, already-infatuated targets.
    if choice.move_id == Choices::ATTRACT && !is_self && !is_ally && !is_field {
        let att_gender = state.side(side).genders[idx as usize].clone();
        let def = state.side(target_side).active[target_slot].pokemon_index;
        let def_gender = def.and_then(|di| state.side(target_side).genders[di as usize].clone());
        let oblivious = def
            .map(|di| state.side(target_side).team_member(di).ability == Abilities::OBLIVIOUS)
            .unwrap_or(false);
        let opposite = matches!(
            (att_gender.as_deref(), def_gender.as_deref()),
            (Some("M"), Some("F")) | (Some("F"), Some("M"))
        );
        if !opposite || oblivious {
            push_order();
            return;
        }
    }

    // Accuracy roll (drawn even at 100%; skipped for always-hit / 0.0).
    if std::env::var("PE_DEBUG_DMG").is_ok() {
        eprintln!("REACHED_ACC {} {} target_slot={} has_target={}", name, choice.move_id, target_slot, has_target);
    }
    if is_spread {
        // PS `hitStepAccuracy`: one roll per target, in target order.
        if !choice.always_hit && !toxic_hits && choice.accuracy > 0.0 {
            let ab = state.side(side).slot_boosts(slot).map(|b| b.acc).unwrap_or(0);
            let st = &*state;
            spread_list.retain(|&(ts, tsl)| {
                let eb = st.side(ts).slot_boosts(tsl).map(|b| b.eva).unwrap_or(0);
                roll_accuracy_boosted(choice.accuracy, ab, eb, rng)
            });
        }
        if spread_list.is_empty() {
            push_order();
            return;
        }
        target_side = spread_list[0].0;
        target_slot = spread_list[0].1;
    } else if !choice.always_hit && !toxic_hits && choice.accuracy > 0.0 {
        let ab = state.side(side).slot_boosts(slot).map(|b| b.acc).unwrap_or(0);
        let eb = state.side(target_side).slot_boosts(target_slot).map(|b| b.eva).unwrap_or(0);
        if !roll_accuracy_boosted(choice.accuracy, ab, eb, rng) {
            state.side_mut(side).active[slot].move_result_this_turn = Some(false);
            push_order();
            return;
        }
    }

    // Substitute `onTryPrimaryHit` (after the accuracy step): a Status move
    // aimed at another pokemon's substitute fails (getDamage -> undefined)
    // unless it is a sound move or the user has Infiltrator.
    if choice.category == MoveCategory::Status
        && !is_self
        && !is_field
        && (target_side, target_slot) != (side, slot)
        && state.side(target_side).active[target_slot].substitute_health > 0
        && !choice.flags.sound
        && state.side(side).team_member(idx).ability != Abilities::INFILTRATOR
    {
        push_order();
        return;
    }

    // Multi-hit count.
    let hit_count: u32 = match choice.multihit {
        0 => 1,
        20 => {
            // 35-35-15-15 out of 100 for 2-3-4-5 hits (PS sample of 20).
            const TABLE: [u32; 20] = [2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5];
            TABLE[rng.label("multihit").random(20) as usize]
        }
        n => n as u32,
    };

    let is_attack = matches!(choice.category, MoveCategory::Physical | MoveCategory::Special);
    let mut total_damage: i16 = 0;
    // A hit that did nothing ends PS's hit loop with no `eachEvent`.
    // TODO(PS parity): other failing status effects (boosts at +6, ...).
    let mut move_failed = false;
    let mut hits_done = 0u32;

    for _ in 0..hit_count {
      let hit_targets: Vec<(Side, usize)> =
          if is_spread { spread_list.clone() } else { vec![(target_side, target_slot)] };
      // PS `getSpreadDamage`: every target's crit + damage roll comes before
      // any per-target effect.
      let pre_damage: Vec<(i16, bool)> = if is_spread && is_attack {
          let sp = spread_count > 1;
          hit_targets
              .iter()
              .map(|&(ts, tsl)| compute_hit_damage(state, rng, side, slot, idx, ts, tsl, choice, ate_boosted, sp))
              .collect()
      } else {
          Vec::new()
      };
      for (ti, &(hts, htsl)) in hit_targets.iter().enumerate() {
        target_side = hts;
        target_slot = htsl;
        // Damage this hit dealt (PS `damagedTargets`), for DamagingHit.
        let mut hit_damage: Option<i16> = None;
        if std::env::var("PE_DEBUG_DMG").is_ok() {
            eprintln!(
                "HITLOOP {} is_attack={} is_self={} is_field={} has_target={} target_alive={}",
                choice.move_id,
                is_attack,
                is_self,
                is_field,
                has_target,
                slot_alive(state, target_side, target_slot)
            );
        }
        if !is_self && !is_field && !slot_alive(state, target_side, target_slot) {
            if is_spread {
                continue;
            }
            break;
        }

        if is_attack && !is_self && !is_field && has_target {
            let (damage, was_crit) = pre_damage
                .get(ti)
                .copied()
                .unwrap_or_else(|| compute_hit_damage(state, rng, side, slot, idx, target_side, target_slot, choice, ate_boosted, false));
            // Substitute `onTryPrimaryHit`: a damaging move hits the substitute
            // instead (sound moves and Infiltrator bypass it). No further
            // effects on that target (`HIT_SUBSTITUTE`).
            // TODO(PS parity): AfterSubDamage events.
            let sub_hp = state.side(target_side).active[target_slot].substitute_health;
            if sub_hp > 0
                && (target_side, target_slot) != (side, slot)
                && !choice.flags.sound
                && state.side(side).team_member(idx).ability != Abilities::INFILTRATOR
            {
                let d = damage.min(sub_hp);
                state.side_mut(target_side).active[target_slot].substitute_health = sub_hp - d;
                total_damage = total_damage.saturating_add(d);
                if let Some(frac) = choice.drain {
                    let heal = ((d as f32) * frac).ceil() as i16;
                    if heal > 0 {
                        let s = state.side_mut(side);
                        let pm = s.team_member_mut(idx);
                        pm.hp = (pm.hp + heal).min(pm.maxhp);
                    }
                }
                // `secondaries` still rolls once per entry (the target is
                // `null`, not `false`), but nothing is applied.
                let rolls = match choice.move_id {
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
                };
                for _ in 0..rolls {
                    let _ = rng.label("secondary").random(100);
                }
                continue;
            }
            // Endure `onDamage` (priority -10): a move that would KO leaves 1 HP.
            let damage = match state.side(target_side).slot_pokemon(target_slot) {
                Some(d) if state.side(target_side).active[target_slot].endure && damage >= d.hp && d.hp > 0 => d.hp - 1,
                _ => damage,
            };
            // PS damage dealt is capped by the target's remaining HP (drain,
            // recoil and Counter use the damage actually dealt).
            let damage = match state.side(target_side).slot_pokemon(target_slot) {
                Some(d) => damage.min(d.hp.max(0)),
                None => damage,
            };
            // Anger Point `onHit`: a critical hit maxes Attack.
            if was_crit {
                if let Some(didx) = state.side(target_side).active[target_slot].pokemon_index {
                    if state.side(target_side).team_member(didx).ability == Abilities::ANGERPOINT
                        && state.side(target_side).team_member(didx).hp > damage
                    {
                        state.side_mut(target_side).team_boosts[didx as usize].atk = 6;
                    }
                }
            }
            if damage > 0 && apply_damage(state, target_side, target_slot, damage) {
                // Target fainted. Nothing else to do here; remaining hits
                // will bail out via the slot_alive check above.
            }
            total_damage = total_damage.saturating_add(damage);
            hit_damage = Some(damage);
            // Counter / Mirror Coat condition `onDamagingHit`.
            if target_side != side && damage > 0 {
                let t = &mut state.side_mut(target_side).active[target_slot];
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
                if heal > 0 {
                    let s = state.side_mut(side);
                    if let Some(p) = s.active[slot].pokemon_index {
                        let pm = s.team_member_mut(p);
                        pm.hp = (pm.hp + heal).min(pm.maxhp);
                    }
                }
            }

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
                _ => {}
            }
            let b = &b;
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
            if move_ty == "allies" {
                // Howl & co. (`target: 'allies'`): the user and its ally, in
                // slot order.
                for sl in 0..2 {
                    if slot_alive(state, side, sl) {
                        if let Some(tidx) = state.side(side).active[sl].pokemon_index {
                            apply_boosts(state, side, tidx, &b.boosts, false);
                        }
                    }
                }
            } else if let Some(tidx) = state.side(ts).active[tslot].pokemon_index {
                apply_boosts(state, ts, tidx, &b.boosts, ts != side);
            }
        }

        // Healing (Moonlight, Synthesis, etc.) — no draw.
        if let Some(h) = &choice.heal {
            let (ts, tslot) = if h.target == MoveTarget::User || is_self {
                (side, slot)
            } else if h.target == MoveTarget::Ally {
                (side, ally_slot)
            } else {
                (target_side, target_slot)
            };
            let weather = state.weather.weather_type;
            if let Some(tidx) = state.side(ts).active[tslot].pokemon_index {
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
                    // Memento / Healing Wish style self-KO (`selfdestruct`).
                    let hp = pm.hp;
                    let _ = pm;
                    apply_damage(state, ts, tslot, hp);
                } else {
                    // PS `heal` returns false (the move fails) at full HP.
                    if pm.hp >= pm.maxhp {
                        move_failed = true;
                    }
                    pm.hp = (pm.hp + amount).min(pm.maxhp);
                }
            }
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
            if let Some(tidx) = state.side(ts).active[tslot].pokemon_index {
                // Already in a status, or immune: the move fails.
                if status_immune(state, ts, tslot, st.status, ts != side, side, slot) {
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
                PokemonVolatileStatus::PROTECT => {
                    state.side_mut(ts).active[tslot].protected = true;

                    // onHit: `addVolatile('stall')` (start: counter 3; restart: x3).
                    let a = &mut state.side_mut(ts).active[tslot];
                    a.stall = match a.stall {
                        None => Some((3, 2)),
                        Some((c, _)) => Some((if c < 729 { c * 3 } else { c }, 2)),
                    };
                }
                PokemonVolatileStatus::CONFUSION => try_confuse(state, rng, ts, tslot),
                PokemonVolatileStatus::SUBSTITUTE => {
                    // PS substitute `onTryHit` (fails if already up or hp <=
                    // maxhp/4), `onHit` (directDamage maxhp/4), `onStart`
                    // (sub hp = floor(maxhp/4); ends partial trapping).
                    let (hp, maxhp) = state
                        .side(ts)
                        .slot_pokemon(tslot)
                        .map(|p| (p.hp, p.maxhp))
                        .unwrap_or((0, 1));
                    if state.side(ts).active[tslot].substitute_health > 0
                        || (hp as i32) * 4 <= maxhp as i32
                        || maxhp == 1
                    {
                        move_failed = true;
                    } else if let Some(tidx) = state.side(ts).active[tslot].pokemon_index {
                        let cost = (maxhp / 4).max(1);
                        state.side_mut(ts).team_member_mut(tidx).hp -= cost;
                        let a = &mut state.side_mut(ts).active[tslot];
                        a.substitute_health = maxhp / 4;
                        a.partial_trap = None;
                    }
                }
                PokemonVolatileStatus::FOCUSENERGY => {
                    if state.side(ts).active[tslot].focus_energy {
                        move_failed = true;
                    }
                    state.side_mut(ts).active[tslot].focus_energy = true;
                }
                PokemonVolatileStatus::AQUARING => {
                    // `addVolatile` fails if already present.
                    if state.side(ts).active[tslot].aqua_ring {
                        move_failed = true;
                    }
                    state.side_mut(ts).active[tslot].aqua_ring = true;
                }
                PokemonVolatileStatus::ENDURE => {
                    state.side_mut(ts).active[tslot].endure = true;

                    // onHit: `addVolatile('stall')` (start: counter 3; restart: x3).
                    let a = &mut state.side_mut(ts).active[tslot];
                    a.stall = match a.stall {
                        None => Some((3, 2)),
                        Some((c, _)) => Some((if c < 729 { c * 3 } else { c }, 2)),
                    };
                }
                PokemonVolatileStatus::LOCKEDMOVE => {
                    // PS lockedmove `onStart`: trueDuration = random(2, 4);
                    // `onRestart`: duration back to 2 while trueDuration >= 2.
                    match state.side(ts).active[tslot].locked_move {
                        None => {
                            let t = rng.label("volatile duration").random_range(2, 4) as u8;
                            state.side_mut(ts).active[tslot].locked_move = Some((2, t));
                        }
                        Some((d, t)) => {
                            if t >= 2 {
                                state.side_mut(ts).active[tslot].locked_move = Some((2, t));
                            } else {
                                state.side_mut(ts).active[tslot].locked_move = Some((d, t));
                            }
                        }
                    }
                }
                PokemonVolatileStatus::YAWN => {
                    // PS yawn `onTryHit`: fails on any status or sleep
                    // immunity; `addVolatile` fails if already yawning.
                    let has_status = state
                        .side(ts)
                        .slot_pokemon(tslot)
                        .map(|p| p.status != PokemonStatus::NONE)
                        .unwrap_or(true);
                    if has_status
                        || state.side(ts).active[tslot].yawn > 0
                        || status_immune(state, ts, tslot, PokemonStatus::SLEEP, ts != side, side, slot)
                    {
                        move_failed = true;
                    } else {
                        let a = &mut state.side_mut(ts).active[tslot];
                        a.yawn = 2;
                        a.yawn_source = Some((side, idx));
                    }
                }
                PokemonVolatileStatus::PARTIALLYTRAPPED => {
                    try_partial_trap(state, rng, ts, tslot, side, idx)
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
            }
        }

        // Psych Up (`onHit`): copy the target's stat stages onto the user.
        // TODO(PS parity): also copies crit-stage volatiles (Focus Energy,
        // Dragon Cheer).
        if choice.move_id == Choices::PSYCHUP {
            if let Some(tidx) = state.side(target_side).active[target_slot].pokemon_index {
                let copied = state.side(target_side).team_boosts[tidx as usize].clone();
                state.side_mut(side).team_boosts[idx as usize] = copied;
            }
        }

        // Uproar self volatile (`addVolatile('uproar')`, duration 3).
        if choice.move_id == Choices::UPROAR && ti == 0 && state.side(side).active[slot].uproar == 0 {
            state.side_mut(side).active[slot].uproar = 3;
        }

        if choice.move_id == Choices::NORETREAT && ti == 0 {
            state.side_mut(side).active[slot].no_retreat = true;
        }

        // Attract: `addVolatile('attract')` (fails if already infatuated).
        if choice.move_id == Choices::ATTRACT && ti == 0 {
            if state.side(target_side).active[target_slot].attract.is_some() {
                move_failed = true;
            } else {
                state.side_mut(target_side).active[target_slot].attract = Some((side, idx));
            }
        }

        // Rest `onHit`: `setStatus('slp')` (draws the sleep duration), then the
        // sleep counter is forced to 3 and the user heals fully.
        if choice.move_id == Choices::REST && ti == 0 {
            if status_immune(state, side, slot, PokemonStatus::SLEEP, false, side, slot) {
                move_failed = true;
            } else {
                set_status(state, rng, side, idx, PokemonStatus::SLEEP);
                let pm = state.side_mut(side).team_member_mut(idx);
                pm.sleep_turns = 3;
                pm.hp = pm.maxhp;
            }
        }

        // Haze `onHitField`: clear every active pokemon's stat stages.
        if choice.move_id == Choices::HAZE && ti == 0 {
            for sd in [Side::One, Side::Two] {
                for sl in 0..2 {
                    if let Some(ai) = state.side(sd).active[sl].pokemon_index {
                        state.side_mut(sd).clear_member_boosts(ai);
                    }
                }
            }
        }

        // Ally Switch `onHit`: swap positions with the (alive) ally.
        if choice.move_id == Choices::ALLYSWITCH && ti == 0 {
            if slot_alive(state, side, 1 - slot) {
                let sd = state.side_mut(side);
                sd.active.swap(0, 1);
                sd.ps_order.swap(0, 1);
            } else {
                move_failed = true;
            }
        }

        // Pain Split (`onHit`): both end at the floored average HP, clamped
        // to each one's max HP (`sethp`).
        if choice.move_id == Choices::PAINSPLIT && !move_failed {
            if let Some(tidx) = state.side(target_side).active[target_slot].pokemon_index {
                let thp = state.side(target_side).team_member(tidx).hp;
                let uhp = state.side(side).team_member(idx).hp;
                let avg = (((thp + uhp) / 2) as i16).max(1);
                let t = state.side_mut(target_side).team_member_mut(tidx);
                t.hp = avg.min(t.maxhp);
                let u = state.side_mut(side).team_member_mut(idx);
                u.hp = avg.min(u.maxhp);
            }
        }

        if choice.move_id == Choices::HELPINGHAND {
            state.side_mut(side).active[1 - slot].helping_hand_mult *= 1.5;
        }

        // Side conditions (Reflect, Light Screen, Spikes, Tailwind, ...).
        if let Some(sc) = &choice.side_condition {
            let cs = if sc.target == MoveTarget::User {
                side
            } else if sc.target == MoveTarget::Ally {
                side
            } else {
                target_side
            };
            if !apply_side_condition(state, cs, sc.condition) {
                move_failed = true;
            }
        }

        // Weather / terrain / pseudo-weather.
        if weather_id != 0 {
            let w = match weather_id {
                1 => Weather::RAIN,
                2 => Weather::SUN,
                3 => Weather::SAND,
                4 => Weather::HAIL,
                _ => Weather::NONE,
            };
            // PS weather `onFieldStart` / `setWeather`: fails if already up.
            if state.weather.weather_type == w {
                move_failed = true;
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
        if choice.move_id == Choices::WONDERROOM {
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
        let custom_secondary = match choice.move_id {
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
                    try_set_status(state, rng, target_side, target_slot, st, side, idx);
                }
                true
            }
            // Its 30% roll is the base-power draw in `calculate_damage`.
            Choices::FICKLEBEAM => true,
            Choices::SALTCURE => {
                let _ = rng.label("secondary").random(100);
                if slot_alive(state, target_side, target_slot) {
                    state.side_mut(target_side).active[target_slot].salt_cure = true;
                }
                true
            }
            // TODO(PS parity): trapping (Spirit Shackle), Throat Chop's sound
            // lock, Eerie Spell's PP cut, Psychic Noise's Heal Block. The
            // roll is consumed; hazards for Ceaseless Edge / Stone Axe are
            // applied via `side_condition` above.
            Choices::SPIRITSHACKLE
            | Choices::THROATCHOP
            | Choices::EERIESPELL
            | Choices::PSYCHICNOISE
            | Choices::CEASELESSEDGE
            | Choices::STONEAXE => {
                let _ = rng.label("secondary").random(100);
                true
            }
            _ => false,
        };

        // Secondary effects (flinch/burn/poison/boosts after a hit).
        if let Some(secs) = choice.secondaries.as_ref().filter(|_| !custom_secondary) {
            for sec in secs {
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
                        if !state.side(ts).active[tslot].focus_punch {
                            state.side_mut(ts).active[tslot].flinched = true;
                        }
                    }
                    Effect::Boost(b) => {
                        if let Some(tidx) = state.side(ts).active[tslot].pokemon_index {
                            apply_boosts(state, ts, tidx, b, ts != side);
                        }
                    }
                    Effect::Status(st) => {
                        if let Some(tidx) = state.side(ts).active[tslot].pokemon_index {
                            if !status_immune(state, ts, tslot, *st, ts != side, side, slot) {
                                set_status(state, rng, ts, tidx, *st);
                                if ts != side {
                                    maybe_synchronize(state, ts, tidx, side, idx, *st);
                                }
                            }
                        }
                    }
                    Effect::Heal(amount) => {
                        if let Some(tidx) = state.side(ts).active[tslot].pokemon_index {
                            let s = state.side_mut(ts);
                            let pm = s.team_member_mut(tidx);
                            let amt = (pm.maxhp as f32 * amount).round() as i16;
                            pm.hp = (pm.hp + amt).min(pm.maxhp);
                        }
                    }
                    Effect::VolatileStatus(PokemonVolatileStatus::FLINCH) => {
                        // Focus Punch's volatile blocks flinch (`onTryAddVolatile`).
                        if !state.side(ts).active[tslot].focus_punch {
                            state.side_mut(ts).active[tslot].flinched = true;
                        }
                    }
                    Effect::VolatileStatus(PokemonVolatileStatus::CONFUSION) => {
                        try_confuse(state, rng, ts, tslot);
                    }
                    Effect::VolatileStatus(vv) => {
                        if *vv == PokemonVolatileStatus::PROTECT {
                            state.side_mut(ts).active[tslot].protected = true;
                        }
                        // TODO(PS parity): other secondary volatiles.
                    }
                    Effect::IfStatsRaised { status } => {
                        // The chance roll was consumed above; the effect is
                        // gated on the target having raised a stat this turn
                        // (PS `statsRaisedThisTurn`).
                        if state.side(ts).active[tslot].stats_raised_this_turn {
                            if let Some(st) = status {
                                let st = *st;
                                if let Some(tidx) = state.side(ts).active[tslot].pokemon_index {
                                    let s = state.side_mut(ts);
                                    let pm = s.team_member_mut(tidx);
                                    if pm.status == PokemonStatus::NONE {
                                        pm.status = st;
                                        if st == PokemonStatus::TOXIC {
                                            s.toxic_stage[tidx as usize] = 1;
                                        }
                                    }
                                }
                            }
                            // TODO(PS parity): Alluring Voice's confusion.
                        }
                    }
                    Effect::RemoveItem => {
                        // TODO(PS parity): item removal.
                    }
                }
            }
        }

        // PS `forceSwitch` step (after secondaries): Roar / Whirlwind /
        // Dragon Tail / Circle Throw flag the target to be dragged out.
        // TODO(PS parity): Ingrain, Commander, Dynamax block it too.
        if choice.force_switch && !move_failed {
            if let Some(d) = state.side(target_side).slot_pokemon(target_slot) {
                let blocked = matches!(d.ability, Abilities::SUCTIONCUPS | Abilities::GUARDDOG);
                let attacker_alive = state.side(side).slot_pokemon(slot).map(|p| p.hp > 0).unwrap_or(false);
                if d.hp > 0 && attacker_alive && !blocked && !possible_switches(state, target_side).is_empty() {
                    state.side_mut(target_side).active[target_slot].drag_out = true;
                }
            }
        }

        // Focus Punch `condition.onHit`: a damaging move breaks the focus.
        if hit_damage.is_some() && state.side(target_side).active[target_slot].focus_punch {
            state.side_mut(target_side).active[target_slot].lost_focus = true;
        }

        // DamagingHit (after secondaries, PS `spreadMoveHit`).
        if hit_damage.is_some() {
            contact_abilities(state, rng, side, slot, idx, target_side, target_slot, choice);
            cursed_body(state, rng, side, slot, target_side, target_slot, choice);
            // Weak Armor, Steam Engine, ... (shared with the singles engine).
            if let Some(didx) = state.side(target_side).active[target_slot].pokemon_index {
                let ability = state.side(target_side).team_member(didx).ability;
                if let Some(b) = crate::engine::mechanics::on_damaging_hit_boosts(ability, choice) {
                    apply_boosts(state, target_side, didx, &b, false);
                }
                // Gooey / Tangling Hair (`onDamagingHit`): a contact hit lowers
                // the attacker's Speed by 1 stage.
                // TODO(PS parity): `checkMoveMakesContact` (Long Reach, Protective Pads).
                if matches!(ability, Abilities::GOOEY | Abilities::TANGLINGHAIR) && choice.flags.contact {
                    apply_boosts(
                        state,
                        side,
                        idx,
                        &StatBoosts {
                            speed: -1,
                            ..Default::default()
                        },
                        true,
                    );
                }
            }
            maybe_wandering_spirit(state, side, slot, target_side, target_slot, choice, rng);
        }

      }
        // PS `hitStepMoveHitLoop`: `eachEvent('Update')` after every hit that
        // did something. Field moves (`tryMoveHit`) have no hit loop.
        if is_field || move_failed {
            break;
        }
        hits_done += 1;
        each_event_update(state, rng);
    }

    if move_failed {
        state.side_mut(side).active[slot].move_result_this_turn = Some(false);
    }

    // lockedmove `onAfterMove`: removed when duration is 1; `onEnd` then
    // confuses the user unless trueDuration > 1.
    if let Some((1, t)) = state.side(side).active[slot].locked_move {
        state.side_mut(side).active[slot].locked_move = None;
        if t <= 1 {
            try_confuse(state, rng, side, slot);
        }
    }

    // Recoil (after all hits): round(damage * fraction), min 1.
    if total_damage > 0 {
        if let Some(frac) = choice.recoil {
            let recoil = ((total_damage as f32) * frac).round().max(1.0) as i16;
            apply_damage(state, side, slot, recoil);
        }
    }

    // PS `resolveFaints` right after the hit loop.
    state.unresolved_faints.clear();

    // End of PS's hit loop: one more `eachEvent('Update')` (after recoil).
    if hits_done > 0 {
        each_event_update(state, rng);
    }

    // Self-destruct moves faint the user after dealing damage.
    // PS `useMoveInner`: `selfdestruct: 'always'` faints the user (HP 0).
    if choice.selfdestruct {
        let hp = state.side(side).slot_pokemon(slot).map(|p| p.hp).unwrap_or(0);
        apply_damage(state, side, slot, hp);
    }

    // Spit Up: `onAfterMove` removes the Stockpile volatile.
    if choice.move_id == Choices::SPITUP {
        end_stockpile(state, side, slot);
    }

    // Recharge flag (Hyper Beam, Solar Beam): cannot act next turn.
    if choice.flags.recharge {
        state.side_mut(side).active[slot].recharging = true;
    }

    // Pivot moves (PS `selfSwitch`, battle-actions.ts `spreadMoveHit`): a
    // move that did something flags its still-alive user to switch out when
    // a replacement exists (`canSwitch`). run_battle_fuzz performs the switch
    // right after this action, from the controller's switch answer.
    // TODO(PS parity): Parting Shot (only if a drop landed), Baton Pass
    // (copies boosts), Shed Tail (Substitute), Chilly Reception (snow).
    let self_switch = match choice.move_id {
        Choices::UTURN | Choices::VOLTSWITCH | Choices::FLIPTURN => total_damage > 0,
        Choices::TELEPORT => true,
        _ => false,
    };
    let baton = choice.move_id == Choices::BATONPASS;
    if baton && state.side(side).find_replacement().is_none() {
        // Baton Pass `onHit`: `-fail` without a replacement.
        state.side_mut(side).active[slot].move_result_this_turn = Some(false);
    }
    if (self_switch || baton)
        && state.side(side).slot_pokemon(slot).map(|p| p.hp > 0).unwrap_or(false)
        && state.side(side).find_replacement().is_some()
    {
        state.side_mut(side).active[slot].force_switch = true;
        state.side_mut(side).active[slot].baton_passing = baton;
    }

    push_order();
}

/// Base power after the PS `BasePower` event (`getDamage`): the dynamic
/// base power, then every `onBasePower` handler chained into one 4096-based
/// modifier (`chainModify`), applied once with `modify`. Handlers run in
/// priority order: Tough Claws (21) before Helping Hand (10).
fn base_power_with_modifiers(
    choice: &Choice,
    attacker: &crate::state::Pokemon,
    defender: &crate::state::Pokemon,
    helping_hand_mult: f32,
    speeds: (i32, i32),
    terrain: crate::engine::state::Terrain,
    ate_boosted: bool,
) -> f32 {
    let dynamic = dynamic_base_power(choice, attacker, Some(defender), speeds);
    let bp = if dynamic > 0.0 { dynamic } else { choice.base_power };
    // (onBasePowerPriority, 4096-based modifier); PS runs higher priority
    // first. Values from data/abilities.ts / data/moves.ts `onBasePower`.
    let mut mods: Vec<(i32, i64)> = Vec::new();
    let f = &choice.flags;
    let physical = choice.category == MoveCategory::Physical;
    let special = choice.category == MoveCategory::Special;
    match attacker.ability {
        // Checks the base power after higher-priority modifiers; it has the
        // highest priority, so that is the plain base power.
        Abilities::TECHNICIAN if bp <= 60.0 => mods.push((30, 6144)),
        Abilities::IRONFIST if f.punch => mods.push((23, 4915)),
        Abilities::RECKLESS if choice.recoil.is_some() || choice.crash.is_some() => mods.push((23, 4915)),
        Abilities::TOUGHCLAWS if f.contact => mods.push((21, 5325)),
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
        // TODO(PS parity): Rivalry (genders), Sand Force (weather), Analytic,
        // Sheer Force, Supreme Overlord.
        _ => {}
    }
    if ate_boosted {
        mods.push((23, 4915));
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
    if let Some(m) = crate::engine::mechanics::terrain_base_power_modifier(terrain, choice, attacker, defender) {
        mods.push((6, m));
    }
    // Helping Hand stacks one x1.5 per use (`helping_hand_mult` = 1.5^n).
    let mut hh = helping_hand_mult;
    while hh > 1.01 {
        mods.push((10, 6144));
        hh /= 1.5;
    }
    if mods.is_empty() {
        return dynamic;
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
fn dynamic_base_power(
    choice: &Choice,
    attacker: &crate::state::Pokemon,
    defender: Option<&crate::state::Pokemon>,
    // PS `getStat('spe')` of attacker and defender (boosts + modifiers, no
    // Trick Room): Electro Ball / Gyro Ball.
    speeds: (i32, i32),
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
        let a = &mut state.side_mut(side).active[slot];
        let r = (a.stockpile_def, a.stockpile_spd);
        a.stockpile = 0;
        a.stockpile_def = 0;
        a.stockpile_spd = 0;
        r
    };
    if let Some(idx) = state.side(side).active[slot].pokemon_index {
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
/// boosted Atk and Def, no crit, one damage roll, at least 1.
fn confusion_damage(state: &DoublesState, rng: &mut Gen5Rng, side: Side, slot: usize, idx: PokemonIndex) -> i16 {
    let s = state.side(side);
    let p = s.team_member(idx);
    let b = s.slot_boosts(slot).cloned().unwrap_or_default();
    let atk = multiply_boost(b.atk, p.attack).max(1) as i64;
    let def = multiply_boost(b.def, p.defense).max(1) as i64;
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
fn try_confuse(state: &mut DoublesState, rng: &mut Gen5Rng, ts: Side, tslot: usize) {
    let Some(p) = state.side(ts).slot_pokemon(tslot) else { return };
    if p.hp <= 0 || p.ability == Abilities::OWNTEMPO || state.side(ts).active[tslot].confusion > 0 {
        return;
    }
    let t = rng.label("volatile duration").random_range(2, 6) as u8;
    state.side_mut(ts).active[tslot].confusion = t;
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
    if p.hp <= 0 || state.side(ts).active[tslot].partial_trap.is_some() {
        return;
    }
    let d = rng.label("volatile duration").random_range(5, 7) as u8;
    state.side_mut(ts).active[tslot].partial_trap = Some(crate::doubles::state::PartialTrap {
        duration: d,
        source_side: src_side,
        source_idx: src_idx,
        divisor: 8,
    });
}

/// PS `trySetStatus` from a secondary `onHit`: fails on an existing status
/// or a type immunity; sleep draws its duration.
fn try_set_status(
    state: &mut DoublesState,
    rng: &mut Gen5Rng,
    ts: Side,
    tslot: usize,
    st: PokemonStatus,
    src_side: Side,
    src_idx: PokemonIndex,
) {
    let Some(tidx) = state.side(ts).active[tslot].pokemon_index else { return };
    let src_slot = (0..2).find(|&sl| state.side(src_side).active[sl].pokemon_index == Some(src_idx));
    if status_immune(state, ts, tslot, st, ts != src_side, src_side, src_slot.unwrap_or(0)) {
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
    let Some(target) = state.side(ts).slot_pokemon(tslot) else { return true };
    // Uproar `onAnySetStatus`: nobody can fall asleep while one is active.
    if status == PokemonStatus::SLEEP
        && [Side::One, Side::Two]
            .iter()
            .any(|&sd| (0..2).any(|sl| state.side(sd).active[sl].uproar > 0))
    {
        return true;
    }
    let ctx = crate::engine::mechanics::StatusContext {
        sun: matches!(state.weather.weather_type, Weather::SUN | Weather::HARSHSUN),
        misty_terrain: state.terrain.terrain_type == crate::engine::state::Terrain::MISTYTERRAIN,
        electric_terrain: state.terrain.terrain_type == crate::engine::state::Terrain::ELECTRICTERRAIN,
        safeguard: state.side(ts).side_conditions.safeguard > 0,
        substitute: state.side(ts).active[tslot].substitute_health > 0,
        from_opponent,
        sleep_clause: false,
        source_ability: state
            .side(src_side)
            .slot_pokemon(src_slot)
            .map(|p| p.ability)
            .unwrap_or(Abilities::NONE),
    };
    crate::engine::mechanics::immune_to_status(target, &status, &ctx)
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
    let speeds = (stat_speed(state, side, slot), stat_speed(state, target_side, target_slot));
    let attacker = state.side(side).team_member(idx);
    let defender = state
        .side(target_side)
        .slot_pokemon(target_slot)
        .expect("target must be present");
    let mut damage: i16 = 0;
    let mut crit = false;
    // Fixed-damage and OHKO moves make no crit/randomizer draws.
    if choice.ohko {
        damage = defender.maxhp;
    } else if matches!(choice.move_id, Choices::COUNTER | Choices::MIRRORCOAT) {
        // damageCallback: the recorded `2 * damage` (1 if none).
        let a = &state.side(side).active[slot];
        let rec = if choice.move_id == Choices::COUNTER { a.counter_vol } else { a.mirror_coat_vol };
        damage = rec.map(|(_, d)| d).filter(|&d| d > 0).unwrap_or(1);
    } else if let Some(fd) = choice.move_id.fixed_damage(attacker, defender) {
        damage = fd;
    } else {
        // Boosted stats, with crit rules (crit ignores negative attack /
        // positive defense boosts) applied inside `calculate_damage`
        // after the crit roll.
        let att_boost = state.side(side).slot_boosts(slot).cloned().unwrap_or_default();
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
        let (base_atk, atk_boost_stat) = if choice.move_id == Choices::FOULPLAY {
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
        let ctx = DamageCtx {
            attack_base: base_atk,
            attack_boost: atk_boost_stat,
            defense_base: base_def,
            defense_boost: def_boost_stat,
            weather: state.weather.weather_type,
            reflect: state.side(target_side).side_conditions.reflect > 0,
            light_screen: state.side(target_side).side_conditions.light_screen > 0,
            defender_mod: crate::engine::mechanics::defender_damage_modifier(defender, choice),
            guts: attacker.ability == Abilities::GUTS,
            attack_mod: crate::engine::mechanics::attacker_stat_modifier(attacker, choice),
        crit_bonus: if state.side(side).active[slot].focus_energy { 2 } else { 0 },
            // Dynamic base power (Reversal / Flail) and Helping Hand
            // (x1.5) are combined into the override.
            base_power_override: base_power_with_modifiers(
                choice,
                attacker,
                defender,
                state.side(side).active[slot].helping_hand_mult,
                speeds,
                state.terrain.terrain_type,
                ate_boosted,
            ),
        };
        let mut ctx = ctx;
        if choice.move_id == Choices::STOMPINGTANTRUM
            && state.side(side).active[slot].move_result_last_turn == Some(false)
        {
            // PS `basePowerCallback`: doubled after a failed move.
            ctx.base_power_override = if ctx.base_power_override > 0.0 {
                ctx.base_power_override
            } else {
                choice.base_power
            } * 2.0;
        }
        if matches!(choice.move_id, Choices::STOREDPOWER | Choices::POWERTRIP) {
            // PS `basePowerCallback`: 20 + 20 * positiveBoosts().
            let b = state.side(side).slot_boosts(slot).cloned().unwrap_or_default();
            let pos: i32 = [b.atk, b.def, b.spa, b.spd, b.spe, b.acc, b.eva]
                .iter()
                .map(|&v| (v as i32).max(0))
                .sum();
            let bp = (20 + 20 * pos) as f32;
            // Helping Hand-style multipliers ride on the override ratio.
            let ratio = if choice.base_power > 0.0 && ctx.base_power_override > 0.0 {
                ctx.base_power_override / choice.base_power
            } else {
                1.0
            };
            ctx.base_power_override = bp * ratio;
        }
        if choice.move_id == Choices::SPITUP {
            // PS `basePowerCallback`: 100 per Stockpile layer.
            ctx.base_power_override = 100.0 * state.side(side).active[slot].stockpile as f32;
        }
        if let Some(outcome) = calculate_damage(attacker, defender, choice, rng, spread, &ctx) {
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
    let def_idx = state.side(target_side).active[target_slot].pokemon_index.or_else(|| {
        state
            .unresolved_faints
            .iter()
            .find(|(s, sl, _)| *s == target_side && *sl == target_slot)
            .map(|&(_, _, i)| i)
    });
    let Some(def_idx) = def_idx else { return };
    let ability = state.side(target_side).team_member(def_idx).ability;
    match contact_ability(ability) {
        Some(ContactAbility::Status30(st)) => {
            if rng.label("ability").random_chance(3, 10) {
                try_set_status(state, rng, side, slot, st, target_side, def_idx);
            }
        }
        Some(ContactAbility::EffectSpore) => {
            let attacker = state.side(side).team_member(idx);
            if !powder_immune(attacker) {
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
                    try_set_status(state, rng, side, slot, st, target_side, def_idx);
                }
            }
        }
        Some(ContactAbility::Recoil8) => {
            let maxhp = state.side(side).team_member(idx).maxhp;
            apply_damage(state, side, slot, (maxhp / 8).max(1));
        }
        None => {}
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
    let def_idx = state.side(target_side).active[target_slot].pokemon_index.or_else(|| {
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
    if state.side(side).active[slot].disabled.is_some() || choice.move_id == Choices::STRUGGLE {
        return;
    }
    if rng.label("ability").random_chance(3, 10) {
        state.side_mut(side).active[slot].disabled = Some((choice.move_id, 4));
    }
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
    let (Some(ai), Some(di)) = (
        state.side(att_side).active[att_slot].pokemon_index,
        state.side(def_side).active[def_slot].pokemon_index,
    ) else {
        return;
    };
    let _ = (ai, di);
    if state.side(def_side).team_member(di).ability != Abilities::WANDERINGSPIRIT {
        return;
    }
    skill_swap(state, att_side, att_slot, def_side, def_slot, rng);
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
        state.side(src_side).active[src_slot].pokemon_index,
        state.side(tgt_side).active[tgt_slot].pokemon_index,
    ) else {
        return false;
    };
    let src_ability = state.side(src_side).team_member(si).ability;
    let tgt_ability = state.side(tgt_side).team_member(ti).ability;
    if state.side(src_side).team_member(si).hp <= 0
        || state.side(tgt_side).team_member(ti).hp <= 0
        || crate::engine::mechanics::fails_skill_swap(src_ability)
        || crate::engine::mechanics::fails_skill_swap(tgt_ability)
    {
        return false;
    }
    state.side_mut(src_side).team_member_mut(si).ability = tgt_ability;
    state.side_mut(tgt_side).team_member_mut(ti).ability = src_ability;
    // The new abilities' Start events (Trace's draw included) run through
    // `apply_on_switch_abilities`.
    // TODO(PS parity): the TerrainChange speed sort after a swapped-in Surge.
    let _ = apply_on_switch_abilities(state, tgt_side, tgt_slot, rng);
    let _ = apply_on_switch_abilities(state, src_side, src_slot, rng);
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
    if source_side == target_side {
        return;
    }
    let has_sync = state.side(target_side).team_member(target_idx).ability == Abilities::SYNCHRONIZE;
    if !has_sync {
        return;
    }
    let s = state.side_mut(source_side);
    let pm = s.team_member_mut(source_idx);
    if pm.status == PokemonStatus::NONE {
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
fn apply_boosts(
    state: &mut DoublesState,
    side: Side,
    idx: PokemonIndex,
    b: &StatBoosts,
    source_is_foe: bool,
) {
    // Defiant / Competitive (shared `mechanics::stat_drop_reaction`).
    let reaction = if source_is_foe
        && (b.attack < 0
            || b.defense < 0
            || b.special_attack < 0
            || b.special_defense < 0
            || b.speed < 0
            || b.accuracy < 0
            || b.evasion < 0)
    {
        crate::engine::mechanics::stat_drop_reaction(state.side(side).team_member(idx).ability)
    } else {
        None
    };
    {
        let s = state.side_mut(side);
        let tb = &mut s.team_boosts[idx as usize];
        tb.atk = (tb.atk + b.attack).clamp(-6, 6);
        tb.def = (tb.def + b.defense).clamp(-6, 6);
        tb.spa = (tb.spa + b.special_attack).clamp(-6, 6);
        tb.spd = (tb.spd + b.special_defense).clamp(-6, 6);
        tb.spe = (tb.spe + b.speed).clamp(-6, 6);
        tb.acc = (tb.acc + b.accuracy).clamp(-6, 6);
        tb.eva = (tb.eva + b.evasion).clamp(-6, 6);
    }
    if let Some((stat, amount)) = reaction {
        let s = state.side_mut(side);
        let tb = &mut s.team_boosts[idx as usize];
        match stat {
            PokemonBoostableStat::Attack => tb.atk = (tb.atk + amount).clamp(-6, 6),
            PokemonBoostableStat::SpecialAttack => tb.spa = (tb.spa + amount).clamp(-6, 6),
            _ => {}
        }
    }
    // statsRaisedThisTurn (PS `boost`): set when any applied boost is
    // positive.
    let raised = b.attack > 0
        || b.defense > 0
        || b.special_attack > 0
        || b.special_defense > 0
        || b.speed > 0
        || b.accuracy > 0
        || b.evasion > 0;
    if raised {
        let s = state.side_mut(side);
        for a in &mut s.active {
            if a.pokemon_index == Some(idx) {
                a.stats_raised_this_turn = true;
            }
        }
    }
}

/// Abilities with the `notrace` flag (PS `data/abilities.ts`): Trace cannot
/// copy them.
fn has_notrace(a: Abilities) -> bool {
    matches!(
        a,
        Abilities::ASONEGLASTRIER
            | Abilities::ASONESPECTRIER
            | Abilities::BATTLEBOND
            | Abilities::COMATOSE
            | Abilities::COMMANDER
            | Abilities::DISGUISE
            | Abilities::EMBODYASPECTCORNERSTONE
            | Abilities::EMBODYASPECTHEARTHFLAME
            | Abilities::EMBODYASPECTTEAL
            | Abilities::EMBODYASPECTWELLSPRING
            | Abilities::FLOWERGIFT
            | Abilities::FORECAST
            | Abilities::HUNGERSWITCH
            | Abilities::ICEFACE
            | Abilities::ILLUSION
            | Abilities::IMPOSTER
            | Abilities::MULTITYPE
            | Abilities::NEUTRALIZINGGAS
            | Abilities::POISONPUPPETEER
            | Abilities::POWERCONSTRUCT
            | Abilities::POWEROFALCHEMY
            | Abilities::PROTOSYNTHESIS
            | Abilities::QUARKDRIVE
            | Abilities::RECEIVER
            | Abilities::RKSSYSTEM
            | Abilities::SCHOOLING
            | Abilities::SHIELDSDOWN
            | Abilities::STANCECHANGE
            | Abilities::TERAFORMZERO
            | Abilities::TERASHELL
            | Abilities::TERASHIFT
            | Abilities::TRACE
            | Abilities::ZENMODE
            | Abilities::ZEROTOHERO
    )
}

/// Run the Start effects of the ability of the pokemon in `side`'s
/// `active[slot]` (switch-in, Skill Swap): Trace, weather / terrain setters,
/// Curious Medicine, Intimidate, Supersweet Syrup. Returns true when it set a
/// terrain (the caller runs PS's TerrainChange speed sort).
// TODO(PS parity): other Start abilities (Download, Frisk, ...).
fn apply_on_switch_abilities(
    state: &mut DoublesState,
    side: Side,
    slot: usize,
    rng: &mut Gen5Rng,
) -> bool {
    let mut ability = {
        let s = state.side(side);
        match s.active[slot].pokemon_index {
            Some(idx) => s.team_member(idx).ability,
            None => return false,
        }
    };
    // Trace (`trace.onStart` -> `onUpdate`): copies a RANDOM adjacent foe's
    // ability, drawing `sample(possibleTargets)`. Foes with the `notrace`
    // flag or No Ability are excluded. Must happen before the copied
    // ability's own on-switch effects run.
    if ability == Abilities::TRACE {
        // `onStart`: a foe with No Ability stops Trace (`seek = false`).
        // TODO(PS parity): `onUpdate` keeps seeking while no foe is valid
        // (e.g. Trace activates later when a valid foe switches in).
        let foe = side.other();
        let foes: Vec<PokemonIndex> = (0..2).filter_map(|f| state.side(foe).active[f].pokemon_index).collect();
        let blocked = foes
            .iter()
            .any(|&fidx| state.side(foe).team_member(fidx).ability == Abilities::NOABILITY);
        let targets: Vec<PokemonIndex> = foes
            .into_iter()
            .filter(|&fidx| !has_notrace(state.side(foe).team_member(fidx).ability))
            .collect();
        if !blocked && !targets.is_empty() {
            let pick = targets[rng.label("trace").random(targets.len() as u32) as usize];
            let copied = state.side(foe).team_member(pick).ability;
            if let Some(idx) = state.side(side).active[slot].pokemon_index {
                state.side_mut(side).team_member_mut(idx).ability = copied;
            }
            ability = copied;
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
                return true;
            }
            return false;
        }
        None => {}
    }

    // Curious Medicine (`curiousmedicine.onStart`): clear the adjacent ally's
    // stat changes. Doubles-only (singles has no ally).
    if ability == Abilities::CURIOUSMEDICINE {
        if let Some(ally) = state.side(side).active[1 - slot].pokemon_index {
            state.side_mut(side).clear_member_boosts(ally);
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
            let me = state.side(side).active[slot].pokemon_index.unwrap();
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
        .active
        .iter()
        .filter(|a| a.pokemon_index.is_some() && a.substitute_health <= 0)
        .filter_map(|a| a.pokemon_index)
        .collect();
    for fidx in targets {
        apply_boosts(state, foe, fidx, &drop, true);
    }
    false
}

fn apply_side_condition(state: &mut DoublesState, side: Side, c: PokemonSideCondition) -> bool {
    let snow = matches!(state.weather.weather_type, Weather::HAIL | Weather::SNOW);
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
        | PokemonSideCondition::StickyWeb => return false,
        _ => {}
    }
    true
}

/// Put battle-team member `position` into `slot`, replacing whatever is
/// there (an alive occupant is switched out: PS `switchIn` ->
/// `clearVolatile` resets its boosts), and apply entry hazards. On-switch
/// abilities are applied by the caller.
fn switch_in(state: &mut DoublesState, side: Side, slot: usize, position: usize) {
    // PS `switchIn` swaps the two pokemon's places in `side.pokemon`.
    {
        let order = &mut state.side_mut(side).ps_order;
        if let Some(p) = order.iter().position(|&t| t == position) {
            order.swap(slot, p);
        }
    }
    if let Some(old) = state.side(side).active[slot].pokemon_index {
        // PS `clearVolatile`: boosts reset, ability back to the base one.
        let s = state.side_mut(side);
        s.clear_member_boosts(old);
        let p = s.team_member_mut(old);
        p.ability = p.base_ability;
        if let Some(t) = s.active[slot].type_change {
            s.team_member_mut(old).types = t;
        }
    }
    let pos = position.min(5);
    let idx = PokemonIndex::deserialize(&pos.to_string());
    // Read hazard layers + incoming pokemon data before mutating.
    let (spikes, toxic_spikes) = {
        let s = state.side(side);
        (s.side_conditions.spikes, s.side_conditions.toxic_spikes)
    };
    let (alive, types) = {
        let s = state.side(side);
        let p = &s.team[pos];
        (p.hp > 0 && p.id != crate::pokemon::PokemonName::NONE, p.types)
    };
    let s = state.side_mut(side);
    s.active[slot] = ActiveSlot::empty();
    if alive {
        s.active[slot].pokemon_index = Some(idx);
        let boots = s.team_member(idx).item == crate::engine::items::Items::HEAVYDUTYBOOTS;
        let grounded = s.team_member(idx).is_grounded();
        // Stealth Rock `onSwitchIn`: maxhp * 2^typeMod / 8 (`damage` truncates,
        // min 1). Applied before Spikes (side-condition insertion order).
        let rock_dmg = if s.side_conditions.stealth_rock > 0 && !boots {
            let mult = crate::engine::damage_calc::type_effectiveness_modifier(
                &PokemonType::ROCK,
                s.team_member(idx),
            );
            ((s.team_member(idx).maxhp as f32 * mult / 8.0) as i16).max(1)
        } else {
            0
        };
        let spikes_dmg = if grounded && spikes > 0 && !boots {
            let mult = [0, 3, 4, 6][spikes.min(3) as usize];
            ((s.team_member(idx).maxhp as i32 * mult / 24) as i16).max(1)
        } else {
            0
        };
        let spikes_dmg = spikes_dmg + rock_dmg;
        let new_status = if grounded && toxic_spikes > 0 {
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
        let clears_toxicspikes = grounded
            && toxic_spikes > 0
            && (types.0 == PokemonType::POISON || types.1 == PokemonType::POISON)
            && s.team_member(idx).status == PokemonStatus::NONE;
        // Apply hazard damage.
        if spikes_dmg > 0 {
            let p = &mut s.team_member_mut(idx);
            p.hp -= spikes_dmg;
        }
        // Apply toxic spikes status.
        if let Some(st) = new_status {
            let p = &mut s.team_member_mut(idx);
            p.status = st;
        }
        if clears_toxicspikes {
            s.side_conditions.toxic_spikes = 0;
        }
        if new_status == Some(PokemonStatus::TOXIC) {
            s.toxic_stage[idx as usize] = 1;
        }
        if s.team_member(idx).hp <= 0 {
            s.team_member_mut(idx).hp = 0;
            faint_pokemon(state, side, slot);
        }
    }
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
    /// Weather `onFieldResidual` (sandstorm damage via `eachEvent('Weather')`).
    Weather,
    /// brn / psn / tox `onResidual`.
    Status(Side, usize),
    /// `aquaring` `onResidual` (order 6).
    AquaRing(Side, usize),
    /// Wonder Room duration countdown (order 27, subOrder 5).
    WonderRoom,
    /// Terrain duration countdown (order 27, subOrder 7).
    Terrain,
    /// `allyswitch` duration countdown (duration 2).
    AllySwitch(Side, usize),
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
        push(27, 0, 7, ResidualKind::Terrain);
    }
    if state.trick_room.active {
        push(27, 0, 1, ResidualKind::Marker);
    }
    if state.wonder_room.active {
        push(27, 0, 5, ResidualKind::WonderRoom);
    }
    for side in [Side::One, Side::Two] {
        // Side conditions with a duration (`onSideResidualOrder` 26).
        let sc = &state.side(side).side_conditions;
        for (active, sub) in [
            (sc.reflect > 0, 1),
            (sc.light_screen > 0, 2),
            (sc.safeguard > 0, 3),
            (sc.mist > 0, 4),
            (sc.tailwind > 0, 5),
            (sc.lucky_chant > 0, 6),
            (sc.aurora_veil > 0, 10),
        ] {
            if active {
                push(26, 0, sub, ResidualKind::Marker);
            }
        }
        for slot in 0..2 {
            let a = &state.side(side).active[slot];
            let Some(idx) = a.pokemon_index else { continue };
            let p = state.side(side).team_member(idx);
            if p.hp <= 0 {
                continue;
            }
            let speed = cached_speed(state, side, slot);
            // Status (effectType 'Status': subOrder 0).
            match p.status {
                PokemonStatus::BURN => push(10, speed, 0, ResidualKind::Status(side, slot)),
                PokemonStatus::POISON | PokemonStatus::TOXIC => {
                    push(9, speed, 0, ResidualKind::Status(side, slot))
                }
                _ => {}
            }
            // Volatiles with a duration (Conditions: subOrder 2, no order).
            if let Some((m, _)) = a.charging {
                if !a.charge_aborted {
                    push(NO_ORDER, speed, 2, ResidualKind::Marker); // twoturnmove
                }
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
                    push(NO_ORDER, speed, 2, ResidualKind::Marker); // the move's own condition
                }
            }
            if a.protected || a.endure {
                push(NO_ORDER, speed, 2, ResidualKind::Marker); // protect / endure
            }
            if a.stall.is_some() {
                push(NO_ORDER, speed, 2, ResidualKind::Stall(side, slot));
            }
            if a.flinched {
                push(NO_ORDER, speed, 2, ResidualKind::Marker);
            }
            if a.helping_hand_mult > 1.0 {
                push(NO_ORDER, speed, 2, ResidualKind::Marker);
            }
            if a.focus_punch {
                push(NO_ORDER, speed, 2, ResidualKind::Marker); // focuspunch (duration 1)
            }
            if a.recharging {
                push(NO_ORDER, speed, 2, ResidualKind::Marker); // mustrecharge
            }
            if a.salt_cure {
                push(13, speed, 2, ResidualKind::SaltCure(side, slot));
            }
            if a.partial_trap.is_some() {
                push(13, speed, 2, ResidualKind::PartialTrap(side, slot));
            }
            if a.yawn > 0 {
                push(23, speed, 2, ResidualKind::Yawn(side, slot));
            }
            if a.aqua_ring {
                push(6, speed, 2, ResidualKind::AquaRing(side, slot));
            }
            if a.counter_vol.is_some() {
                push(NO_ORDER, speed, 2, ResidualKind::Marker); // counter (duration 1)
            }
            if a.mirror_coat_vol.is_some() {
                push(NO_ORDER, speed, 2, ResidualKind::Marker); // mirrorcoat (duration 1)
            }
            if a.ally_switch.is_some() {
                push(NO_ORDER, speed, 2, ResidualKind::AllySwitch(side, slot));
            }
            if a.uproar > 0 {
                push(28, speed, 1, ResidualKind::Uproar(side, slot));
            }
            if a.locked_move.is_some() {
                push(NO_ORDER, speed, 2, ResidualKind::Locked(side, slot));
            }
            if a.disabled.is_some() {
                push(17, speed, 2, ResidualKind::Disable(side, slot));
            }
            // Grassy Terrain heal (`onResidualOrder` 5, subOrder 2).
            if state.terrain.terrain_type == crate::engine::state::Terrain::GRASSYTERRAIN {
                push(5, speed, 2, ResidualKind::GrassyHeal(side, slot));
            }
            // Abilities (subOrder 7 unless set).
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
    let key = |h: &ResidualHandler| (h.order, -(h.priority as i64), -(h.speed as i64), h.sub_order as i64);
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
            shuffle(&mut hs, sorted, sorted + next_indexes.len(), rng, label);
        }
        sorted += next_indexes.len();
    }
    hs
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

/// PS `eachEvent('Update')` (sim/battle.ts): speed-sorts all actives, which
/// draws on speed ties. Called after every action and every hit.
fn each_event_update(state: &DoublesState, rng: &mut Gen5Rng) {
    let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
}

/// PS `eachEvent` / `runSwitch`: the actives speed-sorted (speed only), ties
/// shuffled.
fn sorted_actives(state: &DoublesState, rng: &mut Gen5Rng, label: &'static str) -> Vec<(Side, usize)> {
    // Actives in slot order; a pokemon that fainted in the current hit loop
    // still counts until faints are resolved (`unresolved_faints`).
    let mut hs: Vec<ResidualHandler> = Vec::new();
    for (s, sl) in [(Side::One, 0usize), (Side::One, 1), (Side::Two, 0), (Side::Two, 1)] {
        let speed = if state.side(s).slot_pokemon(sl).map(|p| p.hp > 0).unwrap_or(false) {
            cached_speed(state, s, sl)
        } else if let Some(&(_, _, idx)) =
            state.unresolved_faints.iter().find(|(fs, fsl, _)| *fs == s && *fsl == sl)
        {
            state.side(s).speed_cache[idx as usize] as i32
        } else {
            continue;
        };
        hs.push(ResidualHandler {
            order: 0,
            priority: 0,
            speed,
            sub_order: 0,
            kind: ResidualKind::Status(s, sl),
        });
    }
    sort_residual_handlers(hs, rng, label)
        .into_iter()
        .filter_map(|h| match h.kind {
            ResidualKind::Status(s, sl) => Some((s, sl)),
            _ => None,
        })
        .collect()
}

/// End-of-turn processing, mirroring pokemon-showdown's `update()`:
/// resolve faints (already done on damage), apply residual status/weather
/// damage to the actives in speed order, switch in replacements (with
/// hazard damage), then tick down weather / terrain / side-condition durations.
fn end_of_turn(state: &mut DoublesState, rng: &mut Gen5Rng, pending_switches: &[(Side, usize, usize)]) {
    // Residual phase: PS `runAction('residual')` -> `fieldEvent('Residual')`
    // collects every Residual handler (field, side, per-active status /
    // volatiles / ability) and speed-sorts them with `comparePriority`,
    // shuffling exact ties (consuming PRNG draws), then runs them in order.
    // `runAction('residual')` -> `updateSpeed()` first.
    update_speed(state);
    let handlers = collect_residual_handlers(state);
    let handlers = sort_residual_handlers(handlers, rng, "speed tie (residual)");
    for h in handlers {
        match h.kind {
            ResidualKind::Marker => {}
            ResidualKind::Weather => {
                // PS `fieldEvent`: the duration drops before the callback; at 0
                // the weather just ends (no `onFieldResidual`, no damage).
                if state.weather.turns_remaining > 0 {
                    state.weather.turns_remaining -= 1;
                    if state.weather.turns_remaining == 0 {
                        state.weather.weather_type = Weather::NONE;
                        // `clearWeather` -> `eachEvent('WeatherChange')`.
                        let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
                        continue;
                    }
                }
                // `onFieldResidual` -> `eachEvent('Weather')` then
                // `eachEvent('Update')`: two speed sorts of the actives.
                let order = sorted_actives(state, rng, "speed tie (eachEvent)");
                for (side, slot) in order {
                    if state.weather.weather_type != Weather::SAND {
                        continue;
                    }
                    let Some(idx) = state.side(side).active[slot].pokemon_index else { continue };
                    let p2 = state.side(side).team_member(idx);
                    let types = p2.types;
                    let immune = [PokemonType::ROCK, PokemonType::GROUND, PokemonType::STEEL]
                        .iter()
                        .any(|t| types.0 == *t || types.1 == *t);
                    if !immune {
                        apply_damage(state, side, slot, (p2.maxhp / 16) as i16);
                    }
                }
                each_event_update(state, rng);
            }
            ResidualKind::Status(side, slot) => {
                let idx = match state.side(side).active[slot].pokemon_index {
                    Some(i) => i,
                    None => continue,
                };
    let p = state.side(side).team_member(idx);
            let dmg: i16 = match p.status {
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
            if dmg > 0 {
                apply_damage(state, side, slot, dmg);
            }
            }
            ResidualKind::AquaRing(side, slot) => {
                if let Some(p) = state.side(side).slot_pokemon(slot) {
                    if p.hp > 0 {
                        let idx = state.side(side).active[slot].pokemon_index.unwrap();
                        let pm = state.side_mut(side).team_member_mut(idx);
                        pm.hp = (pm.hp + (pm.maxhp / 16).max(1)).min(pm.maxhp);
                    }
                }
            }
            ResidualKind::WonderRoom => {
                if state.wonder_room.turns_remaining > 0 {
                    state.wonder_room.turns_remaining -= 1;
                    if state.wonder_room.turns_remaining == 0 {
                        state.wonder_room.active = false;
                    }
                }
            }
            ResidualKind::Terrain => {
                if state.terrain.turns_remaining > 0 {
                    state.terrain.turns_remaining -= 1;
                    if state.terrain.turns_remaining == 0 {
                        state.terrain.terrain_type = crate::engine::state::Terrain::NONE;
                        // `clearTerrain` -> `eachEvent('TerrainChange')`.
                        let _ = sorted_actives(state, rng, "speed tie (eachEvent)");
                    }
                }
            }
            ResidualKind::AllySwitch(side, slot) => {
                let a = &mut state.side_mut(side).active[slot];
                if let Some((c, d)) = a.ally_switch {
                    a.ally_switch = if d <= 1 { None } else { Some((c, d - 1)) };
                }
            }
            ResidualKind::Uproar(side, slot) => {
                let a = &mut state.side_mut(side).active[slot];
                a.uproar = a.uproar.saturating_sub(1);
            }
            ResidualKind::Stall(side, slot) => {
                let a = &mut state.side_mut(side).active[slot];
                if let Some((c, d)) = a.stall {
                    a.stall = if d <= 1 { None } else { Some((c, d - 1)) };
                }
            }
            ResidualKind::Locked(side, slot) => {
                if state.side(side).slot_pokemon(slot).map(|p| p.hp <= 0).unwrap_or(true) {
                    continue;
                }
                let Some((mut d, mut t)) = state.side(side).active[slot].locked_move else { continue };
                d -= 1;
                if d == 0 {
                    // Duration ran out: `onEnd`.
                    state.side_mut(side).active[slot].locked_move = None;
                    if t <= 1 {
                        try_confuse(state, rng, side, slot);
                    }
                    continue;
                }
                if state.side(side).slot_pokemon(slot).map(|p| p.status == PokemonStatus::SLEEP).unwrap_or(false) {
                    state.side_mut(side).active[slot].locked_move = None;
                    continue;
                }
                t = t.saturating_sub(1);
                state.side_mut(side).active[slot].locked_move = Some((d, t));
            }
            ResidualKind::Disable(side, slot) => {
                let a = &mut state.side_mut(side).active[slot];
                if let Some((m, d)) = a.disabled {
                    a.disabled = if d <= 1 { None } else { Some((m, d - 1)) };
                }
            }
            ResidualKind::Yawn(side, slot) => {
                // PS fieldEvent: duration--; at 0 `onEnd` -> `trySetStatus('slp')`.
                if state.side(side).slot_pokemon(slot).map(|p| p.hp <= 0).unwrap_or(true) {
                    continue;
                }
                let a = &mut state.side_mut(side).active[slot];
                if a.yawn == 0 {
                    continue;
                }
                a.yawn -= 1;
                if a.yawn == 0 {
                    let src = a.yawn_source.take();
                    if let Some((ss, si)) = src {
                        let already = state.side(side).slot_pokemon(slot).map(|p| p.status != PokemonStatus::NONE).unwrap_or(true);
                        if !already {
                            try_set_status(state, rng, side, slot, PokemonStatus::SLEEP, ss, si);
                        }
                    }
                }
            }
            ResidualKind::PartialTrap(side, slot) => {
                // PS `fieldEvent`: duration-- before the callback; at 0 the
                // volatile ends without damage. `onResidual`: ends if the
                // binder is no longer active, else damages baseMaxhp/divisor.
                let Some(mut t) = state.side(side).active[slot].partial_trap else { continue };
                if state.side(side).slot_pokemon(slot).map(|p| p.hp <= 0).unwrap_or(true) {
                    continue;
                }
                t.duration -= 1;
                if t.duration == 0 {
                    state.side_mut(side).active[slot].partial_trap = None;
                    continue;
                }
                state.side_mut(side).active[slot].partial_trap = Some(t);
                let src_active = (0..2).any(|sl| {
                    let a = &state.side(t.source_side).active[sl];
                    a.pokemon_index == Some(t.source_idx)
                        && state.side(t.source_side).team_member(t.source_idx).hp > 0
                });
                if !src_active {
                    state.side_mut(side).active[slot].partial_trap = None;
                    continue;
                }
                let maxhp = state.side(side).slot_pokemon(slot).map(|p| p.maxhp).unwrap_or(0);
                apply_damage(state, side, slot, (maxhp / t.divisor).max(1));
            }
            ResidualKind::SaltCure(side, slot) => {
                if let Some(p) = state.side(side).slot_pokemon(slot) {
                    let has = |t| p.types.0 == t || p.types.1 == t;
                    let div = if has(PokemonType::WATER) || has(PokemonType::STEEL) { 8 } else { 16 };
                    let dmg = (p.maxhp / div).max(1);
                    apply_damage(state, side, slot, dmg);
                }
            }
            ResidualKind::GrassyHeal(side, slot) => {
                // PS `grassyterrain.onFieldResidual`: heal grounded actives by
                // 1/16 max HP (gen 8+).
                let heal = {
                    let s = state.side(side);
                    let Some(idx) = s.active[slot].pokemon_index else { continue };
                    let p = s.team_member(idx);
                    if p.is_grounded() && p.hp < p.maxhp {
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
                // TODO(PS parity): only when `activeTurns` > 0.
                if let Some(idx) = state.side(side).active[slot].pokemon_index {
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
    }

    // Aborted charge: the move's own condition (duration 2) has now ended.
    for side in [Side::One, Side::Two] {
        for slot in 0..2 {
            let a = &mut state.side_mut(side).active[slot];
            if a.charge_aborted {
                a.charge_aborted = false;
                a.charging = None;
            }
        }
    }

    // End of the `residual` action (faints are resolved before its eachEvent).
    state.unresolved_faints.clear();
    if state.battle_is_over() == 0.0 {
        each_event_update(state, rng);
    }
    check_fainted(state);

    // Switch in replacements (`instaswitch` actions), applying entry hazards,
    // then one `runSwitch` action for all of them. The switch answers go
    // through `commitChoices` -> `updateSpeed()` first.
    if !pending_switches.is_empty() {
        update_speed(state);
    }
    for (side, slot, position) in pending_switches {
        switch_in(state, *side, *slot, *position);
    }
    if !pending_switches.is_empty() {
        each_event_update(state, rng);
        let _ = sorted_actives(state, rng, "speed tie (switch-in)");
    }

    // On-switch abilities (Intimidate) for each newly switched-in pokemon.
    // TODO(PS parity): `fieldEvent('SwitchIn')` handler ties.
    for (side, slot, _) in pending_switches {
        if apply_on_switch_abilities(state, *side, *slot, rng) {
            let _ = sorted_actives(state, rng, "speed tie (switch-in)");
        }
    }
    if !pending_switches.is_empty() {
        each_event_update(state, rng);
    }

    // Tick down durations.

    if state.trick_room.active && state.trick_room.turns_remaining > 0 {
        state.trick_room.turns_remaining -= 1;
        if state.trick_room.turns_remaining == 0 {
            state.trick_room.active = false;
        }
    }
    for side in [Side::One, Side::Two] {
        let s = state.side_mut(side);
        let sc = &mut s.side_conditions;
        for d in [
            &mut sc.reflect,
            &mut sc.light_screen,
            &mut sc.aurora_veil,
            &mut sc.tailwind,
            &mut sc.safeguard,
            &mut sc.lucky_chant,
            &mut sc.mist,
        ] {
            if *d > 0 {
                *d -= 1;
            }
        }
        // Clear per-turn volatiles.
        for a in &mut s.active {
            a.counter_vol = None;
            a.mirror_coat_vol = None;
            a.move_result_last_turn = a.move_result_this_turn;
            a.move_result_this_turn = None;
            a.flinched = false;
            a.protected = false;
            a.endure = false;
            a.helping_hand_mult = 1.0;
            a.focus_punch = false;
            a.lost_focus = false;
            a.stats_raised_this_turn = false;
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
                if let Some(idx) = s.active[slot].pokemon_index {
                    if let Some(choice) = active_move(state, side, slot) {
                        let p = s.team_member(idx);
                        actions.push(Action {
                            side,
                            slot,
                            pokemon_index: idx,
                            choice,
                            priority: 0,
                            speed: effective_speed(state, side, slot).unwrap_or(0),
                            target: 0,
                            target_loc: 0,
                            fractional: 0,
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
                .active[action.slot]
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
                let alive_foes = state.side(target_side).active.iter().filter(|s| {
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
                    attack_base,
                    attack_boost,
                    defense_base,
                    defense_boost,
                    weather: state.weather.weather_type,
                    reflect: false,
                    light_screen: false,
                    defender_mod: 4096,
                    guts: attacker.ability == Abilities::GUTS,
                    attack_mod: crate::engine::mechanics::attacker_stat_modifier(attacker, &action.choice),
                    crit_bonus: 0,
                    base_power_override: 0.0,
                };
                let outcome: DamageOutcome = calculate_damage(attacker, defender, &action.choice, rng, false, &ctx)
                    .unwrap_or(DamageOutcome { damage: 0, crit: false });
                dmg_draw = Some(rng.draw_log.len() as u32 - 1);
                let defender_ability = defender.ability;
                if apply_damage(state, target_side, target_slot, outcome.damage) {
                    // Re-resolve the targets of ALL remaining actions (turnLoop): each
                    // consumes 1 sample sized to that action's target side.
                    for j in (i + 1)..n {
                        let a2 = &actions[j];
                        let foe_side = a2.side.other();
                        let alive_foes = state.side(foe_side).active.iter().filter(|s| {
                            s.pokemon_index
                                .map(|idx| state.side(foe_side).team_member(idx).hp > 0)
                                .unwrap_or(false)
                        }).count() as u32;
                        rng.random(alive_foes.max(1));
                    }
                    pending_switches.push((target_side, target_slot));
                }
                // Static: 30% chance to paralyze the attacker on a physical hit.
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

        if std::env::var("PE_SLOT").is_ok() {
            let slotname = |s: &crate::doubles::state::DoublesSide| {
                s.active
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
