//! PS event-handler sorts (`runEvent` -> `findEventHandlers` -> `speedSort`).
//!
//! PS collects every handler of an event from the target, its allies, its
//! foes, the source, both sides and the field, then speed-sorts them and
//! shuffles exact ties (one PRNG draw per tie group member). The engine
//! models the handlers' *effects* where the mechanic lives; this module only
//! makes the sort's draws, from:
//! - the handler table generated from PS (`event_handlers.rs`), and
//! - ONE list of what effects the engine state holds (`pokemon_effects`,
//!   `side_effects`, `field_effects`).
//!
//! A new volatile or side condition goes into those lists once and is then
//! sorted in every event PS gives it a handler for.

use super::battle::{cached_speed, real_ability};
use super::event_handlers::{handlers, HandlerKind};
use super::rng::Gen5Rng;
use super::state::{DoublesState, Side};
use crate::choices::Choices;
use crate::engine::state::Weather;
use crate::state::PokemonStatus;

/// PS `comparePriority`'s "no order" (`order || 4294967296`).
const NO_ORDER: i64 = 4294967296;

/// Events whose handlers PS sorts left to right instead of `speedSort`
/// (no draws). PS sim/battle.ts runEvent.
const LEFT_TO_RIGHT: [&str; 4] = ["Invulnerability", "TryHit", "DamagingHit", "EntryHazard"];

/// Events run through `eachEvent`: only `on<Event>` handlers, no prefixed ones.
/// PS sim/battle.ts findEventHandlers (`prefixedHandlers`).
const UNPREFIXED: [&str; 5] = ["BeforeTurn", "Update", "Weather", "WeatherChange", "TerrainChange"];

/// PS volatile id per `ActiveSlot::protect_kind`.
pub const PROTECT_IDS: [&str; 6] = ["protect", "kingsshield", "spikyshield", "banefulbunker", "silktrap", "burningbulwark"];

/// The status, ability and volatile ids (PS ids) of the pokemon in a slot,
/// plus its slot conditions. Abilities are reported even when suppressed:
/// PS sorts a suppressed handler and skips it only when it would run.
/// TODO(PS parity): the ability under Gastro Acid (the engine stores NONE).
pub fn pokemon_effects(state: &DoublesState, side: Side, slot: usize) -> Vec<(String, HandlerKind)> {
    use HandlerKind::*;
    let mut v: Vec<(String, HandlerKind)> = Vec::new();
    let s = state.side(side);
    let a = &s.active[slot];
    let Some(idx) = a.pokemon_index else { return v };
    let p = s.team_member(idx);
    let st = match p.status {
        PokemonStatus::BURN => Some("brn"),
        PokemonStatus::PARALYZE => Some("par"),
        PokemonStatus::SLEEP => Some("slp"),
        PokemonStatus::FREEZE => Some("frz"),
        PokemonStatus::POISON => Some("psn"),
        PokemonStatus::TOXIC => Some("tox"),
        _ => None,
    };
    if let Some(st) = st {
        v.push((st.to_string(), Status));
    }
    let ab = real_ability(state, side, idx);
    v.push((ab.to_string().to_lowercase(), Ability));
    let mut vol = |on: bool, id: &str| {
        if on {
            v.push((id.to_string(), Volatile));
        }
    };
    vol(a.confusion > 0, "confusion");
    vol(a.flinched, "flinch");
    vol(a.recharging, "mustrecharge");
    if let Some((m, _)) = a.charging.filter(|_| !a.charge_aborted) {
        vol(true, "twoturnmove");
        if matches!(
            m,
            Choices::DIG | Choices::DIVE | Choices::FLY | Choices::BOUNCE | Choices::PHANTOMFORCE
                | Choices::SHADOWFORCE | Choices::SKYDROP
        ) {
            vol(true, &m.to_string().to_lowercase());
        }
    }
    vol(a.protected, PROTECT_IDS[(a.protect_kind as usize).min(5)]);
    vol(a.endure, "endure");
    vol(a.substitute_health > 0, "substitute");
    vol(a.salt_cure, "saltcure");
    vol(a.stockpile > 0, "stockpile");
    vol(a.focus_punch, "focuspunch");
    vol(a.beak_blast, "beakblast");
    vol(a.gastro_acid, "gastroacid");
    vol(a.helping_hand_mult > 1.0, "helpinghand");
    vol(a.destinybond, "destinybond");
    vol(a.partial_trap.is_some(), "partiallytrapped");
    vol(a.yawn > 0, "yawn");
    vol(a.throat_chop > 0, "throatchop");
    vol(a.taunt > 0, "taunt");
    vol(a.magnet_rise > 0, "magnetrise");
    vol(a.smackdown, "smackdown");
    vol(a.cursed, "curse");
    vol(a.follow_me > 0, "followme");
    vol(a.perish > 0, "perishsong");
    vol(a.flash_fire, "flashfire");
    vol(a.charge, "charge");
    vol(a.no_retreat, "noretreat");
    vol(a.focus_energy, "focusenergy");
    vol(a.attract.is_some(), "attract");
    vol(a.leech_seed.is_some(), "leechseed");
    vol(a.dragon_cheer.is_some(), "dragoncheer");
    vol(a.glaive_rush, "glaiverush");
    vol(a.minimize, "minimize");
    vol(a.ingrain, "ingrain");
    vol(a.heal_block > 0, "healblock");
    vol(a.syrup_bomb.is_some(), "syrupbomb");
    vol(a.octolock.is_some(), "octolock");
    // No Retreat `onStart` adds the `trapped` volatile (PS data/moves.ts noretreat).
    vol(a.trapped || a.no_retreat, "trapped");
    vol(a.encore.is_some(), "encore");
    vol(a.torment, "torment");
    vol(a.imprison, "imprison");
    vol(a.roost_orig.is_some(), "roost");
    vol(a.aqua_ring, "aquaring");
    vol(a.counter_vol.is_some(), "counter");
    vol(a.mirror_coat_vol.is_some(), "mirrorcoat");
    vol(a.ally_switch.is_some(), "allyswitch");
    vol(a.lock_on.is_some(), "lockon");
    vol(a.uproar > 0, "uproar");
    vol(a.stall.is_some(), "stall");
    vol(a.locked_move.is_some(), "lockedmove");
    vol(a.disabled.is_some(), "disable");
    vol(a.electrify, "electrify");
    vol(a.power_trick, "powertrick");
    if s.future_move[slot].is_some() {
        v.push(("futuremove".to_string(), Slot));
    }
    if s.wish[slot].is_some() {
        v.push(("wish".to_string(), Slot));
    }
    v
}

/// The side conditions of a side (PS ids).
pub fn side_effects(state: &DoublesState, side: Side) -> Vec<&'static str> {
    let sc = &state.side(side).side_conditions;
    [
        (sc.reflect, "reflect"),
        (sc.light_screen, "lightscreen"),
        (sc.aurora_veil, "auroraveil"),
        (sc.safeguard, "safeguard"),
        (sc.mist, "mist"),
        (sc.lucky_chant, "luckychant"),
        (sc.tailwind, "tailwind"),
        (sc.quick_guard, "quickguard"),
        (sc.wide_guard, "wideguard"),
        (sc.crafty_shield, "craftyshield"),
        (sc.mat_block, "matblock"),
        (sc.spikes, "spikes"),
        (sc.toxic_spikes, "toxicspikes"),
        (sc.stealth_rock, "stealthrock"),
        (sc.sticky_web, "stickyweb"),
    ]
    .iter()
    .filter(|(n, _)| *n > 0)
    .map(|&(_, id)| id)
    .collect()
}

/// Weather, terrain and pseudo-weathers (PS ids).
pub fn field_effects(state: &DoublesState) -> Vec<&'static str> {
    use crate::engine::state::Terrain;
    let mut v = Vec::new();
    match state.weather.weather_type {
        Weather::RAIN => v.push("raindance"),
        Weather::HEAVYRAIN => v.push("primordialsea"),
        Weather::SUN => v.push("sunnyday"),
        Weather::HARSHSUN => v.push("desolateland"),
        Weather::SAND => v.push("sandstorm"),
        Weather::HAIL => v.push("hail"),
        Weather::SNOW => v.push("snowscape"),
        _ => {}
    }
    match state.terrain.terrain_type {
        Terrain::ELECTRICTERRAIN => v.push("electricterrain"),
        Terrain::GRASSYTERRAIN => v.push("grassyterrain"),
        Terrain::MISTYTERRAIN => v.push("mistyterrain"),
        Terrain::PSYCHICTERRAIN => v.push("psychicterrain"),
        _ => {}
    }
    for (on, id) in [
        (state.trick_room.active, "trickroom"),
        (state.gravity.active, "gravity"),
        (state.magic_room.active, "magicroom"),
        (state.wonder_room.active, "wonderroom"),
        (state.fairy_lock.active, "fairylock"),
    ] {
        if on {
            v.push(id);
        }
    }
    v
}

/// One collected handler: PS `comparePriority` keys.
#[derive(Debug, Clone, Copy)]
struct Key {
    order: i64,
    priority: i32,
    speed: i32,
    sub_order: i32,
}

fn push_pokemon(
    state: &DoublesState,
    out: &mut Vec<Key>,
    side: Side,
    slot: usize,
    callback: &str,
    debug: &mut Vec<String>,
) {
    let hs = handlers(callback);
    if hs.is_empty() {
        return;
    }
    let speed = cached_speed(state, side, slot);
    for (id, kind) in pokemon_effects(state, side, slot) {
        for h in hs.iter().filter(|h| h.kind == kind && h.id == id) {
            // PS resolvePriority: Magic Bounce's `onAllyTryHitSide` uses
            // `getStat('spe', true, true)` (no boosts, no modifiers).
            let speed = if callback == "onAllyTryHitSide" && h.id == "magicbounce" {
                state.side(side).slot_pokemon(slot).map(|p| p.speed as i32).unwrap_or(speed)
            } else {
                speed
            };
            out.push(Key {
                order: if h.order == 0 { NO_ORDER } else { h.order },
                priority: h.priority,
                speed,
                sub_order: h.sub_order,
            });
            if super::debug::flags().events {
                debug.push(format!("{} {} ({:?}{})", callback, h.id, side, slot));
            }
        }
    }
}

fn push_global(out: &mut Vec<Key>, ids: &[&str], kinds: &[HandlerKind], callback: &str, debug: &mut Vec<String>) {
    for h in handlers(callback).iter().filter(|h| kinds.contains(&h.kind) && ids.contains(&h.id)) {
        out.push(Key {
            order: if h.order == 0 { NO_ORDER } else { h.order },
            priority: h.priority,
            speed: 0,
            sub_order: h.sub_order,
        });
        if super::debug::flags().events {
            debug.push(format!("{} {}", callback, h.id));
        }
    }
}

fn alive(state: &DoublesState, side: Side, slot: usize) -> bool {
    state.side(side).slot_pokemon(slot).is_some_and(|p| p.hp > 0)
}

/// PS `runEvent(event, target, source)` for a Pokemon target: collect the
/// handlers like `findEventHandlers` and `speedSort` them, drawing for every
/// tie (label "speed tie (move order)", PS's purpose for runEvent sorts).
/// Only the draws; the handlers' effects are modelled where they apply.
/// PS sim/battle.ts runEvent / findEventHandlers / findPokemonEventHandlers /
/// resolvePriority / comparePriority.
pub fn run_event_ties(
    state: &DoublesState,
    rng: &mut Gen5Rng,
    event: &str,
    target: (Side, usize),
    source: Option<(Side, usize)>,
) {
    run_event_ties_on_effect(state, rng, event, target, source, None);
}

/// `run_event_ties` for a `runEvent(..., onEffect = true)` call: the source
/// effect's own `on<Event>` handler (a move's `onBasePower`) is added first,
/// held by the target, with its `on<Event>Priority` (`effect_priority`).
/// A Move effect has subOrder 0 (PS resolvePriority `effectTypeOrder`).
pub fn run_event_ties_on_effect(
    state: &DoublesState,
    rng: &mut Gen5Rng,
    event: &str,
    target: (Side, usize),
    source: Option<(Side, usize)>,
    effect_priority: Option<i32>,
) {
    if LEFT_TO_RIGHT.contains(&event) {
        return;
    }
    let prefixed = !UNPREFIXED.contains(&event);
    let mut hs: Vec<Key> = Vec::new();
    let mut dbg: Vec<String> = Vec::new();
    let (ts, tsl) = target;
    if let Some(priority) = effect_priority {
        hs.push(Key { order: NO_ORDER, priority, speed: cached_speed(state, ts, tsl), sub_order: 0 });
        if super::debug::flags().events {
            dbg.push(format!("on{} (move)", event));
        }
    }
    let on = format!("on{}", event);
    let target_active = state.side(ts).active[tsl].pokemon_index.is_some();
    let source_active = source.is_some_and(|(s, sl)| state.side(s).active[sl].pokemon_index.is_some());
    if target_active || source_active {
        push_pokemon(state, &mut hs, ts, tsl, &on, &mut dbg);
        if prefixed {
            let (ally, any) = (format!("onAlly{}", event), format!("onAny{}", event));
            let foe = format!("onFoe{}", event);
            for sl in 0..2 {
                if alive(state, ts, sl) {
                    push_pokemon(state, &mut hs, ts, sl, &ally, &mut dbg);
                    push_pokemon(state, &mut hs, ts, sl, &any, &mut dbg);
                }
            }
            for sl in 0..2 {
                if alive(state, ts.other(), sl) {
                    push_pokemon(state, &mut hs, ts.other(), sl, &foe, &mut dbg);
                    push_pokemon(state, &mut hs, ts.other(), sl, &any, &mut dbg);
                }
            }
        }
    }
    if let Some((ss, ssl)) = source.filter(|_| prefixed) {
        push_pokemon(state, &mut hs, ss, ssl, &format!("onSource{}", event), &mut dbg);
    }
    for sd in [Side::One, Side::Two] {
        let ids = side_effects(state, sd);
        if sd == ts {
            push_global(&mut hs, &ids, &[HandlerKind::Side], &on, &mut dbg);
        } else if prefixed {
            push_global(&mut hs, &ids, &[HandlerKind::Side], &format!("onFoe{}", event), &mut dbg);
        }
        if prefixed {
            push_global(&mut hs, &ids, &[HandlerKind::Side], &format!("onAny{}", event), &mut dbg);
        }
    }
    let field = field_effects(state);
    push_global(&mut hs, &field, &[HandlerKind::Field, HandlerKind::Weather], &on, &mut dbg);
    if crate::doubles::debug::flags().events && hs.len() > 1 {
        eprintln!("EVENT {} target={:?}{} handlers: {}", event, ts, tsl, dbg.join(", "));
    }
    speed_sort_keys(&mut hs, rng);
}

/// PS `speedSort` (selection sort, shuffling each group of equal keys).
fn speed_sort_keys(hs: &mut [Key], rng: &mut Gen5Rng) {
    let cmp = |a: &Key, b: &Key| {
        a.order
            .cmp(&b.order)
            .then(b.priority.cmp(&a.priority))
            .then(b.speed.cmp(&a.speed))
            .then(a.sub_order.cmp(&b.sub_order))
    };
    let mut sorted = 0usize;
    while sorted + 1 < hs.len() {
        let mut next = vec![sorted];
        for i in (sorted + 1)..hs.len() {
            match cmp(&hs[i], &hs[next[0]]) {
                std::cmp::Ordering::Less => next = vec![i],
                std::cmp::Ordering::Equal => next.push(i),
                std::cmp::Ordering::Greater => {}
            }
        }
        for (i, &index) in next.iter().enumerate() {
            if index != sorted + i {
                hs.swap(sorted + i, index);
            }
        }
        if next.len() > 1 {
            let end = sorted + next.len();
            let mut i = sorted;
            while i < end - 1 {
                let j = rng.label("speed tie (move order)").random_range(i, end);
                hs.swap(i, j);
                i += 1;
            }
        }
        sorted += next.len();
    }
}
