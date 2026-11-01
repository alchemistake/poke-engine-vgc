//! PS event-handler sorts (`runEvent` -> `findEventHandlers` -> `speedSort`).
//!
//! PS collects every handler of an event from the target, its allies, its
//! foes, the source, both sides and the field, then speed-sorts them and
//! shuffles exact ties (one PRNG draw per tie group member). The engine
//! models the handlers' *effects* where the mechanic lives; this module only
//! makes the sort's draws, from:
//! - the handler table generated from PS (`event_handlers.rs`), and
//! - ONE list of what effects the engine state holds (`pokemon_effects`,
//!   `side_mask`, `field_mask`).
//!
//! A new volatile or side condition goes into those lists once and is then
//! sorted in every event PS gives it a handler for.

use std::convert::TryInto;
use super::battle::{cached_speed, real_ability};
use super::event_handlers::{handlers, HandlerKind};
use super::rng::Gen5Rng;
use super::state::{DoublesState, Side};
use crate::choices::Choices;
use crate::engine::state::Weather;
use crate::state::PokemonStatus;

/// PS `comparePriority`'s "no order" (`order || 4294967296`).
const NO_ORDER: i64 = 4294967296;

/// PS volatile id per `ActiveSlot::protect_kind`.
pub const PROTECT_IDS: [&str; 6] = ["protect", "kingsshield", "spikyshield", "banefulbunker", "silktrap", "burningbulwark"];

/// The volatile id a charging semi-invulnerable move adds (PS ids).
pub fn semi_invulnerable_id(m: Choices) -> Option<&'static str> {
    match m {
        Choices::DIG => Some("dig"),
        Choices::DIVE => Some("dive"),
        Choices::FLY => Some("fly"),
        Choices::BOUNCE => Some("bounce"),
        Choices::PHANTOMFORCE => Some("phantomforce"),
        Choices::SHADOWFORCE => Some("shadowforce"),
        Choices::SKYDROP => Some("skydrop"),
        _ => None,
    }
}

/// One effect of a pokemon reported by `visit_pokemon_effects`.
#[derive(Clone, Copy)]
pub enum Effect {
    /// A status, volatile or slot condition: its PS id.
    Id(&'static str, HandlerKind),
    /// The ability (PS id: the lowercased name).
    Ability(crate::engine::abilities::Abilities),
}

/// The status, ability and volatile ids (PS ids) of the pokemon in a slot,
/// plus its slot conditions. Abilities are reported even when suppressed:
/// PS sorts a suppressed handler and skips it only when it would run.
pub fn pokemon_effects(state: &DoublesState, side: Side, slot: usize) -> Vec<(String, HandlerKind)> {
    let mut v = Vec::new();
    visit_pokemon_effects(state, side, slot, |e| match e {
        Effect::Id(id, kind) => v.push((id.to_string(), kind)),
        Effect::Ability(ab) => v.push((ab.to_string().to_lowercase(), HandlerKind::Ability)),
    });
    v
}

/// The volatile and slot conditions `visit_pokemon_effects` reports, in
/// report order (`cond_mask`).
const CONDS: [(&str, HandlerKind); 67] = [
    ("confusion", HandlerKind::Volatile),
    ("flinch", HandlerKind::Volatile),
    ("mustrecharge", HandlerKind::Volatile),
    ("twoturnmove", HandlerKind::Volatile),
    ("dig", HandlerKind::Volatile),
    ("dive", HandlerKind::Volatile),
    ("fly", HandlerKind::Volatile),
    ("bounce", HandlerKind::Volatile),
    ("phantomforce", HandlerKind::Volatile),
    ("shadowforce", HandlerKind::Volatile),
    ("skydrop", HandlerKind::Volatile),
    ("protect", HandlerKind::Volatile),
    ("kingsshield", HandlerKind::Volatile),
    ("spikyshield", HandlerKind::Volatile),
    ("banefulbunker", HandlerKind::Volatile),
    ("silktrap", HandlerKind::Volatile),
    ("burningbulwark", HandlerKind::Volatile),
    ("endure", HandlerKind::Volatile),
    ("substitute", HandlerKind::Volatile),
    ("saltcure", HandlerKind::Volatile),
    ("stockpile", HandlerKind::Volatile),
    ("focuspunch", HandlerKind::Volatile),
    ("beakblast", HandlerKind::Volatile),
    ("gastroacid", HandlerKind::Volatile),
    ("helpinghand", HandlerKind::Volatile),
    ("destinybond", HandlerKind::Volatile),
    ("partiallytrapped", HandlerKind::Volatile),
    ("yawn", HandlerKind::Volatile),
    ("throatchop", HandlerKind::Volatile),
    ("taunt", HandlerKind::Volatile),
    ("magnetrise", HandlerKind::Volatile),
    ("smackdown", HandlerKind::Volatile),
    ("curse", HandlerKind::Volatile),
    ("followme", HandlerKind::Volatile),
    ("perishsong", HandlerKind::Volatile),
    ("flashfire", HandlerKind::Volatile),
    ("charge", HandlerKind::Volatile),
    ("noretreat", HandlerKind::Volatile),
    ("focusenergy", HandlerKind::Volatile),
    ("attract", HandlerKind::Volatile),
    ("leechseed", HandlerKind::Volatile),
    ("dragoncheer", HandlerKind::Volatile),
    ("glaiverush", HandlerKind::Volatile),
    ("minimize", HandlerKind::Volatile),
    ("ingrain", HandlerKind::Volatile),
    ("healblock", HandlerKind::Volatile),
    ("syrupbomb", HandlerKind::Volatile),
    ("octolock", HandlerKind::Volatile),
    ("trapped", HandlerKind::Volatile),
    ("encore", HandlerKind::Volatile),
    ("torment", HandlerKind::Volatile),
    ("imprison", HandlerKind::Volatile),
    ("roost", HandlerKind::Volatile),
    ("aquaring", HandlerKind::Volatile),
    ("counter", HandlerKind::Volatile),
    ("mirrorcoat", HandlerKind::Volatile),
    ("allyswitch", HandlerKind::Volatile),
    ("lockon", HandlerKind::Volatile),
    ("uproar", HandlerKind::Volatile),
    ("stall", HandlerKind::Volatile),
    ("lockedmove", HandlerKind::Volatile),
    ("disable", HandlerKind::Volatile),
    ("electrify", HandlerKind::Volatile),
    ("powertrick", HandlerKind::Volatile),
    ("choicelock", HandlerKind::Volatile),
    ("futuremove", HandlerKind::Slot),
    ("wish", HandlerKind::Slot),
];

/// `CONDS` bits of the slot conditions (after every volatile).
const FUTURE_BIT: u32 = CONDS.len() as u32 - 2;
const WISH_BIT: u32 = CONDS.len() as u32 - 1;


/// The move a slot is charging (a two-turn move not aborted).
fn charging(a: &crate::doubles::state::ActiveSlot) -> Option<Choices> {
    a.charging.filter(|_| !a.charge_aborted).map(|(m, _)| m)
}

/// The conditions of `CONDS` that are up, as bits (bit i = `CONDS[i]`).
fn cond_mask(s: &crate::doubles::state::DoublesSide, slot: usize) -> u128 {
    let mut m = s.cached_conds(slot, vol_mask);
    if s.future_move[slot].is_some() {
        m |= 1 << FUTURE_BIT;
    }
    if s.wish[slot].is_some() {
        m |= 1 << WISH_BIT;
    }
    m
}

/// The ids of the volatile conditions up in a slot, sorted (fuzz snapshot).
/// A charging move also has a volatile of its own id (PS `solarbeam`,
/// `meteorbeam`, ...) beside `twoturnmove`.
pub(crate) fn volatile_ids(a: &crate::doubles::state::ActiveSlot) -> Vec<String> {
    let mut m = vol_mask(a);
    let mut v = Vec::new();
    while m != 0 {
        let id = CONDS[m.trailing_zeros() as usize].0;
        // One bit holds both redirect volatiles.
        v.push(if id == "followme" && a.follow_me == 2 { "ragepowder" } else { id }.to_string());
        m &= m - 1;
    }
    if let Some(mv) = charging(a) {
        let id = format!("{:?}", mv).to_lowercase();
        if !v.contains(&id) {
            v.push(id);
        }
    }
    // Chilly Reception's `priorityChargeCallback` volatile (no handlers).
    if a.chilly_reception {
        v.push("chillyreception".to_string());
    }
    v.sort();
    v
}

/// The volatile conditions of `CONDS` (every bit below `FUTURE_BIT`) that are
/// up in a slot.
pub(crate) fn vol_mask(a: &crate::doubles::state::ActiveSlot) -> u128 {
    if a.pokemon_index.is_none() {
        return 0;
    }
    let mut m = 0u128;
    if a.choice_lock.is_some() {
        m |= 1 << 64;
    }
    if a.confusion > 0 {
        m |= 1 << 0;
    }
    if a.flinched {
        m |= 1 << 1;
    }
    if a.recharging {
        m |= 1 << 2;
    }
    if charging(a).is_some() {
        m |= 1 << 3;
    }
    if charging(a) == Some(Choices::DIG) {
        m |= 1 << 4;
    }
    if charging(a) == Some(Choices::DIVE) {
        m |= 1 << 5;
    }
    if charging(a) == Some(Choices::FLY) {
        m |= 1 << 6;
    }
    if charging(a) == Some(Choices::BOUNCE) {
        m |= 1 << 7;
    }
    if charging(a) == Some(Choices::PHANTOMFORCE) {
        m |= 1 << 8;
    }
    if charging(a) == Some(Choices::SHADOWFORCE) {
        m |= 1 << 9;
    }
    if charging(a) == Some(Choices::SKYDROP) {
        m |= 1 << 10;
    }
    if a.protected && (a.protect_kind as usize).min(5) == 0 {
        m |= 1 << 11;
    }
    if a.protected && (a.protect_kind as usize).min(5) == 1 {
        m |= 1 << 12;
    }
    if a.protected && (a.protect_kind as usize).min(5) == 2 {
        m |= 1 << 13;
    }
    if a.protected && (a.protect_kind as usize).min(5) == 3 {
        m |= 1 << 14;
    }
    if a.protected && (a.protect_kind as usize).min(5) == 4 {
        m |= 1 << 15;
    }
    if a.protected && (a.protect_kind as usize).min(5) == 5 {
        m |= 1 << 16;
    }
    if a.endure {
        m |= 1 << 17;
    }
    if a.substitute_health > 0 {
        m |= 1 << 18;
    }
    if a.salt_cure {
        m |= 1 << 19;
    }
    if a.stockpile > 0 {
        m |= 1 << 20;
    }
    if a.focus_punch {
        m |= 1 << 21;
    }
    if a.beak_blast {
        m |= 1 << 22;
    }
    if a.gastro_acid {
        m |= 1 << 23;
    }
    if a.helping_hand_mult > 1.0 {
        m |= 1 << 24;
    }
    if a.destinybond {
        m |= 1 << 25;
    }
    if a.partial_trap.is_some() {
        m |= 1 << 26;
    }
    if a.yawn > 0 {
        m |= 1 << 27;
    }
    if a.throat_chop > 0 {
        m |= 1 << 28;
    }
    if a.taunt > 0 {
        m |= 1 << 29;
    }
    if a.magnet_rise > 0 {
        m |= 1 << 30;
    }
    if a.smackdown {
        m |= 1 << 31;
    }
    if a.cursed {
        m |= 1 << 32;
    }
    if a.follow_me > 0 {
        m |= 1 << 33;
    }
    if a.perish > 0 {
        m |= 1 << 34;
    }
    if a.flash_fire {
        m |= 1 << 35;
    }
    if a.charge {
        m |= 1 << 36;
    }
    if a.no_retreat {
        m |= 1 << 37;
    }
    if a.focus_energy {
        m |= 1 << 38;
    }
    if a.attract.is_some() {
        m |= 1 << 39;
    }
    if a.leech_seed.is_some() {
        m |= 1 << 40;
    }
    if a.dragon_cheer.is_some() {
        m |= 1 << 41;
    }
    if a.glaive_rush {
        m |= 1 << 42;
    }
    if a.minimize {
        m |= 1 << 43;
    }
    if a.ingrain {
        m |= 1 << 44;
    }
    if a.heal_block > 0 {
        m |= 1 << 45;
    }
    if a.syrup_bomb.is_some() {
        m |= 1 << 46;
    }
    if a.octolock.is_some() {
        m |= 1 << 47;
    }
    // No Retreat `onStart` adds the `trapped` volatile (PS data/moves.ts noretreat).
    if a.trapped || a.no_retreat {
        m |= 1 << 48;
    }
    if a.encore.is_some() {
        m |= 1 << 49;
    }
    if a.torment {
        m |= 1 << 50;
    }
    if a.imprison {
        m |= 1 << 51;
    }
    if a.roost {
        m |= 1 << 52;
    }
    if a.aqua_ring {
        m |= 1 << 53;
    }
    if a.counter_vol.is_some() {
        m |= 1 << 54;
    }
    if a.mirror_coat_vol.is_some() {
        m |= 1 << 55;
    }
    if a.ally_switch.is_some() {
        m |= 1 << 56;
    }
    if a.lock_on.is_some() {
        m |= 1 << 57;
    }
    if a.uproar > 0 {
        m |= 1 << 58;
    }
    if a.stall.is_some() {
        m |= 1 << 59;
    }
    if a.locked_move.is_some() {
        m |= 1 << 60;
    }
    if a.disabled.is_some() {
        m |= 1 << 61;
    }
    if a.electrify {
        m |= 1 << 62;
    }
    if a.power_trick {
        m |= 1 << 63;
    }
    m
}

/// `pokemon_effects` without allocating: calls `f` for each effect, in the
/// same order (the order PS collects their handlers in).
/// TODO(PS parity): the ability under Gastro Acid (the engine stores NONE).
pub fn visit_pokemon_effects(state: &DoublesState, side: Side, slot: usize, mut f: impl FnMut(Effect)) {
    use HandlerKind::*;
    let s = state.side(side);
    let a = &s.active()[slot];
    let Some(idx) = a.pokemon_index else { return };
    let p = s.team_member(idx);
    if let Some(st) = status_id(p.status) {
        f(Effect::Id(st, Status));
    }
    f(Effect::Ability(real_ability(state, side, idx)));
    let mut m = cond_mask(s, slot);
    while m != 0 {
        let (id, kind) = CONDS[m.trailing_zeros() as usize];
        f(Effect::Id(id, kind));
        m &= m - 1;
    }
}

/// Status ids (PS ids), indexed by `status_index`.
const STATUS_IDS: [&str; 6] = ["brn", "par", "slp", "frz", "psn", "tox"];

fn status_index(st: PokemonStatus) -> Option<usize> {
    match st {
        PokemonStatus::BURN => Some(0),
        PokemonStatus::PARALYZE => Some(1),
        PokemonStatus::SLEEP => Some(2),
        PokemonStatus::FREEZE => Some(3),
        PokemonStatus::POISON => Some(4),
        PokemonStatus::TOXIC => Some(5),
        _ => None,
    }
}

fn status_id(st: PokemonStatus) -> Option<&'static str> {
    status_index(st).map(|i| STATUS_IDS[i])
}

/// Side condition ids (PS ids); a side's active ones are a bit mask over
/// this list (`side_mask`).
const SIDE_IDS: [&str; 15] = [
    "reflect", "lightscreen", "auroraveil", "safeguard", "mist", "luckychant", "tailwind", "quickguard",
    "wideguard", "craftyshield", "matblock", "spikes", "toxicspikes", "stealthrock", "stickyweb",
];

/// Byte offset of each `SIDE_IDS` condition in `SideConditions`.
const SIDE_OFFSETS: [usize; 15] = {
    use crate::state::SideConditions as S;
    use std::mem::offset_of;
    [
        offset_of!(S, reflect),
        offset_of!(S, light_screen),
        offset_of!(S, aurora_veil),
        offset_of!(S, safeguard),
        offset_of!(S, mist),
        offset_of!(S, lucky_chant),
        offset_of!(S, tailwind),
        offset_of!(S, quick_guard),
        offset_of!(S, wide_guard),
        offset_of!(S, crafty_shield),
        offset_of!(S, mat_block),
        offset_of!(S, spikes),
        offset_of!(S, toxic_spikes),
        offset_of!(S, stealth_rock),
        offset_of!(S, sticky_web),
    ]
};

const _: () = assert!(std::mem::size_of::<crate::state::SideConditions>() <= 32);

/// The bytes of a `SideConditions` (all `i8` fields), zero-padded to 32.
#[inline]
fn side_bytes(sc: &crate::state::SideConditions) -> [u128; 2] {
    let mut b = [0u8; 32];
    // SAFETY: `SideConditions` is all `i8` fields (no padding) and fits in `b`.
    unsafe {
        std::ptr::copy_nonoverlapping(
            sc as *const _ as *const u8,
            b.as_mut_ptr(),
            std::mem::size_of::<crate::state::SideConditions>(),
        )
    };
    [u128::from_ne_bytes(b[..16].try_into().unwrap()), u128::from_ne_bytes(b[16..].try_into().unwrap())]
}

/// The side conditions of a side, as bits over `SIDE_IDS`.
fn side_mask(state: &DoublesState, side: Side) -> u32 {
    let sc = &state.side(side).side_conditions;
    [
        sc.reflect,
        sc.light_screen,
        sc.aurora_veil,
        sc.safeguard,
        sc.mist,
        sc.lucky_chant,
        sc.tailwind,
        sc.quick_guard,
        sc.wide_guard,
        sc.crafty_shield,
        sc.mat_block,
        sc.spikes,
        sc.toxic_spikes,
        sc.stealth_rock,
        sc.sticky_web,
    ]
    .iter()
    .enumerate()
    .fold(0, |m, (i, &n)| if n > 0 { m | 1 << i } else { m })
}

/// Weather, terrain and pseudo-weather ids (PS ids); the field's active ones
/// are a bit mask over this list (`field_mask`).
const FIELD_IDS: [&str; 16] = [
    "raindance", "primordialsea", "sunnyday", "desolateland", "sandstorm", "hail", "snowscape",
    "electricterrain", "grassyterrain", "mistyterrain", "psychicterrain",
    "trickroom", "gravity", "magicroom", "wonderroom", "fairylock",
];

/// The weathers / terrains of `FIELD_IDS[0..7]` / `FIELD_IDS[7..11]`.
const FIELD_WEATHERS: [Weather; 7] =
    [Weather::RAIN, Weather::HEAVYRAIN, Weather::SUN, Weather::HARSHSUN, Weather::SAND, Weather::HAIL, Weather::SNOW];
const FIELD_TERRAINS: [crate::engine::state::Terrain; 4] = {
    use crate::engine::state::Terrain;
    [Terrain::ELECTRICTERRAIN, Terrain::GRASSYTERRAIN, Terrain::MISTYTERRAIN, Terrain::PSYCHICTERRAIN]
};

/// Weather, terrain and pseudo-weathers, as bits over `FIELD_IDS`.
fn field_mask(state: &DoublesState) -> u32 {
    use crate::engine::state::Terrain;
    let weather = match state.weather.weather_type {
        Weather::RAIN => Some(0),
        Weather::HEAVYRAIN => Some(1),
        Weather::SUN => Some(2),
        Weather::HARSHSUN => Some(3),
        Weather::SAND => Some(4),
        Weather::HAIL => Some(5),
        Weather::SNOW => Some(6),
        _ => None,
    };
    let terrain = match state.terrain.terrain_type {
        Terrain::ELECTRICTERRAIN => Some(7),
        Terrain::GRASSYTERRAIN => Some(8),
        Terrain::MISTYTERRAIN => Some(9),
        Terrain::PSYCHICTERRAIN => Some(10),
        _ => None,
    };
    let mut m = 0;
    for bit in [weather, terrain].iter().flatten() {
        m |= 1 << bit;
    }
    for (i, on) in [
        state.trick_room.active,
        state.gravity.active,
        state.magic_room.active,
        state.wonder_room.active,
        state.fairy_lock.active,
    ]
    .iter()
    .enumerate()
    {
        if *on {
            m |= 1 << (11 + i);
        }
    }
    m
}

/// The events the engine sorts handlers for (`run_event_ties`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    Accuracy,
    AfterMega,
    BasePower,
    BeforeMove,
    ChangeBoost,
    Damage,
    DisableMove,
    Immunity,
    MaybeTrapPokemon,
    ModifyAccuracy,
    ModifyAtk,
    ModifyBoost,
    ModifyCritRatio,
    ModifyDamage,
    ModifyDef,
    ModifySpA,
    ModifySpD,
    MoveAborted,
    SetStatus,
    TrapPokemon,
    TryAddVolatile,
    TryBoost,
    TryHitSide,
    TryMove,
}

impl Event {
    const ALL: [Event; 24] = [
        Event::Accuracy,
        Event::AfterMega,
        Event::BasePower,
        Event::BeforeMove,
        Event::ChangeBoost,
        Event::Damage,
        Event::DisableMove,
        Event::Immunity,
        Event::MaybeTrapPokemon,
        Event::ModifyAccuracy,
        Event::ModifyAtk,
        Event::ModifyBoost,
        Event::ModifyCritRatio,
        Event::ModifyDamage,
        Event::ModifyDef,
        Event::ModifySpA,
        Event::ModifySpD,
        Event::MoveAborted,
        Event::SetStatus,
        Event::TrapPokemon,
        Event::TryAddVolatile,
        Event::TryBoost,
        Event::TryHitSide,
        Event::TryMove,
    ];

    /// The PS event name (`"ModifyAtk"`).
    pub fn name(self) -> &'static str {
        match self {
            Event::Accuracy => "Accuracy",
            Event::AfterMega => "AfterMega",
            Event::BasePower => "BasePower",
            Event::BeforeMove => "BeforeMove",
            Event::ChangeBoost => "ChangeBoost",
            Event::Damage => "Damage",
            Event::DisableMove => "DisableMove",
            Event::Immunity => "Immunity",
            Event::MaybeTrapPokemon => "MaybeTrapPokemon",
            Event::ModifyAccuracy => "ModifyAccuracy",
            Event::ModifyAtk => "ModifyAtk",
            Event::ModifyBoost => "ModifyBoost",
            Event::ModifyCritRatio => "ModifyCritRatio",
            Event::ModifyDamage => "ModifyDamage",
            Event::ModifyDef => "ModifyDef",
            Event::ModifySpA => "ModifySpA",
            Event::ModifySpD => "ModifySpD",
            Event::MoveAborted => "MoveAborted",
            Event::SetStatus => "SetStatus",
            Event::TrapPokemon => "TrapPokemon",
            Event::TryAddVolatile => "TryAddVolatile",
            Event::TryBoost => "TryBoost",
            Event::TryHitSide => "TryHitSide",
            Event::TryMove => "TryMove",
        }
    }
}

/// Callback prefixes of pokemon handlers (`POKE_PREFIXES[i]` + event name).
const POKE_PREFIXES: [&str; 5] = ["on", "onAlly", "onAny", "onFoe", "onSource"];
const ON: usize = 0;
const ALLY: usize = 1;
const ANY: usize = 2;
const FOE: usize = 3;
const SOURCE: usize = 4;
/// Callback prefixes of side handlers (`side[i]` of an `EventTable`).
const SIDE_ON: usize = 0;
const SIDE_FOE: usize = 1;
const SIDE_ANY: usize = 2;

/// A pokemon handler of one callback: `bit` is its effect (`Tables::fx`).
/// Most pokemon effects (ability, status, volatile, slot condition, item)
/// with an event handler: the width of `FxMask`.
const MAX_FX: usize = 512;

/// A set of pokemon effect bits (`Tables` effect numbering), `MAX_FX` wide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct FxMask([u64; MAX_FX / 64]);

impl FxMask {
    const ZERO: FxMask = FxMask([0; MAX_FX / 64]);
    #[inline]
    fn bit(b: u16) -> FxMask {
        let mut m = FxMask::ZERO;
        m.0[b as usize >> 6] = 1 << (b & 63);
        m
    }
    #[inline]
    fn has(&self, b: u16) -> bool {
        self.0[b as usize >> 6] >> (b & 63) & 1 != 0
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.0.iter().all(|&w| w == 0)
    }
    #[inline]
    fn intersects(&self, o: &FxMask) -> bool {
        self.0.iter().zip(o.0.iter()).any(|(a, b)| a & b != 0)
    }
}

impl std::ops::BitOrAssign for FxMask {
    #[inline]
    fn bitor_assign(&mut self, o: FxMask) {
        for (a, b) in self.0.iter_mut().zip(o.0) {
            *a |= b;
        }
    }
}

struct PokeEntry {
    bit: u16,
    id: &'static str,
    order: i64,
    priority: i32,
    sub_order: i32,
    /// Magic Bounce's `onAllyTryHitSide` sorts by `getStat('spe', true, true)`
    /// (no boosts, no modifiers). PS sim/battle.ts resolvePriority.
    raw_speed: bool,
}

/// A side / field handler of one callback: `idx` is its bit in `SIDE_IDS` /
/// `FIELD_IDS`.
struct GlobalEntry {
    idx: u8,
    id: &'static str,
    order: i64,
    priority: i32,
    sub_order: i32,
}

/// The handlers of one event, per callback prefix, in the generated table's
/// order (`event_handlers::handlers`), resolved once.
struct EventTable {
    /// PS sorts these left to right instead of `speedSort` (no draws).
    left_to_right: bool,
    /// `eachEvent` events only have unprefixed `on<Event>` handlers.
    prefixed: bool,
    poke: [Vec<PokeEntry>; 5],
    /// `poke[i]` positions grouped by effect bit, table order kept within a
    /// bit: `by_bit[i][first[i][bit] as usize..][..count[i][bit]]`.
    by_bit: [Vec<u16>; 5],
    first: [[u16; MAX_FX]; 5],
    count: [[u8; MAX_FX]; 5],
    /// Union of the `bit`s in `poke[i]`.
    poke_mask: [FxMask; 5],
    /// The `CONDS` with a pokemon handler in this event (bit i = `CONDS[i]`):
    /// only these are checked on a slot.
    conds: u128,
    /// `on` / `onFoe` / `onAny` side condition handlers.
    side: [Vec<GlobalEntry>; 3],
    /// `on` weather / terrain / pseudo-weather handlers.
    field: Vec<GlobalEntry>,
    /// Union of `poke_mask`, and of the `SIDE_IDS` / `FIELD_IDS` bits with a
    /// handler: an event whose effects are all absent has nothing to sort.
    any_poke: FxMask,
    any_side: u32,
    any_field: u32,
    /// `any_side` as a byte mask over `SideConditions` (`side_bytes`): a
    /// quick test that no side condition with a handler is up.
    side_bytes: [u128; 2],
    /// `any_field` by `Weather` / `Terrain` discriminant, and the
    /// pseudo-weathers (bit i = `FIELD_IDS[11 + i]`).
    field_weather: u64,
    field_terrain: u64,
    field_pseudo: u32,
    /// Callback names, for `PE_DEBUG_EVENTS`.
    poke_names: [&'static str; 5],
    side_names: [&'static str; 3],
}

struct Tables {
    /// Bit of each ability (indexed by `Abilities as usize`), or `NO_BIT`.
    ability_bit: Vec<u16>,
    /// Bit of each held item (indexed by `Items as usize`), or `NO_BIT`.
    item_bit: Vec<u16>,
    /// Effect bit of each `CONDS` entry, or `NO_BIT`.
    cond_fx: [u16; CONDS.len()],
    /// Effect bit of each `STATUS_IDS` entry, or `NO_BIT`.
    status_fx: [u16; 6],
    events: Vec<EventTable>,
}

const NO_BIT: u16 = u16::MAX;

/// Events whose handlers PS sorts left to right instead of `speedSort`
/// (no draws). PS sim/battle.ts runEvent.
const LEFT_TO_RIGHT: [&str; 4] = ["Invulnerability", "TryHit", "DamagingHit", "EntryHazard"];

/// Events run through `eachEvent`: only `on<Event>` handlers, no prefixed ones.
/// PS sim/battle.ts findEventHandlers (`prefixedHandlers`).
const UNPREFIXED: [&str; 5] = ["BeforeTurn", "Update", "Weather", "WeatherChange", "TerrainChange"];

static TABLES: std::sync::LazyLock<Tables> = std::sync::LazyLock::new(build_tables);

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

fn build_tables() -> Tables {
    use crate::engine::abilities::Abilities;
    use std::str::FromStr;
    // Bit of each pokemon effect (kind, PS id) that has a handler in some
    // event table. Effects without one never join a sort.
    let mut fx: std::collections::HashMap<(u8, &'static str), u16> = std::collections::HashMap::new();
    let mut ability_bit: Vec<u16> = Vec::new();
    let mut item_bit: Vec<u16> = Vec::new();
    let mut events: Vec<EventTable> = Vec::new();
    for ev in Event::ALL {
        let name = ev.name();
        let poke_names = POKE_PREFIXES.map(|p| leak(format!("{}{}", p, name)));
        let side_names = [poke_names[ON], poke_names[FOE], poke_names[ANY]];
        let mut poke: [Vec<PokeEntry>; 5] = Default::default();
        let mut poke_mask = [FxMask::ZERO; 5];
        for (pi, cb) in poke_names.iter().enumerate() {
            for h in handlers(cb) {
                if matches!(h.kind, HandlerKind::Side | HandlerKind::Field | HandlerKind::Weather) {
                    continue;
                }
                let next = fx.len();
                let bit = *fx.entry((h.kind as u8, h.id)).or_insert_with(|| {
                    assert!(next < MAX_FX, "more than MAX_FX pokemon effects have event handlers");
                    next as u16
                });
                if h.kind == HandlerKind::Ability {
                    let ab = Abilities::from_str(h.id).unwrap();
                    if ab.to_string().to_lowercase() == h.id {
                        let i = ab as usize;
                        if ability_bit.len() <= i {
                            ability_bit.resize(i + 1, NO_BIT);
                        }
                        ability_bit[i] = bit;
                    }
                }
                if h.kind == HandlerKind::Item {
                    if let Ok(it) = crate::engine::items::Items::from_str(h.id) {
                        let i = it as usize;
                        if item_bit.len() <= i {
                            item_bit.resize(i + 1, NO_BIT);
                        }
                        item_bit[i] = bit;
                    }
                }
                poke_mask[pi] |= FxMask::bit(bit);
                poke[pi].push(PokeEntry {
                    bit,
                    id: h.id,
                    order: if h.order == 0 { NO_ORDER } else { h.order },
                    priority: h.priority,
                    sub_order: h.sub_order,
                    raw_speed: pi == ALLY && ev == Event::TryHitSide && h.id == "magicbounce",
                });
            }
        }
        let global = |cb: &str, ids: &[&str], kinds: &[HandlerKind]| -> Vec<GlobalEntry> {
            handlers(cb)
                .iter()
                .filter(|h| kinds.contains(&h.kind))
                .filter_map(|h| {
                    let idx = ids.iter().position(|&id| id == h.id)?;
                    Some(GlobalEntry {
                        idx: idx as u8,
                        id: h.id,
                        order: if h.order == 0 { NO_ORDER } else { h.order },
                        priority: h.priority,
                        sub_order: h.sub_order,
                    })
                })
                .collect()
        };
        let side = side_names.map(|cb| global(cb, &SIDE_IDS, &[HandlerKind::Side]));
        let field = global(poke_names[ON], &FIELD_IDS, &[HandlerKind::Field, HandlerKind::Weather]);
        let mut by_bit: [Vec<u16>; 5] = Default::default();
        let mut first = [[0u16; MAX_FX]; 5];
        let mut count = [[0u8; MAX_FX]; 5];
        for pi in 0..5 {
            for bit in 0..MAX_FX as u16 {
                first[pi][bit as usize] = by_bit[pi].len() as u16;
                for (k, h) in poke[pi].iter().enumerate() {
                    if h.bit == bit {
                        by_bit[pi].push(k as u16);
                        count[pi][bit as usize] += 1;
                    }
                }
            }
        }
        events.push(EventTable {
            left_to_right: LEFT_TO_RIGHT.contains(&name),
            prefixed: !UNPREFIXED.contains(&name),
            by_bit,
            first,
            count,
            poke,
            poke_mask,
            conds: 0,
            any_poke: FxMask::ZERO,
            any_side: 0,
            any_field: 0,
            side_bytes: [0; 2],
            field_weather: 0,
            field_terrain: 0,
            field_pseudo: 0,
            side,
            field,
            poke_names,
            side_names,
        });
    }
    // Effect bits of the conditions and statuses, and each event's conditions.
    let cond_fx = CONDS.map(|(id, kind)| fx.get(&(kind as u8, id)).copied().unwrap_or(NO_BIT));
    let status_fx = STATUS_IDS.map(|id| fx.get(&(HandlerKind::Status as u8, id)).copied().unwrap_or(NO_BIT));
    for ev in events.iter_mut() {
        let mut all = FxMask::ZERO;
        for &m in &ev.poke_mask {
            all |= m;
        }
        ev.any_poke = all;
        ev.any_side = ev.side.iter().flatten().fold(0, |m, h| m | 1 << h.idx);
        ev.any_field = ev.field.iter().fold(0, |m, h| m | 1 << h.idx);
        let mut bytes = [0u8; 32];
        for i in 0..SIDE_IDS.len() {
            if ev.any_side & (1 << i) != 0 {
                bytes[SIDE_OFFSETS[i]] = 0xff;
            }
        }
        ev.side_bytes = [
            u128::from_ne_bytes(bytes[..16].try_into().unwrap()),
            u128::from_ne_bytes(bytes[16..].try_into().unwrap()),
        ];
        for i in 0..FIELD_IDS.len() {
            if ev.any_field & (1 << i) == 0 {
                continue;
            }
            match i {
                0..=6 => ev.field_weather |= 1 << FIELD_WEATHERS[i] as u32,
                7..=10 => ev.field_terrain |= 1 << FIELD_TERRAINS[i - 7] as u32,
                _ => ev.field_pseudo |= 1 << (i - 11),
            }
        }
        for (i, &bit) in cond_fx.iter().enumerate() {
            if bit != NO_BIT && all.has(bit) {
                ev.conds |= 1 << i;
            }
        }
    }
    Tables { ability_bit, item_bit, cond_fx, status_fx, events }
}

/// The effects of the pokemon in a slot that have a handler somewhere, as
/// bits, in `visit_pokemon_effects` order.
#[derive(Clone, Copy)]
struct SlotEffects {
    bits: [u16; 16],
    n: u8,
    mask: FxMask,
}

fn slot_effects(t: &Tables, state: &DoublesState, side: Side, slot: usize, conds: u128) -> SlotEffects {
    let mut e = SlotEffects { bits: [0; 16], n: 0, mask: FxMask::ZERO };
    let mut add = |bit: u16| {
        if bit != NO_BIT {
            assert!((e.n as usize) < e.bits.len(), "too many pokemon effects with handlers");
            e.bits[e.n as usize] = bit;
            e.n += 1;
            e.mask |= FxMask::bit(bit);
        }
    };
    // Same order as `visit_pokemon_effects`: status, ability, conditions.
    let s = state.side(side);
    let a = &s.active()[slot];
    let Some(idx) = a.pokemon_index else { return e };
    if let Some(i) = status_index(s.team_member(idx).status) {
        add(t.status_fx[i]);
    }
    add(t.ability_bit.get(real_ability(state, side, idx) as usize).copied().unwrap_or(NO_BIT));
    add(t.item_bit.get(s.team_member(idx).item as usize).copied().unwrap_or(NO_BIT));
    if conds != 0 {
        let mut m = cond_mask(s, slot) & conds;
        while m != 0 {
            add(t.cond_fx[m.trailing_zeros() as usize]);
            m &= m - 1;
        }
    }
    e
}

/// One collected handler: PS `comparePriority` keys.
#[derive(Debug, Clone, Copy)]
struct Key {
    order: i64,
    priority: i32,
    speed: i32,
    sub_order: i32,
}

/// Whether any active pokemon, side or the field holds an effect with a
/// handler for the event (any prefix). Most event sorts have none.
fn has_any_handler(t: &Tables, ev: &EventTable, state: &DoublesState) -> bool {
    if ev.any_side != 0 {
        for sd in [Side::One, Side::Two] {
            let b = side_bytes(&state.side(sd).side_conditions);
            if (b[0] & ev.side_bytes[0]) | (b[1] & ev.side_bytes[1]) != 0 && ev.any_side & side_mask(state, sd) != 0 {
                return true;
            }
        }
    }
    if ev.any_field != 0 {
        let pseudo = state.trick_room.active as u32
            | (state.gravity.active as u32) << 1
            | (state.magic_room.active as u32) << 2
            | (state.wonder_room.active as u32) << 3
            | (state.fairy_lock.active as u32) << 4;
        if (ev.field_weather >> (state.weather.weather_type as u32 & 63)) & 1 != 0
            || (ev.field_terrain >> (state.terrain.terrain_type as u32 & 63)) & 1 != 0
            || ev.field_pseudo & pseudo != 0
        {
            return true;
        }
    }
    if ev.any_poke.is_zero() {
        return false;
    }
    fx_union(t, state).intersects(&ev.any_poke)
}

/// `slot_fx` of all four slots, OR-ed; cached in `DoublesState::fx_union`
/// while no slot or team member changed (and no Mold Breaker list is up).
fn fx_union(t: &Tables, state: &DoublesState) -> FxMask {
    let (a, b) = (&state.side_one, &state.side_two);
    let extra = a.future_move[0].is_some() as u8
        | (a.future_move[1].is_some() as u8) << 1
        | (b.future_move[0].is_some() as u8) << 2
        | (b.future_move[1].is_some() as u8) << 3
        | (a.wish[0].is_some() as u8) << 4
        | (a.wish[1].is_some() as u8) << 5
        | (b.wish[0].is_some() as u8) << 6
        | (b.wish[1].is_some() as u8) << 7;
    let cacheable = state.mold_saved.is_empty();
    if cacheable {
        if let Some((va, vb, e, u)) = state.fx_union.get() {
            if va == a.ver && vb == b.ver && e == extra {
                return u;
            }
        }
    }
    let mut u = FxMask::ZERO;
    for side in [Side::One, Side::Two] {
        for slot in 0..2 {
            u |= slot_fx(t, state, side, slot);
        }
    }
    if cacheable {
        state.fx_union.set(Some((a.ver, b.ver, extra, u)));
    }
    u
}

/// The effect bits (`Tables::fx`) of everything the slot's occupant holds:
/// status, ability and conditions. Cached per slot (`DoublesSide::fx_cache`).
fn slot_fx(t: &Tables, state: &DoublesState, side: Side, slot: usize) -> FxMask {
    let s = state.side(side);
    let Some(idx) = s.active()[slot].pokemon_index else { return FxMask::ZERO };
    let status = s.team_member(idx).status;
    let ability = real_ability(state, side, idx);
    let item = s.team_member(idx).item;
    let extra = s.future_move[slot].is_some() as u8 | (s.wish[slot].is_some() as u8) << 1;
    if let Some(c) = s.fx_cache[slot].get() {
        if c.status == status && c.ability == ability && c.item == item && c.extra == extra {
            return c.fx;
        }
    }
    let bit = |b: u16| if b == NO_BIT { FxMask::ZERO } else { FxMask::bit(b) };
    let mut fx = status_index(status).map_or(FxMask::ZERO, |i| bit(t.status_fx[i]));
    fx |= bit(t.ability_bit.get(ability as usize).copied().unwrap_or(NO_BIT));
    fx |= bit(t.item_bit.get(item as usize).copied().unwrap_or(NO_BIT));
    let mut m = cond_mask(s, slot);
    while m != 0 {
        fx |= bit(t.cond_fx[m.trailing_zeros() as usize]);
        m &= m - 1;
    }
    s.fx_cache[slot].set(Some(crate::doubles::state::FxCache { status, ability, item, extra, fx }));
    fx
}

/// Collects the handlers of one `run_event_ties` call. Each slot's effects
/// are read once and reused by every prefix.
struct Collector<'a> {
    t: &'a Tables,
    ev: &'a EventTable,
    state: &'a DoublesState,
    slots: [Option<SlotEffects>; 4],
    hs: smallvec::SmallVec<[Key; 16]>,
    debug: Option<Vec<String>>,
}

impl Collector<'_> {
    fn push_pokemon(&mut self, side: Side, slot: usize, prefix: usize) {
        if self.ev.poke_mask[prefix].is_zero() {
            return;
        }
        let si = side as usize * 2 + slot;
        let (t, state, conds) = (self.t, self.state, self.ev.conds);
        let fx = *self.slots[si].get_or_insert_with(|| slot_effects(t, state, side, slot, conds));
        if !fx.mask.intersects(&self.ev.poke_mask[prefix]) {
            return;
        }
        let speed = cached_speed(self.state, side, slot);
        for &bit in &fx.bits[..fx.n as usize] {
            if !self.ev.poke_mask[prefix].has(bit) {
                continue;
            }
            let (f, n) = (self.ev.first[prefix][bit as usize] as usize, self.ev.count[prefix][bit as usize] as usize);
            for &k in &self.ev.by_bit[prefix][f..f + n] {
                let h = &self.ev.poke[prefix][k as usize];
                let speed = if h.raw_speed {
                    self.state.side(side).slot_pokemon(slot).map(|p| p.speed as i32).unwrap_or(speed)
                } else {
                    speed
                };
                self.hs.push(Key { order: h.order, priority: h.priority, speed, sub_order: h.sub_order });
                if let Some(d) = &mut self.debug {
                    d.push(format!("{} {} ({:?}{})", self.ev.poke_names[prefix], h.id, side, slot));
                }
            }
        }
    }

    fn push_global(&mut self, entries: &[GlobalEntry], active: u32, callback: &str) {
        for h in entries.iter().filter(|h| active & (1 << h.idx) != 0) {
            self.hs.push(Key { order: h.order, priority: h.priority, speed: 0, sub_order: h.sub_order });
            if let Some(d) = &mut self.debug {
                d.push(format!("{} {}", callback, h.id));
            }
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
    event: Event,
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
    event: Event,
    target: (Side, usize),
    source: Option<(Side, usize)>,
    effect_priority: Option<i32>,
) {
    let t: &Tables = &TABLES;
    let ev = &t.events[event as usize];
    if ev.left_to_right || !has_any_handler(t, ev, state) {
        // At most the effect's own handler: a sort of one key draws nothing.
        return;
    }
    let prefixed = ev.prefixed;
    let mut c = Collector {
        t,
        ev,
        state,
        slots: [None; 4],
        hs: smallvec::SmallVec::new(),
        debug: crate::doubles::debug::flags().events.then(Vec::new),
    };
    let (ts, tsl) = target;
    if let Some(priority) = effect_priority {
        c.hs.push(Key { order: NO_ORDER, priority, speed: cached_speed(state, ts, tsl), sub_order: 0 });
        if let Some(d) = &mut c.debug {
            d.push(format!("on{} (move)", event.name()));
        }
    }
    let target_active = state.side(ts).active()[tsl].pokemon_index.is_some();
    let source_active = source.is_some_and(|(s, sl)| state.side(s).active()[sl].pokemon_index.is_some());
    if target_active || source_active {
        c.push_pokemon(ts, tsl, ON);
        if prefixed {
            for sl in 0..2 {
                if alive(state, ts, sl) {
                    c.push_pokemon(ts, sl, ALLY);
                    c.push_pokemon(ts, sl, ANY);
                }
            }
            for sl in 0..2 {
                if alive(state, ts.other(), sl) {
                    c.push_pokemon(ts.other(), sl, FOE);
                    c.push_pokemon(ts.other(), sl, ANY);
                }
            }
        }
    }
    if let Some((ss, ssl)) = source.filter(|_| prefixed) {
        c.push_pokemon(ss, ssl, SOURCE);
    }
    for sd in [Side::One, Side::Two] {
        let active = side_mask(state, sd);
        if active == 0 {
            continue;
        }
        if sd == ts {
            c.push_global(&ev.side[SIDE_ON], active, ev.side_names[SIDE_ON]);
        } else if prefixed {
            c.push_global(&ev.side[SIDE_FOE], active, ev.side_names[SIDE_FOE]);
        }
        if prefixed {
            c.push_global(&ev.side[SIDE_ANY], active, ev.side_names[SIDE_ANY]);
        }
    }
    if !ev.field.is_empty() {
        c.push_global(&ev.field, field_mask(state), ev.poke_names[ON]);
    }
    if let Some(d) = &c.debug {
        if c.hs.len() > 1 {
            eprintln!("EVENT {} target={:?}{} handlers: {}", event.name(), ts, tsl, d.join(", "));
        }
    }
    // A runEvent inside the residual's `fieldEvent` (no `eachEvent` between).
    let label = if state.tl.in_field_event.get() { "speed tie (residual)" } else { "speed tie (move order)" };
    speed_sort_keys(&mut c.hs, rng, label);
}

/// PS `speedSort` (selection sort, shuffling each group of equal keys).
fn speed_sort_keys(hs: &mut [Key], rng: &mut Gen5Rng, label: &'static str) {
    let cmp = |a: &Key, b: &Key| {
        a.order
            .cmp(&b.order)
            .then(b.priority.cmp(&a.priority))
            .then(b.speed.cmp(&a.speed))
            .then(a.sub_order.cmp(&b.sub_order))
    };
    let mut sorted = 0usize;
    // Indices of the smallest keys found so far (reused across rounds).
    let mut next: smallvec::SmallVec<[usize; 16]> = smallvec::SmallVec::new();
    while sorted + 1 < hs.len() {
        next.clear();
        next.push(sorted);
        for i in (sorted + 1)..hs.len() {
            match cmp(&hs[i], &hs[next[0]]) {
                std::cmp::Ordering::Less => {
                    next.clear();
                    next.push(i);
                }
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
                let j = rng.label(label).random_range(i, end);
                hs.swap(i, j);
                i += 1;
            }
        }
        sorted += next.len();
    }
}
