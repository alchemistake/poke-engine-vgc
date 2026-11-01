//! Damage calculation matching pokemon-showdown's `gen9` damage formula exactly
//! (with the `gen9championsvgc2026regmc` modifiers, which do not change the
//! formula itself).
//!
//! Reference: `dist/sim/battle-actions.js` `getDamage` + `modifyDamage`, and
//! `dist/sim/battle.js` `randomizer` / `modify` / `trunc`.
//!
//! The random draws are consumed in the exact order pokemon-showdown makes them:
//!   1. accuracy  -> `random(100)`        (in `useMove`, before damage)
//!   2. critical  -> `random(critMult)`   (in `getDamage`)
//!   3. damage    -> `random(16)`         (in `modifyDamage` -> `randomizer`)
//!
//! Type-immune targets and fixed-damage moves make NO draws at all.

use crate::choices::{Choice, MoveCategory};
use std::borrow::Cow;
use crate::doubles::rng::Gen5Rng;
use crate::engine::state::Weather;
use crate::state::{Pokemon, PokemonType};

pub const CRIT_MULTIPLIER: f64 = 1.5;

/// gen 6+ critical-hit multiplier table: `critMult[critRatio]`.
fn crit_mult(crit_ratio: i32) -> i32 {
    match crit_ratio.clamp(0, 4) {
        0 => 0,
        1 => 24,
        2 => 8,
        3 => 2,
        _ => 1,
    }
}

fn trunc(x: f64) -> i64 {
    x.floor() as i64
}

/// `battle.modify(value, numerator, denominator = 1)`.
pub fn modify(value: i64, numerator: f64, denominator: f64) -> i64 {
    let modifier = trunc(numerator * 4096.0 / denominator);
    let inner = trunc(value as f64 * modifier as f64);
    (inner + 2047) / 4096
}

/// `battle.randomizer(baseDamage)`.
fn randomizer(base_damage: i64, rng: &mut Gen5Rng) -> i64 {
    let r = rng.label("damage roll").random(16);
    let step1 = trunc(base_damage as f64 * (100.0 - r as f64));
    step1 / 100
}

/// Type effectiveness as an integer number of ×2 / ÷2 steps, matching how
/// pokemon-showdown applies it (integer doubling / halving, not a float).
/// Returns `None` when the defender is immune.
thread_local! {
    /// Gravity is up (set by the executing action): every pokemon is grounded.
    static GRAVITY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub fn set_gravity(on: bool) {
    GRAVITY.with(|g| g.set(on));
}

thread_local! {
    /// The running move's user had Scrappy / Mind's Eye when the move started:
    /// `onModifyMove` set `ignoreImmunity` for the whole move, so a Mummy
    /// swap mid-move (Double Hit) does not undo it.
    static SCRAPPY_MOVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub fn set_scrappy_move(on: bool) {
    SCRAPPY_MOVE.with(|g| g.set(on));
}

thread_local! {
    /// The running move's user had Infiltrator when the move started
    /// (`move.infiltrates` from `onModifyMove`): a Mummy swap mid-move does not
    /// undo it.
    static INFILTRATOR_MOVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub fn set_infiltrator_move(on: bool) {
    INFILTRATOR_MOVE.with(|g| g.set(on));
}

pub fn infiltrator_move() -> bool {
    INFILTRATOR_MOVE.with(|g| g.get())
}

thread_local! {
    /// The running move ignores abilities (Mold Breaker family): PS
    /// `suppressingAbility` for any other pokemon, e.g. a dragged-in Levitate
    /// holder meeting the hazards.
    static MOLD_MOVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub fn set_mold_move(on: bool) {
    MOLD_MOVE.with(|g| g.set(on));
}

pub fn mold_move() -> bool {
    MOLD_MOVE.with(|g| g.get())
}

/// PS `isGrounded` for a Ground move against a Smack Down'd holder: the
/// `smackdown` volatile grounds it before the Flying typing and Levitate are
/// checked (data/moves.ts smackdown, sim/pokemon.ts isGrounded). The chart's
/// Ground -> Flying entry is then neutral (sim/dex.ts getEffectiveness), so
/// that view drops the Flying typing and Levitate for this type check only.
pub fn smackdown_ground_view<'a>(
    move_type: &PokemonType,
    defender: &'a Pokemon,
    smackdown: bool,
    item_ignored: bool,
) -> Cow<'a, Pokemon> {
    use crate::engine::abilities::Abilities;
    use crate::engine::items::Items;
    // Ring Target `onNegateImmunity: false` (PS data/items.ts ringtarget): the
    // holder loses its type immunities in `runImmunity` (a Ground move only
    // skips the Flying check of `isGrounded(true)`; Levitate, Air Balloon and
    // Magnet Rise still lift it). Its immune chart entries are neutral in
    // `getEffectiveness`, so the view drops those types.
    if defender.item == Items::RINGTARGET && !item_ignored {
        if let Some(d) = ring_target_view(move_type, defender) {
            return Cow::Owned(smackdown_ground_view(move_type, &d, smackdown, item_ignored).into_owned());
        }
    }
    // An ignored Air Balloon (Klutz, Magic Room) does not lift its holder,
    // nor does an ignored Iron Ball ground it.
    let balloon = (defender.item == Items::AIRBALLOON && (item_ignored || smackdown))
        || (defender.item == Items::IRONBALL && item_ignored);
    let airborne_typing = defender.types.0 == PokemonType::FLYING || defender.types.1 == PokemonType::FLYING;
    let levitates = matches!(defender.ability, Abilities::LEVITATE | Abilities::EELEVATE);
    if *move_type != PokemonType::GROUND || !(balloon || (smackdown && (airborne_typing || levitates))) {
        return Cow::Borrowed(defender);
    }
    let mut d = defender.clone();
    if balloon {
        d.item = Items::NONE;
    }
    if !smackdown {
        return Cow::Owned(d);
    }
    if d.types.0 == PokemonType::FLYING {
        d.types.0 = PokemonType::TYPELESS;
    }
    if d.types.1 == PokemonType::FLYING {
        d.types.1 = PokemonType::TYPELESS;
    }
    if levitates {
        d.ability = Abilities::NONE;
    }
    Cow::Owned(d)
}

/// `defender` with every type immune to `move_type` dropped (TYPELESS), or None
/// when it has no such type. For a Ring Target holder (`smackdown_ground_view`).
pub fn ring_target_view(move_type: &PokemonType, defender: &Pokemon) -> Option<Pokemon> {
    let immune = |t: PokemonType| {
        let mut single = defender.clone();
        single.types = (t, PokemonType::TYPELESS);
        t != PokemonType::TYPELESS && crate::engine::damage_calc::type_effectiveness_modifier(move_type, &single) == 0.0
    };
    let (a, b) = (immune(defender.types.0), immune(defender.types.1));
    if !a && !b {
        return None;
    }
    let mut d = defender.clone();
    if a {
        d.types.0 = PokemonType::TYPELESS;
    }
    if b {
        d.types.1 = PokemonType::TYPELESS;
    }
    Some(d)
}

/// `type_mod` for a given attacker: Scrappy / Mind's Eye let Normal and Fighting
/// moves hit Ghost types (PS data/abilities.ts scrappy.onModifyMove ignoreImmunity).
pub fn type_mod_by(attacker: &Pokemon, attacking_type: &PokemonType, defender: &Pokemon) -> Option<i32> {
    use crate::engine::abilities::Abilities;
    if (matches!(attacker.ability, Abilities::SCRAPPY | Abilities::MINDSEYE) || SCRAPPY_MOVE.with(|g| g.get()))
        && matches!(*attacking_type, PokemonType::NORMAL | PokemonType::FIGHTING)
        && (defender.types.0 == PokemonType::GHOST || defender.types.1 == PokemonType::GHOST)
    {
        let mut d = defender.clone();
        if d.types.0 == PokemonType::GHOST {
            d.types.0 = PokemonType::TYPELESS;
        }
        if d.types.1 == PokemonType::GHOST {
            d.types.1 = PokemonType::TYPELESS;
        }
        return type_mod(attacking_type, &d);
    }
    type_mod(attacking_type, defender)
}

/// `type_mod_by` plus the target's `addedType` (Trick-or-Treat Ghost, Forest's
/// Curse Grass). PS `runEffectiveness` / `runImmunity` multiply the added type's
/// chart entry in as a second type (Normal vs Ghost-added: immune).
/// The hit's `getMoveHitData(move).typeMod` for an item / ability that reads
/// it after the hit (Enigma Berry, Weakness Policy): the chart with the added
/// type, Freeze-Dry's and Flying Press's `onEffectiveness`, Tar Shot.
pub fn hit_type_mod(attacker: &Pokemon, choice: &Choice, defender: &Pokemon, added_type: PokemonType, tar_shot: bool) -> Option<i32> {
    let mut tm = type_mod_with_added(attacker, &choice.move_type, defender, added_type)?;
    if choice.move_id == crate::choices::Choices::FREEZEDRY {
        tm += 2 * [defender.types.0, defender.types.1].iter().filter(|t| **t == PokemonType::WATER).count() as i32;
    }
    if choice.move_id == crate::choices::Choices::FLYINGPRESS {
        tm += crate::engine::damage_calc::type_effectiveness_modifier(&PokemonType::FLYING, defender).log2().round() as i32;
    }
    if tar_shot && choice.move_type == PokemonType::FIRE {
        tm += 1;
    }
    Some(tm)
}

pub fn type_mod_with_added(
    attacker: &Pokemon,
    attacking_type: &PokemonType,
    defender: &Pokemon,
    added_type: PokemonType,
) -> Option<i32> {
    let mut tm = type_mod_by(attacker, attacking_type, defender)?;
    if added_type != PokemonType::TYPELESS {
        let mut with_added = defender.clone();
        with_added.types = (added_type, PokemonType::TYPELESS);
        tm += type_mod_by(attacker, attacking_type, &with_added)?;
    }
    Some(tm)
}

pub fn type_mod(attacking_type: &PokemonType, defender: &Pokemon) -> Option<i32> {
    // Gravity grounds Flying types: a Ground move ignores their Flying typing
    // (the chart's Ground -> Flying 0 is not applied). PS gravity onFieldStart
    // (isGrounded) and the chart entry for the grounded Flying type.
    if *attacking_type == PokemonType::GROUND
        && GRAVITY.with(|g| g.get())
        && (defender.types.0 == PokemonType::FLYING || defender.types.1 == PokemonType::FLYING)
    {
        let mut d = defender.clone();
        if d.types.0 == PokemonType::FLYING {
            d.types.0 = PokemonType::TYPELESS;
        }
        if d.types.1 == PokemonType::FLYING {
            d.types.1 = PokemonType::TYPELESS;
        }
        return type_mod(attacking_type, &d);
    }
    let mult = crate::engine::damage_calc::type_effectiveness_modifier(attacking_type, defender);
    if mult == 0.0 {
        return None;
    }
    // Ground moves miss airborne targets (Levitate, Air Balloon): PS
    // `runImmunity('Ground')` -> `isGrounded`.
    // TODO(PS parity): Gravity, Smack Down / Ingrain, Thousand Arrows.
    if *attacking_type == PokemonType::GROUND && !defender.is_grounded() && !GRAVITY.with(|g| g.get()) {
        return None;
    }
    Some(mult.log2().round() as i32)
}

#[derive(Debug, Clone, Copy)]
pub struct DamageOutcome {
    pub damage: i16,
    pub crit: bool,
    /// The defender ate its resist berry (`onSourceModifyDamage`).
    pub berry: bool,
    /// The defender's Tera Shell made this hit not very effective.
    pub tera_shell: bool,
}

/// Context for a single damage calculation: effective (boosted) stats and the
/// field state, mirroring the values pokemon-showdown feeds into `getDamage` /
/// `modifyDamage`.
#[derive(Debug, Clone, Copy)]
pub struct DamageCtx {
    /// PS `addedType` of the defender (TYPELESS = none). Adds a third type to
    /// the type chart lookup in `calculate_damage`.
    pub added_type: PokemonType,
    /// The defender has the `tarshot` volatile (Fire moves +1 step).
    pub tar_shot: bool,
    /// PS `addedType` of the attacker (TYPELESS = none). `hasType` counts it,
    /// so a Forest's Curse / Trick-or-Treat holder gets STAB on that type.
    pub attacker_added_type: PokemonType,
    /// Base offensive stat (boosts applied in `calculate_damage`; negative
    /// boosts dropped on a crit).
    pub attack_base: i16,
    /// Offensive stat stage.
    pub attack_boost: i8,
    /// Base defensive stat (boosts applied in `calculate_damage`; positive
    /// boosts dropped on a crit).
    pub defense_base: i16,
    /// Defensive stat stage.
    pub defense_boost: i8,
    pub weather: Weather,
    /// Hydro Steam with the attacker's `effectiveWeather()` 'sunnyday': x1.5
    /// instead of sun's Water drop (PS data/conditions.ts sunnyday).
    pub hydro_steam_sun: bool,
    /// Defender side has Reflect active (halves physical damage).
    pub reflect: bool,
    /// Defender side has Light Screen active (halves special damage).
    pub light_screen: bool,
    /// The defender ability's ModifyDamage modifier (4096 = none), from
    /// `mechanics::defender_damage_modifier`.
    pub defender_mod: i64,
    /// Attacker has the Guts ability (burn does not halve its damage).
    pub guts: bool,
    /// The attacker ability's ModifyAtk / ModifySpA modifier (4096 = none),
    /// from `mechanics::attacker_stat_modifier`.
    pub attack_mod: i64,
    /// Defender's Disguise is intact: neutral effectiveness, no critical hit.
    pub disguise: bool,
    /// The defender ability's ModifyDef / ModifySpD modifier (4096 = none).
    pub defense_mod: i64,
    /// Snow (Ice Def) / Sandstorm (Rock SpD): the weather's `this.modify(x, 1.5)`
    /// returns a new stat, which the chained `defense_mod` then scales.
    pub weather_def: bool,
    /// Extra critical-hit stages (Focus Energy +2).
    pub crit_bonus: i32,
    /// Number of screen side conditions (Reflect / Light Screen / Aurora
    /// Veil, both sides) on the field. Each has an `onAnyModifyDamage`
    /// handler; they tie in the `ModifyDamage` event's `speedSort`, which
    /// shuffles (n - 1 draws).
    pub screen_handlers: u8,
    /// An alive ally of the defender has Friend Guard (`onAnyModifyDamage`:
    /// x0.75 on its allies). PS data/abilities.ts friendguard.
    pub friend_guard: bool,
    /// Sizes (>= 2) of the speed-tie groups of Friend Guard holders on the
    /// field, in descending speed order (0 = unused). Every active Friend
    /// Guard is an `onAnyModifyDamage` handler of the ModifyDamage event, even
    /// when it cannot affect the hit; equal-speed holders shuffle.
    pub friend_guard_ties: [u8; 4],
    /// Defender has Filter / Solid Rock / Prism Armor (`onSourceModifyDamage`:
    /// x0.75 on a super-effective hit; PS data/abilities.ts filter).
    pub solid_rock: bool,
    /// Shell Armor / Battle Armor `onCriticalHit` (the crit roll is still drawn).
    pub no_crit: bool,
    /// Attacker has Sniper (`onModifyDamage`: x1.5 on a critical hit).
    pub sniper: bool,
    /// Tinted Lens `onModifyDamage`: x2 on a not-very-effective hit.
    pub tinted_lens: bool,
    /// Effective base power after multiplier effects (Helping Hand);
    /// 0.0 = use `choice.base_power`.
    pub base_power_override: f32,
    /// Attacker's Life Orb (`onModifyDamage` x5324/4096).
    pub life_orb: bool,
    /// Attacker's Expert Belt (`onModifyDamage` x4915/4096 when super effective).
    pub expert_belt: bool,
    /// Defender's resist berry type (Chilan: Normal at any effectiveness), and
    /// whether Ripen doubles it. `None` when it cannot be eaten (Substitute).
    pub resist_berry: Option<(PokemonType, bool)>,
    /// Ripen `onSourceModifyDamage`: a resist berry eaten outside a hit
    /// (`abilityState.berryWeaken`) halves the next damage.
    pub ripen_weaken: bool,
    /// The hit goes through Protect (PS `bypassProtect`): x0.25 after ModifyDamage.
    pub bypass_protect: bool,
    /// Parental Bond's second hit (x0.25 after the spread modifier).
    pub parental_second: bool,
    /// Metronome (item) `onModifyDamage` modifier (4096 = none).
    pub metronome_mod: i64,
    /// Iron Ball `onEffectiveness`: a Ground move at a Flying holder (not
    /// already grounded) is neutral (every type returns 0).
    pub iron_ball_neutral: bool,
    /// Tera Shell (`runEffectiveness`): 1 = the full-HP Terapagos-Terastal
    /// resists a hit that is not already resisted; 2 = a later hit of a move it
    /// already resisted (typeMod -1 whatever the HP).
    pub tera_shell: u8,
}

/// Compute the damage a single-target (or one hit of a spread) move deals to
/// `defender`. Draws the critical and damage rolls from `rng` — in that order,
/// exactly like pokemon-showdown. Returns `None` if the defender is immune
/// (no draws are made in that case).
/// The `runEvent` sorts inside PS `getDamage` the caller draws the handler
/// ties for (`calculate_damage_with`'s hook), in PS order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DamageEvent {
    /// `runEvent('ModifyCritRatio', source, target)`, before the crit roll.
    ModifyCritRatio,
    /// `runEvent('BasePower', source, target, move, bp, true)`, after the crit roll.
    BasePower,
    /// `runEvent('ModifyAtk' / 'ModifySpA', source, target)`, then
    /// `runEvent('ModifyDef' / 'ModifySpD', target, source)`.
    ModifyStats,
    /// `runEvent('ModifyDamage', source, target)` (champions `modifyDamage`).
    ModifyDamage,
}

pub fn calculate_damage(
    attacker: &Pokemon,
    defender: &Pokemon,
    choice: &Choice,
    rng: &mut Gen5Rng,
    spread: bool,
    ctx: &DamageCtx,
) -> Option<DamageOutcome> {
    calculate_damage_with(attacker, defender, choice, rng, spread, ctx, &mut |_, _| false)
}

/// `calculate_damage`, calling `on_event` at each PS `getDamage` `runEvent`
/// (so the caller can draw its handler speed ties at the right place). It
/// returns whether it drew the ties (else the ModifyDamage fallback below).
pub fn calculate_damage_with(
    attacker: &Pokemon,
    defender: &Pokemon,
    choice: &Choice,
    rng: &mut Gen5Rng,
    spread: bool,
    ctx: &DamageCtx,
    on_event: &mut dyn FnMut(DamageEvent, &mut Gen5Rng) -> bool,
) -> Option<DamageOutcome> {
    if choice.category == MoveCategory::Status || choice.category == MoveCategory::Switch {
        return Some(DamageOutcome { damage: 0, crit: false, berry: false, tera_shell: false });
    }
    let base_power = if ctx.base_power_override > 0.0 {
        ctx.base_power_override
    } else {
        choice.base_power
    };
    if base_power == 0.0 {
        return Some(DamageOutcome { damage: 0, crit: false, berry: false, tera_shell: false });
    }

    // Type immunity is checked in getDamage BEFORE any draws.
    // PS `runEffectiveness` / `runImmunity` also check `addedType`.
    let mut tm = type_mod_with_added(attacker, &choice.move_type, defender, ctx.added_type)?;
    // PS data/moves.ts freezedry.onEffectiveness: `if (type === 'Water') return 1;`
    // Each Water defender type is x2 (+1 step) whatever the chart says. Ice vs
    // Water is resisted (-1 step), so the Water part moves from -1 to +1: +2.
    if choice.move_id == crate::choices::Choices::FREEZEDRY {
        let water_types = [defender.types.0, defender.types.1].iter().filter(|t| **t == PokemonType::WATER).count() as i32;
        tm += 2 * water_types;
    }
    // PS data/moves.ts flyingpress.onEffectiveness: `typeMod +
    // getEffectiveness('Flying', type)` for each defender type.
    if choice.move_id == crate::choices::Choices::FLYINGPRESS {
        let m = crate::engine::damage_calc::type_effectiveness_modifier(&PokemonType::FLYING, defender);
        tm += m.log2().round() as i32;
    }
    // tarshot.onEffectiveness: `if (type !== 'Fire') return; return typeMod + 1`.
    if ctx.tar_shot && choice.move_type == PokemonType::FIRE {
        tm += 1;
    }
    if ctx.disguise || ctx.iron_ball_neutral {
        // disguise.onEffectiveness returns 0 (after the immunity check).
        tm = 0;
    }
    // PS sim/pokemon.ts runEffectiveness: Tera Shell returns -1 after the type loop.
    let tera_shell = ctx.tera_shell == 2 || (ctx.tera_shell == 1 && tm >= 0);
    if tera_shell {
        tm = -1;
    }

    let level = attacker.level;
    let is_physical = choice.category == MoveCategory::Physical;

    // critical roll (drawn in getDamage, before the damage roll).
    // Standard moves have critRatio 1 (1/24); crit-boosting moves (1/8) are
    // handled via `crit_ratio`; guaranteed-crit moves skip the roll.
    let _ = on_event(DamageEvent::ModifyCritRatio, rng);
    let cm = crit_mult(choice.crit_ratio + ctx.crit_bonus);
    let mut crit = choice.move_id.guaranteed_crit();
    if !crit && cm > 0 {
        crit = rng.label("crit").random_chance(1, cm as u32);
    }
    // PS `runEvent('BasePower')` (handlers sorted before Fickle Beam's
    // `onBasePower` draws below).
    let _ = on_event(DamageEvent::BasePower, rng);
    if ctx.disguise || ctx.no_crit {
        // disguise / shellarmor onCriticalHit returns false (the roll is still drawn).
        crit = false;
    }

    // Effective stats. A crit ignores negative offensive boosts and positive
    // defensive boosts (PS `getDamage`: `ignoreNegativeOffensive` /
    // `ignorePositiveDefensive`), so the boosts are applied here, after the
    // crit roll.
    let mut atk_b = ctx.attack_boost;
    let mut def_b = ctx.defense_boost;
    if crit {
        if atk_b < 0 {
            atk_b = 0;
        }
        if def_b > 0 {
            def_b = 0;
        }
    }
    let mut attack = crate::engine::mechanics::boost_stat(atk_b, ctx.attack_base).max(1);
    // Hustle `onModifyAtk` (priority 5): `this.modify(atk, 1.5)` applied
    // directly to the stat, before the chained modifiers. PS data/abilities.ts.
    // Body Press runs `ModifyAtk` too (PS getDamage resets `attackStat`).
    if attacker.ability == crate::engine::abilities::Abilities::HUSTLE
        && choice.category == crate::choices::MoveCategory::Physical
    {
        attack = ((attack as i64 * 6144 + 2047) / 4096) as i16;
    }
    // PS `runEvent('ModifyAtk'/'ModifySpA')`: chained modifier applied once.
    if ctx.attack_mod != 4096 {
        attack = ((attack as i64 * ctx.attack_mod + 2047) / 4096) as i16;
    }
    let mut defense = crate::engine::mechanics::boost_stat(def_b, ctx.defense_base).max(1);
    if ctx.weather_def {
        defense = ((defense as i64 * 6144 + 2047) / 4096) as i16;
    }
    if ctx.defense_mod != 4096 {
        defense = ((defense as i64 * ctx.defense_mod + 2047) / 4096) as i16;
    }

    // Fickle Beam (PS `onBasePower`, drawn in the BasePower event after
    // the crit roll): 30% chance of double power.
    let mut base_power = base_power;
    if choice.move_id == crate::choices::Choices::FICKLEBEAM && rng.label("fickle beam").random_chance(3, 10) {
        base_power *= 2.0;
    }
    let _ = on_event(DamageEvent::ModifyStats, rng);

    // base damage
    let level_f = level as f64;
    let s1 = trunc(2.0 * level_f / 5.0 + 2.0);
    let s2 = trunc(s1 as f64 * base_power as f64 * attack as f64);
    let s3 = trunc(s2 as f64 / defense as f64);
    let mut base = trunc(s3 as f64 / 50.0);

    // modifyDamage
    base += 2;
    if spread {
        base = modify(base, 0.75, 1.0);
    } else if ctx.parental_second {
        base = modify(base, 0.25, 1.0);
    }
    // weather modifier (after spread, before crit)
    if ctx.hydro_steam_sun {
        base = modify(base, 1.5, 1.0);
    } else {
    match ctx.weather {
        Weather::RAIN | Weather::HEAVYRAIN => {
            if choice.move_type == PokemonType::WATER {
                base = modify(base, 1.5, 1.0);
            } else if choice.move_type == PokemonType::FIRE {
                base = modify(base, 0.5, 1.0);
            }
        }
        Weather::SUN | Weather::HARSHSUN => {
            if choice.move_type == PokemonType::FIRE {
                base = modify(base, 1.5, 1.0);
            } else if choice.move_type == PokemonType::WATER {
                base = modify(base, 0.5, 1.0);
            }
        }
        _ => {}
    }
    }
    if crit {
        base = trunc(base as f64 * CRIT_MULTIPLIER);
    }
    base = randomizer(base, rng);

    // STAB
    if choice.move_type != PokemonType::TYPELESS {
        // PS `source.hasType(type)`: the attacker's types include its addedType.
        let is_stab = attacker.types.0 == choice.move_type
            || attacker.types.1 == choice.move_type
            || ctx.attacker_added_type == choice.move_type;
        if is_stab {
            // Adaptability `onModifySTAB`: 2 instead of 1.5.
            let stab = if attacker.ability == crate::engine::abilities::Abilities::ADAPTABILITY { 2.0 } else { 1.5 };
            base = modify(base, stab, 1.0);
        }
    }

    // type effectiveness (integer steps)
    if tm > 0 {
        for _ in 0..tm {
            base *= 2;
        }
    } else if tm < 0 {
        for _ in 0..(-tm) {
            base = trunc(base as f64 / 2.0);
        }
    }

    // burn halving for physical moves (Guts prevents it)
    // (Facade ignores it in gen 6+.)
    if attacker.status == crate::state::PokemonStatus::BURN
        && is_physical
        && !ctx.guts
        && choice.move_id != crate::choices::Choices::FACADE
    {
        base = modify(base, 0.5, 1.0);
    }

    // PS `runEvent('ModifyDamage')`: every handler's modifier is chained
    // (`chainModify`) and applied once. Screens are x2732/4096 in doubles
    // and do nothing on a critical hit.
    let mut chain: i64 = 4096;
    let mut chain_in = |m: i64| chain = (chain * m + 2048) >> 12;
    if ctx.defender_mod != 4096 {
        chain_in(ctx.defender_mod);
    }
    if ctx.solid_rock && tm > 0 {
        chain_in(3072);
    }
    if ctx.sniper && crit {
        chain_in(6144);
    }
    if ctx.tinted_lens && tm < 0 {
        chain_in(8192);
    }
    // PS `runEvent('ModifyDamage')` -> `speedSort` over the handlers, drawn by
    // the caller's hook. Fallback (no hook): Friend Guard holders (subOrder 7,
    // speed > 0) sort before the screens (speed 0); each tie group shuffles
    // (group - 1 draws).
    if !on_event(DamageEvent::ModifyDamage, rng) {
        for &g in ctx.friend_guard_ties.iter().filter(|&&g| g > 0) {
            for m in (2..=g as u32).rev() {
                let _ = rng.label("speed tie (move order)").random(m);
            }
        }
        // The screens share order / priority / subOrder (4) / speed, so they shuffle.
        for m in (2..=ctx.screen_handlers as u32).rev() {
            let _ = rng.label("speed tie (move order)").random(m);
        }
    }
    if !crit && ((ctx.reflect && is_physical) || (ctx.light_screen && !is_physical)) {
        chain_in(2732);
    }
    if ctx.friend_guard {
        chain_in(3072);
    }
    // Held items (PS data/items.ts, subOrder 8: after the abilities).
    if ctx.life_orb {
        chain_in(5324);
    }
    if ctx.metronome_mod != 4096 {
        chain_in(ctx.metronome_mod);
    }
    if ctx.expert_belt && tm > 0 {
        chain_in(4915);
    }
    // Resist berries `onSourceModifyDamage`: eaten, x0.5 (x0.25 with Ripen).
    let mut berry = false;
    if let Some((t, ripen)) = ctx.resist_berry {
        if choice.move_type == t && (t == PokemonType::NORMAL || tm > 0) {
            berry = true;
            chain_in(if ripen { 1024 } else { 2048 });
        }
    }
    if ctx.ripen_weaken && !berry {
        chain_in(2048);
    }
    if chain != 4096 {
        base = (base * chain + 2047) / 4096;
    }
    if ctx.bypass_protect {
        base = modify(base, 0.25, 1.0);
    }
    // Minimum 1 damage (gen 6+): after the final ModifyDamage modifiers
    // (Multiscale, screens), before the 16-bit truncation.
    // PS data/mods/champions/scripts.ts modifyDamage (`gen !== 5 && !baseDamage`).
    if base == 0 {
        base = 1;
    }

    // 16-bit trunc
    let result = (base % 65536 + 65536) % 65536;
    if crate::doubles::debug::flags().dmg {
        eprintln!(
            "DMGDBG {} (BP {} {}) vs {}: atk={} def={} base_final={} crit={} tm={:?} type={:?} bp={} ps={} lo={} sp={} scr={}",
            attacker.id, base_power, choice.move_type, defender.id, attack, defense, result, crit, tm, defender.types, ctx.bypass_protect, ctx.parental_second, ctx.life_orb, spread, ctx.screen_handlers
        );
    }
    Some(DamageOutcome {
        damage: result as i16,
        crit,
        berry,
        tera_shell,
    })
}

/// Accuracy roll: `randomChance(accuracy, 100)`. Returns true if the move hits.
///
/// pokemon-showdown draws the roll even at 100% accuracy, but NOT for moves
/// whose accuracy is `true` (always-hit, e.g. Spikes / Protect) — the engine
/// encodes those as `accuracy == 0.0`.
/// PS `hitStepAccuracy`: accuracy scaled by the user's accuracy stage minus
/// the target's evasion stage (each clamped to +-6), then `randomChance`.
// TODO(PS parity): ModifyAccuracy / Accuracy events (Compound Eyes, items, ...).
pub fn roll_accuracy_boosted(
    accuracy: f32,
    acc_boost: i8,
    eva_boost: i8,
    // `onModifyAccuracy` chain pieces from abilities: (priority 0, priority -1),
    // 4096-based (see `accuracy_ability_mods` in battle.rs).
    ability_mods: (i32, i32),
    rng: &mut Gen5Rng,
) -> bool {
    roll_accuracy_boosted_labeled("accuracy", accuracy, acc_boost, eva_boost, ability_mods, rng)
}

/// `roll_accuracy_boosted` with the draw's purpose label. Triple Axel / Triple
/// Kick hits after the first use `multiaccuracy` (PS hitStepMoveHitLoop
/// `randomChance(accuracy, 100)` after the same ModifyAccuracy chain).
pub fn roll_accuracy_boosted_labeled(
    label: &'static str,
    accuracy: f32,
    acc_boost: i8,
    eva_boost: i8,
    ability_mods: (i32, i32),
    rng: &mut Gen5Rng,
) -> bool {
    let mut acc = accuracy as i32;
    // `runEvent('ModifyAccuracy')`: handlers chain, applied once, higher
    // priority first. Gravity `onModifyAccuracy` (priority 0):
    // chainModify([6840, 4096]); abilities (Hustle, Compound Eyes, Sand Veil,
    // Snow Cloak, Tangled Feet) are priority -1.
    let mut m: i32 = 4096;
    if GRAVITY.with(|g| g.get()) {
        m = (m * 6840 + 2048) >> 12;
    }
    m = (m * ability_mods.0 + 2048) >> 12;
    m = (m * ability_mods.1 + 2048) >> 12;
    if m != 4096 {
        acc = (acc * m + 2047) / 4096;
    }
    let boost = (acc_boost.clamp(-6, 6) as i32 - eva_boost as i32).clamp(-6, 6);
    if boost > 0 {
        acc = acc * (3 + boost) / 3;
    } else if boost < 0 {
        acc = acc * 3 / (3 - boost);
    }
    rng.label(label).random_chance(acc.max(0) as u32, 100)
}

/// `multiaccuracy` roll (Triple Axel / Triple Kick / Population Bomb, hits 2+).
/// PS sim/battle-actions.ts hitStepMoveHitLoop: the boost tables are applied as
/// floats (`accuracy *= boostTable[n]`, no truncation), so the float accuracy
/// goes to `randomChance` unrounded. Without ability / Gravity modifiers this is
/// exact; otherwise it falls back to `roll_accuracy_boosted_labeled`.
// TODO(PS parity): ModifyAccuracy handlers (abilities, Gravity) in PS's float chain.
pub fn roll_multiaccuracy(
    accuracy: f32,
    acc_boost: i8,
    eva_boost: i8,
    ability_mods: (i32, i32),
    rng: &mut Gen5Rng,
) -> bool {
    const TABLE: [f64; 7] = [1.0, 4.0 / 3.0, 5.0 / 3.0, 2.0, 7.0 / 3.0, 8.0 / 3.0, 3.0];
    let mut acc = accuracy as f64;
    let b = acc_boost.clamp(-6, 6) as i32;
    if b > 0 {
        acc *= TABLE[b as usize];
    } else {
        acc /= TABLE[(-b) as usize];
    }
    let e = eva_boost.clamp(-6, 6) as i32;
    if e > 0 {
        acc /= TABLE[e as usize];
    } else if e < 0 {
        acc *= TABLE[(-e) as usize];
    }
    // `runEvent('ModifyAccuracy')` applies its chained modifiers (Gravity,
    // Hustle, Compound Eyes, ...) only to a non-negative integer (PS
    // sim/battle.ts runEvent): a fractional accuracy after the boosts keeps
    // its value. PS champions scripts.ts hitStepMoveHitLoop.
    if (GRAVITY.with(|g| g.get()) || ability_mods != (4096, 4096)) && acc == acc.floor() && acc >= 0.0 {
        return roll_accuracy_boosted_labeled("multiaccuracy", accuracy, acc_boost, eva_boost, ability_mods, rng);
    }
    // PS `randomChance(accuracy, 100)` = `random(100) < accuracy` (float compare).
    (rng.label("multiaccuracy").random(100) as f64) < acc
}

pub fn roll_accuracy(accuracy: f32, rng: &mut Gen5Rng) -> bool {
    if accuracy >= 100.0 {
        // pokemon-showdown still draws the roll even at 100%.
        let _ = rng.label("accuracy").random(100);
        return true;
    }
    rng.label("accuracy").random_chance(accuracy as u32, 100)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doubles::state::{compute_stats, Evs};
    use crate::pokemon::PokemonName;
    use crate::state::{
        Move, PokemonMoves, PokemonNature, PokemonStatus, PokemonType as T,
    };

    fn mk_pokemon(
        name: PokemonName,
        t0: T,
        t1: T,
        atk: i16,
        def: i16,
    ) -> Pokemon {
        let mut p = Pokemon::default();
        p.id = name;
        p.level = 50;
        p.types = (t0, t1);
        p.base_types = (t0, t1);
        p.attack = atk;
        p.defense = def;
        p.special_attack = atk;
        p.special_defense = def;
        p.speed = 100;
        p.status = PokemonStatus::NONE;
        p.moves = PokemonMoves {
            m0: Move::default(),
            m1: Move::default(),
            m2: Move::default(),
            m3: Move::default(),
        };
        p
    }

    #[test]
    fn stat_calc_matches_ps() {
        // Gourgeist: base HP 65, 84 effective EV, 31 IV, level 50 -> 151 HP
        let evs = Evs {
            hp: 11,
            atk: 11,
            def: 11,
            spa: 11,
            spd: 11,
            spe: 11,
        };
        let base = (65, 90, 75, 58, 65, 80);
        let (hp, atk, def, spa, spd, spe) = compute_stats(base, &evs, 50, PokemonNature::SERIOUS);
        assert_eq!(hp, 151);
        assert_eq!((atk, def, spa, spd, spe), (121, 106, 89, 96, 111));
    }
}
