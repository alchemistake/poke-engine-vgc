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
fn modify(value: i64, numerator: f64, denominator: f64) -> i64 {
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
pub fn type_mod(attacking_type: &PokemonType, defender: &Pokemon) -> Option<i32> {
    let mult = crate::engine::damage_calc::type_effectiveness_modifier(attacking_type, defender);
    if mult == 0.0 {
        return None;
    }
    // Ground moves miss airborne targets (Levitate, Air Balloon): PS
    // `runImmunity('Ground')` -> `isGrounded`.
    // TODO(PS parity): Gravity, Smack Down / Ingrain, Thousand Arrows.
    if *attacking_type == PokemonType::GROUND && !defender.is_grounded() {
        return None;
    }
    Some(mult.log2().round() as i32)
}

#[derive(Debug, Clone, Copy)]
pub struct DamageOutcome {
    pub damage: i16,
    pub crit: bool,
}

/// Context for a single damage calculation: effective (boosted) stats and the
/// field state, mirroring the values pokemon-showdown feeds into `getDamage` /
/// `modifyDamage`.
#[derive(Debug, Clone, Copy)]
pub struct DamageCtx {
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
    /// Effective base power after multiplier effects (Helping Hand);
    /// 0.0 = use `choice.base_power`.
    pub base_power_override: f32,
}

/// Compute the damage a single-target (or one hit of a spread) move deals to
/// `defender`. Draws the critical and damage rolls from `rng` — in that order,
/// exactly like pokemon-showdown. Returns `None` if the defender is immune
/// (no draws are made in that case).
pub fn calculate_damage(
    attacker: &Pokemon,
    defender: &Pokemon,
    choice: &Choice,
    rng: &mut Gen5Rng,
    spread: bool,
    ctx: &DamageCtx,
) -> Option<DamageOutcome> {
    if choice.category == MoveCategory::Status || choice.category == MoveCategory::Switch {
        return Some(DamageOutcome { damage: 0, crit: false });
    }
    let base_power = if ctx.base_power_override > 0.0 {
        ctx.base_power_override
    } else {
        choice.base_power
    };
    if base_power == 0.0 {
        return Some(DamageOutcome { damage: 0, crit: false });
    }

    // Type immunity is checked in getDamage BEFORE any draws.
    let tm = type_mod(&choice.move_type, defender)?;

    let level = attacker.level;
    let is_physical = choice.category == MoveCategory::Physical;

    // critical roll (drawn in getDamage, before the damage roll).
    // Standard moves have critRatio 1 (1/24); crit-boosting moves (1/8) are
    // handled via `crit_ratio`; guaranteed-crit moves skip the roll.
    let cm = crit_mult(choice.crit_ratio);
    let mut crit = choice.move_id.guaranteed_crit();
    if !crit && cm > 0 {
        crit = rng.label("crit").random_chance(1, cm as u32);
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
    // PS `runEvent('ModifyAtk'/'ModifySpA')`: chained modifier applied once.
    if ctx.attack_mod != 4096 {
        attack = ((attack as i64 * ctx.attack_mod + 2047) / 4096) as i16;
    }
    let defense = crate::engine::mechanics::boost_stat(def_b, ctx.defense_base).max(1);

    // Fickle Beam (PS `onBasePower`, drawn in the BasePower event after
    // the crit roll): 30% chance of double power.
    let mut base_power = base_power;
    if choice.move_id == crate::choices::Choices::FICKLEBEAM && rng.label("fickle beam").random_chance(3, 10) {
        base_power *= 2.0;
    }

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
    }
    // weather modifier (after spread, before crit)
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
    if crit {
        base = trunc(base as f64 * CRIT_MULTIPLIER);
    }
    base = randomizer(base, rng);

    // STAB
    if choice.move_type != PokemonType::TYPELESS {
        let is_stab = attacker.types.0 == choice.move_type || attacker.types.1 == choice.move_type;
        if is_stab {
            base = modify(base, 1.5, 1.0);
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

    // Minimum 1 damage (gen 6+): checked before the final ModifyDamage
    // modifiers (Reflect / Light Screen) and the 16-bit truncation.
    if base == 0 {
        base = 1;
    }
    // PS `runEvent('ModifyDamage')`: every handler's modifier is chained
    // (`chainModify`) and applied once. Screens are x2732/4096 in doubles
    // and do nothing on a critical hit.
    let mut chain: i64 = 4096;
    let mut chain_in = |m: i64| chain = (chain * m + 2048) >> 12;
    if ctx.defender_mod != 4096 {
        chain_in(ctx.defender_mod);
    }
    if !crit && ((ctx.reflect && is_physical) || (ctx.light_screen && !is_physical)) {
        chain_in(2732);
    }
    if chain != 4096 {
        base = (base * chain + 2047) / 4096;
    }

    // 16-bit trunc
    let result = (base % 65536 + 65536) % 65536;
    if std::env::var("PE_DEBUG_DMG").is_ok() {
        eprintln!(
            "DMGDBG {} (BP {} {}) vs {}: atk={} def={} base_final={} crit={} tm={:?} type={:?}",
            attacker.id, base_power, choice.move_type, defender.id, attack, defense, result, crit, tm, defender.types
        );
    }
    Some(DamageOutcome {
        damage: result as i16,
        crit,
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
pub fn roll_accuracy_boosted(accuracy: f32, acc_boost: i8, eva_boost: i8, rng: &mut Gen5Rng) -> bool {
    let mut acc = accuracy as i32;
    let boost = (acc_boost.clamp(-6, 6) as i32 - eva_boost as i32).clamp(-6, 6);
    if boost > 0 {
        acc = acc * (3 + boost) / 3;
    } else if boost < 0 {
        acc = acc * 3 / (3 - boost);
    }
    rng.label("accuracy").random_chance(acc.max(0) as u32, 100)
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
