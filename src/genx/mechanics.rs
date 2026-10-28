//! Battle mechanics shared by the singles engine (`generate_instructions`,
//! `choice_effects`) and the doubles engine (`src/doubles/`).
//!
//! Functions here take plain values (a `Pokemon`, a move id, a few field
//! facts) instead of an engine's `State`, so both engines can call them.
//! Formulas follow pokemon-showdown (`data/moves.ts`, `data/abilities.ts`,
//! `data/conditions.ts`) exactly, including its integer math.

use super::abilities::Abilities;
use crate::choices::{Choice, Choices, MoveCategory, StatBoosts};
use crate::state::{Pokemon, PokemonBoostableStat, PokemonStatus, PokemonType};

/// Moves with the PS `defrost` flag: usable while frozen, and thaw the user.
pub const THAWS_USER: [Choices; 12] = [
    Choices::FLAMEWHEEL,
    Choices::SACREDFIRE,
    Choices::FLAREBLITZ,
    Choices::FUSIONFLARE,
    Choices::SCALD,
    Choices::STEAMERUPTION,
    Choices::BURNUP,
    Choices::PYROBALL,
    Choices::SCORCHINGSANDS,
    Choices::MATCHAGOTCHA,
    Choices::HYDROSTEAM,
    Choices::SIZZLYSLIDE,
];

/// The field facts `immune_to_status` needs, gathered by each engine from its
/// own state.
pub struct StatusContext {
    /// Sun or harsh sun is active.
    pub sun: bool,
    pub misty_terrain: bool,
    pub electric_terrain: bool,
    /// The target's side has Safeguard.
    pub safeguard: bool,
    /// The target is behind a Substitute.
    pub substitute: bool,
    /// The status comes from an opponent's move (Substitute / Safeguard /
    /// sleep clause only block those).
    pub from_opponent: bool,
    /// Sleep clause: the target's side already has a pokemon asleep.
    pub sleep_clause: bool,
    /// The ability of the pokemon inflicting the status (Corrosion).
    pub source_ability: Abilities,
}

/// Whether `target` cannot get `status` (PS `setStatus` -> `runStatusImmunity`
/// and the ability / field effects that block it).
pub fn immune_to_status(target: &Pokemon, status: &PokemonStatus, ctx: &StatusContext) -> bool {
    // General status immunity
    match target.ability {
        Abilities::SHIELDSDOWN => return target.hp > target.maxhp / 2,
        Abilities::PURIFYINGSALT => return true,
        Abilities::COMATOSE => return true,
        Abilities::LEAFGUARD => return ctx.sun,
        _ => {}
    }

    if target.status != PokemonStatus::NONE || target.hp <= 0 {
        true
    } else if ctx.misty_terrain && target.is_grounded() {
        true
    } else if (ctx.substitute || ctx.safeguard) && ctx.from_opponent {
        // substitute/safeguard don't block if the target is yourself (eg. rest)
        true
    } else {
        // Specific status immunity
        match status {
            PokemonStatus::BURN => {
                target.has_type(&PokemonType::FIRE)
                    || [
                        Abilities::WATERVEIL,
                        Abilities::WATERBUBBLE,
                        Abilities::THERMALEXCHANGE,
                    ]
                    .contains(&target.ability)
            }
            PokemonStatus::FREEZE => {
                target.has_type(&PokemonType::ICE) || target.ability == Abilities::MAGMAARMOR || ctx.sun
            }
            PokemonStatus::SLEEP => {
                (ctx.electric_terrain && target.is_grounded())
                    || [
                        Abilities::INSOMNIA,
                        Abilities::SWEETVEIL,
                        Abilities::VITALSPIRIT,
                    ]
                    .contains(&target.ability)
                    || (ctx.from_opponent && ctx.sleep_clause)
            }

            #[cfg(any(
                feature = "gen6",
                feature = "gen7",
                feature = "gen8",
                feature = "gen9",
                feature = "champions"
            ))]
            PokemonStatus::PARALYZE => {
                target.has_type(&PokemonType::ELECTRIC) || target.ability == Abilities::LIMBER
            }

            #[cfg(any(feature = "gen4", feature = "gen5"))]
            PokemonStatus::PARALYZE => target.ability == Abilities::LIMBER,

            PokemonStatus::POISON | PokemonStatus::TOXIC => {
                ((target.has_type(&PokemonType::POISON) || target.has_type(&PokemonType::STEEL))
                    && ctx.source_ability != Abilities::CORROSION)
                    || [Abilities::IMMUNITY, Abilities::PASTELVEIL].contains(&target.ability)
            }
            _ => false,
        }
    }
}

/// PS `getWeight()` in hectograms (at least 1): Heavy Metal doubles it, Light
/// Metal halves it (truncated). `ignore_ability` for Mold Breaker and co.
// TODO(PS parity): Float Stone, Autotomize.
pub fn weight_hg(pkmn: &Pokemon, ignore_ability: bool) -> i32 {
    let hg = (pkmn.weight_kg * 10.0).round() as i32;
    let hg = if ignore_ability {
        hg
    } else {
        match pkmn.ability {
            Abilities::HEAVYMETAL => hg * 2,
            Abilities::LIGHTMETAL => hg / 2,
            _ => hg,
        }
    };
    hg.max(1)
}

/// Grass Knot / Low Kick base power from the target's weight (hectograms).
pub fn grass_knot_power(target_hg: i32) -> f32 {
    match target_hg {
        w if w >= 2000 => 120.0,
        w if w >= 1000 => 100.0,
        w if w >= 500 => 80.0,
        w if w >= 250 => 60.0,
        w if w >= 100 => 40.0,
        _ => 20.0,
    }
}

/// Heavy Slam / Heat Crash base power from user and target weight.
pub fn heavy_slam_power(user_hg: i32, target_hg: i32) -> f32 {
    if user_hg >= target_hg * 5 {
        120.0
    } else if user_hg >= target_hg * 4 {
        100.0
    } else if user_hg >= target_hg * 3 {
        80.0
    } else if user_hg >= target_hg * 2 {
        60.0
    } else {
        40.0
    }
}

/// Electro Ball base power from `getStat('spe')` of user and target.
pub fn electro_ball_power(user_spe: i32, target_spe: i32) -> f32 {
    let ratio = if target_spe > 0 { (user_spe / target_spe) as usize } else { 0 };
    [40.0, 60.0, 80.0, 120.0, 150.0][ratio.min(4)]
}

/// Gyro Ball base power: `floor(25 * target / user) + 1`, at most 150.
pub fn gyro_ball_power(user_spe: i32, target_spe: i32) -> f32 {
    if user_spe <= 0 {
        return 1.0;
    }
    ((25 * target_spe / user_spe) + 1).min(150) as f32
}

/// Reversal / Flail base power: `max(floor(hp * 48 / maxhp), 1)` thresholds.
pub fn reversal_power(hp: i16, maxhp: i16) -> f32 {
    let ratio = ((hp.max(0) as i32 * 48) / (maxhp.max(1) as i32)).max(1);
    match ratio {
        r if r < 2 => 200.0,
        r if r < 5 => 150.0,
        r if r < 10 => 100.0,
        r if r < 17 => 80.0,
        r if r < 33 => 40.0,
        _ => 20.0,
    }
}

/// Base power that scales with the target's remaining HP: Hard Press (max
/// 100), Crush Grip / Wring Out (max 120).
/// PS: `floor(floor((max * (100 * floor(hp * 4096 / maxHP)) + 2047) / 4096) / 100) || 1`.
pub fn target_hp_power(max_power: i64, hp: i16, maxhp: i16) -> f32 {
    if maxhp <= 0 {
        return max_power as f32;
    }
    let frac = (hp.max(0) as i64 * 4096) / maxhp as i64;
    let bp = ((max_power * (100 * frac) + 2047) / 4096) / 100;
    bp.max(1) as f32
}

/// Defiant / Competitive (`onAfterEachBoost`): the stat raised by +2 when a
/// foe lowers any of the holder's stats.
pub fn stat_drop_reaction(ability: Abilities) -> Option<(PokemonBoostableStat, i8)> {
    match ability {
        Abilities::DEFIANT => Some((PokemonBoostableStat::Attack, 2)),
        Abilities::COMPETITIVE => Some((PokemonBoostableStat::SpecialAttack, 2)),
        _ => None,
    }
}

/// Stat changes the defender's ability gives itself when a move hits it
/// (PS `onDamagingHit`): Weak Armor, Steam Engine, Thermal Exchange, Water
/// Compaction, Justified.
pub fn on_damaging_hit_boosts(defender_ability: Abilities, choice: &Choice) -> Option<StatBoosts> {
    let b = |attack, defense, speed| StatBoosts {
        attack,
        defense,
        special_attack: 0,
        special_defense: 0,
        speed,
        accuracy: 0,
        evasion: 0,
    };
    let t = choice.move_type;
    match defender_ability {
        #[cfg(any(feature = "champions", feature = "gen9", feature = "gen8", feature = "gen7"))]
        Abilities::WEAKARMOR if choice.category == MoveCategory::Physical => Some(b(0, -1, 2)),
        #[cfg(any(feature = "gen6", feature = "gen5", feature = "gen4"))]
        Abilities::WEAKARMOR if choice.category == MoveCategory::Physical => Some(b(0, -1, 1)),
        Abilities::STEAMENGINE if t == PokemonType::WATER || t == PokemonType::FIRE => Some(b(0, 0, 6)),
        Abilities::THERMALEXCHANGE if t == PokemonType::FIRE => Some(b(1, 0, 0)),
        #[cfg(any(feature = "champions", feature = "gen9", feature = "gen8", feature = "gen7"))]
        Abilities::WATERCOMPACTION if t == PokemonType::WATER => Some(b(0, 2, 0)),
        Abilities::JUSTIFIED if t == PokemonType::DARK => Some(b(1, 0, 0)),
        _ => None,
    }
}

/// An ability's `onFractionalPriority` (a tie-breaker inside a priority
/// bracket, in tenths).
pub enum FractionalPriority {
    /// Always this many tenths (Stall -1, Mycelium Might -1 on status moves).
    Fixed(i8),
    /// Quick Draw: +1 tenth with chance `3/10` on non-status moves; the
    /// engine rolls it (PS `randomChance(3, 10)`).
    QuickDraw,
}

pub fn fractional_priority(ability: Abilities, category: MoveCategory) -> Option<FractionalPriority> {
    match ability {
        Abilities::STALL => Some(FractionalPriority::Fixed(-1)),
        Abilities::MYCELIUMMIGHT if category == MoveCategory::Status => Some(FractionalPriority::Fixed(-1)),
        Abilities::QUICKDRAW if category != MoveCategory::Status => Some(FractionalPriority::QuickDraw),
        _ => None,
    }
}

/// Abilities with the `failskillswap` flag (data/abilities.ts).
pub fn fails_skill_swap(a: Abilities) -> bool {
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
            | Abilities::HUNGERSWITCH
            | Abilities::ICEFACE
            | Abilities::ILLUSION
            | Abilities::MULTITYPE
            | Abilities::NEUTRALIZINGGAS
            | Abilities::POISONPUPPETEER
            | Abilities::POWERCONSTRUCT
            | Abilities::PROTOSYNTHESIS
            | Abilities::QUARKDRIVE
            | Abilities::RKSSYSTEM
            | Abilities::SCHOOLING
            | Abilities::SHIELDSDOWN
            | Abilities::STANCECHANGE
            | Abilities::TERAFORMZERO
            | Abilities::TERASHELL
            | Abilities::TERASHIFT
            | Abilities::WONDERGUARD
            | Abilities::ZENMODE
            | Abilities::ZEROTOHERO
    )
}

/// Weather or terrain an ability sets when its holder switches in (same
/// abilities as the singles `ability_on_switch_in`). The duration is
/// `abilities::WEATHER_ABILITY_TURNS` for weather and 5 turns for terrain.
pub enum SwitchInField {
    Weather(super::state::Weather),
    Terrain(super::state::Terrain),
}

pub fn switch_in_field(ability: Abilities) -> Option<SwitchInField> {
    use super::state::{Terrain, Weather};
    Some(match ability {
        Abilities::DRIZZLE => SwitchInField::Weather(Weather::RAIN),
        Abilities::DROUGHT | Abilities::ORICHALCUMPULSE => SwitchInField::Weather(Weather::SUN),
        Abilities::SANDSTREAM => SwitchInField::Weather(Weather::SAND),
        #[cfg(any(feature = "gen9", feature = "champions"))]
        Abilities::SNOWWARNING => SwitchInField::Weather(Weather::SNOW),
        #[cfg(not(any(feature = "gen9", feature = "champions")))]
        Abilities::SNOWWARNING => SwitchInField::Weather(Weather::HAIL),
        Abilities::ELECTRICSURGE | Abilities::HADRONENGINE => SwitchInField::Terrain(Terrain::ELECTRICTERRAIN),
        Abilities::PSYCHICSURGE => SwitchInField::Terrain(Terrain::PSYCHICTERRAIN),
        Abilities::GRASSYSURGE => SwitchInField::Terrain(Terrain::GRASSYTERRAIN),
        Abilities::MISTYSURGE => SwitchInField::Terrain(Terrain::MISTYTERRAIN),
        _ => return None,
    })
}

/// What a defender's ability does to an attacker whose move makes contact
/// (PS `onDamagingHit` + `checkMoveMakesContact`). The singles engine
/// approximates these in `abilities::ability_modify_attack_against`.
pub enum ContactAbility {
    /// Static / Flame Body / Poison Point: `randomChance(3, 10)`, then the
    /// status on the attacker.
    Status30(PokemonStatus),
    /// Effect Spore: if the attacker is not powder-immune, `random(100)`:
    /// < 11 sleep, < 21 paralysis, < 30 poison.
    EffectSpore,
    /// Rough Skin / Iron Barbs: the attacker loses 1/8 of its max HP.
    Recoil8,
}

pub fn contact_ability(defender_ability: Abilities) -> Option<ContactAbility> {
    match defender_ability {
        Abilities::STATIC => Some(ContactAbility::Status30(PokemonStatus::PARALYZE)),
        Abilities::FLAMEBODY => Some(ContactAbility::Status30(PokemonStatus::BURN)),
        Abilities::POISONPOINT => Some(ContactAbility::Status30(PokemonStatus::POISON)),
        Abilities::EFFECTSPORE => Some(ContactAbility::EffectSpore),
        Abilities::ROUGHSKIN | Abilities::IRONBARBS => Some(ContactAbility::Recoil8),
        _ => None,
    }
}

/// PS `runStatusImmunity('powder')`: Grass types and Overcoat.
// TODO(PS parity): Safety Goggles (items are not used by the fuzzer yet).
pub fn powder_immune(p: &Pokemon) -> bool {
    p.has_type(&PokemonType::GRASS) || p.ability == Abilities::OVERCOAT
}
