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

/// Moves with the PS `thawsTarget` flag: a hit cures a frozen target
/// (data/conditions.ts `frz.onAfterMoveSecondary`).
pub const THAWS_TARGET: [Choices; 5] = [
    Choices::SCORCHINGSANDS,
    Choices::STEAMERUPTION,
    Choices::SCALD,
    Choices::MATCHAGOTCHA,
    Choices::HYDROSTEAM,
];

/// PS `move.flags.futuremove` (data/moves.ts): the move targets a slot, so
/// a fainted target is kept (sim/pokemon.ts getMoveTargets).
pub fn is_future_move(m: Choices) -> bool {
    matches!(m, Choices::FUTURESIGHT | Choices::DOOMDESIRE)
}

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
    /// PS `target.isGrounded() && !target.isSemiInvulnerable()` (Misty /
    /// Electric Terrain): the engine's own check, which knows Gravity and the
    /// slot's volatiles.
    pub target_grounded: bool,
}

/// Whether `target` cannot get `status` (PS `setStatus` -> `runStatusImmunity`
/// and the ability / field effects that block it).
pub fn immune_to_status(target: &Pokemon, status: &PokemonStatus, ctx: &StatusContext) -> bool {
    // General status immunity
    match target.ability {
        Abilities::SHIELDSDOWN => return target.hp > target.maxhp / 2,
        Abilities::PURIFYINGSALT => return true,
        Abilities::COMATOSE => return true,
        // PS data/abilities.ts leafguard.onSetStatus: only blocks in sun, otherwise
        // falls through to the existing-status check below.
        Abilities::LEAFGUARD if ctx.sun => return true,
        _ => {}
    }

    if target.status != PokemonStatus::NONE || target.hp <= 0 {
        true
    } else if ctx.misty_terrain && ctx.target_grounded {
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
                (ctx.electric_terrain && ctx.target_grounded)
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

/// Eruption / Water Spout / Dragon Energy base power.
/// PS: `floor(150 * hp / maxhp)`, at least 1.
pub fn eruption_power(hp: i16, maxhp: i16) -> f32 {
    ((150 * hp.max(0) as i32 / maxhp.max(1) as i32).max(1)) as f32
}

/// Stored Power / Power Trip base power from the user's stat stages
/// (atk, def, spa, spd, spe, acc, eva). PS: `20 + 20 * positiveBoosts()`.
pub fn stored_power(boosts: [i8; 7]) -> f32 {
    let pos: i32 = boosts.iter().map(|&v| (v as i32).max(0)).sum();
    (20 + 20 * pos) as f32
}

/// Last Respects base power. PS: `50 + 50 * side.totalFainted`.
pub fn last_respects_power(total_fainted: u32) -> f32 {
    50.0 + 50.0 * total_fainted as f32
}

/// Hex / Infernal Parade double their base power on a target with a
/// status (or Comatose). PS: hex / infernalparade `basePowerCallback`.
pub fn hex_boosted(target: &Pokemon) -> bool {
    target.status != PokemonStatus::NONE || target.ability == Abilities::COMATOSE
}

/// Venoshock / Barb Barrage double their base power on a poisoned target.
/// PS: venoshock / barbbarrage `basePowerCallback`.
pub fn venoshock_boosted(target: &Pokemon) -> bool {
    matches!(target.status, PokemonStatus::POISON | PokemonStatus::TOXIC)
}

/// Apply a stat stage to a base stat, matching pokemon-showdown's
/// `calculateStat` stage multipliers (integer, truncating).
pub fn boost_stat(boost: i8, base: i16) -> i16 {
    match boost {
        -6 => base * 2 / 8,
        -5 => base * 2 / 7,
        -4 => base * 2 / 6,
        -3 => base * 2 / 5,
        -2 => base * 2 / 4,
        -1 => base * 2 / 3,
        0 => base,
        1 => base * 3 / 2,
        2 => base * 4 / 2,
        3 => base * 5 / 2,
        4 => base * 6 / 2,
        5 => base * 7 / 2,
        6 => base * 8 / 2,
        _ => base,
    }
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
        Abilities::RATTLED if matches!(t, PokemonType::DARK | PokemonType::BUG | PokemonType::GHOST) => Some(b(0, 0, 1)),
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

/// Abilities with the `failroleplay` flag (PS data/abilities.ts): Role Play
/// fails when the target has one of these (`onTryHit`).
pub fn fail_role_play(a: Abilities) -> bool {
    crate::doubles::ability_flags::has_flag(a, crate::doubles::ability_flags::FAILROLEPLAY)
}

/// Abilities with the `cantsuppress` flag (data/abilities.ts): Entrainment,
/// Mummy, Lingering Aroma and friends can't replace them (`setAbility`).
pub fn cant_suppress(a: Abilities) -> bool {
    crate::doubles::ability_flags::has_flag(a, crate::doubles::ability_flags::CANTSUPPRESS)
}

/// Abilities with the `noentrain` flag (data/abilities.ts): they can't be
/// copied by Entrainment (`onTryHit`).
pub fn no_entrain(a: Abilities) -> bool {
    crate::doubles::ability_flags::has_flag(a, crate::doubles::ability_flags::NOENTRAIN)
}

/// Abilities with the `failskillswap` flag (data/abilities.ts).
pub fn fails_skill_swap(a: Abilities) -> bool {
    crate::doubles::ability_flags::has_flag(a, crate::doubles::ability_flags::FAILSKILLSWAP)
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

/// Terrain `onBasePower` (priority 6) as a 4096-based modifier: Electric /
/// Grassy / Psychic Terrain boost a grounded user's matching moves x1.3;
/// Grassy Terrain halves Earthquake / Bulldoze / Magnitude and Misty Terrain
/// halves Dragon moves against a grounded target. PS grassyterrain /
/// mistyterrain `onBasePower`: a semi-invulnerable defender is not weakened.
pub fn terrain_base_power_modifier(
    terrain: super::state::Terrain,
    choice: &Choice,
    attacker_grounded: bool,
    defender_grounded: bool,
    defender_semi_invulnerable: bool,
) -> Option<i64> {
    use super::state::Terrain;
    let t = choice.move_type;
    match terrain {
        Terrain::ELECTRICTERRAIN if t == PokemonType::ELECTRIC && attacker_grounded => Some(5325),
        Terrain::PSYCHICTERRAIN if t == PokemonType::PSYCHIC && attacker_grounded => Some(5325),
        Terrain::GRASSYTERRAIN
            if matches!(choice.move_id, Choices::EARTHQUAKE | Choices::BULLDOZE | Choices::MAGNITUDE)
                && defender_grounded
                && !defender_semi_invulnerable =>
        {
            Some(2048)
        }
        Terrain::GRASSYTERRAIN if t == PokemonType::GRASS && attacker_grounded => Some(5325),
        // PS mistyterrain `onBasePower`: `defender.isGrounded() && !defender.isSemiInvulnerable()`.
        Terrain::MISTYTERRAIN if t == PokemonType::DRAGON && defender_grounded && !defender_semi_invulnerable => {
            Some(2048)
        }
        _ => None,
    }
}

/// Terrain Pulse (`onModifyType` / `onModifyMove`): on a grounded user in a
/// terrain, the move takes the terrain's type and doubles its base power.
pub fn terrain_pulse_type(terrain: super::state::Terrain, user_grounded: bool) -> Option<PokemonType> {
    use super::state::Terrain;
    if !user_grounded {
        return None;
    }
    match terrain {
        Terrain::ELECTRICTERRAIN => Some(PokemonType::ELECTRIC),
        Terrain::GRASSYTERRAIN => Some(PokemonType::GRASS),
        Terrain::MISTYTERRAIN => Some(PokemonType::FAIRY),
        Terrain::PSYCHICTERRAIN => Some(PokemonType::PSYCHIC),
        Terrain::NONE => None,
    }
}

/// Weather Ball (`onModifyType` / `onModifyMove`): in weather (or with Mega
/// Sol: sun) it becomes the weather's type with 100 base power.
pub fn weather_ball_type(weather: super::state::Weather, mega_sol: bool) -> Option<PokemonType> {
    use super::state::Weather;
    if mega_sol {
        return Some(PokemonType::FIRE);
    }
    match weather {
        Weather::SUN | Weather::HARSHSUN => Some(PokemonType::FIRE),
        Weather::RAIN | Weather::HEAVYRAIN => Some(PokemonType::WATER),
        Weather::SAND => Some(PokemonType::ROCK),
        Weather::HAIL | Weather::SNOW => Some(PokemonType::ICE),
        Weather::NONE | Weather::DELTASTREAM => None,
    }
}

/// -ate abilities (`onModifyType`, gen 7+): Pixilate / Aerilate / Refrigerate
/// / Galvanize turn Normal moves into their type, Normalize turns every move
/// Normal; a changed move gets x1.2 (`typeChangerBoosted`, BasePower
/// priority 23). The singles engine applies the same rule in
/// `abilities::ability_modify_attack_being_used`.
pub fn ate_type_change(ability: Abilities, choice: &Choice) -> Option<PokemonType> {
    let no_modify = matches!(
        choice.move_id,
        Choices::JUDGMENT
            | Choices::MULTIATTACK
            | Choices::NATURALGIFT
            | Choices::REVELATIONDANCE
            | Choices::TECHNOBLAST
            | Choices::TERRAINPULSE
            | Choices::WEATHERBALL
    );
    if no_modify {
        return None;
    }
    let from_normal = |t| if choice.move_type == PokemonType::NORMAL { Some(t) } else { None };
    match ability {
        Abilities::PIXILATE => from_normal(PokemonType::FAIRY),
        Abilities::AERILATE => from_normal(PokemonType::FLYING),
        Abilities::REFRIGERATE => from_normal(PokemonType::ICE),
        Abilities::GALVANIZE => from_normal(PokemonType::ELECTRIC),
        Abilities::DRAGONIZE => from_normal(PokemonType::DRAGON),
        Abilities::NORMALIZE => Some(PokemonType::NORMAL),
        _ => None,
    }
}

/// Gen 8+: a Poison-type user's Toxic always hits, even a semi-invulnerable
/// target, with no accuracy roll (PS `hitStepInvulnerabilityEvent` /
/// `hitStepAccuracy`).
pub fn toxic_always_hits(move_id: Choices, user: &Pokemon) -> bool {
    move_id == Choices::TOXIC && user.has_type(&PokemonType::POISON)
}

/// Moves with the `minimize` flag (PS data/moves.ts): against a Minimized
/// target they skip the accuracy roll (Minimize `onAccuracy`) and deal x2
/// damage (Minimize `onSourceModifyDamage`).
pub fn minimize_move(move_id: Choices) -> bool {
    matches!(
        move_id,
        Choices::STOMP
            | Choices::BODYSLAM
            | Choices::DRAGONRUSH
            | Choices::FLYINGPRESS
            | Choices::HEATCRASH
            | Choices::HEAVYSLAM
            | Choices::STEAMROLLER
            | Choices::SUPERCELLSLAM
    )
}

/// The attacker's ability in PS's `ModifyAtk` / `ModifySpA` step, as a
/// 4096-based modifier (4096 = none): Overgrow / Blaze / Torrent / Swarm
/// (`move.type === X && attacker.hp <= attacker.maxhp / 3` -> x1.5).
// TODO(PS parity): Huge Power, Gorilla Tactics, Hustle, Defeatist, ...
pub fn attacker_stat_modifier(attacker: &Pokemon, choice: &Choice) -> i64 {
    // Huge Power / Pure Power `onModifyAtk`: x2 (physical moves).
    if matches!(attacker.ability, Abilities::HUGEPOWER | Abilities::PUREPOWER) && choice.category == MoveCategory::Physical {
        return 8192;
    }
    // Guts `onModifyAtk`: x1.5 for physical moves while statused (any status).
    // PS data/abilities.ts guts.
    if attacker.ability == Abilities::GUTS
        && attacker.status != PokemonStatus::NONE
        && choice.category == MoveCategory::Physical
    {
        return 6144;
    }
    // Water Bubble `onModifyAtk` / `onModifySpA`: x2 on Water moves (both
    // categories). PS data/abilities.ts waterbubble.
    if attacker.ability == Abilities::WATERBUBBLE && choice.move_type == PokemonType::WATER {
        return 8192;
    }
    // Type `onModifyAtk` / `onModifySpA` (priority 5, both categories):
    // Steelworker / Dragon's Maw / Rocky Payload x1.5, Transistor x1.3 (gen 9).
    // Defeatist x0.5 at half HP or less; Gorilla Tactics x1.5 Atk (priority 1).
    // PS data/abilities.ts.
    if choice.category != MoveCategory::Status {
        let m = match attacker.ability {
            Abilities::STEELWORKER if choice.move_type == PokemonType::STEEL => 6144,
            Abilities::DRAGONSMAW if choice.move_type == PokemonType::DRAGON => 6144,
            Abilities::ROCKYPAYLOAD if choice.move_type == PokemonType::ROCK => 6144,
            Abilities::TRANSISTOR if choice.move_type == PokemonType::ELECTRIC => 5325,
            Abilities::DEFEATIST if (attacker.hp as i32) * 2 <= attacker.maxhp as i32 => 2048,
            Abilities::GORILLATACTICS if choice.category == MoveCategory::Physical => 6144,
            _ => 4096,
        };
        if m != 4096 {
            return m;
        }
    }
    let pinch = (attacker.hp as i32) * 3 <= attacker.maxhp as i32;
    let boosted = match attacker.ability {
        Abilities::OVERGROW => choice.move_type == PokemonType::GRASS,
        Abilities::BLAZE => choice.move_type == PokemonType::FIRE,
        Abilities::TORRENT => choice.move_type == PokemonType::WATER,
        Abilities::SWARM => choice.move_type == PokemonType::BUG,
        _ => false,
    };
    if boosted && pinch {
        6144
    } else {
        4096
    }
}

/// Solar Power `onModifySpA` (priority 5): x1.5 on special moves while Sun
/// (`sunnyday` / `desolateland`) is the effective weather. PS data/abilities.ts
/// solarpower. 4096 = none.
pub fn solar_power_spa_modifier(attacker: &Pokemon, choice: &Choice, sun: bool) -> i64 {
    if attacker.ability == Abilities::SOLARPOWER && choice.category == MoveCategory::Special && sun {
        6144
    } else {
        4096
    }
}

/// The defender's `onSourceModifyAtk` / `onSourceModifySpA` ability modifier
/// (4096 = none): Thick Fat (Fire / Ice), Heatproof and Water Bubble (Fire),
/// Purifying Salt (Ghost). PS data/abilities.ts (priority 6, before the
/// attacker's own handlers).
pub fn defender_source_stat_modifier(defender: &Pokemon, choice: &Choice) -> i64 {
    let hit = match defender.ability {
        Abilities::THICKFAT => matches!(choice.move_type, PokemonType::FIRE | PokemonType::ICE),
        Abilities::HEATPROOF | Abilities::WATERBUBBLE => choice.move_type == PokemonType::FIRE,
        Abilities::PURIFYINGSALT => choice.move_type == PokemonType::GHOST,
        _ => false,
    };
    if hit {
        2048
    } else {
        4096
    }
}

/// The defender's ability in PS's `ModifyDamage` step
/// (`onSourceModifyDamage`), as a 4096-based modifier (4096 = none): Punk
/// Rock / Ice Scales / Multiscale / Shadow Shield / Fluffy.
// TODO(PS parity): Filter / Solid Rock / Prism Armor (need type effectiveness).
pub fn defender_damage_modifier(defender: &Pokemon, choice: &Choice) -> i64 {
    match defender.ability {
        Abilities::PUNKROCK if choice.flags.sound => 2048,
        Abilities::ICESCALES if choice.category == MoveCategory::Special => 2048,
        Abilities::MULTISCALE | Abilities::SHADOWSHIELD if defender.hp >= defender.maxhp => 2048,
        Abilities::FLUFFY => {
            // `mod = 1; Fire x2; contact /2`, applied as one chainModify.
            let mut m = 4096;
            if choice.move_type == PokemonType::FIRE {
                m *= 2;
            }
            if choice.flags.contact {
                m /= 2;
            }
            m
        }
        _ => 4096,
    }
}

/// PS `onSwitchInPriority` of an ability (data/abilities.ts); 0 when unset.
/// Orders its Start against the other incoming pokemon's SwitchIn handlers.
pub fn switch_in_priority(a: Abilities) -> i8 {
    match a {
        Abilities::NEUTRALIZINGGAS | Abilities::TERASHIFT => 2,
        Abilities::ASONEGLASTRIER | Abilities::ASONESPECTRIER | Abilities::KLUTZ | Abilities::UNNERVE => 1,
        Abilities::MIMICRY | Abilities::SCHOOLING | Abilities::SHIELDSDOWN => -1,
        Abilities::COSTAR
        | Abilities::FLOWERGIFT
        | Abilities::FORECAST
        | Abilities::HOSPITALITY
        | Abilities::ICEFACE
        | Abilities::PROTOSYNTHESIS
        | Abilities::QUARKDRIVE => -2,
        _ => 0,
    }
}
