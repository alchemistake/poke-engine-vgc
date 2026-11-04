//! Static behaviour bitsets for moves (and later abilities / items).
//!
//! Every bit is a *superset* of the trigger condition of the block family it
//! guards in `execute_move_body`: if a block can do something for a move, that
//! move has the bit. A plain damaging move has no bits set, so the guarded
//! runs are skipped with one `mb & X` test each.
//!
//! The lists are built from the engine code (the exact moves the guarded
//! blocks test for), not from PS data. Field-derived parts (e.g. "has a
//! `status` effect") come from the engine's own `MOVES` table.

use crate::choices::{Choices, MOVES};
use crate::engine::state::PokemonVolatileStatus;

/// What a move can need beyond the plain damaging path (one bit per stage of
/// `execute_move_body` that the move may enter).
pub(crate) type MoveBits = u64;

/// Pre-hit checks: `onTry` / `onTryHit` / `onPrepareHit` / `basePowerCallback`
/// failures and per-move tweaks before the hit loop (Round, Shell Trap,
/// Stockpile family, Counter, Fling, pledges, protect-likes, ...).
pub(crate) const MB_TRY: MoveBits = 1 << 0;

/// The per-target blocks inside the hit loop: status / volatile / side
/// condition / weather / terrain / pseudo-weather application and the
/// `ti == 0` status-move effects (Skill Swap, Defog, Haze, Sleep Talk, ...).
pub(crate) const MB_HIT: MoveBits = 1 << 1;

/// Post-hit blocks after the hit loops: recoil-style cleanup and
/// `onAfterMove` / `selfSwitch` steps (Spit Up, pivots, Relic Song, ...).
pub(crate) const MB_POST: MoveBits = 1 << 2;

static MOVE_BITS: std::sync::LazyLock<Vec<MoveBits>> = std::sync::LazyLock::new(|| {
    (0..=u16::MAX)
        .map(Choices::from)
        .enumerate()
        .take_while(|&(i, m)| m as usize == i)
        .map(|(_, m)| move_bits(m))
        .collect()
});

/// The single source of truth: one `match` per bit, listing exactly the moves
/// the guarded blocks test for (plus the moves carrying the fields those
/// blocks read).
fn move_bits(m: Choices) -> MoveBits {
    let mut b = 0;
    if matches!(
        m,
        Choices::ROUND
            | Choices::SHELLTRAP
            | Choices::POLLENPUFF
            | Choices::DOUBLESHOCK
            | Choices::BURNUP
            | Choices::STOCKPILE
            | Choices::SNORE
            | Choices::SLEEPTALK
            | Choices::POLTERGEIST
            | Choices::HEALINGWISH
            | Choices::LUNARDANCE
            | Choices::SUCKERPUNCH
            | Choices::THUNDERCLAP
            | Choices::LASTRESORT
            | Choices::SPITUP
            | Choices::SWALLOW
            | Choices::TELEPORT
            | Choices::AURAWHEEL
            | Choices::DARKVOID
            | Choices::HYPERSPACEFURY
            | Choices::ALLYSWITCH
            | Choices::QUICKGUARD
            | Choices::WIDEGUARD
            | Choices::CRAFTYSHIELD
            | Choices::MATBLOCK
            | Choices::COUNTER
            | Choices::MIRRORCOAT
            | Choices::METALBURST
            | Choices::COMEUPPANCE
            | Choices::REST
            | Choices::FLING
            | Choices::ECHOEDVOICE
            | Choices::NATURALGIFT
            | Choices::ASSURANCE
            | Choices::PURSUIT
            | Choices::PAYBACK
            | Choices::FISHIOUSREND
            | Choices::BOLTBEAK
            | Choices::FUTURESIGHT
            | Choices::DOOMDESIRE
            | Choices::UPPERHAND
            | Choices::NORETREAT
            | Choices::STUFFCHEEKS
            | Choices::BELCH
            | Choices::AURORAVEIL
            | Choices::FIREPLEDGE
            | Choices::GRASSPLEDGE
            | Choices::WATERPLEDGE
            // Damp `onAnyTryMove` explosion block.
            | Choices::EXPLOSION
            | Choices::MINDBLOWN
            | Choices::MISTYEXPLOSION
            | Choices::SELFDESTRUCT
    ) {
        b |= MB_TRY;
    }
    if matches!(
        m,
        Choices::ATTRACT
            | Choices::DISABLE
            | Choices::ENCORE
            | Choices::HEALBLOCK
            | Choices::TAUNT
            | Choices::TORMENT
            | Choices::SKILLSWAP
            | Choices::ENTRAINMENT
            | Choices::PSYCHUP
            | Choices::UPROAR
            | Choices::NORETREAT
            | Choices::CHARGE
            | Choices::CURSE
            | Choices::CLANGOROUSSOUL
            | Choices::HEALPULSE
            | Choices::FLORALHEALING
            | Choices::REFRESH
            | Choices::CAMOUFLAGE
            | Choices::CONVERSION
            | Choices::RAGE
            | Choices::DESTINYBOND
            | Choices::IMPRISON
            | Choices::SPITE
            | Choices::DEFOG
            | Choices::ACUPRESSURE
            | Choices::ROLEPLAY
            | Choices::DOODLE
            | Choices::STRENGTHSAP
            | Choices::PERISHSONG
            | Choices::ROOST
            | Choices::FORESTSCURSE
            | Choices::TRICKORTREAT
            | Choices::QUASH
            | Choices::AFTERYOU
            | Choices::MIMIC
            | Choices::SKETCH
            | Choices::INSTRUCT
            | Choices::MAGNETICFLUX
            | Choices::GEARUP
            | Choices::ROTOTILLER
            | Choices::FLOWERSHIELD
            | Choices::DRAGONCHEER
            | Choices::GUARDSPLIT
            | Choices::POWERSPLIT
            | Choices::TRANSFORM
            | Choices::CORROSIVEGAS
            | Choices::TRICK
            | Choices::SWITCHEROO
            | Choices::NIGHTMARE
            | Choices::BESTOW
            | Choices::GASTROACID
            | Choices::REVIVALBLESSING
            | Choices::SHEDTAIL
            | Choices::POWERTRICK
            | Choices::AUTOTOMIZE
            | Choices::POWERSHIFT
            | Choices::ELECTRIFY
            | Choices::BELLYDRUM
            | Choices::FILLETAWAY
            | Choices::SOAK
            | Choices::MAGICPOWDER
            | Choices::CONVERSION2
            | Choices::TOPSYTURVY
            | Choices::SIMPLEBEAM
            | Choices::WORRYSEED
            | Choices::COREENFORCER
            | Choices::SMELLINGSALTS
            | Choices::WAKEUPSLAP
            | Choices::PSYCHOSHIFT
            | Choices::PURIFY
            | Choices::IONDELUGE
            | Choices::PLASMAFISTS
            | Choices::THOUSANDWAVES
            | Choices::CLEARSMOG
            | Choices::FLAMEBURST
            | Choices::RECYCLE
            | Choices::LEECHSEED
            | Choices::REST
            | Choices::SLEEPTALK
            | Choices::NATUREPOWER
            | Choices::ASSIST
            | Choices::METRONOME
            | Choices::MIRRORMOVE
            | Choices::MEFIRST
            | Choices::COPYCAT
            | Choices::TIDYUP
            | Choices::TEATIME
            | Choices::STUFFCHEEKS
            | Choices::HAZE
            | Choices::GUARDSWAP
            | Choices::POWERSWAP
            | Choices::HEARTSWAP
            | Choices::WISH
            | Choices::ALLYSWITCH
            | Choices::HEALINGWISH
            | Choices::LUNARDANCE
            | Choices::LOCKON
            | Choices::MINDREADER
            | Choices::PAINSPLIT
            | Choices::HELPINGHAND
            // Side-condition hazard `onAfterHit` guard.
            | Choices::STONEAXE
            | Choices::CEASELESSEDGE
            | Choices::COURTCHANGE
            | Choices::GRAVITY
            | Choices::MAGICROOM
            | Choices::MUDSPORT
            | Choices::WATERSPORT
            | Choices::FAIRYLOCK
            | Choices::WONDERROOM
            // Weather moves the upstream table has no `weather_id` for (the
            // engine's `weather_id` match special-cases them).
            | Choices::RAINDANCE
            | Choices::SANDSTORM
            | Choices::SNOWSCAPE
            | Choices::CHILLYRECEPTION
    ) {
        b |= MB_HIT;
    }
    if matches!(
        m,
        Choices::PRESENT
            | Choices::FIREPLEDGE
            | Choices::GRASSPLEDGE
            | Choices::WATERPLEDGE
            | Choices::DOUBLESHOCK
            | Choices::BURNUP
            | Choices::STEELROLLER
            | Choices::SPARKLINGARIA
            | Choices::CLANGINGSCALES
            | Choices::HEALINGWISH
            | Choices::LUNARDANCE
            | Choices::RAPIDSPIN
            | Choices::MORTALSPIN
            | Choices::ICESPINNER
            | Choices::JAWLOCK
            | Choices::SPEEDSWAP
            | Choices::FELLSTINGER
            | Choices::ORDERUP
            | Choices::SPITUP
            // `selfSwitch` steps.
            | Choices::UTURN
            | Choices::VOLTSWITCH
            | Choices::FLIPTURN
            | Choices::TELEPORT
            | Choices::PARTINGSHOT
            | Choices::CHILLYRECEPTION
            | Choices::REVIVALBLESSING
            | Choices::SHEDTAIL
            | Choices::BATONPASS
            | Choices::RELICSONG
    ) {
        b |= MB_POST;
    }

    // Field-derived: the guarded blocks also read these move fields.
    if let Some(c) = MOVES.get(&m) {
        // `Steel Roller onTry` (`requires_terrain`) and the stalling-moves
        // (`protect`-like volatile) block in the pre-hit checks.
        if c.requires_terrain {
            b |= MB_TRY;
        }
        if let Some(v) = &c.volatile_status {
            if matches!(
                v.volatile_status,
                PokemonVolatileStatus::PROTECT
                    | PokemonVolatileStatus::ENDURE
                    | PokemonVolatileStatus::KINGSSHIELD
                    | PokemonVolatileStatus::SPIKYSHIELD
                    | PokemonVolatileStatus::BANEFULBUNKER
                    | PokemonVolatileStatus::SILKTRAP
                    | PokemonVolatileStatus::BURNINGBULWARK
            ) {
                b |= MB_TRY;
            }
        }
        // Hit-loop blocks keyed on move-effect fields.
        if c.status.is_some()
            || c.volatile_status.is_some()
            || c.side_condition.is_some()
            || c.force_switch
            || c.weather_id != 0
            || c.terrain_id != 0
            || c.pseudo_weather_id != 0
        {
            b |= MB_HIT;
        }
        // Post-hit `multihit` block (Clanging Scales style).
        if c.multihit != 0 {
            b |= MB_POST;
        }
    }
    b
}

#[inline]
pub(crate) fn move_bits_of(m: Choices) -> MoveBits {
    MOVE_BITS[m as usize]
}
