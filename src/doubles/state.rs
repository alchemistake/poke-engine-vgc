//! Doubles battle state model.
//!
//! This is a *separate*, self-contained state model for doubles (2 active
//! Pokémon per side). It intentionally does NOT modify the existing singles
//! `Side`/`State` so the singles engine and its tests are untouched. It reuses
//! the shared value types (`Pokemon`, `Choices`, `MOVES`, `Abilities`,
//! `Items`, `SideConditions`, ...) from the rest of the crate.

use crate::choices::{Choice, Choices, MOVES};
use crate::engine::abilities::Abilities;
use crate::engine::items::Items;
use crate::engine::state::{PokemonVolatileStatus, Terrain, Weather};
use crate::pokemon::PokemonName;
use crate::state::{
    DamageDealt, LastUsedMove, Pokemon, PokemonIndex, PokemonMoveIndex, PokemonNature,
    PokemonSideCondition, PokemonStatus, PokemonType, SideConditions, StateTerrain,
    StateTrickRoom, StateWeather, VolatileStatusBitset, VolatileStatusDurations,
};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// One on-field slot. In doubles each side has two of these.
#[derive(Debug, Clone)]
pub struct ActiveSlot {
    /// Team member currently occupying this slot, or `None` when empty.
    pub pokemon_index: Option<PokemonIndex>,
    /// The weather the Forecast holder last synced its type to (PS
    /// `forecast.onWeatherChange`, which runs on a weather change or switch-in
    /// only, so a Soak / Magic Powder type sticks until the next change).
    pub forecast_weather: Option<Weather>,
    pub volatile_statuses: VolatileStatusBitset,
    pub volatile_status_durations: VolatileStatusDurations,
    pub substitute_health: i16,
    pub attack_boost: i8,
    pub defense_boost: i8,
    pub special_attack_boost: i8,
    pub special_defense_boost: i8,
    pub speed_boost: i8,
    pub accuracy_boost: i8,
    pub evasion_boost: i8,
    pub last_used_move: LastUsedMove,
    pub damage_dealt: DamageDealt,
    pub force_switch: bool,
    pub force_trapped: bool,
    pub slow_uturn_move: bool,
    pub baton_passing: bool,
    /// Electrify (single turn): the next move becomes Electric.
    pub electrify: bool,
    /// Power Trick is active (Attack and Defense are swapped).
    pub power_trick: bool,
    /// PS `move.hit` (1-based) of the current hit of a multi-hit move.
    pub hit_number: u8,
    pub shed_tailing: bool,
    /// Revival Blessing hit: the slot's switch answer revives a fainted
    /// party member instead of switching (PS `revivalblessing` slot condition).
    pub reviving: bool,
    /// Flinched by the previous move's secondary effect; cannot act this turn.
    pub flinched: bool,
    /// Used a move with the `recharge` flag (Hyper Beam, Solar Beam, ...);
    /// must recharge this turn.
    pub recharging: bool,
    /// Two-turn move charging (Dig, Meteor Beam, ...): the move id being
    /// charged and how many charge turns remain before it fires. Mirrors
    /// pokemon-showdown's `twoturnmove` / per-move volatile pair.
    pub charging: Option<(crate::choices::Choices, u8)>,
    /// Foe slot chosen on the charge turn; the locked follow-up turn keeps it
    /// (PS `twoturnmove.onStart` stores `targetLoc` on the move volatile).
    pub charge_target: usize,
    /// Salt Cure volatile: residual damage each turn (champions: 1/16 max
    /// HP, 1/8 for Water / Steel).
    pub salt_cure: bool,
    /// Stockpile layers (0-3) and the Def / SpD stages they actually added
    /// (PS `stockpile` volatile; removed again when it ends).
    pub stockpile: u8,
    /// Confusion volatile: PS `volatiles.confusion.time` (0 = not confused).
    pub confusion: u8,
    /// Hit by Roar / Whirlwind / Dragon Tail / Circle Throw this action
    /// (PS `forceSwitchFlag`): dragged out after the move.
    pub drag_out: bool,
    /// Focus Punch volatile (`focuspunch`, duration 1) and its `lostFocus`.
    pub focus_punch: bool,
    pub lost_focus: bool,
    /// Beak Blast volatile (`beakblast`, set by its priority charge, removed
    /// after its own move): a contact hit on this slot burns the attacker.
    pub beak_blast: bool,
    /// `gastroacid` volatile (Gastro Acid landed on this slot): a second
    /// Gastro Acid fails (PS addVolatile on an existing volatile). Cleared on
    /// switch-out.
    pub gastro_acid: bool,
    /// The ability the `gastroacid` volatile suppresses (the engine blanks the
    /// team member's ability to No Ability). PS `getAbility()` still returns
    /// it: Skill Swap / Role Play read and move it.
    pub gastro_ability: crate::engine::abilities::Abilities,
    /// Chilly Reception volatile (`chillyreception`, duration 1, set by its
    /// priority charge): only takes part in the residual sort.
    pub chilly_reception: bool,
    pub stockpile_def: i8,
    pub stockpile_spd: i8,
    /// Used Protect this turn; incoming attacks fail until end of turn.
    pub protected: bool,
    /// Helping Hand multiplier on this pokemon's next move's base power this
    /// turn (PS `helpinghand.condition.onBasePower`); 1.0 = none. Stacks by
    /// x1.5 per Helping Hand (onRestart). Cleared at end of turn.
    pub helping_hand_mult: f32,
    /// Raised at least one of its own stats this turn (PS
    /// `statsRaisedThisTurn`); gates Alluring Voice / Burning Jealousy
    /// secondaries. Reset at end of turn.
    pub stats_raised_this_turn: bool,
    /// PS `statsLoweredThisTurn` (Lash Out).
    pub stats_lowered_this_turn: bool,
    /// Destiny Bond volatile: when the holder faints, the last attacker also
    /// faints (PS data/conditions.ts destinybond.onAfterFaint).
    pub destinybond: bool,
    /// The last pokemon that dealt damage to this slot's occupant
    /// (side, team index), for Destiny Bond `onAfterFaint`.
    pub last_attacker: Option<(Side, crate::state::PokemonIndex)>,
    /// `partiallytrapped` volatile (Whirlpool, Fire Spin, ...): remaining
    /// duration, the binder (side, team index) and the damage divisor.
    pub partial_trap: Option<PartialTrap>,
    /// `yawn` volatile: remaining duration (0 = none) and the user who
    /// applied it (for the sleep `trySetStatus` when it ends).
    pub yawn: u8,
    /// `throatchop` volatile: remaining duration (sound moves blocked).
    pub throat_chop: u8,
    /// Set while a move is run through `useMove` (Copycat): no PP, no BeforeMove.
    pub external_call: bool,
    /// The move being run was bounced (PS `move.hasBounced`): no second bounce.
    pub bounced: bool,
    /// Beat Up: base power for the current hit / the remaining hits' base powers.
    pub bp_fixed: f32,
    pub beat_up_queue: Vec<f32>,
    /// Supreme Overlord `fallen` count fixed at switch-in (max 5).
    pub fallen: u8,
    /// `taunt` volatile: remaining duration (0 = none).
    pub taunt: u8,
    /// `magnetrise` volatile: remaining duration (0 = none). Makes the holder
    /// ungrounded (PS `isGrounded`) until it ends (PS data/moves.ts magnetrise).
    pub magnet_rise: u8,
    /// `smackdown` volatile (no duration): the holder is grounded (PS
    /// `isGrounded`) until it switches out (PS data/moves.ts smackdown).
    pub smackdown: bool,
    /// PS `activeTurns`: end-of-turn count since switch-in.
    pub active_turns: u8,
    /// PS `activeMoveActions`: move actions since switch-in, counted at the start
    /// of `runMove` (Fake Out / First Impression `onTry` fail when it is > 1).
    pub active_move_actions: u8,
    /// Illusion ended (hit by a damaging move): the log shows the real species.
    pub illusion_broken: bool,
    /// Opportunist `effectState.boosts`: stat raises copied from the foes
    /// (PS `onFoeAfterBoost`), applied when the holder's `onAnyAfterMove`,
    /// `onAnySwitchIn` or `onResidual` runs. Indexed like `StatBoosts`.
    pub opportunist_boosts: [i8; 7],
    /// Illusion's disguise, fixed at switch-in (`onBeforeSwitchIn`): the last
    /// unfainted pokemon after the holder then. `Some(None)` = no disguise.
    pub illusion_target: Option<Option<usize>>,
    /// `curse` volatile (Ghost Curse): loses 1/4 max HP each residual (order 12).
    pub cursed: bool,
    /// Set by the executing move's base-power callback precondition (Payback).
    pub bp_double: bool,
    /// `followme` (1) / `ragepowder` (2) volatile (duration 1): redirects foes' moves.
    pub follow_me: u8,
    /// Which protect-like move set `protected`: 0 Protect/Detect, 1 King's
    /// Shield, 2 Spiky Shield, 3 Baneful Bunker, 4 Silk Trap, 5 Burning Bulwark.
    pub protect_kind: u8,
    /// `perishsong` volatile: remaining duration (0 = none); faints at 0.
    pub perish: u8,
    /// PS ids of the volatiles in insertion order (PS `pokemon.volatiles` key
    /// order), kept by `sync_vol_order`. Orders the Residual handlers.
    pub vol_order: Vec<String>,
    /// `flashfire` volatile: Fire moves x1.5 (ModifyAtk / ModifySpA).
    pub flash_fire: bool,
    /// `charge` volatile: the next Electric move has doubled power.
    pub charge: bool,
    /// `noretreat` volatile (No Retreat already used).
    pub no_retreat: bool,
    /// `focusenergy` volatile: +2 critical-hit stages.
    pub focus_energy: bool,
    /// `attract` volatile: the infatuating pokemon (side, team index).
    pub attract: Option<(Side, PokemonIndex)>,
    /// `leechseed` volatile: the source's slot (side, slot) — PS
    /// `effectState.sourceSlot`, resolved with `getAtSlot` at residual time.
    pub leech_seed: Option<(Side, usize)>,
    /// `dragoncheer` volatile: `Some(hasDragonType)` fixed at start
    /// (crit ratio +2 if Dragon, else +1). PS data/moves.ts dragoncheer.
    pub dragon_cheer: Option<bool>,
    /// PS `pokemon.lastMove`: the last move this pokemon used (Encore).
    pub last_move: Option<crate::choices::Choices>,
    /// `glaiverush` volatile: moves against it always hit and deal x2 damage
    /// until its holder next tries to move (PS data/moves.ts glaiverush).
    pub glaive_rush: bool,
    /// Minimize volatile (PS `minimize`): see `mechanics::minimize_move`.
    pub minimize: bool,
    /// `ingrain` volatile: heals 1/16 max HP each residual (order 7).
    pub ingrain: bool,
    /// `healblock` volatile: remaining duration (0 = none).
    pub heal_block: u8,
    /// Trace `seek` (PS data/abilities.ts trace.effectState.seek): set by Trace's
    /// `onStart` (switch-in, Role Play / Skill Swap), cleared when a foe with No
    /// Ability is seen then; while set, every Update event retries the copy.
    pub trace_seek: bool,
    /// `syrupbomb` volatile: (source side, source team index, duration).
    pub syrup_bomb: Option<(Side, PokemonIndex, u8)>,
    /// `octolock` volatile: the binder (side, team index). No duration; it
    /// ends once the binder is no longer active.
    pub octolock: Option<(Side, PokemonIndex)>,
    /// `trapped` volatile (Block, Mean Look, Spider Web, Jaw Lock): no switching;
    /// ends with the volatiles. PS data/conditions.ts trapped.
    pub trapped: bool,
    /// The pokemon whose `trapped` link (`linkedPokemon`, PS `addVolatile`
    /// `linkedStatus` = 'trapper') holds this slot: released when it leaves the
    /// field (PS `removeLinkedVolatiles` on faint / `clearVolatile`).
    pub trapper: Option<(Side, PokemonIndex)>,
    /// PS `pokemon.hurtThisTurn`: lost HP earlier this turn (Assurance).
    pub hurt_this_turn: bool,
    /// PS `hurtThisTurn`'s value: the HP left after that damage (0 if none).
    /// Champions `spreadMoveHit` uses it as Emergency Exit's original HP.
    pub hurt_hp: i16,
    /// PS `pokemon.newlySwitched`: switched in since the turn started (Payback).
    pub newly_switched: bool,
    /// The `twoturnmove` volatile (duration 2) is still around for this
    /// turn's residual after the charged move fired.
    pub twoturn_fired: bool,
    /// `encore` volatile: the forced move and remaining duration.
    pub encore: Option<(crate::choices::Choices, u8)>,
    /// `torment` volatile (no duration): the last move is disabled while it lasts.
    pub torment: bool,
    /// `imprison` volatile (no duration): the user's moves are blocked for the
    /// foes (not modelled yet). Set once; a second Imprison fails.
    pub imprison: bool,
    /// Forest's Curse added Grass to a single-typed pokemon (PS `addedType`):
    /// the second type slot is restored on switch-out / faint.
    pub added_grass: bool,
    /// PS `addedType` on a dual-typed pokemon (Trick-or-Treat Ghost, Forest's
    /// Curse Grass): a third type, reset on switch-out / faint. TYPELESS = none.
    pub added_type: PokemonType,
    /// `roost` volatile (duration 1): the pokemon's types before Flying was
    /// filtered out (`onType`); restored at residual / switch-out / faint.
    pub roost_orig: Option<(PokemonType, PokemonType)>,
    /// `aquaring` volatile: heals 1/16 max HP each residual (order 6).
    pub aqua_ring: bool,
    /// `counter` / `mirrorcoat` volatiles (duration 1): the last foe slot that
    /// hit this pokemon with a physical / special move, and twice the damage.
    pub counter_vol: Option<(Option<(Side, usize)>, i16)>,
    pub mirror_coat_vol: Option<(Option<(Side, usize)>, i16)>,
    /// PS `moveThisTurnResult` / `moveLastTurnResult` (false = the move failed;
    /// None = undefined or null). Stomping Tantrum reads the last turn's.
    pub move_result_this_turn: Option<bool>,
    pub move_result_last_turn: Option<bool>,
    /// `allyswitch` volatile: (counter, duration).
    pub ally_switch: Option<(u16, u8)>,
    /// `lockon` volatile on the user (PS data/moves.ts lockon): the locked foe
    /// (side, team index) and the remaining duration. Its moves always hit it.
    pub lock_on: Option<(Side, PokemonIndex, u8)>,
    /// `uproar` volatile: remaining duration (0 = none). Locks the move.
    pub uproar: u8,
    /// `stall` volatile (Protect / Detect / Endure counter): (counter, duration).
    pub stall: Option<(u16, u8)>,
    /// `endure` volatile (duration 1): survive moves at 1 HP.
    pub endure: bool,
    /// PS `lastMoveTargetLoc`: the target location of the last move (signed:
    /// foe slot + 1, own slot -(slot + 1)); a locked move reuses it.
    pub last_target_loc: i8,
    /// `lockedmove` volatile (Outrage, Thrash, Petal Dance): (duration,
    /// trueDuration) from PS data/conditions.ts.
    pub locked_move: Option<(u8, u8)>,
    /// The move that set `locked_move` (a copied Petal Dance locks its copier
    /// into Petal Dance). PS `getLockedMove` replaces the chosen move with it.
    pub locked_move_id: Option<Choices>,
    /// `disable` volatile: the disabled move and remaining duration.
    pub disabled: Option<(Choices, u8)>,
    /// BeforeMove aborted the locked charge turn: PS `twoturnmove` is gone
    /// (`onMoveAborted`) but the move's own condition (Dive, ...) lasts to
    /// the end of the turn.
    pub charge_aborted: bool,
    pub yawn_source: Option<(Side, PokemonIndex)>,
    /// Libero / Protean already used this switch-in (PS `effectState.libero`);
    /// holds the original types, restored on switch-out.
    pub type_change: Option<(crate::state::PokemonType, crate::state::PokemonType)>,
    /// PS `effectState.protean` / `effectState.libero`: the holder's type change
    /// already happened for its current ability. A new ability (Skill Swap,
    /// Trace, Role Play, ...) makes a fresh `effectState`, so it resets.
    pub protean_done: bool,
}

/// PS `partiallytrapped` effectState (data/conditions.ts).
#[derive(Debug, Clone, Copy)]
pub struct PartialTrap {
    pub duration: u8,
    pub source_side: Side,
    pub source_idx: PokemonIndex,
    pub divisor: i16,
}

impl ActiveSlot {
    pub fn empty() -> Self {
        Self {
            pokemon_index: None,
            forecast_weather: None,
            volatile_statuses: VolatileStatusBitset::default(),
            volatile_status_durations: VolatileStatusDurations::default(),
            substitute_health: 0,
            attack_boost: 0,
            defense_boost: 0,
            special_attack_boost: 0,
            special_defense_boost: 0,
            speed_boost: 0,
            accuracy_boost: 0,
            evasion_boost: 0,
            last_used_move: LastUsedMove::None,
            damage_dealt: DamageDealt::default(),
            force_switch: false,
            force_trapped: false,
            slow_uturn_move: false,
            baton_passing: false,
            electrify: false,
            power_trick: false,
            hit_number: 1,
            shed_tailing: false,
            reviving: false,
            flinched: false,
            recharging: false,
            charging: None,
            charge_target: 0,
            salt_cure: false,
            stockpile: 0,
            confusion: 0,
            drag_out: false,
            focus_punch: false,
            beak_blast: false,
            gastro_acid: false,
            gastro_ability: crate::engine::abilities::Abilities::NONE,
            chilly_reception: false,
            lost_focus: false,
            stockpile_def: 0,
            stockpile_spd: 0,
            protected: false,
            helping_hand_mult: 1.0,
            stats_raised_this_turn: false,
            stats_lowered_this_turn: false,
            destinybond: false,
            last_attacker: None,
            partial_trap: None,
            yawn: 0,
            throat_chop: 0,
            external_call: false,
            bounced: false,
            bp_fixed: 0.0,
            beat_up_queue: Vec::new(),
            fallen: 0,
            taunt: 0,
            magnet_rise: 0,
            smackdown: false,
            active_turns: 0,
            active_move_actions: 0,
            illusion_broken: false,
            opportunist_boosts: [0; 7],
            illusion_target: None,
            cursed: false,
            bp_double: false,
            follow_me: 0,
            protect_kind: 0,
            perish: 0,
            vol_order: Vec::new(),
            flash_fire: false,
            charge: false,
            no_retreat: false,
            focus_energy: false,
            attract: None,
            leech_seed: None,
            dragon_cheer: None,
            last_move: None,
            glaive_rush: false,
            minimize: false,
            ingrain: false,
            heal_block: 0,
            trace_seek: false,
            syrup_bomb: None,
            octolock: None,
            trapped: false,
            trapper: None,
            hurt_this_turn: false,
            hurt_hp: 0,
            newly_switched: false,
            twoturn_fired: false,
            encore: None,
            torment: false,
            imprison: false,
            added_grass: false,
            added_type: PokemonType::TYPELESS,
            roost_orig: None,
            aqua_ring: false,
            counter_vol: None,
            mirror_coat_vol: None,
            move_result_this_turn: None,
            move_result_last_turn: None,
            ally_switch: None,
            lock_on: None,
            uproar: 0,
            stall: None,
            endure: false,
            last_target_loc: 0,
            locked_move: None,
            locked_move_id: None,
            disabled: None,
            charge_aborted: false,
            yawn_source: None,
            type_change: None,
            protean_done: false,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.pokemon_index.is_none()
    }

    /// Effective speed including the current speed boost.
    pub fn boosted_speed(&self, base_speed: i16) -> i16 {
        crate::doubles::state::multiply_boost(self.speed_boost, base_speed)
    }
}

#[derive(Debug, Clone, Copy, Default)]
/// Per-team-member stat boosts. Boosts live on the pokemon (like
/// pokemon-showdown), survive bench time, are cleared on faint, and are
/// cleared on switch-in by the champions `clearVolatile` override.
pub struct TeamBoosts {
    pub atk: i8,
    pub def: i8,
    pub spa: i8,
    pub spd: i8,
    pub spe: i8,
    pub acc: i8,
    pub eva: i8,
}

#[derive(Debug, Clone)]
pub struct DoublesSide {
    pub team: [Pokemon; 6],
    pub active: [ActiveSlot; 2],
    pub side_conditions: SideConditions,
    /// Original species names (including formes) for each team slot, used for
    /// name-based target matching in the replay test.
    pub species_names: [String; 6],
    /// Gender per team member: "M" / "F" / None (genderless). Used by Attract.
    pub genders: [Option<String>; 6],
    /// Stat boosts per team member (see `TeamBoosts`).
    pub team_boosts: [TeamBoosts; 6],
    /// Toxic counter per team member (0 = not toxic; reset on switch-in).
    pub toxic_stage: [i8; 6],
    /// PS `pokemon.speed`: the speed cached at the last `updateSpeed()`
    /// (turn start, residual, queue re-sort). `eachEvent`, `runSwitch` and
    /// Residual handlers sort by it; a fresh switch-in keeps a stale value.
    pub speed_cache: [i16; 6],
    /// PS `side.pokemon` order (team indices). Switch-ins swap positions
    /// (PS `switchIn`); random drag-ins sample from positions 2.. of it.
    pub ps_order: [usize; 6],
    /// Team member that fainted in an active slot this turn and is still
    /// sitting there (PS keeps fainted actives in `side.active` until they
    /// are replaced; their volatiles are cleared on faint, but status /
    /// ability / slot-condition / field handlers stay in the Residual sort).
    /// `(team index, the occupant's stall volatile)`. Cleared on switch-in.
    pub fainted_in_slot: [Option<(PokemonIndex, Option<(u16, u8)>)>; 2],
    /// Wish slot condition per position: (heal amount, residuals seen).
    pub wish: [Option<(i16, u8)>; 2],
    /// `futuremove` slot condition per position: (move, source side, source index, residuals seen).
    pub future_move: [Option<(Choices, Side, PokemonIndex, u8)>; 2],
    /// Side conditions in insertion order (PS object key order).
    pub sc_order: Vec<String>,
    /// Supersweet Syrup already triggered (PS `syrupTriggered`, once per
    /// battle per pokemon).
    pub syrup_triggered: [bool; 6],
    /// Healing Wish / Lunar Dance slot conditions per active slot (PS
    /// `slotCondition: 'healingwish' | 'lunardance'`): they wait for the next
    /// pokemon switched into that slot.
    pub healing_wish_slot: [bool; 2],
    /// PS `moveSlot.used` per team member / move slot (set by `deductPP`).
    pub moves_used: [[bool; 4]; 6],
    /// PS `pokemon.attackedBy` entries with damage (`getLastDamagedBy`), per
    /// team member: (attacker side, attacker slot, damage, turn).
    /// PS `attackedBy`: (attacker side, attacker slot at the hit, damage, turn,
    /// attacker team index). Pruned at end of turn unless the attacker is still
    /// on the field and alive (PS `endTurn`: `source.isActive`).
    pub damaged_by: [Vec<(Side, usize, i16, u32, usize)>; 6],
    /// Original (atk, def, spa, spd) of a team member whose `storedStats` were
    /// changed (Guard / Power Split); restored by `setSpecies` on switch-in.
    pub saved_stats: [Option<[i16; 4]>; 6],
    /// Original raw speed of a team member whose `storedStats.spe` was swapped
    /// (Speed Swap); restored by `setSpecies` on switch-in.
    pub saved_speed: [Option<i16>; 6],
}

impl DoublesSide {
    pub fn team_member(&self, index: PokemonIndex) -> &Pokemon {
        &self.team[index as usize]
    }
    pub fn team_member_mut(&mut self, index: PokemonIndex) -> &mut Pokemon {
        &mut self.team[index as usize]
    }

    /// The Pokémon in an active slot, if any.
    pub fn slot_pokemon(&self, slot: usize) -> Option<&Pokemon> {
        self.active[slot].pokemon_index.map(|i| self.team_member(i))
    }
    pub fn slot_pokemon_mut(&mut self, slot: usize) -> Option<&mut Pokemon> {
        let idx = self.active[slot].pokemon_index?;
        Some(self.team_member_mut(idx))
    }

    /// Team-member boosts for a slot, if occupied.
    pub fn slot_boosts(&self, slot: usize) -> Option<&TeamBoosts> {
        let i = self.active[slot].pokemon_index?;
        Some(&self.team_boosts[i as usize])
    }
    pub fn slot_boosts_mut(&mut self, slot: usize) -> Option<&mut TeamBoosts> {
        let i = self.active[slot].pokemon_index?;
        Some(&mut self.team_boosts[i as usize])
    }

    /// Clear a team member's boosts (faint / champions switch-in reset).
    pub fn clear_member_boosts(&mut self, index: PokemonIndex) {
        self.team_boosts[index as usize] = TeamBoosts::default();
    }

    pub fn alive_team_count(&self) -> usize {
        self.team
            .iter()
            .filter(|p| p.hp > 0 && p.id != crate::pokemon::PokemonName::NONE)
            .count()
    }

    pub fn find_replacement(&self) -> Option<PokemonIndex> {
        for i in 0..6 {
            let idx = PokemonIndex::deserialize(&i.to_string());
            if self.team[i].hp > 0
                && self.team[i].id != crate::pokemon::PokemonName::NONE
                && !self
                    .active
                    .iter()
                    .any(|s| s.pokemon_index == Some(idx))
            {
                return Some(idx);
            }
        }
        None
    }

    pub fn get_side_condition(&self, c: PokemonSideCondition) -> i8 {
        match c {
            PokemonSideCondition::Reflect => self.side_conditions.reflect,
            PokemonSideCondition::LightScreen => self.side_conditions.light_screen,
            PokemonSideCondition::AuroraVeil => self.side_conditions.aurora_veil,
            PokemonSideCondition::Tailwind => self.side_conditions.tailwind,
            PokemonSideCondition::Protect => self.side_conditions.protect,
            PokemonSideCondition::WideGuard => self.side_conditions.wide_guard,
            PokemonSideCondition::QuickGuard => self.side_conditions.quick_guard,
            PokemonSideCondition::CraftyShield => self.side_conditions.crafty_shield,
            PokemonSideCondition::MatBlock => self.side_conditions.mat_block,
            PokemonSideCondition::Safeguard => self.side_conditions.safeguard,
            PokemonSideCondition::LuckyChant => self.side_conditions.lucky_chant,
            PokemonSideCondition::Mist => self.side_conditions.mist,
            PokemonSideCondition::Spikes => self.side_conditions.spikes,
            PokemonSideCondition::Stealthrock => self.side_conditions.stealth_rock,
            PokemonSideCondition::StickyWeb => self.side_conditions.sticky_web,
            PokemonSideCondition::ToxicSpikes => self.side_conditions.toxic_spikes,
            PokemonSideCondition::ToxicCount => self.side_conditions.toxic_count,
            PokemonSideCondition::HealingWish => self.side_conditions.healing_wish,
            PokemonSideCondition::LunarDance => self.side_conditions.lunar_dance,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DoublesState {
    /// PS `battle.lastMove`: the last move used by anyone (Copycat).
    pub last_move: Option<Choices>,
    /// Set when the current action's `onHit` ran a Copycat / Sleep Talk
    /// `useMove`: PS's `clearActiveMove` of the copying move is then a no-op,
    /// so the copied move (not the copier) is `battle.lastMove`.
    pub copy_called: bool,
    pub side_one: DoublesSide,
    pub side_two: DoublesSide,
    pub weather: StateWeather,
    pub terrain: StateTerrain,
    pub trick_room: StateTrickRoom,
    /// Wonder Room pseudo-weather (Def and Sp. Def swapped), duration 5.
    pub wonder_room: StateTrickRoom,
    /// Magic Room pseudo-weather (items suppressed), duration 5.
    pub magic_room: StateTrickRoom,
    /// Gravity pseudo-weather (everything grounded, accuracy x1.67), duration 5.
    pub gravity: StateTrickRoom,
    /// Fairy Lock pseudo-weather (everything trapped), duration 2.
    pub fairy_lock: StateTrickRoom,
    pub turn: u32,
    /// Pokemon that reached 0 HP during the current move's hit loop. PS only
    /// resolves faints after the loop, so its per-hit `eachEvent` still sorts
    /// them: (side, slot, team index).
    pub unresolved_faints: Vec<(crate::doubles::state::Side, usize, PokemonIndex)>,
    /// The `disabled` volatile each pokemon had when it fainted. PS keeps the
    /// volatiles until `faintMessages`, so a DamagingHit handler that runs after
    /// a Destiny Bond KO of its attacker still sees the attacker's Disable.
    pub faint_disabled: Vec<(Side, PokemonIndex, Option<(Choices, u8)>)>,
    /// Attackers KO'd by a Destiny Bond `onAfterFaint` during the current move.
    /// PS only resolves their faint after the move, so until then they still
    /// have HP for the rest of the hit (e.g. a lock volatile's `addVolatile`).
    /// The attacker's lock (`locked_move`) at its KO, for the same reason.
    pub db_alive_pending: Vec<(Side, PokemonIndex, Option<(u8, u8)>)>,
    /// Disguise holders whose hit was absorbed by a spread move. PS busts the
    /// forme in `onUpdate`, i.e. inside the next `eachEvent('Update')` after
    /// its speed sort (the sort still sees the stale speed).
    pub pending_disguise: Vec<(crate::doubles::state::Side, usize)>,
    /// Sides of every fainted pokemon in `faint()` call order: with both sides
    /// wiped out at once, PS gives the win to the side of the last one
    /// (`checkWin(faintData)`, sim/battle.ts).
    pub faint_log: Vec<Side>,
    /// Attacker of a move hit that fainted a pokemon, with its
    /// `onSourceAfterFaint` ability (Moxie, Chilling Neigh, Grim Neigh): (side,
    /// slot, team index, ability). PS runs that event once per `resolveFaints`
    /// batch, with `length` = the batch size, and only if the battle is not over.
    pub pending_after_faint: Option<(Side, usize, PokemonIndex, Abilities)>,
    /// Encore `onStart` (champions): the target's queued move becomes this move
    /// (`queue.changeAction`), planned during the move and applied by the battle loop.
    pub queue_change: Option<crate::doubles::battle::QueuePlan>,
    /// Round `onTry` ran this action: the battle loop prioritizes the next
    /// queued Round with BP doubled (`queue.prioritizeAction`).
    pub queue_round: bool,
    /// After You `onHit`: the target's queued action moves to the front
    /// (`queue.prioritizeAction`), applied by the battle loop.
    pub queue_prioritize: Option<(Side, usize)>,
    /// Quash `onHit`: the target's queued action is moved last (`order = 201`).
    pub queue_quash: Option<(Side, usize)>,
    /// Instruct `onHit`: the target's last move is queued again at the front of
    /// the queue (`resolveAction` + `prioritizeAction`). (side, slot, pokemon,
    /// move id, the target's signed `lastMoveTargetLoc`).
    pub queue_instruct: Option<(Side, usize, PokemonIndex, Choices, i8)>,
    /// A move is executing (Destiny Bond only reacts to move damage).
    pub in_move: bool,
    /// Gravity `onFieldStart`: queued moves cancelled (`queue.cancelMove`).
    pub queue_cancel: Vec<(Side, usize)>,
    /// Wonder Room's `onModifyMove` has run since it started (PS fixes the
    /// handler's subOrder only once `state.target` is set).
    pub wonder_room_seen: bool,
}

impl DoublesState {
    pub fn battle_is_over(&self) -> f32 {
        if self.side_one.alive_team_count() == 0 && self.side_two.alive_team_count() == 0 {
            // PS `checkWin`: the side of the last faint wins.
            return if self.faint_log.last() == Some(&Side::One) { 1.0 } else { -1.0 };
        }
        if self.side_one.alive_team_count() == 0 {
            return -1.0;
        }
        if self.side_two.alive_team_count() == 0 {
            return 1.0;
        }
        0.0
    }

    pub fn side(&self, which: Side) -> &DoublesSide {
        match which {
            Side::One => &self.side_one,
            Side::Two => &self.side_two,
        }
    }
    pub fn side_mut(&mut self, which: Side) -> &mut DoublesSide {
        match which {
            Side::One => &mut self.side_one,
            Side::Two => &mut self.side_two,
        }
    }

    /// All four on-field Pokémon in field-position order:
    /// [A0, A1, B0, B1].
    pub fn all_active(&self) -> [Option<(&Pokemon, Side, usize)>; 4] {
        [
            self.side_one
                .slot_pokemon(0)
                .map(|p| (p, Side::One, 0)),
            self.side_one
                .slot_pokemon(1)
                .map(|p| (p, Side::One, 1)),
            self.side_two
                .slot_pokemon(0)
                .map(|p| (p, Side::Two, 0)),
            self.side_two
                .slot_pokemon(1)
                .map(|p| (p, Side::Two, 1)),
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    One,
    Two,
}

impl Side {
    pub fn other(self) -> Side {
        match self {
            Side::One => Side::Two,
            Side::Two => Side::One,
        }
    }
}

/// Stat-stage multiplier, matching pokemon-showdown's boost table.
pub fn multiply_boost(boost: i8, stat: i16) -> i16 {
    match boost {
        -6 => stat * 2 / 8,
        -5 => stat * 2 / 7,
        -4 => stat * 2 / 6,
        -3 => stat * 2 / 5,
        -2 => stat * 2 / 4,
        -1 => stat * 2 / 3,
        0 => stat,
        1 => stat * 3 / 2,
        2 => stat * 4 / 2,
        3 => stat * 5 / 2,
        4 => stat * 6 / 2,
        5 => stat * 7 / 2,
        6 => stat * 8 / 2,
        _ => stat,
    }
}

// ---------------------------------------------------------------------------
// Team / input data (serde)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct TeamMember {
    pub pos: Option<u8>,
    pub species: String,
    pub name: Option<String>,
    pub level: u8,
    pub evs: Evs,
    pub ivs: Option<Ivs>,
    pub nature: String,
    pub ability: String,
    pub item: String,
    pub gender: Option<String>,
    pub moves: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Evs {
    pub hp: u16,
    pub atk: u16,
    pub def: u16,
    pub spa: u16,
    pub spd: u16,
    pub spe: u16,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Ivs {
    #[allow(dead_code)]
    pub hp: u8,
    #[allow(dead_code)]
    pub atk: u8,
    #[allow(dead_code)]
    pub def: u8,
    #[allow(dead_code)]
    pub spa: u8,
    #[allow(dead_code)]
    pub spd: u8,
    #[allow(dead_code)]
    pub spe: u8,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BattleInput {
    pub seed: [u64; 4],
    pub teams: Teams,
    pub leads: Option<Leads>,
    /// Explicit per-turn controller actions (fuzz testing). When present, the
    /// battle is driven by these choices instead of the auto-move policy.
    #[serde(default)]
    pub choices: Option<Vec<FuzzTurn>>,
    /// Stop after this many turns WITHOUT reporting an error (stepwise fuzz
    /// mode: the driver re-runs the engine with one more turn each step).
    #[serde(default)]
    pub stop_after: Option<u32>,
}

/// One controller action for one active slot.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum FuzzAction {
    /// Use the move in slot `slot` (0-3). `target` is a signed PS location:
    /// 1/2 = foe slot, negative = own slot (ally moves); ignored by the
    /// engine for non-attack moves. `struggle` marks a forced Struggle.
    Move {
        slot: usize,
        target: i8,
        #[serde(default)]
        struggle: bool,
    },
    /// Do nothing (empty active slot with no replacement available).
    Pass,

    /// Switch in the team member at 0-based team `position`.
    Switch {
        position: usize,
    },
}

/// The controller's actions for one full turn: two per side, in active-slot
/// order (index 0 = left active, index 1 = right active).
#[derive(Debug, Clone, Deserialize)]
pub struct FuzzTurn {
    pub a: Vec<FuzzAction>,
    pub b: Vec<FuzzAction>,
    /// Forced switches pokemon-showdown requested during this turn, in
    /// request order: mid-turn pivots (Volt Switch / U-turn) and end-of-turn
    /// faint replacements.
    #[serde(default)]
    pub switches: Vec<FuzzSwitch>,
}

/// One answer to a pokemon-showdown switch request.
#[derive(Debug, Clone, Deserialize)]
pub struct FuzzSwitch {
    /// "a" or "b".
    pub side: String,
    /// Active slot being replaced.
    pub slot: usize,
    /// 0-based battle-team position switched in; `None` = pass (no
    /// replacement available).
    #[serde(default)]
    pub position: Option<usize>,
    /// True for a mid-turn pivot switch (the slot's pokemon was still alive).
    #[serde(default)]
    pub pivot: bool,
    /// Revival Blessing answer: `position` is the fainted pokemon revived.
    #[serde(default)]
    pub revive: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Teams {
    #[serde(alias = "A")]
    pub a: Vec<TeamMember>,
    #[serde(alias = "B")]
    pub b: Vec<TeamMember>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Leads {
    #[serde(alias = "A")]
    pub a: Vec<usize>,
    #[serde(alias = "B")]
    pub b: Vec<usize>,
}

/// A single move from a pokemon-showdown battle log, used by the replay test to
/// validate the engine's battle loop (faints, switches, retargeting, winner)
/// against ground-truth damage.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplayMove {
    #[serde(default)]
    pub turn: Option<usize>,
    pub att_side: String,
    pub att_slot: usize,
    pub att: String,
    pub tgt_side: String,
    pub tgt_slot: usize,
    pub tgt: String,
    pub damage: i16,
    #[serde(default)]
    pub crit: bool,
    #[serde(default)]
    pub self_faint: bool,
    #[serde(default)]
    pub transform_from: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReplayInput {
    pub seed: Option<[u64; 4]>,
    pub teams: Teams,
    pub leads: Option<Leads>,
    pub moves: Vec<ReplayMove>,
    pub winner: String,
}

pub fn to_id(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect()
}

fn parse_nature(s: &str) -> PokemonNature {
    let id = to_id(s);
    PokemonNature::from_str(&id.to_uppercase()).unwrap_or(PokemonNature::SERIOUS)
}

fn parse_ability(s: &str) -> Abilities {
    let id = to_id(s);
    Abilities::from_str(&id.to_uppercase()).unwrap_or(Abilities::NONE)
}

fn parse_item(s: &str) -> Items {
    if s.is_empty() {
        return Items::NONE;
    }
    let id = to_id(s);
    Items::from_str(&id.to_uppercase()).unwrap_or(Items::NONE)
}

/// Convert a champions "stat points" EV value into effective EVs, matching
/// `genx::state::stat_points_to_effective_ev`.
fn stat_points_to_effective_ev(stat_points: u16) -> u16 {
    if stat_points == 0 {
        0
    } else {
        8 * stat_points - 4
    }
}

fn common_stat_calc(base: u16, ev: u16, level: u16) -> u16 {
    // 31 IV (matches the random-team generator and the singles formula).
    ((2 * base + 31 + ev / 4) * level) / 100
}

/// Build the full stat block from base stats / evs / level / nature, matching
/// pokemon-showdown for level-50, 31-IV teams.
pub fn compute_stats(
    base: (i16, i16, i16, i16, i16, i16),
    evs: &Evs,
    level: u8,
    nature: PokemonNature,
) -> (i16, i16, i16, i16, i16, i16) {
    let lvl = level as u16;
    let e = (
        stat_points_to_effective_ev(evs.hp),
        stat_points_to_effective_ev(evs.atk),
        stat_points_to_effective_ev(evs.def),
        stat_points_to_effective_ev(evs.spa),
        stat_points_to_effective_ev(evs.spd),
        stat_points_to_effective_ev(evs.spe),
    );
    let mut hp = (common_stat_calc(base.0 as u16, e.0, lvl) + lvl + 10) as i16;
    let mut atk = (common_stat_calc(base.1 as u16, e.1, lvl) + 5) as i16;
    let mut def = (common_stat_calc(base.2 as u16, e.2, lvl) + 5) as i16;
    let mut spa = (common_stat_calc(base.3 as u16, e.3, lvl) + 5) as i16;
    let mut spd = (common_stat_calc(base.4 as u16, e.4, lvl) + 5) as i16;
    let mut spe = (common_stat_calc(base.5 as u16, e.5, lvl) + 5) as i16;

    // nature: +110% / -90%
    //
    // NOTE: this repo's pokemon-showdown data/natures.ts remaps Sassy to
    // +spd/-spe (standard is +spd/-spa); all other natures match standard.
    let (bonus, penalty) = match nature {
        PokemonNature::LONELY => (0, 1),
        PokemonNature::ADAMANT => (0, 2),
        PokemonNature::NAUGHTY => (0, 3),
        PokemonNature::BRAVE => (0, 4),
        PokemonNature::BOLD => (1, 0),
        PokemonNature::IMPISH => (1, 2),
        PokemonNature::LAX => (1, 3),
        PokemonNature::RELAXED => (1, 4),
        PokemonNature::MODEST => (2, 0),
        PokemonNature::MILD => (2, 1),
        PokemonNature::RASH => (2, 3),
        PokemonNature::QUIET => (2, 4),
        PokemonNature::CALM => (3, 0),
        PokemonNature::GENTLE => (3, 1),
        PokemonNature::CAREFUL => (3, 2),
        PokemonNature::QUIRKY => (9, 9),
        PokemonNature::SASSY => (3, 4),
        PokemonNature::TIMID => (4, 0),
        PokemonNature::HASTY => (4, 1),
        PokemonNature::JOLLY => (4, 2),
        PokemonNature::NAIVE => (4, 3),
        _ => (9, 9),
    };
    let apply = |v: i16, idx: usize| -> i16 {
        if idx == bonus {
            v * 11 / 10
        } else if idx == penalty {
            v * 9 / 10
        } else {
            v
        }
    };
    atk = apply(atk, 0);
    def = apply(def, 1);
    spa = apply(spa, 2);
    spd = apply(spd, 3);
    spe = apply(spe, 4);
    (hp, atk, def, spa, spd, spe)
}

fn build_pokemon(m: &TeamMember) -> Pokemon {
    // Forme names (e.g. "Furfrou-Debutante") may not be in the table or the
    // PokemonName enum; fall back to the base species (before the "-").
    let base = m.species.split('-').next().unwrap_or(&m.species);
    let id = PokemonName::from_str(&to_id(&m.species).to_uppercase())
        .or_else(|_| PokemonName::from_str(&to_id(base).to_uppercase()))
        .unwrap_or_else(|_| panic!("unknown species {}", m.species));
    let data = crate::doubles::species_types::species_data(&to_id(&m.species))
        .or_else(|| crate::doubles::species_types::species_data(&to_id(base)))
        .unwrap_or_else(|| panic!("no species data for {}", m.species));
    let base = data.base_stats;
    let (t0, t1) = data.types;
    let types = (
        PokemonType::from(t0),
        PokemonType::from(t1),
    );
    let nature = parse_nature(&m.nature);
    let (hp, atk, def, spa, spd, spe) =
        compute_stats(base, &m.evs, m.level, nature.clone());
    let mut moves = vec![MoveSlot::default(); 4];
    for (i, mv) in m.moves.iter().enumerate().take(4) {
        let id = to_id(mv);
        let choices = Choices::from_str(&id.to_uppercase())
            .unwrap_or_else(|_| panic!("unknown move {mv}"));
        let choice: Choice = MOVES.get(&choices).cloned().unwrap();
        // PS `baseMoveSlots` pp: `calculatePP(move, 3)`, champions
        // scripts.ts `(move.pp / 5 + 1) * 4` (stored as `Choice.pp`).
        moves[i] = MoveSlot {
            id: choices,
            pp: choice.pp as i8,
            choice,
        };
    }
    Pokemon {
        id,
        level: m.level as i8,
        types,
        base_types: types,
        hp,
        maxhp: hp,
        ability: parse_ability(&m.ability),
        base_ability: parse_ability(&m.ability),
        item: parse_item(&m.item),
        nature,
        evs: (
            m.evs.hp as u8,
            m.evs.atk as u8,
            m.evs.def as u8,
            m.evs.spa as u8,
            m.evs.spd as u8,
            m.evs.spe as u8,
        ),
        attack: atk,
        defense: def,
        special_attack: spa,
        special_defense: spd,
        speed: spe,
        status: PokemonStatus::NONE,
        rest_turns: 0,
        sleep_turns: 0,
        weight_kg: crate::doubles::species_weights::species_weight_kg(&to_id(&m.species))
            .or_else(|| {
                let base_species = m.species.split('-').next().unwrap_or(&m.species);
                crate::doubles::species_weights::species_weight_kg(&to_id(base_species))
            })
            .unwrap_or(1.0),
        mega_evolved: false,
        terastallized: false,
        tera_type: PokemonType::NORMAL,
        attacked_by: Vec::new(),
        moves: crate::state::PokemonMoves {
            m0: crate::state::Move {
                id: moves[0].id,
                disabled: false,
                pp: moves[0].pp,
                choice: moves[0].choice.clone(),
            },
            m1: crate::state::Move {
                id: moves[1].id,
                disabled: false,
                pp: moves[1].pp,
                choice: moves[1].choice.clone(),
            },
            m2: crate::state::Move {
                id: moves[2].id,
                disabled: false,
                pp: moves[2].pp,
                choice: moves[2].choice.clone(),
            },
            m3: crate::state::Move {
                id: moves[3].id,
                disabled: false,
                pp: moves[3].pp,
                choice: moves[3].choice.clone(),
            },
        },
    }
}

#[derive(Debug, Clone)]
struct MoveSlot {
    id: Choices,
    choice: Choice,
    pp: i8,
}
impl Default for MoveSlot {
    fn default() -> Self {
        Self {
            id: Choices::NONE,
            choice: Choice::default(),
            pp: 0,
        }
    }
}

impl DoublesState {
    pub fn from_input(input: &BattleInput) -> Self {
        Self::from_teams(&input.teams)
    }

    pub fn from_teams(teams: &Teams) -> Self {
        let mut side_one = DoublesSide {
            team: std::array::from_fn(|_| Pokemon::default()),
            active: [ActiveSlot::empty(), ActiveSlot::empty()],
            side_conditions: SideConditions::default(),
            species_names: std::array::from_fn(|_| String::new()),
            genders: std::array::from_fn(|_| None),
            team_boosts: std::array::from_fn(|_| TeamBoosts::default()),
            toxic_stage: [0; 6],
            speed_cache: [0; 6],
            ps_order: [0, 1, 2, 3, 4, 5],
            fainted_in_slot: [None, None],
            wish: [None, None],
            future_move: [None, None],
            sc_order: Vec::new(),
            syrup_triggered: [false; 6],
            healing_wish_slot: [false; 2],
            moves_used: [[false; 4]; 6],
            damaged_by: std::array::from_fn(|_| Vec::new()),
            saved_stats: [None; 6],
            saved_speed: [None; 6],
        };
        let mut side_two = DoublesSide {
            team: std::array::from_fn(|_| Pokemon::default()),
            active: [ActiveSlot::empty(), ActiveSlot::empty()],
            side_conditions: SideConditions::default(),
            species_names: std::array::from_fn(|_| String::new()),
            genders: std::array::from_fn(|_| None),
            team_boosts: std::array::from_fn(|_| TeamBoosts::default()),
            toxic_stage: [0; 6],
            speed_cache: [0; 6],
            ps_order: [0, 1, 2, 3, 4, 5],
            fainted_in_slot: [None, None],
            wish: [None, None],
            future_move: [None, None],
            sc_order: Vec::new(),
            syrup_triggered: [false; 6],
            healing_wish_slot: [false; 2],
            moves_used: [[false; 4]; 6],
            damaged_by: std::array::from_fn(|_| Vec::new()),
            saved_stats: [None; 6],
            saved_speed: [None; 6],
        };
        for (i, m) in teams.a.iter().enumerate().take(6) {
            side_one.team[i] = build_pokemon(m);
            // PS `setSpecies`: speed = storedStats.spe.
            side_one.speed_cache[i] = side_one.team[i].speed;
            side_one.species_names[i] = m.species.clone();
            side_one.genders[i] = m.gender.clone();
        }
        for (i, m) in teams.b.iter().enumerate().take(6) {
            side_two.team[i] = build_pokemon(m);
            side_two.speed_cache[i] = side_two.team[i].speed;
            side_two.species_names[i] = m.species.clone();
            side_two.genders[i] = m.gender.clone();
        }
        let state = Self {
            last_move: None,
            copy_called: false,
            queue_change: None,
            queue_round: false,
            queue_prioritize: None,
            queue_quash: None,
            queue_instruct: None,
            in_move: false,
            queue_cancel: Vec::new(),
            wonder_room_seen: false,
            side_one,
            side_two,
            weather: StateWeather {
                weather_type: Weather::NONE,
                turns_remaining: -1,
            },
            terrain: StateTerrain {
                terrain_type: Terrain::NONE,
                turns_remaining: 0,
            },
            trick_room: StateTrickRoom {
                active: false,
                turns_remaining: 0,
            },
            wonder_room: StateTrickRoom {
                active: false,
                turns_remaining: 0,
            },
            magic_room: StateTrickRoom {
                active: false,
                turns_remaining: 0,
            },
            gravity: StateTrickRoom {
                active: false,
                turns_remaining: 0,
            },
            fairy_lock: StateTrickRoom {
                active: false,
                turns_remaining: 0,
            },
            turn: 1,
            unresolved_faints: Vec::new(),
            faint_disabled: Vec::new(),
            db_alive_pending: Vec::new(),
            pending_disguise: Vec::new(),
            faint_log: Vec::new(),
            pending_after_faint: None,
        };
        state
    }

    /// Send out the lead Pokémon into the two active slots.
    pub fn send_out_leads(&mut self, leads_a: &[usize], leads_b: &[usize]) {
        // PS starts the battle with an end-of-turn advance: the leads already
        // have `activeTurns` 1 at the first residual (Speed Boost fires).
        for (slot, &lead_pos) in leads_a.iter().take(2).enumerate() {
            let idx = PokemonIndex::deserialize(&lead_pos.to_string());
            self.side_one.active[slot].pokemon_index = Some(idx);
            self.side_one.active[slot].active_turns = 1;
        }
        for (slot, &lead_pos) in leads_b.iter().take(2).enumerate() {
            let idx = PokemonIndex::deserialize(&lead_pos.to_string());
            self.side_two.active[slot].pokemon_index = Some(idx);
            self.side_two.active[slot].active_turns = 1;
        }
    }
}

#[derive(Debug, Serialize)]
pub struct SlotHp {
    pub name: String,
    pub hp: i16,
    pub maxhp: i16,
    pub fainted: bool,
}

#[derive(Debug, Serialize)]
pub struct TurnHp {
    pub a: Vec<Option<SlotHp>>,
    pub b: Vec<Option<SlotHp>>,
}

/// Full-team per-turn HP snapshot (all team members, not just actives), used by
/// the replay test to compare the HP trajectory against pokemon-showdown.
#[derive(Debug, Serialize)]
pub struct ReplayTurnHp {
    pub turn: u32,
    pub a: Vec<SlotHp>,
    pub b: Vec<SlotHp>,
}

// ---------------------------------------------------------------------------
// Fuzz battle output: full-state per-turn snapshots
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct Boosts {
    pub atk: i8,
    pub def: i8,
    pub spa: i8,
    pub spd: i8,
    pub spe: i8,
    pub acc: i8,
    pub eva: i8,
}

impl Default for Boosts {
    fn default() -> Self {
        Self {
            atk: 0,
            def: 0,
            spa: 0,
            spd: 0,
            spe: 0,
            acc: 0,
            eva: 0,
        }
    }
}

/// One team member's full state (in team position order, 0-5).
#[derive(Debug, Clone, Serialize)]
pub struct FullSlot {
    pub species: String,
    pub hp: i16,
    pub maxhp: i16,
    pub fainted: bool,
    /// Normalized status: none | burn | sleep | freeze | paralyze | poison | toxic
    pub status: String,
    /// Active slot index (0/1) if on field, else null.
    pub active_slot: Option<usize>,
    pub boosts: Boosts,
}

/// Side conditions as (id, duration) pairs, ids lowercase (e.g. "reflect").
#[derive(Debug, Clone, Serialize)]
pub struct SideConditionEntry {
    pub id: String,
    /// `None` for conditions without a duration (hazards): PS reports none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<i8>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FuzzTurnState {
    pub turn: u32,
    pub a: Vec<FullSlot>,
    pub b: Vec<FullSlot>,
    pub weather: String,
    pub terrain: String,
    pub trick_room: bool,
    /// Active pseudo-weather ids (sorted), e.g. ["trickroom"], ["tailwind"].
    pub pseudo_weather: Vec<String>,
    pub a_side_conditions: Vec<SideConditionEntry>,
    pub b_side_conditions: Vec<SideConditionEntry>,
    /// Execution order this turn, entries "SIDE<slot>:Species:movename".
    pub move_order: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FuzzBattleResult {
    pub winner: String,
    pub total_turns: u32,
    pub turns: Vec<FuzzTurnState>,
    /// Set when the engine hit an internal problem (e.g. exhausted choices).
    #[serde(default)]
    pub error: Option<String>,
    /// Every PRNG draw made during the battle (diagnostics for draw-order
    /// parity with pokemon-showdown).
    #[serde(default)]
    pub prng_draws: Vec<u32>,
    /// What each draw was used for: `"<label>|<context>"` (e.g.
    /// `"crit|turn 2: B1 Weavile ICESHARD"`), parallel to `prng_draws`.
    #[serde(default)]
    pub prng_draw_ctx: Vec<String>,
}

pub fn normalize_status(s: PokemonStatus) -> &'static str {
    match s {
        PokemonStatus::NONE => "none",
        PokemonStatus::BURN => "burn",
        PokemonStatus::SLEEP => "sleep",
        PokemonStatus::FREEZE => "freeze",
        PokemonStatus::PARALYZE => "paralyze",
        PokemonStatus::POISON => "poison",
        PokemonStatus::TOXIC => "toxic",
    }
}

/// The side conditions currently up (fixed order; see `sc_order`).
pub fn side_condition_entries(sc: &SideConditions) -> Vec<SideConditionEntry> {
    let mut v = vec![];
    if sc.reflect != 0 {
        v.push(SideConditionEntry { id: "reflect".into(), duration: Some(sc.reflect) });
    }
    if sc.light_screen != 0 {
        v.push(SideConditionEntry { id: "lightscreen".into(), duration: Some(sc.light_screen) });
    }
    if sc.aurora_veil != 0 {
        v.push(SideConditionEntry { id: "auroraveil".into(), duration: Some(sc.aurora_veil) });
    }
    if sc.tailwind != 0 {
        v.push(SideConditionEntry { id: "tailwind".into(), duration: Some(sc.tailwind) });
    }
    if sc.safeguard != 0 {
        v.push(SideConditionEntry { id: "safeguard".into(), duration: Some(sc.safeguard) });
    }
    if sc.lucky_chant != 0 {
        v.push(SideConditionEntry { id: "luckychant".into(), duration: Some(sc.lucky_chant) });
    }
    if sc.spikes != 0 {
        v.push(SideConditionEntry { id: "spikes".into(), duration: None });
    }
    if sc.stealth_rock != 0 {
        v.push(SideConditionEntry { id: "stealthrock".into(), duration: None });
    }
    if sc.sticky_web != 0 {
        v.push(SideConditionEntry { id: "stickyweb".into(), duration: None });
    }
    if sc.toxic_spikes != 0 {
        v.push(SideConditionEntry { id: "toxicspikes".into(), duration: None });
    }
    if sc.protect != 0 {
        v.push(SideConditionEntry { id: "protect".into(), duration: Some(sc.protect) });
    }
    if sc.quick_guard != 0 {
        v.push(SideConditionEntry { id: "quickguard".into(), duration: Some(sc.quick_guard) });
    }
    if sc.wide_guard != 0 {
        v.push(SideConditionEntry { id: "wideguard".into(), duration: Some(sc.wide_guard) });
    }
    v
}

/// Snapshot the full state of both sides for fuzz comparison.
pub fn fuzz_snapshot(state: &DoublesState, move_order: Vec<String>) -> FuzzTurnState {
    let snap_side = |s: &DoublesSide| -> Vec<FullSlot> {
        s.team
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                // Skip empty team slots (team preview picks 4 of 6).
                !s.species_names[*i].is_empty()
                    || s.team[*i].id != crate::pokemon::PokemonName::NONE
            })
            .map(|(i, p)| {
                let active_slot = s
                    .active
                    .iter()
                    .position(|sl| sl.pokemon_index == Some(PokemonIndex::deserialize(&i.to_string())));
                let boosts = {
                    let b = &s.team_boosts[i];
                    Boosts {
                        atk: b.atk,
                        def: b.def,
                        spa: b.spa,
                        spd: b.spd,
                        spe: b.spe,
                        acc: b.acc,
                        eva: b.eva,
                    }
                };
                FullSlot {
                    species: s.species_names[i].clone(),
                    hp: p.hp,
                    maxhp: p.maxhp,
                    fainted: p.hp <= 0,
                    status: normalize_status(p.status).to_string(),
                    active_slot,
                    boosts,
                }
            })
            .collect()
    };
    let sc_entries = |sc: &SideConditions, order: &[String]| -> Vec<SideConditionEntry> {
        // The harness lists `sideConditions` sorted by id.
        let _ = order;
        let mut v = side_condition_entries(sc);
        v.sort_by(|a, b| a.id.cmp(&b.id));
        v
    };
    let mut pseudo_weather: Vec<String> = vec![];
    if state.trick_room.active {
        pseudo_weather.push("trickroom".into());
    }
    if state.wonder_room.active {
        pseudo_weather.push("wonderroom".into());
    }
    if state.magic_room.active {
        pseudo_weather.push("magicroom".into());
    }
    if state.gravity.active {
        pseudo_weather.push("gravity".into());
    }
    if state.fairy_lock.active {
        pseudo_weather.push("fairylock".into());
    }
    pseudo_weather.sort();
    // pokemon-showdown reports the weather CONDITION id (e.g. "raindance",
    // "sunnyday", "sandstorm"), not the engine's enum name.
    let weather_name = match state.weather.weather_type {
        Weather::NONE => "none",
        Weather::SUN => "sunnyday",
        Weather::HARSHSUN => "harshsun",
        Weather::RAIN => "raindance",
        Weather::HEAVYRAIN => "heavyrain",
        Weather::SAND => "sandstorm",
        Weather::HAIL => "hail",
        Weather::SNOW => "snowscape",
        Weather::DELTASTREAM => "deltastream",
    };
    FuzzTurnState {
        turn: state.turn,
        a: snap_side(&state.side_one),
        b: snap_side(&state.side_two),
        weather: weather_name.to_string(),
        terrain: format!("{}", state.terrain.terrain_type).to_lowercase(),
        trick_room: state.trick_room.active,
        pseudo_weather,
        a_side_conditions: sc_entries(&state.side_one.side_conditions, &state.side_one.sc_order),
        b_side_conditions: sc_entries(&state.side_two.side_conditions, &state.side_two.sc_order),
        move_order,
    }
}

#[derive(Debug, Serialize)]
pub struct BattleResult {
    pub winner: String,
    pub total_turns: u32,
    pub prng_draws: Vec<u32>,
    pub turns: Vec<TurnEntry>,
    #[serde(default)]
    pub replay_hp: Vec<ReplayTurnHp>,
}

#[derive(Debug, Serialize)]
pub struct TurnEntry {
    pub turn: u32,
    pub order: Vec<String>,
    pub moves: Vec<MoveResult>,
    pub switches: Vec<SwitchResult>,
    pub hp: TurnHp,
}

#[derive(Debug, Serialize)]
pub struct MoveResult {
    pub attacker: String,
    pub move_name: String,
    pub target: String,
    pub damage: i16,
    pub crit: bool,
    pub missed: bool,
    #[serde(default)]
    pub dmg_draw: Option<u32>,
}

#[derive(Debug, Serialize)]
pub struct SwitchResult {
    pub side: String,
    pub slot: usize,
    pub in_pokemon: String,
}

/// A Pokémon's effective types. Filled in from the species data at setup.
impl Pokemon {
    pub fn set_types(&mut self, t0: PokemonType, t1: PokemonType) {
        self.types = (t0, t1);
        self.base_types = (t0, t1);
    }
}



#[cfg(test)]
mod oddish_test {
    use super::*;
    use crate::state::PokemonNature;
    #[test]
    fn check() {
        let evs = Evs { hp: 11, atk: 11, def: 11, spa: 11, spd: 11, spe: 11 };
        let oddish_base = (45, 49, 65, 49, 65, 30);
        let s = compute_stats(oddish_base, &evs, 50, PokemonNature::SERIOUS);
        eprintln!("Oddish stats (hp,atk,def,spa,spd,spe): {:?}", s);
    }
}


impl DoublesSide {
    /// Keep `sc_order` in step with the conditions currently up: new ones are
    /// appended, ended ones dropped.
    pub fn sync_sc_order(&mut self) {
        let present: Vec<String> = side_condition_entries(&self.side_conditions).into_iter().map(|e| e.id).collect();
        self.sc_order.retain(|o| present.contains(o));
        for id in present {
            if !self.sc_order.contains(&id) {
                self.sc_order.push(id);
            }
        }
    }
}
