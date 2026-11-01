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
    /// Switching out from Eject Button (PS `switchFlag === true`).
    pub eject_switch: bool,
    /// Eject Pack `effectState.eject`: a stat was lowered, the pack fires at
    /// the next AfterMove.
    pub eject_pack: bool,
    pub force_trapped: bool,
    pub slow_uturn_move: bool,
    pub baton_passing: bool,
    /// Electrify (single turn): the next move becomes Electric.
    pub electrify: bool,
    /// Power Trick is active (Attack and Defense are swapped).
    pub power_trick: bool,
    /// Attack and Defense are swapped now (Power Trick). A Mega Evolution
    /// recomputes them unswapped while the volatile stays.
    pub pt_swapped: bool,
    /// PS `move.hit` (1-based) of the current hit of a multi-hit move.
    pub hit_number: u8,
    pub shed_tailing: bool,
    /// Unburden's volatile: the holder lost its item (Speed x2).
    pub unburden: bool,
    /// Slow Start's `abilityState.counter` (turns left at x0.5 Atk / Spe).
    pub slow_start: u8,
    /// The `gem` volatile: a Normal Gem powers up this move (x1.3).
    pub gem: bool,
    /// The running move is Parental Bond's two-hit version.
    pub parental_bond: bool,
    /// Metronome (item) volatile: the last move and its consecutive uses.
    pub metronome_last: Option<Choices>,
    pub metronome_count: u8,
    /// The Metronome item's `metronome` volatile is up (item `onStart`).
    pub metronome_vol: bool,
    /// Cud Chew `effectState`: the berry eaten and the residuals left until it
    /// is eaten again.
    pub cud_chew: Option<(Items, u8)>,
    /// Ripen `abilityState.berryWeaken` (a `Cell`: the damage calc, which
    /// holds the state shared, consumes it).
    pub berry_weaken: std::cell::Cell<bool>,
    /// PS `usedItemThisTurn` (Pickup).
    pub used_item_this_turn: bool,
    /// The terrain a terrain seed last checked (its `onStart` /
    /// `onTerrainChange`); `None` until the holder's item starts.
    pub seed_terrain: Option<crate::engine::state::Terrain>,
    /// Choice item `choicelock` volatile: the locked move.
    pub choice_lock: Option<Choices>,
    /// PS `moveThisTurnResult = null` (a NOT_FAIL hit: Protect), this / last turn.
    pub move_result_null: bool,
    pub move_null_last_turn: bool,
    /// The item this slot's Fling threw (its base power and effect).
    pub fling_item: Items,
    /// Revival Blessing hit: the slot's switch answer revives a fainted
    /// party member instead of switching (PS `revivalblessing` slot condition).
    pub reviving: bool,
    /// Flinched by the previous move's secondary effect; cannot act this turn.
    pub flinched: bool,
    /// Used a move with the `recharge` flag (Hyper Beam, Solar Beam, ...);
    /// must recharge this turn.
    pub recharging: bool,
    /// Turn ends `recharging` has lived through (PS `mustrecharge` duration 2:
    /// it ends at the second residual even if the turn was never spent).
    pub recharge_age: u8,
    /// Two-turn move charging (Dig, Meteor Beam, ...): the move id being
    /// charged and how many charge turns remain before it fires. Mirrors
    /// pokemon-showdown's `twoturnmove` / per-move volatile pair.
    pub charging: Option<(crate::choices::Choices, u8)>,
    /// Signed PS target location chosen on the charge turn; the locked follow-up turn keeps it
    /// (PS `twoturnmove.onStart` stores `targetLoc` on the move volatile).
    pub charge_target_loc: i8,
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
    /// Shell Trap's `shelltrap` volatile (duration 1): 1 set, 2 `gotHit` by a
    /// foe's physical move. PS data/moves.ts shelltrap.condition.
    pub shell_trap: u8,
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
    /// Zen Mode's `zenmode` volatile (no handlers of its own): up while the
    /// Darmanitan is in its Zen forme from its ability.
    pub zen_mode: bool,
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
    /// The move being run was taken by Snatch (PS `move.sourceEffect ===
    /// 'snatch'`): no second Snatch.
    pub snatched: bool,
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
    /// `telekinesis` volatile duration (3): floats; non-OHKO moves always hit.
    pub telekinesis: u8,
    /// `rollout` volatile (Rollout / Ice Ball lock): (hitCount, contactHitCount,
    /// duration, the base-power multiplier of the current hit).
    pub rollout: Option<(u8, u8, u8, u8)>,
    /// The `rollout` volatile is Ice Ball's (`onLockMove: 'iceball'`), not Rollout's.
    pub rollout_ice: bool,
    /// `defensecurl` volatile (doubles Rollout / Ice Ball).
    pub defense_curl: bool,
    /// Mimic replaced move slot `k` (Mimic's id and PP saved, restored when the
    /// pokemon leaves: PS `moveSlots` vs `baseMoveSlots`).
    pub mimic: Option<(u8, crate::choices::Choices, i8)>,
    /// Lifted by a Sky Drop user (its side and team index): can't act, and is
    /// semi-invulnerable like the user.
    pub sky_lifted_by: Option<(Side, crate::state::PokemonIndex)>,
    /// Rage's `rage` volatile: +1 Atk when hit by a damaging move.
    pub rage: bool,
    /// Magic Coat's single-turn volatile (reflects like Magic Bounce).
    pub magic_coat: bool,
    /// Snatch's single-turn volatile: takes the next snatchable move used by
    /// any pokemon (PS data/moves.ts snatch.condition).
    pub snatch: bool,
    /// Grudge's volatile: fainting to a move zeroes that move's PP.
    pub grudge: bool,
    /// Micle Berry's `micleberry` volatile duration (next accuracy x1.2).
    pub micle: u8,
    /// `embargo` volatile duration (5): the holder's item is ignored.
    pub embargo: u8,
    /// `foresight` volatile (Foresight / Odor Sleuth).
    pub foresight: bool,
    /// `miracleeye` volatile.
    pub miracle_eye: bool,
    /// Protosynthesis / Quark Drive volatile: (best stat 0..4 = Atk, Def, SpA,
    /// SpD, Spe; from Booster Energy).
    pub paradox: Option<(u8, bool)>,
    /// The paradox volatile is Quark Drive's (else Protosynthesis's).
    pub paradox_quark: bool,
    /// `bide` volatile: (duration, total move damage taken, last damaging pokemon).
    pub bide: Option<(u8, i16, Option<(Side, crate::state::PokemonIndex)>)>,
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
    /// Mirror Herb's `effectState.boosts` (the same copy as Opportunist).
    pub mirror_herb_boosts: [i8; 7],
    /// Me First's `mefirst` volatile: the called move has x1.5 power.
    pub me_first: bool,
    /// The `mefirst` volatile itself (duration 1, ended at the residual):
    /// reported while the turn lasts. PS data/moves.ts mefirst.condition.
    pub me_first_vol: bool,
    /// Illusion's disguise, fixed at switch-in (`onBeforeSwitchIn`): the last
    /// unfainted pokemon after the holder then. `Some(None)` = no disguise.
    pub illusion_target: Option<Option<usize>>,
    /// `curse` volatile (Ghost Curse): loses 1/4 max HP each residual (order 12).
    pub cursed: bool,
    /// `nightmare` volatile: a sleeping holder loses 1/4 max HP each residual
    /// (order 11); it ends when the holder's sleep ends.
    pub nightmare: bool,
    /// `tarshot` volatile: Fire moves get one more effectiveness step
    /// (`onEffectiveness`). PS data/moves.ts tarshot.condition.
    pub tar_shot: bool,
    /// `powershift` volatile: Attack and Defense swapped (PS data/moves.ts
    /// powershift.condition; no `onRestart`, so a second use fails).
    pub power_shift: bool,
    /// Autotomize lowered `weighthg`; `clearVolatile` -> `setSpecies` restores it.
    pub autotomized: bool,
    /// Commander's `commanding` volatile (Tatsugiri inside its Dondozo ally):
    /// skips its actions, moves miss it, cannot be dragged out.
    /// PS data/conditions.ts commanding.
    pub commanding: bool,
    /// Commander's `commanded` volatile (Dondozo): the Tatsugiri's team index.
    /// PS data/conditions.ts commanded.
    pub commanded: Option<PokemonIndex>,
    /// `powder` volatile (duration 1): the holder's Fire move fails and hurts
    /// it 1/4 max HP (`onTryMove`). PS data/moves.ts powder.condition.
    pub powder: bool,
    /// Set by the executing move's base-power callback precondition (Payback).
    pub bp_double: bool,
    /// PS `beingCalledBack`: set by `switchIn` on the pokemon leaving the
    /// field, while Pursuit (`onFoeBeforeSwitchOut`) hits it.
    pub being_called_back: bool,
    /// `followme` (1) / `ragepowder` (2) volatile (duration 1): redirects foes' moves.
    pub follow_me: u8,
    /// Which protect-like move set `protected`: 0 Protect/Detect, 1 King's
    /// Shield, 2 Spiky Shield, 3 Baneful Bunker, 4 Silk Trap, 5 Burning Bulwark.
    pub protect_kind: u8,
    /// `perishsong` volatile: remaining duration (0 = none); faints at 0.
    pub perish: u8,
    /// PS ids of the volatiles in insertion order (PS `pokemon.volatiles` key
    /// order), kept by `sync_vol_order`. Orders the Residual handlers.
    pub vol_order: Vec<&'static str>,
    /// The volatile conditions (`events::vol_mask`) `vol_order` was last
    /// synced with (`sync_vol_order`); `u64::MAX` forces a sync.
    pub vol_synced: u128,
    /// `flashfire` volatile: Fire moves x1.5 (ModifyAtk / ModifySpA).
    pub flash_fire: bool,
    /// `charge` volatile: the next Electric move has doubled power.
    pub charge: bool,
    /// `noretreat` volatile (No Retreat already used).
    pub no_retreat: bool,
    /// The `trapped` volatile No Retreat's `onStart` added itself (none when the
    /// user was already trapped: that link's `trapped` stays, and goes with it).
    pub no_retreat_trap: bool,
    /// Truant's `truant` volatile: the next move is skipped (`cant`).
    pub truant: bool,
    /// `laserfocus` volatile duration (2 when used; the next move crits).
    pub laser_focus: u8,
    /// `furycutter` volatile: (multiplier, duration).
    pub fury_cutter: Option<(u8, u8)>,
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
    /// The `roost` volatile itself (duration 1), also on a pokemon with no
    /// Flying type to filter (its residual handler still exists).
    pub roost: bool,
    /// `aquaring` volatile: heals 1/16 max HP each residual (order 6).
    pub aqua_ring: bool,
    /// `counter` / `mirrorcoat` volatiles (duration 1): the last foe slot that
    /// hit this pokemon with a physical / special move, and twice the damage.
    pub counter_vol: Option<(Option<(Side, usize)>, i16)>,
    pub mirror_coat_vol: Option<(Option<(Side, usize)>, i16)>,
    /// `pursuit` volatile (duration 1), added by Pursuit's `beforeTurnCallback`.
    pub pursuit_vol: bool,
    /// `spotlight` volatile (duration 1): foes' single-target moves are
    /// redirected to this pokemon (`onFoeRedirectTarget`, priority 2).
    pub spotlight: bool,
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
    /// PS `lastMoveUsed.type`: the type of the last move this pokemon used,
    /// after `ModifyType` (Conversion 2 reads it). Cleared on switch-out.
    pub last_move_used_type: Option<PokemonType>,
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
            eject_switch: false,
            eject_pack: false,
            force_trapped: false,
            slow_uturn_move: false,
            baton_passing: false,
            electrify: false,
            power_trick: false,
            pt_swapped: false,
            hit_number: 1,
            shed_tailing: false,
            unburden: false,
            slow_start: 0,
            gem: false,
            parental_bond: false,
            metronome_last: None,
            metronome_count: 0,
            metronome_vol: false,
            cud_chew: None,
            berry_weaken: std::cell::Cell::new(false),
            used_item_this_turn: false,
            seed_terrain: None,
            choice_lock: None,
            move_result_null: false,
            move_null_last_turn: false,
            fling_item: Items::NONE,
            reviving: false,
            flinched: false,
            recharging: false,
            recharge_age: 0,
            charging: None,
            charge_target_loc: 0,
            salt_cure: false,
            stockpile: 0,
            confusion: 0,
            drag_out: false,
            focus_punch: false,
            beak_blast: false,
            shell_trap: 0,
            gastro_acid: false,
            gastro_ability: crate::engine::abilities::Abilities::NONE,
            chilly_reception: false,
            zen_mode: false,
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
            snatched: false,
            bp_fixed: 0.0,
            beat_up_queue: Vec::new(),
            fallen: 0,
            taunt: 0,
            magnet_rise: 0,
            telekinesis: 0,
            rollout: None,
            rollout_ice: false,
            defense_curl: false,
            mimic: None,
            sky_lifted_by: None,
            rage: false,
            magic_coat: false,
            snatch: false,
            grudge: false,
            micle: 0,
            embargo: 0,
            foresight: false,
            miracle_eye: false,
            paradox: None,
            paradox_quark: false,
            bide: None,
            smackdown: false,
            active_turns: 0,
            active_move_actions: 0,
            illusion_broken: false,
            opportunist_boosts: [0; 7],
            mirror_herb_boosts: [0; 7],
            me_first: false,
            me_first_vol: false,
            illusion_target: None,
            cursed: false,
            nightmare: false,
            tar_shot: false,
            power_shift: false,
            autotomized: false,
            commanding: false,
            commanded: None,
            powder: false,
            bp_double: false,
            being_called_back: false,
            follow_me: 0,
            protect_kind: 0,
            perish: 0,
            vol_order: Vec::new(),
            vol_synced: 0,
            flash_fire: false,
            charge: false,
            no_retreat: false,
            no_retreat_trap: false,
            truant: false,
            laser_focus: 0,
            fury_cutter: None,
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
            roost: false,
            aqua_ring: false,
            counter_vol: None,
            mirror_coat_vol: None,
            pursuit_vol: false,
            spotlight: false,
            move_result_this_turn: None,
            move_result_last_turn: None,
            ally_switch: None,
            lock_on: None,
            uproar: 0,
            stall: None,
            endure: false,
            last_target_loc: 0,
            last_move_used_type: None,
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

/// What Transform overwrote (PS `transformInto`), and the species copied.
#[derive(Debug, Clone)]
pub struct TransformSave {
    pub id: PokemonName,
    pub moves: crate::state::PokemonMoves,
    pub types: (crate::state::PokemonType, crate::state::PokemonType),
    /// atk, def, spa, spd, spe.
    pub stats: [i16; 5],
    pub weight_kg: f32,
    /// The species name the pokemon now shows (the target's).
    pub as_species: &'static str,
}

/// A slot's cached event-handler effect bits and the inputs outside the
/// slot they depend on (status, ability, Future Sight / Wish).
#[derive(Debug, Clone, Copy)]
pub(crate) struct FxCache {
    pub status: crate::state::PokemonStatus,
    pub ability: crate::engine::abilities::Abilities,
    pub item: crate::engine::items::Items,
    pub extra: u8,
    pub fx: crate::doubles::events::FxMask,
}

#[derive(Debug, Clone)]
pub struct DoublesSide {
    pub team: [Pokemon; 6],
    slots: [ActiveSlot; 2],
    /// Per slot: the volatile conditions (`events::vol_mask`) as of the last
    /// read; cleared by `active_mut` / `active_slot_mut`.
    cond_cache: [std::cell::Cell<Option<u128>>; 2],
    /// Per slot: the event-handler effect bits of the occupant
    /// (`events::slot_fx`), valid while the key fields still match. Cleared
    /// with `cond_cache`.
    pub(crate) fx_cache: [std::cell::Cell<Option<FxCache>>; 2],
    /// Bumped by every change to the slots or team members (`active_mut`,
    /// `active_slot_mut`, `team_member_mut`, direct `team` writes): keys
    /// `DoublesState::fx_union`.
    pub(crate) ver: u32,
    pub side_conditions: SideConditions,
    /// Original species names (including formes) for each team slot, used for
    /// name-based target matching in the replay test.
    pub species_names: [&'static str; 6],
    /// Gender per team member: "M" / "F" / None (genderless). Used by Attract.
    pub genders: [Option<&'static str>; 6],
    /// Stat boosts per team member (see `TeamBoosts`).
    pub team_boosts: [TeamBoosts; 6],
    /// Toxic counter per team member (0 = not toxic; reset on switch-in).
    pub toxic_stage: [i8; 6],
    /// PS `pokemon.speed`: the speed cached at the last `updateSpeed()`
    /// (turn start, residual, queue re-sort). `eachEvent`, `runSwitch` and
    /// Residual handlers sort by it; a fresh switch-in keeps a stale value.
    pub speed_cache: [i16; 6],
    /// PS `abilityState.effectOrder` per team member: when its ability state
    /// started (switch-in). Breaks speed ties in redirect order
    /// (`compareRedirectOrder`).
    pub ability_order: [u32; 6],
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
    pub sc_order: Vec<&'static str>,
    /// Supersweet Syrup already triggered (PS `syrupTriggered`, once per
    /// battle per pokemon).
    pub syrup_triggered: [bool; 6],
    /// A Mega Evolution was used (PS: `canMegaEvo = false` for the whole side).
    pub mega_used: bool,
    /// PS `side.faintedThisTurn` / `faintedLastTurn` (Retaliate).
    pub fainted_this_turn: bool,
    pub fainted_last_turn: bool,
    /// PS `pokemon.swordBoost` / `shieldBoost`: Intrepid Sword / Dauntless Shield
    /// boost once per battle.
    pub once_boosted: [bool; 6],
    /// PS `pokemon.lastItem` per team member (Recycle, Harvest).
    pub last_item: [Items; 6],
    /// PS `pokemon.ateBerry` per team member (Belch).
    pub ate_berry: [bool; 6],
    /// Healing Wish / Lunar Dance slot conditions per active slot (PS
    /// `slotCondition: 'healingwish' | 'lunardance'`): they wait for the next
    /// pokemon switched into that slot.
    pub healing_wish_slot: [bool; 2],
    /// The slot's `healing_wish_slot` condition is Lunar Dance (also restores PP).
    pub lunar_dance_slot: [bool; 2],
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
    /// A transformed team member's own species data, restored when it leaves
    /// the field (PS `transformed`, `clearVolatile` -> `setSpecies(baseSpecies)`).
    pub transform_save: [Option<Box<TransformSave>>; 6],
}

impl DoublesSide {
    pub fn team_member(&self, index: PokemonIndex) -> &Pokemon {
        &self.team[index as usize]
    }
    /// A team member, for changes to fields other than status, ability and
    /// HP (PP, `attacked_by`, ...): `ver` is not bumped.
    #[inline]
    pub fn team_member_scratch_mut(&mut self, index: PokemonIndex) -> &mut Pokemon {
        &mut self.team[index as usize]
    }
    /// Write PS `pokemon.speed` (`speed_cache`); a change bumps `ver`.
    #[inline]
    pub fn set_speed_cache(&mut self, index: usize, speed: i16) {
        if self.speed_cache[index] != speed {
            self.speed_cache[index] = speed;
            self.ver = self.ver.wrapping_add(1);
        }
    }
    pub fn team_member_mut(&mut self, index: PokemonIndex) -> &mut Pokemon {
        self.ver = self.ver.wrapping_add(1);
        &mut self.team[index as usize]
    }

    /// The two on-field slots.
    #[inline]
    pub fn active(&self) -> &[ActiveSlot; 2] {
        &self.slots
    }

    /// The two on-field slots, for changes.
    #[inline]
    pub fn active_mut(&mut self) -> &mut [ActiveSlot; 2] {
        self.cond_cache = Default::default();
        self.fx_cache = Default::default();
        self.ver = self.ver.wrapping_add(1);
        &mut self.slots
    }

    /// One on-field slot, for changes (only its cache entry is cleared).
    #[inline]
    pub fn active_slot_mut(&mut self, slot: usize) -> &mut ActiveSlot {
        self.cond_cache[slot].set(None);
        self.fx_cache[slot].set(None);
        self.ver = self.ver.wrapping_add(1);
        &mut self.slots[slot]
    }

    /// One on-field slot, for changes to fields that `events::vol_mask` does
    /// not read (its cached mask is kept). Debug builds check every cache hit,
    /// so a volatile write through here fails the debug fuzz run.
    #[inline]
    pub fn slot_scratch_mut(&mut self, slot: usize) -> &mut ActiveSlot {
        &mut self.slots[slot]
    }

    /// `compute(slot)`, cached until the slot is next borrowed mutably.
    #[inline]
    pub fn cached_conds(&self, slot: usize, compute: impl Fn(&ActiveSlot) -> u128) -> u128 {
        if let Some(v) = self.cond_cache[slot].get() {
            debug_assert_eq!(v, compute(&self.slots[slot]), "stale cond_cache");
            return v;
        }
        let v = compute(&self.slots[slot]);
        self.cond_cache[slot].set(Some(v));
        v
    }

    /// The Pokémon in an active slot, if any.
    pub fn slot_pokemon(&self, slot: usize) -> Option<&Pokemon> {
        self.active()[slot].pokemon_index.map(|i| self.team_member(i))
    }
    pub fn slot_pokemon_mut(&mut self, slot: usize) -> Option<&mut Pokemon> {
        let idx = self.active()[slot].pokemon_index?;
        Some(self.team_member_mut(idx))
    }

    /// Team-member boosts for a slot, if occupied.
    pub fn slot_boosts(&self, slot: usize) -> Option<&TeamBoosts> {
        let i = self.active()[slot].pokemon_index?;
        Some(&self.team_boosts[i as usize])
    }
    pub fn slot_boosts_mut(&mut self, slot: usize) -> Option<&mut TeamBoosts> {
        let i = self.active()[slot].pokemon_index?;
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
                    .active()
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

/// Per-action scratch flags of the running move (PS battle-level fields such
/// as `activeTarget`); `Cell`s so read-only helpers can set them.
#[derive(Debug, Clone, Default)]
pub struct MoveScratch {
    /// PS `queue.willMove(target)` is false for every other active: Analytic.
    pub analytic_no_move: std::cell::Cell<bool>,
    /// Slots (bit `side * 2 + slot`) with a move still queued this turn (`queue.willMove`).
    pub will_move: std::cell::Cell<u8>,
    /// Slots a Piercing Drill / Unseen Fist hit reached through Protect this move.
    pub bypass_protect: std::cell::Cell<u8>,
    /// Slots (bit `side * 2 + slot`) whose Tera Shell resisted this move's
    /// first hit (PS `abilityState.resisted`): its later hits stay resisted.
    pub tera_shell_resisted: std::cell::Cell<u8>,
    /// Focus Band rolls a spread hit already drew (bit slot: rolled, bit slot + 4: saved).
    pub band_pre: std::cell::Cell<u8>,
    /// The residual action is running (PS `!queue.peek()`).
    pub in_residual: std::cell::Cell<bool>,
    /// The residual `fieldEvent` handler loop is running (draw labels).
    pub in_field_event: std::cell::Cell<bool>,
    /// The `selfdestruct: 'always'` user whose move is running (fainted for PS).
    pub selfdestructing: std::cell::Cell<Option<(Side, crate::state::PokemonIndex)>>,
    /// The running move's Pressure PP (side, slot, user, move slot, move), still
    /// owed until `pressure_done`: PS takes it before the hit, so a Leppa Berry
    /// eaten mid-move sees it.
    pub pressure_owed: std::cell::Cell<Option<(crate::doubles::state::Side, usize, crate::state::PokemonIndex, crate::state::PokemonMoveIndex, crate::choices::Choices)>>,
    pub pressure_done: std::cell::Cell<bool>,
    /// The caller move (user side, user, move slot, move) of a move run by
    /// `useMove` from a move's handler (Mirror Move, Sleep Talk, ...): PS
    /// `useMoveInner` `callerMoveForPressure` takes the called move's Pressure
    /// PP from the caller's slot.
    pub pressure_caller: std::cell::Cell<Option<(Side, crate::state::PokemonIndex, crate::state::PokemonMoveIndex, crate::choices::Choices)>>,
    /// Slots (bit `side * 2 + slot`) whose Berserk / Anger Shell has not yet
    /// checked this move's damage (`checkedBerserk` false): no healing berry.
    pub berserk_pending: std::cell::Cell<u8>,
    /// The target the running move resolved to (after any retarget / redirect),
    /// as PS `getMoveTargets` returns it: Pressure's `pressureTargets`. `None`
    /// until `execute_action_inner` resolves it for this action.
    pub pressure_target: std::cell::Cell<Option<Option<(Side, usize)>>>,
    /// Pressure PP extra counted when the target was resolved (PS `pressureTargets`
    /// are read before the hit, so a target that swaps its ability mid-move still
    /// costs PP). -1 when not yet resolved.
    pub pressure_extra: std::cell::Cell<i8>,
    /// The running move's `AfterMove` handler sort was already drawn by
    /// `execute_action_inner` (before a duration-1 lockedmove release, whose
    /// `onEnd` confusion must come after the sort). Reset per action.
    pub after_move_tie_drawn: std::cell::Cell<bool>,
    /// The running move found no targets (PS useMoveInner `!targets.length`):
    /// it returns false without `MoveFail`, so no crash / Steel Beam fail damage.
    pub no_target_fail: std::cell::Cell<bool>,
    /// The running move reached its accuracy step (PS `hitStepAccuracy`):
    /// Micle Berry's `onSourceAccuracy` only ends there, not on an earlier fail.
    pub acc_step: std::cell::Cell<bool>,
    /// PS `battle.activeTarget` while a spread move's Substitute `onTryPrimaryHit`
    /// computes its damage (set only then): the last target of the previous
    /// per-target loop, not the substitute holder. Unaware reads it.
    pub sub_active_target: std::cell::Cell<Option<(Side, usize)>>,
    /// The target that ate its resist berry in the last damage calculation
    /// (PS `onSourceModifyDamage` -> `eatItem`); consumed by the caller.
    pub berry_eaten: std::cell::Cell<Option<(Side, usize)>>,
    /// The running move stopped in its BeforeMove handlers (PS runMove
    /// `!willTryMove`), so `runEvent('MoveAborted')` follows.
    pub before_move_aborted: std::cell::Cell<bool>,
    /// Set while Opportunist applies its own copied boosts (PS `effect.name ===
    /// 'Opportunist'`): those raises are not copied again.
    pub opportunist_busy: std::cell::Cell<bool>,
    /// Slots (bit `side * 2 + slot`) a spread hit KOs after the target whose
    /// effects are running: PS `spreadDamage` already left them at 0 HP, so
    /// their abilities no longer act (Flower Veil).
    pub spread_ko_later: std::cell::Cell<u8>,
    /// A spread hit's Substitute drain heal was already applied in the
    /// damage pre-pass (`substitute.onTryPrimaryHit`), so the per-target loop skips it.
    pub sub_drain_done: std::cell::Cell<bool>,
    /// The running move is a Dancer copy (PS `runMove` with `externalMove`):
    /// no PP, no `moveUsed`, no Pressure, and no further Dancer.
    pub dancing: std::cell::Cell<bool>,
    /// PS `battle.activeTarget` of the running move (its resolved target),
    /// read by Dancer after the move.
    pub dance_target: std::cell::Cell<Option<(Side, usize)>>,
}

#[derive(Debug, Clone)]
pub struct DoublesState {
    /// Scratch flags of the running move (were thread-locals).
    pub tl: MoveScratch,
    /// Pseudo-weathers (bits over `events::FIELD_IDS`) whose PS effect state
    /// has a `target`: PS sets it the first time one of their handlers runs
    /// in `runEvent`; until then they sort as a plain Condition (subOrder 2).
    pub pseudo_targeted: std::cell::Cell<u32>,
    /// Union of the four slots' event-handler effect bits (`events`), keyed
    /// by both sides' `ver` and the Future Sight / Wish bits.
    pub(crate) fx_union: std::cell::Cell<Option<(u32, u32, u8, crate::doubles::events::FxMask)>>,
    /// `sorted_actives`' last untied order, keyed by both sides' `ver`.
    pub(crate) actives_order: std::cell::Cell<Option<((u32, u32), [(Side, u8); 4], u8)>>,
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
    /// Mud Sport / Water Sport pseudo-weathers (duration 5).
    pub mud_sport: StateTrickRoom,
    pub water_sport: StateTrickRoom,
    /// Echoed Voice's pseudo-weather: (multiplier, duration).
    pub echoed_voice: Option<(u8, i8)>,
    pub turn: u32,
    /// Pokemon that reached 0 HP during the current move's hit loop. PS only
    /// resolves faints after the loop, so its per-hit `eachEvent` still sorts
    /// them: (side, slot, team index).
    pub unresolved_faints: Vec<(crate::doubles::state::Side, usize, PokemonIndex)>,
    /// KO'd pokemon (still in `unresolved_faints`) whose Attract and Syrup
    /// Bomb volatiles still tie in `eachEvent('Update')`.
    pub ko_update_ties: Vec<(crate::doubles::state::Side, usize, PokemonIndex)>,
    /// Abilities blanked by Mold Breaker for the running move: (side, team
    /// index, real ability). PS `battle.suppressingAbility(target)`.
    pub mold_saved: arrayvec::ArrayVec<(Side, PokemonIndex, Abilities), 3>,
    /// Abilities blanked by an active Neutralizing Gas holder: (side, team
    /// index, real ability). PS sim/pokemon.ts ignoringAbility.
    pub gas_saved: arrayvec::ArrayVec<(Side, PokemonIndex, Abilities), 4>,
    /// The Neutralizing Gas holder whose `onEnd` is running (PS
    /// `abilityState.ending`): it no longer suppresses anything.
    pub gas_ending: Option<(Side, PokemonIndex)>,
    /// Build the fuzz output (per-turn snapshots, move-order entries). Off
    /// for plain simulation; the battle itself is the same either way.
    pub record: bool,
    /// Record typed battle events into `events` (off by default; the battle
    /// is the same either way).
    pub record_events: bool,
    /// Compute `a_options` / `b_options` in snapshots (off in the fuzzer's
    /// `--batch --check` mode, which never reads them).
    pub snap_options: bool,
    /// The events of the current turn (cleared at turn start) when
    /// `record_events` is on.
    pub events: Vec<BattleEvent>,
    /// The move being executed (user side, user team index, move): the
    /// `actor` of the damage that move deals.
    pub ev_move: Option<(Side, PokemonIndex, Choices)>,
    /// The next switch-in is a drag (Roar, Whirlwind, Dragon Tail, Red Card).
    pub ev_drag: bool,
    /// Ion Deluge's one-turn `iondeluge` pseudo-weather: Normal moves are Electric.
    pub ion_deluge: bool,
    /// PS `battle.lastSuccessfulMoveThisTurn`: the last move whose `useMove`
    /// did something this turn (Fusion Flare / Fusion Bolt).
    pub last_success_move: Option<Choices>,
    /// The field / status / item / ability values the events have reported
    /// so far (`ev_sync` emits the differences).
    pub ev_snap: Option<Box<EvSnap>>,
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
    /// Index in `faint_log` of a Destiny Bond victim of the running action:
    /// PS queues that faint during `faintMessages`, after the hit's other
    /// faints, so it stays last (`checkWin` reads the last one).
    pub db_faint_pos: Option<usize>,
    /// PS `battle.effectOrder` counter for `ability_order`.
    pub effect_order: u32,
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
    /// Pokemon whose Fling `fling` volatile drops the item at the next Update.
    pub fling_pending: Vec<(Side, PokemonIndex)>,
    /// Switch requests this turn made with no answer in the turn's choices:
    /// (side, slot, revive). A step-by-step controller (`run_fuzz_turn`) reads
    /// these to ask for the answer and re-run the turn.
    pub unanswered_switches: Vec<(Side, usize, bool)>,
    /// The state when the first unanswered switch request was made.
    pub request_state: Option<Box<DoublesState>>,
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
    /// Switches PS requested before turn 1 (an Eject Pack set off during the
    /// leads' send-out), answered before the turn-0 snapshot.
    #[serde(default)]
    pub pre_switches: Vec<FuzzSwitch>,
    /// Record typed battle events into each turn's snapshot (`events`).
    #[serde(default)]
    pub events: bool,
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
        /// Mega Evolve before moving (PS `move N mega`).
        #[serde(default)]
        mega: bool,
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

/// `to_id` into a stack buffer (no allocation); `None` if the id is longer
/// than the buffer.
fn id_buf(s: &str) -> Option<arrayvec::ArrayString<64>> {
    let mut b = arrayvec::ArrayString::<64>::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            b.try_push(c.to_ascii_lowercase()).ok()?;
        } else if !c.is_ascii() {
            for l in c.to_lowercase().filter(|l| l.is_ascii_alphanumeric()) {
                b.try_push(l).ok()?;
            }
        }
    }
    Some(b)
}

/// Id -> enum map over every variant of a `define_enum_with_from_str` enum
/// (variant names lowercased), built once. `from(i)` gives the variant with
/// discriminant `i` for `i < n`, or the enum's default past the last one
/// (the walk stops at the first repeated name).
fn enum_ids<T: Clone + std::fmt::Debug>(n: usize, from: impl Fn(usize) -> T) -> FxMap<&'static str, T> {
    let mut m = FxMap::default();
    for i in 0..n {
        let v = from(i);
        let id = format!("{:?}", v).to_lowercase();
        if m.contains_key(id.as_str()) {
            break;
        }
        m.insert(&*Box::leak(id.into_boxed_str()), v);
    }
    m
}

type FxMap<K, V> = rustc_hash::FxHashMap<K, V>;

fn lookup_id<T: Clone>(map: &FxMap<&'static str, T>, s: &str) -> Option<T> {
    match id_buf(s) {
        Some(b) => map.get(b.as_str()).cloned(),
        None => map.get(to_id(s).as_str()).cloned(),
    }
}

static NATURE_IDS: std::sync::LazyLock<FxMap<&'static str, PokemonNature>> =
    std::sync::LazyLock::new(|| enum_ids(25, |i| PokemonNature::from(i as u8)));
static ABILITY_IDS: std::sync::LazyLock<FxMap<&'static str, Abilities>> =
    std::sync::LazyLock::new(|| enum_ids(i16::MAX as usize, |i| Abilities::from(i as i16)));
static ITEM_IDS: std::sync::LazyLock<FxMap<&'static str, Items>> =
    std::sync::LazyLock::new(|| enum_ids(1024, |i| Items::from(i as u16)));

/// Lowercase id of each enum value, indexed by discriminant ("" for NONE):
/// the fuzz snapshot's species / item / ability ids, without allocating.
fn enum_names<T: std::fmt::Debug>(n: usize, from: impl Fn(usize) -> T) -> Vec<&'static str> {
    let mut v: Vec<&'static str> = Vec::new();
    for i in 0..n {
        let id = format!("{:?}", from(i)).to_lowercase();
        if i > 0 && id == "none" {
            break; // past the last variant: `from` falls back to NONE
        }
        v.push(if id == "none" { "" } else { Box::leak(id.into_boxed_str()) });
    }
    v
}

static SPECIES_NAMES: std::sync::LazyLock<Vec<&'static str>> =
    std::sync::LazyLock::new(|| enum_names(i16::MAX as usize, |i| PokemonName::from(i as i16)));
static ABILITY_NAMES: std::sync::LazyLock<Vec<&'static str>> =
    std::sync::LazyLock::new(|| enum_names(i16::MAX as usize, |i| Abilities::from(i as i16)));
static ITEM_NAMES: std::sync::LazyLock<Vec<&'static str>> =
    std::sync::LazyLock::new(|| enum_names(1024, |i| Items::from(i as u16)));

fn parse_nature(s: &str) -> PokemonNature {
    lookup_id(&NATURE_IDS, s).unwrap_or(PokemonNature::SERIOUS)
}

fn parse_ability(s: &str) -> Abilities {
    lookup_id(&ABILITY_IDS, s).unwrap_or(Abilities::NONE)
}

fn parse_item(s: &str) -> Items {
    if s.is_empty() {
        return Items::NONE;
    }
    lookup_id(&ITEM_IDS, s).unwrap_or(Items::NONE)
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
    // Shedinja (base HP 1) always has 1 HP (PS sim/dex-data.ts `statModify`).
    let mut hp = if base.0 == 1 { 1 } else { (common_stat_calc(base.0 as u16, e.0, lvl) + lvl + 10) as i16 };
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

thread_local! {
    static INTERNED: std::cell::RefCell<rustc_hash::FxHashSet<&'static str>> =
        std::cell::RefCell::new(Default::default());
}

/// A `&'static` copy of a name (species / forme names): each distinct name
/// is leaked once, so states can hold names without cloning strings.
pub fn intern(name: &str) -> &'static str {
    INTERNED.with(|set| {
        let mut set = set.borrow_mut();
        if let Some(&s) = set.get(name) {
            return s;
        }
        let s: &'static str = Box::leak(name.to_string().into_boxed_str());
        set.insert(s);
        s
    })
}

/// Species data resolved from a team member's species name.
#[derive(Clone, Copy)]
struct SpeciesInfo {
    id: PokemonName,
    base_stats: (i16, i16, i16, i16, i16, i16),
    types: (u8, u8),
    weight_kg: f32,
}

thread_local! {
    /// Name lookups are string matches over every species / move; a battle
    /// is built from few distinct names, so resolve each name once.
    static SPECIES_CACHE: std::cell::RefCell<FxMap<String, SpeciesInfo>> =
        std::cell::RefCell::new(Default::default());
    static MOVE_CACHE: std::cell::RefCell<FxMap<String, Choices>> =
        std::cell::RefCell::new(Default::default());
}

/// Forme names (e.g. "Furfrou-Debutante") may not be in the table or the
/// PokemonName enum; fall back to the base species (before the "-").
fn species_lookup(species: &str) -> SpeciesInfo {
    if let Some(sp) = SPECIES_CACHE.with(|c| c.borrow().get(species).copied()) {
        return sp;
    }
    let base = species.split('-').next().unwrap_or(species);
    let id = PokemonName::from_str(&to_id(species).to_uppercase())
        .or_else(|_| PokemonName::from_str(&to_id(base).to_uppercase()))
        .unwrap_or_else(|_| panic!("unknown species {}", species));
    let data = crate::doubles::species_types::species_data(&to_id(species))
        .or_else(|| crate::doubles::species_types::species_data(&to_id(base)))
        .unwrap_or_else(|| panic!("no species data for {}", species));
    let weight_kg = crate::doubles::species_weights::species_weight_kg(&to_id(species))
        .or_else(|| crate::doubles::species_weights::species_weight_kg(&to_id(base)))
        .unwrap_or(1.0);
    let sp = SpeciesInfo { id, base_stats: data.base_stats, types: data.types, weight_kg };
    SPECIES_CACHE.with(|c| c.borrow_mut().insert(species.to_string(), sp));
    sp
}

fn move_lookup(mv: &str) -> Choices {
    match MOVE_CACHE.with(|c| c.borrow().get(mv).copied()) {
        Some(c) => c,
        None => {
            let c = Choices::from_str(&to_id(mv).to_uppercase()).unwrap_or_else(|_| panic!("unknown move {mv}"));
            MOVE_CACHE.with(|m| m.borrow_mut().insert(mv.to_string(), c));
            c
        }
    }
}

fn build_pokemon(m: &TeamMember) -> Pokemon {
    // Zacian / Zamazenta `onBattleStart` with their Rusted item: the Crowned
    // forme (stats, types, weight) and Iron Head becomes its Behemoth move.
    // PS data/conditions.ts zacian / zamazenta.
    let crowned = match (to_id(&m.species).as_str(), to_id(&m.item).as_str()) {
        ("zacian", "rustedsword") => Some(("Zacian-Crowned", "behemothblade")),
        ("zamazenta", "rustedshield") => Some(("Zamazenta-Crowned", "behemothbash")),
        _ => None,
    };
    let crowned_member;
    let m = match crowned {
        Some((species, behemoth)) => {
            let mut c = m.clone();
            c.species = species.to_string();
            for mv in c.moves.iter_mut() {
                if to_id(mv) == "ironhead" {
                    *mv = behemoth.to_string();
                }
            }
            crowned_member = c;
            &crowned_member
        }
        None => m,
    };
    // Forme names (e.g. "Furfrou-Debutante") may not be in the table or the
    // PokemonName enum; fall back to the base species (before the "-").
    let sp = species_lookup(&m.species);
    let id = sp.id;
    let base = sp.base_stats;
    let (t0, t1) = sp.types;
    let types = (
        PokemonType::from(t0),
        PokemonType::from(t1),
    );
    let nature = parse_nature(&m.nature);
    let ability = parse_ability(&m.ability);
    let item = parse_item(&m.item);
    // Arceus / Silvally species `onType` (every forme): with Multitype / RKS
    // System the single type is the held Plate / Memory's, else Normal.
    // PS data/conditions.ts arceus, silvally.
    let base_species = m.species.split('-').next().unwrap_or(&m.species);
    let types = match (base_species, ability) {
        ("Arceus", Abilities::MULTITYPE) => (plate_type(item).unwrap_or(PokemonType::NORMAL), PokemonType::TYPELESS),
        ("Silvally", Abilities::RKSSYSTEM) => (
            crate::doubles::item_data::memory_type(item).unwrap_or(PokemonType::NORMAL),
            PokemonType::TYPELESS,
        ),
        _ => types,
    };
    let (hp, atk, def, spa, spd, spe) =
        compute_stats(base, &m.evs, m.level, nature.clone());
    let mut moves: [MoveSlot; 4] = Default::default();
    for (i, mv) in m.moves.iter().enumerate().take(4) {
        let choices = move_lookup(mv);
        // PS `baseMoveSlots` pp: `calculatePP(move, 3)`, champions
        // scripts.ts `(move.pp / 5 + 1) * 4` (stored as `Choice.pp`).
        moves[i] = MoveSlot { id: choices, pp: MOVES[&choices].pp as i8 };
    }
    let [m0, m1, m2, m3] = moves.map(|ms| crate::state::Move { id: ms.id, disabled: false, pp: ms.pp });
    Pokemon {
        id,
        level: m.level as i8,
        types,
        base_types: types,
        hp,
        maxhp: hp,
        ability: ability,
        base_ability: ability,
        item,
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
        weight_kg: sp.weight_kg,
        mega_evolved: false,
        terastallized: false,
        tera_type: PokemonType::NORMAL,
        attacked_by: Vec::new(),
        moves: crate::state::PokemonMoves { m0, m1, m2, m3 },
    }
}

/// Plates' `onPlate` (the type Multitype gives Arceus). PS data/items.ts.
pub(crate) fn plate_type(item: Items) -> Option<PokemonType> {
    Some(match item {
        Items::DRACOPLATE => PokemonType::DRAGON,
        Items::DREADPLATE => PokemonType::DARK,
        Items::EARTHPLATE => PokemonType::GROUND,
        Items::FISTPLATE => PokemonType::FIGHTING,
        Items::FLAMEPLATE => PokemonType::FIRE,
        Items::ICICLEPLATE => PokemonType::ICE,
        Items::INSECTPLATE => PokemonType::BUG,
        Items::IRONPLATE => PokemonType::STEEL,
        Items::MEADOWPLATE => PokemonType::GRASS,
        Items::MINDPLATE => PokemonType::PSYCHIC,
        Items::PIXIEPLATE => PokemonType::FAIRY,
        Items::SKYPLATE => PokemonType::FLYING,
        Items::SPLASHPLATE => PokemonType::WATER,
        Items::SPOOKYPLATE => PokemonType::GHOST,
        Items::STONEPLATE => PokemonType::ROCK,
        Items::TOXICPLATE => PokemonType::POISON,
        Items::ZAPPLATE => PokemonType::ELECTRIC,
        _ => return None,
    })
}

#[derive(Debug, Clone)]
struct MoveSlot {
    id: Choices,
    pp: i8,
}
impl Default for MoveSlot {
    fn default() -> Self {
        Self { id: Choices::NONE, pp: 0 }
    }
}

impl DoublesState {
    pub fn from_input(input: &BattleInput) -> Self {
        Self::from_teams(&input.teams)
    }

    pub fn from_teams(teams: &Teams) -> Self {
        let mut side_one = DoublesSide {
            team: std::array::from_fn(|_| Pokemon::default()),
            slots: [ActiveSlot::empty(), ActiveSlot::empty()],
            cond_cache: Default::default(),
            fx_cache: Default::default(),
            ver: 0,
            side_conditions: SideConditions::default(),
            species_names: [""; 6],
            genders: std::array::from_fn(|_| None),
            team_boosts: std::array::from_fn(|_| TeamBoosts::default()),
            toxic_stage: [0; 6],
            speed_cache: [0; 6],
            ability_order: [0; 6],
            ps_order: [0, 1, 2, 3, 4, 5],
            fainted_in_slot: [None, None],
            wish: [None, None],
            future_move: [None, None],
            sc_order: Vec::new(),
            syrup_triggered: [false; 6],
            mega_used: false,
            fainted_this_turn: false,
            fainted_last_turn: false,
            once_boosted: [false; 6],
            last_item: [Items::NONE; 6],
            ate_berry: [false; 6],
            healing_wish_slot: [false; 2],
            lunar_dance_slot: [false; 2],
            moves_used: [[false; 4]; 6],
            damaged_by: std::array::from_fn(|_| Vec::new()),
            saved_stats: [None; 6],
            saved_speed: [None; 6],
            transform_save: Default::default(),
        };
        let mut side_two = DoublesSide {
            team: std::array::from_fn(|_| Pokemon::default()),
            slots: [ActiveSlot::empty(), ActiveSlot::empty()],
            cond_cache: Default::default(),
            fx_cache: Default::default(),
            ver: 0,
            side_conditions: SideConditions::default(),
            species_names: [""; 6],
            genders: std::array::from_fn(|_| None),
            team_boosts: std::array::from_fn(|_| TeamBoosts::default()),
            toxic_stage: [0; 6],
            speed_cache: [0; 6],
            ability_order: [0; 6],
            ps_order: [0, 1, 2, 3, 4, 5],
            fainted_in_slot: [None, None],
            wish: [None, None],
            future_move: [None, None],
            sc_order: Vec::new(),
            syrup_triggered: [false; 6],
            mega_used: false,
            fainted_this_turn: false,
            fainted_last_turn: false,
            once_boosted: [false; 6],
            last_item: [Items::NONE; 6],
            ate_berry: [false; 6],
            healing_wish_slot: [false; 2],
            lunar_dance_slot: [false; 2],
            moves_used: [[false; 4]; 6],
            damaged_by: std::array::from_fn(|_| Vec::new()),
            saved_stats: [None; 6],
            saved_speed: [None; 6],
            transform_save: Default::default(),
        };
        for (i, m) in teams.a.iter().enumerate().take(6) {
            side_one.team[i] = build_pokemon(m);
            // PS `setSpecies`: speed = storedStats.spe.
            side_one.speed_cache[i] = side_one.team[i].speed;
            side_one.species_names[i] = intern(&m.species);
            side_one.genders[i] = m.gender.as_deref().map(intern);
        }
        for (i, m) in teams.b.iter().enumerate().take(6) {
            side_two.team[i] = build_pokemon(m);
            side_two.speed_cache[i] = side_two.team[i].speed;
            side_two.species_names[i] = intern(&m.species);
            side_two.genders[i] = m.gender.as_deref().map(intern);
        }
        let state = Self {
            tl: MoveScratch { pressure_extra: std::cell::Cell::new(-1), ..Default::default() },
            pseudo_targeted: std::cell::Cell::new(0),
            fx_union: Default::default(),
            actives_order: Default::default(),
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
            fling_pending: Vec::new(),
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
            mud_sport: StateTrickRoom {
                active: false,
                turns_remaining: 0,
            },
            water_sport: StateTrickRoom {
                active: false,
                turns_remaining: 0,
            },
            echoed_voice: None,
            turn: 1,
            unresolved_faints: Vec::new(),
            ko_update_ties: Vec::new(),
            mold_saved: arrayvec::ArrayVec::new(),
            gas_saved: arrayvec::ArrayVec::new(),
            gas_ending: None,
            record: true,
            record_events: false,
            snap_options: true,
            events: Vec::new(),
            ev_move: None,
            ev_drag: false,
            ion_deluge: false,
            last_success_move: None,
            ev_snap: None,
            faint_disabled: Vec::new(),
            db_alive_pending: Vec::new(),
            pending_disguise: Vec::new(),
            faint_log: Vec::new(),
            db_faint_pos: None,
            effect_order: 0,
            pending_after_faint: None,
            unanswered_switches: Vec::new(),
            request_state: None,
        };
        state
    }

    /// Send out the lead Pokémon into the two active slots.
    pub fn send_out_leads(&mut self, leads_a: &[usize], leads_b: &[usize]) {
        // PS starts the battle with an end-of-turn advance: the leads already
        // have `activeTurns` 1 at the first residual (Speed Boost fires).
        for (slot, &lead_pos) in leads_a.iter().take(2).enumerate() {
            let idx = PokemonIndex::deserialize(&lead_pos.to_string());
            self.side_one.active_slot_mut(slot).pokemon_index = Some(idx);
            self.side_one.active_slot_mut(slot).active_turns = 1;
        }
        for (slot, &lead_pos) in leads_b.iter().take(2).enumerate() {
            let idx = PokemonIndex::deserialize(&lead_pos.to_string());
            self.side_two.active_slot_mut(slot).pokemon_index = Some(idx);
            self.side_two.active_slot_mut(slot).active_turns = 1;
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
    pub species: &'static str,
    pub hp: i16,
    pub maxhp: i16,
    pub fainted: bool,
    /// Normalized status: none | burn | sleep | freeze | paralyze | poison | toxic
    pub status: &'static str,
    /// Active slot index (0/1) if on field, else null.
    pub active_slot: Option<usize>,
    pub boosts: Boosts,
    /// Current species / forme id (PS `species.id`, e.g. "absolmega").
    pub forme: &'static str,
    /// Held item id ("" = none).
    pub item: &'static str,
    /// Current ability id.
    pub ability: &'static str,
    /// PP left per move slot.
    pub pp: Vec<i8>,
    /// Volatile condition ids while active (sorted), e.g. ["confusion"].
    pub volatiles: Vec<String>,
}

/// Side conditions as (id, duration) pairs, ids lowercase (e.g. "reflect").
#[derive(Debug, Clone, Serialize)]
pub struct SideConditionEntry {
    pub id: &'static str,
    /// `None` for conditions without a duration (hazards): PS reports none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<i8>,
}

/// What a `BattleEvent` reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Move,
    Switch,
    Drag,
    Faint,
    Damage,
    Heal,
    Status,
    CureStatus,
    Boost,
    Unboost,
    Weather,
    FieldStart,
    FieldEnd,
    SideStart,
    SideEnd,
    Crit,
    SuperEffective,
    Resisted,
    Immune,
    Miss,
    Fail,
    Item,
    EndItem,
    Ability,
    Mega,
    Cant,
}

impl EventKind {
    pub fn id(self) -> &'static str {
        match self {
            EventKind::Move => "move",
            EventKind::Switch => "switch",
            EventKind::Drag => "drag",
            EventKind::Faint => "faint",
            EventKind::Damage => "damage",
            EventKind::Heal => "heal",
            EventKind::Status => "status",
            EventKind::CureStatus => "curestatus",
            EventKind::Boost => "boost",
            EventKind::Unboost => "unboost",
            EventKind::Weather => "weather",
            EventKind::FieldStart => "fieldstart",
            EventKind::FieldEnd => "fieldend",
            EventKind::SideStart => "sidestart",
            EventKind::SideEnd => "sideend",
            EventKind::Crit => "crit",
            EventKind::SuperEffective => "supereffective",
            EventKind::Resisted => "resisted",
            EventKind::Immune => "immune",
            EventKind::Miss => "miss",
            EventKind::Fail => "fail",
            EventKind::Item => "item",
            EventKind::EndItem => "enditem",
            EventKind::Ability => "ability",
            EventKind::Mega => "mega",
            EventKind::Cant => "cant",
        }
    }
}

/// What the recorded events have reported (see `DoublesState::ev_snap`).
#[derive(Debug, Clone, PartialEq)]
pub struct EvSnap {
    pub weather: Weather,
    pub terrain: Terrain,
    pub pseudo: [bool; 7],
    pub sc: [[i8; 13]; 2],
    pub status: [[PokemonStatus; 6]; 2],
    pub item: [[Items; 6]; 2],
    pub ability: [[Abilities; 6]; 2],
}

/// Pseudo-weather ids in `EvSnap::pseudo` order.
pub const EV_PSEUDO: [&str; 7] = ["trickroom", "wonderroom", "magicroom", "gravity", "fairylock", "mudsport", "watersport"];
/// Side condition ids in `EvSnap::sc` order.
pub const EV_SIDE: [&str; 13] = [
    "reflect", "lightscreen", "auroraveil", "tailwind", "safeguard", "luckychant", "mist", "spikes",
    "toxicspikes", "stealthrock", "stickyweb", "quickguard", "wideguard",
];

impl EvSnap {
    pub fn of(state: &DoublesState) -> Self {
        let sc = |s: &DoublesSide| {
            let c = &s.side_conditions;
            [
                c.reflect, c.light_screen, c.aurora_veil, c.tailwind, c.safeguard, c.lucky_chant, c.mist, c.spikes,
                c.toxic_spikes, c.stealth_rock, c.sticky_web, c.quick_guard, c.wide_guard,
            ]
        };
        EvSnap {
            weather: state.weather.weather_type,
            terrain: state.terrain.terrain_type,
            pseudo: [
                state.trick_room.active,
                state.wonder_room.active,
                state.magic_room.active,
                state.gravity.active,
                state.fairy_lock.active,
                state.mud_sport.active,
                state.water_sport.active,
            ],
            sc: [sc(&state.side_one), sc(&state.side_two)],
            status: [
                std::array::from_fn(|i| state.side_one.team[i].status),
                std::array::from_fn(|i| state.side_two.team[i].status),
            ],
            item: [
                std::array::from_fn(|i| state.side_one.team[i].item),
                std::array::from_fn(|i| state.side_two.team[i].item),
            ],
            ability: [
                std::array::from_fn(|i| state.side_one.team[i].ability),
                std::array::from_fn(|i| state.side_two.team[i].ability),
            ],
        }
    }
}

/// One typed battle event (`DoublesState::events`).
///
/// `side` / `actor` is the pokemon that acts (a move's user, the pokemon
/// that switches in, faints, mega evolves or can't move, the attacker whose
/// move deals damage); `target_side` / `target` is the pokemon it happens
/// to (a move's target, the pokemon that takes damage, heals, gets a status
/// or a boost, loses an item). Team indexes, -1 for none. `value` is the
/// exact HP change (damage, heal), the boost amount (boost, unboost), the
/// slot (switch, drag), the layer count (sidestart) or the turns (weather,
/// fieldstart).
#[derive(Debug, Clone, Copy)]
pub struct BattleEvent {
    pub kind: EventKind,
    pub side: Option<Side>,
    pub actor: i8,
    pub target_side: Option<Side>,
    pub target: i8,
    pub mv: Choices,
    pub status: PokemonStatus,
    pub effect: &'static str,
    pub item: Items,
    pub ability: Abilities,
    pub value: i32,
}

impl BattleEvent {
    pub fn new(kind: EventKind) -> Self {
        BattleEvent {
            kind,
            side: None,
            actor: -1,
            target_side: None,
            target: -1,
            mv: Choices::NONE,
            status: PokemonStatus::NONE,
            effect: "",
            item: Items::NONE,
            ability: Abilities::NONE,
            value: 0,
        }
    }
    pub fn actor(mut self, side: Side, idx: impl EvIndex) -> Self {
        self.side = Some(side);
        self.actor = idx.ev_index();
        self
    }
    pub fn target(mut self, side: Side, idx: impl EvIndex) -> Self {
        self.target_side = Some(side);
        self.target = idx.ev_index();
        self
    }
    pub fn mv(mut self, mv: Choices) -> Self {
        self.mv = mv;
        self
    }
    pub fn status(mut self, status: PokemonStatus) -> Self {
        self.status = status;
        self
    }
    pub fn effect(mut self, effect: &'static str) -> Self {
        self.effect = effect;
        self
    }
    pub fn item(mut self, item: Items) -> Self {
        self.item = item;
        self
    }
    pub fn ability(mut self, ability: Abilities) -> Self {
        self.ability = ability;
        self
    }
    pub fn value(mut self, value: i32) -> Self {
        self.value = value;
        self
    }
}

/// A team index for `BattleEvent` (a `PokemonIndex` or a team position).
pub trait EvIndex {
    fn ev_index(self) -> i8;
}
impl EvIndex for PokemonIndex {
    fn ev_index(self) -> i8 {
        self as i8
    }
}
impl EvIndex for usize {
    fn ev_index(self) -> i8 {
        self as i8
    }
}

impl Serialize for BattleEvent {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let side = |x: Option<Side>| match x {
            Some(Side::One) => "a",
            Some(Side::Two) => "b",
            None => "",
        };
        let mv = if self.mv == Choices::NONE { String::new() } else { format!("{:?}", self.mv).to_lowercase() };
        let status = if self.status == PokemonStatus::NONE { "" } else { normalize_status(self.status) };
        let mut m = s.serialize_map(Some(11))?;
        m.serialize_entry("kind", self.kind.id())?;
        m.serialize_entry("side", side(self.side))?;
        m.serialize_entry("actor", &self.actor)?;
        m.serialize_entry("target_side", side(self.target_side))?;
        m.serialize_entry("target", &self.target)?;
        m.serialize_entry("move", &mv)?;
        m.serialize_entry("status", status)?;
        m.serialize_entry("effect", self.effect)?;
        m.serialize_entry("item", if self.item == Items::NONE { "" } else { ITEM_NAMES[self.item as usize] })?;
        m.serialize_entry(
            "ability",
            if self.ability == Abilities::NONE { "" } else { ABILITY_NAMES[self.ability as usize] },
        )?;
        m.serialize_entry("value", &self.value)?;
        m.end()
    }
}

/// PS weather condition id of `w` ("sunnyday", "raindance", ...).
pub fn weather_id(w: Weather) -> &'static str {
    match w {
        Weather::NONE => "none",
        Weather::SUN => "sunnyday",
        Weather::HARSHSUN => "harshsun",
        Weather::RAIN => "raindance",
        Weather::HEAVYRAIN => "heavyrain",
        Weather::SAND => "sandstorm",
        Weather::HAIL => "hail",
        Weather::SNOW => "snowscape",
        Weather::DELTASTREAM => "deltastream",
    }
}

/// One move-order entry; serialized as `"<A|B><slot>:<species>:<MOVE>"`.
#[derive(Debug, Clone, Copy)]
pub struct OrderEntry {
    pub side: Side,
    pub slot: u8,
    pub species: &'static str,
    pub mv: Choices,
}

impl std::fmt::Display for OrderEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let tag = match self.side {
            Side::One => "A",
            Side::Two => "B",
        };
        write!(f, "{}{}:{}:{:?}", tag, self.slot, self.species, self.mv)
    }
}

impl Serialize for OrderEntry {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct FuzzTurnState {
    pub turn: u32,
    pub a: Vec<FullSlot>,
    pub b: Vec<FullSlot>,
    pub weather: &'static str,
    pub terrain: &'static str,
    pub trick_room: bool,
    /// Active pseudo-weather ids (sorted), e.g. ["trickroom"], ["tailwind"].
    pub pseudo_weather: Vec<&'static str>,
    /// Turns left (PS `duration`) of the weather, terrain and each
    /// pseudo-weather in `pseudo_weather` order.
    pub weather_turns: i8,
    pub terrain_turns: i8,
    pub pseudo_weather_turns: Vec<i8>,
    pub a_side_conditions: Vec<SideConditionEntry>,
    pub b_side_conditions: Vec<SideConditionEntry>,
    /// Execution order this turn, entries "SIDE<slot>:Species:movename".
    pub move_order: Vec<OrderEntry>,
    /// Legal choices per active slot for the next move request (None = empty
    /// slot), read by the legality audit (`fuzz/legal_audit.js`).
    pub a_options: Vec<Option<crate::doubles::legal::SlotOptions>>,
    pub b_options: Vec<Option<crate::doubles::legal::SlotOptions>>,
    /// The turn's events (`record_events`), absent when off.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<BattleEvent>,
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
/// `side_condition_entries` into a stack list (no allocation).
pub fn side_condition_entries_into(sc: &SideConditions, v: &mut arrayvec::ArrayVec<SideConditionEntry, 24>) {
    if sc.reflect != 0 {
        v.push(SideConditionEntry { id: "reflect", duration: Some(sc.reflect) });
    }
    if sc.light_screen != 0 {
        v.push(SideConditionEntry { id: "lightscreen", duration: Some(sc.light_screen) });
    }
    if sc.aurora_veil != 0 {
        v.push(SideConditionEntry { id: "auroraveil", duration: Some(sc.aurora_veil) });
    }
    if sc.tailwind != 0 {
        v.push(SideConditionEntry { id: "tailwind", duration: Some(sc.tailwind) });
    }
    if sc.safeguard != 0 {
        v.push(SideConditionEntry { id: "safeguard", duration: Some(sc.safeguard) });
    }
    if sc.lucky_chant != 0 {
        v.push(SideConditionEntry { id: "luckychant", duration: Some(sc.lucky_chant) });
    }
    if sc.mist != 0 {
        v.push(SideConditionEntry { id: "mist", duration: Some(sc.mist) });
    }
    if sc.spikes != 0 {
        v.push(SideConditionEntry { id: "spikes", duration: None });
    }
    if sc.stealth_rock != 0 {
        v.push(SideConditionEntry { id: "stealthrock", duration: None });
    }
    if sc.sticky_web != 0 {
        v.push(SideConditionEntry { id: "stickyweb", duration: None });
    }
    if sc.toxic_spikes != 0 {
        v.push(SideConditionEntry { id: "toxicspikes", duration: None });
    }
    if sc.protect != 0 {
        v.push(SideConditionEntry { id: "protect", duration: Some(sc.protect) });
    }
    if sc.quick_guard != 0 {
        v.push(SideConditionEntry { id: "quickguard", duration: Some(sc.quick_guard) });
    }
    if sc.wide_guard != 0 {
        v.push(SideConditionEntry { id: "wideguard", duration: Some(sc.wide_guard) });
    }
    if sc.crafty_shield != 0 {
        v.push(SideConditionEntry { id: "craftyshield", duration: Some(sc.crafty_shield) });
    }
    if sc.mat_block != 0 {
        v.push(SideConditionEntry { id: "matblock", duration: Some(sc.mat_block) });
    }
}

pub fn side_condition_entries(sc: &SideConditions) -> Vec<SideConditionEntry> {
    let mut v = arrayvec::ArrayVec::new();
    side_condition_entries_into(sc, &mut v);
    v.into_iter().collect()
}

/// Snapshot the full state of both sides for fuzz comparison.
pub fn fuzz_snapshot(state: &DoublesState, move_order: Vec<OrderEntry>) -> FuzzTurnState {
    let snap_side = |s: &DoublesSide, sd: Side| -> Vec<FullSlot> {
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
                    .active()
                    .iter()
                    .position(|sl| sl.pokemon_index.map(|x| x as usize) == Some(i));
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
                    species: s.transform_save[i].as_ref().map_or(s.species_names[i], |t| t.as_species),
                    forme: SPECIES_NAMES[p.id as usize],
                    hp: p.hp,
                    maxhp: p.maxhp,
                    fainted: p.hp <= 0,
                    status: normalize_status(p.status),
                    active_slot,
                    boosts,
                    item: ITEM_NAMES[p.item as usize],
                    // Gastro Acid / Neutralizing Gas keep the ability id (the
                    // engine stores NONE).
                    ability: match active_slot {
                        Some(sl) if s.active()[sl].gastro_acid && p.ability == Abilities::NONE => {
                            ABILITY_NAMES[s.active()[sl].gastro_ability as usize]
                        }
                        _ => match state.gas_saved.iter().find(|&&(gs, gi, _)| gs == sd && gi as usize == i) {
                            Some(&(_, _, a)) if p.ability == Abilities::NONE => ABILITY_NAMES[a as usize],
                            _ => ABILITY_NAMES[p.ability as usize],
                        },
                    },
                    pp: [&p.moves.m0, &p.moves.m1, &p.moves.m2, &p.moves.m3]
                        .iter()
                        .filter(|m| m.id != Choices::NONE)
                        .map(|m| m.pp)
                        .collect(),
                    volatiles: active_slot
                        .map(|sl| {
                            let a = &s.active()[sl];
                            let mut v = crate::doubles::events::volatile_ids(a);
                            // Volatiles kept outside `CONDS`.
                            if a.unburden {
                                v.push("unburden".to_string());
                            }
                            // `trapper`: the other end of a `trapped` link this pokemon made.
                            let me = Some((sd, PokemonIndex::deserialize(&i.to_string())));
                            if [&state.side_one, &state.side_two]
                                .iter()
                                .any(|o| o.active().iter().any(|x| x.trapped && x.trapper == me && x.pokemon_index.is_some()))
                            {
                                v.push("trapper".to_string());
                            }
                            // A fired two-turn move's `twoturnmove` lasts until the residual.
                            if a.twoturn_fired && a.charging.is_none() && !v.iter().any(|x| x == "twoturnmove") {
                                v.push("twoturnmove".to_string());
                            }
                            if a.zen_mode {
                                v.push("zenmode".to_string());
                            }
                            if a.gem {
                                v.push("gem".to_string());
                            }
                            if a.metronome_vol {
                                v.push("metronome".to_string());
                            }
                            v.sort();
                            v
                        })
                        .unwrap_or_default(),
                }
            })
            .collect()
    };
    let sc_entries = |sc: &SideConditions, order: &[&'static str]| -> Vec<SideConditionEntry> {
        // The harness lists `sideConditions` sorted by id.
        let _ = order;
        let mut v = side_condition_entries(sc);
        v.sort_by(|a, b| a.id.cmp(&b.id));
        v
    };
    let mut pseudo: Vec<(&'static str, i8)> = [
        ("trickroom", &state.trick_room),
        ("wonderroom", &state.wonder_room),
        ("magicroom", &state.magic_room),
        ("gravity", &state.gravity),
        ("fairylock", &state.fairy_lock),
        ("mudsport", &state.mud_sport),
        ("watersport", &state.water_sport),
    ]
    .iter()
    .filter(|(_, f)| f.active)
    .map(|(id, f)| (*id, f.turns_remaining))
    .collect();
    if let Some((_, d)) = state.echoed_voice {
        pseudo.push(("echoedvoice", d));
    }
    // Ion Deluge (duration 1): only seen when the battle ends mid-turn.
    if state.ion_deluge {
        pseudo.push(("iondeluge", 1));
    }
    pseudo.sort();
    let pseudo_weather: Vec<&'static str> = pseudo.iter().map(|(id, _)| *id).collect();
    let pseudo_weather_turns: Vec<i8> = pseudo.iter().map(|(_, t)| *t).collect();
    // pokemon-showdown reports the weather CONDITION id (e.g. "raindance",
    // "sunnyday", "sandstorm"), not the engine's enum name.
    let weather_name = weather_id(state.weather.weather_type);
    FuzzTurnState {
        turn: state.turn,
        a: snap_side(&state.side_one, Side::One),
        b: snap_side(&state.side_two, Side::Two),
        weather: weather_name,
        terrain: match state.terrain.terrain_type {
            Terrain::NONE => "none",
            Terrain::ELECTRICTERRAIN => "electricterrain",
            Terrain::PSYCHICTERRAIN => "psychicterrain",
            Terrain::MISTYTERRAIN => "mistyterrain",
            Terrain::GRASSYTERRAIN => "grassyterrain",
        },
        trick_room: state.trick_room.active,
        pseudo_weather,
        weather_turns: if state.weather.weather_type == Weather::NONE { 0 } else { state.weather.turns_remaining },
        terrain_turns: if state.terrain.terrain_type == Terrain::NONE { 0 } else { state.terrain.turns_remaining },
        pseudo_weather_turns,
        a_side_conditions: sc_entries(&state.side_one.side_conditions, &state.side_one.sc_order),
        b_side_conditions: sc_entries(&state.side_two.side_conditions, &state.side_two.sc_order),
        move_order,
        a_options: if state.snap_options {
            (0..2).map(|sl| crate::doubles::legal::slot_options(state, Side::One, sl)).collect()
        } else {
            Vec::new()
        },
        b_options: if state.snap_options {
            (0..2).map(|sl| crate::doubles::legal::slot_options(state, Side::Two, sl)).collect()
        } else {
            Vec::new()
        },
        events: state.events.clone(),
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
        let mut present = arrayvec::ArrayVec::new();
        side_condition_entries_into(&self.side_conditions, &mut present);
        if present.is_empty() && self.sc_order.is_empty() {
            return;
        }
        self.sc_order.retain(|o| present.iter().any(|e| e.id == *o));
        for e in present {
            if !self.sc_order.contains(&e.id) {
                self.sc_order.push(e.id);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fuzzer fast check: the canonical form of a turn snapshot
// ---------------------------------------------------------------------------

/// PS-side `squash` in fuzz/fuzz.js: lowercase, keep only `[a-z0-9]`.
fn fuzz_squash(s: &str) -> String {
    s.to_lowercase().chars().filter(|c| c.is_ascii_lowercase() || c.is_ascii_digit()).collect()
}

/// `squashOrderEntry` in fuzz/fuzz.js: drop the slot digits after the side
/// tag, drop the species' forme suffix, squash.
fn fuzz_squash_order_entry(s: &str) -> String {
    let b = s.as_bytes();
    let s = if !b.is_empty() && (b[0] == b'A' || b[0] == b'B') && b.len() > 1 && b[1].is_ascii_digit() {
        let n = b[1..].iter().take_while(|c| c.is_ascii_digit()).count();
        format!("{}{}", &s[..1], &s[1 + n..])
    } else {
        s.to_string()
    };
    let mut parts: Vec<&str> = s.split(':').collect();
    if parts.len() == 3 {
        parts[1] = parts[1].split('-').next().unwrap_or("");
    }
    fuzz_squash(&parts.join(":"))
}

fn json_str(s: &str) -> String {
    serde_json::to_string(s).unwrap()
}

/// `JSON.stringify` of fuzz/fuzz.js `normalizePETurn(turn)` without `turn` and
/// `moveOrderRaw`: the string the fuzzer's fast path compares with the same
/// form of PS's snapshot. Keep it in step with `normalizePETurn`.
pub fn fuzz_canonical(t: &FuzzTurnState) -> String {
    use std::fmt::Write;
    let mut o = String::with_capacity(2048);
    let side = |o: &mut String, mons: &[FullSlot], sc: &[SideConditionEntry]| {
        o.push_str("{\"pokemon\":[");
        for (i, p) in mons.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            let b = &p.boosts;
            let _ = write!(
                o,
                "{{\"species\":{},\"position\":{},\"hp\":{},\"maxhp\":{},\"status\":{},\"fainted\":{},\"activeSlot\":{},\
\"boosts\":{{\"atk\":{},\"def\":{},\"spa\":{},\"spd\":{},\"spe\":{},\"acc\":{},\"eva\":{}}},\
\"forme\":{},\"item\":{},\"ability\":{},\"pp\":{},\"volatiles\":{}}}",
                json_str(&fuzz_squash(p.species.split('-').next().unwrap_or(""))),
                i,
                p.hp,
                p.maxhp,
                json_str(p.status),
                p.fainted,
                p.active_slot.map_or("null".to_string(), |x| x.to_string()),
                b.atk, b.def, b.spa, b.spd, b.spe, b.acc, b.eva,
                json_str(&fuzz_squash(p.forme)),
                json_str(p.item),
                json_str(p.ability),
                serde_json::to_string(&p.pp).unwrap(),
                serde_json::to_string(&p.volatiles).unwrap(),
            );
        }
        o.push_str("],\"sideConditions\":[");
        for (i, c) in sc.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            match c.duration {
                Some(d) => {
                    let _ = write!(o, "{{\"id\":{},\"duration\":{}}}", json_str(c.id), d);
                }
                None => {
                    let _ = write!(o, "{{\"id\":{}}}", json_str(c.id));
                }
            }
        }
        o.push_str("]}");
    };
    // WEATHER_MAP / TERRAIN_MAP in fuzz/fuzz.js, then lowercase.
    let weather = match t.weather {
        "sandstorm" => "sand",
        "primariesun" => "harshsun",
        "primaryrain" => "heavyrain",
        w => w,
    }
    .to_lowercase();
    let terrain = match t.terrain {
        "electric" => "electricterrain",
        "psyxic" => "psychicterrain",
        "misty" => "mistyterrain",
        "grassy" => "grassyterrain",
        w => w,
    }
    .to_lowercase();
    let mut pseudo: Vec<String> = t.pseudo_weather.iter().map(|s| fuzz_squash(s)).collect();
    pseudo.sort();
    let order: Vec<String> = t.move_order.iter().map(|e| fuzz_squash_order_entry(&e.to_string())).collect();
    o.push_str("{\"a\":");
    side(&mut o, &t.a, &t.a_side_conditions);
    o.push_str(",\"b\":");
    side(&mut o, &t.b, &t.b_side_conditions);
    let _ = write!(
        o,
        ",\"weather\":{},\"terrain\":{},\"trickRoom\":{},\"pseudoWeather\":{},\"weatherTurns\":{},\"terrainTurns\":{},\"pseudoWeatherTurns\":{},\"moveOrder\":{}}}",
        json_str(&weather),
        json_str(&terrain),
        t.trick_room,
        serde_json::to_string(&pseudo).unwrap(),
        t.weather_turns,
        t.terrain_turns,
        serde_json::to_string(&t.pseudo_weather_turns).unwrap(),
        serde_json::to_string(&order).unwrap(),
    );
    o
}

/// 64-bit FNV-1a of `fuzz_canonical` as 16 hex digits (`psCanonical` in
/// fuzz/fuzz.js hashes PS's side the same way).
pub fn fuzz_canonical_hash(t: &FuzzTurnState) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in fuzz_canonical(t).bytes() {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    format!("{:016x}", h)
}
