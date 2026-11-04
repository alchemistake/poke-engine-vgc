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
    substitute_health: i16,
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
    /// Power Trick is active (Attack and Defense are swapped).
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
    choice_lock: Option<Choices>,
    /// PS `moveThisTurnResult = null` (a NOT_FAIL hit: Protect), this / last turn.
    pub move_result_null: bool,
    pub move_null_last_turn: bool,
    /// The item this slot's Fling threw (its base power and effect).
    pub fling_item: Items,
    /// Revival Blessing hit: the slot's switch answer revives a fainted
    /// party member instead of switching (PS `revivalblessing` slot condition).
    pub reviving: bool,
    /// Flinched by the previous move's secondary effect; cannot act this turn.
    /// Used a move with the `recharge` flag (Hyper Beam, Solar Beam, ...);
    /// must recharge this turn.
    /// Turn ends `recharging` has lived through (PS `mustrecharge` duration 2:
    /// it ends at the second residual even if the turn was never spent).
    pub recharge_age: u8,
    /// Two-turn move charging (Dig, Meteor Beam, ...): the move id being
    /// charged and how many charge turns remain before it fires. Mirrors
    /// pokemon-showdown's `twoturnmove` / per-move volatile pair.
    charging: Option<(crate::choices::Choices, u8)>,
    /// Signed PS target location chosen on the charge turn; the locked follow-up turn keeps it
    /// (PS `twoturnmove.onStart` stores `targetLoc` on the move volatile).
    pub charge_target_loc: i8,
    /// Salt Cure volatile: residual damage each turn (champions: 1/16 max
    /// HP, 1/8 for Water / Steel).
    /// Stockpile layers (0-3) and the Def / SpD stages they actually added
    /// (PS `stockpile` volatile; removed again when it ends).
    stockpile: u8,
    /// Confusion volatile: PS `volatiles.confusion().time` (0 = not confused).
    confusion: u8,
    /// Hit by Roar / Whirlwind / Dragon Tail / Circle Throw this action
    /// (PS `forceSwitchFlag`): dragged out after the move.
    pub drag_out: bool,
    /// Focus Punch volatile (`focuspunch`, duration 1) and its `lostFocus`.
    pub lost_focus: bool,
    /// Beak Blast volatile (`beakblast`, set by its priority charge, removed
    /// after its own move): a contact hit on this slot burns the attacker.
    /// Shell Trap's `shelltrap` volatile (duration 1): 1 set, 2 `gotHit` by a
    /// foe's physical move. PS data/moves.ts shelltrap.condition.
    pub shell_trap: u8,
    /// `gastroacid` volatile (Gastro Acid landed on this slot): a second
    /// Gastro Acid fails (PS addVolatile on an existing volatile). Cleared on
    /// switch-out.
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
    protected: bool,
    /// Helping Hand multiplier on this pokemon's next move's base power this
    /// turn (PS `helpinghand.condition.onBasePower`); 1.0 = none. Stacks by
    /// x1.5 per Helping Hand (onRestart). Cleared at end of turn.
    helping_hand_mult: f32,
    /// Raised at least one of its own stats this turn (PS
    /// `statsRaisedThisTurn`); gates Alluring Voice / Burning Jealousy
    /// secondaries. Reset at end of turn.
    pub stats_raised_this_turn: bool,
    /// PS `statsLoweredThisTurn` (Lash Out).
    pub stats_lowered_this_turn: bool,
    /// Destiny Bond volatile: when the holder faints, the last attacker also
    /// faints (PS data/conditions.ts destinybond.onAfterFaint).
    /// The last pokemon that dealt damage to this slot's occupant
    /// (side, team index), for Destiny Bond `onAfterFaint`.
    pub last_attacker: Option<(Side, crate::state::PokemonIndex)>,
    /// `partiallytrapped` volatile (Whirlpool, Fire Spin, ...): remaining
    /// duration, the binder (side, team index) and the damage divisor.
    partial_trap: Option<PartialTrap>,
    /// `yawn` volatile: remaining duration (0 = none) and the user who
    /// applied it (for the sleep `trySetStatus` when it ends).
    yawn: u8,
    /// `throatchop` volatile: remaining duration (sound moves blocked).
    throat_chop: u8,
    /// Set while a move is run through `useMove` (Copycat): no PP, no BeforeMove.
    pub external_call: bool,
    /// The move being run was bounced (PS `move.hasBounced`): no second bounce.
    pub bounced: bool,
    /// The move being run was taken by Snatch (PS `move.sourceEffect ===
    /// 'snatch'`): no second Snatch.
    pub snatched: bool,
    /// Beat Up: base power for the current hit / the remaining hits' base powers.
    pub bp_fixed: f32,
    pub beat_up_queue: crate::copyvec::CopyVec<u8, 6>,
    /// Supreme Overlord `fallen` count fixed at switch-in (max 5).
    pub fallen: u8,
    /// `taunt` volatile: remaining duration (0 = none).
    taunt: u8,
    /// `magnetrise` volatile: remaining duration (0 = none). Makes the holder
    /// ungrounded (PS `isGrounded`) until it ends (PS data/moves.ts magnetrise).
    magnet_rise: u8,
    /// `telekinesis` volatile duration (3): floats; non-OHKO moves always hit.
    telekinesis: u8,
    /// `rollout` volatile (Rollout / Ice Ball lock): (hitCount, contactHitCount,
    /// duration, the base-power multiplier of the current hit).
    rollout: Option<(u8, u8, u8, u8)>,
    /// The `rollout` volatile is Ice Ball's (`onLockMove: 'iceball'`), not Rollout's.
    rollout_ice: bool,
    /// `defensecurl` volatile (doubles Rollout / Ice Ball).
    /// Mimic replaced move slot `k` (Mimic's id and PP saved, restored when the
    /// pokemon leaves: PS `moveSlots` vs `baseMoveSlots`).
    pub mimic: Option<(u8, crate::choices::Choices, i8)>,
    /// Lifted by a Sky Drop user (its side and team index): can't act, and is
    /// semi-invulnerable like the user.
    pub sky_lifted_by: Option<(Side, crate::state::PokemonIndex)>,
    /// Rage's `rage` volatile: +1 Atk when hit by a damaging move.
    /// Magic Coat's single-turn volatile (reflects like Magic Bounce).
    /// Snatch's single-turn volatile: takes the next snatchable move used by
    /// any pokemon (PS data/moves.ts snatch.condition).
    pub snatch: bool,
    /// Grudge's volatile: fainting to a move zeroes that move's PP.
    /// Micle Berry's `micleberry` volatile duration (next accuracy x1.2).
    micle: u8,
    /// `embargo` volatile duration (5): the holder's item is ignored.
    embargo: u8,
    /// `foresight` volatile (Foresight / Odor Sleuth).
    /// `miracleeye` volatile.
    /// Protosynthesis / Quark Drive volatile: (best stat 0..4 = Atk, Def, SpA,
    /// SpD, Spe; from Booster Energy).
    paradox: Option<(u8, bool)>,
    /// The paradox volatile is Quark Drive's (else Protosynthesis's).
    paradox_quark: bool,
    /// `bide` volatile: (duration, total move damage taken, last damaging pokemon).
    bide: Option<(u8, i16, Option<(Side, crate::state::PokemonIndex)>)>,
    /// `smackdown` volatile (no duration): the holder is grounded (PS
    /// `isGrounded`) until it switches out (PS data/moves.ts smackdown).
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
    pub illusion_target: Option<Option<u8>>,
    /// `curse` volatile (Ghost Curse): loses 1/4 max HP each residual (order 12).
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
    /// Berserk / Anger Shell `checkedBerserk` is false: a non-multihit move hit
    /// the holder and its `onAfterMoveSecondary` has not run, so its healing
    /// berries (`onTryEatItem`) wait.
    pub berserk_hold: bool,
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
    follow_me: u8,
    /// Which protect-like move set `protected`: 0 Protect/Detect, 1 King's
    /// Shield, 2 Spiky Shield, 3 Baneful Bunker, 4 Silk Trap, 5 Burning Bulwark.
    protect_kind: u8,
    /// `perishsong` volatile: remaining duration (0 = none); faints at 0.
    perish: u8,
    /// The volatiles (`events::CONDS` indices) in insertion order (PS
    /// `pokemon.volatiles` key order), kept by `sync_vol_order`. Orders the
    /// Residual handlers.
    pub vol_order: crate::copyvec::CopyVec<u8, 32>,
    /// `flashfire` volatile: Fire moves x1.5 (ModifyAtk / ModifySpA).
    /// `charge` volatile: the next Electric move has doubled power.
    /// `noretreat` volatile (No Retreat already used).
    /// The `trapped` volatile No Retreat's `onStart` added itself (none when the
    /// user was already trapped: that link's `trapped` stays, and goes with it).
    no_retreat_trap: bool,
    /// Truant's `truant` volatile: the next move is skipped (`cant`).
    /// `laserfocus` volatile duration (2 when used; the next move crits).
    laser_focus: u8,
    /// `furycutter` volatile: (multiplier, duration).
    fury_cutter: Option<(u8, u8)>,
    /// `focusenergy` volatile: +2 critical-hit stages.
    /// `attract` volatile: the infatuating pokemon (side, team index).
    attract: Option<(Side, PokemonIndex)>,
    /// `leechseed` volatile: the source's slot (side, slot) — PS
    /// `effectState.sourceSlot`, resolved with `getAtSlot` at residual time.
    leech_seed: Option<(Side, u8)>,
    /// `dragoncheer` volatile: `Some(hasDragonType)` fixed at start
    /// (crit ratio +2 if Dragon, else +1). PS data/moves.ts dragoncheer.
    dragon_cheer: Option<bool>,
    /// PS `pokemon.lastMove`: the last move this pokemon used (Encore).
    pub last_move: Option<crate::choices::Choices>,
    /// `glaiverush` volatile: moves against it always hit and deal x2 damage
    /// until its holder next tries to move (PS data/moves.ts glaiverush).
    /// Minimize volatile (PS `minimize`): see `mechanics::minimize_move`.
    /// `ingrain` volatile: heals 1/16 max HP each residual (order 7).
    /// `healblock` volatile: remaining duration (0 = none).
    heal_block: u8,
    /// Trace `seek` (PS data/abilities.ts trace.effectState.seek): set by Trace's
    /// `onStart` (switch-in, Role Play / Skill Swap), cleared when a foe with No
    /// Ability is seen then; while set, every Update event retries the copy.
    pub trace_seek: bool,
    /// `syrupbomb` volatile: (source side, source team index, duration).
    syrup_bomb: Option<(Side, PokemonIndex, u8)>,
    /// `octolock` volatile: the binder (side, team index). No duration; it
    /// ends once the binder is no longer active.
    octolock: Option<(Side, PokemonIndex)>,
    /// `trapped` volatile (Block, Mean Look, Spider Web, Jaw Lock): no switching;
    /// ends with the volatiles. PS data/conditions.ts trapped.
    trapped: bool,
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
    /// Switched in, its ability `onStart` not run yet (Unnerve's
    /// `effectState.unnerved` is still unset).
    pub ability_unstarted: bool,
    /// The `twoturnmove` volatile (duration 2) is still around for this
    /// turn's residual after the charged move fired.
    pub twoturn_fired: bool,
    /// `encore` volatile: the forced move and remaining duration.
    encore: Option<(crate::choices::Choices, u8)>,
    /// `torment` volatile (no duration): the last move is disabled while it lasts.
    /// `imprison` volatile (no duration): the user's moves are blocked for the
    /// foes (not modelled yet). Set once; a second Imprison fails.
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
    /// `aquaring` volatile: heals 1/16 max HP each residual (order 6).
    /// `counter` / `mirrorcoat` volatiles (duration 1): the last foe slot that
    /// hit this pokemon with a physical / special move, and twice the damage.
    counter_vol: Option<(Option<(Side, u8)>, i16)>,
    mirror_coat_vol: Option<(Option<(Side, u8)>, i16)>,
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
    ally_switch: Option<(u16, u8)>,
    /// `lockon` volatile on the user (PS data/moves.ts lockon): the locked foe
    /// (side, team index) and the remaining duration. Its moves always hit it.
    lock_on: Option<(Side, PokemonIndex, u8)>,
    /// `uproar` volatile: remaining duration (0 = none). Locks the move.
    uproar: u8,
    /// `stall` volatile (Protect / Detect / Endure counter): (counter, duration).
    stall: Option<(u16, u8)>,
    /// `endure` volatile (duration 1): survive moves at 1 HP.
    /// PS `lastMoveTargetLoc`: the target location of the last move (signed:
    /// foe slot + 1, own slot -(slot + 1)); a locked move reuses it.
    pub last_target_loc: i8,
    /// PS `lastMoveUsed.type`: the type of the last move this pokemon used,
    /// after `ModifyType` (Conversion 2 reads it). Cleared on switch-out.
    pub last_move_used_type: Option<PokemonType>,
    /// `lockedmove` volatile (Outrage, Thrash, Petal Dance): (duration,
    /// trueDuration) from PS data/conditions.ts.
    locked_move: Option<(u8, u8)>,
    /// The move that set `locked_move` (a copied Petal Dance locks its copier
    /// into Petal Dance). PS `getLockedMove` replaces the chosen move with it.
    pub locked_move_id: Option<Choices>,
    /// `disable` volatile: the disabled move and remaining duration.
    disabled: Option<(Choices, u8)>,
    /// BeforeMove aborted the locked charge turn: PS `twoturnmove` is gone
    /// (`onMoveAborted`) but the move's own condition (Dive, ...) lasts to
    /// the end of the turn.
    charge_aborted: bool,
    pub yawn_source: Option<(Side, PokemonIndex)>,
    /// Libero / Protean already used this switch-in (PS `effectState.libero`);
    /// holds the original types, restored on switch-out.
    pub type_change: Option<(crate::state::PokemonType, crate::state::PokemonType)>,
    /// PS `effectState.protean` / `effectState.libero`: the holder's type change
    /// already happened for its current ability. A new ability (Skill Swap,
    /// Trace, Role Play, ...) makes a fresh `effectState`, so it resets.
    pub protean_done: bool,
    /// The volatile conditions that are up (bit i = `events::CONDS[i]`): the
    /// on/off ones live only here; the valued ones keep their bit in sync
    /// through their setters.
    vols: u128,
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
            recharge_age: 0,
            charging: None,
            charge_target_loc: 0,
            stockpile: 0,
            confusion: 0,
            drag_out: false,
            shell_trap: 0,
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
            last_attacker: None,
            partial_trap: None,
            yawn: 0,
            throat_chop: 0,
            external_call: false,
            bounced: false,
            snatched: false,
            bp_fixed: 0.0,
            beat_up_queue: crate::copyvec::CopyVec::new(),
            fallen: 0,
            taunt: 0,
            magnet_rise: 0,
            telekinesis: 0,
            rollout: None,
            rollout_ice: false,
            mimic: None,
            sky_lifted_by: None,
            snatch: false,
            micle: 0,
            embargo: 0,
            paradox: None,
            paradox_quark: false,
            bide: None,
            active_turns: 0,
            active_move_actions: 0,
            illusion_broken: false,
            opportunist_boosts: [0; 7],
            mirror_herb_boosts: [0; 7],
            me_first: false,
            me_first_vol: false,
            illusion_target: None,
            nightmare: false,
            tar_shot: false,
            power_shift: false,
            autotomized: false,
            berserk_hold: false,
            commanding: false,
            commanded: None,
            powder: false,
            bp_double: false,
            being_called_back: false,
            follow_me: 0,
            protect_kind: 0,
            perish: 0,
            vol_order: crate::copyvec::CopyVec::new(),
            no_retreat_trap: false,
            laser_focus: 0,
            fury_cutter: None,
            attract: None,
            leech_seed: None,
            dragon_cheer: None,
            last_move: None,
            heal_block: 0,
            trace_seek: false,
            syrup_bomb: None,
            octolock: None,
            trapped: false,
            trapper: None,
            hurt_this_turn: false,
            hurt_hp: 0,
            newly_switched: false,
            ability_unstarted: false,
            twoturn_fired: false,
            encore: None,
            added_grass: false,
            added_type: PokemonType::TYPELESS,
            roost_orig: None,
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
            last_target_loc: 0,
            last_move_used_type: None,
            locked_move: None,
            locked_move_id: None,
            disabled: None,
            charge_aborted: false,
            yawn_source: None,
            type_change: None,
            protean_done: false,
            vols: 0,
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
#[derive(Debug, Clone, Copy)]
pub struct TransformSave {
    pub id: PokemonName,
    pub moves: crate::state::PokemonMoves,
    pub types: (crate::state::PokemonType, crate::state::PokemonType),
    /// atk, def, spa, spd, spe.
    pub stats: [i16; 5],
    pub weight_kg: f32,
    /// The species name the pokemon now shows (the target's).
    pub as_species: SpeciesName,
}

#[derive(Debug, Clone)]
pub struct DoublesSide {
    pub team: [Pokemon; 6],
    slots: [ActiveSlot; 2],
    pub side_conditions: SideConditions,
    /// Original species names (including formes) for each team slot, used for
    /// name-based target matching in the replay test.
    pub species_names: [SpeciesName; 6],
    /// Gender per team member (None = genderless). Used by Attract.
    pub genders: [Option<Gender>; 6],
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
    pub ps_order: [u8; 6],
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
    pub sc_order: crate::copyvec::CopyVec<u8, 16>,
    /// Supersweet Syrup already triggered (PS `syrupTriggered`, once per
    /// battle per pokemon).
    pub syrup_triggered: Bits8,
    /// A Mega Evolution was used (PS: `canMegaEvo = false` for the whole side).
    pub mega_used: bool,
    /// PS `side.faintedThisTurn` / `faintedLastTurn` (Retaliate).
    pub fainted_this_turn: bool,
    pub fainted_last_turn: bool,
    /// PS `pokemon.swordBoost` / `shieldBoost`: Intrepid Sword / Dauntless Shield
    /// boost once per battle.
    pub once_boosted: Bits8,
    /// PS `pokemon.lastItem` per team member (Recycle, Harvest).
    pub last_item: [Items; 6],
    /// PS `pokemon.ateBerry` per team member (Belch).
    pub ate_berry: Bits8,
    /// Healing Wish / Lunar Dance slot conditions per active slot (PS
    /// `slotCondition: 'healingwish' | 'lunardance'`): they wait for the next
    /// pokemon switched into that slot.
    pub healing_wish_slot: Bits8,
    /// The slot's Healing Wish / Lunar Dance was visited by an Ally Switch
    /// `runEvent('Swap')` without being used: PS's `runEvent` set its
    /// `effectState.target` to that pokemon, so `fieldEvent('SwitchIn')` skips
    /// it from then on (only another Swap can still use it).
    pub healing_wish_detached: Bits8,
    /// Grass / Fire / Water Pledge side conditions (turns left, 0 = none).
    pub pledges: [i8; 3],
    /// PS `side.totalFainted`: every faint resolved, revived pokemon included.
    pub total_fainted: u8,
    /// Gluttony's `abilityState.gluttony` reset by Neutralizing Gas ending:
    /// the pinch-berry threshold stays 1/4 until the holder takes damage.
    pub gluttony_off: Bits8,
    /// The slot's `healing_wish_slot` condition is Lunar Dance (also restores PP).
    pub lunar_dance_slot: Bits8,
    /// PS `moveSlot.used` per team member / move slot (set by `deductPP`).
    pub moves_used: MoveBits,
    /// PS `pokemon.attackedBy` entries with damage (`getLastDamagedBy`), per
    /// team member: (attacker side, attacker slot, damage, turn).
    /// PS `attackedBy`: (attacker side, attacker slot at the hit, damage, turn,
    /// attacker team index). Pruned at end of turn unless the attacker is still
    /// on the field and alive (PS `endTurn`: `source.isActive`).
    /// Only each foe attacker's latest entry is kept (`record_damaged_by`).
    pub damaged_by: [crate::copyvec::CopyVec<(Side, u8, i16, u16, u8), 6>; 6],
    /// Original (atk, def, spa, spd) of a team member whose `storedStats` were
    /// changed (Guard / Power Split); restored by `setSpecies` on switch-in.
    pub saved_stats: [Option<[i16; 4]>; 6],
    /// Original raw speed of a team member whose `storedStats.spe` was swapped
    /// (Speed Swap); restored by `setSpecies` on switch-in.
    pub saved_speed: [Option<i16>; 6],
    /// A transformed team member's own species data, restored when it leaves
    /// the field (PS `transformed`, `clearVolatile` -> `setSpecies(baseSpecies)`).
    pub transform_save: [Option<TransformSave>; 6],
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
    /// Write PS `pokemon.speed` (`speed_cache`).
    #[inline]
    pub fn set_speed_cache(&mut self, index: usize, speed: i16) {
        self.speed_cache[index] = speed;
    }
    #[inline]
    pub fn team_member_mut(&mut self, index: PokemonIndex) -> &mut Pokemon {
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
        &mut self.slots
    }

    /// One on-field slot, for changes.
    #[inline]
    pub fn active_slot_mut(&mut self, slot: usize) -> &mut ActiveSlot {
        &mut self.slots[slot]
    }

    /// One on-field slot, for changes (same as `active_slot_mut`).
    #[inline]
    pub fn slot_scratch_mut(&mut self, slot: usize) -> &mut ActiveSlot {
        &mut self.slots[slot]
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
    /// A delayed hit's source no longer in its slot (fainted / switched): PS
    /// still collects its handlers (`onSource...`) from the pokemon itself.
    /// The event collector reads it for that side's empty slot.
    pub ev_source_idx: std::cell::Cell<Option<(Side, PokemonIndex)>>,
    /// Heal Block that Psychic Noise's secondary just added this move: PS ran
    /// the target's item `onHit` (Enigma Berry) before the secondaries.
    pub heal_block_fresh: std::cell::Cell<Option<(Side, usize)>>,
    /// An Update between `switchIn` and its `runSwitch`: White Herb (no
    /// `onUpdate`) waits for its `onAnySwitchIn` there.
    pub no_white_herb: std::cell::Cell<bool>,
    /// The residual action is running (PS `!queue.peek()`).
    pub in_residual: std::cell::Cell<bool>,
    /// The running move is a combined Pledge (`move.forceSTAB`), by its user.
    pub pledge_stab: std::cell::Cell<Option<(Side, PokemonIndex)>>,
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
    /// Micle Berry's `onSourceAccuracy` ran this move (it removes the
    /// volatile: a spread move's later targets get no boost).
    pub micle_spent: std::cell::Cell<bool>,
    /// Team positions a queued switch action will bring in (bit `side * 8 +
    /// position`): PS `possibleSwitches` leaves them out (Eject Pack).
    pub queued_switch_ins: std::cell::Cell<u16>,
    /// Spread-move targets (bit `side * 2 + slot`) this hit KOs: no Ally /
    /// Foe / Any handlers during the earlier targets' secondaries.
    pub spread_kod: std::cell::Cell<u8>,
    /// Lum Berry waits for Poison Puppeteer's confusion (`set_status`).
    pub defer_lum: std::cell::Cell<bool>,
    /// The move failed in `BeforeMove` after its `|move|` line (choicelock):
    /// `useMove` never ran, so no Pressure PP.
    pub before_move_failed: std::cell::Cell<bool>,
    /// A pivot switch-in: Tera Shift waits for the `runSwitch` speed sort.
    pub defer_tera_shift: std::cell::Cell<bool>,
    /// A Magic Coat / Bounce reflection ran inside this move: PS's
    /// `activePokemon` is now the bouncer (Unaware no longer sees the user).
    pub active_bounced: std::cell::Cell<bool>,
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
    /// PS `battle.activeMove` after `useMove`: the last move run in this action
    /// (a Metronome-called Feather Dance), which Dancer copies.
    pub dance_move: std::cell::Cell<Option<Choices>>,
}

#[derive(Debug)]
pub struct DoublesState {
    /// Scratch flags of the running move (were thread-locals).
    pub tl: MoveScratch,
    /// Pseudo-weathers (bits over `events::FIELD_IDS`) whose PS effect state
    /// has a `target`: PS sets it the first time one of their handlers runs
    /// in `runEvent`; until then they sort as a plain Condition (subOrder 2).
    pub pseudo_targeted: std::cell::Cell<u32>,
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
    pub unresolved_faints: crate::copyvec::CopyVec<(crate::doubles::state::Side, u8, PokemonIndex), 4>,
    /// PS `checkWin` ended the battle (`this.ended`): later KOs in the same
    /// callback (Life Orb recoil after the last foe fainted) are never processed.
    pub battle_ended: bool,
    /// Boosts of a pokemon KO'd but not yet through `faintMessages`: PS keeps
    /// them until `clearVolatile` (a move whose user fell to Spiky Shield
    /// still uses its stages). The engine clears them at the KO.
    pub fainted_boosts: crate::copyvec::CopyVec<(crate::doubles::state::Side, PokemonIndex, TeamBoosts), 8>,
    /// KO'd pokemon (still in `unresolved_faints`) whose Attract and Syrup
    /// Bomb volatiles still tie in `eachEvent('Update')`.
    pub ko_update_ties: crate::copyvec::CopyVec<(crate::doubles::state::Side, u8, PokemonIndex), 4>,
    /// Abilities blanked by Mold Breaker for the running move: (side, team
    /// index, real ability). PS `battle.suppressingAbility(target)`.
    pub mold_saved: crate::copyvec::CopyVec<(Side, PokemonIndex, Abilities), 3>,
    /// Abilities blanked by an active Neutralizing Gas holder: (side, team
    /// index, real ability). PS sim/pokemon.ts ignoringAbility.
    pub gas_saved: crate::copyvec::CopyVec<(Side, PokemonIndex, Abilities), 4>,
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
    /// A move's `runEvent('ModifyType')` ran the `iondeluge` handler, which set
    /// its `effectState.target` (the Field): its Residual subOrder is 5, not 2.
    pub ion_deluge_target: bool,
    /// Fairy Lock's `onTrapPokemon` ran (a switch-out while it is active),
    /// which set its pseudoWeather `effectState.target` (the Field): its
    /// Residual subOrder is 5, not 2. PS sim/battle.ts runEvent.
    pub fairy_lock_target: bool,
    /// PS `battle.lastSuccessfulMoveThisTurn`: the last move whose `useMove`
    /// did something this turn (Fusion Flare / Fusion Bolt).
    pub last_success_move: Option<Choices>,
    /// The field / status / item / ability values the events have reported
    /// so far (`ev_sync` emits the differences).
    pub ev_snap: Option<Box<EvSnap>>,
    /// The `disabled` volatile each pokemon had when it fainted. PS keeps the
    /// volatiles until `faintMessages`, so a DamagingHit handler that runs after
    /// a Destiny Bond KO of its attacker still sees the attacker's Disable.
    pub faint_disabled: crate::copyvec::CopyVec<(Side, PokemonIndex, Option<(Choices, u8)>), 16>,
    /// KO'd pokemon that were Embargoed: the volatile still holds until
    /// `faintMessages` (a Jaboca Berry on the KO hit stays unused).
    pub faint_embargo: crate::copyvec::CopyVec<(Side, PokemonIndex), 16>,
    /// Attackers KO'd by a Destiny Bond `onAfterFaint` during the current move.
    /// PS only resolves their faint after the move, so until then they still
    /// have HP for the rest of the hit (e.g. a lock volatile's `addVolatile`).
    /// The attacker's lock (`locked_move`) at its KO, for the same reason.
    pub db_alive_pending: crate::copyvec::CopyVec<(Side, PokemonIndex, Option<(u8, u8)>), 16>,
    /// Destiny Bond `onFaint` waits for `faintMessages` (`resolve_faints`):
    /// (fainted side, fainted index, attacker side, attacker index, the
    /// attacker's active slot at the KO, the attacker's lock at the KO). A
    /// spread move's later hits still see the attacker alive (Spicy Spray).
    pub db_pending: crate::copyvec::CopyVec<(Side, PokemonIndex, Side, PokemonIndex, u8, Option<(u8, u8)>), 16>,
    /// Grudge `onFaint` waits for `faintMessages`: (victim side, victim index,
    /// attacker side, attacker index, attacker's last move).
    pub grudge_pending: crate::copyvec::CopyVec<(Side, PokemonIndex, Side, PokemonIndex, Choices), 16>,
    /// Disguise holders whose hit was absorbed by a spread move. PS busts the
    /// forme in `onUpdate`, i.e. inside the next `eachEvent('Update')` after
    /// its speed sort (the sort still sees the stale speed).
    pub pending_disguise: crate::copyvec::CopyVec<(crate::doubles::state::Side, u8), 4>,
    /// Sides of every fainted pokemon in `faint()` call order: with both sides
    /// wiped out at once, PS gives the win to the side of the last one
    /// (`checkWin(faintData)`, sim/battle.ts).
    pub faint_log: crate::copyvec::CopyVec<Side, 32>,
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
    pub queue_change: Option<Box<crate::doubles::battle::QueuePlan>>,
    /// Round `onTry` ran this action: the battle loop prioritizes the next
    /// queued Round with BP doubled (`queue.prioritizeAction`).
    pub queue_round: bool,
    /// The Round that queued the next one ignores abilities (Mold Breaker):
    /// the called Round inherits `move.ignoreAbility` from its sourceEffect.
    pub queue_round_mold: bool,
    /// After You `onHit`: the target's queued action moves to the front
    /// (`queue.prioritizeAction`), applied by the battle loop.
    pub queue_prioritize: Option<(Side, PokemonIndex)>,
    /// Pledge `onPrepareHit`: the ally's prioritized action runs as a combo
    /// (`action.sourceEffect` = this Pledge).
    pub queue_pledge: Option<(Side, PokemonIndex, crate::choices::Choices)>,
    /// Quash `onHit`: the target's queued action is moved last (`order = 201`).
    pub queue_quash: Option<(Side, usize)>,
    /// Instruct `onHit`: the target's last move is queued again at the front of
    /// the queue (`resolveAction` + `prioritizeAction`). (side, slot, pokemon,
    /// move id, the target's signed `lastMoveTargetLoc`).
    pub queue_instruct: Option<(Side, usize, PokemonIndex, Choices, i8)>,
    /// A move is executing (Destiny Bond only reacts to move damage).
    pub in_move: bool,
    /// Gravity `onFieldStart`: queued moves cancelled (`queue.cancelMove`).
    pub queue_cancel: crate::copyvec::CopyVec<(Side, u8), 4>,
    /// Wonder Room's `onModifyMove` has run since it started (PS fixes the
    /// handler's subOrder only once `state.target` is set).
    pub wonder_room_seen: bool,
    /// Pokemon whose Fling `fling` volatile drops the item at the next Update.
    pub fling_pending: crate::copyvec::CopyVec<(Side, PokemonIndex), 4>,
    /// Switch requests this turn made with no answer in the turn's choices:
    /// (side, slot, revive). A step-by-step controller (`run_fuzz_turn`) reads
    /// these to ask for the answer and re-run the turn.
    pub unanswered_switches: crate::copyvec::CopyVec<(Side, u8, bool), 4>,
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

/// A species / forme name as a 2-byte id into a process-wide, append-only
/// name table (`SpeciesName::of` registers a name once; reads take no lock).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpeciesName(u16);

const MAX_SPECIES_NAMES: usize = 4096;
static SPECIES_NAME_TABLE: [std::sync::OnceLock<&'static str>; MAX_SPECIES_NAMES] =
    [const { std::sync::OnceLock::new() }; MAX_SPECIES_NAMES];
static SPECIES_NAME_IDS: std::sync::Mutex<Option<rustc_hash::FxHashMap<&'static str, u16>>> =
    std::sync::Mutex::new(None);

impl SpeciesName {
    /// The empty name (no team member).
    pub const EMPTY: SpeciesName = SpeciesName(0);

    /// The id of `name`, registering it on first use.
    pub fn of(name: &str) -> SpeciesName {
        if name.is_empty() {
            return SpeciesName::EMPTY;
        }
        let mut ids = SPECIES_NAME_IDS.lock().unwrap_or_else(|e| e.into_inner());
        let ids = ids.get_or_insert_with(Default::default);
        if let Some(&i) = ids.get(name) {
            return SpeciesName(i);
        }
        let i = ids.len() + 1;
        assert!(i < MAX_SPECIES_NAMES, "more than {} species names", MAX_SPECIES_NAMES);
        let s = intern(name);
        let _ = SPECIES_NAME_TABLE[i].set(s);
        ids.insert(s, i as u16);
        SpeciesName(i as u16)
    }

    #[inline]
    pub fn as_str(self) -> &'static str {
        if self.0 == 0 {
            return "";
        }
        SPECIES_NAME_TABLE[self.0 as usize].get().copied().unwrap_or("")
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl std::ops::Deref for SpeciesName {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for SpeciesName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
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
        attacked_by: crate::state::AttackedBy::default(),
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
            side_conditions: SideConditions::default(),
            species_names: [SpeciesName::EMPTY; 6],
            genders: [None; 6],
            team_boosts: std::array::from_fn(|_| TeamBoosts::default()),
            toxic_stage: [0; 6],
            speed_cache: [0; 6],
            ability_order: [0; 6],
            ps_order: [0, 1, 2, 3, 4, 5],
            fainted_in_slot: [None, None],
            wish: [None, None],
            future_move: [None, None],
            sc_order: crate::copyvec::CopyVec::new(),
            syrup_triggered: Bits8::default(),
            mega_used: false,
            fainted_this_turn: false,
            fainted_last_turn: false,
            once_boosted: Bits8::default(),
            last_item: [Items::NONE; 6],
            ate_berry: Bits8::default(),
            healing_wish_slot: Bits8::default(),
            healing_wish_detached: Bits8::default(),
            pledges: [0; 3],
            total_fainted: 0,
            gluttony_off: Bits8::default(),
            lunar_dance_slot: Bits8::default(),
            moves_used: MoveBits::default(),
            damaged_by: Default::default(),
            saved_stats: [None; 6],
            saved_speed: [None; 6],
            transform_save: Default::default(),
        };
        let mut side_two = DoublesSide {
            team: std::array::from_fn(|_| Pokemon::default()),
            slots: [ActiveSlot::empty(), ActiveSlot::empty()],
            side_conditions: SideConditions::default(),
            species_names: [SpeciesName::EMPTY; 6],
            genders: [None; 6],
            team_boosts: std::array::from_fn(|_| TeamBoosts::default()),
            toxic_stage: [0; 6],
            speed_cache: [0; 6],
            ability_order: [0; 6],
            ps_order: [0, 1, 2, 3, 4, 5],
            fainted_in_slot: [None, None],
            wish: [None, None],
            future_move: [None, None],
            sc_order: crate::copyvec::CopyVec::new(),
            syrup_triggered: Bits8::default(),
            mega_used: false,
            fainted_this_turn: false,
            fainted_last_turn: false,
            once_boosted: Bits8::default(),
            last_item: [Items::NONE; 6],
            ate_berry: Bits8::default(),
            healing_wish_slot: Bits8::default(),
            healing_wish_detached: Bits8::default(),
            pledges: [0; 3],
            total_fainted: 0,
            gluttony_off: Bits8::default(),
            lunar_dance_slot: Bits8::default(),
            moves_used: MoveBits::default(),
            damaged_by: Default::default(),
            saved_stats: [None; 6],
            saved_speed: [None; 6],
            transform_save: Default::default(),
        };
        for (i, m) in teams.a.iter().enumerate().take(6) {
            side_one.team[i] = build_pokemon(m);
            // PS `setSpecies`: speed = storedStats.spe.
            side_one.speed_cache[i] = side_one.team[i].speed;
            side_one.species_names[i] = SpeciesName::of(&m.species);
            side_one.genders[i] = Gender::parse(m.gender.as_deref());
        }
        for (i, m) in teams.b.iter().enumerate().take(6) {
            side_two.team[i] = build_pokemon(m);
            side_two.speed_cache[i] = side_two.team[i].speed;
            side_two.species_names[i] = SpeciesName::of(&m.species);
            side_two.genders[i] = Gender::parse(m.gender.as_deref());
        }
        let state = Self {
            tl: MoveScratch { pressure_extra: std::cell::Cell::new(-1), ..Default::default() },
            pseudo_targeted: std::cell::Cell::new(0),
            last_move: None,
            copy_called: false,
            queue_change: None,
            queue_round: false,
            queue_round_mold: false,
            queue_prioritize: None,
            queue_pledge: None,
            queue_quash: None,
            queue_instruct: None,
            in_move: false,
            queue_cancel: crate::copyvec::CopyVec::new(),
            wonder_room_seen: false,
            fling_pending: crate::copyvec::CopyVec::new(),
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
            unresolved_faints: crate::copyvec::CopyVec::new(),
            battle_ended: false,
            fainted_boosts: crate::copyvec::CopyVec::new(),
            ko_update_ties: crate::copyvec::CopyVec::new(),
            mold_saved: crate::copyvec::CopyVec::new(),
            gas_saved: crate::copyvec::CopyVec::new(),
            gas_ending: None,
            record: true,
            record_events: false,
            snap_options: true,
            events: Vec::new(),
            ev_move: None,
            ev_drag: false,
            ion_deluge: false,
            ion_deluge_target: false,
            fairy_lock_target: false,
            last_success_move: None,
            ev_snap: None,
            faint_disabled: crate::copyvec::CopyVec::new(),
            faint_embargo: crate::copyvec::CopyVec::new(),
            db_alive_pending: crate::copyvec::CopyVec::new(),
            db_pending: crate::copyvec::CopyVec::new(),
            grudge_pending: crate::copyvec::CopyVec::new(),
            pending_disguise: crate::copyvec::CopyVec::new(),
            faint_log: crate::copyvec::CopyVec::new(),
            db_faint_pos: None,
            effect_order: 0,
            pending_after_faint: None,
            unanswered_switches: crate::copyvec::CopyVec::new(),
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

/// The side-condition ids `side_condition_entries` lists; `sc_order` holds
/// indices into this table.
pub const SC_IDS: [&str; 16] = [
    "reflect", "lightscreen", "auroraveil", "tailwind", "safeguard", "luckychant", "mist", "spikes",
    "stealthrock", "stickyweb", "toxicspikes", "wideguard", "quickguard", "protect", "matblock", "craftyshield",
];

/// `SC_IDS` index of a side-condition id.
pub fn sc_index(id: &str) -> u8 {
    SC_IDS.iter().position(|&s| s == id).unwrap_or_else(|| panic!("unknown side condition {}", id)) as u8
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
                    species: s.transform_save[i].as_ref().map_or(s.species_names[i], |t| t.as_species).as_str(),
                    forme: SPECIES_NAMES[p.id as usize],
                    hp: p.hp,
                    maxhp: p.maxhp,
                    // A KO after the battle ended stays unprocessed in its slot.
                    fainted: p.hp <= 0 && !(state.battle_ended && active_slot.is_some()),
                    status: normalize_status(p.status),
                    active_slot,
                    boosts,
                    item: ITEM_NAMES[p.item as usize],
                    // Gastro Acid / Neutralizing Gas keep the ability id (the
                    // engine stores NONE).
                    ability: match active_slot {
                        Some(sl) if s.active()[sl].gastro_acid() && p.ability == Abilities::NONE => {
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
                                .any(|o| o.active().iter().any(|x| x.trapped() && x.trapper == me && x.pokemon_index.is_some()))
                            {
                                v.push("trapper".to_string());
                            }
                            // A fired two-turn move's `twoturnmove` lasts until the residual.
                            if a.twoturn_fired && a.charging().is_none() && !v.iter().any(|x| x == "twoturnmove") {
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
    let sc_entries = |sc: &SideConditions, order: &[u8], pledges: [i8; 3]| -> Vec<SideConditionEntry> {
        // The harness lists `sideConditions` sorted by id.
        let _ = order;
        let mut v = side_condition_entries(sc);
        for (&n, id) in pledges.iter().zip(["grasspledge", "firepledge", "waterpledge"]) {
            if n != 0 {
                v.push(SideConditionEntry { id, duration: Some(n) });
            }
        }
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
        a_side_conditions: sc_entries(&state.side_one.side_conditions, &state.side_one.sc_order, state.side_one.pledges),
        b_side_conditions: sc_entries(&state.side_two.side_conditions, &state.side_two.sc_order, state.side_two.pledges),
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
        self.sc_order.retain(|&o| present.iter().any(|e| sc_index(e.id) == o));
        for e in present {
            let k = sc_index(e.id);
            if !self.sc_order.contains(&k) {
                self.sc_order.push(k);
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

/// Volatile-condition accessors (`ActiveSlot::vols`).
impl ActiveSlot {
    /// The volatile conditions that are up (bit i = `events::CONDS[i]`).
    #[inline]
    pub fn vols(&self) -> u128 {
        self.vols
    }
    #[inline]
    pub fn truant(&self) -> bool {
        self.vols >> 65 & 1 != 0
    }
    #[inline]
    pub fn set_truant(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 65)) | ((v as u128) << 65);
    }
    #[inline]
    pub fn update_truant(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.truant();
        g(&mut v);
        self.set_truant(v);
    }
    #[inline]
    pub fn take_truant(&mut self) -> bool {
        let v = self.truant();
        self.set_truant(false);
        v
    }
    #[inline]
    pub fn defense_curl(&self) -> bool {
        self.vols >> 70 & 1 != 0
    }
    #[inline]
    pub fn set_defense_curl(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 70)) | ((v as u128) << 70);
    }
    #[inline]
    pub fn update_defense_curl(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.defense_curl();
        g(&mut v);
        self.set_defense_curl(v);
    }
    #[inline]
    pub fn take_defense_curl(&mut self) -> bool {
        let v = self.defense_curl();
        self.set_defense_curl(false);
        v
    }
    #[inline]
    pub fn rage(&self) -> bool {
        self.vols >> 72 & 1 != 0
    }
    #[inline]
    pub fn set_rage(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 72)) | ((v as u128) << 72);
    }
    #[inline]
    pub fn update_rage(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.rage();
        g(&mut v);
        self.set_rage(v);
    }
    #[inline]
    pub fn take_rage(&mut self) -> bool {
        let v = self.rage();
        self.set_rage(false);
        v
    }
    #[inline]
    pub fn magic_coat(&self) -> bool {
        self.vols >> 75 & 1 != 0
    }
    #[inline]
    pub fn set_magic_coat(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 75)) | ((v as u128) << 75);
    }
    #[inline]
    pub fn update_magic_coat(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.magic_coat();
        g(&mut v);
        self.set_magic_coat(v);
    }
    #[inline]
    pub fn take_magic_coat(&mut self) -> bool {
        let v = self.magic_coat();
        self.set_magic_coat(false);
        v
    }
    #[inline]
    pub fn grudge(&self) -> bool {
        self.vols >> 76 & 1 != 0
    }
    #[inline]
    pub fn set_grudge(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 76)) | ((v as u128) << 76);
    }
    #[inline]
    pub fn update_grudge(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.grudge();
        g(&mut v);
        self.set_grudge(v);
    }
    #[inline]
    pub fn take_grudge(&mut self) -> bool {
        let v = self.grudge();
        self.set_grudge(false);
        v
    }
    #[inline]
    pub fn foresight(&self) -> bool {
        self.vols >> 79 & 1 != 0
    }
    #[inline]
    pub fn set_foresight(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 79)) | ((v as u128) << 79);
    }
    #[inline]
    pub fn update_foresight(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.foresight();
        g(&mut v);
        self.set_foresight(v);
    }
    #[inline]
    pub fn take_foresight(&mut self) -> bool {
        let v = self.foresight();
        self.set_foresight(false);
        v
    }
    #[inline]
    pub fn miracle_eye(&self) -> bool {
        self.vols >> 80 & 1 != 0
    }
    #[inline]
    pub fn set_miracle_eye(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 80)) | ((v as u128) << 80);
    }
    #[inline]
    pub fn update_miracle_eye(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.miracle_eye();
        g(&mut v);
        self.set_miracle_eye(v);
    }
    #[inline]
    pub fn take_miracle_eye(&mut self) -> bool {
        let v = self.miracle_eye();
        self.set_miracle_eye(false);
        v
    }
    #[inline]
    pub fn flinched(&self) -> bool {
        self.vols >> 1 & 1 != 0
    }
    #[inline]
    pub fn set_flinched(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 1)) | ((v as u128) << 1);
    }
    #[inline]
    pub fn update_flinched(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.flinched();
        g(&mut v);
        self.set_flinched(v);
    }
    #[inline]
    pub fn take_flinched(&mut self) -> bool {
        let v = self.flinched();
        self.set_flinched(false);
        v
    }
    #[inline]
    pub fn recharging(&self) -> bool {
        self.vols >> 2 & 1 != 0
    }
    #[inline]
    pub fn set_recharging(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 2)) | ((v as u128) << 2);
    }
    #[inline]
    pub fn update_recharging(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.recharging();
        g(&mut v);
        self.set_recharging(v);
    }
    #[inline]
    pub fn take_recharging(&mut self) -> bool {
        let v = self.recharging();
        self.set_recharging(false);
        v
    }
    #[inline]
    pub fn endure(&self) -> bool {
        self.vols >> 17 & 1 != 0
    }
    #[inline]
    pub fn set_endure(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 17)) | ((v as u128) << 17);
    }
    #[inline]
    pub fn update_endure(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.endure();
        g(&mut v);
        self.set_endure(v);
    }
    #[inline]
    pub fn take_endure(&mut self) -> bool {
        let v = self.endure();
        self.set_endure(false);
        v
    }
    #[inline]
    pub fn salt_cure(&self) -> bool {
        self.vols >> 19 & 1 != 0
    }
    #[inline]
    pub fn set_salt_cure(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 19)) | ((v as u128) << 19);
    }
    #[inline]
    pub fn update_salt_cure(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.salt_cure();
        g(&mut v);
        self.set_salt_cure(v);
    }
    #[inline]
    pub fn take_salt_cure(&mut self) -> bool {
        let v = self.salt_cure();
        self.set_salt_cure(false);
        v
    }
    #[inline]
    pub fn focus_punch(&self) -> bool {
        self.vols >> 21 & 1 != 0
    }
    #[inline]
    pub fn set_focus_punch(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 21)) | ((v as u128) << 21);
    }
    #[inline]
    pub fn update_focus_punch(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.focus_punch();
        g(&mut v);
        self.set_focus_punch(v);
    }
    #[inline]
    pub fn take_focus_punch(&mut self) -> bool {
        let v = self.focus_punch();
        self.set_focus_punch(false);
        v
    }
    #[inline]
    pub fn beak_blast(&self) -> bool {
        self.vols >> 22 & 1 != 0
    }
    #[inline]
    pub fn set_beak_blast(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 22)) | ((v as u128) << 22);
    }
    #[inline]
    pub fn update_beak_blast(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.beak_blast();
        g(&mut v);
        self.set_beak_blast(v);
    }
    #[inline]
    pub fn take_beak_blast(&mut self) -> bool {
        let v = self.beak_blast();
        self.set_beak_blast(false);
        v
    }
    #[inline]
    pub fn gastro_acid(&self) -> bool {
        self.vols >> 23 & 1 != 0
    }
    #[inline]
    pub fn set_gastro_acid(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 23)) | ((v as u128) << 23);
    }
    #[inline]
    pub fn update_gastro_acid(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.gastro_acid();
        g(&mut v);
        self.set_gastro_acid(v);
    }
    #[inline]
    pub fn take_gastro_acid(&mut self) -> bool {
        let v = self.gastro_acid();
        self.set_gastro_acid(false);
        v
    }
    #[inline]
    pub fn destinybond(&self) -> bool {
        self.vols >> 25 & 1 != 0
    }
    #[inline]
    pub fn set_destinybond(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 25)) | ((v as u128) << 25);
    }
    #[inline]
    pub fn update_destinybond(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.destinybond();
        g(&mut v);
        self.set_destinybond(v);
    }
    #[inline]
    pub fn take_destinybond(&mut self) -> bool {
        let v = self.destinybond();
        self.set_destinybond(false);
        v
    }
    #[inline]
    pub fn smackdown(&self) -> bool {
        self.vols >> 31 & 1 != 0
    }
    #[inline]
    pub fn set_smackdown(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 31)) | ((v as u128) << 31);
    }
    #[inline]
    pub fn update_smackdown(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.smackdown();
        g(&mut v);
        self.set_smackdown(v);
    }
    #[inline]
    pub fn take_smackdown(&mut self) -> bool {
        let v = self.smackdown();
        self.set_smackdown(false);
        v
    }
    #[inline]
    pub fn cursed(&self) -> bool {
        self.vols >> 32 & 1 != 0
    }
    #[inline]
    pub fn set_cursed(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 32)) | ((v as u128) << 32);
    }
    #[inline]
    pub fn update_cursed(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.cursed();
        g(&mut v);
        self.set_cursed(v);
    }
    #[inline]
    pub fn take_cursed(&mut self) -> bool {
        let v = self.cursed();
        self.set_cursed(false);
        v
    }
    #[inline]
    pub fn flash_fire(&self) -> bool {
        self.vols >> 35 & 1 != 0
    }
    #[inline]
    pub fn set_flash_fire(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 35)) | ((v as u128) << 35);
    }
    #[inline]
    pub fn update_flash_fire(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.flash_fire();
        g(&mut v);
        self.set_flash_fire(v);
    }
    #[inline]
    pub fn take_flash_fire(&mut self) -> bool {
        let v = self.flash_fire();
        self.set_flash_fire(false);
        v
    }
    #[inline]
    pub fn charge(&self) -> bool {
        self.vols >> 36 & 1 != 0
    }
    #[inline]
    pub fn set_charge(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 36)) | ((v as u128) << 36);
    }
    #[inline]
    pub fn update_charge(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.charge();
        g(&mut v);
        self.set_charge(v);
    }
    #[inline]
    pub fn take_charge(&mut self) -> bool {
        let v = self.charge();
        self.set_charge(false);
        v
    }
    #[inline]
    pub fn no_retreat(&self) -> bool {
        self.vols >> 37 & 1 != 0
    }
    #[inline]
    pub fn set_no_retreat(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 37)) | ((v as u128) << 37);
    }
    #[inline]
    pub fn update_no_retreat(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.no_retreat();
        g(&mut v);
        self.set_no_retreat(v);
    }
    #[inline]
    pub fn take_no_retreat(&mut self) -> bool {
        let v = self.no_retreat();
        self.set_no_retreat(false);
        v
    }
    #[inline]
    pub fn focus_energy(&self) -> bool {
        self.vols >> 38 & 1 != 0
    }
    #[inline]
    pub fn set_focus_energy(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 38)) | ((v as u128) << 38);
    }
    #[inline]
    pub fn update_focus_energy(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.focus_energy();
        g(&mut v);
        self.set_focus_energy(v);
    }
    #[inline]
    pub fn take_focus_energy(&mut self) -> bool {
        let v = self.focus_energy();
        self.set_focus_energy(false);
        v
    }
    #[inline]
    pub fn glaive_rush(&self) -> bool {
        self.vols >> 42 & 1 != 0
    }
    #[inline]
    pub fn set_glaive_rush(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 42)) | ((v as u128) << 42);
    }
    #[inline]
    pub fn update_glaive_rush(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.glaive_rush();
        g(&mut v);
        self.set_glaive_rush(v);
    }
    #[inline]
    pub fn take_glaive_rush(&mut self) -> bool {
        let v = self.glaive_rush();
        self.set_glaive_rush(false);
        v
    }
    #[inline]
    pub fn minimize(&self) -> bool {
        self.vols >> 43 & 1 != 0
    }
    #[inline]
    pub fn set_minimize(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 43)) | ((v as u128) << 43);
    }
    #[inline]
    pub fn update_minimize(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.minimize();
        g(&mut v);
        self.set_minimize(v);
    }
    #[inline]
    pub fn take_minimize(&mut self) -> bool {
        let v = self.minimize();
        self.set_minimize(false);
        v
    }
    #[inline]
    pub fn ingrain(&self) -> bool {
        self.vols >> 44 & 1 != 0
    }
    #[inline]
    pub fn set_ingrain(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 44)) | ((v as u128) << 44);
    }
    #[inline]
    pub fn update_ingrain(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.ingrain();
        g(&mut v);
        self.set_ingrain(v);
    }
    #[inline]
    pub fn take_ingrain(&mut self) -> bool {
        let v = self.ingrain();
        self.set_ingrain(false);
        v
    }
    #[inline]
    pub fn torment(&self) -> bool {
        self.vols >> 50 & 1 != 0
    }
    #[inline]
    pub fn set_torment(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 50)) | ((v as u128) << 50);
    }
    #[inline]
    pub fn update_torment(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.torment();
        g(&mut v);
        self.set_torment(v);
    }
    #[inline]
    pub fn take_torment(&mut self) -> bool {
        let v = self.torment();
        self.set_torment(false);
        v
    }
    #[inline]
    pub fn imprison(&self) -> bool {
        self.vols >> 51 & 1 != 0
    }
    #[inline]
    pub fn set_imprison(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 51)) | ((v as u128) << 51);
    }
    #[inline]
    pub fn update_imprison(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.imprison();
        g(&mut v);
        self.set_imprison(v);
    }
    #[inline]
    pub fn take_imprison(&mut self) -> bool {
        let v = self.imprison();
        self.set_imprison(false);
        v
    }
    #[inline]
    pub fn roost(&self) -> bool {
        self.vols >> 52 & 1 != 0
    }
    #[inline]
    pub fn set_roost(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 52)) | ((v as u128) << 52);
    }
    #[inline]
    pub fn update_roost(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.roost();
        g(&mut v);
        self.set_roost(v);
    }
    #[inline]
    pub fn take_roost(&mut self) -> bool {
        let v = self.roost();
        self.set_roost(false);
        v
    }
    #[inline]
    pub fn aqua_ring(&self) -> bool {
        self.vols >> 53 & 1 != 0
    }
    #[inline]
    pub fn set_aqua_ring(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 53)) | ((v as u128) << 53);
    }
    #[inline]
    pub fn update_aqua_ring(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.aqua_ring();
        g(&mut v);
        self.set_aqua_ring(v);
    }
    #[inline]
    pub fn take_aqua_ring(&mut self) -> bool {
        let v = self.aqua_ring();
        self.set_aqua_ring(false);
        v
    }
    #[inline]
    pub fn electrify(&self) -> bool {
        self.vols >> 62 & 1 != 0
    }
    #[inline]
    pub fn set_electrify(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 62)) | ((v as u128) << 62);
    }
    #[inline]
    pub fn update_electrify(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.electrify();
        g(&mut v);
        self.set_electrify(v);
    }
    #[inline]
    pub fn take_electrify(&mut self) -> bool {
        let v = self.electrify();
        self.set_electrify(false);
        v
    }
    #[inline]
    pub fn power_trick(&self) -> bool {
        self.vols >> 63 & 1 != 0
    }
    #[inline]
    pub fn set_power_trick(&mut self, v: bool) {
        self.vols = (self.vols & !(1u128 << 63)) | ((v as u128) << 63);
    }
    #[inline]
    pub fn update_power_trick(&mut self, g: impl FnOnce(&mut bool)) {
        let mut v = self.power_trick();
        g(&mut v);
        self.set_power_trick(v);
    }
    #[inline]
    pub fn take_power_trick(&mut self) -> bool {
        let v = self.power_trick();
        self.set_power_trick(false);
        v
    }
    #[inline]
    pub fn choice_lock(&self) -> Option<Choices> {
        self.choice_lock
    }
    #[inline]
    pub fn set_choice_lock(&mut self, v: Option<Choices>) {
        self.choice_lock = v;
        let v = self.choice_lock;
        self.vols = (self.vols & !(1u128 << 64)) | (((v.is_some()) as u128) << 64);
    }
    #[inline]
    pub fn update_choice_lock(&mut self, g: impl FnOnce(&mut Option<Choices>)) {
        g(&mut self.choice_lock);
        let v = self.choice_lock;
        self.vols = (self.vols & !(1u128 << 64)) | (((v.is_some()) as u128) << 64);
    }
    #[inline]
    pub fn take_choice_lock(&mut self) -> Option<Choices> {
        let v = std::mem::take(&mut self.choice_lock);
        let v = self.choice_lock;
        self.vols = (self.vols & !(1u128 << 64)) | (((v.is_some()) as u128) << 64);
        v
    }
    #[inline]
    pub fn laser_focus(&self) -> u8 {
        self.laser_focus
    }
    #[inline]
    pub fn set_laser_focus(&mut self, v: u8) {
        self.laser_focus = v;
        let v = self.laser_focus;
        self.vols = (self.vols & !(1u128 << 66)) | (((v > 0) as u128) << 66);
    }
    #[inline]
    pub fn update_laser_focus(&mut self, g: impl FnOnce(&mut u8)) {
        g(&mut self.laser_focus);
        let v = self.laser_focus;
        self.vols = (self.vols & !(1u128 << 66)) | (((v > 0) as u128) << 66);
    }
    #[inline]
    pub fn take_laser_focus(&mut self) -> u8 {
        let v = std::mem::take(&mut self.laser_focus);
        let v = self.laser_focus;
        self.vols = (self.vols & !(1u128 << 66)) | (((v > 0) as u128) << 66);
        v
    }
    #[inline]
    pub fn fury_cutter(&self) -> Option<(u8, u8)> {
        self.fury_cutter
    }
    #[inline]
    pub fn set_fury_cutter(&mut self, v: Option<(u8, u8)>) {
        self.fury_cutter = v;
        let v = self.fury_cutter;
        self.vols = (self.vols & !(1u128 << 67)) | (((v.is_some()) as u128) << 67);
    }
    #[inline]
    pub fn update_fury_cutter(&mut self, g: impl FnOnce(&mut Option<(u8, u8)>)) {
        g(&mut self.fury_cutter);
        let v = self.fury_cutter;
        self.vols = (self.vols & !(1u128 << 67)) | (((v.is_some()) as u128) << 67);
    }
    #[inline]
    pub fn take_fury_cutter(&mut self) -> Option<(u8, u8)> {
        let v = std::mem::take(&mut self.fury_cutter);
        let v = self.fury_cutter;
        self.vols = (self.vols & !(1u128 << 67)) | (((v.is_some()) as u128) << 67);
        v
    }
    #[inline]
    pub fn telekinesis(&self) -> u8 {
        self.telekinesis
    }
    #[inline]
    pub fn set_telekinesis(&mut self, v: u8) {
        self.telekinesis = v;
        let v = self.telekinesis;
        self.vols = (self.vols & !(1u128 << 68)) | (((v > 0) as u128) << 68);
    }
    #[inline]
    pub fn update_telekinesis(&mut self, g: impl FnOnce(&mut u8)) {
        g(&mut self.telekinesis);
        let v = self.telekinesis;
        self.vols = (self.vols & !(1u128 << 68)) | (((v > 0) as u128) << 68);
    }
    #[inline]
    pub fn take_telekinesis(&mut self) -> u8 {
        let v = std::mem::take(&mut self.telekinesis);
        let v = self.telekinesis;
        self.vols = (self.vols & !(1u128 << 68)) | (((v > 0) as u128) << 68);
        v
    }
    #[inline]
    pub fn bide(&self) -> Option<(u8, i16, Option<(Side, crate::state::PokemonIndex)>)> {
        self.bide
    }
    #[inline]
    pub fn set_bide(&mut self, v: Option<(u8, i16, Option<(Side, crate::state::PokemonIndex)>)>) {
        self.bide = v;
        let v = self.bide;
        self.vols = (self.vols & !(1u128 << 71)) | (((v.is_some()) as u128) << 71);
    }
    #[inline]
    pub fn update_bide(&mut self, g: impl FnOnce(&mut Option<(u8, i16, Option<(Side, crate::state::PokemonIndex)>)>)) {
        g(&mut self.bide);
        let v = self.bide;
        self.vols = (self.vols & !(1u128 << 71)) | (((v.is_some()) as u128) << 71);
    }
    #[inline]
    pub fn take_bide(&mut self) -> Option<(u8, i16, Option<(Side, crate::state::PokemonIndex)>)> {
        let v = std::mem::take(&mut self.bide);
        let v = self.bide;
        self.vols = (self.vols & !(1u128 << 71)) | (((v.is_some()) as u128) << 71);
        v
    }
    #[inline]
    pub fn micle(&self) -> u8 {
        self.micle
    }
    #[inline]
    pub fn set_micle(&mut self, v: u8) {
        self.micle = v;
        let v = self.micle;
        self.vols = (self.vols & !(1u128 << 77)) | (((v > 0) as u128) << 77);
    }
    #[inline]
    pub fn update_micle(&mut self, g: impl FnOnce(&mut u8)) {
        g(&mut self.micle);
        let v = self.micle;
        self.vols = (self.vols & !(1u128 << 77)) | (((v > 0) as u128) << 77);
    }
    #[inline]
    pub fn take_micle(&mut self) -> u8 {
        let v = std::mem::take(&mut self.micle);
        let v = self.micle;
        self.vols = (self.vols & !(1u128 << 77)) | (((v > 0) as u128) << 77);
        v
    }
    #[inline]
    pub fn embargo(&self) -> u8 {
        self.embargo
    }
    #[inline]
    pub fn set_embargo(&mut self, v: u8) {
        self.embargo = v;
        let v = self.embargo;
        self.vols = (self.vols & !(1u128 << 78)) | (((v > 0) as u128) << 78);
    }
    #[inline]
    pub fn update_embargo(&mut self, g: impl FnOnce(&mut u8)) {
        g(&mut self.embargo);
        let v = self.embargo;
        self.vols = (self.vols & !(1u128 << 78)) | (((v > 0) as u128) << 78);
    }
    #[inline]
    pub fn take_embargo(&mut self) -> u8 {
        let v = std::mem::take(&mut self.embargo);
        let v = self.embargo;
        self.vols = (self.vols & !(1u128 << 78)) | (((v > 0) as u128) << 78);
        v
    }
    #[inline]
    pub fn confusion(&self) -> u8 {
        self.confusion
    }
    #[inline]
    pub fn set_confusion(&mut self, v: u8) {
        self.confusion = v;
        let v = self.confusion;
        self.vols = (self.vols & !(1u128 << 0)) | (((v > 0) as u128) << 0);
    }
    #[inline]
    pub fn update_confusion(&mut self, g: impl FnOnce(&mut u8)) {
        g(&mut self.confusion);
        let v = self.confusion;
        self.vols = (self.vols & !(1u128 << 0)) | (((v > 0) as u128) << 0);
    }
    #[inline]
    pub fn take_confusion(&mut self) -> u8 {
        let v = std::mem::take(&mut self.confusion);
        let v = self.confusion;
        self.vols = (self.vols & !(1u128 << 0)) | (((v > 0) as u128) << 0);
        v
    }
    #[inline]
    pub fn substitute_health(&self) -> i16 {
        self.substitute_health
    }
    #[inline]
    pub fn set_substitute_health(&mut self, v: i16) {
        self.substitute_health = v;
        let v = self.substitute_health;
        self.vols = (self.vols & !(1u128 << 18)) | (((v > 0) as u128) << 18);
    }
    #[inline]
    pub fn update_substitute_health(&mut self, g: impl FnOnce(&mut i16)) {
        g(&mut self.substitute_health);
        let v = self.substitute_health;
        self.vols = (self.vols & !(1u128 << 18)) | (((v > 0) as u128) << 18);
    }
    #[inline]
    pub fn take_substitute_health(&mut self) -> i16 {
        let v = std::mem::take(&mut self.substitute_health);
        let v = self.substitute_health;
        self.vols = (self.vols & !(1u128 << 18)) | (((v > 0) as u128) << 18);
        v
    }
    #[inline]
    pub fn stockpile(&self) -> u8 {
        self.stockpile
    }
    #[inline]
    pub fn set_stockpile(&mut self, v: u8) {
        self.stockpile = v;
        let v = self.stockpile;
        self.vols = (self.vols & !(1u128 << 20)) | (((v > 0) as u128) << 20);
    }
    #[inline]
    pub fn update_stockpile(&mut self, g: impl FnOnce(&mut u8)) {
        g(&mut self.stockpile);
        let v = self.stockpile;
        self.vols = (self.vols & !(1u128 << 20)) | (((v > 0) as u128) << 20);
    }
    #[inline]
    pub fn take_stockpile(&mut self) -> u8 {
        let v = std::mem::take(&mut self.stockpile);
        let v = self.stockpile;
        self.vols = (self.vols & !(1u128 << 20)) | (((v > 0) as u128) << 20);
        v
    }
    #[inline]
    pub fn partial_trap(&self) -> Option<PartialTrap> {
        self.partial_trap
    }
    #[inline]
    pub fn set_partial_trap(&mut self, v: Option<PartialTrap>) {
        self.partial_trap = v;
        let v = self.partial_trap;
        self.vols = (self.vols & !(1u128 << 26)) | (((v.is_some()) as u128) << 26);
    }
    #[inline]
    pub fn update_partial_trap(&mut self, g: impl FnOnce(&mut Option<PartialTrap>)) {
        g(&mut self.partial_trap);
        let v = self.partial_trap;
        self.vols = (self.vols & !(1u128 << 26)) | (((v.is_some()) as u128) << 26);
    }
    #[inline]
    pub fn take_partial_trap(&mut self) -> Option<PartialTrap> {
        let v = std::mem::take(&mut self.partial_trap);
        let v = self.partial_trap;
        self.vols = (self.vols & !(1u128 << 26)) | (((v.is_some()) as u128) << 26);
        v
    }
    #[inline]
    pub fn yawn(&self) -> u8 {
        self.yawn
    }
    #[inline]
    pub fn set_yawn(&mut self, v: u8) {
        self.yawn = v;
        let v = self.yawn;
        self.vols = (self.vols & !(1u128 << 27)) | (((v > 0) as u128) << 27);
    }
    #[inline]
    pub fn update_yawn(&mut self, g: impl FnOnce(&mut u8)) {
        g(&mut self.yawn);
        let v = self.yawn;
        self.vols = (self.vols & !(1u128 << 27)) | (((v > 0) as u128) << 27);
    }
    #[inline]
    pub fn take_yawn(&mut self) -> u8 {
        let v = std::mem::take(&mut self.yawn);
        let v = self.yawn;
        self.vols = (self.vols & !(1u128 << 27)) | (((v > 0) as u128) << 27);
        v
    }
    #[inline]
    pub fn throat_chop(&self) -> u8 {
        self.throat_chop
    }
    #[inline]
    pub fn set_throat_chop(&mut self, v: u8) {
        self.throat_chop = v;
        let v = self.throat_chop;
        self.vols = (self.vols & !(1u128 << 28)) | (((v > 0) as u128) << 28);
    }
    #[inline]
    pub fn update_throat_chop(&mut self, g: impl FnOnce(&mut u8)) {
        g(&mut self.throat_chop);
        let v = self.throat_chop;
        self.vols = (self.vols & !(1u128 << 28)) | (((v > 0) as u128) << 28);
    }
    #[inline]
    pub fn take_throat_chop(&mut self) -> u8 {
        let v = std::mem::take(&mut self.throat_chop);
        let v = self.throat_chop;
        self.vols = (self.vols & !(1u128 << 28)) | (((v > 0) as u128) << 28);
        v
    }
    #[inline]
    pub fn taunt(&self) -> u8 {
        self.taunt
    }
    #[inline]
    pub fn set_taunt(&mut self, v: u8) {
        self.taunt = v;
        let v = self.taunt;
        self.vols = (self.vols & !(1u128 << 29)) | (((v > 0) as u128) << 29);
    }
    #[inline]
    pub fn update_taunt(&mut self, g: impl FnOnce(&mut u8)) {
        g(&mut self.taunt);
        let v = self.taunt;
        self.vols = (self.vols & !(1u128 << 29)) | (((v > 0) as u128) << 29);
    }
    #[inline]
    pub fn take_taunt(&mut self) -> u8 {
        let v = std::mem::take(&mut self.taunt);
        let v = self.taunt;
        self.vols = (self.vols & !(1u128 << 29)) | (((v > 0) as u128) << 29);
        v
    }
    #[inline]
    pub fn magnet_rise(&self) -> u8 {
        self.magnet_rise
    }
    #[inline]
    pub fn set_magnet_rise(&mut self, v: u8) {
        self.magnet_rise = v;
        let v = self.magnet_rise;
        self.vols = (self.vols & !(1u128 << 30)) | (((v > 0) as u128) << 30);
    }
    #[inline]
    pub fn update_magnet_rise(&mut self, g: impl FnOnce(&mut u8)) {
        g(&mut self.magnet_rise);
        let v = self.magnet_rise;
        self.vols = (self.vols & !(1u128 << 30)) | (((v > 0) as u128) << 30);
    }
    #[inline]
    pub fn take_magnet_rise(&mut self) -> u8 {
        let v = std::mem::take(&mut self.magnet_rise);
        let v = self.magnet_rise;
        self.vols = (self.vols & !(1u128 << 30)) | (((v > 0) as u128) << 30);
        v
    }
    #[inline]
    pub fn follow_me(&self) -> u8 {
        self.follow_me
    }
    #[inline]
    pub fn set_follow_me(&mut self, v: u8) {
        self.follow_me = v;
        let v = self.follow_me;
        self.vols = (self.vols & !(1u128 << 33)) | (((v > 0) as u128) << 33);
    }
    #[inline]
    pub fn update_follow_me(&mut self, g: impl FnOnce(&mut u8)) {
        g(&mut self.follow_me);
        let v = self.follow_me;
        self.vols = (self.vols & !(1u128 << 33)) | (((v > 0) as u128) << 33);
    }
    #[inline]
    pub fn take_follow_me(&mut self) -> u8 {
        let v = std::mem::take(&mut self.follow_me);
        let v = self.follow_me;
        self.vols = (self.vols & !(1u128 << 33)) | (((v > 0) as u128) << 33);
        v
    }
    #[inline]
    pub fn perish(&self) -> u8 {
        self.perish
    }
    #[inline]
    pub fn set_perish(&mut self, v: u8) {
        self.perish = v;
        let v = self.perish;
        self.vols = (self.vols & !(1u128 << 34)) | (((v > 0) as u128) << 34);
    }
    #[inline]
    pub fn update_perish(&mut self, g: impl FnOnce(&mut u8)) {
        g(&mut self.perish);
        let v = self.perish;
        self.vols = (self.vols & !(1u128 << 34)) | (((v > 0) as u128) << 34);
    }
    #[inline]
    pub fn take_perish(&mut self) -> u8 {
        let v = std::mem::take(&mut self.perish);
        let v = self.perish;
        self.vols = (self.vols & !(1u128 << 34)) | (((v > 0) as u128) << 34);
        v
    }
    #[inline]
    pub fn attract(&self) -> Option<(Side, PokemonIndex)> {
        self.attract
    }
    #[inline]
    pub fn set_attract(&mut self, v: Option<(Side, PokemonIndex)>) {
        self.attract = v;
        let v = self.attract;
        self.vols = (self.vols & !(1u128 << 39)) | (((v.is_some()) as u128) << 39);
    }
    #[inline]
    pub fn update_attract(&mut self, g: impl FnOnce(&mut Option<(Side, PokemonIndex)>)) {
        g(&mut self.attract);
        let v = self.attract;
        self.vols = (self.vols & !(1u128 << 39)) | (((v.is_some()) as u128) << 39);
    }
    #[inline]
    pub fn take_attract(&mut self) -> Option<(Side, PokemonIndex)> {
        let v = std::mem::take(&mut self.attract);
        let v = self.attract;
        self.vols = (self.vols & !(1u128 << 39)) | (((v.is_some()) as u128) << 39);
        v
    }
    #[inline]
    pub fn leech_seed(&self) -> Option<(Side, u8)> {
        self.leech_seed
    }
    #[inline]
    pub fn set_leech_seed(&mut self, v: Option<(Side, u8)>) {
        self.leech_seed = v;
        let v = self.leech_seed;
        self.vols = (self.vols & !(1u128 << 40)) | (((v.is_some()) as u128) << 40);
    }
    #[inline]
    pub fn update_leech_seed(&mut self, g: impl FnOnce(&mut Option<(Side, u8)>)) {
        g(&mut self.leech_seed);
        let v = self.leech_seed;
        self.vols = (self.vols & !(1u128 << 40)) | (((v.is_some()) as u128) << 40);
    }
    #[inline]
    pub fn take_leech_seed(&mut self) -> Option<(Side, u8)> {
        let v = std::mem::take(&mut self.leech_seed);
        let v = self.leech_seed;
        self.vols = (self.vols & !(1u128 << 40)) | (((v.is_some()) as u128) << 40);
        v
    }
    #[inline]
    pub fn dragon_cheer(&self) -> Option<bool> {
        self.dragon_cheer
    }
    #[inline]
    pub fn set_dragon_cheer(&mut self, v: Option<bool>) {
        self.dragon_cheer = v;
        let v = self.dragon_cheer;
        self.vols = (self.vols & !(1u128 << 41)) | (((v.is_some()) as u128) << 41);
    }
    #[inline]
    pub fn update_dragon_cheer(&mut self, g: impl FnOnce(&mut Option<bool>)) {
        g(&mut self.dragon_cheer);
        let v = self.dragon_cheer;
        self.vols = (self.vols & !(1u128 << 41)) | (((v.is_some()) as u128) << 41);
    }
    #[inline]
    pub fn take_dragon_cheer(&mut self) -> Option<bool> {
        let v = std::mem::take(&mut self.dragon_cheer);
        let v = self.dragon_cheer;
        self.vols = (self.vols & !(1u128 << 41)) | (((v.is_some()) as u128) << 41);
        v
    }
    #[inline]
    pub fn heal_block(&self) -> u8 {
        self.heal_block
    }
    #[inline]
    pub fn set_heal_block(&mut self, v: u8) {
        self.heal_block = v;
        let v = self.heal_block;
        self.vols = (self.vols & !(1u128 << 45)) | (((v > 0) as u128) << 45);
    }
    #[inline]
    pub fn update_heal_block(&mut self, g: impl FnOnce(&mut u8)) {
        g(&mut self.heal_block);
        let v = self.heal_block;
        self.vols = (self.vols & !(1u128 << 45)) | (((v > 0) as u128) << 45);
    }
    #[inline]
    pub fn take_heal_block(&mut self) -> u8 {
        let v = std::mem::take(&mut self.heal_block);
        let v = self.heal_block;
        self.vols = (self.vols & !(1u128 << 45)) | (((v > 0) as u128) << 45);
        v
    }
    #[inline]
    pub fn syrup_bomb(&self) -> Option<(Side, PokemonIndex, u8)> {
        self.syrup_bomb
    }
    #[inline]
    pub fn set_syrup_bomb(&mut self, v: Option<(Side, PokemonIndex, u8)>) {
        self.syrup_bomb = v;
        let v = self.syrup_bomb;
        self.vols = (self.vols & !(1u128 << 46)) | (((v.is_some()) as u128) << 46);
    }
    #[inline]
    pub fn update_syrup_bomb(&mut self, g: impl FnOnce(&mut Option<(Side, PokemonIndex, u8)>)) {
        g(&mut self.syrup_bomb);
        let v = self.syrup_bomb;
        self.vols = (self.vols & !(1u128 << 46)) | (((v.is_some()) as u128) << 46);
    }
    #[inline]
    pub fn take_syrup_bomb(&mut self) -> Option<(Side, PokemonIndex, u8)> {
        let v = std::mem::take(&mut self.syrup_bomb);
        let v = self.syrup_bomb;
        self.vols = (self.vols & !(1u128 << 46)) | (((v.is_some()) as u128) << 46);
        v
    }
    #[inline]
    pub fn octolock(&self) -> Option<(Side, PokemonIndex)> {
        self.octolock
    }
    #[inline]
    pub fn set_octolock(&mut self, v: Option<(Side, PokemonIndex)>) {
        self.octolock = v;
        let v = self.octolock;
        self.vols = (self.vols & !(1u128 << 47)) | (((v.is_some()) as u128) << 47);
    }
    #[inline]
    pub fn update_octolock(&mut self, g: impl FnOnce(&mut Option<(Side, PokemonIndex)>)) {
        g(&mut self.octolock);
        let v = self.octolock;
        self.vols = (self.vols & !(1u128 << 47)) | (((v.is_some()) as u128) << 47);
    }
    #[inline]
    pub fn take_octolock(&mut self) -> Option<(Side, PokemonIndex)> {
        let v = std::mem::take(&mut self.octolock);
        let v = self.octolock;
        self.vols = (self.vols & !(1u128 << 47)) | (((v.is_some()) as u128) << 47);
        v
    }
    #[inline]
    pub fn encore(&self) -> Option<(crate::choices::Choices, u8)> {
        self.encore
    }
    #[inline]
    pub fn set_encore(&mut self, v: Option<(crate::choices::Choices, u8)>) {
        self.encore = v;
        let v = self.encore;
        self.vols = (self.vols & !(1u128 << 49)) | (((v.is_some()) as u128) << 49);
    }
    #[inline]
    pub fn update_encore(&mut self, g: impl FnOnce(&mut Option<(crate::choices::Choices, u8)>)) {
        g(&mut self.encore);
        let v = self.encore;
        self.vols = (self.vols & !(1u128 << 49)) | (((v.is_some()) as u128) << 49);
    }
    #[inline]
    pub fn take_encore(&mut self) -> Option<(crate::choices::Choices, u8)> {
        let v = std::mem::take(&mut self.encore);
        let v = self.encore;
        self.vols = (self.vols & !(1u128 << 49)) | (((v.is_some()) as u128) << 49);
        v
    }
    #[inline]
    pub fn counter_vol(&self) -> Option<(Option<(Side, u8)>, i16)> {
        self.counter_vol
    }
    #[inline]
    pub fn set_counter_vol(&mut self, v: Option<(Option<(Side, u8)>, i16)>) {
        self.counter_vol = v;
        let v = self.counter_vol;
        self.vols = (self.vols & !(1u128 << 54)) | (((v.is_some()) as u128) << 54);
    }
    #[inline]
    pub fn update_counter_vol(&mut self, g: impl FnOnce(&mut Option<(Option<(Side, u8)>, i16)>)) {
        g(&mut self.counter_vol);
        let v = self.counter_vol;
        self.vols = (self.vols & !(1u128 << 54)) | (((v.is_some()) as u128) << 54);
    }
    #[inline]
    pub fn take_counter_vol(&mut self) -> Option<(Option<(Side, u8)>, i16)> {
        let v = std::mem::take(&mut self.counter_vol);
        let v = self.counter_vol;
        self.vols = (self.vols & !(1u128 << 54)) | (((v.is_some()) as u128) << 54);
        v
    }
    #[inline]
    pub fn mirror_coat_vol(&self) -> Option<(Option<(Side, u8)>, i16)> {
        self.mirror_coat_vol
    }
    #[inline]
    pub fn set_mirror_coat_vol(&mut self, v: Option<(Option<(Side, u8)>, i16)>) {
        self.mirror_coat_vol = v;
        let v = self.mirror_coat_vol;
        self.vols = (self.vols & !(1u128 << 55)) | (((v.is_some()) as u128) << 55);
    }
    #[inline]
    pub fn update_mirror_coat_vol(&mut self, g: impl FnOnce(&mut Option<(Option<(Side, u8)>, i16)>)) {
        g(&mut self.mirror_coat_vol);
        let v = self.mirror_coat_vol;
        self.vols = (self.vols & !(1u128 << 55)) | (((v.is_some()) as u128) << 55);
    }
    #[inline]
    pub fn take_mirror_coat_vol(&mut self) -> Option<(Option<(Side, u8)>, i16)> {
        let v = std::mem::take(&mut self.mirror_coat_vol);
        let v = self.mirror_coat_vol;
        self.vols = (self.vols & !(1u128 << 55)) | (((v.is_some()) as u128) << 55);
        v
    }
    #[inline]
    pub fn ally_switch(&self) -> Option<(u16, u8)> {
        self.ally_switch
    }
    #[inline]
    pub fn set_ally_switch(&mut self, v: Option<(u16, u8)>) {
        self.ally_switch = v;
        let v = self.ally_switch;
        self.vols = (self.vols & !(1u128 << 56)) | (((v.is_some()) as u128) << 56);
    }
    #[inline]
    pub fn update_ally_switch(&mut self, g: impl FnOnce(&mut Option<(u16, u8)>)) {
        g(&mut self.ally_switch);
        let v = self.ally_switch;
        self.vols = (self.vols & !(1u128 << 56)) | (((v.is_some()) as u128) << 56);
    }
    #[inline]
    pub fn take_ally_switch(&mut self) -> Option<(u16, u8)> {
        let v = std::mem::take(&mut self.ally_switch);
        let v = self.ally_switch;
        self.vols = (self.vols & !(1u128 << 56)) | (((v.is_some()) as u128) << 56);
        v
    }
    #[inline]
    pub fn lock_on(&self) -> Option<(Side, PokemonIndex, u8)> {
        self.lock_on
    }
    #[inline]
    pub fn set_lock_on(&mut self, v: Option<(Side, PokemonIndex, u8)>) {
        self.lock_on = v;
        let v = self.lock_on;
        self.vols = (self.vols & !(1u128 << 57)) | (((v.is_some()) as u128) << 57);
    }
    #[inline]
    pub fn update_lock_on(&mut self, g: impl FnOnce(&mut Option<(Side, PokemonIndex, u8)>)) {
        g(&mut self.lock_on);
        let v = self.lock_on;
        self.vols = (self.vols & !(1u128 << 57)) | (((v.is_some()) as u128) << 57);
    }
    #[inline]
    pub fn take_lock_on(&mut self) -> Option<(Side, PokemonIndex, u8)> {
        let v = std::mem::take(&mut self.lock_on);
        let v = self.lock_on;
        self.vols = (self.vols & !(1u128 << 57)) | (((v.is_some()) as u128) << 57);
        v
    }
    #[inline]
    pub fn uproar(&self) -> u8 {
        self.uproar
    }
    #[inline]
    pub fn set_uproar(&mut self, v: u8) {
        self.uproar = v;
        let v = self.uproar;
        self.vols = (self.vols & !(1u128 << 58)) | (((v > 0) as u128) << 58);
    }
    #[inline]
    pub fn update_uproar(&mut self, g: impl FnOnce(&mut u8)) {
        g(&mut self.uproar);
        let v = self.uproar;
        self.vols = (self.vols & !(1u128 << 58)) | (((v > 0) as u128) << 58);
    }
    #[inline]
    pub fn take_uproar(&mut self) -> u8 {
        let v = std::mem::take(&mut self.uproar);
        let v = self.uproar;
        self.vols = (self.vols & !(1u128 << 58)) | (((v > 0) as u128) << 58);
        v
    }
    #[inline]
    pub fn stall(&self) -> Option<(u16, u8)> {
        self.stall
    }
    #[inline]
    pub fn set_stall(&mut self, v: Option<(u16, u8)>) {
        self.stall = v;
        let v = self.stall;
        self.vols = (self.vols & !(1u128 << 59)) | (((v.is_some()) as u128) << 59);
    }
    #[inline]
    pub fn update_stall(&mut self, g: impl FnOnce(&mut Option<(u16, u8)>)) {
        g(&mut self.stall);
        let v = self.stall;
        self.vols = (self.vols & !(1u128 << 59)) | (((v.is_some()) as u128) << 59);
    }
    #[inline]
    pub fn take_stall(&mut self) -> Option<(u16, u8)> {
        let v = std::mem::take(&mut self.stall);
        let v = self.stall;
        self.vols = (self.vols & !(1u128 << 59)) | (((v.is_some()) as u128) << 59);
        v
    }
    #[inline]
    pub fn locked_move(&self) -> Option<(u8, u8)> {
        self.locked_move
    }
    #[inline]
    pub fn set_locked_move(&mut self, v: Option<(u8, u8)>) {
        self.locked_move = v;
        let v = self.locked_move;
        self.vols = (self.vols & !(1u128 << 60)) | (((v.is_some()) as u128) << 60);
    }
    #[inline]
    pub fn update_locked_move(&mut self, g: impl FnOnce(&mut Option<(u8, u8)>)) {
        g(&mut self.locked_move);
        let v = self.locked_move;
        self.vols = (self.vols & !(1u128 << 60)) | (((v.is_some()) as u128) << 60);
    }
    #[inline]
    pub fn take_locked_move(&mut self) -> Option<(u8, u8)> {
        let v = std::mem::take(&mut self.locked_move);
        let v = self.locked_move;
        self.vols = (self.vols & !(1u128 << 60)) | (((v.is_some()) as u128) << 60);
        v
    }
    #[inline]
    pub fn disabled(&self) -> Option<(Choices, u8)> {
        self.disabled
    }
    #[inline]
    pub fn set_disabled(&mut self, v: Option<(Choices, u8)>) {
        self.disabled = v;
        let v = self.disabled;
        self.vols = (self.vols & !(1u128 << 61)) | (((v.is_some()) as u128) << 61);
    }
    #[inline]
    pub fn update_disabled(&mut self, g: impl FnOnce(&mut Option<(Choices, u8)>)) {
        g(&mut self.disabled);
        let v = self.disabled;
        self.vols = (self.vols & !(1u128 << 61)) | (((v.is_some()) as u128) << 61);
    }
    #[inline]
    pub fn take_disabled(&mut self) -> Option<(Choices, u8)> {
        let v = std::mem::take(&mut self.disabled);
        let v = self.disabled;
        self.vols = (self.vols & !(1u128 << 61)) | (((v.is_some()) as u128) << 61);
        v
    }
    #[inline]
    pub fn helping_hand_mult(&self) -> f32 {
        self.helping_hand_mult
    }
    #[inline]
    pub fn set_helping_hand_mult(&mut self, v: f32) {
        self.helping_hand_mult = v;
        let v = self.helping_hand_mult;
        self.vols = (self.vols & !(1u128 << 24)) | (((v > 1.0) as u128) << 24);
    }
    #[inline]
    pub fn update_helping_hand_mult(&mut self, g: impl FnOnce(&mut f32)) {
        g(&mut self.helping_hand_mult);
        let v = self.helping_hand_mult;
        self.vols = (self.vols & !(1u128 << 24)) | (((v > 1.0) as u128) << 24);
    }
    #[inline]
    pub fn take_helping_hand_mult(&mut self) -> f32 {
        let v = std::mem::take(&mut self.helping_hand_mult);
        let v = self.helping_hand_mult;
        self.vols = (self.vols & !(1u128 << 24)) | (((v > 1.0) as u128) << 24);
        v
    }
    #[inline]
    pub fn rollout(&self) -> Option<(u8, u8, u8, u8)> {
        self.rollout
    }
    #[inline]
    pub fn set_rollout(&mut self, v: Option<(u8, u8, u8, u8)>) {
        self.rollout = v;
        self.sync_rollout();
    }
    #[inline]
    pub fn update_rollout(&mut self, g: impl FnOnce(&mut Option<(u8, u8, u8, u8)>)) {
        g(&mut self.rollout);
        self.sync_rollout();
    }
    #[inline]
    pub fn take_rollout(&mut self) -> Option<(u8, u8, u8, u8)> {
        let v = std::mem::take(&mut self.rollout);
        self.sync_rollout();
        v
    }
    #[inline]
    pub fn rollout_ice(&self) -> bool {
        self.rollout_ice
    }
    #[inline]
    pub fn set_rollout_ice(&mut self, v: bool) {
        self.rollout_ice = v;
        self.sync_rollout();
    }
    #[inline]
    pub fn update_rollout_ice(&mut self, g: impl FnOnce(&mut bool)) {
        g(&mut self.rollout_ice);
        self.sync_rollout();
    }
    #[inline]
    pub fn take_rollout_ice(&mut self) -> bool {
        let v = std::mem::take(&mut self.rollout_ice);
        self.sync_rollout();
        v
    }
    #[inline]
    pub fn paradox(&self) -> Option<(u8, bool)> {
        self.paradox
    }
    #[inline]
    pub fn set_paradox(&mut self, v: Option<(u8, bool)>) {
        self.paradox = v;
        self.sync_paradox();
    }
    #[inline]
    pub fn update_paradox(&mut self, g: impl FnOnce(&mut Option<(u8, bool)>)) {
        g(&mut self.paradox);
        self.sync_paradox();
    }
    #[inline]
    pub fn take_paradox(&mut self) -> Option<(u8, bool)> {
        let v = std::mem::take(&mut self.paradox);
        self.sync_paradox();
        v
    }
    #[inline]
    pub fn paradox_quark(&self) -> bool {
        self.paradox_quark
    }
    #[inline]
    pub fn set_paradox_quark(&mut self, v: bool) {
        self.paradox_quark = v;
        self.sync_paradox();
    }
    #[inline]
    pub fn update_paradox_quark(&mut self, g: impl FnOnce(&mut bool)) {
        g(&mut self.paradox_quark);
        self.sync_paradox();
    }
    #[inline]
    pub fn take_paradox_quark(&mut self) -> bool {
        let v = std::mem::take(&mut self.paradox_quark);
        self.sync_paradox();
        v
    }
    #[inline]
    pub fn charging(&self) -> Option<(crate::choices::Choices, u8)> {
        self.charging
    }
    #[inline]
    pub fn set_charging(&mut self, v: Option<(crate::choices::Choices, u8)>) {
        self.charging = v;
        self.sync_charging();
    }
    #[inline]
    pub fn update_charging(&mut self, g: impl FnOnce(&mut Option<(crate::choices::Choices, u8)>)) {
        g(&mut self.charging);
        self.sync_charging();
    }
    #[inline]
    pub fn take_charging(&mut self) -> Option<(crate::choices::Choices, u8)> {
        let v = std::mem::take(&mut self.charging);
        self.sync_charging();
        v
    }
    #[inline]
    pub fn charge_aborted(&self) -> bool {
        self.charge_aborted
    }
    #[inline]
    pub fn set_charge_aborted(&mut self, v: bool) {
        self.charge_aborted = v;
        self.sync_charging();
    }
    #[inline]
    pub fn update_charge_aborted(&mut self, g: impl FnOnce(&mut bool)) {
        g(&mut self.charge_aborted);
        self.sync_charging();
    }
    #[inline]
    pub fn take_charge_aborted(&mut self) -> bool {
        let v = std::mem::take(&mut self.charge_aborted);
        self.sync_charging();
        v
    }
    #[inline]
    pub fn protected(&self) -> bool {
        self.protected
    }
    #[inline]
    pub fn set_protected(&mut self, v: bool) {
        self.protected = v;
        self.sync_protect();
    }
    #[inline]
    pub fn update_protected(&mut self, g: impl FnOnce(&mut bool)) {
        g(&mut self.protected);
        self.sync_protect();
    }
    #[inline]
    pub fn take_protected(&mut self) -> bool {
        let v = std::mem::take(&mut self.protected);
        self.sync_protect();
        v
    }
    #[inline]
    pub fn protect_kind(&self) -> u8 {
        self.protect_kind
    }
    #[inline]
    pub fn set_protect_kind(&mut self, v: u8) {
        self.protect_kind = v;
        self.sync_protect();
    }
    #[inline]
    pub fn update_protect_kind(&mut self, g: impl FnOnce(&mut u8)) {
        g(&mut self.protect_kind);
        self.sync_protect();
    }
    #[inline]
    pub fn take_protect_kind(&mut self) -> u8 {
        let v = std::mem::take(&mut self.protect_kind);
        self.sync_protect();
        v
    }
    #[inline]
    pub fn trapped(&self) -> bool {
        self.trapped
    }
    #[inline]
    pub fn set_trapped(&mut self, v: bool) {
        self.trapped = v;
        self.sync_trap();
    }
    #[inline]
    pub fn update_trapped(&mut self, g: impl FnOnce(&mut bool)) {
        g(&mut self.trapped);
        self.sync_trap();
    }
    #[inline]
    pub fn take_trapped(&mut self) -> bool {
        let v = std::mem::take(&mut self.trapped);
        self.sync_trap();
        v
    }
    #[inline]
    pub fn no_retreat_trap(&self) -> bool {
        self.no_retreat_trap
    }
    #[inline]
    pub fn set_no_retreat_trap(&mut self, v: bool) {
        self.no_retreat_trap = v;
        self.sync_trap();
    }
    #[inline]
    pub fn update_no_retreat_trap(&mut self, g: impl FnOnce(&mut bool)) {
        g(&mut self.no_retreat_trap);
        self.sync_trap();
    }
    #[inline]
    pub fn take_no_retreat_trap(&mut self) -> bool {
        let v = std::mem::take(&mut self.no_retreat_trap);
        self.sync_trap();
        v
    }
    fn sync_rollout(&mut self) {
        let m = (1u128 << 69) | (1u128 << 81);
        let b = match self.rollout { Some(_) if self.rollout_ice => 1u128 << 81, Some(_) => 1u128 << 69, None => 0 };
        self.vols = (self.vols & !m) | b;
    }
    fn sync_paradox(&mut self) {
        let m = (1u128 << 73) | (1u128 << 74);
        let b = match self.paradox { Some(_) if self.paradox_quark => 1u128 << 74, Some(_) => 1u128 << 73, None => 0 };
        self.vols = (self.vols & !m) | b;
    }
    fn sync_charging(&mut self) {
        // Bit 3 = twoturnmove; 4..10 = the semi-invulnerable / Sky Drop moves.
        let m = 0xFFu128 << 3;
        let mut b = 0u128;
        if let Some((mv, _)) = self.charging.filter(|_| !self.charge_aborted) {
            b |= 1 << 3;
            b |= match mv {
                Choices::DIG => 1 << 4,
                Choices::DIVE => 1 << 5,
                Choices::FLY => 1 << 6,
                Choices::BOUNCE => 1 << 7,
                Choices::PHANTOMFORCE => 1 << 8,
                Choices::SHADOWFORCE => 1 << 9,
                Choices::SKYDROP => 1 << 10,
                _ => 0,
            };
        }
        self.vols = (self.vols & !m) | b;
    }
    fn sync_protect(&mut self) {
        // protect_kind 0..=5 -> bits 11..=16, 6 -> bit 82.
        let m = (0x3Fu128 << 11) | (1u128 << 82);
        let b = match (self.protected, self.protect_kind) {
            (false, _) => 0,
            (true, k @ 0..=5) => 1u128 << (11 + k as u32),
            (true, 6) => 1u128 << 82,
            _ => 0,
        };
        self.vols = (self.vols & !m) | b;
    }
    fn sync_trap(&mut self) {
        let b = (self.trapped || self.no_retreat_trap) as u128;
        self.vols = (self.vols & !(1u128 << 48)) | (b << 48);
    }
}

/// Up to 8 flags, one per team member or slot (bit i = entry i).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Bits8(u8);

impl Bits8 {
    #[inline]
    pub fn get(self, i: usize) -> bool {
        self.0 >> i & 1 != 0
    }
    #[inline]
    pub fn set(&mut self, i: usize, v: bool) {
        self.0 = (self.0 & !(1 << i)) | ((v as u8) << i);
    }
}

/// PS `moveSlot.used` for 6 team members x 4 move slots (bit `member * 4 + slot`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MoveBits(u32);

impl MoveBits {
    #[inline]
    pub fn get(self, member: usize, slot: usize) -> bool {
        self.0 >> (member * 4 + slot) & 1 != 0
    }
    #[inline]
    pub fn set(&mut self, member: usize, slot: usize, v: bool) {
        let b = member * 4 + slot;
        self.0 = (self.0 & !(1 << b)) | ((v as u32) << b);
    }
    /// Clear all four move slots of a team member.
    #[inline]
    pub fn clear(&mut self, member: usize) {
        self.0 &= !(0xF << (member * 4));
    }
}

/// A pokemon's gender (PS `M` / `F`; genderless is `None`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gender {
    M,
    F,
}

impl Gender {
    /// The input's `"M"` / `"F"`; anything else is genderless.
    pub fn parse(s: Option<&str>) -> Option<Gender> {
        match s {
            Some("M") => Some(Gender::M),
            Some("F") => Some(Gender::F),
            _ => None,
        }
    }
    /// Both gendered and different (Attract, Captivate).
    pub fn opposite(a: Option<Gender>, b: Option<Gender>) -> bool {
        matches!((a, b), (Some(x), Some(y)) if x != y)
    }
}

// A state copy is a `memcpy`: no field below the top level owns heap memory.
const _: () = assert!(!std::mem::needs_drop::<ActiveSlot>());
const _: () = assert!(!std::mem::needs_drop::<DoublesSide>());

impl Clone for DoublesState {
    /// A `memcpy` of the state, then real clones of the four fields that own
    /// heap memory: the event log and its snapshot, the saved request state
    /// (all empty outside event-recording / step-by-step runs) and a planned
    /// Encore queue change (its `Choice` may own its secondaries).
    fn clone(&self) -> Self {
        let mut c = std::mem::MaybeUninit::<DoublesState>::uninit();
        let p = c.as_mut_ptr();
        // SAFETY: every other field is plain data (no drop glue: checked by
        // `PLAIN_FIELDS`), so a bitwise copy is a valid clone of it; the four
        // owning fields are overwritten (never dropped) with their own clones.
        unsafe {
            std::ptr::copy_nonoverlapping(self, p, 1);
            std::ptr::addr_of_mut!((*p).events).write(self.events.clone());
            std::ptr::addr_of_mut!((*p).ev_snap).write(self.ev_snap.clone());
            std::ptr::addr_of_mut!((*p).request_state).write(self.request_state.clone());
            std::ptr::addr_of_mut!((*p).queue_change).write(self.queue_change.clone());
            c.assume_init()
        }
    }
}

/// The type of every `DoublesState` field except `events`, `ev_snap`,
/// `request_state` and `queue_change` has no drop glue (`Clone` copies them
/// bitwise). A new field that owns heap memory fails here: add it to `Clone`
/// or make it plain data.
const PLAIN_FIELDS: () = {
    assert!(!std::mem::needs_drop::<DoublesSide>());
    assert!(!std::mem::needs_drop::<MoveScratch>());
    assert!(!std::mem::needs_drop::<Option<(Side, PokemonIndex)>>());
    assert!(!std::mem::needs_drop::<Option<(Side, PokemonIndex, Choices)>>());
    assert!(!std::mem::needs_drop::<Option<(Side, PokemonIndex, crate::choices::Choices)>>());
    assert!(!std::mem::needs_drop::<Option<(Side, usize)>>());
    assert!(!std::mem::needs_drop::<Option<(Side, usize, PokemonIndex, Abilities)>>());
    assert!(!std::mem::needs_drop::<Option<(Side, usize, PokemonIndex, Choices, i8)>>());
    assert!(!std::mem::needs_drop::<Option<(u8, i8)>>());
    assert!(!std::mem::needs_drop::<Option<Choices>>());
    assert!(!std::mem::needs_drop::<Option<usize>>());
    assert!(!std::mem::needs_drop::<StateTerrain>());
    assert!(!std::mem::needs_drop::<StateTrickRoom>());
    assert!(!std::mem::needs_drop::<StateWeather>());
    assert!(!std::mem::needs_drop::<bool>());
    assert!(!std::mem::needs_drop::<crate::copyvec::CopyVec<(Side, PokemonIndex), 16>>());
    assert!(!std::mem::needs_drop::<crate::copyvec::CopyVec<(Side, PokemonIndex), 8>>());
    assert!(!std::mem::needs_drop::<crate::copyvec::CopyVec<(Side, PokemonIndex, Abilities), 3>>());
    assert!(!std::mem::needs_drop::<crate::copyvec::CopyVec<(Side, PokemonIndex, Abilities), 4>>());
    assert!(!std::mem::needs_drop::<crate::copyvec::CopyVec<(Side, PokemonIndex, Option<(Choices, u8)>), 16>>());
    assert!(!std::mem::needs_drop::<crate::copyvec::CopyVec<(Side, PokemonIndex, Option<(u8, u8)>), 16>>());
    assert!(!std::mem::needs_drop::<crate::copyvec::CopyVec<(Side, PokemonIndex, Side, PokemonIndex, Choices), 16>>());
    assert!(!std::mem::needs_drop::<crate::copyvec::CopyVec<(Side, usize), 8>>());
    assert!(!std::mem::needs_drop::<crate::copyvec::CopyVec<(Side, usize, bool), 8>>());
    assert!(!std::mem::needs_drop::<crate::copyvec::CopyVec<(crate::doubles::state::Side, PokemonIndex, TeamBoosts), 8>>());
    assert!(!std::mem::needs_drop::<crate::copyvec::CopyVec<(crate::doubles::state::Side, usize), 8>>());
    assert!(!std::mem::needs_drop::<crate::copyvec::CopyVec<(crate::doubles::state::Side, usize, PokemonIndex), 8>>());
    assert!(!std::mem::needs_drop::<crate::copyvec::CopyVec<Side, 32>>());
    assert!(!std::mem::needs_drop::<std::cell::Cell<u32>>());
    assert!(!std::mem::needs_drop::<u32>());
};
const _: () = PLAIN_FIELDS;
