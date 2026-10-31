//! Legal choices for an active slot at a move request, mirroring PS
//! `pokemon.getMoveRequestData()` (sim/pokemon.ts): the hard-locked move or
//! every known move with its `disabled` flag, Struggle when nothing is usable,
//! and whether the pokemon is trapped. Read between turns, when PS makes its
//! move request.

use serde::Serialize;

use crate::choices::{Choices, MoveCategory};
use crate::doubles::battle::{has_gravity_flag, imprison_blocks, try_trap_calls};
use crate::doubles::move_targets::{ps_move_target, Tgt};
use crate::doubles::state::{DoublesState, Side};
use crate::engine::items::Items;
use crate::state::{PokemonIndex, PokemonType};

/// One move a slot can be asked to use.
#[derive(Debug, Clone, Serialize)]
pub struct MoveOption {
    /// PS move id ("recharge" and "struggle" included).
    pub id: String,
    /// The move slot (0-3) to put in the `Move` action.
    pub move_slot: usize,
    /// The `struggle` flag of the `Move` action.
    pub struggle: bool,
    /// PP left (0 for recharge / Struggle / a locked move).
    pub pp: i8,
    pub disabled: bool,
    /// Valid signed target locations (1/2 = foe slot, -1/-2 = own slot);
    /// `[0]` for a move whose target the player does not choose.
    pub targets: Vec<i8>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SlotOptions {
    /// Every move PS offers (the one locked move, or all known moves); a
    /// `disabled` one can not be chosen.
    pub moves: Vec<MoveOption>,
    /// Locked into its move (recharge, two-turn, Outrage, Uproar).
    pub locked: bool,
    pub trapped: bool,
    /// Team positions this slot may switch to (empty when trapped).
    pub switches: Vec<usize>,
}

fn ps_id(id: Choices) -> String {
    id.to_string().to_lowercase()
}

/// PS `getLockedMove` (`LockMove` event): the move a hard-locked pokemon must use.
fn locked_move(state: &DoublesState, side: Side, slot: usize) -> Option<Choices> {
    let a = &state.side(side).active()[slot];
    if a.recharging {
        // PS `getMoves` returns the "recharge" pseudo-move.
        return Some(Choices::NONE);
    }
    if let Some((mv, _)) = a.charging {
        return Some(mv);
    }
    if a.locked_move.is_some() {
        return a.locked_move_id.or(a.last_move);
    }
    if a.uproar > 0 {
        return Some(Choices::UPROAR);
    }
    None
}

/// The `onDisableMove` handlers (and 0 PP) that disable a known move.
fn move_disabled(state: &DoublesState, side: Side, slot: usize, idx: PokemonIndex, m: &crate::state::Move) -> bool {
    let a = &state.side(side).active()[slot];
    let p = state.side(side).team_member(idx);
    m.pp <= 0
        || m.disabled
        || a.disabled.is_some_and(|(d, _)| d == m.id)
        || imprison_blocks(state, side, m.id)
        || (a.taunt > 0 && m.choice().category == MoveCategory::Status)
        // PS data/mods/champions/moves.ts fakeout / firstimpression.
        || (matches!(m.id, Choices::FAKEOUT | Choices::FIRSTIMPRESSION) && a.active_move_actions > 0)
        || (state.gravity.active && has_gravity_flag(m.id))
        || (a.heal_block > 0 && m.choice().flags.heal)
        || (a.throat_chop > 0 && m.choice().flags.sound)
        || (a.torment && a.last_move.is_some_and(|l| l == m.id && l != Choices::STRUGGLE))
        // PS data/moves.ts encore `onDisableMove`: only the encored move.
        || a.encore.is_some_and(|(e, _)| e != m.id)
        // PS data/items.ts choicescarf -> `choicelock` `onDisableMove`.
        || (p.item == Items::CHOICESCARF
            && a.last_move.is_some_and(|l| l != m.id && l != Choices::STRUGGLE && p.moves.into_iter().any(|k| k.id == l)))
        // PS data/moves.ts gigatonhammer / bloodmoon `onDisableMove`.
        || (matches!(m.id, Choices::GIGATONHAMMER | Choices::BLOODMOON) && a.last_move == Some(m.id))
}

/// PS target locations for a move used from `slot` (see `MoveOption::targets`).
fn targets(state: &DoublesState, side: Side, slot: usize, id: Choices) -> Vec<i8> {
    let p = state.side(side).slot_pokemon(slot);
    let mut tgt = Tgt::from_ps(ps_move_target(&ps_id(id)));
    // PS `getMoves` target overrides.
    match id {
        Choices::CURSE if !p.is_some_and(|p| p.has_type(&PokemonType::GHOST)) => tgt = Tgt::SelfTarget,
        Choices::POLLENPUFF if state.side(side).active()[slot].heal_block > 0 => tgt = Tgt::AdjacentFoe,
        _ => {}
    }
    let own = -(slot as i8 + 1);
    let ally = -(2 - slot as i8);
    let ally_alive = state.side(side).slot_pokemon(1 - slot).is_some_and(|p| p.hp > 0);
    match tgt {
        Tgt::Normal | Tgt::Any => {
            let mut t = vec![1, 2];
            if ally_alive {
                t.push(ally);
            }
            t
        }
        Tgt::AdjacentFoe => vec![1, 2],
        // An absent ally is still a legal (failing) target.
        Tgt::AdjacentAlly => vec![ally],
        Tgt::AdjacentAllyOrSelf => {
            if ally_alive {
                vec![own, ally]
            } else {
                vec![own]
            }
        }
        _ => vec![0],
    }
}

/// The choices PS offers the pokemon in `slot` of `side`; `None` for an empty
/// or fainted slot (its action is a pass).
pub fn slot_options(state: &DoublesState, side: Side, slot: usize) -> Option<SlotOptions> {
    let s = state.side(side);
    let idx = s.active()[slot].pokemon_index?;
    let p = s.team_member(idx);
    if p.hp <= 0 {
        return None;
    }
    let locked = locked_move(state, side, slot);
    let mut moves = Vec::new();
    match locked {
        Some(Choices::NONE) => moves.push(MoveOption {
            id: "recharge".to_string(),
            move_slot: 0,
            struggle: false,
            pp: 0,
            disabled: false,
            targets: vec![0],
        }),
        Some(mv) => {
            // PS sends no target for a locked move; the engine reuses
            // `lastMoveTargetLoc`.
            let move_slot = p.moves.into_iter().position(|m| m.id == mv).unwrap_or(0);
            moves.push(MoveOption { id: ps_id(mv), move_slot, struggle: false, pp: 0, disabled: false, targets: vec![0] });
        }
        None => {
            for (i, m) in p.moves.into_iter().enumerate() {
                if m.id == Choices::NONE {
                    continue;
                }
                moves.push(MoveOption {
                    id: ps_id(m.id),
                    move_slot: i,
                    struggle: false,
                    pp: m.pp,
                    disabled: move_disabled(state, side, slot, idx, m),
                    targets: targets(state, side, slot, m.id),
                });
            }
            if moves.iter().all(|m| m.disabled) {
                moves = vec![MoveOption {
                    id: "struggle".to_string(),
                    move_slot: 0,
                    struggle: true,
                    pp: 0,
                    disabled: false,
                    targets: vec![0],
                }];
            }
        }
    }

    // PS `endTurn` `TrapPokemon`: every source calls `tryTrap` (No Retreat's
    // too, PS data/moves.ts noretreat), which Ghost types are immune to
    // (Trick-or-Treat's added type included). Champions Run Away clears it
    // last (data/mods/champions/abilities.ts runaway). A hard-locked pokemon
    // is trapped anyway (`getMoveRequestData`).
    let a = &s.active()[slot];
    let ghost = p.has_type(&PokemonType::GHOST) || a.added_type == PokemonType::GHOST;
    let trap_sources = try_trap_calls(state, side, slot) + a.no_retreat as usize;
    let trapped = locked.is_some()
        || (!ghost && trap_sources > 0 && p.ability != crate::engine::abilities::Abilities::RUNAWAY);
    let switches = if trapped {
        Vec::new()
    } else {
        (0..6)
            .filter(|&i| {
                let q = &s.team[i];
                q.id != crate::pokemon::PokemonName::NONE
                    && q.hp > 0
                    && !s.active().iter().any(|a| a.pokemon_index.map(|x| x as usize) == Some(i))
            })
            .collect()
    };
    Some(SlotOptions { moves, locked: locked.is_some(), trapped, switches })
}
