//! Python bindings for the doubles engine (`poke_engine::doubles`).
//!
//! The doubles engine simulates a whole battle from two teams, a PRNG seed and
//! (optionally) explicit per-turn choices. It has no instruction generation, so
//! these bindings expose the battle runners and their results.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::{pyfunction, pymethods, wrap_pyfunction, Bound, PyResult};
use std::panic::{catch_unwind, AssertUnwindSafe};

use poke_engine::choices::Choices;
use poke_engine::doubles::battle::{run_battle, run_battle_fuzz, run_fuzz_turn, start_fuzz_battle};
use poke_engine::doubles::legal::{slot_options, MoveOption, SlotOptions};
use poke_engine::doubles::rng::Gen5Rng;
use poke_engine::doubles::state::{
    fuzz_snapshot, BattleInput, BattleResult, Boosts, DoublesSide, DoublesState, Evs, FullSlot,
    FuzzAction, FuzzSwitch, FuzzTurn, Ivs, MoveResult, OrderEntry, Side, SideConditionEntry,
    SlotHp, SwitchResult, TeamMember, Teams, TurnEntry,
};
use poke_engine::pokemon::PokemonName;

/// Run `f`, turning an engine panic into a Python `RuntimeError`.
fn catch_engine_panic<T>(f: impl FnOnce() -> T) -> PyResult<T> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|e| {
        let msg = e
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        PyRuntimeError::new_err(format!("doubles engine panic: {}", msg))
    })
}

fn build_state(teams: &Teams, leads_a: &[usize], leads_b: &[usize]) -> DoublesState {
    let mut state = DoublesState::from_teams(teams);
    state.send_out_leads(leads_a, leads_b);
    state
}

fn check_leads(leads: &[usize], team_size: usize, side: &str) -> PyResult<()> {
    if leads.len() != 2 || leads[0] == leads[1] || leads.iter().any(|&i| i >= team_size) {
        return Err(PyValueError::new_err(format!(
            "leads_{} must be two distinct team positions below {}, got {:?}",
            side, team_size, leads
        )));
    }
    Ok(())
}

fn check_team(team: &[PyDoublesPokemon], side: &str) -> PyResult<()> {
    if team.len() < 2 || team.len() > 6 {
        return Err(PyValueError::new_err(format!(
            "team_{} must have 2 to 6 pokemon, got {}",
            side,
            team.len()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

#[derive(Clone)]
#[pyclass(name = "DoublesPokemon", module = "poke_engine", get_all, from_py_object)]
pub struct PyDoublesPokemon {
    pub species: String,
    pub level: u8,
    pub evs: (u16, u16, u16, u16, u16, u16),
    pub ivs: Option<(u8, u8, u8, u8, u8, u8)>,
    pub nature: String,
    pub ability: String,
    pub item: String,
    pub moves: Vec<String>,
    pub gender: Option<String>,
    pub name: Option<String>,
}

impl Into<TeamMember> for PyDoublesPokemon {
    fn into(self) -> TeamMember {
        TeamMember {
            pos: None,
            species: self.species,
            name: self.name,
            level: self.level,
            evs: Evs {
                hp: self.evs.0,
                atk: self.evs.1,
                def: self.evs.2,
                spa: self.evs.3,
                spd: self.evs.4,
                spe: self.evs.5,
            },
            ivs: self.ivs.map(|ivs| Ivs {
                hp: ivs.0,
                atk: ivs.1,
                def: ivs.2,
                spa: ivs.3,
                spd: ivs.4,
                spe: ivs.5,
            }),
            nature: self.nature,
            ability: self.ability,
            item: self.item,
            gender: self.gender,
            moves: self.moves,
        }
    }
}

#[pymethods]
impl PyDoublesPokemon {
    #[new]
    #[pyo3(signature = (
        species,
        moves,
        ability="".to_string(),
        item="".to_string(),
        nature="serious".to_string(),
        evs=(0, 0, 0, 0, 0, 0),
        level=50,
        ivs=None,
        gender=None,
        name=None,
    ))]
    fn new(
        species: String,
        moves: Vec<String>,
        ability: String,
        item: String,
        nature: String,
        evs: (u16, u16, u16, u16, u16, u16),
        level: u8,
        ivs: Option<(u8, u8, u8, u8, u8, u8)>,
        gender: Option<String>,
        name: Option<String>,
    ) -> PyResult<Self> {
        if moves.is_empty() || moves.len() > 4 {
            return Err(PyValueError::new_err(format!(
                "{} must have 1 to 4 moves, got {}",
                species,
                moves.len()
            )));
        }
        Ok(PyDoublesPokemon {
            species,
            level,
            evs,
            ivs,
            nature,
            ability,
            item,
            moves,
            gender,
            name,
        })
    }
    fn __repr__(&self) -> String {
        format!(
            "DoublesPokemon({} @ {}, {}, {:?})",
            self.species, self.item, self.ability, self.moves
        )
    }
}

/// One controller action for one active slot.
#[derive(Clone)]
#[pyclass(name = "DoublesAction", module = "poke_engine", get_all, from_py_object)]
pub struct PyDoublesAction {
    /// "move", "switch" or "pass"
    pub kind: String,
    pub slot: usize,
    pub target: i8,
    pub struggle: bool,
    pub position: usize,
    /// Mega Evolve before moving (the pokemon holds its Mega Stone).
    pub mega: bool,
}

impl Into<FuzzAction> for PyDoublesAction {
    fn into(self) -> FuzzAction {
        match self.kind.as_str() {
            "move" => FuzzAction::Move {
                slot: self.slot,
                target: self.target,
                struggle: self.struggle,
                mega: self.mega,
            },
            "switch" => FuzzAction::Switch {
                position: self.position,
            },
            _ => FuzzAction::Pass,
        }
    }
}

#[pymethods]
impl PyDoublesAction {
    /// Use the move in move slot `slot` (0-3). `target` is a PS location:
    /// 1/2 = foe slot, -1/-2 = own slot (ally moves); ignored for moves
    /// that do not take a target.
    #[staticmethod]
    #[pyo3(name = "move", signature = (slot, target=1, struggle=false, mega=false))]
    fn move_(slot: usize, target: i8, struggle: bool, mega: bool) -> PyResult<Self> {
        if slot > 3 {
            return Err(PyValueError::new_err(format!("move slot must be 0-3, got {}", slot)));
        }
        Ok(PyDoublesAction {
            kind: "move".to_string(),
            slot,
            target,
            struggle,
            position: 0,
            mega,
        })
    }
    /// Switch in the team member at 0-based team `position`.
    #[staticmethod]
    fn switch(position: usize) -> Self {
        PyDoublesAction {
            kind: "switch".to_string(),
            slot: 0,
            target: 0,
            struggle: false,
            position,
            mega: false,
        }
    }
    /// Do nothing (empty active slot with no replacement available).
    #[staticmethod]
    #[pyo3(name = "pass_")]
    fn pass() -> Self {
        PyDoublesAction {
            kind: "pass".to_string(),
            slot: 0,
            target: 0,
            struggle: false,
            position: 0,
            mega: false,
        }
    }
    fn __repr__(&self) -> String {
        match self.kind.as_str() {
            "move" => format!(
                "DoublesAction.move({}, target={}{}{})",
                self.slot,
                self.target,
                if self.struggle { ", struggle=True" } else { "" },
                if self.mega { ", mega=True" } else { "" }
            ),
            "switch" => format!("DoublesAction.switch({})", self.position),
            _ => "DoublesAction.pass_()".to_string(),
        }
    }
}

/// One answer to a switch request made during a turn (pivot or faint).
#[derive(Clone)]
#[pyclass(name = "DoublesForcedSwitch", module = "poke_engine", get_all, from_py_object)]
pub struct PyDoublesForcedSwitch {
    /// "a" or "b"
    pub side: String,
    pub slot: usize,
    pub position: Option<usize>,
    pub pivot: bool,
    pub revive: bool,
}

impl Into<FuzzSwitch> for PyDoublesForcedSwitch {
    fn into(self) -> FuzzSwitch {
        FuzzSwitch {
            side: self.side,
            slot: self.slot,
            position: self.position,
            pivot: self.pivot,
            revive: self.revive,
        }
    }
}

#[pymethods]
impl PyDoublesForcedSwitch {
    #[new]
    #[pyo3(signature = (side, slot, position=None, pivot=false, revive=false))]
    fn new(
        side: String,
        slot: usize,
        position: Option<usize>,
        pivot: bool,
        revive: bool,
    ) -> PyResult<Self> {
        let side = side.to_lowercase();
        if side != "a" && side != "b" {
            return Err(PyValueError::new_err(format!("side must be 'a' or 'b', got {}", side)));
        }
        if slot > 1 {
            return Err(PyValueError::new_err(format!("active slot must be 0 or 1, got {}", slot)));
        }
        Ok(PyDoublesForcedSwitch {
            side,
            slot,
            position,
            pivot,
            revive,
        })
    }
}

/// Both sides' choices for one turn. `a` / `b` hold one action per active
/// slot (index 0 = left, 1 = right).
#[derive(Clone)]
#[pyclass(name = "DoublesTurnChoices", module = "poke_engine", get_all, from_py_object)]
pub struct PyDoublesTurnChoices {
    pub a: Vec<PyDoublesAction>,
    pub b: Vec<PyDoublesAction>,
    pub switches: Vec<PyDoublesForcedSwitch>,
}

impl Into<FuzzTurn> for PyDoublesTurnChoices {
    fn into(self) -> FuzzTurn {
        FuzzTurn {
            a: self.a.into_iter().map(|a| a.into()).collect(),
            b: self.b.into_iter().map(|a| a.into()).collect(),
            switches: self.switches.into_iter().map(|s| s.into()).collect(),
        }
    }
}

impl From<&FuzzTurn> for PyDoublesTurnChoices {
    fn from(turn: &FuzzTurn) -> Self {
        let action = |a: &FuzzAction| match a {
            FuzzAction::Move { slot, target, struggle, mega } => PyDoublesAction {
                kind: "move".to_string(),
                slot: *slot,
                target: *target,
                struggle: *struggle,
                position: 0,
                mega: *mega,
            },
            FuzzAction::Switch { position } => PyDoublesAction::switch(*position),
            FuzzAction::Pass => PyDoublesAction::pass(),
        };
        PyDoublesTurnChoices {
            a: turn.a.iter().map(action).collect(),
            b: turn.b.iter().map(action).collect(),
            switches: turn
                .switches
                .iter()
                .map(|s| PyDoublesForcedSwitch {
                    side: s.side.clone(),
                    slot: s.slot,
                    position: s.position,
                    pivot: s.pivot,
                    revive: s.revive,
                })
                .collect(),
        }
    }
}

#[pymethods]
impl PyDoublesTurnChoices {
    #[new]
    #[pyo3(signature = (a, b, switches=Vec::new()))]
    fn new(
        a: Vec<PyDoublesAction>,
        b: Vec<PyDoublesAction>,
        switches: Vec<PyDoublesForcedSwitch>,
    ) -> PyResult<Self> {
        if a.len() != 2 || b.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "each side needs 2 actions (one per active slot), got a={} b={}",
                a.len(),
                b.len()
            )));
        }
        Ok(PyDoublesTurnChoices { a, b, switches })
    }
}

// ---------------------------------------------------------------------------
// Results of `run_doubles_battle_with_choices` (full-state snapshots)
// ---------------------------------------------------------------------------

#[derive(Clone)]
#[pyclass(name = "DoublesBoosts", module = "poke_engine", get_all, from_py_object)]
pub struct PyDoublesBoosts {
    pub atk: i8,
    #[pyo3(name = "def_")]
    pub def: i8,
    pub spa: i8,
    pub spd: i8,
    pub spe: i8,
    pub acc: i8,
    pub eva: i8,
}

impl From<Boosts> for PyDoublesBoosts {
    fn from(other: Boosts) -> Self {
        PyDoublesBoosts {
            atk: other.atk,
            def: other.def,
            spa: other.spa,
            spd: other.spd,
            spe: other.spe,
            acc: other.acc,
            eva: other.eva,
        }
    }
}

#[derive(Clone)]
#[pyclass(name = "DoublesPokemonState", module = "poke_engine", get_all, from_py_object)]
pub struct PyDoublesPokemonState {
    /// Team position (0-5), the `position` used by switch actions.
    pub position: usize,
    pub species: String,
    pub level: i8,
    pub hp: i16,
    pub maxhp: i16,
    pub fainted: bool,
    /// none | burn | sleep | freeze | paralyze | poison | toxic
    pub status: String,
    /// Active slot (0/1) if on the field, else None.
    pub active_slot: Option<usize>,
    pub boosts: PyDoublesBoosts,
    pub types: (String, String),
    pub ability: String,
    pub item: String,
    /// Unboosted stats (atk, def, spa, spd, spe).
    pub stats: (i16, i16, i16, i16, i16),
    pub moves: Vec<PyDoublesMoveState>,
}

#[derive(Clone)]
#[pyclass(name = "DoublesMoveState", module = "poke_engine", get_all, from_py_object)]
pub struct PyDoublesMoveState {
    pub id: String,
    pub pp: i8,
    pub disabled: bool,
}

fn engine_name(name: impl ToString) -> String {
    name.to_string().to_lowercase()
}

/// The team's state in team position order, from the engine's snapshot plus
/// the team members' data.
fn pokemon_states(side: &DoublesSide, slots: Vec<FullSlot>) -> Vec<PyDoublesPokemonState> {
    let positions = (0..6).filter(|&i| {
        !side.species_names[i].is_empty() || side.team[i].id != PokemonName::NONE
    });
    positions
        .zip(slots)
        .map(|(i, slot)| {
            let p = &side.team[i];
            PyDoublesPokemonState {
                position: i,
                species: slot.species.to_string(),
                level: p.level,
                hp: slot.hp,
                maxhp: slot.maxhp,
                fainted: slot.fainted,
                status: slot.status.to_string(),
                active_slot: slot.active_slot,
                boosts: slot.boosts.into(),
                types: (engine_name(p.types.0), engine_name(p.types.1)),
                ability: engine_name(p.ability),
                item: engine_name(p.item),
                stats: (
                    p.attack,
                    p.defense,
                    p.special_attack,
                    p.special_defense,
                    p.speed,
                ),
                moves: p
                    .moves
                    .into_iter()
                    .filter(|m| m.id != Choices::NONE)
                    .map(|m| PyDoublesMoveState {
                        id: engine_name(m.id),
                        pp: m.pp,
                        disabled: m.disabled,
                    })
                    .collect(),
            }
        })
        .collect()
}

#[pymethods]
impl PyDoublesPokemonState {
    fn __repr__(&self) -> String {
        format!(
            "DoublesPokemonState({} {}/{} {}{})",
            self.species,
            self.hp,
            self.maxhp,
            self.status,
            match self.active_slot {
                Some(s) => format!(" active={}", s),
                None => String::new(),
            }
        )
    }
}

#[derive(Clone)]
#[pyclass(name = "DoublesSideCondition", module = "poke_engine", get_all, from_py_object)]
pub struct PyDoublesSideCondition {
    pub id: String,
    /// None for conditions without a duration (hazards).
    pub duration: Option<i8>,
}

impl From<SideConditionEntry> for PyDoublesSideCondition {
    fn from(other: SideConditionEntry) -> Self {
        PyDoublesSideCondition {
            id: other.id.to_string(),
            duration: other.duration,
        }
    }
}

#[derive(Clone)]
#[pyclass(name = "DoublesTurnState", module = "poke_engine", get_all, from_py_object)]
pub struct PyDoublesTurnState {
    pub turn: u32,
    /// Side A's team in team position order.
    pub a: Vec<PyDoublesPokemonState>,
    /// Side B's team in team position order.
    pub b: Vec<PyDoublesPokemonState>,
    pub weather: String,
    pub terrain: String,
    pub trick_room: bool,
    pub pseudo_weather: Vec<String>,
    pub a_side_conditions: Vec<PyDoublesSideCondition>,
    pub b_side_conditions: Vec<PyDoublesSideCondition>,
    /// Execution order this turn, entries "<A|B><slot>:<species>:<MOVE>".
    pub move_order: Vec<String>,
}

impl PyDoublesTurnState {
    fn from_state(state: &DoublesState, turn: u32, move_order: Vec<OrderEntry>) -> Self {
        let other = fuzz_snapshot(state, move_order);
        PyDoublesTurnState {
            turn,
            a: pokemon_states(&state.side_one, other.a),
            b: pokemon_states(&state.side_two, other.b),
            weather: other.weather.to_string(),
            terrain: other.terrain.to_string(),
            trick_room: other.trick_room,
            pseudo_weather: other.pseudo_weather.iter().map(|s| s.to_string()).collect(),
            a_side_conditions: other.a_side_conditions.into_iter().map(|s| s.into()).collect(),
            b_side_conditions: other.b_side_conditions.into_iter().map(|s| s.into()).collect(),
            move_order: other.move_order.iter().map(|o| o.to_string()).collect(),
        }
    }
}

#[derive(Clone)]
#[pyclass(name = "DoublesChoicesResult", module = "poke_engine", get_all, from_py_object)]
pub struct PyDoublesChoicesResult {
    /// "A", "B", or "draw" (also "draw" when stopped before the battle ends).
    pub winner: String,
    pub total_turns: u32,
    /// State after each turn; `turns[0]` is the state after the leads are sent out.
    pub turns: Vec<PyDoublesTurnState>,
    /// Set when the engine hit an internal problem (e.g. ran out of choices).
    pub error: Option<String>,
    pub prng_draws: Vec<u32>,
    /// What each PRNG draw was used for (only filled when `trace=True`).
    pub prng_draw_ctx: Vec<String>,
}


// ---------------------------------------------------------------------------
// Results of `run_doubles_battle` (auto-move policy)
// ---------------------------------------------------------------------------

#[derive(Clone)]
#[pyclass(name = "DoublesSlotHp", module = "poke_engine", get_all, from_py_object)]
pub struct PyDoublesSlotHp {
    pub name: String,
    pub hp: i16,
    pub maxhp: i16,
    pub fainted: bool,
}

impl From<SlotHp> for PyDoublesSlotHp {
    fn from(other: SlotHp) -> Self {
        PyDoublesSlotHp {
            name: other.name,
            hp: other.hp,
            maxhp: other.maxhp,
            fainted: other.fainted,
        }
    }
}

#[derive(Clone)]
#[pyclass(name = "DoublesMoveResult", module = "poke_engine", get_all, from_py_object)]
pub struct PyDoublesMoveResult {
    pub attacker: String,
    pub move_name: String,
    pub target: String,
    pub damage: i16,
    pub crit: bool,
    pub missed: bool,
}

impl From<MoveResult> for PyDoublesMoveResult {
    fn from(other: MoveResult) -> Self {
        PyDoublesMoveResult {
            attacker: other.attacker,
            move_name: other.move_name,
            target: other.target,
            damage: other.damage,
            crit: other.crit,
            missed: other.missed,
        }
    }
}

#[derive(Clone)]
#[pyclass(name = "DoublesSwitchResult", module = "poke_engine", get_all, from_py_object)]
pub struct PyDoublesSwitchResult {
    pub side: String,
    pub slot: usize,
    pub in_pokemon: String,
}

impl From<SwitchResult> for PyDoublesSwitchResult {
    fn from(other: SwitchResult) -> Self {
        PyDoublesSwitchResult {
            side: other.side,
            slot: other.slot,
            in_pokemon: other.in_pokemon,
        }
    }
}

#[derive(Clone)]
#[pyclass(name = "DoublesTurn", module = "poke_engine", get_all, from_py_object)]
pub struct PyDoublesTurn {
    pub turn: u32,
    pub order: Vec<String>,
    pub moves: Vec<PyDoublesMoveResult>,
    pub switches: Vec<PyDoublesSwitchResult>,
    /// Active slots' HP at the end of the turn (None = empty slot).
    pub hp_a: Vec<Option<PyDoublesSlotHp>>,
    pub hp_b: Vec<Option<PyDoublesSlotHp>>,
}

impl From<TurnEntry> for PyDoublesTurn {
    fn from(other: TurnEntry) -> Self {
        PyDoublesTurn {
            turn: other.turn,
            order: other.order,
            moves: other.moves.into_iter().map(|m| m.into()).collect(),
            switches: other.switches.into_iter().map(|s| s.into()).collect(),
            hp_a: other.hp.a.into_iter().map(|h| h.map(|h| h.into())).collect(),
            hp_b: other.hp.b.into_iter().map(|h| h.map(|h| h.into())).collect(),
        }
    }
}

#[derive(Clone)]
#[pyclass(name = "DoublesBattleResult", module = "poke_engine", get_all, from_py_object)]
pub struct PyDoublesBattleResult {
    pub winner: String,
    pub total_turns: u32,
    pub turns: Vec<PyDoublesTurn>,
    pub prng_draws: Vec<u32>,
}

impl From<BattleResult> for PyDoublesBattleResult {
    fn from(other: BattleResult) -> Self {
        PyDoublesBattleResult {
            winner: other.winner,
            total_turns: other.total_turns,
            turns: other.turns.into_iter().map(|t| t.into()).collect(),
            prng_draws: other.prng_draws,
        }
    }
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

fn to_teams(team_a: Vec<PyDoublesPokemon>, team_b: Vec<PyDoublesPokemon>) -> Teams {
    Teams {
        a: team_a.into_iter().map(|p| p.into()).collect(),
        b: team_b.into_iter().map(|p| p.into()).collect(),
    }
}

/// Run a full doubles battle where both sides use the engine's built-in
/// move policy.
#[pyfunction]
#[pyo3(signature = (team_a, team_b, seed, leads_a=vec![0, 1], leads_b=vec![0, 1]))]
fn run_doubles_battle(
    team_a: Vec<PyDoublesPokemon>,
    team_b: Vec<PyDoublesPokemon>,
    seed: [u64; 4],
    leads_a: Vec<usize>,
    leads_b: Vec<usize>,
) -> PyResult<PyDoublesBattleResult> {
    check_team(&team_a, "a")?;
    check_team(&team_b, "b")?;
    check_leads(&leads_a, team_a.len(), "a")?;
    check_leads(&leads_b, team_b.len(), "b")?;
    let teams = to_teams(team_a, team_b);
    let result = catch_engine_panic(|| {
        let mut state = build_state(&teams, &leads_a, &leads_b);
        let mut rng = Gen5Rng::new(seed);
        run_battle(&mut state, &mut rng)
    })?;
    Ok(result.into())
}

/// Run a doubles battle driven by explicit per-turn choices. Stops when the
/// battle ends, the choices run out, or after `stop_after` turns.
#[pyfunction]
#[pyo3(signature = (
    team_a,
    team_b,
    seed,
    choices,
    leads_a=vec![0, 1],
    leads_b=vec![0, 1],
    stop_after=None,
    trace=false,
))]
fn run_doubles_battle_with_choices(
    team_a: Vec<PyDoublesPokemon>,
    team_b: Vec<PyDoublesPokemon>,
    seed: [u64; 4],
    choices: Vec<PyDoublesTurnChoices>,
    leads_a: Vec<usize>,
    leads_b: Vec<usize>,
    stop_after: Option<u32>,
    trace: bool,
) -> PyResult<PyDoublesChoicesResult> {
    check_team(&team_a, "a")?;
    check_team(&team_b, "b")?;
    check_leads(&leads_a, team_a.len(), "a")?;
    check_leads(&leads_b, team_b.len(), "b")?;
    let teams = to_teams(team_a, team_b);
    let choices: Vec<FuzzTurn> = choices.into_iter().map(|c| c.into()).collect();
    // `run_battle_fuzz`'s loop, with a full snapshot after each turn.
    catch_engine_panic(|| {
        let mut state = build_state(&teams, &leads_a, &leads_b);
        let mut rng = Gen5Rng::new(seed);
        rng.trace = trace;
        start_fuzz_battle(&mut state, &mut rng);
        let mut turns = vec![PyDoublesTurnState::from_state(&state, 0, Vec::new())];
        let mut error = None;
        for (ci, turn) in choices.iter().enumerate() {
            if stop_after.is_some_and(|n| ci as u32 >= n) || state.battle_is_over() != 0.0 {
                break;
            }
            let move_order = run_fuzz_turn(&mut state, &mut rng, turn, &mut error);
            if error.is_some() {
                break;
            }
            turns.push(PyDoublesTurnState::from_state(&state, state.turn, move_order));
            state.turn += 1;
        }
        let ran = (turns.len() - 1) as u32;
        if error.is_none()
            && state.battle_is_over() == 0.0
            && stop_after.map_or(true, |n| ran < n)
        {
            error = Some(format!(
                "choices exhausted at turn {} but the battle is not over",
                state.turn
            ));
        }
        PyDoublesChoicesResult {
            winner: winner_name(&state).unwrap_or("draw").to_string(),
            total_turns: state.turn,
            turns,
            error,
            prng_draws: rng.draw_log.clone(),
            prng_draw_ctx: rng.draw_ctx.clone(),
        }
    })
}

fn winner_name(state: &DoublesState) -> Option<&'static str> {
    match state.battle_is_over() {
        x if x > 0.0 => Some("A"),
        x if x < 0.0 => Some("B"),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Legal choices
// ---------------------------------------------------------------------------

#[derive(Clone)]
#[pyclass(name = "DoublesMoveOption", module = "poke_engine", get_all, from_py_object)]
pub struct PyDoublesMoveOption {
    /// Move id ("recharge" and "struggle" included).
    pub id: String,
    /// The move slot to pass to `DoublesAction.move`.
    pub move_slot: usize,
    pub struggle: bool,
    pub pp: i8,
    pub disabled: bool,
    /// Valid target locations (1/2 = foe slot, -1/-2 = own slot); [0] when
    /// the player does not choose the target.
    pub targets: Vec<i8>,
}

impl From<MoveOption> for PyDoublesMoveOption {
    fn from(o: MoveOption) -> Self {
        PyDoublesMoveOption {
            id: o.id,
            move_slot: o.move_slot,
            struggle: o.struggle,
            pp: o.pp,
            disabled: o.disabled,
            targets: o.targets,
        }
    }
}

#[derive(Clone)]
#[pyclass(name = "DoublesSlotOptions", module = "poke_engine", get_all, from_py_object)]
pub struct PyDoublesSlotOptions {
    /// Every move offered (the locked move, Struggle, or all known moves);
    /// a `disabled` one can not be chosen.
    pub moves: Vec<PyDoublesMoveOption>,
    pub locked: bool,
    pub trapped: bool,
    /// Team positions this slot may switch to.
    pub switches: Vec<usize>,
}

impl From<SlotOptions> for PyDoublesSlotOptions {
    fn from(o: SlotOptions) -> Self {
        PyDoublesSlotOptions {
            moves: o.moves.into_iter().map(|m| m.into()).collect(),
            locked: o.locked,
            trapped: o.trapped,
            switches: o.switches,
        }
    }
}

impl PyDoublesSlotOptions {
    fn actions(&self) -> Vec<PyDoublesAction> {
        let mut out = Vec::new();
        for m in self.moves.iter().filter(|m| !m.disabled) {
            for &target in &m.targets {
                out.push(PyDoublesAction {
                    kind: "move".to_string(),
                    slot: m.move_slot,
                    target,
                    struggle: m.struggle,
                    position: 0,
                    mega: false,
                });
            }
        }
        out.extend(self.switches.iter().map(|&p| PyDoublesAction::switch(p)));
        out
    }

    fn allows(&self, act: &PyDoublesAction) -> bool {
        match act.kind.as_str() {
            "switch" => self.switches.contains(&act.position),
            "move" => self.moves.iter().any(|m| {
                !m.disabled
                    && m.move_slot == act.slot
                    && m.struggle == act.struggle
                    && (m.targets == [0] || m.targets.contains(&act.target))
            }),
            _ => false,
        }
    }
}

#[pymethods]
impl PyDoublesSlotOptions {
    /// Every legal action for this slot: each usable move with each valid
    /// target, then each switch.
    #[pyo3(name = "actions")]
    fn py_actions(&self) -> Vec<PyDoublesAction> {
        self.actions()
    }
}

fn side_from_key(side: &str) -> PyResult<Side> {
    match side.to_lowercase().as_str() {
        "a" => Ok(Side::One),
        "b" => Ok(Side::Two),
        _ => Err(PyValueError::new_err(format!("side must be 'a' or 'b', got {}", side))),
    }
}

fn options_for(state: &DoublesState, side: Side) -> Vec<Option<PyDoublesSlotOptions>> {
    (0..2).map(|sl| slot_options(state, side, sl).map(|o| o.into())).collect()
}

/// Per slot: the legal actions (a pass for an empty slot).
fn slot_actions(options: &[Option<PyDoublesSlotOptions>]) -> Vec<Vec<PyDoublesAction>> {
    options
        .iter()
        .map(|o| match o {
            Some(o) => o.actions(),
            None => vec![PyDoublesAction::pass()],
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Step-by-step battle
// ---------------------------------------------------------------------------

fn side_key(side: Side) -> &'static str {
    match side {
        Side::One => "a",
        Side::Two => "b",
    }
}

/// A switch the battle needs before the turn can go on.
#[derive(Clone)]
#[pyclass(name = "DoublesSwitchRequest", module = "poke_engine", get_all, from_py_object)]
pub struct PyDoublesSwitchRequest {
    /// "a" or "b"
    pub side: String,
    /// Active slot to fill (0 or 1).
    pub slot: usize,
    /// "faint" (end-of-turn replacement), "pivot" (U-turn, Volt Switch,
    /// Baton Pass, Emergency Exit, ...) or "revive" (Revival Blessing: pick a
    /// fainted team member to revive).
    pub kind: String,
    /// Team positions that are valid answers.
    pub options: Vec<usize>,
    /// True when `None` (no switch) is a valid answer.
    pub can_pass: bool,
}

#[pymethods]
impl PyDoublesSwitchRequest {
    fn __repr__(&self) -> String {
        format!(
            "DoublesSwitchRequest(side={}, slot={}, kind={}, options={:?}, can_pass={})",
            self.side, self.slot, self.kind, self.options, self.can_pass
        )
    }
}

/// A turn whose actions are chosen but which waits for switch answers.
struct PendingTurn {
    turn: FuzzTurn,
    /// The battle at the request: mid-turn for pivots, end of turn for faints.
    view: DoublesState,
    requests: Vec<PyDoublesSwitchRequest>,
}

/// A doubles battle played one decision at a time.
///
/// Each turn, `choose` takes both sides' actions. If the turn needs a switch
/// decision (a pivot move, a faint replacement, ...), the battle pauses:
/// `requests` lists what is needed and `switch` answers them one at a time.
/// The engine is deterministic for a seed, so the turn is re-simulated from
/// its start with the answers; the result is the same as giving all the
/// answers up front.
#[pyclass(name = "DoublesBattle", module = "poke_engine", unsendable, skip_from_py_object)]
pub struct PyDoublesBattle {
    /// The battle at the start of the next turn.
    state: DoublesState,
    rng: Gen5Rng,
    pending: Option<PendingTurn>,
    last_move_order: Vec<OrderEntry>,
    history: Vec<FuzzTurn>,
}

impl Clone for PyDoublesBattle {
    fn clone(&self) -> Self {
        PyDoublesBattle {
            state: self.state.clone(),
            rng: self.rng.clone(),
            pending: self.pending.as_ref().map(|p| PendingTurn {
                turn: p.turn.clone(),
                view: p.view.clone(),
                requests: p.requests.clone(),
            }),
            last_move_order: self.last_move_order.clone(),
            history: self.history.clone(),
        }
    }
}

impl PyDoublesBattle {
    /// Team positions that can switch into a slot of `side` in `state`.
    fn bench(state: &DoublesState, side: Side) -> Vec<usize> {
        let s = state.side(side);
        (0..6)
            .filter(|&i| {
                let p = &s.team[i];
                p.id != PokemonName::NONE
                    && p.hp > 0
                    && !s.active().iter().any(|a| a.pokemon_index.map(|x| x as usize) == Some(i))
            })
            .collect()
    }

    fn fainted(state: &DoublesState, side: Side) -> Vec<usize> {
        let s = state.side(side);
        (0..6)
            .filter(|&i| s.team[i].id != PokemonName::NONE && s.team[i].hp <= 0)
            .collect()
    }

    /// Simulate the current turn from its start. Either it finishes (the
    /// battle moves on) or it stops at the first switch request with no answer.
    fn run_turn(&mut self, turn: FuzzTurn) -> PyResult<()> {
        let mut state = self.state.clone();
        let mut rng = self.rng.clone();
        state.unanswered_switches.clear();
        state.request_state = None;
        let mut error = None;
        let move_order = catch_engine_panic(|| run_fuzz_turn(&mut state, &mut rng, &turn, &mut error))?;
        if let Some(e) = error {
            self.pending = None;
            return Err(PyValueError::new_err(e));
        }

        if let Some(&(side, slot, revive)) = state.unanswered_switches.first() {
            let view = *state.request_state.take().expect("request state");
            let options = if revive {
                Self::fainted(&view, side)
            } else {
                Self::bench(&view, side)
            };
            let request = PyDoublesSwitchRequest {
                side: side_key(side).to_string(),
                slot,
                kind: if revive { "revive" } else { "pivot" }.to_string(),
                options,
                can_pass: false,
            };
            self.pending = Some(PendingTurn { turn, view, requests: vec![request] });
            return Ok(());
        }

        // End-of-turn faint replacements: every empty slot of a side that
        // still has a pokemon to send in.
        if state.battle_is_over() == 0.0 {
            let mut requests = Vec::new();
            for side in [Side::One, Side::Two] {
                let bench = Self::bench(&state, side);
                let empty: Vec<usize> = (0..2)
                    .filter(|&sl| state.side(side).slot_pokemon(sl).map_or(true, |p| p.hp <= 0))
                    .collect();
                if bench.is_empty() {
                    continue;
                }
                for &slot in &empty {
                    requests.push(PyDoublesSwitchRequest {
                        side: side_key(side).to_string(),
                        slot,
                        kind: "faint".to_string(),
                        options: bench.clone(),
                        can_pass: bench.len() < empty.len(),
                    });
                }
            }
            if !requests.is_empty() {
                self.pending = Some(PendingTurn { turn, view: state, requests });
                return Ok(());
            }
        }

        state.turn += 1;
        self.state = state;
        self.rng = rng;
        self.pending = None;
        self.last_move_order = move_order;
        self.history.push(turn);
        Ok(())
    }

    fn snapshot(&self) -> PyDoublesTurnState {
        match &self.pending {
            Some(p) => PyDoublesTurnState::from_state(&p.view, p.view.turn, Vec::new()),
            None => PyDoublesTurnState::from_state(
                &self.state,
                self.state.turn - 1,
                self.last_move_order.clone(),
            ),
        }
    }
}

#[pymethods]
impl PyDoublesBattle {
    #[new]
    #[pyo3(signature = (team_a, team_b, seed, leads_a=vec![0, 1], leads_b=vec![0, 1], trace=false, events=false))]
    fn new(
        team_a: Vec<PyDoublesPokemon>,
        team_b: Vec<PyDoublesPokemon>,
        seed: [u64; 4],
        leads_a: Vec<usize>,
        leads_b: Vec<usize>,
        trace: bool,
        events: bool,
    ) -> PyResult<Self> {
        check_team(&team_a, "a")?;
        check_team(&team_b, "b")?;
        check_leads(&leads_a, team_a.len(), "a")?;
        check_leads(&leads_b, team_b.len(), "b")?;
        let teams = to_teams(team_a, team_b);
        let (state, rng) = catch_engine_panic(|| {
            let mut state = build_state(&teams, &leads_a, &leads_b);
            state.record_events = events;
            let mut rng = Gen5Rng::new(seed);
            rng.trace = trace;
            start_fuzz_battle(&mut state, &mut rng);
            (state, rng)
        })?;
        Ok(PyDoublesBattle {
            state,
            rng,
            pending: None,
            last_move_order: Vec::new(),
            history: Vec::new(),
        })
    }

    /// Play a turn: one action per active slot for each side (left, right).
    /// Returns the new state. If the turn needs a switch decision, the
    /// returned state is the battle at that point and `requests` is set.
    ///
    /// With `check=True` an action that `legal_options` does not allow
    /// raises ValueError (and so do both slots switching to one pokemon).
    #[pyo3(signature = (a, b, check=true))]
    fn choose(
        &mut self,
        a: Vec<PyDoublesAction>,
        b: Vec<PyDoublesAction>,
        check: bool,
    ) -> PyResult<PyDoublesTurnState> {
        if self.state.battle_is_over() != 0.0 {
            return Err(PyValueError::new_err("the battle is over"));
        }
        if self.pending.is_some() {
            return Err(PyValueError::new_err(
                "the turn waits for switch answers: call switch() first",
            ));
        }
        let choices = PyDoublesTurnChoices::new(a, b, Vec::new())?;
        if check {
            for (key, side, acts) in [("a", Side::One, &choices.a), ("b", Side::Two, &choices.b)] {
                let options = options_for(&self.state, side);
                for (slot, (act, opt)) in acts.iter().zip(&options).enumerate() {
                    let ok = match opt {
                        Some(o) => o.allows(act),
                        None => act.kind == "pass",
                    };
                    if !ok {
                        return Err(PyValueError::new_err(format!(
                            "illegal action for side {} slot {}: {}",
                            key,
                            slot,
                            act.__repr__()
                        )));
                    }
                }
                if acts[0].kind == "switch" && acts[1].kind == "switch" && acts[0].position == acts[1].position {
                    return Err(PyValueError::new_err(format!(
                        "side {}: both slots switch to position {}",
                        key, acts[0].position
                    )));
                }
            }
        }
        let turn: FuzzTurn = choices.into();
        self.run_turn(turn)?;
        Ok(self.snapshot())
    }

    /// Answer one of `requests`: switch team `position` into `slot` of
    /// `side` (None = no switch, only when the request allows it). Once all
    /// requests are answered the turn goes on; returns the new state.
    #[pyo3(signature = (side, slot, position=None))]
    fn switch(
        &mut self,
        side: String,
        slot: usize,
        position: Option<usize>,
    ) -> PyResult<PyDoublesTurnState> {
        let side = side.to_lowercase();
        let pending = self
            .pending
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("no switch request is pending"))?;
        let k = pending
            .requests
            .iter()
            .position(|r| r.side == side && r.slot == slot)
            .ok_or_else(|| {
                PyValueError::new_err(format!("no switch request for side {} slot {}", side, slot))
            })?;
        let request = &pending.requests[k];
        match position {
            Some(p) if !request.options.contains(&p) => {
                return Err(PyValueError::new_err(format!(
                    "position {} is not a valid answer, options: {:?}",
                    p, request.options
                )))
            }
            None if !request.can_pass => {
                return Err(PyValueError::new_err("this request needs a team position"))
            }
            _ => {}
        }
        let request = pending.requests.remove(k);
        // A faint replacement can not be sent into two slots.
        let empty_left = pending.requests.iter().filter(|r| r.side == side).count();
        for r in pending.requests.iter_mut().filter(|r| r.side == side) {
            r.options.retain(|&o| Some(o) != position);
            r.can_pass = r.options.len() < empty_left;
        }
        pending.turn.switches.push(FuzzSwitch {
            side,
            slot,
            position,
            pivot: request.kind != "faint",
            revive: request.kind == "revive",
        });
        if pending.requests.is_empty() {
            let turn = self.pending.take().unwrap().turn;
            self.run_turn(turn)?;
        }
        Ok(self.snapshot())
    }

    /// Legal choices for `side`'s two active slots at the next turn (None =
    /// empty slot, which passes).
    fn legal_options(&self, side: String) -> PyResult<Vec<Option<PyDoublesSlotOptions>>> {
        Ok(options_for(&self.state, side_from_key(&side)?))
    }

    /// Per active slot of `side`: every legal action for the next turn.
    fn legal_actions(&self, side: String) -> PyResult<Vec<Vec<PyDoublesAction>>> {
        Ok(slot_actions(&options_for(&self.state, side_from_key(&side)?)))
    }

    /// Every legal (left, right) action pair for `side` at the next turn
    /// (the two slots never switch to the same pokemon).
    fn legal_joint_actions(&self, side: String) -> PyResult<Vec<(PyDoublesAction, PyDoublesAction)>> {
        let per_slot = slot_actions(&options_for(&self.state, side_from_key(&side)?));
        let mut out = Vec::new();
        for x in &per_slot[0] {
            for y in &per_slot[1] {
                if x.kind == "switch" && y.kind == "switch" && x.position == y.position {
                    continue;
                }
                out.push((x.clone(), y.clone()));
            }
        }
        Ok(out)
    }

    /// The current state: the battle after the last turn, or at the pending
    /// switch request.
    #[getter]
    fn state(&self) -> PyDoublesTurnState {
        self.snapshot()
    }

    /// Switch requests to answer with `switch` before the turn goes on.
    /// Empty when the battle waits for `choose` or is over.
    #[getter]
    fn requests(&self) -> Vec<PyDoublesSwitchRequest> {
        self.pending.as_ref().map(|p| p.requests.clone()).unwrap_or_default()
    }

    /// The next decision: "turn" (call `choose`), "switch" (call `switch`)
    /// or "over".
    #[getter]
    fn waiting_for(&self) -> &'static str {
        if self.pending.is_some() {
            "switch"
        } else if self.state.battle_is_over() != 0.0 {
            "over"
        } else {
            "turn"
        }
    }

    #[getter]
    fn is_over(&self) -> bool {
        self.state.battle_is_over() != 0.0
    }

    /// "A" or "B" once the battle is over, else None.
    #[getter]
    fn winner(&self) -> Option<&'static str> {
        winner_name(&self.state)
    }

    /// The number of the next turn to play (1 before the first turn).
    #[getter]
    fn turn(&self) -> u32 {
        self.state.turn
    }

    /// Every finished turn's choices, switch answers included. Pass them to
    /// `run_doubles_battle_with_choices` to replay the battle.
    #[getter]
    fn history(&self) -> Vec<PyDoublesTurnChoices> {
        self.history.iter().map(PyDoublesTurnChoices::from).collect()
    }

    /// The typed events of the last turn (list of dicts: kind, side, actor,
    /// target_side, target, move, status, effect, item, ability, value), up
    /// to the pending switch request while one waits. Battle start events
    /// (lead abilities) before the first turn. Empty unless the battle was
    /// made with `events=True`.
    #[getter]
    fn events<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let events = match &self.pending {
            Some(p) => &p.view.events,
            None => &self.state.events,
        };
        pythonize::pythonize(py, events).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn prng_draws(&self) -> Vec<u32> {
        self.rng.draw_log.clone()
    }

    /// An independent copy of the battle (for search / rollouts).
    fn copy(&self) -> Self {
        self.clone()
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
}

/// Run a battle from the JSON input used by the `doubles-battle` /
/// `doubles-fuzz-battle` CLI subcommands and return the CLI's JSON output.
/// Inputs with `choices` use the choice-driven runner.
#[pyfunction]
fn run_doubles_battle_json(input_json: String) -> PyResult<String> {
    let input: BattleInput = serde_json::from_str(&input_json)
        .map_err(|e| PyValueError::new_err(format!("bad doubles battle JSON: {}", e)))?;
    catch_engine_panic(|| {
        let (leads_a, leads_b) = match &input.leads {
            Some(l) => (l.a.clone(), l.b.clone()),
            None => (vec![0, 1], vec![0, 1]),
        };
        let mut state = build_state(&input.teams, &leads_a, &leads_b);
        let mut rng = Gen5Rng::new(input.seed);
        match &input.choices {
            Some(choices) => serde_json::to_string(&run_battle_fuzz(
                &mut state,
                &mut rng,
                choices,
                input.stop_after,
            )),
            None => serde_json::to_string(&run_battle(&mut state, &mut rng)),
        }
    })?
    .map_err(|e| PyRuntimeError::new_err(format!("serialize: {}", e)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_doubles_battle, m)?)?;
    m.add_function(wrap_pyfunction!(run_doubles_battle_with_choices, m)?)?;
    m.add_function(wrap_pyfunction!(run_doubles_battle_json, m)?)?;
    m.add_class::<PyDoublesPokemon>()?;
    m.add_class::<PyDoublesAction>()?;
    m.add_class::<PyDoublesForcedSwitch>()?;
    m.add_class::<PyDoublesTurnChoices>()?;
    m.add_class::<PyDoublesBattle>()?;
    m.add_class::<PyDoublesSwitchRequest>()?;
    m.add_class::<PyDoublesMoveState>()?;
    m.add_class::<PyDoublesMoveOption>()?;
    m.add_class::<PyDoublesSlotOptions>()?;
    m.add_class::<PyDoublesBoosts>()?;
    m.add_class::<PyDoublesPokemonState>()?;
    m.add_class::<PyDoublesSideCondition>()?;
    m.add_class::<PyDoublesTurnState>()?;
    m.add_class::<PyDoublesChoicesResult>()?;
    m.add_class::<PyDoublesSlotHp>()?;
    m.add_class::<PyDoublesMoveResult>()?;
    m.add_class::<PyDoublesSwitchResult>()?;
    m.add_class::<PyDoublesTurn>()?;
    m.add_class::<PyDoublesBattleResult>()?;
    Ok(())
}
