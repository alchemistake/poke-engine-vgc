from typing import List, Tuple, Set, Optional

from poke_engine import Weather, Terrain, PokemonIndex

class VolatileStatusDurations:
    """
    Durations for volatile statuses on a SideSlot.

    :param confusion: Confusion turns remaining
    :type confusion: int
    :param encore: Encore turns remaining
    :type encore: int
    :param lockedmove: Locked move turns remaining
    :type lockedmove: int
    :param slowstart: Slow Start turns remaining
    :type slowstart: int
    :param taunt: Taunt turns remaining
    :type taunt: int
    :param yawn: Yawn turns remaining
    :type yawn: int
    """

    confusion: int
    encore: int
    lockedmove: int
    slowstart: int
    taunt: int
    yawn: int

    def __init__(
        self,
        confusion: int = 0,
        encore: int = 0,
        lockedmove: int = 0,
        slowstart: int = 0,
        taunt: int = 0,
        yawn: int = 0,
    ) -> None: ...

class SideConditions:
    """
    Side conditions affecting a Side.

    :param aurora_veil: Aurora Veil turns remaining
    :type aurora_veil: int
    :param crafty_shield: Crafty Shield turns remaining
    :type crafty_shield: int
    :param healing_wish: Healing Wish turns remaining
    :type healing_wish: int
    :param light_screen: Light Screen turns remaining
    :type light_screen: int
    :param lucky_chant: Lucky Chant turns remaining
    :type lucky_chant: int
    :param lunar_dance: Lunar Dance turns remaining
    :type lunar_dance: int
    :param mat_block: Mat Block turns remaining
    :type mat_block: int
    :param mist: Mist turns remaining
    :type mist: int
    :param protect: Protect turns remaining
    :type protect: int
    :param quick_guard: Quick Guard turns remaining
    :type quick_guard: int
    :param reflect: Reflect turns remaining
    :type reflect: int
    :param safeguard: Safeguard turns remaining
    :type safeguard: int
    :param spikes: Number of Spikes layers
    :type spikes: int
    :param stealth_rock: Stealth Rock present
    :type stealth_rock: int
    :param sticky_web: Sticky Web present
    :type sticky_web: int
    :param tailwind: Tailwind turns remaining
    :type tailwind: int
    :param toxic_count: Toxic counter
    :type toxic_count: int
    :param toxic_spikes: Number of Toxic Spikes layers
    :type toxic_spikes: int
    :param wide_guard: Wide Guard turns remaining
    :type wide_guard: int
    """

    aurora_veil: int
    crafty_shield: int
    healing_wish: int
    light_screen: int
    lucky_chant: int
    lunar_dance: int
    mat_block: int
    mist: int
    protect: int
    quick_guard: int
    reflect: int
    safeguard: int
    spikes: int
    stealth_rock: int
    sticky_web: int
    tailwind: int
    toxic_count: int
    toxic_spikes: int
    wide_guard: int

    def __init__(
        self,
        spikes: int = 0,
        toxic_spikes: int = 0,
        stealth_rock: int = 0,
        sticky_web: int = 0,
        tailwind: int = 0,
        lucky_chant: int = 0,
        lunar_dance: int = 0,
        reflect: int = 0,
        light_screen: int = 0,
        aurora_veil: int = 0,
        crafty_shield: int = 0,
        safeguard: int = 0,
        mist: int = 0,
        protect: int = 0,
        healing_wish: int = 0,
        mat_block: int = 0,
        quick_guard: int = 0,
        toxic_count: int = 0,
        wide_guard: int = 0,
    ) -> None: ...

class Move:
    """
    Represents a Pokemon move.

    :param id: Move name
    :type id: str
    :param disabled: Whether the move is disabled
    :type disabled: bool
    :param pp: Remaining PP
    :type pp: int
    """

    id: str
    disabled: bool
    pp: int

    def __init__(
        self,
        id: str = "none",
        pp: int = 16,
        disabled: bool = False,
    ) -> None: ...

class Pokemon:
    """
    Represents a Pokemon with all its battle properties.

    :param id: Pokemon name
    :type id: str
    :param level: Pokemon level
    :type level: int
    :param types: Current types
    :type types: tuple[str, str]
    :param base_types: Base types. Different from types if the Pokemon had its types changed by a move or ability.
    :type base_types: tuple[str, str]
    :param hp: Current HP
    :type hp: int
    :param maxhp: Maximum HP
    :type maxhp: int
    :param ability: Current ability
    :type ability: str
    :param base_ability: Base ability. Different from ability if the Pokemon had its ability changed by a move or ability.
    :type base_ability: str
    :param item: Held item
    :type item: str
    :param nature: Nature
    :type nature: str
    :param evs: Effort values
    :type evs: tuple[int, int, int, int, int, int]
    :param attack: Attack stat
    :type attack: int
    :param defense: Defense stat
    :type defense: int
    :param special_attack: Special Attack stat
    :type special_attack: int
    :param special_defense: Special Defense stat
    :type special_defense: int
    :param speed: Speed stat
    :type speed: int
    :param status: Status condition
    :type status: str
    :param rest_turns: Rest turns remaining. Decrements from 2.
    :type rest_turns: int
    :param sleep_turns: Turns spent asleep. Increments from 0.
    :type sleep_turns: int
    :param weight_kg: Weight in kilograms
    :type weight_kg: float
    :param mega_evolved: if the Pokemon is mega-evolved
    :type mega_evolved: bool
    :param terastallized: if the Pokemon is terastallized
    :type terastallized: bool
    :param tera_type: Tera type
    :type tera_type: str
    :param moves: List of moves
    :type moves: list[Move]
    :param times_attacked: The number of times this Pokemon has been attacked
    :type times_attacked: int
    :param stellar_boosted_types: For pokemon terastallized into the Stellar type, this keeps track the types that have been boosted by Stellar Boost.
    :type stellar_boosted_types: list[str]
    """

    id: str
    level: int
    types: Tuple[str, str]
    base_types: Tuple[str, str]
    hp: int
    maxhp: int
    ability: str
    base_ability: str
    item: str
    nature: str
    evs: Tuple[int, int, int, int, int, int]
    attack: int
    defense: int
    special_attack: int
    special_defense: int
    speed: int
    status: str
    rest_turns: int
    sleep_turns: int
    weight_kg: float
    mega_evolved: bool
    terastallized: bool
    tera_type: str
    moves: List[Move]
    times_attacked: int
    stellar_boosted_types: Optional[list[str]]

    def __init__(
        self,
        id: str = "pikachu",
        level: int = 50,
        types: Tuple[str, str] = ("normal", "typeless"),
        base_types: Tuple[str, str] = ("normal", "typeless"),
        hp: int = 100,
        maxhp: int = 100,
        ability: str = "none",
        base_ability: str = "",
        item: str = "none",
        nature: str = "serious",
        evs: Tuple[int, int, int, int, int, int] = (85, 85, 85, 85, 85, 85),
        attack: int = 100,
        defense: int = 100,
        special_attack: int = 100,
        special_defense: int = 100,
        speed: int = 100,
        status: str = "none",
        rest_turns: int = 0,
        sleep_turns: int = 0,
        weight_kg: float = 0.0,
        moves: Optional[List[Move]] = None,
        mega_evolved: bool = False,
        terastallized: bool = False,
        tera_type: str = "typeless",
        times_attacked: int = 0,
        stellar_boosted_types: Optional[list[str]] = None,
    ) -> None: ...
    @staticmethod
    def create_fainted() -> Pokemon: ...

class Side:
    """
    Represents a Side

    :param active_index: Index of the active Pokemon
    :type active_index: str
    :param baton_passing: Whether Baton Pass is active
    :type baton_passing: bool
    :param shed_tailing: Whether Shed Tail is active
    :type shed_tailing: bool
    :param volatile_status_durations: Durations for volatile statuses
    :type volatile_status_durations: VolatileStatusDurations
    :param wish: Wish status (turns, HP)
    :type wish: tuple[int, int]
    :param future_sight: Future Sight status (turns, target)
    :type future_sight: tuple[int, str]
    :param force_switch: Whether forced to switch
    :type force_switch: bool
    :param force_trapped: Whether trapped
    :type force_trapped: bool
    :param slow_uturn_move: Whether slow U-turn is active
    :type slow_uturn_move: bool
    :param volatile_statuses: Set of volatile status names
    :type volatile_statuses: set[str]
    :param substitute_health: Substitute HP
    :type substitute_health: int
    :param attack_boost: Attack boost
    :type attack_boost: int
    :param defense_boost: Defense boost
    :type defense_boost: int
    :param special_attack_boost: Special Attack boost
    :type special_attack_boost: int
    :param special_defense_boost: Special Defense boost
    :type special_defense_boost: int
    :param speed_boost: Speed boost
    :type speed_boost: int
    :param accuracy_boost: Accuracy boost
    :type accuracy_boost: int
    :param evasion_boost: Evasion boost
    :type evasion_boost: int
    :param last_used_move: Last used move
    :type last_used_move: str
    :param switch_out_move_second_saved_move: A move waiting to be used after another Pokemon completes it's pivot move
    :type switch_out_move_second_saved_move: str
    """

    pokemon: List[Pokemon]
    side_conditions: SideConditions
    active_index: PokemonIndex
    baton_passing: bool
    shed_tailing: bool
    volatile_status_durations: VolatileStatusDurations
    wish: Tuple[int, int]
    future_sight: Tuple[int, str]
    force_switch: bool
    force_trapped: bool
    slow_uturn_move: bool
    volatile_statuses: Set[str]
    substitute_health: int
    attack_boost: int
    defense_boost: int
    special_attack_boost: int
    special_defense_boost: int
    speed_boost: int
    accuracy_boost: int
    evasion_boost: int
    last_used_move: str
    switch_out_move_second_saved_move: str

    def __init__(
        self,
        pokemon: List[Pokemon] = None,
        active_index: PokemonIndex = PokemonIndex.P0,
        baton_passing: bool = False,
        shed_tailing: bool = False,
        volatile_status_durations: Optional[VolatileStatusDurations] = None,
        wish: Tuple[int, int] = (0, 0),
        future_sight: Tuple[int, str] = (0, "0"),
        force_switch: bool = False,
        force_trapped: bool = False,
        slow_uturn_move: bool = False,
        volatile_statuses: Optional[Set[str]] = None,
        substitute_health: int = 0,
        attack_boost: int = 0,
        defense_boost: int = 0,
        special_attack_boost: int = 0,
        special_defense_boost: int = 0,
        speed_boost: int = 0,
        accuracy_boost: int = 0,
        evasion_boost: int = 0,
        last_used_move: str = "move:none",
        switch_out_move_second_saved_move: str = "none",
        side_conditions: Optional[SideConditions] = None,
    ) -> None: ...

class StateInstructions:
    """State instructions with percentage and instruction list."""

    percentage: float
    instruction_list: List["Instruction"]

    def __repr__(self) -> str: ...

class Instruction:
    """Represents a single instruction."""
    def __repr__(self) -> str: ...

class State:
    """
    Represents a Pokemon battle state

    :param side_one: The first side of the battle
    :type side_one: Side
    :param side_two: The second side of the battle
    :type side_two: Side
    :param weather: The current weather condition
    :type weather: Weather
    :param weather_turns_remaining: The number of turns remaining for the current weather condition
    :type weather_turns_remaining: int
    :param terrain: The current terrain condition
    :type terrain: str
    :param terrain_turns_remaining: The number of turns remaining for the current terrain condition
    :type terrain_turns_remaining: int
    :param trick_room: Whether Trick Room is active
    :type trick_room: bool
    """

    side_one: Side
    side_two: Side
    weather: str
    weather_turns_remaining: int
    terrain: str
    terrain_turns_remaining: int
    trick_room: bool
    trick_room_turns_remaining: int
    team_preview: bool

    def __init__(
        self,
        side_one: Side = None,
        side_two: Side = None,
        weather: Weather = Weather.NONE,
        weather_turns_remaining: int = 0,
        terrain: Terrain = Terrain.NONE,
        terrain_turns_remaining: int = 0,
        trick_room: bool = False,
        trick_room_turns_remaining: int = 0,
        team_preview: bool = False,
    ) -> None: ...
    def apply_instructions(self, instructions: StateInstructions) -> State: ...
    def reverse_instructions(self, instructions: StateInstructions) -> State: ...
    @classmethod
    def from_string(cls, state_str: str) -> State: ...
    def to_string(self) -> str: ...

class MctsSideResult:
    """Result for one side from MCTS."""

    move_choice: Tuple[str, str]
    total_score: float
    visits: int

class MctsResult:
    """Result from MCTS algorithm."""

    side_one: List[MctsSideResult]
    side_two: List[MctsSideResult]
    iteration_count: int

def mcts(
    py_state: State, duration_ms: int, iterations: int, threads: int
) -> MctsResult:
    """
    Perform Monte Carlo Tree Search on the given state.

    :param py_state: The game state to analyze
    :param duration_ms: Duration in milliseconds to run MCTS. ignored if iterations > 0.
    :param iterations: Exact number of monte-carlo iterations to run
    :param threads: Number of threads to use for MCTS
    :return: MCTS results for both sides
    """
    ...

def generate_instructions(
    py_state: State,
    side_one_move: str,
    side_two_move: str,
) -> List[StateInstructions]:
    """
    Generate instructions for a turn given moves for both sides.

    :param py_state: The current game state
    :param side_one_move: Move for side one
    :param side_two_move: Move for side two
    :return: List of state instructions
    """
    ...

def calculate_damage(
    py_state: State,
    side_one_move: str,
    side_two_move: str,
    side_one_moves_first: bool,
) -> List[int]:
    """
    Calculate damage rolls for a move.

    :param py_state: The current game state
    :param side_one_move: The move used for side_one
    :param side_two_move: The move used for side_two
    :param side_one_moves_first: if side_one moves first
    :return: List of damage values
    """
    ...

# ---------------------------------------------------------------------------
# Doubles engine (gen9championsvgc2026regmc).
# Only present when built with `--no-default-features --features champions`.
# ---------------------------------------------------------------------------

class DoublesPokemon:
    """
    A team member for a doubles battle. Stats are computed by the engine
    from the species' base stats.

    :param species: Species name, e.g. "Incineroar" or "Furfrou-Debutante"
    :param moves: 1 to 4 move names
    :param ability: Ability name
    :param item: Item name ("" for no item)
    :param nature: Nature name
    :param evs: Champions stat points (hp, atk, def, spa, spd, spe)
    :param level: Level
    :param ivs: IVs (hp, atk, def, spa, spd, spe); the engine uses 31
    :param gender: "M", "F" or None
    :param name: Nickname
    """

    species: str
    moves: List[str]
    ability: str
    item: str
    nature: str
    evs: Tuple[int, int, int, int, int, int]
    level: int
    ivs: Optional[Tuple[int, int, int, int, int, int]]
    gender: Optional[str]
    name: Optional[str]

    def __init__(
        self,
        species: str,
        moves: List[str],
        ability: str = "",
        item: str = "",
        nature: str = "serious",
        evs: Tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
        level: int = 50,
        ivs: Optional[Tuple[int, int, int, int, int, int]] = None,
        gender: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None: ...

class DoublesAction:
    """
    One action for one active slot. Create with ``DoublesAction.move``,
    ``DoublesAction.switch`` or ``DoublesAction.pass_``.
    """

    kind: str
    slot: int
    target: int
    struggle: bool
    position: int

    @staticmethod
    def move(slot: int, target: int = 1, struggle: bool = False) -> DoublesAction:
        """
        Use the move in move slot ``slot`` (0-3). ``target`` is a
        pokemon-showdown location: 1/2 = foe slot, -1/-2 = own slot.
        It is ignored for moves without a target.
        """
        ...
    @staticmethod
    def switch(position: int) -> DoublesAction:
        """Switch in the team member at 0-based team ``position``."""
        ...
    @staticmethod
    def pass_() -> DoublesAction:
        """Do nothing (empty slot with no replacement available)."""
        ...

class DoublesForcedSwitch:
    """
    The answer to a switch request made during a turn: a pivot move
    (U-turn / Volt Switch) or a faint replacement.

    :param side: "a" or "b"
    :param slot: Active slot being replaced (0 or 1)
    :param position: Team position to switch in; None = pass
    :param pivot: True for a mid-turn pivot switch
    :param revive: True for a Revival Blessing answer
    """

    side: str
    slot: int
    position: Optional[int]
    pivot: bool
    revive: bool

    def __init__(
        self,
        side: str,
        slot: int,
        position: Optional[int] = None,
        pivot: bool = False,
        revive: bool = False,
    ) -> None: ...

class DoublesTurnChoices:
    """
    Both sides' choices for one turn.

    :param a: Side A's actions, one per active slot (left, right)
    :param b: Side B's actions, one per active slot (left, right)
    :param switches: Answers to switch requests made during the turn, in order
    """

    a: List[DoublesAction]
    b: List[DoublesAction]
    switches: List[DoublesForcedSwitch]

    def __init__(
        self,
        a: List[DoublesAction],
        b: List[DoublesAction],
        switches: List[DoublesForcedSwitch] = [],
    ) -> None: ...

class DoublesBoosts:
    atk: int
    def_: int  # `def` is a Python keyword
    spa: int
    spd: int
    spe: int
    acc: int
    eva: int

class DoublesMoveState:
    id: str
    pp: int
    disabled: bool

class DoublesPokemonState:
    """One team member's state."""

    position: int
    """Team position (0-5), the ``position`` used by switch actions."""
    species: str
    level: int
    hp: int
    maxhp: int
    fainted: bool
    status: str
    active_slot: Optional[int]
    boosts: DoublesBoosts
    types: Tuple[str, str]
    ability: str
    item: str
    stats: Tuple[int, int, int, int, int]
    """Unboosted stats (atk, def, spa, spd, spe)."""
    moves: List[DoublesMoveState]

class DoublesSideCondition:
    id: str
    duration: Optional[int]

class DoublesTurnState:
    """The full battle state after one turn."""

    turn: int
    a: List[DoublesPokemonState]
    b: List[DoublesPokemonState]
    weather: str
    terrain: str
    trick_room: bool
    pseudo_weather: List[str]
    a_side_conditions: List[DoublesSideCondition]
    b_side_conditions: List[DoublesSideCondition]
    move_order: List[str]

class DoublesChoicesResult:
    """Result of ``run_doubles_battle_with_choices``."""

    winner: str
    total_turns: int
    turns: List[DoublesTurnState]
    error: Optional[str]
    prng_draws: List[int]
    prng_draw_ctx: List[str]

class DoublesMoveOption:
    id: str
    """Move id ("recharge" and "struggle" included)."""
    move_slot: int
    """The move slot to pass to ``DoublesAction.move``."""
    struggle: bool
    pp: int
    disabled: bool
    targets: List[int]
    """Valid target locations (1/2 = foe slot, -1/-2 = own slot); ``[0]``
    when the player does not choose the target."""

class DoublesSlotOptions:
    """Legal choices for one active slot, as PS's move request gives them."""

    moves: List[DoublesMoveOption]
    """Every move offered (the locked move, Struggle, or all known moves);
    a ``disabled`` one can not be chosen."""
    locked: bool
    trapped: bool
    switches: List[int]
    """Team positions this slot may switch to."""

    def actions(self) -> List[DoublesAction]:
        """Every legal action: each usable move with each target, then each switch."""
        ...

class DoublesSwitchRequest:
    """A switch the battle needs before the turn can go on."""

    side: str
    slot: int
    kind: str
    """"faint" (end-of-turn replacement), "pivot" (U-turn, Volt Switch,
    Baton Pass, Emergency Exit, ...) or "revive" (Revival Blessing: pick a
    fainted team member)."""
    options: List[int]
    """Team positions that are valid answers."""
    can_pass: bool
    """True when ``None`` (no switch) is a valid answer."""

class DoublesBattle:
    """
    A doubles battle played one decision at a time.

    Each turn, ``choose`` takes both sides' actions. If the turn needs a switch
    decision (a pivot move, a faint replacement, ...), the battle pauses:
    ``requests`` lists what is needed and ``switch`` answers them one at a
    time. The turn is then re-simulated from its start with the answers, so the
    result is the same as giving all the answers up front.

    :param team_a: Side A's team (2 to 6 pokemon)
    :param team_b: Side B's team (2 to 6 pokemon)
    :param seed: pokemon-showdown Gen5 PRNG seed
    :param leads_a: Side A's two lead team positions
    :param leads_b: Side B's two lead team positions
    :param trace: Record every PRNG draw in ``prng_draws``
    """

    state: DoublesTurnState
    """The battle after the last turn, or at the pending switch request."""
    requests: List[DoublesSwitchRequest]
    """Requests to answer with ``switch``; empty unless ``waiting_for == "switch"``."""
    waiting_for: str
    """"turn" (call ``choose``), "switch" (call ``switch``) or "over"."""
    is_over: bool
    winner: Optional[str]
    """"A" or "B" once the battle is over, else None."""
    turn: int
    """The number of the next turn to play (1 before the first turn)."""
    history: List[DoublesTurnChoices]
    """Every finished turn's choices, switch answers included."""
    prng_draws: List[int]

    def __init__(
        self,
        team_a: List[DoublesPokemon],
        team_b: List[DoublesPokemon],
        seed: Tuple[int, int, int, int],
        leads_a: List[int] = [0, 1],
        leads_b: List[int] = [0, 1],
        trace: bool = False,
    ) -> None: ...
    def choose(
        self, a: List[DoublesAction], b: List[DoublesAction], check: bool = True
    ) -> DoublesTurnState:
        """
        Play a turn: one action per active slot for each side (left, right).
        Returns the new state. If the turn needs a switch decision, the state
        is the battle at that point and ``requests`` is set. With ``check``,
        an action ``legal_options`` does not allow raises ValueError.

        :raises ValueError: if the battle is over, a switch is pending, or the
            actions are invalid
        """
        ...
    def legal_options(self, side: str) -> List[Optional[DoublesSlotOptions]]:
        """Legal choices for ``side``'s two active slots (None = empty slot, which passes)."""
        ...
    def legal_actions(self, side: str) -> List[List[DoublesAction]]:
        """Per active slot of ``side``: every legal action for the next turn."""
        ...
    def legal_joint_actions(self, side: str) -> List[Tuple[DoublesAction, DoublesAction]]:
        """Every legal (left, right) action pair (no two switches to one pokemon)."""
        ...
    def switch(
        self, side: str, slot: int, position: Optional[int] = None
    ) -> DoublesTurnState:
        """
        Answer one of ``requests``: switch team ``position`` into ``slot`` of
        ``side`` (None = no switch, only when ``can_pass``). Once all requests
        are answered the turn goes on. Returns the new state.

        :raises ValueError: if there is no such request or the answer is invalid
        """
        ...
    def copy(self) -> DoublesBattle:
        """An independent copy of the battle (for search / rollouts)."""
        ...
    def __copy__(self) -> DoublesBattle: ...
    def __deepcopy__(self, memo: dict) -> DoublesBattle: ...

class DoublesSlotHp:
    name: str
    hp: int
    maxhp: int
    fainted: bool

class DoublesMoveResult:
    attacker: str
    move_name: str
    target: str
    damage: int
    crit: bool
    missed: bool

class DoublesSwitchResult:
    side: str
    slot: int
    in_pokemon: str

class DoublesTurn:
    turn: int
    order: List[str]
    moves: List[DoublesMoveResult]
    switches: List[DoublesSwitchResult]
    hp_a: List[Optional[DoublesSlotHp]]
    hp_b: List[Optional[DoublesSlotHp]]

class DoublesBattleResult:
    """Result of ``run_doubles_battle``."""

    winner: str
    total_turns: int
    turns: List[DoublesTurn]
    prng_draws: List[int]

def run_doubles_battle(
    team_a: List[DoublesPokemon],
    team_b: List[DoublesPokemon],
    seed: Tuple[int, int, int, int],
    leads_a: List[int] = [0, 1],
    leads_b: List[int] = [0, 1],
) -> DoublesBattleResult:
    """
    Run a full doubles battle where both sides use the engine's built-in
    move policy.

    :param team_a: Side A's team (2 to 6 pokemon)
    :param team_b: Side B's team (2 to 6 pokemon)
    :param seed: pokemon-showdown Gen5 PRNG seed
    :param leads_a: Side A's two lead team positions
    :param leads_b: Side B's two lead team positions
    :raises RuntimeError: if the engine fails (e.g. unknown species or move)
    """
    ...

def run_doubles_battle_with_choices(
    team_a: List[DoublesPokemon],
    team_b: List[DoublesPokemon],
    seed: Tuple[int, int, int, int],
    choices: List[DoublesTurnChoices],
    leads_a: List[int] = [0, 1],
    leads_b: List[int] = [0, 1],
    stop_after: Optional[int] = None,
    trace: bool = False,
) -> DoublesChoicesResult:
    """
    Run a doubles battle driven by explicit per-turn choices. Stops when the
    battle ends, the choices run out (``error`` is set), or after
    ``stop_after`` turns.

    :param trace: Record what each PRNG draw was used for in ``prng_draw_ctx``
    :raises RuntimeError: if the engine fails (e.g. unknown species or move)
    """
    ...

def run_doubles_battle_json(input_json: str) -> str:
    """
    Run a battle from the JSON input of the ``doubles-battle`` /
    ``doubles-fuzz-battle`` CLI subcommands and return the CLI's JSON output.
    """
    ...
