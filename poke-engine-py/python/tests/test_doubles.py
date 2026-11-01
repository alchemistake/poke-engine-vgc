import json

import pytest

import poke_engine

if not hasattr(poke_engine, "run_doubles_battle"):
    pytest.skip(
        "poke_engine was built without the `champions` feature",
        allow_module_level=True,
    )

from poke_engine import (
    DoublesAction,
    DoublesBattle,
    DoublesForcedSwitch,
    DoublesPokemon,
    DoublesTurnChoices,
    run_doubles_battle,
    run_doubles_battle_json,
    run_doubles_battle_with_choices,
)

SEED = (1, 2, 3, 4)


def team(*species):
    return [DoublesPokemon(s, ["tackle", "protect"]) for s in species]


TEAM_A = team("Incineroar", "Rillaboom", "Pikachu", "Garchomp")
TEAM_B = team("Amoonguss", "Kingambit", "Dragonite", "Sneasler")


def attack_turn():
    return DoublesTurnChoices(
        a=[DoublesAction.move(0, 1), DoublesAction.move(0, 2)],
        b=[DoublesAction.move(0, 1), DoublesAction.move(0, 2)],
    )


def test_run_doubles_battle_finishes():
    result = run_doubles_battle(TEAM_A, TEAM_B, SEED)
    assert result.winner in ("A", "B")
    assert len(result.turns) > 0
    assert len(result.turns[0].hp_a) == 2


def test_run_doubles_battle_is_deterministic():
    r1 = run_doubles_battle(TEAM_A, TEAM_B, SEED)
    r2 = run_doubles_battle(TEAM_A, TEAM_B, SEED)
    assert r1.winner == r2.winner
    assert r1.prng_draws == r2.prng_draws


def test_run_doubles_battle_with_choices_snapshots():
    result = run_doubles_battle_with_choices(
        TEAM_A, TEAM_B, SEED, [attack_turn()], stop_after=1
    )
    assert result.error is None
    last = result.turns[-1]
    assert [p.species for p in last.a] == [
        "Incineroar",
        "Rillaboom",
        "Pikachu",
        "Garchomp",
    ]
    assert [p.active_slot for p in last.a] == [0, 1, None, None]
    assert len(last.move_order) == 4
    # every active pokemon used tackle and took damage from the other side
    for p in last.a[:2] + last.b[:2]:
        assert p.hp < p.maxhp


def test_protect_blocks_damage():
    turn = DoublesTurnChoices(
        a=[DoublesAction.move(1), DoublesAction.move(1)],
        b=[DoublesAction.move(0, 1), DoublesAction.move(0, 2)],
    )
    result = run_doubles_battle_with_choices(
        TEAM_A, TEAM_B, SEED, [turn], stop_after=1
    )
    for p in result.turns[-1].a[:2]:
        assert p.hp == p.maxhp


def test_switch_action():
    turn = DoublesTurnChoices(
        a=[DoublesAction.switch(2), DoublesAction.move(1)],
        b=[DoublesAction.move(1), DoublesAction.move(1)],
    )
    result = run_doubles_battle_with_choices(
        TEAM_A, TEAM_B, SEED, [turn], stop_after=1
    )
    slots = {p.species: p.active_slot for p in result.turns[-1].a}
    assert slots["Pikachu"] == 0
    assert slots["Incineroar"] is None


def test_choices_exhausted_sets_error():
    result = run_doubles_battle_with_choices(TEAM_A, TEAM_B, SEED, [attack_turn()])
    assert result.error is not None


def test_trace_records_draw_context():
    result = run_doubles_battle_with_choices(
        TEAM_A, TEAM_B, SEED, [attack_turn()], stop_after=1, trace=True
    )
    assert len(result.prng_draw_ctx) == len(result.prng_draws) > 0


def test_unknown_species_raises():
    with pytest.raises(RuntimeError, match="NotAPokemon"):
        run_doubles_battle(team("NotAPokemon", "Pikachu"), TEAM_B, SEED)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        run_doubles_battle(TEAM_A, TEAM_B, SEED, leads_a=[0, 0])
    with pytest.raises(ValueError):
        DoublesTurnChoices(a=[DoublesAction.move(0)], b=[])
    with pytest.raises(ValueError):
        DoublesAction.move(4)
    with pytest.raises(ValueError):
        DoublesForcedSwitch("c", 0)
    with pytest.raises(ValueError):
        DoublesPokemon("Pikachu", [])


def test_run_doubles_battle_json_matches_typed_api():
    def member(species):
        return {
            "species": species,
            "level": 50,
            "evs": {"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
            "nature": "serious",
            "ability": "",
            "item": "",
            "moves": ["tackle", "protect"],
        }

    battle = {
        "seed": list(SEED),
        "teams": {
            "a": [member(p.species) for p in TEAM_A],
            "b": [member(p.species) for p in TEAM_B],
        },
    }
    out = json.loads(run_doubles_battle_json(json.dumps(battle)))
    typed = run_doubles_battle(TEAM_A, TEAM_B, SEED)
    assert out["winner"] == typed.winner
    assert out["prng_draws"] == typed.prng_draws

    with pytest.raises(ValueError):
        run_doubles_battle_json("{}")


PIVOT_A = [
    DoublesPokemon("Incineroar", ["uturn", "protect"]),
    DoublesPokemon("Rillaboom", ["tackle", "protect"]),
    DoublesPokemon("Pikachu", ["tackle", "protect"]),
    DoublesPokemon("Garchomp", ["tackle", "protect"]),
]


def play_out(battle, rng):
    """Play random legal-looking choices to the end; return the requests seen."""
    seen = []
    while battle.waiting_for != "over" and battle.turn < 60:
        if battle.waiting_for == "switch":
            req = battle.requests[0]
            seen.append(req.kind)
            options = req.options + ([None] if req.can_pass else [])
            battle.switch(req.side, req.slot, rng.choice(options))
            continue

        battle.choose(
            *[list(rng.choice(battle.legal_joint_actions(side))) for side in "ab"]
        )
    return seen


def test_battle_starts_waiting_for_a_turn():
    battle = DoublesBattle(TEAM_A, TEAM_B, SEED)
    assert battle.waiting_for == "turn"
    assert battle.turn == 1
    assert battle.requests == []
    state = battle.state
    assert state.turn == 0
    assert [p.active_slot for p in state.a] == [0, 1, None, None]
    assert [m.id for m in state.a[0].moves] == ["tackle", "protect"]
    assert state.a[2].position == 2


def test_battle_choose_matches_batch_run():
    battle = DoublesBattle(TEAM_A, TEAM_B, SEED)
    state = battle.choose(*[[DoublesAction.move(0, 1), DoublesAction.move(0, 2)]] * 2)
    batch = run_doubles_battle_with_choices(
        TEAM_A, TEAM_B, SEED, [attack_turn()], stop_after=1
    )
    expected = batch.turns[-1]
    assert state.turn == 1 and battle.turn == 2
    assert [p.hp for p in state.a + state.b] == [p.hp for p in expected.a + expected.b]
    assert state.move_order == expected.move_order


def test_pivot_pauses_for_switch():
    battle = DoublesBattle(PIVOT_A, TEAM_B, SEED)
    state = battle.choose(
        [DoublesAction.move(0, 1), DoublesAction.move(1)],
        [DoublesAction.move(0, 1), DoublesAction.move(0, 1)],
    )
    assert battle.waiting_for == "switch"
    (req,) = battle.requests
    assert (req.side, req.slot, req.kind) == ("a", 0, "pivot")
    assert sorted(req.options) == [2, 3] and not req.can_pass
    # the state is the battle at the request: Incineroar still on the field
    assert state.a[0].active_slot == 0

    with pytest.raises(ValueError):
        battle.choose([DoublesAction.move(0)] * 2, [DoublesAction.move(0)] * 2)
    with pytest.raises(ValueError):
        battle.switch("a", 0, 1)  # already active
    with pytest.raises(ValueError):
        battle.switch("a", 0)  # pass not allowed

    state = battle.switch("a", 0, 3)
    assert battle.waiting_for == "turn"
    slots = {p.species: p.active_slot for p in state.a}
    assert slots["Garchomp"] == 0 and slots["Incineroar"] is None
    assert battle.history[0].switches[0].pivot


def test_battle_copy_is_independent():
    battle = DoublesBattle(TEAM_A, TEAM_B, SEED)
    other = battle.copy()
    other.choose([DoublesAction.move(0, 1)] * 2, [DoublesAction.move(0, 1)] * 2)
    assert battle.turn == 1 and other.turn == 2


@pytest.mark.parametrize("seed", range(20))
def test_stepwise_play_matches_replay(seed):
    import random

    battle = DoublesBattle(PIVOT_A, TEAM_B, (seed, 2, 3, 4), trace=True)
    seen = play_out(battle, random.Random(seed))
    assert battle.is_over
    assert "faint" in seen

    replay = run_doubles_battle_with_choices(
        PIVOT_A, TEAM_B, (seed, 2, 3, 4), battle.history, trace=True
    )
    assert replay.error is None
    assert replay.winner == battle.winner
    assert replay.prng_draws == battle.prng_draws

    def key(s):
        return [(p.species, p.hp, p.status, p.active_slot) for p in s.a + s.b]

    assert key(replay.turns[-1]) == key(battle.state)


def test_legal_options():
    battle = DoublesBattle(PIVOT_A, TEAM_B, SEED)
    (left, right) = battle.legal_options("a")
    assert [m.id for m in left.moves] == ["uturn", "protect"]
    assert not left.trapped and not left.locked
    assert sorted(left.switches) == [2, 3]
    uturn = left.moves[0]
    assert sorted(uturn.targets) == [-2, 1, 2]  # foes and the ally
    assert left.moves[1].targets == [0]  # Protect: no target choice
    assert len(left.actions()) == 3 + 1 + 2

    per_slot = battle.legal_actions("a")
    assert [len(x) for x in per_slot] == [6, 6]
    joint = battle.legal_joint_actions("a")
    # both slots may not switch in the same pokemon
    assert len(joint) == 6 * 6 - 2


def test_choose_rejects_illegal_actions():
    battle = DoublesBattle(TEAM_A, TEAM_B, SEED)
    ok = [DoublesAction.move(0, 1), DoublesAction.move(0, 1)]
    with pytest.raises(ValueError, match="illegal"):
        battle.choose([DoublesAction.move(3, 1), DoublesAction.move(0, 1)], ok)
    with pytest.raises(ValueError, match="illegal"):
        battle.choose([DoublesAction.switch(1), DoublesAction.move(0, 1)], ok)
    with pytest.raises(ValueError, match="same|both"):
        battle.choose([DoublesAction.switch(2), DoublesAction.switch(2)], ok)
    assert battle.turn == 1
    battle.choose([DoublesAction.switch(2), DoublesAction.switch(3)], ok)
    assert battle.turn == 2


def test_fake_out_only_on_first_turn():
    team = [DoublesPokemon("Incineroar", ["fakeout", "tackle"])] + TEAM_A[1:]
    battle = DoublesBattle(team, TEAM_B, SEED)
    fake_out = lambda: battle.legal_options("a")[0].moves[0]
    assert fake_out().id == "fakeout" and not fake_out().disabled
    battle.choose([DoublesAction.move(0, 1), DoublesAction.move(0, 1)], [DoublesAction.move(0, 1)] * 2)
    assert fake_out().disabled


def test_events_off_by_default():
    battle = DoublesBattle(TEAM_A, TEAM_B, SEED)
    battle.choose([DoublesAction.move(0, 1)] * 2, [DoublesAction.move(0, 1)] * 2)
    assert battle.events == []


@pytest.mark.parametrize("seed", range(10))
def test_events_match_hp_and_faints(seed):
    import random

    battle = DoublesBattle(PIVOT_A, TEAM_B, (seed, 2, 3, 4), events=True)
    plain = DoublesBattle(PIVOT_A, TEAM_B, (seed, 2, 3, 4))
    rng = random.Random(seed)
    before = battle.state
    while battle.waiting_for != "over" and battle.turn < 60:
        if battle.waiting_for == "switch":
            req = battle.requests[0]
            options = req.options + ([None] if req.can_pass else [])
            pick = rng.choice(options)
            battle.switch(req.side, req.slot, pick)
            plain.switch(req.side, req.slot, pick)
        else:
            acts = [list(rng.choice(battle.legal_joint_actions(side))) for side in "ab"]
            battle.choose(*acts)
            plain.choose(*acts)
        if battle.waiting_for == "switch":
            continue
        after = battle.state
        events = battle.events
        assert all(e["kind"] != "move" or e["move"] for e in events)
        for side in "ab":
            for i, (p0, p1) in enumerate(zip(getattr(before, side), getattr(after, side))):
                mine = [e for e in events if e["target_side"] == side and e["target"] == i]
                dmg = sum(e["value"] for e in mine if e["kind"] == "damage")
                heal = sum(e["value"] for e in mine if e["kind"] == "heal")
                assert dmg - heal == p0.hp - p1.hp, (battle.turn, side, i, events)
                faints = [e for e in events if e["kind"] == "faint" and e["side"] == side and e["actor"] == i]
                assert len(faints) == (1 if p0.hp > 0 and p1.hp == 0 else 0)
        before = after
    # Recording does not change the battle.
    assert battle.winner == plain.winner
    assert [p.hp for p in battle.state.a + battle.state.b] == [p.hp for p in plain.state.a + plain.state.b]
