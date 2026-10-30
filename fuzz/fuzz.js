// Fuzz tester: runs poke-engine and pokemon-showdown on the same random
// battles and compares full per-turn state.
//
//   node fuzz.js --seed <testerSeed> --count <numTests> [--workdir dir] [--engine path]
//
// - The tester seed drives the controller PRNG (team generation, battle seed
//   selection, and all in-game move/switch/team-preview choices). Re-running
//   with the same seed reproduces the exact same test cases.
// - The battle seed (fed to both simulators) is separate from the controller
//   randomness, exactly like a real battle.
// - Exits immediately on the first mismatch, printing the mismatch details
//   and the state from both simulators. Work files (input.json / ps.json /
//   pe.json / report.txt) are kept in the workdir for the failing test.

'use strict';

const path = require('path');
const fs = require('fs');
const { spawnSync } = require('child_process');

// Parallel sweep parent: start the workers BEFORE loading pokemon-showdown
// (the parent only needs it to print a report). See runParallel / main.
const PARALLEL_PARENT = (() => {
  const o = parseArgs();
  return o.jobs > 1 && o.count > 1 && !o.checkOnly && !o.verbose ? o : null;
})();
if (PARALLEL_PARENT) runParallel(PARALLEL_PARENT);

const PS_DIR = path.resolve(__dirname, '..', '..', 'pokemon-showdown');
const { genTeam, effectiveStats } = require('./teams.js');

const FORMAT = 'gen9championsvgc2026regmc';

// pokemon-showdown, loaded on first use: a sweep whose PS battles are all
// cached never loads it. MOD_DEX is the format's own dex (applies
// data/mods/champions/*): its species table defines which species are legal.
let PS_SIM = null;
function ps() {
  if (!PS_SIM) {
    const Sim = require(path.join(PS_DIR, 'dist', 'sim'));
    const { PRNG } = require(path.join(PS_DIR, 'dist', 'sim', 'prng.js'));
    PS_SIM = { Dex: Sim.Dex, Battle: Sim.Battle, PRNG, MOD_DEX: Sim.Dex.forFormat(FORMAT) };
  }
  return PS_SIM;
}

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------

function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {
    seed: null,
    count: null,
    workdir: path.join(__dirname, 'work'),
    engine: path.resolve(__dirname, '..', 'target', 'release', 'poke-engine'),
    verbose: false,
    // Internal (parallel workers): test seeds are seed, seed+step, ...; on
    // the first failure print `FAIL <seed>` and exit 1 without a report.
    step: 1,
    checkOnly: false,
    jobs: Math.max(1, require('os').cpus().length),
    // Reuse cached PS battles on the fast path (see psCacheDir).
    cache: true,
  };
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--seed') opts.seed = parseInt(args[++i], 10);
    else if (args[i] === '--count') opts.count = parseInt(args[++i], 10);
    else if (args[i] === '--workdir') opts.workdir = path.resolve(args[++i]);
    else if (args[i] === '--engine') opts.engine = path.resolve(args[++i]);
    else if (args[i] === '--verbose') opts.verbose = true;
    else if (args[i] === '--jobs') opts.jobs = Math.max(1, parseInt(args[++i], 10));
    else if (args[i] === '--step') opts.step = parseInt(args[++i], 10);
    else if (args[i] === '--check-only') opts.checkOnly = true;
    else if (args[i] === '--no-cache') opts.cache = false;
    else {
      console.error(`unknown argument: ${args[i]}`);
      process.exit(2);
    }
  }
  if (opts.seed === null || opts.count === null || Number.isNaN(opts.seed) || Number.isNaN(opts.count)) {
    console.error('usage: node fuzz.js --seed <n> --count <n> [--workdir dir] [--engine path] [--verbose]');
    process.exit(2);
  }
  return opts;
}

// ---------------------------------------------------------------------------
// Normalization (so both simulators' states use the same vocabulary)
// ---------------------------------------------------------------------------

const STATUS_MAP = {
  '': 'none',
  par: 'paralyze',
  brn: 'burn',
  psn: 'poison',
  tox: 'toxic',
  slp: 'sleep',
  frz: 'freeze',
  // PS marks a fainted pokemon left in its slot (no replacement) 'fnt';
  // fainting itself is compared through `fainted`.
  fnt: 'none',
};

const WEATHER_MAP = {
  sandstorm: 'sand',
  primariesun: 'harshsun',
  primaryrain: 'heavyrain',
};

const TERRAIN_MAP = {
  electric: 'electricterrain',
  psyxic: 'psychicterrain',
  misty: 'mistyterrain',
  grassy: 'grassyterrain',
};

function squash(s) {
  return String(s).toLowerCase().replace(/[^a-z0-9]/g, '');
}

// "side:Species:move" -> squashed, forme suffix removed from species
function squashOrderEntry(s) {
  const parts = String(s).replace(/^([AB])\d+/, '$1').split(':');
  if (parts.length === 3) parts[1] = parts[1].split('-')[0];
  return squash(parts.join(':'));
}

function normalizePSState(snap) {
  const normSide = (side) => ({
    pokemon: side.pokemon.map((p) => ({
      species: squash(p.species.split('-')[0]),
      position: p.position,
      hp: p.hp,
      maxhp: p.maxhp,
      status: STATUS_MAP[p.status] || p.status,
      fainted: p.fainted,
      activeSlot: p.activeSlot < 0 ? null : p.activeSlot,
      boosts: { ...p.boosts },
    })),
    sideConditions: side.sideConditions.map((c) => ({ id: c.id, duration: c.duration })),
  });
  return {
    turn: snap.turn,
    a: normSide(snap.a),
    b: normSide(snap.b),
    weather: (WEATHER_MAP[snap.weather] || snap.weather).toLowerCase(),
    terrain: (TERRAIN_MAP[snap.terrain] || snap.terrain).toLowerCase(),
    trickRoom: snap.trickRoom,
    pseudoWeather: snap.pseudoWeather.map(squash).sort(),
    moveOrder: snap.moveOrder.map(squashOrderEntry),
    moveOrderRaw: snap.moveOrder,
  };
}

function normalizePETurn(turn) {
  const normSide = (side, sc) => ({
    pokemon: side.map((p, i) => ({
      species: squash(p.species.split('-')[0]),
      position: i,
      hp: p.hp,
      maxhp: p.maxhp,
      status: p.status,
      fainted: p.fainted,
      activeSlot: p.active_slot === null ? null : p.active_slot,
      boosts: {
        atk: p.boosts.atk,
        def: p.boosts.def,
        spa: p.boosts.spa,
        spd: p.boosts.spd,
        spe: p.boosts.spe,
        acc: p.boosts.acc,
        eva: p.boosts.eva,
      },
    })),
    sideConditions: sc.map((c) => ({ id: c.id, duration: c.duration })),
  });
  return {
    turn: turn.turn,
    a: normSide(turn.a, turn.a_side_conditions),
    b: normSide(turn.b, turn.b_side_conditions),
    weather: (WEATHER_MAP[turn.weather] || turn.weather).toLowerCase(),
    terrain: (TERRAIN_MAP[turn.terrain] || turn.terrain).toLowerCase(),
    trickRoom: turn.trick_room,
    pseudoWeather: (turn.pseudo_weather || []).map(squash).sort(),
    moveOrder: (turn.move_order || []).map(squashOrderEntry),
    moveOrderRaw: turn.move_order || [],
  };
}

// ---------------------------------------------------------------------------
// Deep diff
// ---------------------------------------------------------------------------

function deepDiff(ps, pe, p, out) {
  if (ps === pe) return;
  if (
    typeof ps !== typeof pe ||
    ps === null ||
    pe === null ||
    typeof ps !== 'object' ||
    Array.isArray(ps) !== Array.isArray(pe)
  ) {
    out.push({ path: p, ps, pe });
    return;
  }
  if (Array.isArray(ps)) {
    if (ps.length !== pe.length) {
      out.push({ path: `${p}.length`, ps: ps.length, pe: pe.length });
    }
    const n = Math.max(ps.length, pe.length);
    for (let i = 0; i < n; i++) deepDiff(ps[i], pe[i], `${p}[${i}]`, out);
    return;
  }
  const keys = new Set([...Object.keys(ps), ...Object.keys(pe)]);
  for (const k of keys) deepDiff(ps[k], pe[k], p ? `${p}.${k}` : k, out);
}

// ---------------------------------------------------------------------------
// PS battle driving
// ---------------------------------------------------------------------------

// Snapshot the battle state. `orderA`/`orderB` map each pokemon object to its
// ORIGINAL battle-team index: PS reorders side.pokemon when pokemon switch in
// (team array follows active order), so we must emit a stable order to compare
// against poke-engine's fixed team array.
function makeSnapshot(battle, logIdx, orderA, orderB) {
  const sideSnap = (side, order) => {
    const mons = side.pokemon
      .slice()
      .sort((x, y) => order.get(x) - order.get(y));
    return {
      pokemon: mons.map((p, pos) => ({
        species: p.species.name,
        position: pos,
      hp: p.hp,
      maxhp: p.maxhp,
      status: p.status || '',
      fainted: !!p.fainted,
      // A fainted pokemon with no replacement stays in PS's side.active,
      // but its slot is empty (the engine models it that way).
      activeSlot: p.fainted ? -1 : side.active.indexOf(p),
      boosts: {
        atk: p.boosts.atk || 0,
        def: p.boosts.def || 0,
        spa: p.boosts.spa || 0,
        spd: p.boosts.spd || 0,
        spe: p.boosts.spe || 0,
        acc: p.boosts.accuracy || 0,
        eva: p.boosts.evasion || 0,
      },
      })),
      sideConditions: Object.keys(side.sideConditions)
        .map((id) => ({ id, duration: side.sideConditions[id].duration }))
        .sort((x, y) => x.id.localeCompare(y.id)),
    };
  };
  const moveOrder = [];
  for (let i = logIdx; i < battle.log.length; i++) {
    let L = battle.log[i];
    if (typeof L === 'string') L = L.split('|');
    if (L[1] === 'move') {
      const src = L[2]; // "p1a: Pikachu"
      const sideTag = src.startsWith('p1') ? 'A' : 'B';
      const name = src.slice(src.indexOf(':') + 1).trim();
      moveOrder.push(`${sideTag}:${name}:${L[3]}`);
    }
  }
  return {
    turn: battle.turn,
    a: sideSnap(battle.p1, orderA),
    b: sideSnap(battle.p2, orderB),
    weather: battle.field.weather || 'none',
    terrain: battle.field.terrain || 'none',
    trickRoom: !!battle.field.pseudoWeather.trickroom,
    pseudoWeather: Object.keys(battle.field.pseudoWeather),
    moveOrder,
  };
}

// ---------------------------------------------------------------------------
// PRNG draw tracing (diagnostics)
// ---------------------------------------------------------------------------
// Both simulators consume ONE shared PRNG stream, so the n-th draw has the
// same value on both sides; a bug shows up as a draw used for a different
// PURPOSE (a missing or extra draw shifts everything after it). Each side
// tags every draw with a purpose from the same vocabulary: PS from the call
// stack, the engine via `Gen5Rng::label` (`prng_draw_ctx` in pe.json).

let PS_TRACE = null; // { battle, draws: [{ turn, cat, ctx, stack }] }
// Draw tracing captures a call stack per draw (slow); only reports need it.
let TRACE_DRAWS = true;

const { psDrawPurpose } = require('./fuzz_purpose.js');

function tracePSDraws(battle) {
  const trace = { battle, draws: [] };
  const rng = battle.prng.rng;
  const next = rng.next.bind(rng);
  rng.next = () => {
    const v = next();
    const fns = (new Error().stack || '')
      .split('\n')
      .slice(2)
      .map((l) => (l.match(/at (?:[\w$]+\.)?([\w$<>]+) \(/) || [])[1])
      .filter(Boolean);
    let ctx = '';
    const p = battle.activePokemon;
    if (battle.activeMove && p) {
      ctx = `${p.side.id === 'p1' ? 'A' : 'B'}${p.position} ${p.name} ${battle.activeMove.name}`;
    } else if (fns.includes('fieldEvent') && !fns.includes('runSwitch')) {
      ctx = 'end of turn';
    } else if (fns.includes('runSwitch')) {
      ctx = 'switch-in';
    } else if (fns.includes('commitChoices')) {
      ctx = 'move order';
    }
    trace.draws.push({
      turn: battle.turn,
      cat: psDrawPurpose(fns),
      ctx: `turn ${battle.turn}: ${ctx}`,
      stack: fns.slice(0, 6).join(' < '),
    });
    return v;
  };
  return trace;
}

// The PS protocol log of one turn, reduced to the lines that explain what
// happened (moves, damage, misses, ability/item activations, ...).
function psTurnLog(battle, turn) {
  const keep = /^\|(move|-damage|-heal|-miss|-fail|-immune|-activate|-ability|-crit|-supereffective|-resisted|-status|-curestatus|-boost|-unboost|-setboost|-clearallboost|faint|switch|drag|cant|-weather|-sidestart|-sideend|-start|-end|-fieldstart|-fieldend|-singleturn|-singlemove|-hitcount|-prepare|-item|-enditem|-notarget|-nothing|-transform|-mustrecharge)\|/;
  const log = battle.log;
  let start = turn === 0 ? 0 : log.indexOf(`|turn|${turn}`);
  if (start < 0) return [];
  let end = log.indexOf(`|turn|${turn + 1}`, start + 1);
  if (end < 0) end = log.length;
  const out = [];
  for (let i = start; i < end; i++) {
    let l = log[i];
    if (l.startsWith('|split|')) {
      l = log[i + 1]; // exact-HP (secret) line; skip the percentage copy
      i += 2;
    }
    if (keep.test(l)) out.push(l);
  }
  return out;
}

// Where each kind of draw happens in PS and what mirrors it in the engine.
const DRAW_HINTS = {
  'speed tie (eachEvent)':
    "PS `eachEvent('Update')` (sim/battle.ts) speed-sorts ALL actives after every action (runAction) " +
    'and after each hit of a move (hitStepMoveHitLoop), shuffling speed ties. Engine: `sorted_actives` ' +
    '(src/doubles/battle.rs).',
  'speed tie (residual)':
    "End-of-turn `fieldEvent('Residual')` sorts every Residual handler (statuses, volatiles with a " +
    'duration, abilities, side conditions, weather) and shuffles ties. Engine: ' +
    '`collect_residual_handlers` is missing or has an extra handler.',
  'speed tie (move order)':
    'Turn-order `speedSort` (priority, then speed, ties shuffled). Engine: `speed_sort` / ' +
    '`effective_speed`; check priority and speed modifiers (boosts, paralysis, Trick Room, Tailwind, abilities).',
  'speed tie (switch-in)':
    '`runSwitch` speed-sorts all actives after a switch-in (sim/battle-actions.ts). Engine: ' +
    '`resort_all_active` (src/doubles/battle.rs).',
  'speed tie (queue insert)': 'A mid-turn action was inserted into the queue among speed ties (`insertChoice`).',
  accuracy:
    'Accuracy roll (`hitStepAccuracy`). One side stopped the move earlier: Protect, semi-invulnerable ' +
    'target, type / ability immunity, `onTry` failure, no target. Moves with `accuracy: true` never roll.',
  crit: 'Crit roll in `getDamage`: one side skipped damage calculation (immunity, fixed damage, miss).',
  'damage roll': 'Damage roll (`randomizer`): one side skipped or repeated damage calculation.',
  secondary:
    'Secondary roll (`secondaries`, one per entry in data/moves.ts). Engine: `choice.secondaries` in ' +
    'src/choices.rs, or the custom-secondary match in `execute_action`.',
  'self boost': 'PS `selfDrops` roll for `self.boosts`. Engine: `self_boosts` in src/choices.rs.',
  multihit: 'Hit-count sample for 2-5 hit moves. Engine: `multihit` in src/choices.rs.',
  multiaccuracy:
    'Accuracy roll for hits 2+ of a `multiaccuracy` move (Triple Axel, Triple Kick, Population Bomb) in ' +
    '`hitStepMoveHitLoop`. Engine: the `multiaccuracy` roll in the hit loop of `execute_action`.',
  retarget: 'The chosen target fainted, so PS picked a random foe (`getRandomTarget`).',
  'status before move':
    'Status `onBeforeMove` roll: full paralysis (champions 1/8) or freeze thaw (1/4), see ' +
    'data/mods/champions/conditions.ts. Engine: the status block at the start of `execute_action`.',
  'sleep duration': 'Sleep duration `sample([2, 3, 3])` when sleep is set (champions `slp.onStart`). Engine: `set_status`.',
  'volatile duration': 'A volatile drew its duration when added (`addVolatile` -> `onStart`, e.g. confusion `random(2, 6)`). Engine: `try_confuse`.',
  'fractional priority': 'Quick Draw `onFractionalPriority` 30% roll when the queue is built (`resolveAction`). Engine: the resolve loop in `run_battle_fuzz` + `mechanics::fractional_priority`.',
  'drag in': 'Roar / Whirlwind / Dragon Tail / Circle Throw pick a random replacement (`dragIn` -> `getRandomSwitchable`, PS `side.pokemon` order). Engine: `drag_in_all`, `possible_switches`.',
  trace: 'Trace `onUpdate` picks a random valid adjacent foe (`sample`). Engine: the Trace branch of `apply_on_switch_abilities`.',
  stall: 'Protect-family success roll after consecutive use (`stall.onStallMove`, `randomChance(1, counter)`). Engine: the stall check in `execute_action`.',
  'residual roll': 'A roll inside an end-of-turn `onResidual` handler (Moody `sample`, Shed Skin / Hydration / Healer chances). Engine: `ResidualKind` handlers in `end_of_turn`.',
  ability: 'An ability `onDamagingHit` roll (Static, Flame Body, Effect Spore, ...) in data/abilities.ts.',
  sample: 'A `sample()` inside a move effect (Tri Attack / Dire Claw status pick, ...).',
  'fickle beam': 'Fickle Beam `onBasePower` 30% roll.',
};

// Compare draw purposes; returns report lines (with a diagnosis).
function prngReport(pe, failTurn) {
  const out = [];
  if (!PS_TRACE || !pe || !pe.prng_draw_ctx) return out;
  const ps = PS_TRACE.draws.filter((d) => d.turn <= Math.max(failTurn, 0) || failTurn === 0);
  const peD = pe.prng_draw_ctx.map((c) => {
    const k = c.indexOf('|');
    return { cat: c.slice(0, k), ctx: c.slice(k + 1) };
  });
  const n = Math.max(ps.length, peD.length);
  let m = -1;
  for (let i = 0; i < n; i++) {
    if (!ps[i] || !peD[i] || ps[i].cat !== peD[i].cat) {
      m = i;
      break;
    }
  }
  out.push(`PRNG draws through turn ${failTurn}: PS=${ps.length} PE=${peD.length}`);
  if (m < 0) {
    out.push(
      'DIAGNOSIS: PRNG draws line up (same count, same purpose, same order). The mismatch is a ' +
        'mechanic / formula difference, not RNG: compare the PS log below with the engine state ' +
        '(damage modifiers, abilities, items, status, move effects).'
    );
    return out;
  }
  const fmt = (d) => (d ? `${d.cat.padEnd(24)} ${d.ctx}` : '(no draw)');
  const a = ps[m];
  const b = peD[m];
  let guess = '';
  if (a && b && ps[m + 1] && ps[m + 1].cat === b.cat) {
    guess = `the engine is MISSING PS's "${a.cat}" draw (${a.ctx})`;
  } else if (a && b && peD[m + 1] && peD[m + 1].cat === a.cat) {
    guess = `the engine makes an EXTRA "${b.cat}" draw (${b.ctx})`;
  } else if (!b) {
    guess = `the engine is MISSING PS's "${a.cat}" draw (${a.ctx}) and everything after it`;
  } else if (!a) {
    guess = `the engine makes EXTRA draws starting with "${b.cat}" (${b.ctx})`;
  } else {
    guess = `PS made a "${a.cat}" draw (${a.ctx}) where the engine made a "${b.cat}" draw (${b.ctx})`;
  }
  out.push(
    `DIAGNOSIS: PRNG misaligned at draw #${m}: ${guess}. Every later roll is shifted, so the ` +
      'state diff below may be a side effect. Fix this draw first.'
  );
  if (a) out.push(`  PS call stack for draw #${m}: ${a.stack}`);
  for (const cat of new Set([a && a.cat, b && b.cat].filter(Boolean))) {
    if (DRAW_HINTS[cat]) out.push(`  hint (${cat}): ${DRAW_HINTS[cat]}`);
  }
  out.push('  draw  PS purpose / context                                    | PE purpose / context');
  for (let i = Math.max(0, m - 4); i < Math.min(n, m + 5); i++) {
    out.push(`  ${i === m ? '>' : ' '}#${String(i).padEnd(4)}${fmt(ps[i]).padEnd(58)} | ${fmt(peD[i])}`);
  }
  return out;
}

// Drive the PS battle turn by turn. `onInit(snap, meta)` is invoked right
// after team preview (before any move) with the initial state; `onTurn(snap,
// turns, snapIdx, meta)` at every turn boundary. The fuzzer runs poke-engine
// on the same state/turns so far and exits immediately on the first divergence.
function stepPSBattle(battleSeed, teamA, teamB, controller, onInit, onTurn) {
  const battle = new (ps().Battle)({
    format: ps().Dex.formats.get(FORMAT),
    seed: battleSeed,
    p1: { name: 'A', team: teamA },
    p2: { name: 'B', team: teamB },
  });

  const turns = []; // {a: [act, act], b: [act, act]} per turn
  const snapshots = [];
  let logIdx = 0;
  let snapshotTurn = battle.turn;
  let cur = null;
  let leads = { a: [0, 1], b: [0, 1] };

  // Surface choice-rejection messages as exceptions.
  battle.strictChoices = true;

  PS_TRACE = TRACE_DRAWS ? tracePSDraws(battle) : null;

  const actsToStr = (acts) =>
    acts
      .filter((act) => act.kind !== 'pass')
      .map((act) =>
        act.kind === 'move'
          ? `move ${(act.choiceIdx ?? act.slot) + 1}${act.hasTarget ? ` ${act.target}` : ''}`
          : `switch ${act.position + 1}`
      )
      .join(', ');

  // A move whose target the player does not choose (self, spread, field).
  const NO_TARGET = 0;
  const moveSlotOf = (p, m) => (m ? p.moveSlots.findIndex((ms) => ms.id === m.id) : -1);

  const moveActionsFor = (sideIdx) => {
    const side = sideIdx === 0 ? battle.p1 : battle.p2;
    const reqs = battle.getRequests('move');
    const req = reqs[sideIdx];
    if (!req || !req.active) throw new Error('no move request for side ' + sideIdx);
    return side.active.map((p, slot) => {
      // Fainted slot with no replacement: pass (no choice is sent for it).
      if (!p || p.fainted || p.hp <= 0) return { kind: 'pass' };
      const data = req.active[slot];
      // Valid target locations for each choosable target type. PS uses SIGNED
      // locations: positive = foe slot (1..2), negative = own slot (-(slot+1)).
      const ally = side.active[1 - slot];
      const allyAlive = ally && !ally.fainted && ally.hp > 0;
      const targetsFor = (targetType) => {
        switch (targetType) {
        case 'normal':
        case 'any':
          return [1, 2]; // we only ever aim at the foe
        case 'adjacentFoe':
          return [slot + 1]; // the foe directly across
        case 'adjacentAlly':
          return allyAlive ? [-(2 - slot)] : [];
        case 'adjacentAllyOrSelf': {
          const t = [-(slot + 1)]; // self
          if (allyAlive) t.push(-(2 - slot));
          return t;
        }
        default:
          // self, spread (allAdjacent / allAdjacentFoes), side / field (all,
          // allySide, foeSide, allies, allyTeam), randomNormal, scripted:
          // the player picks no target; PS resolves it. NO_TARGET means the
          // choice is sent without a target argument.
          return [NO_TARGET];
        }
      };
      // A move disabled with `isHidden` (Imprison) looks enabled in a fresh
      // request; PS reveals it only by rejecting the choice ("Unavailable
      // choice"), which marks it disabled in the side's active request.
      const live = side.activeRequest && side.activeRequest.active && side.activeRequest.active[slot];
      const revealedDisabled = (m) =>
        !!(live && live.moves && live.moves.some((lm) => lm.id === m.id && lm.disabled));
      const valid = data.moves
        .map((m, i) => ({ m, i }))
        .filter((x) => !x.m.disabled && !revealedDisabled(x.m) && x.m.pp > 0 &&
          targetsFor(x.m.target).length > 0);
      if (valid.length === 0) {
        // No usable moves (all disabled / 0 PP / no valid target). Prefer the
        // Struggle entry if the request offers one; otherwise pick the first
        // move by index (PS will reject unusable ones, which the fuzzer reports).
        // An enabled move left out only for lack of a target (Helping Hand
        // with the ally fainted) is still a legal choice: aim it at the empty
        // ally slot, and PS fails the move.
        let idx = data.moves.findIndex((m) => m.id === 'struggle');
        if (idx < 0) idx = data.moves.findIndex((m) => !m.disabled && !revealedDisabled(m) && m.pp > 0);
        idx = Math.max(idx, 0);
        const m = data.moves[idx];
        let t = targetsFor(m && m.target);
        if (t.length === 0 && m && (m.target === 'adjacentAlly' || m.target === 'adjacentAllyOrSelf')) {
          t = [-(2 - slot)];
        }
        const target = t.length ? t[controller.random(t.length)] : NO_TARGET;
        return {
          kind: 'move',
          slot: Math.max(moveSlotOf(p, m), 0),
          choiceIdx: idx,
          struggle: !!m && m.id === 'struggle',
          target,
          hasTarget: target !== NO_TARGET,
        };
      }
      const pick = valid[controller.random(valid.length)];
      const t = targetsFor(pick.m.target);
      const target = t[controller.random(t.length)];
      return {
        kind: 'move',
        // `slot` is the pokemon's real move slot (what the engine uses);
        // `choiceIdx` is the index in PS's request, which shrinks to the one
        // locked move during Dig / Outrage / recharge turns.
        slot: Math.max(moveSlotOf(p, pick.m), 0),
        choiceIdx: pick.i,
        target,
        hasTarget: target !== NO_TARGET,
      };
    });
  };

  // Team preview: random 4 of 6, random order (first two become the leads).
  // The battle team is the picked 4, in pick order (PS reorders side.pokemon).
  let battleTeamA = teamA;
  let battleTeamB = teamB;
  if (battle.requestState === 'teampreview') {
    const pick4 = () => {
      const idxs = [0, 1, 2, 3, 4, 5];
      controller.shuffle(idxs);
      return idxs.slice(0, 4);
    };
    const a = pick4();
    const b = pick4();
    leads = { a: [0, 1], b: [0, 1] };
    battleTeamA = a.map((i) => teamA[i]);
    battleTeamB = b.map((i) => teamB[i]);
    battle.makeChoices(
      'team ' + a.map((x) => x + 1).join(', '),
      'team ' + b.map((x) => x + 1).join(', ')
    );
  }

  // Stable team order for snapshots (PS reorders side.pokemon on switches).
  const orderA = new Map();
  battle.p1.pokemon.forEach((p, i) => orderA.set(p, i));
  const orderB = new Map();
  battle.p2.pokemon.forEach((p, i) => orderB.set(p, i));

  // Turn 0: initial state (leads out, full HP) — checked before simulating
  // anything, so stat-calc / species-data / lead errors fail fast.
  onInit(makeSnapshot(battle, logIdx, orderA, orderB), { leads, battleTeamA, battleTeamB });

  let guard = 0;
  while (!battle.ended) {
    if (++guard > 10000) throw new Error('controller loop guard exceeded');

    if (battle.requestState === 'move') {
      // Choose each side separately and re-read the request at commit time:
      // PS can disable a move reactively as the first side commits (Taunt,
      // Intimidate-free "unavailable" updates, ...), so a pre-fetched request
      // for the second side can go stale.
      // `switches`: every forced switch PS requested during this turn, in
      // request order — mid-turn pivots (Volt Switch / U-turn) and
      // end-of-turn faint replacements. Kept apart from a/b so the slot's
      // original move choice is never overwritten.
      cur = { a: [], b: [], switches: [] };
      turns.push(cur);
      // Like a real player: if PS answers "Unavailable choice" (a hidden
      // disable such as Imprison, now revealed in the active request), choose
      // again. Any other choice error is a harness bug and propagates.
      const chooseSide = (sid, si, key) => {
        const side = si === 0 ? battle.p1 : battle.p2;
        for (let attempt = 0; ; attempt++) {
          const acts = moveActionsFor(si);
          cur[key] = acts;
          try {
            battle.choose(sid, actsToStr(acts));
            return;
          } catch (e) {
            if (attempt < 8 && /^\[Unavailable choice\]/.test(e.message)) {
              side.clearChoice();
              continue;
            }
            throw e;
          }
        }
      };
      chooseSide('p1', 0, 'a');
      chooseSide('p2', 1, 'b');
    } else if (battle.requestState === 'switch') {
      const reqs = battle.getRequests('switch');
      for (const [sid, side, key, order, si] of [
        ['p1', battle.p1, 'a', orderA, 0],
        ['p2', battle.p2, 'b', orderB, 1],
      ]) {
        // A side that only waits gets a request without `forceSwitch`.
        if (side.requestState !== 'switch' || !reqs[si] || !reqs[si].forceSwitch) continue;
        const toks = [];
        // Each flagged slot switches in a DISTINCT replacement while any are
        // left (a pokemon can only switch in once); the rest must pass.
        // `pos` is PS's CURRENT side.pokemon index (used in the choice
        // string); the recorded position is the stable battle-team index.
        const candidates = side.pokemon
          .map((q, pos) => ({ q, pos }))
          .filter((x) => !x.q.fainted && x.q.hp > 0 && !side.active.includes(x.q));
        reqs[si].forceSwitch.forEach((flag, slot) => {
          if (!flag) return;
          const p = side.active[slot];
          const pivot = !!(p && p.hp > 0 && !p.fainted);
          // Revival Blessing: the answer picks a fainted party member to revive.
          if (p && side.slotConditions[p.position]['revivalblessing']) {
            const fainted = side.pokemon.map((q, pos) => ({ q, pos })).filter((x) => x.q.fainted);
            const pick = fainted[controller.random(fainted.length)];
            cur.switches.push({ side: key, slot, kind: 'revive', revive: true, position: order.get(pick.q), pivot });
            toks.push(`switch ${pick.pos + 1}`);
            return;
          }
          if (candidates.length > 0) {
            const pickIdx = controller.random(candidates.length);
            const pick = candidates.splice(pickIdx, 1)[0];
            cur.switches.push({ side: key, slot, kind: 'switch', position: order.get(pick.q), pivot });
            toks.push(`switch ${pick.pos + 1}`);
          } else {
            cur.switches.push({ side: key, slot, kind: 'pass', pivot });
            toks.push('pass');
          }
        });
        if (toks.length > 0) battle.choose(sid, toks.join(', '));
      }
    } else if (battle.requestState === 'teampreview') {
      throw new Error('unexpected second teampreview request');
    } else {
      throw new Error('unexpected requestState: ' + battle.requestState);
    }

    // Turn boundary: a new 'move' request (turn advanced) or battle ended.
    if (battle.ended || (battle.requestState === 'move' && battle.turn !== snapshotTurn)) {
      snapshotTurn = battle.turn;
      const snap = makeSnapshot(battle, logIdx, orderA, orderB);
      logIdx = battle.log.length;
      snapshots.push(snap);
      onTurn(snap, turns, snapshots.length - 1, { leads, battleTeamA, battleTeamB });
    }
  }

  if (snapshots.length === 0) throw new Error('battle ended with zero snapshots');

  const winner =
    battle.winner === 'p1' ? 'A' : battle.winner === 'p2' ? 'B' : battle.winner || 'draw';

  return { winner, totalTurns: battle.turn, turns, snapshots, leads, battleTeamA, battleTeamB };
}

// ---------------------------------------------------------------------------
// poke-engine invocation
// ---------------------------------------------------------------------------

function toEngineMember(m, pos) {
  return {
    pos,
    species: m.species,
    level: m.level,
    evs: m.evs,
    ivs: m.ivs,
    nature: m.nature,
    ability: m.ability,
    item: m.item || '',
    gender: m.gender,
    moves: m.moves,
  };
}

function runEngine(opts, input) {
  if (!fs.existsSync(opts.engine)) {
    throw new Error(
      `engine binary not found at ${opts.engine} — build it first:\n` +
        '  cargo build --release --features champions --no-default-features'
    );
  }
  const r = spawnSync(
    opts.engine,
    ['doubles-fuzz-battle', '--input', input.path, '--output', input.outPath],
    { encoding: 'utf8' }
  );
  if (r.status !== 0) {
    const stderr = r.stderr || '';
    if (/unrecognized subcommand/i.test(stderr)) {
      throw new Error(
        `engine binary at ${opts.engine} does not know the 'doubles-fuzz-battle' subcommand ` +
          '(stale build). Rebuild it with:\n' +
          '  cd poke-engine && cargo build --release --features champions --no-default-features'
      );
    }
    const tail = stderr.split('\n').slice(-40).join('\n');
    throw new Error(`engine exited with status ${r.status}:\n${tail}`);
  }
  const out = JSON.parse(fs.readFileSync(input.outPath, 'utf8'));
  return out;
}

// ---------------------------------------------------------------------------
// Readable rendering (labeled, turn-by-turn, for humans and worker agents)
// ---------------------------------------------------------------------------

const BOOST_KEYS = ['atk', 'def', 'spa', 'spd', 'spe', 'acc', 'eva'];

function prettySpecies(id) {
  try {
    const s = ps().Dex.species.get(id);
    return s && s.name ? s.name : id;
  } catch (e) {
    return id;
  }
}

function prettyMove(id) {
  try {
    const m = ps().Dex.moves.get(id);
    return m && m.name ? m.name : id;
  } catch (e) {
    return id;
  }
}

function fmtBoosts(b) {
  if (!b) return 'none';
  const parts = BOOST_KEYS.filter((k) => b[k] !== 0).map((k) => `${k}: ${b[k] > 0 ? '+' : ''}${b[k]}`);
  return parts.length ? `(${parts.join(', ')})` : 'none';
}

function fmtSideConds(sc) {
  if (!sc || !sc.length) return 'none';
  return `(${sc.map((c) => `${c.id}: ${c.duration}`).join(', ')})`;
}

// "a:copperajah:brickbreak" (or "A0:Species:MOVE" from PE) -> "A Copperajah: Brick Break"
function fmtOrderEntry(e) {
  const parts = String(e).split(':');
  if (parts.length === 3) {
    const side = parts[0].replace(/\d+/g, '').toUpperCase() || '?';
    return `${side} ${prettySpecies(parts[1])}: ${prettyMove(parts[2])}`;
  }
  return e;
}

function fmtMoveOrder(order) {
  return order && order.length
    ? order.map((e, i) => `${i + 1}. ${fmtOrderEntry(e)}`).join(' | ')
    : '(none)';
}

// Per-turn change events for one simulator's normalized state:
// "Happiny: took 27 damage (162 -> 135)", "Magearna: status: none -> sleep",
// "Copperajah: bench -> slot 0", "Charjabug: fainted", ...
function turnEvents(prev, cur) {
  const out = { a: [], b: [] };
  for (const key of ['a', 'b']) {
    const cs = cur[key].pokemon;
    const ps = prev ? prev[key].pokemon : [];
    for (let i = 0; i < cs.length; i++) {
      const c = cs[i];
      const p = ps[i];
      const name = prettySpecies(c.species);
      if (!p) {
        out[key].push(`${name}: appeared`);
        continue;
      }
      if (p.hp !== c.hp) {
        const d = c.hp - p.hp;
        out[key].push(
          d < 0
            ? `${name}: took ${-d} damage (${p.hp} -> ${c.hp})`
            : `${name}: recovered ${d} HP (${p.hp} -> ${c.hp})`
        );
      }
      if (p.status !== c.status) out[key].push(`${name}: status: ${p.status} -> ${c.status}`);
      if (!p.fainted && c.fainted) out[key].push(`${name}: fainted`);
      if (p.fainted && !c.fainted) out[key].push(`${name}: revived`);
      const act = (x) => (x === null ? 'bench' : `slot ${x}`);
      if (act(p.activeSlot) !== act(c.activeSlot)) {
        out[key].push(`${name}: ${act(p.activeSlot)} -> ${act(c.activeSlot)}`);
      }
      const b1 = fmtBoosts(p.boosts);
      const b2 = fmtBoosts(c.boosts);
      if (b1 !== b2) out[key].push(`${name}: stat_change: ${b1} -> ${b2}`);
    }
  }
  return out;
}

// Pseudo initial state (before turn 1): full HP, no status, leads out.
function pseudoInitial(cur) {
  const side = (k) => ({
    pokemon: cur[k].pokemon.map((p, i) => ({
      species: p.species,
      hp: p.maxhp,
      maxhp: p.maxhp,
      status: 'none',
      fainted: false,
      activeSlot: i < 2 ? i : null,
      boosts: { atk: 0, def: 0, spa: 0, spd: 0, spe: 0, acc: 0, eva: 0 },
    })),
  });
  return { a: side('a'), b: side('b') };
}

// Render one turn of BOTH simulators, side by side, with every value labeled.
// Lines that differ between PS and PE are marked "<< MISMATCH".
// The missing side (when turn counts differ) renders as MISSING.
function renderTurnComparison(idx, psN, peN, psPrev, pePrev) {
  const L = [];
  L.push(`--- turn ${idx} ---`);
  const fieldLine = (name, a, b) => {
    const av = a === undefined ? 'MISSING' : String(a);
    const bv = b === undefined ? 'MISSING' : String(b);
    L.push(`  ${name}: ${av === bv ? av : `PS=${av} PE=${bv}  << MISMATCH`}`);
  };
  fieldLine('weather', psN && psN.weather, peN && peN.weather);
  fieldLine('terrain', psN && psN.terrain, peN && peN.terrain);
  fieldLine('trickroom', psN && String(psN.trickRoom), peN && String(peN.trickRoom));
  fieldLine(
    'pseudo-weather',
    psN && (psN.pseudoWeather.join(',') || 'none'),
    peN && (peN.pseudoWeather.join(',') || 'none')
  );
  for (const key of ['a', 'b']) {
    const tag = key.toUpperCase();
    L.push(`  side ${tag}:`);
    const psP = psN ? psN[key].pokemon : [];
    const peP = peN ? peN[key].pokemon : [];
    const n = Math.max(psP.length, peP.length);
    for (let i = 0; i < n; i++) {
      const p = psP[i];
      const q = peP[i];
      const sp = p ? p.species : q ? q.species : '???';
      L.push(`    [${i}] ${prettySpecies(sp)}:`);
      const pf = (name, a, b) => {
        const av = a === undefined ? 'MISSING' : String(a);
        const bv = b === undefined ? 'MISSING' : String(b);
        L.push(`      ${name}: ${av === bv ? av : `PS=${av} PE=${bv}  << MISMATCH`}`);
      };
      pf('hp', p && `${p.hp}/${p.maxhp}`, q && `${q.hp}/${q.maxhp}`);
      pf('status', p && p.status, q && q.status);
      pf('fainted', p && p.fainted, q && q.fainted);
      pf(
        'active',
        p && (p.activeSlot === null ? 'bench' : `slot ${p.activeSlot}`),
        q && (q.activeSlot === null ? 'bench' : `slot ${q.activeSlot}`)
      );
      pf('stat_change', p && fmtBoosts(p.boosts), q && fmtBoosts(q.boosts));
    }
    fieldLine('side conditions', psN && fmtSideConds(psN[key].sideConditions), peN && fmtSideConds(peN[key].sideConditions));
    // What happened this turn, per simulator.
    const evA = psN ? turnEvents(psPrev || pseudoInitial(psN), psN)[key] : undefined;
    const evB = peN ? turnEvents(pePrev || pseudoInitial(peN), peN)[key] : undefined;
    const fmtEv = (ev) => (ev === undefined ? undefined : ev.length ? ev.join(' | ') : '(no changes)');
    fieldLine(`turn events [side ${tag}]`, fmtEv(evA), fmtEv(evB));
  }
  const orderA = psN ? fmtMoveOrder(psN.moveOrderRaw) : 'MISSING';
  const orderB = peN ? fmtMoveOrder(peN.moveOrderRaw) : 'MISSING';
  L.push(`  move order: ${orderA === orderB ? '' : '<< MISMATCH'}`.trim());
  L.push(`    PS: ${orderA}`);
  L.push(`    PE: ${orderB}`);
  return L.join('\n');
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

function formatTeamLabeled(team, battleTeam) {
  const lines = [];
  team.forEach((m, i) => {
    lines.push(`  [${i}] ${m.species}:`);
    lines.push(`      ability: ${m.ability}`);
    lines.push(`      gender: ${m.gender}`);
    lines.push(`      nature: ${m.nature}`);
    lines.push(`      stat points: (hp: ${m.evs.hp}, atk: ${m.evs.atk}, def: ${m.evs.def}, spa: ${m.evs.spa}, spd: ${m.evs.spd}, spe: ${m.evs.spe})`);
    const st = effectiveStats(m, ps().Dex);
    lines.push(
      `      stats (level 50, after stat points + nature): (hp: ${st.hp}, atk: ${st.atk}, def: ${st.def}, spa: ${st.spa}, spd: ${st.spd}, spe: ${st.spe})`
    );
    lines.push(`      moves: ${m.moves.map((x, j) => `${j + 1}: ${x}`).join(', ')}`);
  });
  if (battleTeam) {
    lines.push(
      `  battle team (team preview pick, in pick order): ${battleTeam.map((m) => m.species).join(', ')}`
    );
  }
  return lines.join('\n');
}

function formatChoicesLabeled(turns, ps, upTo) {
  const battleTeamA = ps ? ps.battleTeamA : [];
  const battleTeamB = ps ? ps.battleTeamB : [];
  // Which battle-team position occupies each active slot at the START of the
  // given turn (from the previous turn's snapshot, or the leads for turn 1).
  // Failure snapshots start with the turn-0 initial state, so snapshots[i]
  // is the state at the start of turn i + 1.
  const occupantAt = (turnIdx, sideKey, slot) => {
    let snap;
    if (turnIdx === 0) {
      const leads = (ps && ps.leads && ps.leads[sideKey]) || [0, 1];
      return leads[slot];
    } else if (ps && ps.snapshots[turnIdx]) {
      snap = ps.snapshots[turnIdx][sideKey];
    } else {
      return null;
    }
    const entry = snap.pokemon.find((p) => p.activeSlot === slot);
    return entry ? entry.position : null;
  };
  const lines = [];
  turns.slice(0, upTo).forEach((t, i) => {
    lines.push(`  turn ${i + 1}:`);
    const sideLine = (tag, sideKey, acts, bt) => {
      acts.forEach((act, slot) => {
        if (act.kind === 'move') {
          const occ = occupantAt(i, sideKey, slot);
          const mvName =
            occ !== null && bt[occ] && bt[occ].moves[act.slot]
              ? `${bt[occ].species} uses move ${act.slot + 1} (${prettyMove(bt[occ].moves[act.slot])})`
              : `move ${act.slot + 1}`;
          let targetLabel = 'n/a';
          if (act.hasTarget) {
            if (act.target > 0) targetLabel = `foe slot ${act.target}`;
            else if (act.target === -(slot + 1)) targetLabel = 'self';
            else if (act.target < 0) targetLabel = `ally slot ${-(act.target)}`;
          }
          lines.push(`    ${tag} slot ${slot}: move: ${mvName}  target: ${targetLabel}`);
        } else if (act.kind === 'switch') {
          lines.push(
            `    ${tag} slot ${slot}: switch: in position ${act.position + 1} (${
              bt[act.position] ? bt[act.position].species : '?'
            })`
          );
        } else {
          lines.push(`    ${tag} slot ${slot}: pass: fainted, no replacement available`);
        }
      });
    };
    sideLine('A', 'a', t.a, battleTeamA);
    sideLine('B', 'b', t.b, battleTeamB);
    (t.switches || []).forEach((sw) => {
      const tag = sw.side === 'a' ? 'A' : 'B';
      const bt = sw.side === 'a' ? battleTeamA : battleTeamB;
      const why = sw.revive ? 'Revival Blessing' : sw.pivot ? 'pivot switch (mid-turn)' : 'faint replacement (end of turn)';
      lines.push(
        sw.kind === 'revive'
          ? `    ${tag} slot ${sw.slot}: ${why}: revive position ${sw.position + 1} (${
            bt[sw.position] ? bt[sw.position].species : '?'
          })`
          : sw.kind === 'switch'
          ? `    ${tag} slot ${sw.slot}: ${why}: in position ${sw.position + 1} (${
            bt[sw.position] ? bt[sw.position].species : '?'
          })`
          : `    ${tag} slot ${sw.slot}: ${why}: pass, no replacement available`
      );
    });
  });
  return lines.join('\n');
}

function writeFailure(opts, testIdx, info) {
  const dir = path.join(opts.workdir, `test-${testIdx + 1}`);
  fs.mkdirSync(dir, { recursive: true });
  const lines = [];
  lines.push('=== FUZZ MISMATCH ===');
  lines.push(`test #${testIdx + 1} of ${opts.count}`);
  lines.push(`tester seed: ${opts.seed}`);
  lines.push(`battle seed: ${info.battleSeed.join(', ')}`);
  lines.push('');
  lines.push('Team A (6 random members):');
  lines.push(formatTeamLabeled(info.teamA, info.ps && info.ps.battleTeamA));
  lines.push('Team B (6 random members):');
  lines.push(formatTeamLabeled(info.teamB, info.ps && info.ps.battleTeamB));
  lines.push('');
  if (info.engineError) {
    lines.push('RESULT: ENGINE ERROR (poke-engine) — the engine crashed or reported an error:');
    lines.push(info.engineError);
    if (info.pe && info.ps && info.ps.snapshots.length) {
      const i = Math.min(info.ps.snapshots.length, info.pe.turns.length) - 1;
      if (i >= 0) {
        lines.push('');
        lines.push(`State at the last completed turn (turn ${i}), PE output is partial:`);
        lines.push(
          renderTurnComparison(
            i,
            normalizePSState(info.ps.snapshots[i]),
            normalizePETurn(info.pe.turns[i]),
            i > 0 ? normalizePSState(info.ps.snapshots[i - 1]) : undefined,
            i > 0 ? normalizePETurn(info.pe.turns[i - 1]) : undefined
          )
        );
      }
    }
  } else if (info.psError) {
    lines.push('RESULT: PS ERROR (pokemon-showdown) — the fuzzer hit an error while driving PS:');
    lines.push(info.psError);
  } else {
    const cmp = info.cmp;
    const turnLabel =
      cmp.turn === 0
        ? 'turn 0 (initial state, before any moves were simulated)'
        : `turn ${cmp.turn}`;
    lines.push(`RESULT: STATE MISMATCH at ${turnLabel} — simulation stopped here (early exit)`);
    lines.push('');
    lines.push('Differing fields (machine-readable list):');
    for (const d of cmp.allDiffs.slice(0, 40)) {
      lines.push(`  ${d.path}: PS=${JSON.stringify(d.ps)} PE=${JSON.stringify(d.pe)}`);
    }
    if (cmp.allDiffs.length > 40) lines.push(`  ... and ${cmp.allDiffs.length - 40} more`);
    lines.push('');
    lines.push(...prngReport(info.pe, cmp.turn));
    if (PS_TRACE) {
      lines.push('');
      lines.push(`PS battle log, turn ${cmp.turn}:`);
      lines.push(...psTurnLog(PS_TRACE.battle, cmp.turn).map((l) => '  ' + l));
    }
    lines.push('');
    lines.push(
      `turns simulated so far: PS=${info.ps.snapshots.length - 1}  PE=${
        info.pe ? info.pe.turns.length - 1 : '?'
      }`
    );
    lines.push('');
    lines.push(
      cmp.turn === 0
        ? 'The initial state in detail (PS = pokemon-showdown, PE = poke-engine):'
        : 'The failing turn in detail (PS = pokemon-showdown, PE = poke-engine):'
    );
    lines.push('"<< MISMATCH" marks differences; "turn events" shows per-turn changes.');
    const i = cmp.turn;
    lines.push(
      renderTurnComparison(
        i,
        normalizePSState(info.ps.snapshots[i]),
        info.pe && info.pe.turns[i] ? normalizePETurn(info.pe.turns[i]) : null,
        i > 0 ? normalizePSState(info.ps.snapshots[i - 1]) : undefined,
        info.pe && i > 0 && info.pe.turns[i - 1] ? normalizePETurn(info.pe.turns[i - 1]) : undefined
      )
    );
  }
  lines.push('');
  lines.push('Choices the controller sent to BOTH simulators:');
  const upTo = info.ps ? info.ps.turns.length : 0;
  lines.push(formatChoicesLabeled(info.ps ? info.ps.turns : [], info.ps, upTo));
  const report = lines.join('\n');
  fs.writeFileSync(path.join(dir, 'report.txt'), report);
  console.error(opts.verbose ? report : compactReport(opts, testIdx, info));
  console.error(`\nfull report: ${path.join(dir, 'report.txt')}`);
}

// Console summary of a failure: only what differs, plus enough context to
// reproduce it. The full report (all teams, full state, all turns) is in
// report.txt.
function compactReport(opts, testIdx, info) {
  const out = [];
  out.push(`=== FUZZ MISMATCH === test #${testIdx + 1} of ${opts.count} (tester seed ${opts.seed})`);
  out.push(`reproduce: node fuzz/fuzz.js --seed ${opts.seed + testIdx} --count 1`);
  out.push(`battle seed: ${info.battleSeed.join(', ')}`);
  const teamLine = (tag, bt) => {
    if (!bt) return;
    out.push(`team ${tag}:`);
    bt.forEach((m, i) =>
      out.push(`  [${i}] ${m.species} (${m.ability}): ${m.moves.map(prettyMove).join(', ')}`)
    );
  };
  teamLine('A', info.ps && info.ps.battleTeamA);
  teamLine('B', info.ps && info.ps.battleTeamB);
  out.push('');

  let turn = null;
  if (info.engineError) {
    out.push('RESULT: ENGINE ERROR (poke-engine):');
    out.push(info.engineError);
    turn = info.ps ? info.ps.turns.length : null;
  } else if (info.psError) {
    out.push('RESULT: PS ERROR (pokemon-showdown, harness bug):');
    out.push(info.psError);
    turn = info.ps ? info.ps.turns.length : null;
  } else {
    const cmp = info.cmp;
    turn = cmp.turn;
    out.push(`RESULT: STATE MISMATCH at turn ${cmp.turn}${cmp.turn === 0 ? ' (initial state)' : ''}`);
    for (const d of cmp.allDiffs.slice(0, 40)) {
      out.push(`  ${d.path}: PS=${JSON.stringify(d.ps)} PE=${JSON.stringify(d.pe)}`);
    }
    if (cmp.allDiffs.length > 40) out.push(`  ... and ${cmp.allDiffs.length - 40} more`);
    // Only the differing lines of the turn detail, under their side /
    // pokemon headings.
    const i = cmp.turn;
    const detail = renderTurnComparison(
      i,
      normalizePSState(info.ps.snapshots[i]),
      info.pe && info.pe.turns[i] ? normalizePETurn(info.pe.turns[i]) : null,
      i > 0 ? normalizePSState(info.ps.snapshots[i - 1]) : undefined,
      info.pe && i > 0 && info.pe.turns[i - 1] ? normalizePETurn(info.pe.turns[i - 1]) : undefined
    ).split('\n');
    out.push('');
    out.push(...prngReport(info.pe, cmp.turn));
    out.push('');
    out.push('differences in detail:');
    let side = null;
    let mon = null;
    for (let k = 0; k < detail.length; k++) {
      const l = detail[k];
      if (/^\s*side [AB]:/.test(l)) {
        side = l;
        mon = null;
      } else if (/^\s*\[\d+\] /.test(l)) {
        mon = l;
      }
      if (!l.includes('<< MISMATCH')) continue;
      if (side) out.push(side), (side = null);
      if (mon && !/turn events|side conditions|move order/.test(l)) out.push(mon), (mon = null);
      out.push(l);
      if (/move order:/.test(l)) out.push(detail[k + 1], detail[k + 2]);
    }
  }

  if (PS_TRACE && turn !== null) {
    const lines = psTurnLog(PS_TRACE.battle, turn);
    if (lines.length) {
      out.push('');
      out.push(`PS battle log, turn ${turn} (what pokemon-showdown did):`);
      out.push(...lines.map((l) => '  ' + l));
    }
  }

  // Choices of the failing turn only (earlier turns are in report.txt).
  if (info.ps && turn) {
    const all = formatChoicesLabeled(info.ps.turns, info.ps, info.ps.turns.length).split('\n');
    const start = all.indexOf(`  turn ${turn}:`);
    if (start >= 0) {
      out.push('');
      out.push('choices sent to both simulators:');
      out.push(...all.slice(start));
    }
  }
  return out.join('\n');
}

// Run poke-engine on the first `n` turns (stepwise mode: stop_after = n).
// The engine input for the first `n` turns of a battle.
function engineInput(battleSeed, meta, turns, n) {
  return {
    seed: battleSeed,
    teams: {
      a: meta.battleTeamA.map((m, i) => toEngineMember(m, i)),
      b: meta.battleTeamB.map((m, i) => toEngineMember(m, i)),
    },
    leads: meta.leads,
    choices: turns.slice(0, n),
    stop_after: n,
  };
}

function runEngineTurn(opts, dir, battleSeed, meta, turns, n) {
  const inputPath = path.join(dir, 'input.json');
  const outPath = path.join(dir, 'pe.json');
  fs.writeFileSync(
    inputPath,
    JSON.stringify({
      seed: battleSeed,
      teams: {
        a: meta.battleTeamA.map((m, i) => toEngineMember(m, i)),
        b: meta.battleTeamB.map((m, i) => toEngineMember(m, i)),
      },
      leads: meta.leads,
      choices: turns.slice(0, n),
      stop_after: n,
    })
  );
  return runEngine(opts, { path: inputPath, outPath });
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function failTest(opts, t, info) {
  const dir = path.join(opts.workdir, `test-${t + 1}`);
  fs.mkdirSync(dir, { recursive: true });
  if (info.ps) {
    fs.writeFileSync(path.join(dir, 'ps.json'), JSON.stringify(info.ps, null, 2));
  }
  writeFailure(opts, t, info);
  process.exit(1);
}

// PS battle cache (fast path only). The PS side of a battle depends only on
// the test seed, the harness (this file, teams.js) and the pokemon-showdown
// build, never on the engine; so a battle's PS result (choices, snapshots,
// winner) is stored once and reused by later sweeps. The key hashes all of
// those inputs, so any harness or PS change starts a fresh cache.
// Location: fuzz/work/ps-cache/<key>/<seed>.json (gitignored).
let PS_CACHE_DIR = null;
function psCacheDir() {
  if (PS_CACHE_DIR) return PS_CACHE_DIR;
  const h = require('crypto').createHash('sha1');
  h.update(fs.readFileSync(__filename));
  h.update(fs.readFileSync(path.join(__dirname, 'teams.js')));
  // The PS build that actually runs (dist/): a rebuild rewrites these.
  for (const f of ['sim/battle.js', 'sim/battle-actions.js', 'data/moves.js', 'data/abilities.js',
    'data/conditions.js', 'data/mods/champions/scripts.js']) {
    h.update(`${f}:${fs.statSync(path.join(PS_DIR, 'dist', f)).mtimeMs};`);
  }
  h.update(FORMAT);
  const root = path.join(__dirname, 'work', 'ps-cache');
  PS_CACHE_DIR = path.join(root, h.digest('hex').slice(0, 16));
  if (!fs.existsSync(PS_CACHE_DIR)) {
    // New harness / PS build: older caches can never be used again.
    for (const old of fs.existsSync(root) ? fs.readdirSync(root) : []) {
      fs.rmSync(path.join(root, old), { recursive: true, force: true });
    }
  }
  fs.mkdirSync(PS_CACHE_DIR, { recursive: true });
  return PS_CACHE_DIR;
}
function psCacheRead(seed) {
  try {
    return JSON.parse(fs.readFileSync(path.join(psCacheDir(), `${seed}.json`), 'utf8'));
  } catch (e) {
    return null;
  }
}
function psCacheWrite(seed, game) {
  const file = path.join(psCacheDir(), `${seed}.json`);
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(game));
  fs.renameSync(tmp, file); // atomic: parallel workers never read half a file
}

// Tests checked per engine call on the fast path.
const FAST_CHUNK = 8;
const FAST_CHUNK_CACHED = 128;

// The controller PRNG, battle seed and teams of one test (all derived from
// the test seed, so a test can be rebuilt identically).
function makeTest(testSeed) {
  const controller = new (ps().PRNG)([
    testSeed & 0xffff,
    (testSeed + 1) & 0xffff,
    (testSeed + 2) & 0xffff,
    (testSeed + 3) & 0xffff,
  ]);
  // 1. Battle seed (4 x 16-bit words), from the controller stream.
  const battleSeed = [0, 0, 0, 0].map(() => controller.random(65536));
  // 2. Two fully random teams, from the controller stream.
  const teamA = genTeam(controller, ps().MOD_DEX);
  const teamB = genTeam(controller, ps().MOD_DEX);
  return { controller, battleSeed, teamA, teamB };
}

// Fast pass/fail check of several tests: play each whole PS battle (no draw
// tracing), run the engine ONCE for all of them (`--batch`), and compare
// every turn's state and the winner. The engine is deterministic, so turn n
// of a full run equals the stepwise run with `stop_after: n`. Returns the
// index of the first test that is not a clean pass (PS error, engine error
// or mismatch), or -1; the caller re-runs that test stepwise for the report.
function fastCheckBatch(opts, dir, seeds) {
  const games = [];
  let firstBad = -1;
  TRACE_DRAWS = false;
  for (let k = 0; k < seeds.length; k++) {
    let game = opts.cache ? psCacheRead(seeds[k]) : null;
    if (!game) {
      const { controller, battleSeed, teamA, teamB } = makeTest(seeds[k]);
      let initSnap = null;
      const snaps = [];
      try {
        const meta = stepPSBattle(battleSeed, teamA, teamB, controller,
          (snap) => {
            initSnap = snap;
          },
          (snap) => {
            snaps.push(snap);
          });
        const { turns, leads, battleTeamA, battleTeamB, winner } = meta;
        game = { battleSeed, meta: { turns, leads, battleTeamA, battleTeamB, winner }, initSnap, snaps };
      } catch (e) {
        firstBad = k; // later tests cannot be first (PS errors are not cached)
        break;
      }
      if (opts.cache) psCacheWrite(seeds[k], game);
    }
    games.push(game);
  }
  TRACE_DRAWS = true;
  if (!games.length) return firstBad;

  const inPath = path.join(dir, 'batch-input.json');
  const outPath = path.join(dir, 'batch-output.json');
  fs.writeFileSync(inPath, JSON.stringify(games.map((g) =>
    engineInput(g.battleSeed, g.meta, g.meta.turns, g.meta.turns.length))));
  const r = spawnSync(opts.engine, ['doubles-fuzz-battle', '--batch', '--input', inPath, '--output', outPath],
    { encoding: 'utf8' });
  if (r.status !== 0) return 0; // let the stepwise path explain it
  const results = JSON.parse(fs.readFileSync(outPath, 'utf8'));

  const same = (psSnap, peTurn) => {
    const { moveOrderRaw: _a, turn: _t1, ...psC } = normalizePSState(psSnap);
    const { moveOrderRaw: _b, turn: _t2, ...peC } = normalizePETurn(peTurn);
    const diffs = [];
    deepDiff(psC, peC, '', diffs);
    return diffs.length === 0;
  };
  for (let k = 0; k < games.length; k++) {
    const g = games[k];
    const pe = results[k];
    let ok = pe && !pe.error && g.initSnap && pe.turns.length >= g.snaps.length + 1 &&
      same(g.initSnap, pe.turns[0]);
    for (let n = 1; ok && n <= g.snaps.length; n++) ok = same(g.snaps[n - 1], pe.turns[n]);
    if (!ok || pe.winner !== g.meta.winner) return k;
  }
  return firstBad;
}

// Parallel sweep: worker j checks seeds seed+j, seed+j+jobs, ... (each stops
// at its first failure). The lowest failing seed is then re-run here as a
// single test, which prints the normal report. Same result as a sequential
// sweep: the first failing seed in order.
function runParallel(opts) {
  const { spawn } = require('child_process');
  const jobs = Math.min(opts.jobs, opts.count);
  let pending = jobs;
  let first = Infinity; // lowest failing seed so far
  // checked[j]: worker j has checked every one of its seeds below this one.
  const checked = Array.from({ length: jobs }, (_, j) => opts.seed + j);
  const children = [];
  // A worker can stop once its next seed is above the lowest failure.
  const prune = () => {
    children.forEach((c, j) => {
      if (!c.done && checked[j] > first) c.kill();
    });
  };
  for (let j = 0; j < jobs; j++) {
    const count = Math.ceil((opts.count - j) / jobs);
    const child = spawn(
      process.execPath,
      [__filename, '--seed', String(opts.seed + j), '--count', String(count), '--step', String(jobs),
        '--check-only', '--jobs', '1', '--engine', opts.engine, ...(opts.cache ? [] : ['--no-cache']),
        '--workdir', path.join(opts.workdir, `worker-${j}`)],
      { stdio: ['ignore', 'pipe', 'inherit'] }
    );
    children.push(child);
    let buf = '';
    child.stdout.on('data', (d) => {
      buf += d;
      let nl;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl);
        buf = buf.slice(nl + 1);
        const m = line.match(/^(PASS|FAIL) (\d+)$/);
        if (!m) continue;
        const seed = parseInt(m[2], 10);
        if (m[1] === 'FAIL') first = Math.min(first, seed);
        else checked[j] = seed + jobs;
      }
      prune();
    });
    child.on('close', () => {
      child.done = true;
      if (--pending === 0) {
        if (first === Infinity) process.exit(0);
        // Re-run the first failing seed alone, in this process: full report.
        process.argv = [process.argv[0], __filename, '--seed', String(first), '--count', '1', '--jobs', '1',
          '--engine', opts.engine, '--workdir', opts.workdir];
        main();
      }
    });
  }
}

function main() {
  const opts = parseArgs();
  fs.mkdirSync(opts.workdir, { recursive: true });

  // Passing tests print nothing unless --verbose; a mismatch prints the full
  // report (and exits 1).
  const log = opts.verbose ? console.log : () => {};

  log(`fuzz: tester seed=${opts.seed}, tests=${opts.count}, format=${FORMAT}`);
  log(`fuzz: engine=${opts.engine}`);
  log('');

  let passedUpTo = 0;
  for (let t = 0; t < opts.count; t++) {
    // Controller PRNG: one per test, seeded from (tester seed + test index),
    // so test #k of `--seed S` reproduces alone as `--seed S+k-1 --count 1`.
    // Separate from the battle seed.
    const testSeed = opts.seed + t * opts.step;

    // Fast path, in chunks: tests before `passedUpTo` are known to pass.
    // Only the first failing test is re-run stepwise below (the report).
    if (!opts.verbose && t >= passedUpTo) {
      // A run of cached PS battles costs nothing to load, so it goes to the
      // engine in one batch (up to 128); uncached battles keep small chunks
      // for early stopping.
      const isCached = (k) =>
        opts.cache && fs.existsSync(path.join(psCacheDir(), `${opts.seed + (t + k) * opts.step}.json`));
      let n = 0;
      while (n < FAST_CHUNK_CACHED && t + n < opts.count && isCached(n)) n++;
      if (n === 0) n = Math.min(FAST_CHUNK, opts.count - t);
      const seeds = Array.from({ length: n }, (_, k) => opts.seed + (t + k) * opts.step);
      const bad = fastCheckBatch(opts, opts.workdir, seeds);
      passedUpTo = bad < 0 ? t + n : t + bad;
      // Parallel worker: tell the parent how far this worker got.
      if (opts.checkOnly && passedUpTo > t) {
        process.stdout.write(`PASS ${opts.seed + (passedUpTo - 1) * opts.step}\n`);
      }
    }
    if (!opts.verbose && t < passedUpTo) continue;

    const dir = path.join(opts.workdir, `test-${t + 1}`);
    fs.mkdirSync(dir, { recursive: true });
    if (opts.checkOnly) {
      // Parallel worker: report the failing seed to the parent and stop.
      process.stdout.write(`FAIL ${testSeed}\n`);
      process.exit(1);
    }

    const { controller, battleSeed, teamA, teamB } = makeTest(testSeed);
    const info = { battleSeed, teamA, teamB };

    log(`test #${t + 1}: battle seed=${battleSeed.join(', ')}`);
    log('Team A:');
    log(formatTeamLabeled(teamA, null));
    log('Team B:');
    log(formatTeamLabeled(teamB, null));

    // 3. Stepwise: check the initial state (turn 0) first, then one turn on
    //    PS -> same turns on poke-engine -> compare. Exits immediately at the
    //    first divergence. Engine turns are 1-based behind a turn-0 initial
    //    snapshot: pe.turns[0] = initial, pe.turns[n] = state after turn n.
    let prevPsN = undefined;
    let prevPeN = undefined;
    let lastPe = null;
    let meta;
    let snapshotsSoFar = [];
    let initSnap = null;
    const fail = (extra) => {
      Object.assign(info, extra, {
        ps: {
          winner: '(battle not finished)',
          totalTurns: snapshotsSoFar.length - 1,
          turns: meta ? metaTurns : [],
          snapshots: snapshotsSoFar,
          leads: meta ? meta.leads : undefined,
          battleTeamA: meta ? meta.battleTeamA : undefined,
          battleTeamB: meta ? meta.battleTeamB : undefined,
        },
      });
      failTest(opts, t, info);
    };
    let metaTurns = [];
    try {
      meta = stepPSBattle(
        battleSeed,
        teamA,
        teamB,
        controller,
        (snap, m) => {
          // --- turn 0: initial state check (no moves simulated yet) ---
          meta = m;
          initSnap = snap;
          snapshotsSoFar = [snap];
          let pe;
          try {
            pe = runEngineTurn(opts, dir, battleSeed, m, [], 0);
          } catch (e) {
            info.engineError = e.message;
            fail({});
          }
          lastPe = pe;
          if (pe.error) {
            fail({ engineError: `engine reported error: ${pe.error}`, pe });
          }
          if (!pe.turns.length) {
            fail({
              cmp: {
                turn: 0,
                turnDiffs: [],
                allDiffs: [{ path: 'engineTurnsProduced', ps: 1, pe: 0 }],
              },
              pe,
            });
          }
          const psN = normalizePSState(snap);
          const peN = normalizePETurn(pe.turns[0]);
          const { moveOrderRaw: _a, turn: _t1, ...psC } = psN;
          const { moveOrderRaw: _b, turn: _t2, ...peC } = peN;
          const diffs = [];
          deepDiff(psC, peC, '', diffs);
          if (diffs.length > 0) {
            fail({ cmp: { turn: 0, turnDiffs: diffs, allDiffs: diffs }, pe });
          }
          log('  initial state (turn 0): OK');
          if (opts.verbose) log(renderTurnComparison(0, psN, peN, undefined, undefined));
          prevPsN = psN;
          prevPeN = peN;
        },
        (snap, turns, snapIdx, m) => {
          // --- turn n: state check ---
          meta = m;
          metaTurns = turns;
          const n = turns.length;
          snapshotsSoFar.push(snap);
          let pe;
          try {
            pe = runEngineTurn(opts, dir, battleSeed, m, turns, n);
          } catch (e) {
            info.engineError = e.message;
            fail({});
          }
          lastPe = pe;
          if (pe.error) {
            fail({ engineError: `engine reported error: ${pe.error}`, pe });
          }
          if (pe.turns.length < n + 1) {
            fail({
              cmp: {
                turn: n,
                turnDiffs: [],
                allDiffs: [{ path: 'engineTurnsProduced', ps: n + 1, pe: pe.turns.length }],
              },
              pe,
            });
          }
          const psN = normalizePSState(snap);
          const peN = normalizePETurn(pe.turns[n]);
          // `turn` labels differ by design (PS counts the next turn); ignore it.
          const { moveOrderRaw: _a, turn: _t1, ...psC } = psN;
          const { moveOrderRaw: _b, turn: _t2, ...peC } = peN;
          const diffs = [];
          deepDiff(psC, peC, '', diffs);
          if (diffs.length > 0) {
            fail({ cmp: { turn: n, turnDiffs: diffs, allDiffs: diffs }, pe });
          }
          log(`  turn ${n}: OK`);
          if (opts.verbose) {
            log(renderTurnComparison(n, psN, peN, prevPsN, prevPeN));
          }
          prevPsN = psN;
          prevPeN = peN;
        }
      );
    } catch (e) {
      info.psError = e.stack || String(e);
      fail({});
    }

    // 4. Winner check (the last engine run covers the whole battle).
    const winner = meta.winner;
    if (lastPe && lastPe.winner !== winner) {
      info.cmp = {
        turn: meta.turns.length - 1,
        turnDiffs: [],
        allDiffs: [{ path: 'winner', ps: winner, pe: lastPe.winner }],
      };
      info.pe = lastPe;
      info.ps = {
        winner,
        totalTurns: meta.totalTurns,
        turns: meta.turns,
        snapshots: meta.snapshots,
        leads: meta.leads,
        battleTeamA: meta.battleTeamA,
        battleTeamB: meta.battleTeamB,
      };
      failTest(opts, t, info);
    }

    log(`test #${t + 1} OK  winner=${winner}  turns=${meta.totalTurns}`);
    log('');
  }

  log(`\nall ${opts.count} tests passed`);
  process.exit(0);
}

if (!PARALLEL_PARENT) main();
