#!/usr/bin/env node
// Legal-choice audit: plays the fuzzer's battles in pokemon-showdown and
// checks, at every move request, that the engine's legal choices
// (`a_options` / `b_options` in its turn snapshots, src/doubles/legal.rs)
// match PS: the offered moves, which are disabled (hidden ones included), and
// whether the pokemon is trapped. Does not affect the fuzz score.
//
//   node fuzz/legal_audit.js --seed <S> --count <N> [--engine <path>] [--examples <K>]
//
// Exit code 1 when any mismatch is found; mismatches are grouped by kind.

const path = require('path');
const fs = require('fs');
const os = require('os');
const { spawnSync } = require('child_process');
const { makeTest, stepPSBattle, engineInput, setTraceDraws } = require('./fuzz.js');

function parseArgs() {
  const a = process.argv.slice(2);
  const o = {
    seed: null,
    count: null,
    engine: path.resolve(__dirname, '..', 'target', 'release', 'poke-engine'),
    examples: 3,
  };
  for (let i = 0; i < a.length; i++) {
    if (a[i] === '--seed') o.seed = Number(a[++i]);
    else if (a[i] === '--count') o.count = Number(a[++i]);
    else if (a[i] === '--engine') o.engine = path.resolve(a[++i]);
    else if (a[i] === '--examples') o.examples = Number(a[++i]);
    else throw new Error('unknown argument ' + a[i]);
  }
  if (o.seed === null || o.count === null) throw new Error('usage: --seed S --count N');
  return o;
}

// PS's legal choices for both sides at the current move request.
function psOptions(battle) {
  if (battle.ended || battle.requestState !== 'move') return null;
  return [battle.p1, battle.p2].map((side) =>
    [0, 1].map((slot) => {
      const p = side.active[slot];
      if (!p || p.fainted || p.hp <= 0) return null;
      const req = side.activeRequest && side.activeRequest.active && side.activeRequest.active[slot];
      if (!req) return null;
      return {
        moves: req.moves.map((m) => {
          const ms = p.moveSlots.find((x) => x.id === m.id);
          // `disabled: 'hidden'` (Imprison) shows as enabled in the request.
          // A locked move (no `pp` in the request) is forced, never disabled.
          const locked = m.pp === undefined;
          return { id: m.id, pp: locked || m.id === 'struggle' ? 0 : m.pp,
            disabled: !locked && !!(m.disabled || (ms && ms.disabled)) };
        }),
        trapped: !!p.trapped,
      };
    })
  );
}

function peSlot(o) {
  if (!o) return null;
  return { moves: o.moves.map((m) => ({ id: m.id, pp: m.pp, disabled: m.disabled })), trapped: o.trapped };
}

function classify(ps, pe) {
  if (!ps !== !pe) return 'slot presence';
  if (!ps) return null;
  const ids = (x) => x.moves.map((m) => m.id).join(',');
  if (ids(ps) !== ids(pe)) {
    if (ps.moves.length === 1 || pe.moves.length === 1) return `offered moves (locked/struggle): PS=${ids(ps)} PE=${ids(pe)}`;
    return 'offered moves';
  }
  for (let i = 0; i < ps.moves.length; i++) {
    if (ps.moves[i].disabled !== pe.moves[i].disabled) {
      return `disabled ${ps.moves[i].id}: PS=${ps.moves[i].disabled} PE=${pe.moves[i].disabled}`;
    }
  }
  if (ps.trapped !== pe.trapped) return `trapped: PS=${ps.trapped} PE=${pe.trapped}`;
  for (let i = 0; i < ps.moves.length; i++) {
    if (ps.moves[i].pp !== pe.moves[i].pp) return `pp ${ps.moves[i].id}`;
  }
  return null;
}

function main() {
  const opts = parseArgs();
  setTraceDraws(false);
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'legal-audit-'));
  const kinds = new Map();
  let requests = 0;
  let checked = 0;
  let skipped = 0;
  const CHUNK = 64;
  for (let start = 0; start < opts.count; start += CHUNK) {
    const games = [];
    for (let t = start; t < Math.min(opts.count, start + CHUNK); t++) {
      const seed = opts.seed + t;
      const { controller, battleSeed, teamA, teamB } = makeTest(seed);
      const psOpts = [];
      try {
        const meta = stepPSBattle(battleSeed, teamA, teamB, controller,
          (snap, m, battle) => psOpts.push(psOptions(battle)),
          (snap, turns, idx, m, battle) => psOpts.push(psOptions(battle)));
        games.push({ seed, battleSeed, meta, psOpts });
      } catch (e) {
        skipped++; // a PS-side harness error: the fuzzer reports those
      }
    }
    const inPath = path.join(dir, 'in.json');
    const outPath = path.join(dir, 'out.json');
    fs.writeFileSync(inPath, JSON.stringify(games.map((g) =>
      engineInput(g.battleSeed, g.meta, g.meta.turns, g.meta.turns.length))));
    const r = spawnSync(opts.engine, ['doubles-fuzz-battle', '--batch', '--input', inPath, '--output', outPath],
      { encoding: 'utf8', maxBuffer: 1 << 30 });
    if (r.status !== 0) throw new Error('engine failed: ' + r.stderr.slice(-2000));
    const results = JSON.parse(fs.readFileSync(outPath, 'utf8'));
    games.forEach((g, k) => {
      const pe = results[k];
      if (!pe || !pe.turns) {
        skipped++;
        return;
      }
      checked++;
      // The engine state only matches PS up to the first fuzz divergence;
      // audit the turns before it (all of them for a passing seed).
      g.psOpts.forEach((sides, n) => {
        if (!sides || !pe.turns[n]) return;
        const peTurn = pe.turns[n];
        ['a', 'b'].forEach((key, si) => {
          [0, 1].forEach((slot) => {
            const ps = sides[si][slot];
            if (ps) requests++;
            const kind = classify(ps, peSlot(peTurn[`${key}_options`][slot]));
            if (!kind) return;
            const e = kinds.get(kind) || { count: 0, examples: [] };
            e.count++;
            if (e.examples.length < opts.examples) {
              const mon = peTurn[key].find((p) => p.active_slot === slot);
              e.examples.push(`seed ${g.seed} turn ${n} ${key}${slot} ${mon ? mon.species : '?'}: ` +
                `PS=${JSON.stringify(ps)} PE=${JSON.stringify(peSlot(peTurn[`${key}_options`][slot]))}`);
            }
            kinds.set(kind, e);
          });
        });
      });
    });
  }
  console.log(`audited ${checked} battles (${skipped} skipped), ${requests} slot requests`);
  if (kinds.size === 0) {
    console.log('no mismatches');
    process.exit(0);
  }
  const sorted = [...kinds.entries()].sort((x, y) => y[1].count - x[1].count);
  for (const [kind, e] of sorted) {
    console.log(`\n${e.count}x ${kind}`);
    for (const ex of e.examples) console.log('  ' + ex);
  }
  process.exit(1);
}

main();
