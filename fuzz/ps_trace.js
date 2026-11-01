#!/usr/bin/env node
// Trace what pokemon-showdown did inside one turn of one fuzz battle.
//
//   node fuzz/ps_trace.js --seed <S> --turn <T>
//
// Replays `fuzz.js --seed S --count 1` in-process (stops at its first
// mismatch, like the fuzzer) and prints, in order, for turn T:
//   - protocol lines (|move|, |-damage|, |-activate|, ...),
//   - every modifier PS applied, with the effect that applied it
//     ("Tough Claws (Pinsir) BasePower x1.3"),
//   - every event that changed a value (BasePower 3 -> 4, ModifyAtk, Damage...),
//   - PS debug messages,
//   - every PRNG draw with its purpose (same vocabulary as the fuzzer).
// Use it when the fuzzer says "PRNG draws line up" and a number differs: the
// modifier lines name the ability / item / condition the engine is missing.

const path = require('path');
const { psDrawPurpose } = require('./fuzz_purpose.js');

const args = process.argv.slice(2);
const get = (k) => {
  const i = args.indexOf(k);
  return i >= 0 ? parseInt(args[i + 1], 10) : NaN;
};
const SEED = get('--seed');
const TURN = get('--turn');
const MODE = args.includes('--mode') ? args[args.indexOf('--mode') + 1] : (process.env.FUZZ_MODE || 'random');
if (Number.isNaN(SEED) || Number.isNaN(TURN)) {
  console.error('usage: node fuzz/ps_trace.js --seed <S> --turn <T> [--mode random|champions]');
  process.exit(2);
}

const PS_DIR = path.resolve(__dirname, '..', '..', 'pokemon-showdown');
const Sim = require(path.join(PS_DIR, 'dist', 'sim'));
const B = Sim.Battle.prototype;
const out = [];
const show = (b) => b.turn === TURN;
const name = (a) => (a && typeof a === 'object' ? a.name || a.id || '[obj]' : JSON.stringify(a));
const who = (b) => {
  const e = b.effect;
  const holder = b.effectState && b.effectState.target;
  return `${e ? e.name : '?'}${holder && holder.name ? ` (${holder.name})` : ''}`;
};

// Protocol lines, minus timestamps / blank lines / the percentage copy of
// each HP line (the exact-HP copy is kept).
const proto = (b, parts) => {
  if (!show(b) || typeof parts[0] !== 'string' || parts.some((p) => typeof p === 'function')) return;
  const line = `  |${parts.map((p) => (p && typeof p === 'object' ? p.name || p.id || '' : String(p))).join('|')}`;
  if (/^  \|(t:)?\|?$/.test(line) || /^  \|t:\|/.test(line) || /^  \|(upkeep|)$/.test(line)) return;
  if (/\/100\b/.test(line) && out.length && out[out.length - 1].split('|').slice(0, 3).join('|') ===
    line.split('|').slice(0, 3).join('|')) return;
  out.push(line);
};
const origAdd = B.add;
B.add = function (...parts) {
  proto(this, parts);
  return origAdd.apply(this, parts);
};
const origAddMove = B.addMove;
B.addMove = function (...parts) {
  proto(this, parts);
  return origAddMove.apply(this, parts);
};
const origDebug = B.debug;
B.debug = function (msg) {
  if (show(this)) out.push(`    debug: ${msg}`);
  return origDebug.call(this, msg);
};
const origChain = B.chainModify;
B.chainModify = function (num, den) {
  if (show(this)) {
    const f = Array.isArray(num) ? num[0] / num[1] : den ? num / den : num;
    out.push(`    modifier: ${who(this)} on ${this.event ? this.event.id : '?'} x${+f.toFixed(4)}`);
  }
  return origChain.call(this, num, den);
};
const origRunEvent = B.runEvent;
B.runEvent = function (eventid, target, source, effect, relayVar, ...rest) {
  const r = origRunEvent.call(this, eventid, target, source, effect, relayVar, ...rest);
  if (show(this) && typeof relayVar === 'number' && r !== relayVar) {
    out.push(
      `    event ${eventid}: ${relayVar} -> ${r}  (target ${name(target)}, source ${name(source)}, ${name(effect)})`
    );
  }
  return r;
};

const origRandomTarget = B.getRandomTarget;
B.getRandomTarget = function (pokemon, move) {
  const r = origRandomTarget.call(this, pokemon, move);
  if (show(this)) {
    out.push(`    random target: ${pokemon.name} ${this.dex.moves.get(move).name} -> ${r ? r.name : 'none'}`);
  }
  return r;
};

// Speed sorts that drew (a tie): print the sorted list with its sort keys so
// the tied entries can be compared with the engine's (PE_DEBUG_RES=1 /
// PE_DEBUG_SPEED=1). `*` marks the entries that were in a shuffled tie group.
let drawCount = 0;
const sortItem = (h) => {
  if (h && h.effect !== undefined) {
    const holder = h.effectHolder && h.effectHolder.name ? ` (${h.effectHolder.name})` : '';
    const eff = h.effect ? `${h.effect.effectType || ''} ${h.effect.name}` : '?';
    return `${eff}${holder}  order=${h.order ?? '-'} prio=${h.priority ?? 0} speed=${h.speed ?? 0} sub=${h.subOrder ?? 0}`;
  }
  if (h && h.choice) {
    const mv = h.move ? ` ${h.move.name}` : '';
    return `action ${h.choice}${h.pokemon ? ` ${h.pokemon.name}` : ''}${mv}  order=${h.order ?? '-'} prio=${h.priority ?? 0} speed=${h.speed ?? 0}`;
  }
  if (h && h.side && h.species) return `${h.name}  speed=${h.speed}`;
  return name(h);
};
const origSpeedSort = B.speedSort;
B.speedSort = function (list, comparator) {
  const before = drawCount;
  const r = origSpeedSort.call(this, list, comparator);
  if (show(this) && drawCount > before) {
    const cmp = comparator || this.comparePriority;
    const caller = ((new Error().stack || '').split('\n')[2].match(/at (?:[\w$]+\.)?([\w$<>]+) \(/) || [])[1];
    out.push(`    speedSort tie in ${caller} (${drawCount - before} draws), sorted list (* = tied):`);
    list.forEach((h, i) => {
      const tied = (i > 0 && cmp(list[i - 1], h) === 0) || (i + 1 < list.length && cmp(h, list[i + 1]) === 0);
      out.push(`      ${tied ? '*' : ' '} ${sortItem(h)}`);
    });
  }
  return r;
};

// Tag PRNG draws by purpose (reuses the fuzzer's vocabulary).
const prng = require(path.join(PS_DIR, 'dist', 'sim', 'prng.js'));
let current = null;
const origNext = prng.Gen5RNG.prototype.next;
prng.Gen5RNG.prototype.next = function () {
  drawCount++;
  const v = origNext.call(this);
  if (current && show(current) && current.prng && current.prng.rng === this) {
    const fns = (new Error().stack || '')
      .split('\n')
      .slice(2)
      .map((l) => (l.match(/at (?:[\w$]+\.)?([\w$<>]+) \(/) || [])[1])
      .filter((f) => f && !['next', 'random', 'randomChance'].includes(f));
    out.push(`    draw: ${psDrawPurpose(fns)}  (${fns.slice(0, 9).join(' < ')})`);
  }
  return v;
};
const origStart = B.start;
B.start = function (...a) {
  current = this;
  return origStart.apply(this, a);
};

process.on('exit', () => {
  process.stdout.write(
    `PS trace, seed ${SEED}, turn ${TURN}:\n` + (out.length ? out.join('\n') : '  (turn not reached)') + '\n'
  );
});
console.log = () => {};
console.error = () => {};
// --verbose: no fast path, so the battle is played (and traced) once.
process.argv = [process.argv[0], 'fuzz.js', '--seed', String(SEED), '--count', '1', '--verbose', '--mode', MODE, '--workdir',
  path.join(__dirname, "work", "trace")];
// Loaded as a module, fuzz.js does not run its CLI: call it.
require('./fuzz.js').main();
