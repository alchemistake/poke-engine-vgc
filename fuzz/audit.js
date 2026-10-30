#!/usr/bin/env node
// PS-side audits for mechanic CLASSES (champions format, fuzz species pool).
// Use them when a fix is "X was missing from list Y": check every entry of Y.
//
//   node fuzz/audit.js residual   # every Residual handler: volatiles with a
//                                 # `duration`, onResidual (order / subOrder)
//   node fuzz/audit.js batonpass  # volatiles Baton Pass copies vs noCopy
//
// Compare the output with the engine: `collect_residual_handlers` and the
// Baton Pass `copyVolatileFrom` block in src/doubles/battle.rs.
const path = require('path');
const fs = require('fs');
const { Dex } = require(path.resolve(__dirname, '../../pokemon-showdown/dist/sim'));
const { buildSpeciesPool } = require('./teams.js');
const d = Dex.forFormat('gen9championsvgc2026regmc');
const pool = buildSpeciesPool(d);
const moves = new Set();
const abil = new Set();
for (const { sp, moves: ms } of pool) {
  ms.forEach((m) => moves.add(m));
  Object.values(sp.abilities).forEach((a) => abil.add(d.abilities.get(a).id));
}
const kindOf = (m) => (m.volatileStatus ? 'volatile' : m.sideCondition ? 'side' : m.weather ? 'weather' :
  m.terrain ? 'terrain' : m.pseudoWeather ? 'field' : m.slotCondition ? 'slot' : 'self/other');
const residualKeys = (c) => {
  const k = [];
  if (c.duration) k.push(`duration=${c.duration}`);
  for (const key of Object.keys(c)) {
    if (/Residual$/.test(key)) {
      k.push(key + (c[key + 'Order'] ? ` order=${c[key + 'Order']}` : '') +
        (c[key + 'SubOrder'] ? ` sub=${c[key + 'SubOrder']}` : ''));
    }
  }
  return k;
};
const CONDS = ['brn', 'par', 'slp', 'frz', 'psn', 'tox', 'confusion', 'flinch', 'trapped', 'partiallytrapped',
  'lockedmove', 'twoturnmove', 'mustrecharge', 'futuremove', 'stall', 'rolloutstorage'];
const mode = process.argv[2];
const out = [];
if (mode === 'residual') {
  for (const id of moves) {
    const m = d.moves.get(id);
    if (!m.condition) continue;
    const k = residualKeys(m.condition);
    if (k.length) out.push(`move ${id} [${kindOf(m)}] ${k.join(', ')}`);
  }
  for (const id of abil) {
    const a = d.abilities.get(id);
    if (a.onResidual) out.push(`ability ${id} onResidual order=${a.onResidualOrder ?? '-'} sub=${a.onResidualSubOrder ?? '-'}`);
  }
  for (const id of CONDS) {
    const c = d.conditions.get(id);
    if (!c.exists) continue;
    const k = residualKeys(c);
    if (k.length) out.push(`cond ${id} ${k.join(', ')}`);
  }
  console.log(out.sort().join('\n'));
} else if (mode === 'batonpass') {
  const ids = new Set();
  const PS = path.resolve(__dirname, '../../pokemon-showdown/data');
  const src = ['moves.ts', 'conditions.ts', 'abilities.ts'].map((f) => fs.readFileSync(path.join(PS, f), 'utf8')).join('\n');
  for (const id of moves) {
    const m = d.moves.get(id);
    if (m.volatileStatus) ids.add(m.volatileStatus);
    if (m.self && m.self.volatileStatus) ids.add(m.self.volatileStatus);
    if (m.condition && kindOf(m) === 'self/other') ids.add(id);
    (m.secondaries || []).forEach((s) => s.volatileStatus && ids.add(s.volatileStatus));
  }
  for (const m of src.matchAll(/addVolatile\('(\w+)'/g)) ids.add(m[1]);
  const copy = [];
  const no = [];
  for (const id of [...ids].sort()) {
    const c = d.conditions.get(id);
    if (!c.exists) continue;
    const mv = d.moves.get(id);
    if (mv.exists && kindOf(mv) !== 'volatile' && kindOf(mv) !== 'self/other') continue;
    (c.noCopy ? no : copy).push(id + (c.duration ? `(d${c.duration})` : '') + (c.onCopy ? '[onCopy]' : ''));
  }
  console.log('copied by Baton Pass:', copy.join(' '));
  console.log('noCopy:', no.join(' '));
} else {
  console.error('usage: node fuzz/audit.js residual|batonpass');
  process.exit(2);
}
