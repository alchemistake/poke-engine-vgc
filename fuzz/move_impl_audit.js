#!/usr/bin/env node
// Move coverage audit: every move in the fuzz pool whose PS data has custom
// logic (a callback, a condition, a volatile / side / field effect) but that the
// doubles engine never names (`Choices::<ID>` in src/doubles/*.rs, generated
// tables excluded). Such a move most likely runs as a plain damage / no-op move.
//
//   node fuzz/move_impl_audit.js            # list
//   node fuzz/move_impl_audit.js --json     # machine-readable
'use strict';

const fs = require('fs');
const path = require('path');
const { Dex } = require(path.resolve(__dirname, '../../pokemon-showdown/dist/sim'));
const { buildSpeciesPool } = require('./teams.js');

const d = Dex.forFormat('gen9championsvgc2026regmc');
const pool = new Set();
for (const { moves } of buildSpeciesPool(d)) moves.forEach((m) => pool.add(m));

const GENERATED = new Set(['move_data.rs', 'move_targets.rs', 'event_handlers.rs', 'species_types.rs', 'species_weights.rs', 'ability_flags.rs', 'item_data.rs', 'megas.rs']);
const dir = path.resolve(__dirname, '../src/doubles');
let src = '';
for (const f of fs.readdirSync(dir)) if (f.endsWith('.rs') && !GENERATED.has(f)) src += fs.readFileSync(path.join(dir, f), 'utf8');
src += fs.readFileSync(path.resolve(__dirname, '../src/genx/mechanics.rs'), 'utf8');
const lines = src.split('\n');

// Fields the engine handles from the move table (choices.rs) generically are
// not listed; these need code of their own.
const FIELDS = ['slotCondition', 'callsMove', 'stealsBoosts', 'overrideOffensivePokemon'];
const out = [];
for (const id of [...pool].sort()) {
  const m = d.moves.get(id);
  const hooks = Object.keys(m).filter((k) => typeof m[k] === 'function');
  // A volatile the engine names as `PokemonVolatileStatus::<ID>` counts as handled.
  if (m.condition && !(m.volatileStatus && new RegExp(`PokemonVolatileStatus::${m.volatileStatus.toUpperCase()}\\b`).test(src))
      && !m.sideCondition && !m.pseudoWeather && !m.terrain && !m.weather) hooks.push('condition');
  if (m.secondary && Object.keys(m.secondary).some((k) => typeof m.secondary[k] === 'function')) hooks.push('secondary.onHit');
  if (m.self && Object.keys(m.self).some((k) => typeof m.self[k] === 'function')) hooks.push('self.onHit');
  for (const f of FIELDS) if (m[f] !== undefined && m[f] !== false) hooks.push(f);
  if (!hooks.length) continue;
  // A name inside a long list (`| Choices::X` lines, or a line naming 3+
  // moves) is a flag table, not handling of its own.
  const re = new RegExp(`Choices::${id.toUpperCase()}\\b`);
  const named = lines.some((l) => re.test(l) && !/^\s*\|\s*Choices::\w+\s*(=>.*)?$/.test(l) && (l.match(/Choices::/g) || []).length < 3);
  if (!named) out.push({ id, hooks });
}
if (process.argv.includes('--json')) console.log(JSON.stringify(out));
else {
  console.log(`${out.length} of ${pool.size} pool moves have custom PS logic and no doubles handling`);
  for (const o of out) console.log(`  ${o.id}: ${o.hooks.join(', ')}`);
}
