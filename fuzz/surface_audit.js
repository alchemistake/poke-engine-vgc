#!/usr/bin/env node
// Surface audit: everything the fuzzer can put in a battle (fuzz/teams.js
// pools: species, the formes they reach in battle, abilities, moves, held
// items, Mega Stones) checked against what the engine knows, from its source.
// Run it before a fuzz loop: an entry listed here fails the fuzzer for a
// known reason, so implement it first instead of finding it seed by seed.
//
//   node fuzz/surface_audit.js            # summary + every missing entry
//   node fuzz/surface_audit.js --json     # machine-readable
//
// Checks (static, no battle is run):
//   species  pool species and in-battle formes (Mega picks, battle-only formes)
//            with no `species_data` row (src/doubles/species_types.rs)
//   moves    pool moves missing from the `Choices` enum (src/choices.rs) or
//            from `ps_move_target` (src/doubles/move_targets.rs)
//   abilities  pool abilities missing from the `Abilities` enum
//            (src/genx/abilities.rs), or never referenced by the doubles
//            engine (no `Abilities::X` and no "id" string in src/doubles)
//   items    pool items (Mega Stones included) missing from the `Items` enum
//            (src/genx/items.rs), or never referenced by the doubles engine
// "Never referenced" is a strong hint, not proof: a few effects run from
// generic data. Each missing entry lists how many pool species can use it.
'use strict';

const fs = require('fs');
const path = require('path');
const { Dex } = require(path.resolve(__dirname, '../../pokemon-showdown/dist/sim'));
const { buildSpeciesPool, buildItemPool } = require('./teams.js');

const ROOT = path.resolve(__dirname, '..');
const read = (f) => fs.readFileSync(path.join(ROOT, f), 'utf8');
const d = Dex.forFormat('gen9championsvgc2026regmc');
const pool = buildSpeciesPool(d);

// Variant names of `define_enum_with_from_str! { ... Name { A, B, ... } }`.
function enumIds(file, name) {
  const src = read(file);
  const start = src.indexOf(`${name} {`);
  if (start < 0) throw new Error(`enum ${name} not found in ${file}`);
  const body = src.slice(start + name.length + 2, src.indexOf('}', start));
  return new Set(body.split(',').map((s) => s.replace(/\/\/.*$/gm, '').trim().toLowerCase()).filter(Boolean));
}

// The doubles engine, plus the shared genx helpers it calls
// (`crate::engine::mechanics`).
const doublesSrc = fs.readdirSync(path.join(ROOT, 'src/doubles'))
  .filter((f) => f.endsWith('.rs') && f !== 'event_handlers.rs' && f !== 'species_types.rs' &&
    f !== 'species_weights.rs' && f !== 'move_targets.rs')
  .map((f) => read(`src/doubles/${f}`)).concat(read('src/genx/mechanics.rs')).join('\n');
const referenced = (prefix, id) =>
  new RegExp(`${prefix}::${id.toUpperCase()}\\b`).test(doublesSrc) || doublesSrc.includes(`"${id}"`);

const speciesRows = new Set([...read('src/doubles/species_types.rs').matchAll(/^\s+"([a-z0-9]+)" =>/gm)].map((m) => m[1]));
const moveTargets = new Set([...read('src/doubles/move_targets.rs').matchAll(/^\s+"([a-z0-9]+)" =>/gm)].map((m) => m[1]));
const choices = enumIds('src/choices.rs', 'Choices');
const abilitiesEnum = enumIds('src/genx/abilities.rs', 'Abilities');
const itemsEnum = enumIds('src/genx/items.rs', 'Items');

// Usage counts: how many pool species can use each move / ability / item.
const users = { species: new Map(), moves: new Map(), abilities: new Map(), items: new Map() };
const bump = (m, id) => m.set(id, (m.get(id) || 0) + 1);
const formes = new Set();
for (const { sp, sheet, moves, abilities, stone } of pool) {
  if (stone) {
    formes.add(sp.id); // Mega pick: the forme it Mega Evolves into
    bump(users.items, d.items.get(stone).id);
    for (const a of Object.values(sp.abilities)) bump(users.abilities, d.abilities.get(a).id);
    continue;
  }
  bump(users.species, sheet.id);
  for (const m of moves) bump(users.moves, m);
  for (const a of abilities) bump(users.abilities, d.abilities.get(a).id);
  // Formes reached in battle from this species (Aegislash-Blade, ...).
  for (const f of sp.formes || []) {
    const fs_ = d.species.get(f);
    if (fs_.exists && !fs_.isNonstandard && fs_.battleOnly) formes.add(fs_.id);
  }
}
for (const it of buildItemPool(d)) users.items.set(d.items.get(it).id, pool.length);

const report = {
  species: [...users.species.keys()].filter((id) => !speciesRows.has(id)),
  formes: [...formes].filter((id) => !speciesRows.has(id)).sort(),
  movesUnknown: [...users.moves.keys()].filter((id) => !choices.has(id) || !moveTargets.has(id)),
  abilitiesUnknown: [...users.abilities.keys()].filter((id) => !abilitiesEnum.has(id)),
  abilitiesUnreferenced: [...users.abilities.keys()].filter((id) => abilitiesEnum.has(id) && !referenced('Abilities', id)),
  itemsUnknown: [...users.items.keys()].filter((id) => !itemsEnum.has(id)),
  itemsUnreferenced: [...users.items.keys()].filter((id) => itemsEnum.has(id) && !referenced('Items', id)),
};
const byUse = (kind) => (a, b) => (users[kind].get(b) || 0) - (users[kind].get(a) || 0) || a.localeCompare(b);
report.movesUnknown.sort(byUse('moves'));
report.abilitiesUnknown.sort(byUse('abilities'));
report.abilitiesUnreferenced.sort(byUse('abilities'));
report.itemsUnknown.sort();
report.itemsUnreferenced.sort();

if (process.argv.includes('--json')) {
  console.log(JSON.stringify(report, null, 2));
} else {
  const totals = {
    species: users.species.size, formes: formes.size, movesUnknown: users.moves.size,
    abilitiesUnknown: users.abilities.size, abilitiesUnreferenced: users.abilities.size,
    itemsUnknown: users.items.size, itemsUnreferenced: users.items.size,
  };
  const label = (kind, id) => {
    const k = kind.startsWith('moves') ? 'moves' : kind.startsWith('abil') ? 'abilities' : null;
    return k ? `${id}(${users[k].get(id)})` : id;
  };
  for (const [k, v] of Object.entries(report)) {
    console.log(`${k}: ${v.length} of ${totals[k]}`);
    if (v.length) console.log('  ' + v.map((id) => label(k, id)).join(', '));
  }
}
const missing = Object.values(report).reduce((n, v) => n + v.length, 0);
process.exit(missing ? 1 : 0);
