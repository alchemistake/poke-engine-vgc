#!/usr/bin/env node
// Move data audit: every move in the fuzz pool, the engine's `Choice` entry
// (src/choices.rs, the champions build) against pokemon-showdown's move data
// (champions dex): base power, accuracy, priority, category, type, recoil,
// drain. A mismatch here is a wrong number in the engine, not a mechanic.
//
//   node fuzz/move_data_audit.js
'use strict';

const fs = require('fs');
const path = require('path');
const { Dex } = require(path.resolve(__dirname, '../../pokemon-showdown/dist/sim'));
const { buildSpeciesPool } = require('./teams.js');

const d = Dex.forFormat('gen9championsvgc2026regmc');
const moves = new Set();
for (const { moves: ms } of buildSpeciesPool(d)) ms.forEach((m) => moves.add(m));

// The engine's own move table (`poke-engine dump-moves`).
const { execFileSync } = require('child_process');
const engine = path.resolve(__dirname, '../target/release/poke-engine');
const table = JSON.parse(execFileSync(engine, ['dump-moves'], { maxBuffer: 1 << 26 }).toString());

const out = [];
for (const id of [...moves].sort()) {
  const mv = d.moves.get(id);
  const e = table[id];
  if (!e) {
    out.push(`${id}: no engine entry`);
    continue;
  }
  const bad = [];
  if (mv.basePower !== e.bp && !mv.basePowerCallback) bad.push(`bp ${e.bp} vs ${mv.basePower}`);
  if (mv.accuracy === true) {
    if (!e.always && e.acc < 100) bad.push(`acc ${e.acc} vs true`);
  } else if (e.always || e.acc !== mv.accuracy) {
    bad.push(`acc ${e.always ? 'true' : e.acc} vs ${mv.accuracy}`);
  }
  if (e.priority !== mv.priority) bad.push(`priority ${e.priority} vs ${mv.priority}`);
  if (e.category.toLowerCase() !== mv.category.toLowerCase()) bad.push(`category ${e.category} vs ${mv.category}`);
  const VAR_TYPE = ['hiddenpower', 'weatherball', 'terrainpulse', 'naturalgift', 'judgment', 'multiattack',
    'technoblast', 'revelationdance', 'terablast', 'ivycudgel', 'ragingbull', 'aurawheel'];
  if (e.type.toLowerCase() !== mv.type.toLowerCase() && !VAR_TYPE.includes(id)) bad.push(`type ${e.type} vs ${mv.type}`);
  const psRec = mv.recoil ? mv.recoil[0] / mv.recoil[1] : 0;
  if (Math.abs(e.recoil - psRec) > 0.01) bad.push(`recoil ${e.recoil} vs ${psRec.toFixed(3)}`);
  const psDrain = mv.drain ? mv.drain[0] / mv.drain[1] : 0;
  if (Math.abs(e.drain - psDrain) > 0.01) bad.push(`drain ${e.drain} vs ${psDrain.toFixed(3)}`);
  // Stat changes: PS `boosts` (target or user) / `self.boosts` against the
  // engine's `boost` / `self_boosts` (either table, as the engine stores some
  // self drops in `boost`). Evasion is set in code for a few moves.
  const KEYS = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'];
  const vec = (b) => (b ? KEYS.map((k) => b[k] || 0).join(',') : null);
  const psB = vec(mv.boosts || (mv.self && mv.self.boosts));
  const peB = new Set([e.boosts, e.self_boosts].filter(Boolean).map((v) => v.join(',')));
  const CODE_BOOSTS = ['doubleteam', 'minimize', 'sweetscent', 'growth', 'curse'];
  if (psB && !peB.has(psB) && !CODE_BOOSTS.includes(id)) bad.push(`boosts [${[...peB].join(' | ') || 'none'}] vs [${psB}]`);
  if (mv.status && !e.status) bad.push(`status none vs ${mv.status}`);
  if (mv.secondary && !mv.secondary.self && !e.secondaries) bad.push(`no secondaries vs ${JSON.stringify(Object.keys(mv.secondary))}`);
  if (bad.length) out.push(`${id}: ${bad.join(', ')}`);
}
console.log(`${out.length} of ${moves.size} pool moves differ`);
for (const l of out) console.log('  ' + l);
