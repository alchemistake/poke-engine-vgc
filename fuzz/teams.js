// Hand-rolled random team generator for the VGC Reg M-C fuzz tester.
//
// Fully random: species (from the legal pool), stat points (28 pts, 0-20 per
// stat, champions semantics), nature, ability (within the species' pool),
// moves (4 distinct, from the species' gen-9 learnset pool), gender.
// No items (the previous port has no item support to validate against).
//
// All randomness comes from the caller-provided controller PRNG, so a given
// tester seed reproduces the exact same teams.

'use strict';

const STAT_KEYS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const IVS = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };

/**
 * Effective level-50 stats from a generated set, matching the champions mod
 * exactly (data/mods/champions/scripts.ts, no Level Clause Mod):
 *   hp:    base + statPoints + 75
 *   other: base + statPoints + 20
 * then nature (+10% / -10%, 16-bit truncated like PS).
 */
function effectiveStats(set, Dex) {
  const base = Dex.species.get(set.species).baseStats;
  const nature = Dex.natures.get(set.nature);
  const out = {};
  for (const k of STAT_KEYS) {
    let stat = base[k] + set.evs[k] + (k === 'hp' ? 75 : 20);
    if (nature.plus === k) stat = Math.trunc(Math.trunc(stat * 110, 16) / 100);
    else if (nature.minus === k) stat = Math.trunc(Math.trunc(stat * 90, 16) / 100);
    out[k] = stat;
  }
  return out;
}

let POOL = null;

/**
 * Build the species pool once from the FORMAT's own species table
 * (`Dex.forFormat(FORMAT)`, i.e. data/mods/champions/formats-data.ts):
 * excludes CAP/fan-made, Past-gen and other `isNonstandard` species, plus
 * Illegal / Mythical / Restricted Legendary tiers — exactly what the format
 * allows. Also requires 4+ gen-9 learnset moves in the mod's learnset data.
 *
 * @param modDex Dex.forFormat('gen9championsvgc2026regmc')
 */
function buildSpeciesPool(modDex) {
  if (POOL) return POOL;
  const pool = [];
  for (const sp of modDex.species.all()) {
    if (sp.isNonstandard) continue; // no CAP / fan-made, Past, Hidden, ...
    if (['Illegal', 'Mythical', 'Restricted Legendary'].includes(sp.tier)) continue;
    if (!sp.abilities || Object.keys(sp.abilities).length === 0) continue;
    const entry = modDex.data.Learnsets[sp.id];
    const ls = entry && entry.learnset;
    if (!ls) continue;
    const moves = Object.keys(ls).filter((m) => {
      const e = ls[m];
      return Array.isArray(e) && e.some((x) => x.startsWith('9'));
    });
    if (moves.length < 4) continue;
    pool.push({ sp, moves });
  }
  if (pool.length === 0) throw new Error('empty species pool');
  POOL = pool;
  return pool;
}

/** 28 stat points, each stat capped at 20 (champions / VGC sheet rules). */
function rollEvs(controller) {
  const evs = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
  let left = 28;
  while (left > 0) {
    const s = STAT_KEYS[controller.random(STAT_KEYS.length)];
    if (evs[s] < 20) {
      evs[s] += 1;
      left -= 1;
    }
  }
  return evs;
}

/** Generate one random 6-pokemon team. `modDex` = Dex.forFormat(FORMAT). */
function genTeam(controller, modDex) {
  const pool = buildSpeciesPool(modDex);
  const natures = modDex.natures.all();
  const team = [];
  for (let i = 0; i < 6; i++) {
    const { sp, moves } = pool[controller.random(pool.length)];

    // 4 distinct random moves from the learnset pool.
    const remaining = moves.slice();
    const picked = [];
    for (let m = 0; m < 4; m++) {
      const idx = controller.random(remaining.length);
      picked.push(remaining.splice(idx, 1)[0]);
    }

    // Random ability within the species' pool (base / secondary / hidden).
    const abilityKeys = Object.keys(sp.abilities);
    const ability = sp.abilities[abilityKeys[controller.random(abilityKeys.length)]];

    // Random gender, respecting species gender locks.
    const gender = sp.gender || ['M', 'F', 'N'][controller.random(3)];

    team.push({
      species: sp.name,
      level: 50,
      gender,
      shiny: false,
      nature: natures[controller.random(natures.length)].name,
      ability,
      evs: rollEvs(controller),
      ivs: { ...IVS },
      moves: picked,
      item: '',
    });
  }
  return team;
}

module.exports = { genTeam, buildSpeciesPool, effectiveStats, STAT_KEYS };
