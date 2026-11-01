// Hand-rolled random team generator for the VGC Reg M-C fuzz tester.
//
// Fully random. The pool is wider than the format (WIDE_POOL): every species,
// move and held item that is standard or "Past" in the champions dex (no CAP,
// Custom, LGPE, Future, Z-Crystals, Primal Orbs), with NatDex movepools, so
// the engine is tested beyond Reg M-C. The format's own Species / Item Clause
// still hold. Species that need an item (Zacian-Crowned, plates, ...) hold it.
// With WIDE_POOL off, teams are ones the format's validator accepts
// (pokemon-showdown `TeamValidator`, Flat Rules): species (every team-legal
// species and forme), stat points (66 total, 0-32 per stat, champions
// semantics), nature, ability (within the species' pool), moves (up to 4
// distinct, from the species' move pool), gender, and a held item (Item
// Clause: no item twice per team). Mega formes (-Mega, -Mega-X/Y/Z) are picks
// of their own: picking one puts its base species on the sheet holding that
// Mega Stone. No other pick ever holds a Mega Stone.
//
// All randomness comes from the caller-provided controller PRNG, so a given
// tester seed reproduces the exact same teams.

'use strict';

const path = require('path');

const STAT_KEYS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const IVS = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const FORMAT = 'gen9championsvgc2026regmc';
// Champions stat points (sim/team-validator.ts validateStats, ruleTable.evLimit).
const SP_TOTAL = 66;
const SP_MAX = 32;
// Chance a pokemon holds no item (Acrobatics, Knock Off / Trick no-ops, ...).
const NO_ITEM_CHANCE = 20; // 1 in 20
// Widen the pool past Reg M-C (see the header). A change here is a new harness:
// bump fuzz.js SEED_SPACE with it.
const WIDE_POOL = true;
const wideOk = (e) => e.exists && (!e.isNonstandard || e.isNonstandard === 'Past');

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

let VALIDATOR = null;
function validator() {
  if (!VALIDATOR) {
    const { TeamValidator, Dex } = require(path.resolve(__dirname, '..', '..', 'pokemon-showdown', 'dist', 'sim'));
    VALIDATOR = new TeamValidator(Dex.formats.get(FORMAT));
  }
  return VALIDATOR;
}

// A set is legal for the format (species, forme, ability, moves, item, stats).
function setProblems(set) {
  return validator().validateSet(JSON.parse(JSON.stringify(set)), {}) || null;
}

let POOL = null;

/**
 * Build the species pool once from the FORMAT's own species table
 * (`Dex.forFormat(FORMAT)`, i.e. data/mods/champions/formats-data.ts): every
 * species the validator accepts on a team sheet, plus one pick per Mega
 * forme of those species. Other battle-only formes (Aegislash-Blade,
 * Palafin-Hero, ...) are reached in battle only. Moves come from
 * `getMovePool` (forme learnsets fall back to the base species, like the
 * validator).
 *
 * @param modDex Dex.forFormat('gen9championsvgc2026regmc')
 * @returns [{ sp, sheet, moves, abilities, stone }]: `sp` is the pick (a Mega
 *   forme for a Mega pick), `sheet` the species on the team sheet, `abilities`
 *   the sheet species' legal abilities, `stone` the forced Mega Stone or null.
 */
function buildWideSpeciesPool(modDex) {
  const stonesFor = new Map();
  for (const it of modDex.items.all()) {
    if (!wideOk(it) || !it.megaStone) continue;
    for (const base of Object.keys(it.megaStone)) {
      if (!stonesFor.has(base)) stonesFor.set(base, []);
      stonesFor.get(base).push(it.name);
    }
  }
  const pool = [];
  for (const sp of modDex.species.all()) {
    if (!wideOk(sp) || sp.battleOnly) continue;
    if (!sp.abilities || Object.keys(sp.abilities).length === 0) continue;
    const moves = [...modDex.species.getMovePool(sp.id, true)]
      .filter((m) => { const mv = modDex.moves.get(m); return wideOk(mv) && !mv.isZ && !mv.isMax; })
      .sort();
    if (moves.length === 0) continue;
    const abilities = [...new Set(Object.values(sp.abilities))]
      .filter((a) => wideOk(modDex.abilities.get(a)));
    if (abilities.length === 0) continue;
    // A forme that needs its item (Zacian-Crowned, Arceus plates, ...) holds it.
    const required = sp.requiredItems || (sp.requiredItem ? [sp.requiredItem] : null);
    if (required) {
      // Arceus formes also accept a Z-Crystal: only the Plate (no Z-Moves here).
      for (const it of required.slice().sort()) {
        if (modDex.items.get(it).zMove) continue;
        pool.push({ sp, sheet: sp, moves, abilities, stone: it });
      }
      continue;
    }
    pool.push({ sp, sheet: sp, moves, abilities, stone: null });
    for (const stone of (stonesFor.get(sp.name) || []).sort()) {
      const mega = modDex.species.get(modDex.items.get(stone).megaStone[sp.name]);
      if (!mega.exists) continue;
      pool.push({ sp: mega, sheet: sp, moves, abilities, stone });
    }
  }
  return pool;
}

function buildSpeciesPool(modDex) {
  if (POOL) return POOL;
  if (WIDE_POOL) {
    POOL = buildWideSpeciesPool(modDex);
    return POOL;
  }
  const V = validator();
  const stonesFor = new Map(); // species name -> [mega stone names]
  for (const it of modDex.items.all()) {
    if (it.isNonstandard || !it.megaStone) continue;
    for (const base of Object.keys(it.megaStone)) {
      if (!stonesFor.has(base)) stonesFor.set(base, []);
      stonesFor.get(base).push(it.name);
    }
  }
  const pool = [];
  for (const sp of modDex.species.all()) {
    if (sp.isNonstandard || sp.battleOnly) continue;
    if (!sp.abilities || Object.keys(sp.abilities).length === 0) continue;
    if (V.checkSpecies({ species: sp.name, name: sp.name, moves: [], ability: '' }, sp, sp, {})) continue;
    const moves = [...modDex.species.getMovePool(sp.id)].filter((m) => !V.checkCanLearn(modDex.moves.get(m), sp));
    if (moves.length === 0) continue;
    // Abilities the format allows on this species (Battle Bond Greninja is banned).
    const abilities = [...new Set(Object.values(sp.abilities))].filter(
      (a) => !setProblems({ species: sp.name, ability: a, moves: [moves[0]], nature: 'Hardy',
        evs: { hp: 1, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, ivs: IVS, level: 50, item: '' })
    );
    if (abilities.length === 0) continue;
    pool.push({ sp, sheet: sp, moves, abilities, stone: null });
    for (const stone of (stonesFor.get(sp.name) || []).sort()) {
      const mega = modDex.species.get(modDex.items.get(stone).megaStone[sp.name]);
      pool.push({ sp: mega, sheet: sp, moves, abilities, stone });
    }
  }
  if (pool.length === 0) throw new Error('empty species pool');
  POOL = pool;
  return pool;
}

let ITEMS = null;

/** Every item the format allows, minus Mega Stones (those go only to their species). */
function buildItemPool(modDex) {
  if (ITEMS) return ITEMS;
  ITEMS = modDex.items.all()
    .filter((it) => (WIDE_POOL ? wideOk(it) : !it.isNonstandard) && !it.megaStone && !it.zMove && !it.isPrimalOrb)
    .map((it) => it.name)
    .sort();
  return ITEMS;
}

/** 66 stat points, each stat capped at 32 (champions / VGC sheet rules). */
function rollEvs(controller) {
  const evs = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
  let left = SP_TOTAL;
  while (left > 0) {
    const s = STAT_KEYS[controller.random(STAT_KEYS.length)];
    if (evs[s] < SP_MAX) {
      evs[s] += 1;
      left -= 1;
    }
  }
  return evs;
}

/** Generate one random 6-pokemon team. `modDex` = Dex.forFormat(FORMAT). */
function genTeam(controller, modDex) {
  const pool = buildSpeciesPool(modDex);
  const items = buildItemPool(modDex);
  const natures = modDex.natures.all();
  const team = [];
  const usedBase = new Set(); // Species Clause
  const usedItems = new Set(); // Item Clause = 1
  while (team.length < 6) {
    const { sheet, moves, abilities, stone } = pool[controller.random(pool.length)];
    if (usedBase.has(sheet.baseSpecies)) continue;

    // Up to 4 distinct random moves from the move pool.
    const remaining = moves.slice();
    const picked = [];
    while (picked.length < 4 && remaining.length > 0) {
      picked.push(remaining.splice(controller.random(remaining.length), 1)[0]);
    }

    const ability = abilities[controller.random(abilities.length)];

    // Random gender, respecting species gender locks.
    const gender = sheet.gender || ['M', 'F', 'N'][controller.random(3)];

    // Held item: a Mega pick holds its Mega Stone; any other pick a random
    // unused regular item, sometimes none.
    let item = '';
    if (stone) {
      item = stone;
    } else if (controller.random(NO_ITEM_CHANCE) !== 0) {
      const free = items.filter((it) => !usedItems.has(it));
      item = free[controller.random(free.length)];
    }

    const set = {
      species: sheet.name,
      level: 50,
      gender,
      shiny: false,
      nature: natures[controller.random(natures.length)].name,
      ability,
      evs: rollEvs(controller),
      ivs: { ...IVS },
      moves: picked,
      item,
    };
    const problems = WIDE_POOL ? null : setProblems(set);
    if (problems) throw new Error(`genTeam made an illegal set:\n${problems.join('\n')}\n${JSON.stringify(set)}`);
    usedBase.add(sheet.baseSpecies);
    if (item) usedItems.add(item);
    team.push(set);
  }
  return team;
}

/** Throws unless the whole team passes the format's validator (Species / Item Clause). */
function assertLegalTeam(team) {
  const problems = validator().validateTeam(JSON.parse(JSON.stringify(team)));
  if (problems) throw new Error(`illegal team:\n${problems.join('\n')}`);
}

module.exports = {
  genTeam, buildSpeciesPool, buildItemPool, effectiveStats, assertLegalTeam, STAT_KEYS,
};
