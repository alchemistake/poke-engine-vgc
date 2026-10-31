#!/usr/bin/env node
// Generate the fixed benchmark game set: seeds [from, to) of the fuzz
// harness, played once in pokemon-showdown to record every choice.
//   bench/games.json    engine batch input (same format as fuzz --batch)
//   bench/ps_games.json PS replay script: seed, teams, choose() calls
// Usage: node bench/gen_games.js [from=0] [to=100] [prefix=bench/]
// (a prefix other than bench/ writes <prefix>games.json / <prefix>ps_games.json,
// e.g. a larger test set)
const fs = require('fs');
const path = require('path');
const F = require('./load_fuzz.js');

const from = +(process.argv[2] || 0);
const to = +(process.argv[3] || 100);
const prefix = process.argv[4] || path.join(__dirname, path.sep);

const Battle = F.ps().Battle;
const origChoose = Battle.prototype.choose;
const origMake = Battle.prototype.makeChoices;
let log = null;
Battle.prototype.choose = function (sid, input) {
  const r = origChoose.call(this, sid, input);
  if (log) log.push([sid, input]); // only accepted choices
  return r;
};
Battle.prototype.makeChoices = function (...inputs) {
  if (log) log.push(['*', ...inputs]);
  const saved = log;
  log = null; // makeChoices calls choose() internally
  try {
    return origMake.apply(this, inputs);
  } finally {
    log = saved;
  }
};

const engine = [];
const psGames = [];
for (let s = from; s < to; s++) {
  const { controller, battleSeed, teamA, teamB } = F.makeTest(s);
  log = [];
  const meta = F.stepPSBattle(battleSeed, teamA, teamB, controller, () => {}, () => {});
  psGames.push({ seed: battleSeed, teamA, teamB, choices: log, winner: meta.winner, turns: meta.turns.length });
  log = null;
  engine.push(F.engineInput(battleSeed, meta, meta.turns, meta.turns.length));
}
fs.writeFileSync(prefix + 'games.json', JSON.stringify(engine));
fs.writeFileSync(prefix + 'ps_games.json', JSON.stringify(psGames));
const turns = psGames.reduce((a, g) => a + g.turns, 0);
console.log(`wrote ${psGames.length} games (${turns} turns)`);
