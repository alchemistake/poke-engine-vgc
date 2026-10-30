#!/usr/bin/env node
// Time pokemon-showdown on the benchmark games: replay each recorded game
// (new Battle + the same accepted choose() calls), no snapshots or tracing.
// Prints the best of --reps runs, in microseconds for all games.
// Usage: node bench/ps_bench.js [reps=5]
const path = require('path');
const PS_DIR = path.resolve(__dirname, '..', '..', 'pokemon-showdown');
const Sim = require(path.join(PS_DIR, 'dist', 'sim'));
const games = require('./ps_games.json');
const FORMAT = Sim.Dex.formats.get('gen9championsvgc2026regmc');
const reps = +(process.argv[2] || 5);

function play(g) {
  const b = new Sim.Battle({
    format: FORMAT,
    seed: g.seed,
    p1: { name: 'A', team: g.teamA },
    p2: { name: 'B', team: g.teamB },
  });
  b.strictChoices = true;
  for (const [sid, ...inputs] of g.choices) {
    if (sid === '*') b.makeChoices(...inputs);
    else b.choose(sid, inputs[0]);
  }
  const w = b.winner === 'p1' ? 'A' : b.winner === 'p2' ? 'B' : b.winner || 'draw';
  if (!b.ended || w !== g.winner) throw new Error(`replay diverged (winner ${w} vs ${g.winner})`);
}

let best = Infinity;
for (let r = 0; r < reps; r++) {
  const t0 = process.hrtime.bigint();
  for (const g of games) play(g);
  const us = Number(process.hrtime.bigint() - t0) / 1000;
  best = Math.min(best, us);
  console.error(`rep ${r + 1}: ${Math.round(us)} us`);
}
console.log(Math.round(best));
