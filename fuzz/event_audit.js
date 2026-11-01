#!/usr/bin/env node
// Event coverage audit: checks the engine's typed battle events
// (`DoublesState::events`) against the state they describe. Use it as the
// engine of a fuzz run:
//
//   node fuzz/fuzz.js --seed 0 --count 500 --engine fuzz/event_audit.js
//
// It runs the real engine with `events: true`, checks every turn, appends
// each problem to fuzz/work/event_audit.log and hands fuzz.js the output
// without the events (so the parity check is unchanged). Checks, per
// pokemon and turn:
//   - damage minus heal events equals the HP lost;
//   - a pokemon that faints has a faint event, and only then;
//   - a pokemon that comes into a slot has a switch or drag event for it.
// `node fuzz/event_audit.js --report` prints the log grouped by problem.
// Clear fuzz/work/event_audit.{log,count} before a run.
'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');

const ENGINE = path.resolve(__dirname, '../target/release/poke-engine');
const LOG = path.resolve(__dirname, 'work/event_audit.log');

function checkBattle(seed, res, out) {
  const turns = res.turns || [];
  for (let t = 1; t < turns.length; t++) {
    const prev = turns[t - 1];
    const cur = turns[t];
    const evs = cur.events || [];
    for (const side of ['a', 'b']) {
      cur[side].forEach((p, i) => {
        const q = prev[side][i];
        if (!q) return;
        const mine = evs.filter((e) => e.target_side === side && e.target === i);
        const dmg = mine.filter((e) => e.kind === 'damage').reduce((s, e) => s + e.value, 0);
        const heal = mine.filter((e) => e.kind === 'heal').reduce((s, e) => s + e.value, 0);
        const lost = q.hp - p.hp;
        if (dmg - heal !== lost) {
          out.push(`seed ${seed} turn ${cur.turn} ${side}${i} ${p.species}: hp ${q.hp} -> ${p.hp}, events damage ${dmg} heal ${heal}`);
        }
        const faints = evs.filter((e) => e.kind === 'faint' && e.side === side && e.actor === i).length;
        const fainted = q.hp > 0 && p.hp <= 0;
        if (fainted && faints !== 1) {
          out.push(`seed ${seed} turn ${cur.turn} ${side}${i} ${p.species}: fainted, ${faints} faint events`);
        }
        if (!fainted && faints > 0 && q.hp > 0) {
          out.push(`seed ${seed} turn ${cur.turn} ${side}${i} ${p.species}: faint event but hp ${p.hp}`);
        }
      });
      // Slot changes: the pokemon now in a slot came in by a switch / drag
      // (Ally Switch only moves an active pokemon to the other slot).
      for (let slot = 0; slot < 2; slot++) {
        const now = cur[side].findIndex((p) => p.active_slot === slot);
        const before = prev[side].findIndex((p) => p.active_slot === slot);
        if (now < 0 || now === before) continue;
        if (prev[side][now] && prev[side][now].active_slot != null) continue;
        // Ally Switch moves the pokemon after it came in: any slot counts then.
        const ally = evs.some((e) => e.kind === 'move' && e.move === 'allyswitch' && e.side === side);
        const sw = evs.filter((e) => (e.kind === 'switch' || e.kind === 'drag') && e.side === side &&
          (e.value === slot || (ally && e.actor === now)));
        const last = sw[sw.length - 1];
        if (!last || last.actor !== now) {
          const kinds = evs.map((e) => e.kind + (e.move ? ':' + e.move : '') + (e.side ? ':' + e.side + e.actor : '')).join(' ');
          out.push(`seed ${seed} turn ${cur.turn} ${side} slot ${slot}: ${cur[side][now].species} came in, switch events ${JSON.stringify(sw.map((e) => e.actor))} | ${kinds}`);
        }
      }
    }
  }
}

function strip(res) {
  if (res && res.turns) for (const t of res.turns) delete t.events;
  return res;
}

function wrapper(argv) {
  const ii = argv.indexOf('--input');
  const oi = argv.indexOf('--output');
  const batch = argv.includes('--batch');
  // `--batch --check` (fuzz.js fast path): entries are `{input, expect,
  // winner}`. Audit a plain `--batch` run of the inputs, then hand fuzz.js
  // the real `--check` result.
  const check = argv.includes('--check');
  const input = JSON.parse(fs.readFileSync(argv[ii + 1], 'utf8'));
  const inputs = check ? input.map((e) => e.input) : batch ? input : [input];
  for (const x of inputs) x.events = true;
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'evaudit-'));
  const inPath = path.join(tmp, 'in.json');
  const outPath = path.join(tmp, 'out.json');
  fs.writeFileSync(inPath, JSON.stringify(batch ? inputs : inputs[0]));
  const args = argv.map((a, k) => (k === ii + 1 ? inPath : k === oi + 1 ? outPath : a)).filter((a) => a !== '--check');
  const r = spawnSync(ENGINE, args, { encoding: 'utf8', maxBuffer: 1 << 28 });
  process.stderr.write(r.stderr || '');
  if (r.status !== 0) {
    process.stdout.write(r.stdout || '');
    process.exit(r.status === null ? 1 : r.status);
  }
  const out = JSON.parse(fs.readFileSync(outPath, 'utf8'));
  const results = batch ? out : [out];
  const problems = [];
  results.forEach((res, k) => checkBattle(inputs[k].seed.join('-'), res, problems));
  fs.mkdirSync(path.dirname(LOG), { recursive: true });
  fs.appendFileSync(LOG.replace(/\.log$/, '.count'), `${results.length}\n`);
  if (problems.length) {
    fs.mkdirSync(path.dirname(LOG), { recursive: true });
    fs.appendFileSync(LOG, problems.join('\n') + '\n');
  }
  if (check) {
    const rc = spawnSync(ENGINE, argv, { encoding: 'utf8', maxBuffer: 1 << 28 });
    process.stderr.write(rc.stderr || '');
    process.stdout.write(rc.stdout || '');
    fs.rmSync(tmp, { recursive: true, force: true });
    process.exit(rc.status === null ? 1 : rc.status);
  }
  results.forEach(strip);
  const text = batch ? JSON.stringify(results) : JSON.stringify(results[0], null, 2);
  fs.writeFileSync(argv[oi + 1], text);
  fs.rmSync(tmp, { recursive: true, force: true });
  process.stdout.write(`Wrote ${argv[oi + 1]}\n`);
}

function report() {
  const cnt = LOG.replace(/\.log$/, '.count');
  if (fs.existsSync(cnt)) {
    const n = fs.readFileSync(cnt, 'utf8').split('\n').filter(Boolean).reduce((a, b) => a + +b, 0);
    console.log(`engine runs checked: ${n}`);
  }
  if (!fs.existsSync(LOG)) {
    console.log('no problems logged');
    return;
  }
  const lines = [...new Set(fs.readFileSync(LOG, 'utf8').split('\n').filter(Boolean))];
  const kinds = {};
  for (const l of lines) {
    const k = l.includes('faint') ? 'faint' : l.includes('came in') ? 'switch' : 'hp';
    (kinds[k] = kinds[k] || []).push(l);
  }
  for (const [k, ls] of Object.entries(kinds)) {
    console.log(`${k}: ${ls.length}`);
    for (const l of ls.slice(0, 15)) console.log('  ' + l);
  }
}

if (process.argv.includes('--report')) report();
else wrapper(process.argv.slice(2));
