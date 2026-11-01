// Census: list ALL failing seeds in [from, to), not only the first.
// Runs fuzz.js --check-only workers on contiguous segments; restarts past each FAIL.
// Usage: node fuzz/census.js FROM TO [JOBS]  (run from poke-engine/)
//        FUZZ_ENGINE=fuzz/event_audit.js node fuzz/census.js ...  (engine wrapper)
// Prints one failing seed per line. Diff two runs (before / after a fix) to
// see every battle the fix repaired or broke, not only the first one.
const { spawn } = require('child_process');
const path = require('path');
const [from, to] = [parseInt(process.argv[2], 10), parseInt(process.argv[3], 10)];
const jobs = parseInt(process.argv[4] || '16', 10);
const fuzz = path.resolve('fuzz/fuzz.js');
const fails = [];
const seg = Math.ceil((to - from) / jobs);
let pending = jobs;
for (let j = 0; j < jobs; j++) {
  const lo = from + j * seg, hi = Math.min(to, lo + seg);
  const run = (s) => {
    if (s >= hi) { if (--pending === 0) done(); return; }
    const c = spawn(process.execPath, [fuzz, '--seed', String(s), '--count', String(hi - s), '--check-only',
      '--jobs', '1', '--workdir', path.resolve(`fuzz/work/census-${j}`),
      // FUZZ_ENGINE: an engine wrapper (e.g. fuzz/event_audit.js).
      ...(process.env.FUZZ_ENGINE ? ['--engine', process.env.FUZZ_ENGINE] : [])], { stdio: ['ignore', 'pipe', 'ignore'] });
    let out = '';
    c.stdout.on('data', (d) => (out += d));
    c.on('close', () => {
      const m = out.match(/FAIL (\d+)/);
      if (m) { fails.push(+m[1]); run(+m[1] + 1); } else run(hi);
    });
  };
  run(lo);
}
function done() {
  fails.sort((a, b) => a - b);
  console.log(fails.join('\n'));
}
