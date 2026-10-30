// Load fuzz/fuzz.js as a library: its source ends with a call to main(),
// so compile it with that call removed and the needed helpers exported.
// fuzz.js itself stays untouched (its hash keys the PS cache).
const fs = require('fs');
const path = require('path');
const Module = require('module');

const file = path.join(__dirname, '..', 'fuzz', 'fuzz.js');
let src = fs.readFileSync(file, 'utf8');
src = src.replace(/^if \(!PARALLEL_PARENT\) main\(\);$/m, '');
src = src.replace(/^if \(PARALLEL_PARENT\) runParallel\(PARALLEL_PARENT\);$/m, '');
src = src.replace(/^const PARALLEL_PARENT = \(\(\) => \{[\s\S]*?\}\)\(\);$/m, 'const PARALLEL_PARENT = null;');
src += '\nmodule.exports = { ps, makeTest, stepPSBattle, engineInput };\n';

const m = new Module(file, module);
m.filename = file;
m.paths = Module._nodeModulePaths(path.dirname(file));
m._compile(src, file);
module.exports = m.exports;
