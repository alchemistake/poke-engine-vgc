//! A faithful reimplementation of the Gen 5 LCG PRNG used by
//! pokemon-showdown (`dist/sim/prng.js`, `Gen5RNG`).
//!
//! The battle uses a single shared PRNG. To reproduce pokemon-showdown's
//! damage rolls / critical hits / accuracy rolls exactly, the engine must draw
//! from this PRNG in the *same order* as pokemon-showdown does.
#[derive(Debug, Clone)]
pub struct Gen5Rng {
    /// 64-bit LCG state. The seed words `[s0, s1, s2, s3]` map to
    /// `s0 << 48 | s1 << 32 | s2 << 16 | s3`.
    state: u64,
    /// Total number of `next()` draws made (for logging/verification).
    draws: u64,
    /// Optional log of every raw `next()` value, for parity verification.
    pub draw_log: Vec<u32>,
    /// Fuzz tracing: when on, every draw also records `"<label>|<context>"`
    /// in `draw_ctx` so the fuzzer can tell what each draw was used for.
    pub trace: bool,
    /// What the next draw is for ("accuracy", "crit", ...); reset after it.
    pub label: &'static str,
    /// The action being executed ("turn 3: A0 Weavile ICESHARD", ...).
    pub context: String,
    pub draw_ctx: Vec<String>,
}

const LCG_A: u64 = 0x5D58_8B65_6C07_8965;
const LCG_C: u64 = 0x0000_0000_0026_9EC3;

impl Gen5Rng {
    pub fn new(seed: [u64; 4]) -> Self {
        let state = (seed[0] << 48) | (seed[1] << 32) | (seed[2] << 16) | seed[3];
        Self {
            state,
            draws: 0,
            draw_log: Vec::new(),
            trace: false,
            label: "",
            context: String::new(),
            draw_ctx: Vec::new(),
        }
    }

    /// Name what the next draw is for (fuzz tracing; see `draw_ctx`).
    pub fn label(&mut self, label: &'static str) -> &mut Self {
        self.label = label;
        self
    }

    /// Number of draws made so far.
    pub fn draw_count(&self) -> u64 {
        self.draws
    }

    /// Advance the LCG and return the top 32 bits of the new state.
    /// Equivalent to `Gen5RNG.next()`.
    ///
    /// `#[track_caller]` (here and on the wrappers below): the fuzz trace records
    /// the engine line that drew (`@battle.rs:1234`), shown in the fuzz report's
    /// draw table. `PE_DEBUG_RNG=bt` also prints each draw's engine call stack.
    #[track_caller]
    pub fn next(&mut self) -> u32 {
        self.draws += 1;
        self.state = self.state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
        let v = (self.state >> 32) as u32;
        self.draw_log.push(v);
        if self.trace {
            let label = if self.label.is_empty() { "?" } else { self.label };
            let at = std::panic::Location::caller();
            let file = at.file().rsplit('/').next().unwrap_or(at.file());
            self.draw_ctx.push(format!("{}|{} @{}:{}", label, self.context, file, at.line()));
        }
        self.label = "";
        if let Ok(mode) = std::env::var("PE_DEBUG_RNG") {
            eprintln!("DRAW #{} {}", self.draws - 1, v);
            if mode == "bt" {
                eprintln!("  stack: {}", engine_stack());
            }
        }
        v
    }

    /// `PRNG.random(n)` = `floor(next() * n / 2^32)`, an integer in `[0, n)`.
    #[track_caller]
    pub fn random(&mut self, n: u32) -> u32 {
        let x = self.next() as u64;
        ((x * n as u64) >> 32) as u32
    }

    /// `PRNG.random(from, to)` = `floor(next() * (to - from) / 2^32) + from`,
    /// an integer in `[from, to)`.
    #[track_caller]
    pub fn random_range(&mut self, from: usize, to: usize) -> usize {
        let x = self.next() as u64;
        let span = (to - from) as u64;
        ((x * span) >> 32) as usize + from
    }

    /// `PRNG.randomChance(num, den)` = `random(den) < num`.
    #[track_caller]
    pub fn random_chance(&mut self, num: u32, den: u32) -> bool {
        self.random(den) < num
    }
}

/// The doubles-engine functions on the current call stack, innermost first
/// (`execute_action_body < run_battle_fuzz`), with their lines. Debug only.
fn engine_stack() -> String {
    let bt = std::backtrace::Backtrace::force_capture().to_string();
    let mut out: Vec<String> = Vec::new();
    let mut pending: Option<String> = None;
    for line in bt.lines() {
        let t = line.trim();
        if let Some(rest) = t.split_once(": ").map(|x| x.1).filter(|_| t.chars().next().is_some_and(|c| c.is_ascii_digit())) {
            pending = rest.rsplit("::").nth(0).map(|f| f.trim_end_matches(|c: char| c == '}' || c == '{').to_string());
            if rest.contains("{{closure}}") {
                pending = rest.rsplit("::").nth(1).map(|f| format!("{}{{closure}}", f));
            }
        } else if let Some(loc) = t.strip_prefix("at ") {
            if loc.contains("src/doubles/") && !loc.contains("rng.rs") {
                if let Some(f) = pending.take() {
                    let file_line = loc.rsplit('/').next().unwrap_or(loc);
                    let file_line = file_line.rsplitn(2, ':').nth(1).unwrap_or(file_line);
                    out.push(format!("{} ({})", f, file_line));
                }
            }
            pending = None;
        }
    }
    out.join(" < ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pokemon_showdown() {
        let mut rng = Gen5Rng::new([1, 2, 3, 4]);
        let draws: Vec<u32> = (0..5).map(|_| rng.next()).collect();
        // Values captured directly from pokemon-showdown's Gen5RNG.
        assert_eq!(
            draws,
            vec![2030470262, 3793892072, 2851743046, 574702432, 3620926519]
        );
    }
}
