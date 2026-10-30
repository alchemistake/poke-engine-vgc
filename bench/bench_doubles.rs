//! Speed test: run the fixed benchmark games (`bench/games.json`, made by
//! `bench/gen_games.js`) through the doubles engine and print the best time
//! of `reps` runs, in microseconds for all games. JSON parsing is outside the
//! timed region; each game builds its state from the parsed input.
//!
//! cargo run --release --features champions --no-default-features \
//!     --bin bench-doubles -- [games.json] [reps]
use poke_engine::doubles::battle::run_battle_fuzz;
use poke_engine::doubles::rng::Gen5Rng;
use poke_engine::doubles::state::{BattleInput, DoublesState, FuzzBattleResult, Leads};
use std::time::Instant;

fn play(input: &BattleInput) -> FuzzBattleResult {
    let mut state = DoublesState::from_input(input);
    let leads = input.leads.clone().unwrap_or(Leads {
        a: vec![0, 1],
        b: vec![0, 1],
    });
    state.send_out_leads(&leads.a, &leads.b);
    let mut rng = Gen5Rng::new(input.seed);
    run_battle_fuzz(&mut state, &mut rng, input.choices.as_deref().unwrap(), None)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(String::as_str).unwrap_or("bench/games.json");
    let reps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let json = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {}", path, e));
    let games: Vec<BattleInput> = serde_json::from_str(&json).expect("bad JSON");

    let mut best = u128::MAX;
    let mut turns = 0u64;
    for _ in 0..reps {
        let t0 = Instant::now();
        turns = 0;
        for g in &games {
            let r = std::hint::black_box(play(std::hint::black_box(g)));
            turns += r.turns.len() as u64 - 1;
        }
        best = best.min(t0.elapsed().as_micros());
    }
    eprintln!("{} games, {} turns, best of {} runs", games.len(), turns, reps);
    println!("{}", best);
}
