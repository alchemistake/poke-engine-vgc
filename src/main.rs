#![allow(dead_code)]
use poke_engine::io;
use std::process::exit;

fn main() {
    io::main();
    exit(1);
}

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
