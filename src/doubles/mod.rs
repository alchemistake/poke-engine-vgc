//! Doubles battle engine for `gen9championsvgc2026regmc`.
//!
//! A self-contained doubles (2-active-per-side) engine that reuses the shared
//! value types and the Gen 9 data tables from the rest of the crate. It is
//! separate from the singles `State`/`Side` model so the singles engine and its
//! tests are unaffected.

pub mod battle;
pub mod damage;
pub mod event_handlers;
pub mod events;
pub mod move_targets;
pub mod rng;
pub mod species_types;
pub mod species_weights;
pub mod state;
