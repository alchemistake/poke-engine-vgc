//! Debug output switches, read from the environment once at first use (an
//! `std::env::var` call per RNG draw or event was ~8% of the run time).
//! Set them before the process starts, e.g. `PE_DEBUG_DMG=1`.

use std::sync::LazyLock;

pub struct Flags {
    /// `PE_DEBUG_DMG`: damage calc and turn headers.
    pub dmg: bool,
    /// `PE_DEBUG_SPEED`: action priority / speed before the move-order sort.
    pub speed: bool,
    /// `PE_DEBUG_ACTIVES`: `sorted_actives` input and result.
    pub actives: bool,
    /// `PE_DEBUG_RES`: residual handler order.
    pub res: bool,
    /// `PE_SLOT`: switch-in slot placement.
    pub slot: bool,
    /// `PE_DEBUG_EVENTS`: handlers collected for each event sort.
    pub events: bool,
    /// `PE_DEBUG_RNG`: every PRNG draw; `PE_DEBUG_RNG=bt` adds its call stack.
    pub rng: bool,
    pub rng_bt: bool,
}

static FLAGS: LazyLock<Flags> = LazyLock::new(|| {
    let on = |k: &str| std::env::var(k).is_ok();
    let rng = std::env::var("PE_DEBUG_RNG").ok();
    Flags {
        dmg: on("PE_DEBUG_DMG"),
        speed: on("PE_DEBUG_SPEED"),
        actives: on("PE_DEBUG_ACTIVES"),
        res: on("PE_DEBUG_RES"),
        slot: on("PE_SLOT"),
        events: on("PE_DEBUG_EVENTS"),
        rng_bt: rng.as_deref() == Some("bt"),
        rng: rng.is_some(),
    }
});

#[inline]
pub fn flags() -> &'static Flags {
    &FLAGS
}
