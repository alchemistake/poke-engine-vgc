// Purpose of a pokemon-showdown PRNG draw, from the call stack's function
// names (innermost first). The engine tags its draws with the same strings
// (`Gen5Rng::label` in src/doubles/rng.rs), so both sides can be compared.
function psDrawPurpose(fns) {
  const has = (n) => fns.includes(n);
  if (has('speedSort')) {
    if (has('runSwitch')) return 'speed tie (switch-in)';
    if (has('eachEvent')) return 'speed tie (eachEvent)';
    if (has('fieldEvent')) return 'speed tie (residual)';
    return 'speed tie (move order)';
  }
  if (has('insertChoice')) return 'speed tie (queue insert)';
  if (has('onFractionalPriority')) return 'fractional priority';
  if (has('getRandomSwitchable')) return 'drag in';
  if (has('onUpdate') && has('sample')) return 'trace';
  if (has('setStatus')) return 'sleep duration';
  if (has('addVolatile')) return 'volatile duration';
  if (has('onBasePower')) return 'fickle beam';
  if (has('randomizer')) return 'damage roll';
  if (has('onBeforeMove')) return 'status before move';
  if (has('hitStepAccuracy')) return 'accuracy';
  if (has('secondaries')) return has('sample') ? 'sample' : 'secondary';
  if (has('sample') && has('hitStepMoveHitLoop')) return 'multihit';
  if (has('selfDrops')) return 'self boost';
  if (has('getDamage')) return 'crit';
  if (has('getRandomTarget') || has('randomFoe')) return 'retarget';
  if (has('onDamagingHit') || has('onSourceDamagingHit')) return 'ability';
  if (has('onStallMove')) return 'stall';
  if (has('onResidual')) return 'residual roll';
  const inner = fns.find((f) => !['random', 'randomChance', 'sample', 'next'].includes(f));
  // Hits 2+ of a `multiaccuracy` move roll accuracy in the hit loop itself.
  if (inner === 'hitStepMoveHitLoop') return 'multiaccuracy';
  return 'other: ' + (inner || '?');
}

module.exports = { psDrawPurpose };
