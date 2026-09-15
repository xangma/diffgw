# Frozen sampled-NQC controls

## Question and boundaries

Does derivative/resampling choice, with matching time held fixed, account for
the remaining sampled TEOB FD residual? Reuse the four supported native
trajectories from `../2026-09-15-teob-fd-attribution`. The predeclared 40+10
case remains unsupported by the published target domain. No trajectory
regeneration, oracle-derived coefficients, fitted alignment, gate changes,
public dispatch changes, or runtime promotion is part of this experiment.

## Controls declared before execution

1. Replay the previous diagnostic's reconstructed spacing and verify its FD
   arrays exactly. Compare the original nominal 0.1 M policy separately.
2. At the nominal baseline's exact selected matching knot, use an independent
   rational centered five-point first derivative twice. This closes the
   previous coefficient-only roundoff check through the complete waveform.
3. Replace only the repeated first derivative with the direct five-point
   second derivative. This is a discretization control, not an asserted fix.
4. Differentiate natural cubic splines of the identical auxiliary samples.
5. Resample the identical cut-support interpolants on nine local nodes
   centered at the fixed matching knot, with spacings 0.05, 0.1, 0.2 and
   0.5 M. Recompute the NQC basis from these states, then use the rational
   first derivative twice. Keep zeroth-order matching values fixed. This
   isolates derivative resolution without moving the matching event.

For controls 2–5 apply the changed coefficients to the nominal original and
auxiliary grids; retain the nominal ringdown grid, peak policy, attachment,
support, projection and conditioner. Assert identical TD time arrays and
attachment metadata. Retain all results, including regressions. The 0.1 M
local-node control records floating-point coordinate differences explicitly.
These controls do not change the auxiliary grid used for ringdown.

Run the existing strict acceptance harness on every FD output using the
previous exact-bin/band wrapper. Preserve its 1e-6 complex relative L2 and
1e-8 mismatch requirements, both polarizations, detector and support checks.
Record input/output hashes, runtime source hashes, commands, PID and timings.

Separately localize sampled-grid TD residuals against validated public TD
fixtures. Use elapsed indices and grid attachment metadata, without alignment.
Report energy fractions as attribution of the TD residual only; do not infer
FD acceptance from TD segments, particularly tiny BNS attachment segments.

## Decision rule

Small full-waveform changes exclude a hypothesis at the observed error scale.
Numerical convergence or a better individual score does not justify adopting
a different model convention. A runtime change requires an independently
established implementation defect and appropriate validation.
