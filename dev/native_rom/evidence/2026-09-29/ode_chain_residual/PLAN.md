# Frozen chain-rule residual: preregistered diagnostic

The opt-in chain-rule candidate and complete two-row gate receipts are in
`../ode_chain_derivatives/`. The nonspinning refined ODE complex error is
`2.614471228362189e-6` against `1e-6`; its grid complex errors are
`6.220681283284825e-9` and `8.17864033897492e-9`, so the latter fails
the unchanged decreasing-error test. The spinning row passes both gates.

No numerical rule or gate is changed for this diagnostic. The observable
success condition for any later source-backed correction is that both rows
pass all original ODE and grid gates with the original tolerances, event
definitions, matching targets, and output spacings. A control that only
shrinks an error without passing a gate is recorded as such.

## Competing hypotheses and decisive checks

1. **Retained sampled-spline values.** The chain candidate replaces all
   matching derivative rows, but its amplitude value row still comes from
   splines of sampled `A` and `A*q`. Compare that row with direct evaluation
   of the published base mode and NQC basis at the same continuous match
   state on each frozen nonspinning level. Use the unchanged chain derivative
   rows and targets to solve a counterfactual coefficient vector with only
   the value row replaced. Compare its full five coefficients with the
   active solve. Phase value is not a matching condition and cannot alter
   its two coefficients. If the value-row swap has a material coefficient
   effect, evaluate its waveform effect on the frozen levels before claiming
   a correction; do not infer a gate pass from the solve alone.

2. **Trajectory and match-state contribution.** Compare continuous match
   states, match times, chain vectors, and coefficient drift across
   `ode_0/ode_1/ode_2`. The exact accepted-trajectory hashes in the previous
   receipts differ across levels, so numerical trajectory effects remain
   possible. Replay the frozen waveforms and compare raw pre-NQC mode on
   common pre-attachment samples with the complete attached waveform.
   Check whether changing the value row removes that drift before
   attributing it to the trajectory or event locator. Event differences
   alone do not establish causality.

3. **Output-grid trajectory path.** The previous grid-control receipt found
   that restart index and first trial step depend on the requested grid.
   The chain candidate's nonspinning grid errors are now near `1e-8`, but
   still nonmonotone. First compare the candidate accepted-trajectory hashes,
   match states, and value rows across `ode_2/grid_1/grid_2`. As a diagnostic
   control only, hold the already established nonspinning restart at
   `165 M` and coarse first step at `0.5 M` on all three grids using existing
   opt-in controls. Compare exact nested waveforms and accepted trajectory
   hashes. Identical controlled trajectories and waveforms would locate
   the default-grid sensitivity in the trajectory path; they would not
   justify a row-specific production restart or first step.

Run only the frozen nonspinning row for the causal comparison. If a single
published-equation numerical correction survives it, implement it behind
an opt-in SEOBNR development flag, then run **both** frozen rows and both
gates without changing their definitions. Otherwise preserve the default
generator and report the maximal supported boundary. Use only diffgw's
published-equation implementation, public paper equations, and black-box
outputs; do not read LAL implementation source or private DCC text.
