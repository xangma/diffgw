# Trajectory replay with all assembly grids fixed

Predeclared before execution. Diagnostic only; no runtime promotion or oracle
fitting. This corrects the target-audit report's claim that a common auxiliary
grid and fixed events had not been tried. The strategy experiment already ran
that comparison, including equal actual NQC and attachment times. Its joined
interpolation grids remained different (1089 versus 1573 knots).

## Inputs and scope

Use the immutable 30+20 solar-mass, zero-spin/tide, r~20 convergence cache,
its original `current` radiation policy, published 2017 NQC targets, UIB2016
remnant, 0.1 M auxiliary step, and 0.5 M public TD scoring grid. Verify input
and original replay source hashes before use. The two cached trajectories are
`tight_0p1` and `tight_0p01`; no ODE reintegration or PA change occurs.
This historical TD case differs from the later four-case strict FD contract.

## Controls

Reproduce native outputs and the old common-grid/fixed-event pair first.
Use exactly the previously selected common auxiliary grid and fixed NQC/RD
event times. No selection depends on public waveform values or scores.

For each trajectory save five variants:

1. Native postprocessing.
2. Historical common auxiliary grid plus fixed events, including its recovered
   NQC derivative spacing of 0.1000000000003638 M.
3. Same control with the authoritative 0.1 M NQC derivative scalar. Retain all
   native equations, spline inputs, event rules and ringdown spacing behavior.
4. Variant 3 with the full pre-auxiliary corrected amplitude and phase sampled
   onto the `tight_0p1` accepted prefix. Native join, attachment and ringdown
   follow. Thus both trajectories use identical joined/extended/output grids.
5. Repeat variant 4 with the `tight_0p01` accepted prefix to test dependence on
   the choice of reference interpolation grid.

Resample signed corrected amplitude and lifted phase separately using the
existing natural cubic routine. Exact source knots are gathered directly;
reject extrapolation. This does not hold the earlier trajectory-to-auxiliary
spline knots fixed: their interpolation error remains part of each trajectory
approximation. Prefix resampling also introduces an interpolation operation;
the reverse-reference control measures sensitivity to this choice.

## Checks and interpretation

- Hash source, inputs and previous manifests before/after. Preserve historical
  artifacts and all runtime files.
- Verify native processed-cache reproduction and previous replay metrics.
- Verify the explicit-spacing clone reproduces native NQC coefficients and
  corrected samples bitwise on each trajectory's own native auxiliary grid.
- Save all output arrays, joined/extended grids, NQC sampled inputs,
  coefficients and selected/requested times; score the full public support.
- Assert equal actual NQC sample, RD peak, actual RD attachment, joined,
  extended, native output and public scoring grids for both prefix controls.
- Verify baseline-prefix identity and unchanged auxiliary inputs/coefficients
  when only the final interpolation grid changes.
- Independently recompute pairwise/public norms and grid identities from saved
  arrays. Review the helper and run focused tests for its new controls.

Compare trajectory-to-trajectory differences separately from errors against
the public binary. A small first quantity neither passes the strict FD gate
nor establishes the correct numerical policy. Do not choose a grid or event
because it happens to lower the oracle error.

## Preflight amendment, before any output was generated

The first run stopped at the source-hash guard: the NQC and ringdown modules
have changed since the old replay. The later continuous-matching experiment
documents additions to both modules and reproduction of legacy grid output.
The replay helper and trajectory driver still match their historical hashes.
Keep the failed preflight log. Pin current runtime hashes to the most recent
frozen target-audit source record and record both historical/current hashes.
Require native cache reproduction and historical controlled-pair scores as
compatibility checks before interpreting any new control. Source-hash identity
with the original replay is explicitly not claimed.
