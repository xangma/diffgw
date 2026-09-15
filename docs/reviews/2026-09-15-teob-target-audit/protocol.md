# NQC target and event audit: frozen protocol

This protocol precedes the new numerical runs. Use the four frozen zero-spin,
zero-tide trajectories from `2026-09-15-teob-fd-attribution`, the authoritative
0.1 M policy, and the bitwise nominal baselines in `2026-09-15-teob-grid-nqc`.
No upstream implementation/source access, inferred oracle coefficients,
trajectory regeneration, runtime edits, or acceptance changes.

## Source and convention audit

Check arXiv:1703.06814v1 Tables II/IV and Eq. (A3), including the deliberate
equal-mass branch, against the explicit 2017 provider. Check
arXiv:1806.01772v2 Sec. II.B.4 for version applicability and Sec. II.A for
normalization and event definitions. The newer targets replace the older
ones for q >= 4; all four current cases have q < 4. Do not substitute the
2018 provider into this experiment.

## Numerical controls: 11 per case, 44 full FD evaluations

1. Reproduce the authoritative nominal TD/FD arrays bitwise.
2. At the same selected knot perturb each of the four target values in both
   directions by its maximum printed-coefficient rounding radius. At zero
   spin this is 5e-9 for each equal-mass target, or (1+nu)*5e-9 for each
   asymmetric target. This assumes rounding to nearest at eight decimals;
   it does not represent fit uncertainty, version differences, or arbitrary
   implementation precision. Keep the other three targets fixed.
3. With nominal targets, evaluate the same five-point/repeated-derivative
   equations at the immediately preceding and following auxiliary knots.
   Keep every sampled input, step, output grid and ringdown event fixed.
   These are diagnostic sensitivities, not proposed event prescriptions.

For the target box, independently propagate its radii through the inverse
two-by-two matching matrices. Check its coefficient bounds against all 16
target-box vertices without waveform generation. Record a first-order FD
triangle envelope from the four centered directional responses; label it
as a linear approximation, not a certified nonlinear waveform bound.

Every generated waveform receives the existing complete strict single-case
harness (both polarizations, PSDs, detectors, finite/DC/support checks,
complex relative L2 <= 1e-6, mismatch <= 1e-8). Save all outcomes, including
regressions. No direction, parameter, fit or convention is chosen by oracle
score. Require unchanged attachment, output time grid and runtime hashes.

## Interpretation

Agreement with a paper establishes the named published convention, not
identity with an installed binary. Small target perturbations support a
local sensitivity conclusion only. Adjacent-knot sensitivity measures
event discretization on one frozen trajectory; it cannot identify the
correct physical event or establish trajectory convergence. A runtime
change requires an independently established defect.
