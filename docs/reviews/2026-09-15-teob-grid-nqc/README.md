# Sampled TEOB parity: corrected replay and isolated numerical controls

## Outcome

**The previous replay overstated the BNS error.** Its reconstruction of a
nominal 0.1 M step from two large timestamps changed the selected NQC matching
knot. Saving and replaying the authoritative scalar policy fixes this
diagnostic defect. The corrected catalog error is about twelve times smaller.
These are corrected measurements of the existing native model; runtime
waveform code and public dispatch are unchanged.

All four supported cases still fail the existing strict parity gate. The
new derivative and pre-attachment output-interpolation controls do not close
that gap. No numerical alternative was promoted or selected by its oracle
score. Two Luna agents handled bounded audits and localization, with concise
outputs; the primary agent verified material claims and ran full FD controls.

## Corrected baseline

Complex relative L2, plus / cross, against the existing strict public oracle:

| Masses, solar masses | Previous replay | Authoritative 0.1 M replay |
|---|---:|---:|
| 1.4 + 1.4, catalog | 2.81030e-4 / 1.56066e-3 | **2.36585e-5 / 1.26205e-4** |
| 30 + 20 | 3.27159e-4 / 3.27009e-4 | 3.27159e-4 / 3.27009e-4 |
| 20 + 20 | 1.03035e-4 / 8.25393e-5 | 1.03035e-4 / 8.25394e-5 |
| 1.6 + 1.2 | 2.89770e-4 / 3.36591e-4 | **7.34462e-5 / 1.00234e-4** |

The gate remains 1e-6 complex relative L2 and 1e-8 mismatch, with both
polarizations, detector projections, PSDs, finite values, DC and support
checks. All parameters and exact `[flow,fhigh)` bins match the frozen
experiment. The predeclared 40+10 case remains explicitly unsupported:
its nu=0.16 is outside the published NQC target domain used here.

Evidence: [derivative-controls.json](derivative-controls.json), each case's
`nominal.npz`, and `validated-replay/<case>/matching-controls.json`.

## Why the replay changed

For the BNS caches, subtracting adjacent auxiliary timestamps around
4.6 million M yields `0.09999999962747097`, instead of the stored policy's
intended `0.1`. Rebuilding the grid accumulates a maximum 7.45e-8 M coordinate
change. The `searchsorted(..., right=True)-2` matching rule is discontinuous
at the relevant boundary, so this tiny coordinate difference moves the
selected physical knot by approximately 0.1 M.

Restoring the nominal step changes the catalog index **165 to 166**, and
the unequal BNS index **166 to 167**. The maximum coefficient changes are
0.12319 and 0.12448. The nominal auxiliary grids are bitwise identical to
the original cached grids in all four cases. Smaller absolute BBH times
produce only tiny coefficient changes and retain the selected indices.

The diagnostic cache writer now saves `continuation_fixed_step` and
`nqc_auxiliary_spacing`. The replay reads those values, rejects conflicts,
and requires an explicit independently known `--fixed-step` for older
caches. It also requires a separate output directory. Original reports and
arrays are preserved. The original replay script is saved under `before/`;
[implementation.patch](implementation.patch) contains only this turn's
tooling fix and its regression tests.

Evidence: [spacing-audit.json](spacing-audit.json). The earlier continuous
matching rejection remains valid: its FD errors exceed these corrected
grid baselines in all four supported cases. Historical mixed-policy and
coefficient-attribution results that used the reconstructed grid should not
be treated as attribution of the authoritative BNS baseline.

## What the new controls establish

The [initial protocol](protocol.md) preceded 36 complete FD evaluations:
two baselines and seven derivative controls per case. The matching time,
ringdown attachment and TD output grid are identical for all controls of
the authoritative baseline.

- Independent rational five-point differentiation changes the FD waveform
  by at most **6.17e-11** relative L2. Arithmetic roundoff is negligible at
  the observed residual scale.
- Direct second derivatives and natural-cubic auxiliary derivatives do not
  restore parity. The latter change the BBH waveforms by up to 3.51e-5;
  they worsen all four cases' scores.
- Local derivative spacings 0.05, 0.1, 0.2 and 0.5 M, centered at the same
  matching knot, also fail throughout. Finer spacing is not consistently
  closer to the oracle. These are controls on one frozen interpolated
  trajectory, not a trajectory-convergence study.

The [follow-up protocol](resampling-protocol.md) then isolated output
interpolation in 16 further evaluations. Before the actual grid attachment,
amplitude, phase, or both were resampled directly from the corrected accepted
series. NQC coefficients and every post-attachment mode sample stayed fixed.
The largest FD change was **8.16e-7** relative L2; no case reaches parity.
For the catalog the change is below 2.66e-11. Thus the catalog's concentrated
pre-attachment residual is not explained by this join/interpolation choice.

Evidence: each case's `output-resampling.json`. All controls retain their
full acceptance result, waveform arrays and hashes, including regressions.

## Where the remaining TD residual occurs

Localization compares the authoritative grid waveform with validated public
exact-start TD fixtures at identical elapsed indices, without fitted shifts.
Percentages below are shares of **joint TD squared residual energy**, not FD
error fractions or proof of a particular physical cause.

| Segment | Catalog samples / share | 30+20 samples / share |
|---|---:|---:|
| PA span | 64,251 / 0.0147% | 2,637 / 0.0085% |
| Early ODE | 37 / 1.3719% | 1,364 / 7.7919% |
| Late, before NQC matching knot | **1 / 98.3460%** | 8 / 38.6590% |
| Matching knot to attachment | 0 / empty | 0 / empty |
| At and after attachment | 2 / 0.2674% | 78 / 53.5406% |

The catalog's coarse physical output sampling makes a broad claim about the
matching region inappropriate: the dominant segment has exactly one sample.
Its cumulative residual crosses 10%, 50%, 90% and 99% at index 64288. For
30+20 those crossings occur at indices 4002, 4009, 4013 and 4021.

Evidence: [catalog localization](localization-catalog.json),
[30+20 localization](localization-bbh_reference.json), and their NPZ arrays.

## Validation and next investigation

- **35 focused tests passed**, including large-timestamp metadata regression
  guards and the existing NQC and conditioned-waveform tests.
- Independent polynomial checks through degree four give maximum first /
  second derivative errors 4.98e-14 / 6.19e-13.
- The corrected replay tool reproduces every nominal TD and FD array
  **bitwise** in all four cases.
- Artifact validation verifies all 52 experimental outputs, unchanged
  attachment/grid controls, public-fixture comparisons, and all **130 runtime
  source files unchanged**. Of 76 entries in the historical manifest, 75
  remain unchanged; the deliberately fixed replay script has a verified
  original copy. Every historical report and array remains unchanged.
- No native evolution, public dispatch, threshold, fitted alignment, physical
  coefficient or runtime matching policy changed. No jobs remain running.

See [artifact-validation.json](artifact-validation.json) and
[validation.json](validation.json) for checks and commands.

The next useful investigation is the published NQC **target version,
normalization and event convention**, with these authoritative scalar
policies and frozen trajectories. The present controls do not identify a
target defect, and trajectory/model error remains possible. Further
derivative-roundoff or join-interpolation tuning has little support from
this evidence. Preserve the exact acceptance test while auditing those
remaining physical and timing assumptions.
