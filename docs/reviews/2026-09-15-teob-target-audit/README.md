# TEOB NQC targets and matching-time audit

## Outcome

**This audit found no target transcription, active normalization, sign or
version-selection defect that explains the four remaining parity failures.**
All 44 frozen-trajectory FD controls fail the unchanged strict gate. Printed
coefficient rounding has a small local effect; moving the NQC solve by one
sample in either direction worsens both polarizations in every case. These
results do not establish parity or identify the remaining physical cause.

Two Luna agents handled narrow source/convention audits with concise outputs.
The primary agent verified material claims, ran the controls and independently
checked their saved arrays. No runtime, public dispatch, threshold, physical
coefficient or historical artifact changed. No jobs remain running.

## Published model and conventions

The explicit 2017 provider matches the printed coefficients in Tables II/IV
and Eq. (A3), including the deliberate equal-mass branch. The active assembly
uses `A/(nu*sqrt(24))`, consistent positive mode frequency, and initial-total-
mass time units with the separate remnant-mass ringdown scale.
See [source audit](source-audit.md), [convention audit](convention-audit.md), and
[the 2017 paper](https://arxiv.org/pdf/1703.06814v1).

The 2018 paper replaces the old NQC targets for **q >= 4**, so replacing the
targets in these four q < 4 cases is not supported by that version change.
See [2018 Sec. II.B.4](https://arxiv.org/pdf/1806.01772v2).

**Correction to the previous report's wording:** the local `nu > .16` guard
is a compatibility-selection restriction aligned with this version boundary,
not the published validity domain of the 2017 formulas. The historical report
is preserved. The excluded 40+10 case remains excluded from this experiment;
supporting it requires a separately specified target-version path.

## Frozen controls and results

The [protocol](protocol.md) preceded the runs. Every case uses its original
zero-spin, zero-tide trajectory, authoritative 0.1 M auxiliary step, original
conditioning and exact comparison bins. Each nominal replay reproduces all
six corrected-baseline TD/FD arrays **bitwise**.

Complex relative L2 against the public binary, plus / cross:

| Masses (solar masses) | Nominal | Previous NQC knot | Next NQC knot |
|---|---:|---:|---:|
| 1.4 + 1.4 | 2.36585e-5 / 1.26205e-4 | 2.81031e-4 / 1.56066e-3 | 2.37481e-4 / 1.29247e-3 |
| 30 + 20 | 3.27159e-4 / 3.27009e-4 | 4.09790e-3 / 4.10812e-3 | 3.44966e-3 / 3.45604e-3 |
| 20 + 20 | 1.03035e-4 / 8.25394e-5 | 3.83200e-3 / 3.87460e-3 | 3.87855e-3 / 3.90160e-3 |
| 1.6 + 1.2 | 7.34462e-5 / 1.00234e-4 | 2.89779e-4 / 3.36602e-4 | 4.33628e-4 / 5.20974e-4 |

The adjacent-knot controls reuse every sampled input and the same derivative
equations, changing only their evaluation index by one (approximately 0.1 M).
Ringdown attachment and output time grids remain fixed. These controls measure
sensitivity; they neither define the correct physical event nor justify
choosing an event from its oracle score.

Every one of the 44 outputs receives the complete existing strict harness:
both polarizations, three PSDs, six detector projections, finite/DC/band
checks, complex relative L2 <= 1e-6 and mismatch <= 1e-8. All fail.
Per-case `controls.json` files retain all scores and regressions; matching
NPZ files retain the generated arrays.

### Conditional printed-precision sensitivity

Assuming rounding to nearest at eight decimals, the zero-spin target radius
is 5e-9 for each equal-mass target and `(1+nu)*5e-9` for each unequal-mass
target. Each of the four targets was independently perturbed in both
directions. The linear coefficient bounds were checked against all 16
vertices of the target box at fixed matching matrices.

| Masses | First-order FD triangle envelope, plus / cross |
|---|---:|
| 1.4 + 1.4 | 7.14631e-9 / 1.91187e-8 |
| 30 + 20 | 6.33203e-8 / 6.38929e-8 |
| 20 + 20 | 5.17997e-8 / 5.23211e-8 |
| 1.6 + 1.2 | 6.15543e-9 / 7.11561e-9 |

This sums the norms of four centered directional responses, divided by the
nominal waveform norm in the comparison band. Its maximum is **6.39e-8**;
each entry is below 0.064% of its corresponding nominal oracle residual.
The largest centered-pair symmetry residual is 1.48e-12. This supports a
small local rounding effect. It is **not a certified nonlinear waveform
bound**, physical fit uncertainty, or a bound on unknown binary coefficients.

### Event ledger

Times below are offsets in M from the sampled NQC orbital-frequency peak.
Requested and selected times must remain distinct.

| Case | Requested NQC | Selected NQC | Interpolated RD peak | Actual RD attachment |
|---|---:|---:|---:|---:|
| 1.4 + 1.4 | -1 | -1.1 | +0.0430232 | -1.1 |
| 30 + 20 | -1 | -1.1 | +0.0453051 | -1.1 |
| 20 + 20 | -1 | -1.1 | +0.00231414 | -1.1 |
| 1.6 + 1.2 | -1 | -1.1 | -0.0200135 | -1.2 |

The grid NQC path uses a sampled peak and the documented index rule. Ringdown
uses its separate seven-knot peak estimate and attachment rule. Consequently,
the unequal-mass low-mass case solves NQC one auxiliary knot after attachment.
This is a recorded consequence of existing policies, not evidence that one
should alter them to improve a score. The papers' physical event definitions
do not by themselves specify these numerical index rules.

## Verification

- **35 focused tests passed** (cache metadata, merger NQC and conditioned waveform).
- Independent decimal evaluation confirms all four zero-spin target vectors.
- Independent NumPy checks verify all 44 output hashes, both polarization
  L2 scores against frozen public FD arrays, directional sensitivities,
  unchanged grids, and bitwise nominal replays. Maximum L2 discrepancy
  between the independent calculation and recorded harness is 1.39e-17.
- All **130 runtime source files**, three tracked support/harness files and
  **96 previous manifest entries** remain unchanged. Experimental script,
  protocol, helper and input hashes match their run records.
- All four jobs exited successfully; elapsed times were 4.7–83.5 seconds.

See [artifact validation](artifact-validation.json), [test record](validation.json),
and [artifact manifest](artifact-manifest.json). The scripts are
`scripts/teob_target_event_controls.py` and
`scripts/validate_teob_target_controls.py` at the repository root.

## Next useful experiment

**Separate trajectory changes from event movement.** The earlier
[ODE study](../2026-09-15-no-contact/teob-ode-convergence-clean/conclusion.md)
changed integration tolerances, but each trajectory regenerated its own NQC
grid and attachment: the last two levels moved attachment by 0.0274 M while
their pre-NQC modes changed by only 2.68e-8 relative L2. That study explicitly
did not hold the auxiliary grid and event time fixed. Repeating its tolerance
sweep would leave the same confounding factors.

The next bounded control should first verify compatibility of saved trajectory
caches, then replay two trajectory levels through one predeclared baseline
auxiliary grid, NQC evaluation time and attachment. Compare those results with
the same two trajectories using their own events, within one consistent case
and output contract. This separates the effect of changed trajectory values
from the effect of changed sampling/event policies, without fitting to the
oracle. The present scripts supply parts of the required diagnostic assembly;
this combined experiment has not been run here.

A public full-route NQC-off comparison would provide complementary attribution,
but the [earlier attribution report](../2026-09-15-no-contact/teob-nqc-attribution-clean/conclusion.md)
records that this control was unavailable. It should only be pursued if a
documented public interface exposes it. No numerical result here licenses
inferring hidden coefficients or identifying internal binary behavior.
