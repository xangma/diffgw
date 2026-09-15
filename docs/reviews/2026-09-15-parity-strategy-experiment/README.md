# Parity strategy experiment — 15 September 2026

**The strongest next step is to remove grid-dependent merger matching.** The
controlled EOB experiment isolates a large numerical sensitivity there. The
bounded joint complex surrogate does not establish an advantage over the
completed Eq.42 baseline. Its NR phase convention also needs stronger checks.

## 1. EOB: auxiliary grid dominates the observed amplification

Native reassembly of cached trajectories reproduces the original baseline to
`2.43e-16` relative L2. A local diagnostic clone reproduces the native baseline
and both tight trajectories exactly. The public waveform supplies scores only.

For the same two tight trajectories, changing independent controls gives:

| Controls shared between trajectories | Output difference, relative L2 |
|---|---:|
| Native controls | `2.1647e-3` |
| Auxiliary grid | `1.4726e-7` |
| NQC and ringdown event times | `7.9988e-4` |
| Auxiliary grid and event times | `9.0398e-7` |

The common grid cuts inter-trajectory spread by about **14,700 times**. This
strongly localizes this case's amplification to grid-dependent matching,
including the NQC evaluation times and knot selection. Event times alone do
not remove it. Matching matrices, coefficients, conditions and indices are
recorded; the common grid and fixed event times are explicitly verified.

This is a stability diagnosis, not a runtime fix or LAL parity improvement.
The grid-only replay still differs from the public waveform by about 0.205%,
and the grid-plus-fixed-events replay by about 0.481%. A fixed grid shared
between two cached trajectories is an experimental control, not a general
waveform algorithm.

**Next implementation experiment:** locate the peak continuously and evaluate
NQC matching quantities at a declared physical time, using continuous
interpolation/derivatives rather than a trajectory-dependent selected knot.
Test auxiliary-grid origin, spacing and ODE-tolerance changes independently,
then rerun compatibility gates and independent scientific checks. This is a
proposal supported by the diagnosis; it has not been implemented here.

Details: [EOB report](eob/README.md) and [native replay evidence](eob/native-fixed-event-reassembly.json).
The initial synthetic cache-ratio diagnostic was rejected during review and
is explicitly marked withdrawn; no conclusion relies on it.

## 2. NR: the simple joint complex basis is not a demonstrated improvement

Training-only selection on five families gave mean cross-validation complex
errors of **12.15% for the basis versus 6.39% for the completed baseline**.
The comparator retains the original Eq.42 `(3,3)` amplitude calibration and
adds a simple independent phase interpolation. It is a diagnostic comparator,
not an already implemented full waveform model.

Across six existing interpolation cases, median `(3,3)` complex error is
**1.78% versus 1.04%**. The candidate introduces four amplitude errors above
1%. At q approximately 10 it also extrapolates worse.

Two fresh NR families were chosen before inspecting their waveforms. One
failed the unchanged extraction/resolution gates. The qualified q approximately
6.5 case gives mixed results: the basis slightly improves the two-mode mean
(2.06% versus 2.15%), while `(3,3)` complex error increases
(2.09% versus 0.95%). Resolution sensitivity exceeds the model differences.
The requested two-family qualified coverage remains incomplete.

These results reject this particular candidate as a justified replacement.
They do not rule out a better reduced-basis representation or interpolator.
Only a nonspinning, remnant-conditioned ringdown frequency interval was tested;
this is not full waveform, detector strain, spin, or LAL parity validation.

Details: [NR report](nr/README.md), [frozen protocol](protocol.md),
[fresh inputs](data/qualified-manifest.json), and [coverage gap](data/holdout-status.json).

## 3. Shared-phase normalization needs attention before another fit

For the fresh q approximately 6.5 family, replacing the sampled `(2,2)` peak
with a local continuous quadratic peak changes inter-resolution complex error:

- `(2,2)`: **9.81% → 7.37%**.
- `(3,3)`: **3.58% → 1.78%**.

The relative peak correction is `0.07342 M`. Amplitude differences and phase
residuals after an affine fit are essentially unchanged. Sampled timing
therefore contributes materially to shared-gauge error, but does not explain
all NR sensitivity. This was a post-validation diagnostic, not a refit or a
revision of the frozen model scores.

Before enlarging the training set or adding spins, declare and test a stable
continuous time/orbital-phase convention and a raw joint complex error budget.
Separate NR resolution/extraction sensitivity from gauge sensitivity. A future
model selection experiment needs new prospective validation data.

Details: [peak-gauge diagnostic](nr/peak-gauge-diagnostic/README.md).

## Validation and scope

- 14 tests passed: 10 reduced-basis/comparison tests and 4 replay/report tests.
- An independent rerun reproduced the saved EOB controlled results exactly.
- Verified NR model/source freezes, evaluation model hashes, fresh input hashes,
  and EOB source hashes. Conditional spectrum JVP checks passed.
- New diagnostic scripts, tests and review artifacts only; no package/runtime
  promotion, LAL training, commit, or PR. Existing parity gates were not rerun.
- All work completed locally on RONI2; no jobs remain running from this experiment.

Training/evaluation commands appear in the component reports. Fresh-data
command and completed PID are recorded in [data/run.log](data/run.log).
Two Luna agents handled bounded subtasks; Spark was unavailable in this session.
