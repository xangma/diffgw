# Continuous TEOB matching — 15 September 2026

**Continuous matching reduces the difference between the two tightest cached
integrations by about 2,700 times.** In this case it also lowers error against
the public reference. The implementation is available through
`conditioned_teob_fd(..., matching_method="continuous")`; the default remains
`"grid"`. Full LAL parity has not been established.

## Implementation

1. Locate the unique interior maximum of the orbital-frequency natural cubic
   spline over accepted trajectory samples, solving its interval derivatives.
2. Evaluate the four next-to-quasi-circular (NQC) matching conditions at
   `t_match = t_peak - 1 M`. Spline derivatives of sampled amplitude products
   and phase bases make the corrected accepted-series spline satisfy the
   amplitude, amplitude derivative, frequency, and frequency derivative targets.
3. Pass that same peak to ringdown. Anchor the published ansatz's phase at the
   exact matching time. Evaluate the corrected spline before attachment and
   the analytic ansatz afterward when producing uniformly sampled output.

Auxiliary-grid origin and spacing no longer determine the continuous solve or
phase anchor. The published ansatz retains its own amplitude. Peak selection
and output partitioning are piecewise differentiable; ambiguous or unresolved
peaks are rejected.

Sources: [peak locator](/Users/xangma/repos/torchwave/src/torchwave/approximants/eob/continuous_matching.py),
[NQC](/Users/xangma/repos/torchwave/src/torchwave/approximants/eob/teob_merger_nqc_clean.py),
[ringdown](/Users/xangma/repos/torchwave/src/torchwave/approximants/eob/teob_ringdown_assembly_clean.py),
[adapter](/Users/xangma/repos/torchwave/src/torchwave/approximants/eob/teob_conditioned_waveform_clean.py).

## Measured results

Reassembly uses three previously saved native trajectories for the same
nonspinning 30+20 solar-mass binary. Their relative integration tolerances are
`1e-11`, `1e-12`, and `1e-13`. Native equations and existing published NQC/remnant
fits supply reconstruction quantities; public waveform samples supply scores
only. Comparisons use identical physical sample coordinates without alignment
optimization or an extra interpolation step.

| Measurement: relative complex L2 error | Grid | Continuous |
|---|---:|---:|
| Difference between two tightest integrations | `2.1647e-3` | `8.0130e-7` |
| Baseline versus public reference | `1.7625e-3` | `1.1923e-3` |
| Tight `1e-12` versus public reference | `4.0396e-3` | `1.1917e-3` |
| Tight `1e-13` versus public reference | `2.0543e-3` | `1.1915e-3` |

The finest-trajectory public error decreases by about **42%**, from **0.2054%
to 0.1191%**. The continuous peak changes by only `1.37e-7 M` between the two
tightest integrations. This is a waveform L2 diagnostic, not a detector
mismatch or a parity-gate result.

For each trajectory, auxiliary spacings `0.05`, `0.1`, `0.2`, and `0.5 M`
produce bitwise-identical continuous coefficients, peaks, and waveform values
on shared support. The `0.05 M` case covers `[0, 12393.5] M`; the others cover
the full scored interval `[0, 12468.5] M`. The existing 1,500-sample extension
changes duration with spacing, so this does not establish tail-duration
invariance. Synthetic tests independently vary auxiliary origin and spacing.

The native grid baseline reproduces the saved processed waveform to
`2.57e-16` relative L2. See [comparison evidence](/Users/xangma/repos/torchwave/docs/reviews/2026-09-15-continuous-matching/report.json).

## Validation

- **59 focused tests pass**, covering analytic peak location, event gradients,
  four exact NQC targets, auxiliary-grid changes, nonlinear pre-attachment
  interpolation, signed-amplitude phase handling, batching, adapter policy
  forwarding, and legacy replay. [Test log](/Users/xangma/repos/torchwave/docs/reviews/2026-09-15-continuous-matching/focused-tests.log).
- A fresh native 30+20 solar-mass run completes PA initialization, adaptive
  evolution, NQC, ringdown, projection, and Fourier conditioning in 29.6 s.
  Its 4,087 TD and 2,049 FD samples are finite, and NQC/ringdown share exactly
  the same peak, matching time, and attachment time. [Smoke report](/Users/xangma/repos/torchwave/docs/reviews/2026-09-15-continuous-matching/adapter-smoke.json).
- Snapshot parity gates remain **17/21 catalog + 11/13 holdout = 28/34**.
  Both JSON reports are byte-identical to the prior baseline. The existing
  XHM, XPHM, SEOBNR ODE, and TEOBResumS catalog failures, plus XHM/XPHM holdout
  failures, remain. [Baseline comparison](/Users/xangma/repos/torchwave/docs/reviews/2026-09-15-continuous-matching/parity-baseline-comparison.json).
- Source hashes were unchanged during each run. Gates use an immutable source
  snapshot. Commands, host, completed PIDs, and logs are recorded in
  [jobs.json](/Users/xangma/repos/torchwave/docs/reviews/2026-09-15-continuous-matching/jobs.json).
  All jobs completed locally on RONI2.

## Scope and next step

The public `TEOBResumS` dispatch does not use this clean adapter. Its unchanged
parity gates establish regression protection; they do not certify the opt-in
continuous path. Coverage here comprises one intrinsic binary, three cached
integrations, synthetic checks, and one fresh native waveform.

The observer epoch still uses the first accepted orbital-frequency decrease.
The accepted-trajectory spline retains finite integration error, and the
published NQC/ringdown fits need not agree in amplitude or derivatives at the
join. In continuous mode, unused auxiliary peak-polynomial diagnostics are
documented NaNs; the supplied continuous peak is authoritative.

Next, separate the remaining `~1.19e-3` error into trajectory, NQC target,
ringdown amplitude/derivative, and time/phase-convention contributions on
independent binaries. The improved stability makes those comparisons more
informative. Keep this option experimental until that broader validation is
complete.

## Reproduce

From `/Users/xangma/repos/torchwave`:

```bash
PYTHONPATH=src /Users/xangma/miniconda3/envs/pycbcorig313/bin/python \
  scripts/teob_continuous_matching_experiment.py
```

The comparison records its cache and source hashes. Earlier fixed-event
experiments remain frozen in their original review directory. Two Luna
subagents handled ringdown and comparison work; Spark was unavailable.
