# Pass 33: post-patch IMRPhenomPv2 residual diagnosis

Date: 2026-09-13

This pass measured the implementation after the Pass 32 opening-angle and
zero-cone differentiability corrections. It used only the public LAL binary
waveform API as a black-box comparator and the published precession equations
already recorded in `precession.md`. No LAL, PyCBC, TEOBResumS, or other
implementation source was inspected or copied. No residual was used to fit a
runtime coefficient.

## Measurements

The fresh strict runs are archived in
`verification/pass-33-pv2/catalog.json` and `holdout.json`. The commands
returned the expected strict-suite failure status because other cases remain
unresolved.

| Case | Polarization complex L2 (max) | Detector complex L2 (max) | Result |
|---|---:|---:|---|
| `IMRPhenomPv2_diagnostic` | `2.5723438e-1` | `2.5699423e-1` | FAILED |
| `IMRPhenomPv2_holdout` | `1.4183070` | `1.4506322` | FAILED |

The catalog remains `11/21` and the frozen holdout remains `4/13`. The
diagnostic values are numerically unchanged at the reported precision from
the preceding strict snapshot; the Pass 32 correction is retained because it
is source-supported, but it does not close this model-level parity gap.

## Stage diagnosis

The aligned carrier is not the dominant source of the Pv2 residual. On the
same intrinsic configurations, independent public `IMRPhenomD` black-box
comparisons give TorchWave complex L2 errors of approximately `1.22e-5`
(catalog configuration) and `8.37e-6` (holdout configuration), with flat
mismatches below `8e-11`. In contrast, replacing the aligned carrier with
the precessing twist produces `2.52e-1`/`2.57e-1` catalog polarization errors
and `1.34`/`1.42` holdout errors.

The residual is already large at the reference frequency in the holdout:
the local complex phase ratios at 35 Hz are approximately `-1.66` rad (plus)
and `-1.73` rad (cross), with amplitude ratios `1.17` and `1.44`. The
catalog residual is smaller at 20 Hz and grows strongly above roughly 100 Hz,
indicating both a reference-frame/projection discrepancy and accumulated
precession-angle evolution. The independent magnitude comparison separates
the effects: at the catalog point the plus/cross magnitude errors are
`6.81e-2`/`9.02e-2`, while at the holdout they are
`2.18e-1`/`3.91e-1`.

Thus the largest remaining Pv2 issue is the prescription-specific reference
frame and precession projection, with angle evolution also contributing at
higher frequencies. The current direct Pv2 path does not have a uniquely
source-complete correction for that frame in the audited primary material;
the existing record explicitly leaves its Pv2 reference-frame mapping and
precession-corrected remnant/ringdown prescription unresolved. No speculative
change is made in this pass.

Validation: the two strict harness runs completed, and the result JSON files
were parsed for the metrics above. `git diff --check`, targeted byte-compilation,
and the three precession/IMRPhenomP test files passed (`77 passed`). No source
files or tests were changed. The clean-room boundary remains intact.
