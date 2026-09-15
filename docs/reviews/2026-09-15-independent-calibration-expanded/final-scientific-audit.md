# Final scientific audit: expanded NR33 external validation

**Date:** 2026-09-15  
**Scope:** read-only audit of the frozen expanded validation, its quality replay, and its frozen prediction/target arrays. No fit, refit, waveform download, or LAL rerun was performed for this audit.

## Result

The frozen validator completed five external, holdout, highest-resolution N3 cases:

| target q | case | NR amplitude relative L2 |
|---:|---|---:|
| 3.5 | `SXS:BBH:2483/Lev5/N3` | 0.530415% |
| 4.5 | `SXS:BBH:2484/Lev5/N3` | 0.609547% |
| 5.5 | `SXS:BBH:2486/Lev5/N3` | 0.676441% |
| 7.0 | `SXS:BBH:2491/Lev5/N3` | 0.748382% |
| 10.0 | `SXS:BBH:0185/Lev3/N3` | 1.111440% |

The quality replay covers 5 of 8 requested targets. The `q=1.2` candidate failed its extraction and resolution gates; `q=1.8` and `q=2.5` were absent from the catalog search. Therefore these results make no general parity claim over the requested target grid.

## Independent metric recomputation

I loaded only `validation/nr33-external-predictions.json` and `validation/nr33-external-targets.json` and recomputed the reported geometric amplitude metric as

```
||A_pred - A_target||_2 / ||A_target||_2
```

The prediction and target case IDs, input NPZ hashes, and frequency grids agree exactly for all five rows. The independently computed numerators, denominators, and ratios agree with the corresponding validation report values to at most `3e-18` in the ratio. This is the raw geometric extraction metric; it is distinct from the normalized training amplitude metric.

## LAL frequency and mode contract

All five rows use `SimIMRPhenomXHMFrequencySequenceOneMode`, mode `(3,-3)`, XHM release `122022`, LAL `7.7.0`, and LALSimulation `6.2.0`. For each row, I independently converted the frozen `frequency_Mf` grid using the recorded total mass and 4.925490947641267e-6 seconds per solar mass. The resulting physical-frequency arrays exactly equal the stored LAL arrays (`np.array_equal`; zero maximum absolute difference), with matching lengths and no frequency extrapolation.

## Freeze and provenance checks

- Calibration digest before and after validation: `ade377c62a57a5d9cff513e34f22b0fe43677177a0c1577a9a4af5ad3b8aa561` (both sides, and the pinned expected value).
- All five selected rows are literal `holdout` rows; none is a training row or an SXS training alias.
- The validation report binds the qualified external manifest, quality report, protocol, source fit manifest, prediction artifact, and target artifact by SHA256. The prediction and target artifacts bind the same calibration digest.
- The validation freeze records `calibration_unchanged: true` and the same five heldout IDs. These checks establish that the validation used the frozen calibration and did not mutate or refit it during the run; they do not turn this finite sample into a convergence or universal-accuracy proof.

For the five external cases, the recorded fit-vs-NR / fit-vs-LAL percentages are respectively 0.530/0.695, 0.610/0.469, 0.676/0.482, 0.748/0.578, and 1.111/2.210 (the corresponding NR-vs-LAL value for `q=10` is 1.248). These are scalar report values only; no arrays are reproduced here.

## Limits of interpretation

This is an amplitude-only, finite-band check of five external holdouts. It does not evaluate complex-waveform mismatch, establish all-target coverage, or prove catalog-wide NR/LAL parity. The `q=10` row is above 1% for fit-vs-NR and has the largest fit-vs-LAL discrepancy among these rows, so it should remain visible in downstream summaries rather than being treated as a uniformly representative case.

**Audited inputs:** `validation/nr33-external-validation.json`, `validation/nr33-external-predictions.json`, `validation/nr33-external-targets.json`, `quality-replay.json`, `qualified-replay.json`, and `validation-freeze.json` in this directory.
