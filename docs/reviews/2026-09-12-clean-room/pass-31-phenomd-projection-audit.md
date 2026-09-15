# Pass 31: IMRPhenomD projection holdout audit

This was a bounded clean-room audit of the three remaining aligned-spin
IMRPhenomD holdout failures.  The comparison used only the TorchWave runtime,
the public LALSimulation binary waveform API as an external oracle, the
published PhenomD papers, and the independent detector geometry tests already
in the repository.  No LAL, PyCBC, or TEOBResumS implementation source was
read or copied, and no residual was used to fit a runtime constant.

## Reproduction

The strict holdouts were regenerated with
`tests/parity_harness.py --strict --scope=holdout` in the configured FP64
environment.  The direct polarization and detector-projected metrics were:

| case | direct complex L2 | magnitude L2 | H1 sky0 complex L2 | flat mismatch |
| --- | ---: | ---: | ---: | ---: |
| `IMRPhenomD_holdout_0` | `1.2321902262e-5` | `1.5598413623e-6` | `1.2443719577e-5` | `7.59e-11` |
| `IMRPhenomD_holdout_1` | `8.9629199371e-7` | `1.4005178523e-7` | `1.0135173461e-6` | `4.02e-13` |
| `IMRPhenomD_holdout_2` | `9.6293421814e-7` | `2.8454775770e-7` | `1.1433393422e-6` | `4.63e-13` |

The direct plus and cross channels have the same residuals to displayed
precision.  The projected failures track those same residuals: projection
changes the L2 value by the expected linear combination of plus and cross, but
does not introduce an independent frequency-dependent error.  The flat and
named-PSD mismatches remain many orders of magnitude below the `1e-8` gate,
and the optimized matches exceed `0.999999999997` in all three cases.

## Diagnosis

The complex residual is primarily a small waveform calibration discrepancy,
not a detector convention discrepancy.  For holdout 0, the TorchWave/LAL
amplitude ratio is approximately 1.0 at 20--40 Hz and rises to about
1.0000346 by 320--512 Hz; its unwrapped phase difference rises smoothly from
near zero to roughly `8.5e-5` rad.  Holdouts 1 and 2 show the same smooth,
smaller pattern.  A detector time delay or antenna-factor convention would
affect the projected output without changing the direct polarizations; it
cannot explain these direct residuals.

The detector path is independently constrained by the published detector
geometry and sidereal-time equations in `tests/test_clean_room_detectors.py`.
Those tests cover the antenna tensor, GPS-to-GMST conversion, Earth-center
delay, frequency-domain projection, sky batching, and gradients.  The parity
harness also computes the public-oracle projection as
`(F_plus*h_plus + F_cross*h_cross)*exp(-2*pi*i*f*delta_t)`, matching the
TorchWave projection contract.  No source-supported projection defect was
found.

The remaining waveform discrepancy is consistent with the unresolved
PhenomD calibration provenance recorded in
`pass-18-phenomd-assets-primary.md`: the papers print rounded Table V values
and do not specify the full-precision coefficient supplement, QNM
interpolation knots/order, or an absolute phase/time gauge.  Replacing the
stored extra digits, changing the QNM interpolation, or altering the phase
anchor based on these residuals would be an unsupported fit rather than a
clean-room implementation correction.

## Decision

No runtime or test change is justified by the permitted sources.  The strict
holdout failures remain documented as an unresolved source-data/calibration
blocker.  The next admissible step would be an independently licensed,
author-provided coefficient/QNM asset or a new primary-source specification
that fixes those omitted details; until then, changing the projection or
waveform constants would not be evidence-based.
