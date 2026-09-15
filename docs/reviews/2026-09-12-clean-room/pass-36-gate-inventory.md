# Pass 36 clean-room gate inventory

Date: 2026-09-13  
Scope: strict catalog and deterministic holdout parity gates only.

This bounded inventory reviewed `tests/parity_harness.py`, the focused harness
tests, `docs/model_manifest.json`, the Pass 34, Pass 35, and Pass 36 reports,
and the fresh Pass 35 final artifacts in
`docs/reviews/2026-09-12-clean-room/verification/pass-35-final/`. No waveform
implementation source from LAL, PyCBC, TEOBResumS, or another external
implementation was inspected or copied. No constants were fitted to oracle
outputs and no NR run was performed.

The fresh artifacts contain 11/21 passing catalog cases and 4/13 passing
holdout cases. Thus there are 19 nonpassing strict cases: ten catalog entries
(nine `FAILED` and the unresolved Kerr entry) and nine holdout entries. Both
JSON artifacts parse successfully and report `inventory_valid: true`, empty
`inventory_errors`, and empty `manifest_drift_errors`. The direct complex
relative L2 gate is `1e-6`; the flat and named PSD mismatch gates are `1e-8`.
The harness also checks detector projections. In particular, the two PhenomD
holdouts with polarization L2 below `1e-6` still fail because their largest
projection L2 is `1.0135173461056485e-6` and `1.1433393421595667e-6`.

## Nonpassing strict cases

The value in the metric column is the larger of the plus/cross polarization
L2 and the six detector projection L2 values, respectively, as recorded by the
fresh artifact. It is evidence of the gate failure, not an oracle-fitted
correction.

| Scope and case IDs | Fresh L2 (polarization; projection) | Missing evidence class | Gate conclusion |
| --- | --- | --- | --- |
| `catalog: IMRPhenomD` has no nonpassing case | — | — | Its catalog case passes; its three holdouts are listed below. |
| `catalog: IMRPhenomHM_diagnostic`; `holdout: IMRPhenomHM_holdout_5` | `0.004540949061216069; 0.00451217403413887` / `0.01946459733434688; 0.01915734346459294` | **unresolved convention mapping** | The public HM equations do not determine the remaining phase/time gauge, mode-sign, mixing, and transition mapping needed for strict full-polarization parity. |
| `catalog: IMRPhenomXHM_diagnostic`; `holdout: IMRPhenomXHM_holdout_6` | `0.3820011689068622; 0.3794817271438915` / `0.6586575063318627; 0.6533452980229569` | **incomplete physical dynamics/driver/calibration** | The remaining higher-mode calibration/preprocessing, ringdown and mixing data are not fully public in a reproducible form. |
| `catalog: IMRPhenomPv2_diagnostic`; `holdout: IMRPhenomPv2_holdout` | `0.2572343808555074; 0.2569942337695498` / `1.418307000762992; 1.450632228360452` | **unresolved convention mapping** | The opening-angle correction is already applied. Exact source frame, remnant/spin convention, reference phase, and degenerate continuation remain unspecified. |
| `catalog: IMRPhenomXP_diagnostic`; `holdout: IMRPhenomXP_holdout` | `0.072548555371545; 0.07242449232148719` / `0.4032712778793853; 0.5109112921701383` | **unresolved convention mapping** | The exact MSA/version, frame and remnant choices, phase gauge, and invalid-row continuation are not fixed by the available public equations. |
| `catalog: IMRPhenomXPHM_diagnostic`; `holdout: IMRPhenomXPHM_holdout` | `0.1609264163494752; 0.1611235388005673` / `1.185113526883755; 1.277046119704339` | **unresolved convention mapping** | The carrier-level precession/frame convention remains unresolved; higher-mode calibration cannot repair a carrier-level mismatch without source-backed data. |
| `catalog: IMRPhenomNSBH_diagnostic`; `holdout: IMRPhenomNSBH_holdout` | `2.951698015917103e-05; 2.964991436215297e-05` / `0.001757972684430281; 0.001758092823245954` | **published coefficient/table precision** | The remaining residual requires complete high-precision NSBH fitted coefficients and the authoritative low-Λ correction/gamma convention; the public displayed precision is insufficient for the strict complex gate. |
| `holdout: IMRPhenomD_holdout_0`, `IMRPhenomD_holdout_1`, `IMRPhenomD_holdout_2` | `1.232190226217208e-05; 1.244371957656429e-05` / `8.96291993710199e-07; 1.0135173461056485e-06` / `9.629342181428129e-07; 1.143339342159567e-06` | **published coefficient/table precision** | The published unrounded PhenomD coefficient/QNM interpolation data needed for these off-catalog points are not available at the precision required by the direct and projection gates. |
| `catalog: KerrRingdown_diagnostic` | — (`UNRESOLVED_ORACLE`) | **independent waveform oracle** | No independent full ringdown waveform reference fixes excitation, overtone content, attachment, distance normalization, and phase conventions for this case. |
| `catalog: SEOBNR_ODE_diagnostic` | `1.020698741625642; 1.020583698846942` | **incomplete physical dynamics/driver/calibration** | The public equations do not fully specify the calibrated EOB dynamics, flux, NQC and merger/ringdown driver required for this implementation-level comparison. |
| `catalog: SEOBNR_ROM_diagnostic` | `1.001903219203762; 1.002228437708556` | **incomplete physical dynamics/driver/calibration** | A strict ROM requires the model's calibrated reduced-order basis/surrogate data and versioned construction, which are not independently reconstructible from the public equations alone. |
| `catalog: TEOBResumS_diagnostic` | `1.000916156109546; 1.00074237227128` | **incomplete physical dynamics/driver/calibration** | The reconstructed uncalibrated dynamics still lack the inherited tidal/spin resummation and merger prescription data needed for strict parity. |

The `TaylorF2_HM` entry is intentionally absent from the nonpassing list: its
declared catalog case passes the dominant-mode-only strict check. Its manifest
status records that a full higher-mode independent oracle is unresolved; that
status is a coverage limitation, not a failed strict case.

## Proven local manifest defect and correction

The manifest's stored validation maxima were stale relative to the fresh
Pass 35 artifacts for `IMRPhenomHM` (catalog and holdout), `IMRPhenomNSBH`
(catalog and holdout), and `TEOBResumS` (catalog). The harness's existing
manifest check only compared the set of model names, so it correctly reported
no name drift while missing stale per-model validation measurements. The
manifest maxima were updated from the fresh artifact records. A focused
regression test now derives each model/scope maximum from those artifacts and
compares it with `docs/model_manifest.json`, preventing this class of stale
measurement from passing unnoticed.

This is the only proven local harness/manifest defect. It does not change
runtime waveform code or strict thresholds. The remaining 19 nonpassing cases
are evidence-blocked by the classes above; no source-backed runtime correction
is justified by the reviewed material.

## Validation

- `python -m json.tool docs/model_manifest.json` and parsing all JSON files in
  `verification/pass-35-final/`: passed.
- `PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q tests/test_parity_harness_controls.py tests/test_lal_parity.py`: **64 passed**.
- `git diff --check`: passed.
