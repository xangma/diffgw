# Pass 36: next clean-room parity gap audit

Date: 2026-09-13

## Decision

No remaining strict LAL parity failure has a concrete implementation
correction that is fully supported by the public primary equations or data
currently available to this repository. No runtime, test, manifest, or strict
gate change is made.

The current strict status remains 11/21 catalog cases and 4/13 frozen holdout
cases. The failed cases are already localized by Passes 31--35; the remaining
unknowns are model calibration, convention, or independent-reference inputs.
Choosing values from the public binary residual would be oracle fitting.

## Bounded audit

I reviewed the current `docs/model_manifest.json`, the latest strict status in
`pass-34-summary.md` and `pass-34-failure-triage.md`, all Pass 31--35 reports,
and the implementation/tests for the remaining model families. The audit did
not inspect or copy LAL, PyCBC, TEOBResumS, or other waveform implementation
source, did not fit any constant to LAL output, and did not run NR.

The source-backed candidates are exhausted as follows:

| Failure | Current evidence | Exact information still needed |
| --- | --- | --- |
| IMRPhenomD holdouts | Pass 31 localized the error to carrier calibration precision; Pass 32 found no admissible machine-readable asset. The published equations and rounded tables do not reproduce the native full-precision fit. | An admissibly licensed release of the unrounded PhenomD calibration vectors and the precise Kerr-QNM interpolation knots, interpolation order, and endpoint rule. |
| IMRPhenomPv2/XP/XPHM | Passes 32--34 retain the source-derived Pv2 opening-angle correction. The residual remains carrier-level for XPHM and the precession papers leave selected MSA/version, remnant, frame, and degenerate-row behavior unspecified. | An independently published coprecessing XAS/XP reference that fixes the exact precession version, remnant branch, frame/Table-IV offsets, reference-frequency behavior, and invalid/degenerate-row continuation. |
| IMRPhenomHM/XHM | Pass 33 localizes XHM to higher-mode reconstruction; Pass 35 localizes HM's largest residual to `(2,1)` without finding a missing printed equation. The papers do not fix all fit-target preprocessing, phase/time gauge, or spherical/spheroidal mixing metadata. | Complete author calibration metadata for intermediate and ringdown targets, their preprocessing and absolute gauge, plus the `(3,2)`/`(4,3)` mixing convention. |
| IMRPhenomNSBH | Pass 35 rechecked classification, remnant frequency, disruption windows, ringdown width, and the published branches. The residual is phase-gauge/precision limited after the sentinel correction. | An explicit reference-frequency/time-phase convention for this API and a complete high-precision release of the fitted NSBH coefficients, including the ambiguous low-`\Lambda` notation. |
| KerrRingdown | The strict case is unresolved for lack of an independent complete waveform oracle, rather than a demonstrated equation defect. | Independent excitation/twin-mode and overtone data with distance, energy, angular, and phase conventions. |
| SEOBNR/TEOB/ROM | Passes 30--34 establish equation-level components but not a uniquely complete calibrated waveform assembly. | A declared target version with calibrated dynamics/flux, NQC and merger attachment data, remnant and mode-selection metadata, and independent waveform references. |
| TaylorF2_HM extra modes | Passes 30--35 exhaust the leading source-backed inventory through `l=4`; `(4,1)` and `(5,5)` remain blocked at the source-mode/native negative-`m` SPA convention boundary. | A primary source or author data explicitly mapping those source-frame coefficients into this API's stored mode and SPA phase convention. |

No candidate outside those rows supplied a source-defined replacement value or
identity. In particular, the current MSA final-spin rejection, XHM numeric
threshold deferral, TEOB prototype choices, and Kerr waveform normalization
are explicit unresolved boundaries rather than accidental defaults that can
be repaired from the printed equations alone.

## Validation

The focused source-derived and reconstruction suites for the audited areas
remain green:

```text
PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q \
  tests/test_clean_room_phenomd_baseline.py \
  tests/test_clean_room_msa.py \
  tests/test_clean_higher_modes.py \
  tests/test_clean_room_nsbh.py \
  tests/test_taylorf2_hm.py \
  tests/test_clean_room_eob_published_driver.py \
  tests/test_clean_room_teob_postmerger.py
```

Result: **112 passed in 6.43 s**. `git diff --check` was also run; no source
or test file was changed by this pass.
