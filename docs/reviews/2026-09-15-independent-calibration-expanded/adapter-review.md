# External NR33 adapter review

**Scope.** Read-only review of `screen_nr33_external_validation.py`,
`run_nr33_external_validation.py`, their contract tests, the expanded protocol,
and `run_nr33_frozen_validation.py`. No waveform H5 files, LAL implementation,
or runtime code were inspected or changed.

## Result

The numerical path is equivalent to the frozen runner.  Both use the same
`FFTControls`, metadata-driven independent QNM helper, `f >= 0.95 f_ring,33`
selection (with the helper's fixed `f <= 0.3` limit), eta/delta mapping,
polynomial coefficient evaluation, `a022` factor, `ringdown_shape`, and frozen
`sigma_33` (compare `run_nr33_external_validation.py:245-255` with
`run_nr33_frozen_validation.py:41-52`).  The external path adds admission and
hash checks and does not fit or select using residuals.  I found no material
scientific formula error.

| Area | Finding | Consequence |
|---|---|---|
| External admission | `_check_external` requires holdout, N3, declared target, 0.5% target-q tolerance, positive rank, existing NPZ, and matching NPZ hash (`run_nr33_external_validation.py:164-211`). The screen reuses the original per-simulation quality function and requires exactly one declared holdout primary (`screen_nr33_external_validation.py:57-75`). | Compatible with the frozen nonspinning amplitude policy, subject to the protocol's metadata and candidate inventory being truthful. |
| Provenance binding | The runner pins the v3 calibration, expanded protocol, quality policy, original fit manifest, fitter, and QNM helper hashes before generating artifacts (`run_nr33_external_validation.py:133-161,214-240`). Quality admission binds the qualified manifest to the screened input hash and exact selected IDs (`:82-120`). | End-to-end provenance is adequate for a frozen run. |
| Residual auditability | The detached prediction/target JSON stores calibration, external-manifest, and adapter hashes, but not the quality-report hash, protocol hash, or qualified-manifest hash (`run_nr33_external_validation.py:268-278`). The final validation report does store those hashes (`:300-309`). | Keep the final report with the detached artifacts; the detached pair alone cannot establish quality admission. This is a low-severity packaging gap, not a formula or selection error. |
| Standalone screen invocation | `screen()` accepts any caller-supplied `expected_protocol_sha256` that matches the file (`screen_nr33_external_validation.py:25-26`); it does not itself compare that argument with the frozen protocol constant. The runner later rejects non-frozen protocol/quality hashes. | End-to-end runner use remains pinned, but the screen report is not independently v3-pinned when invoked as a standalone CLI. Treat standalone screen output as advisory unless the caller supplies the frozen digest. |

## Follow-up before validation

The coordinator added protocol, quality-report, screened-input-manifest, and original fit-manifest hashes to both detached artifacts before the validation freeze. The packaging gap above is resolved in the frozen run. The standalone screen remains explicitly caller-pinned; the recorded command and end-to-end runner use the fixed protocol digest.

## Required next data/implementation

This adapter validates only the frozen nonspinning `(3,3)` **amplitude** artifact.
Spin dependence, phase, or other modes require new explicitly frozen calibration
coefficients and corresponding NR metadata/quality gates; they are outside this
adapter's current contract.  No implementation change is recommended from this
review.  Existing v3 resolution and N2/N3 gates are reused by the external
screen rather than newly invented here.

