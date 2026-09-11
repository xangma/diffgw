# TorchWave: reviewed Agy fixes and remaining parity gaps

September 11, 2026. Reviewed dirty working tree based on HEAD `1ed8d50db7fe6b2220f86e41082a2f79764f9330`.

Agy was prompted directly to repair parity gaps using published mathematics and author data, with LALSimulation restricted to public binary oracle calls. Codex reviewed its edits, rejected an incorrect phase-formula change, ran independent checks, and completed the remaining harness validation repairs. **Full waveform parity remains incomplete.**

## Accepted changes

- XAS now uses an analytic inspiral amplitude derivative at the intermediate-region boundary instead of a forward difference with step `1e-6`. The PN helper is checked against complex-step differentiation at five configurations, plus batched/scalar equivalence, execution contexts, and parameter-gradient checks. All nine new derivative tests pass. The published Eq. (95) phase blend was already correct before dispatch and remains intact.
- The harness rejects replaced or altered canonical cases, invalid grids and waveform tensors, invalid PSDs, missing strict-mode oracle dependencies, and unphysical overlaps. All mandatory reported metrics are gated. New controls include corrupting only projection PSD mismatches and injecting invalid projection shapes/NaNs.
- Every executable case checks both polarizations and six H1/L1/V1 sky projections at GPS 1126259462. All three named PSDs (`aLIGOZeroDetHighPower`, `CosmicExplorer`, `EinsteinTelescope`) are checked for every component. Detector response and delay references use independent public LAL binary APIs.
- Default CLI acceptance now covers the manifest, checks manifest drift, and fails for missing coverage or failed models. `--analytic-only` explicitly selects the six-case subset. CI runs the strict catalog command and retains its report even after another step fails. The workflow was reviewed locally; remote CI was not executed.
- Manifest, provenance ledger, and the editable completion report now state measured scope and unresolved coverage. Earlier round evidence remains unchanged.

## Final verification

Host: `client-3792.default.vpn.port.ac.uk`; cwd: `/Users/xangma/repos/torchwave`. Python 3.13.9, PyTorch 2.9.1, NumPy 2.3.5, LAL 7.7.0, LALSimulation 6.2.0, PyCBC 2.10.0. Oracle checks use CPU FP64/complex128.

| Check | Result | Evidence |
|---|---|---|
| Full discovered regression suite | **238 passed**, exit 0 | [pytest log](verification/pytest.log) |
| Focused derivative and harness controls | **41 passed**, exit 0 | [focused log](verification/focused-controls.log) |
| Strict analytic acceptance | **6/6 passed**, exit 0 | [JSON](verification/strict-analytic.json) |
| Strict catalog acceptance | **6/16 passed**, exit 1 | [JSON](verification/strict-catalog.json) |
| Manifest coverage | 17 models: 12 have executable cases; 5 lack validated oracle mappings | [coverage](verification/summary.json) |

The normal pytest discovery pattern does not collect `tests/parity_harness.py`; its catalog acceptance is checked separately by the strict CLI above. Passing regressions therefore does not imply passing catalog acceptance.

The analytic cases comprise **four TaylorF2 and two TaylorF2_Eccentric** configurations. Maximum direct complex relative L2 is `1.6394512e-12` over their polarizations and `3.0713944e-7` over their detector projections. These are finite-sample measurements, not certification over the full parameter domain.

Acceptance thresholds remain `1e-6` for direct complex, magnitude, and norm relative errors, and `1e-8` for direct flat/PSD mismatches. Waveforms are compared without amplitude rescaling, on the declared common half-open frequency band `[flow, fhigh)`. TorchWave DC/out-of-band support is checked; the oracle's native endpoint/padding is not treated as a separate support-parity claim. Optimized matches are independent component diagnostics and do not replace direct gates.

## Remaining model gaps

All ten non-analytic catalog cases fail. Each row is one declared configuration; full physical parameters and both-polarization/projection metrics are in the catalog JSON.

| Model | Plus magnitude relative L2 | Plus raw complex relative L2 | Plus optimized match (diagnostic) |
|---|---:|---:|---:|
| IMRPhenomD | 0.0014835356 | 1.0397082 | 0.99999762 |
| IMRPhenomXAS | 0.097437855 | 1.4410288 | 0.99600154 |
| IMRPhenomHM | 0.19730655 | 1.1409813 | 0.95788246 |
| IMRPhenomXHM | 0.11648058 | 1.42775 | 0.99063268 |
| IMRPhenomPv2 | 0.11820471 | 1.2647583 | 0.98914294 |
| IMRPhenomXP | 0.11775149 | 1.7392219 | 0.99024986 |
| IMRPhenomXPHM | 0.1067059 | 1.6271344 | 0.98941267 |
| IMRPhenomD_NRTidal | 3.9485924e-06 | 0.31626089 | 1.00000000 |
| IMRPhenomD_NRTidalv2 | 0.0020219389 | 0.2809333 | 0.99999798 |
| IMRPhenomNSBH | 0.0080057854 | 1.9696263 | 0.99996877 |

The raw complex errors include unresolved phase/frame convention differences. The magnitude errors separately establish amplitude disagreement at these configurations. No empirical amplitude rescaling, fitted phase/time correction, or oracle-trained coefficient adjustment was applied. High optimized overlaps do not close these gaps.

`TaylorF2_HM`, `KerrRingdown`, `SEOBNR_ODE`, `SEOBNR_ROM`, and `TEOBResumS` lack independently validated oracle mappings in this harness. `UNRESOLVED_ORACLE` does not claim that external references cannot exist. The native ROM's agreement with its own generator does not establish agreement with LAL SEOBNRv4_ROM.

The XAS derivative fix does not materially close its magnitude residual: the five diagnostic configurations still show roughly 3.2–9.7% magnitude relative L2 error. Complete piecewise-amplitude continuity and the normalization/collocation cause of the residual remain unproved. Further model repairs require a source-supported derivation and independent checks; no full-parity completion claim is warranted.

## Clean-room and review record

The XAS derivation uses [Pratten et al., arXiv:2001.11412v1](https://arxiv.org/abs/2001.11412v1), Eqs. (68)–(69), (72), (77), and (95), plus the already-present author Mathematica supplement. Its SHA256 remains `94b9334c19f3a7909864891ea278ab9a411e4f33b7b7e17cc01d927056d9ac52`. LAL implementation source was not used, and no fit coefficients or training assets changed in this run.

Seven Agy dispatches covered the waveform and harness phases; [attempt outcomes](delegation/attempts.json) and [initial waveform](delegation/phase-1-xas.md)/[harness](delegation/phase-2-acceptance.md) prompts are retained. Each phase used the Low → Medium → High quality ladder, with one extra interrupted read-loop attempt in the waveform phase. Codex's final repairs were frequency/projection validation, complete metric gating, negative-control/test corrections, and evidence/documentation reconciliation.

Relative to the retained pre-dispatch snapshot, only six existing files changed, alongside the new derivative tests, harness controls, and this review bundle. All 155 baseline files remain present; all 63 historical review/evidence files checked are unchanged. See [baseline preservation](verification/baseline-preservation.json), [patch against the dirty baseline](verification/changes-from-baseline.patch), and [checksums](verification/sha256.json). The full recoverable baseline and Agy logs are retained at `/Users/xangma/.codex/agy-runs/torchwave-20260911-035215`. No commits, pushes, or pull requests were made.

Exact final commands, environment overrides, process IDs, timestamps, exit codes, and logs are recorded in [commands.json](verification/commands.json). The before/after diagnostic scripts and JSON are retained in the verification directory. All launched checks and Agy processes have finished.
