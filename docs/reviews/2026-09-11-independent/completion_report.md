# Independent waveform reconstruction — current verification

Strict catalog parity remains incomplete. The full repository suite passes **273 tests**; the separate strict catalog passes **6 of 16 declared cases**. Across the 17-model manifest, two models pass their declared cases, ten fail, and five have unresolved oracle coverage. Acceptance gates and case inventory were not changed.

## Implemented and measured

- Replaced runtime Kerr QNM table interpolation with PyTorch Chebyshev evaluation trained on independent Kerr eigenproblem solves. No runtime QNM file or solver dependency. The 2013 unused validation spins have maximum frequency/damping relative errors of about 1.01e-11/2.15e-11 against independent solves. [Derivation and reproducibility](qnm.md).
- Reconstructed IMRPhenomXAS amplitude from published equations and the original author Mathematica fits: stable collocation, amplitude normalization, spin conventions and joined intermediate amplitude. [Amplitude record](xas/audit_record.md).
- Corrected XAS phase intervals, calibration-value frequency, intermediate Lorentzian convention and ringdown join. [Phase record](xas/phase_reconstruction.md).
- Added meaningful normalized strain batching, full-waveform gradient checks and extrapolated one-sided phase derivative checks. All 273 repository tests pass; [log](verification/full-pytest-final.log).
- All 16 declared catalog cases generate finite, nonzero native waveforms with LAL/PyCBC imports blocked and `LAL_DATA_PATH` unset. This checks runtime independence, not parity. [Results](verification/native-runtime.json).

Across 16 separate XAS mass/spin diagnostic configurations with q from 1 to 100, production amplitude relative L2 error is at most **1.07e-13**. After a diagnostic affine phase alignment, the maximum phase-shape residual is **2.25e-11 radians**. Raw complex L2 errors remain **1.22–1.82**; affine alignment is not used for acceptance or runtime calibration. [Measurements](verification/xas-production-holdouts.json).

The strict catalog XAS case has amplitude relative L2 error **1.79e-14** and raw complex relative L2 error **1.4042**. PhenomD has raw complex error **0.0045942** and amplitude error **5.2231e-5**. Both fail. The higher-mode, precessing, tidal and NSBH cases also fail. See the complete [strict report](verification/strict-catalog-current.json).

## Source gaps and remaining work

The [original XHM paper, Sec. II B and Appendix C](https://arxiv.org/html/2001.10914v2#A3) describes additional dominant-mode time calibration and a fit accounting for the difference between curvature and strain peak times. The retrieved XAS author fits and XHM ancillary phase notebook did not identify the needed dominant-mode timing-fit coefficients. A named relative (3,2)-mode time shift in the notebook is a different quantity. The missing dominant-mode fit cannot be replaced with a measured oracle residual while maintaining the requested implementation boundary.

The [XHM author ancillary archive](https://arxiv.org/src/2001.10914v2/anc) was retrieved and 91 author assets extracted for research. Archive SHA256: `cc05e7dbff7bf0c42a10b7da1f4840eefc55b252168d6bfb085155a7bdc5318e`. This search does not establish that no other public author release exists. The archived harmonic fits are useful for future XHM reconstruction, but do not establish current XHM parity.

The PhenomD journal coefficient precision is insufficient to uniquely specify behavior at the strict numerical tolerance under the documented rounding-interval assumption. The independent sensitivity experiment demonstrates this ambiguity; it does not show that rounding explains every remaining error. An unrounded author calibration release has not been located. [Source search and measured sensitivity](phenomd/research_record.md).

The concrete source requirements are the original PhenomX dominant-mode time calibration and more precise PhenomD calibration information. Higher-mode, precessing and tidal reconstruction still needs further implementation and validation; obtaining those assets alone would not certify the remaining catalog.

## Execution and provenance

Agy implemented and revised the bounded phases using native file tools. Codex reviewed diffs, protected pre-existing changes, performed public-output diagnostics and ran all numerical tests. Agy was given primary mathematical sources and observable compatibility specifications; no LAL code, internal tables or port implementations were supplied in this XAS phase. Codex had inspected LAL source separately during earlier review, so no claim of zero source exposure by all participants is made.

The XAS phase used Low, Medium and High attempts. Each saved its changes and a summary but the CLI returned a network error. The saved work was recovered and independently checked; an ERROR result was not treated as successful verification. No Agy job remains running. No commit, push or PR was performed.

## Reproduce

From the repository root, with the relevant test dependencies installed:

```sh
PYTHONPATH=src python -m pytest -q tests/
PYTHONPATH=src python tests/parity_harness.py --strict --json-out /tmp/torchwave-strict.json
PYTHONPATH=src python docs/reviews/2026-09-11-independent/verification/check_native_runtime.py --json-out /tmp/torchwave-native.json
PYTHONPATH=src python docs/reviews/2026-09-11-independent/verification/check_xas_holdouts.py --json-out /tmp/torchwave-xas.json
```

The strict harness is expected to exit nonzero for the current failures. Oracle versions and full model conventions are captured in the strict JSON metadata. Holdout comparison requires LAL only as an external test oracle; native generation blocks it.
