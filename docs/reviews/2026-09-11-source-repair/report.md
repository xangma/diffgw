# Source repair verification — 11 September 2026

**Strict LAL parity is not achieved.** This iteration improves IMRPhenomD's numerical ringdown frequencies and gradients, but leaves calibration precision and time-origin compatibility unresolved.

## Changes and clean-room review

Agy researched the primary papers and implemented a numerical QNM replacement. The IMRPhenomD phase used Low, Medium, a Medium resumption after a denied web read, and High: three quality attempts plus one additional call. Codex rejected an incorrect constant phase shift, checked the actual diffs and numerical outputs, and retained the original reference-phase behavior. No permission-bypass flags were used.

After the final attempt, Codex took over the remaining validation and repair. A newer independent author dataset exposed inaccurate extremal values in the older table. The final implementation uses the unmodified 2025 Lo/Sabani/Cardoso data, an independently constructed natural cubic spline on the signed nonuniform grid, and ordinary cached constants compatible with inference followed by differentiation. SciPy is used only as a test reference. See [source provenance](phenomd/source_record.md).

Changed code: `src/torchwave/approximants/imrphenomd.py`, new `kerr_qnm_interp.py`, two new numerical data files, the package-data rule in `pyproject.toml`, and new `tests/test_phenomd_qnm_regression.py`. Existing work was preserved: only the two intended existing files differ from the 185-file pre-dispatch baseline. The canonical harness, thresholds, prior verification and historical review artifacts are unchanged.

Codex inspected LAL source/documentation for diagnosis. Agy was given abstract compatibility requirements, public-binary measurements and independent primary-source materials. No LAL code, numerical coefficient tables, or source-specific implementation outline was relayed or copied into the implementation. The papers' missing numerical precision was not reconstructed by fitting oracle residuals.

## Verification

- **253 tests passed** in 15.81 seconds; the new focused set has 21 passing checks including the six existing D tests.
- Every primary-data node, off-node values and derivatives were checked against an independent SciPy spline. Cold inference caches, signed-spin gradients, float32/64 batches and normalized waveform derivatives/polarization scaling passed.
- The wheel built successfully. Both bundled data files match their source hashes. IMRPhenomD generation and differentiation also succeeded when importing from the extracted wheel outside the repository.
- **Strict catalog: 6 of 16 cases pass; 10 fail.** The five previously unresolved oracle mappings remain unresolved. Inventory and manifest guards pass.
- The five additional frozen D diagnostic/holdout cases all still fail the original raw-complex and mismatch requirements. No amplitude scaling or time/phase optimization is applied to those acceptance metrics.

The plus-polarization amplitude error improved across all five D cases. The complex error still reflects the unresolved phase discrepancy:

| Case | Before magnitude L2 | After magnitude L2 | After raw complex L2 |
| --- | ---: | ---: | ---: |
| IMRPhenomD_diagnostic | 0.0014835356 | 5.2230718e-05 | 1.0397975 |
| IMRPhenomD_holdout_0 | 0.002847214 | 0.00011997901 | 1.1157741 |
| IMRPhenomD_holdout_1 | 0.002199102 | 8.6774633e-05 | 0.82312595 |
| IMRPhenomD_holdout_2 | 0.013856253 | 0.00046648345 | 1.2335166 |
| IMRPhenomD_holdout_3 | 0.0064045348 | 0.00014181759 | 1.1533603 |

## Remaining work

The original author archives supply rounded Table V and remnant fits, and do not uniquely determine the reference implementation's absolute time origin. An independent full-precision calibration release was not located in this search. The user's decision about allowing LAL-derived numerical calibration data in the clean-room specification is pending; none has been transferred. Other failing model families remain as identified in the full catalog JSON. Passing unit tests must not be interpreted as waveform parity.

Evidence: [test log](verification/final-pytest.log), [full catalog](verification/final-catalog.json), [D holdouts](verification/final-d-holdouts.json), [baseline preservation](verification/baseline-preservation.json). Dispatch prompts and logs are retained at `/Users/xangma/.codex/agy-runs/torchwave-sources-20260911-072309`.
