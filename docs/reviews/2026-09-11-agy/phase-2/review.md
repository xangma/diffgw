# Phase 2: parity acceptance repairs

The final results and limitations are in [the review report](../report.md), with complete machine-readable [catalog evidence](../verification/strict-catalog.json).

The harness now validates exact case IDs, model mappings, physical parameters, frequency grids, and gates against fresh copies of serialized canonical specifications. Six substituted copies of a single TaylorF2 case no longer pass. Invalid waveform shapes, non-finite values, invalid grids and PSDs, and unphysical overlaps fail validation. A reported mandatory PSD mismatch cannot be ignored.

Every executable case evaluates both polarizations and H1/L1/V1 at two sky positions, using GPS 1126259462 and independent public LAL binary detector-response and time-delay APIs. All three named PSDs are gated for each polarization and projection. Missing LAL in strict mode, catalog drift, missing cases, and unresolved oracle mappings prevent catalog acceptance.

The default CLI scope is the 17-model manifest: 16 executable cases cover 12 models, and five models are explicitly unresolved. The analytic subset requires `--analytic-only`. Full catalog acceptance is a separate failing pytest entrypoint and CLI command; the CI workflow runs the strict command and uploads its report even if regression tests fail. CI execution itself was not tested remotely.

Codex verification found defects after each Agy attempt. After the third quality attempt, Codex moved frequency validation ahead of oracle generation, made invalid projections return failure before metric computation, restored gating of every reported required metric, corrected the projection-only PSD negative control, and added invalid-grid/projection regressions.

Final regression results: 238 passed. Strict analytic acceptance: 6/6 passed (four TaylorF2 and two TaylorF2_Eccentric cases). Strict catalog acceptance: 6/16 passed, ten failed, five unresolved models, exit code 1. Both-polarization analytic complex relative L2 is below 1.64e-12; detector-projection complex relative L2 is below 3.08e-7. These are finite-sample results and do not establish parameter-domain parity.
