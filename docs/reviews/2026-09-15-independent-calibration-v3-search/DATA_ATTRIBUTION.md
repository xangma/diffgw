# NR33 calibration data attribution and provenance

## SXS sources

The seven qualified NR inputs are SXS/SpEC simulations. Five were retained from the prior screened set; two were added as replacements after the metadata and waveform quality gates were frozen:

| role | SXS simulation | CaltechDATA record | DOI |
|---|---|---|---|
| reused | SXS:BBH:2331 | [aq9e2-0ye86](https://data.caltech.edu/records/aq9e2-0ye86) | [10.26138/SXS:BBH:2331v3.0](https://doi.org/10.26138/SXS:BBH:2331v3.0) |
| reused | SXS:BBH:1166 | [27s13-fe162](https://data.caltech.edu/records/27s13-fe162) | [10.26138/SXS:BBH:1166v3.0](https://doi.org/10.26138/SXS:BBH:1166v3.0) |
| reused | SXS:BBH:2265 | [258wv-a5t45](https://data.caltech.edu/records/258wv-a5t45) | [10.26138/SXS:BBH:2265v3.0](https://doi.org/10.26138/SXS:BBH:2265v3.0) |
| reused | SXS:BBH:2485 | [5mjz7-2gs59](https://data.caltech.edu/records/5mjz7-2gs59) | [10.26138/SXS:BBH:2485v3.0](https://doi.org/10.26138/SXS:BBH:2485v3.0) |
| reused | SXS:BBH:3619 | [6fw56-9q785](https://data.caltech.edu/records/6fw56-9q785) | [10.26138/SXS:BBH:3619v3.0](https://doi.org/10.26138/SXS:BBH:3619v3.0) |
| added q≈6 holdout | SXS:BBH:2489 | [q7rzh-bzw94](https://data.caltech.edu/records/q7rzh-bzw94) | [10.26138/SXS:BBH:2489v3.0](https://doi.org/10.26138/SXS:BBH:2489v3.0) |
| added q≈8 training | SXS:BBH:2707 | [gwmd0-0rd53](https://data.caltech.edu/records/gwmd0-0rd53) | [10.26138/SXS:BBH:2707v3.0](https://doi.org/10.26138/SXS:BBH:2707v3.0) |

These identities, selected file URLs, and checksums are recorded in `nr33-manifest-round1.json`, `nr33-new-manifest-round1.json`, and `nr33-qualified-manifest.json`. The raw H5/JSON files were fetched from the external CaltechDATA URLs and remain external to this repository; local raw and derived-file paths are recorded in the manifests.

The SXS database describes its waveform data as free for use and requests acknowledgement of the first scientific publication for each simulation: [SXS Gravitational Waveform Database](https://data.black-holes.org/waveforms/). Its documentation describes outermost-extraction and extrapolated N=2,3,4 products and advises choosing extrapolation order using convergence tests: [SXS waveform documentation](https://data.black-holes.org/waveforms/documentation.html). These are the source project's public use and attribution statements. The CaltechDATA API metadata captured for each of the seven records contains no per-record `license` value; this note makes no further licensing claim.

## Decoder and fit provenance

- Extraction used SXS package version `2026.0.3`. Standalone N2 used `scripts.prepare_nr33_calibration_data.extract_case` (script SHA-256 `721e9395aab7f5d15dc82cba2670b70e538825d81a7e681cec386c23a0796496`); embedded N3 used `scripts.prepare_nr33_calibration_v2.extract_embedded_case` with the explicit same-level `/Strain_N3.dir` group (script SHA-256 `777b4492fbd6d73d2ed6202b128be0873b00bf7d0c59d14f011a873e53869c7c`).
- The calibration rule is explicitly **NR-only quality selection** in `nr33-calibration.json`. The five training cases are q≈1.5, 2, 4, 5, 8; q≈3 and q≈6 are holdouts.
- Inputs were frozen before fitting at `2026-09-15T11:46:15.554074+00:00`; fitting completed at `2026-09-15T11:46:16.303813+00:00`. Both timestamps are recorded in `nr33-calibration-freeze.json`. The frozen calibration SHA-256 is `ade377c62a57a5d9cff513e34f22b0fe43677177a0c1577a9a4af5ad3b8aa561`.
- LAL comparisons were generated after that freeze and are validation diagnostics only. They did not provide calibration training data. Historical artifacts and the superseded q≈3/q≈6 source choices remain preserved in the earlier review directories.
