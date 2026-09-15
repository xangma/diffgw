# Expanded NR33 candidate discovery (metadata only)

Date: 2026-09-15. **Provenance update:** the original inventory bytes were lost after an inadvertent rewrite. The authoritative preparation now uses the preserved replacement and explicit replay; see [the incident audit](inventory-incident-audit.md). Candidate order and selections remain those frozen before waveform downloads. This is a read-only expansion around the frozen v3 calibration. It did not download waveform/H5 files, inspect waveform samples, run LAL, or fit coefficients.

## Frozen search scope

The search used the frozen [protocol](protocol.json), including target mass ratios `1.2, 1.8, 2.5, 3.5, 4.5, 5.5, 7, 10`, relative mass-ratio tolerance `0.005`, and the 16 explicit SXS exclusions. Only `SXS:BBH:*` entries were considered. A candidate had to be nondeprecated, lie in its target window, have at least two available Lev values, catalog reference eccentricity `<= 0.001`, and catalog maximum reference spin norm `<= 1e-4`. Ranking is descending highest two available Lev labels, then ascending numeric SXS ID; all ranked entries are retained in `candidate-inventory.json`.

The catalog input is the fresh-checked SXS Catalog Data v3.0.0 snapshot published 2025-05-14, available at [simulations.json](https://raw.githubusercontent.com/sxs-collaboration/sxscatalogdata/v3.0.0/simulations.json). Its SHA256 is `132d38520b3562c8f254b3c9aaf8efd4678428f94dbbabfef4436bc50c1c87c6` (32,618,880 bytes); the separate freshness check found the current release unchanged. This is a bounded search of that catalog and is not a claim that no other SXS source exists.

## Result

There are 181 nondeprecated, unexcluded BBH entries in the eight target windows. Six targets have one or more catalog-eligible candidates; 1.8 and 2.5 have none under the fixed spin/eccentricity gates.

| target q | window | all ranked BBH entries | catalog-eligible | first eligible / rank | exact highest-two metadata fetched |
|---:|---:|---:|---:|---|---|
| 1.2 | [1.194, 1.206] | 16 | 1 | SXS:BBH:0198 / 1 | Lev4, Lev5: pass |
| 1.8 | [1.791, 1.809] | 8 | 0 | absent | none |
| 2.5 | [2.4875, 2.5125] | 27 | 0 | absent | none |
| 3.5 | [3.4825, 3.5175] | 28 | 1 | SXS:BBH:2483 / 9 | Lev4, Lev5: pass |
| 4.5 | [4.4775, 4.5225] | 42 | 2 | SXS:BBH:2484 / 1 | Lev4, Lev5: pass |
| 5.5 | [5.4725, 5.5275] | 9 | 1 | SXS:BBH:2486 / 1 | Lev4, Lev5: pass |
| 7 | [6.965, 7.035] | 15 | 1 | SXS:BBH:2491 / 2 | Lev4, Lev5: pass |
| 10 | [9.95, 10.05] | 36 | 1 | SXS:BBH:0185 / 30 | Lev2, Lev3: pass |

“Pass” here means the fixed metadata gate after fetching each candidate's highest available metadata file; it is not an NR waveform quality screen. Exact metadata were fetched only for the first catalog-eligible candidate in each nonempty target. Alternatives remain catalog-only and are preserved with their catalog metadata and declared file inventories. Thus the two absent-target results and the ranks of uninspected alternatives are preliminary catalog-gate results, not claims that every candidate's highest-Lev metadata has been independently fetched. The six fetched records provide a spot check: their catalog reference spin/e values agree with exact highest-Lev metadata, but that does not prove the catalog field's level provenance for all 181 entries.

## Exact current Caltech records and checks

The six first eligible records were fetched from their current Caltech API records. Record IDs, DOI, raw-record SHA256, and exact downloaded JSON sidecar checksums are machine-readable in `candidate-inventory.json`; API links are included per record and per file.

| SXS ID | Caltech record | DOI | highest two | high-Lev exact spin max / e |
|---|---|---|---|---|
| [0198](https://data.caltech.edu/records/gx28k-wcg91) | `gx28k-wcg91` | [10.26138/SXS:BBH:0198v3.0](https://doi.org/10.26138/SXS:BBH:0198v3.0) | Lev4, Lev5 | `8.5348992e-5` / `2.023e-4` |
| [2483](https://data.caltech.edu/records/z5qsd-ptm37) | `z5qsd-ptm37` | [10.26138/SXS:BBH:2483v3.0](https://doi.org/10.26138/SXS:BBH:2483v3.0) | Lev4, Lev5 | `8.9362081e-5` / `2.336e-4` |
| [2484](https://data.caltech.edu/records/x0q4y-s3440) | `x0q4y-s3440` | [10.26138/SXS:BBH:2484v3.0](https://doi.org/10.26138/SXS:BBH:2484v3.0) | Lev4, Lev5 | `9.0119501e-5` / `8.098e-4` |
| [2486](https://data.caltech.edu/records/1wj32-qdk56) | `1wj32-qdk56` | [10.26138/SXS:BBH:2486v3.0](https://doi.org/10.26138/SXS:BBH:2486v3.0) | Lev4, Lev5 | `1.8991685e-5` / `4.536e-4` |
| [2491](https://data.caltech.edu/records/rx0d1-q7d95) | `rx0d1-q7d95` | [10.26138/SXS:BBH:2491v3.0](https://doi.org/10.26138/SXS:BBH:2491v3.0) | Lev4, Lev5 | `4.1513235e-5` / `3.607e-4` |
| [0185](https://data.caltech.edu/records/y7mbp-x8889) | `y7mbp-x8889` | [10.26138/SXS:BBH:0185v3.0](https://doi.org/10.26138/SXS:BBH:0185v3.0) | Lev2, Lev3 | `1.4339514e-5` / `2.772e-4` |

For every fetched highest-two level, the downloaded `ExtraWaveforms.json` sidecar declares both `/Strain_N3.dir` and `/Psi4_N3.dir`; this records availability in the sidecar only. The expected H5 sizes/checksums are cataloged without downloading H5. Downloaded metadata and sidecars matched the Caltech API entry size and MD5 values. Per-record API license fields were absent; no license conclusion is inferred here.

## Reproducibility and provenance

The temporary metadata-only builder is `/tmp/torchwave-nr33-expanded_build.py`; its cache is `/tmp/torchwave-nr33-expanded/catalog/`. The committed inventory records the source catalog hash, current raw record cache paths and SHA256 hashes, exact metadata/sidecar cache paths and checksums, all ranked candidate IDs, and all target-level absent/exhausted outcomes. The inventory SHA256 at creation is `0ff18f33eecfcc17e46311a49e7777bb376c21ef56c32f62baa5956639732a4a`.
