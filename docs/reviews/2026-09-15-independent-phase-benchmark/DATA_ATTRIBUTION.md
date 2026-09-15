# Data and validation attribution

Numerical-relativity inputs were produced by the SXS Collaboration. These are the seven primary simulations, each accompanied by frozen resolution and extrapolation controls:

| Simulation | CaltechDATA record | Dataset DOI |
|---|---|---|
| SXS:BBH:2265 | [258wv-a5t45](https://data.caltech.edu/records/258wv-a5t45) | [10.26138/SXS:BBH:2265v3.0](https://doi.org/10.26138/SXS:BBH:2265v3.0) |
| SXS:BBH:2483 | [z5qsd-ptm37](https://data.caltech.edu/records/z5qsd-ptm37) | [10.26138/SXS:BBH:2483v3.0](https://doi.org/10.26138/SXS:BBH:2483v3.0) |
| SXS:BBH:2484 | [x0q4y-s3440](https://data.caltech.edu/records/x0q4y-s3440) | [10.26138/SXS:BBH:2484v3.0](https://doi.org/10.26138/SXS:BBH:2484v3.0) |
| SXS:BBH:2486 | [1wj32-qdk56](https://data.caltech.edu/records/1wj32-qdk56) | [10.26138/SXS:BBH:2486v3.0](https://doi.org/10.26138/SXS:BBH:2486v3.0) |
| SXS:BBH:2489 | [q7rzh-bzw94](https://data.caltech.edu/records/q7rzh-bzw94) | [10.26138/SXS:BBH:2489v3.0](https://doi.org/10.26138/SXS:BBH:2489v3.0) |
| SXS:BBH:2491 | [rx0d1-q7d95](https://data.caltech.edu/records/rx0d1-q7d95) | [10.26138/SXS:BBH:2491v3.0](https://doi.org/10.26138/SXS:BBH:2491v3.0) |
| SXS:BBH:0185 | [y7mbp-x8889](https://data.caltech.edu/records/y7mbp-x8889) | [10.26138/SXS:BBH:0185v3.0](https://doi.org/10.26138/SXS:BBH:0185v3.0) |

The [input manifest](input-manifest.json) preserves provider identities, hashes and metadata from the preceding experiments. Captured per-record license fields are null; this document assigns no inferred data license. See the [SXS documentation and acknowledgement guidance](https://data.black-holes.org/waveforms/documentation.html) and the prior [expanded-data attribution](../2026-09-15-independent-calibration-expanded/DATA_ATTRIBUTION.md).

The source-code LICENSE remains MIT. This work adds independently written preprocessing, numerical diagnostics and tests. It adds no LAL implementation or fitted coefficients. Public LAL outputs and the PSD are used only for validation, after the input/code freeze; no calibration was performed. The existing independently NR-fitted amplitude artifact remains byte-identical. Provider data and validation outputs are recorded separately from the code license.
