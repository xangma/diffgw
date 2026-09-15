# Data and validation attribution

The experiment uses numerical-relativity waveforms produced by the SXS Collaboration. See the [SXS waveform catalog](https://data.black-holes.org/waveforms/) and its [documentation and acknowledgement guidance](https://data.black-holes.org/waveforms/documentation.html).

| SXS simulation | Current Caltech record | Dataset DOI |
|---|---|---|
| SXS:BBH:0198 | [gx28k-wcg91](https://data.caltech.edu/records/gx28k-wcg91) | [10.26138/SXS:BBH:0198v3.0](https://doi.org/10.26138/SXS:BBH:0198v3.0) |
| SXS:BBH:2483 | [z5qsd-ptm37](https://data.caltech.edu/records/z5qsd-ptm37) | [10.26138/SXS:BBH:2483v3.0](https://doi.org/10.26138/SXS:BBH:2483v3.0) |
| SXS:BBH:2484 | [x0q4y-s3440](https://data.caltech.edu/records/x0q4y-s3440) | [10.26138/SXS:BBH:2484v3.0](https://doi.org/10.26138/SXS:BBH:2484v3.0) |
| SXS:BBH:2486 | [1wj32-qdk56](https://data.caltech.edu/records/1wj32-qdk56) | [10.26138/SXS:BBH:2486v3.0](https://doi.org/10.26138/SXS:BBH:2486v3.0) |
| SXS:BBH:2491 | [rx0d1-q7d95](https://data.caltech.edu/records/rx0d1-q7d95) | [10.26138/SXS:BBH:2491v3.0](https://doi.org/10.26138/SXS:BBH:2491v3.0) |
| SXS:BBH:0185 | [y7mbp-x8889](https://data.caltech.edu/records/y7mbp-x8889) | [10.26138/SXS:BBH:0185v3.0](https://doi.org/10.26138/SXS:BBH:0185v3.0) |

Full provider creator metadata and per-file source identities are retained in the inventory and manifests. Raw H5 and decoded NPZ files remain outside the repository. The retrieved Caltech records have no populated per-record license field; this report does not assign an inferred license to them.

The source-code `LICENSE` remains MIT and byte-identical to the starting snapshot. This validation adds no LAL implementation or fit coefficients. The previously frozen, independently NR-fitted coefficient artifact is reused unchanged. LAL is called through its public API solely for external comparison after input freezing.

The underlying amplitude equation and fixed settings come from the published [IMRPhenomXHM paper](https://arxiv.org/html/2001.10914v2), as documented in the preceding calibration experiment. This expansion performs no fitting and contacts no authors or maintainers.
