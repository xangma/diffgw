# Replacement waveform data attribution

Waveform creator: **SXS Collaboration**. This experiment used the following
published v3.0 records. Publication references below are copied as identifiers
from each selected highest-resolution `metadata.json` field `citation_dois`.

| Simulation | CaltechDATA record | Dataset DOI | Publication references supplied by metadata |
| --- | --- | --- | --- |
| SXS:BBH:2331 | [aq9e2-0ye86](https://data.caltech.edu/records/aq9e2-0ye86) | [2331v3.0](https://doi.org/10.26138/SXS:BBH:2331v3.0) | [SXS Catalog 3](https://doi.org/10.26138/SXSCatalog3) |
| SXS:BBH:1166 | [27s13-fe162](https://data.caltech.edu/records/27s13-fe162) | [1166v3.0](https://doi.org/10.26138/SXS:BBH:1166v3.0) | [10.1088/1361-6382/ab34e2](https://doi.org/10.1088/1361-6382/ab34e2) |
| SXS:BBH:2265 | [258wv-a5t45](https://data.caltech.edu/records/258wv-a5t45) | [2265v3.0](https://doi.org/10.26138/SXS:BBH:2265v3.0) | [SXS Catalog 3](https://doi.org/10.26138/SXSCatalog3), [10.1103/PhysRevD.99.064045](https://doi.org/10.1103/PhysRevD.99.064045) |
| SXS:BBH:2485 | [5mjz7-2gs59](https://data.caltech.edu/records/5mjz7-2gs59) | [2485v3.0](https://doi.org/10.26138/SXS:BBH:2485v3.0) | [SXS Catalog 3](https://doi.org/10.26138/SXSCatalog3) |
| SXS:BBH:3619 | [6fw56-9q785](https://data.caltech.edu/records/6fw56-9q785) | [3619v3.0](https://doi.org/10.26138/SXS:BBH:3619v3.0) | [SXS Catalog 3](https://doi.org/10.26138/SXSCatalog3) |
| SXS:BBH:3630 | [28pv5-4wj69](https://data.caltech.edu/records/28pv5-4wj69) | [3630v3.0](https://doi.org/10.26138/SXS:BBH:3630v3.0) | [SXS Catalog 3](https://doi.org/10.26138/SXSCatalog3) |
| SXS:BBH:3622 | [r1mdz-3sj90](https://data.caltech.edu/records/r1mdz-3sj90) | [3622v3.0](https://doi.org/10.26138/SXS:BBH:3622v3.0) | [SXS Catalog 3](https://doi.org/10.26138/SXSCatalog3) |

## Usage and license evidence

The current CaltechDATA API records are public/open but expose no explicit
per-record license in the inspected metadata/custom fields. The
[SXS waveform database](https://data.black-holes.org/waveforms/) states that its
data are free to use and requests acknowledgement of the first scientific
publication for each simulation. This document records that site guidance and
the supplied citation identifiers; it does not assign an SPDX license or infer
the license of a current record from a historical record.

The [frozen inventory](candidate-inventory.json) preserves exact API responses,
record provenance, retrieval information, MD5 checksums and sizes. The
[decoded manifest](nr33-manifest-round1.json) identifies the exact H5 key,
embedded group where applicable, physical metadata, waveform sidecar, and local
SHA256 hashes for every decoded case. Numerical data remain under
`/tmp/torchwave-nr33-v2`, outside the repository. The repository's MIT license
was not changed and is not a declaration of ownership or licensing of SXS data.

No LAL implementation or fitted coefficients were incorporated by this
replacement-screening experiment. No external party was contacted.
