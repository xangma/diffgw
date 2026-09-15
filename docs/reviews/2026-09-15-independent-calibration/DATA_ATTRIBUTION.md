# Independent NR (3,3) pilot: data attribution

The independently written preparation, fitting, and validation software is
covered by TorchWave's MIT license. The SXS numerical-relativity inputs retain
their **Creative Commons Attribution 4.0 International** terms:
[license](https://creativecommons.org/licenses/by/4.0/).

The following datasets, by the creators named in their linked records, were
published on Zenodo by the Simulating eXtreme Spacetimes collaboration.
Each exact record independently reports `cc-by-4.0`; the retrieved creator
lists, file checksums, and metadata identities are retained in
[the verification record](verification/dataset-license-records.json).

| Simulation | Approximate mass ratio | Versioned dataset |
| --- | ---: | --- |
| SXS:BBH:0008 | 1.5 | [10.5281/zenodo.13166243](https://doi.org/10.5281/zenodo.13166243) |
| SXS:BBH:0169 | 2 | [10.5281/zenodo.13167391](https://doi.org/10.5281/zenodo.13167391) |
| SXS:BBH:0030 | 3 | [10.5281/zenodo.13167328](https://doi.org/10.5281/zenodo.13167328) |
| SXS:BBH:0182 | 4 | [10.5281/zenodo.13159588](https://doi.org/10.5281/zenodo.13159588) |
| SXS:BBH:0056 | 5 | [10.5281/zenodo.13165791](https://doi.org/10.5281/zenodo.13165791) |
| SXS:BBH:0166 | 6 | [10.5281/zenodo.13167344](https://doi.org/10.5281/zenodo.13167344) |
| SXS:BBH:0063 | 8 | [10.5281/zenodo.13167443](https://doi.org/10.5281/zenodo.13167443) |

Changes made for this pilot include decoding, mode extraction, time resampling,
endpoint tapering, Fourier transformation, and fitting new numerical amplitude
coefficients. The manifest and method record specify the actual processing.
Processed NR arrays and associated calibration data in this review carry this
CC-BY-4.0 attribution notice; the MIT software grant does not remove the source
data's attribution requirements. Retain this notice when redistributing those
data or adaptations. No endorsement by the dataset authors is implied.

This bounded pilot does not establish provenance for inherited calibration
assets elsewhere in TorchWave.
