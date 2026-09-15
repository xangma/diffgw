# Calibration data and the MIT code licence

Checked 12 September 2026 following authorization to use calibration data while
retaining MIT for TorchWave's code. This is an asset-level licensing record,
not a certification of all inherited source or coefficients.

TorchWave can retain MIT for its original evaluator code while identified data
keeps separate attribution terms. We do not represent the combined distribution
as containing only MIT-licensed assets. The root `LICENSE` is unchanged;
[THIRD_PARTY_NOTICES.md](../../../THIRD_PARTY_NOTICES.md) identifies the
bundled data exception, and package builds include both files.

| Asset | Primary licence evidence | Current disposition |
| --- | --- | --- |
| Cook Kerr QNM data, v4 | [Zenodo 16801267](https://zenodo.org/records/16801267) records `odc-by`; [ODC-By-1.0](https://opendatacommons.org/licenses/by/1-0/) supplies the terms. | Generated runtime spline arrays and author-sample fixtures retain these terms and attribution. Original evaluation code remains MIT. |
| `SEOBNRv4ROM_v3.0.hdf5` | [LALSuite waveform data, v1](https://zenodo.org/records/14999310) records `cc-by-4.0`; the file's published MD5 matches the pinned download. | A candidate source for an independently written ROM evaluator under the new data allowance. Currently downloaded only for external verification; no native ROM integration or redistribution was performed in this licensing pass. |
| Inherited unrounded PhenomD fit arrays | No independently obtained author asset with explicit reuse terms found in the bounded search below. | Provenance unresolved. Removed the module header's unsupported claim that its provenance guarantees MIT integrity. No coefficients changed. |

## Cook data obligations

ODC-By sections 3.1 and 4 allow commercial use, extraction and modification,
subject to the licence's conditions. Sections 4.2 and 4.4 require preservation
of the data licence and notices when conveying the database or a derived
database. Section 4.3 covers attribution for publicly used Produced Works.
Section 2.3 excludes database-operating software from its scope; section 2.4
does not grant independent rights in individual contents. These provisions
support our implementation choice to retain ODC-By for the generated arrays
and MIT for the original software. We do not sublicense those arrays as MIT.

The notice identifies the author, version, DOI, licence URL, source filenames,
and the interpolation modification. Each sample fixture has an attribution
or author field and a licence URI. The holdout fixture generator was updated
to preserve the additional notice fields on regeneration. The numerical arrays,
interpolation, holdout rows and waveform behavior are unchanged.

## ROM data route

The Zenodo deposit credits Geraint Pratten, Lucy Maja Thomas and Antoni Ramos
Buades. The HDF5 root metadata additionally identifies Michael Puerrer as the
file creator, version 3.0.0, and references his 2014 and 2016 ROM papers.
No conflicting licence was present in the inspected root attributes. The
download's MD5 is `bb0eab9ae721af6abd06c4872b13e477`.

The [CC-BY-4.0 summary](https://creativecommons.org/licenses/by/4.0/) permits
sharing and adaptation, including commercially, with credit, a licence link,
and an indication of modifications. It forbids additional restrictions on the
licensed material. This supports a route using separately attributed data and
an original MIT evaluator; it does not grant permission to copy the LALSuite
implementation. Before shipping a native ROM asset, retain all file-level
notices and check any additional metadata inside the selected asset. Reading
the archive licence alone does not complete model implementation or validation.

## Unresolved PhenomD assets

The independent source search covered the two PhenomD papers, their author
links, APS, Zenodo, gw-models and author institutions. It found the rounded
coefficient mapping in [PhenomD II, Appendix C/Table V](https://arxiv.org/html/1508.07253v2)
and the calibration description in [PhenomD I](https://arxiv.org/html/1508.07250v2),
but no explicitly licensed complete calibration-hybrid bundle or full-precision
coefficient asset. The [APS article record](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.93.044007)
lists APS copyright and a permissions link, without a supplementary-material
entry. The papers' [arXiv distribution licence](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
grants rights to arXiv; it is not a blanket grant to redistribute separate assets.

This is an unresolved source/permission gap, not a finding that mathematical
facts or independently implemented equations are prohibited. An author-provided
standalone dataset with explicit terms would resolve the acquisition route;
no authors have been contacted. No GPL implementation or translated port was
used as a source, and no waveform-oracle residuals were fitted.

## Evidence and verification

Selected primary record metadata, retrieval timestamps, source-response SHA256
and published file checksums are saved in
[Cook metadata](verification/licensing/16801267-metadata.json) and
[ROM metadata](verification/licensing/14999310-metadata.json). Original Cook
file SHA256 values and extraction details remain in the
[QNM provenance report](pass-3-qnm.md).

Both the wheel and source archive built successfully and included byte-exact
copies of `LICENSE` and `THIRD_PARTY_NOTICES.md`. The focused QNM/angular suite
passed **55 tests in 1.05 s**. Package notice inclusion and these checks are recorded
in [verification results](verification/licensing/checks.json). The earlier pass-6
numerical verification snapshots remain historical records of their exact
inputs; this notice pass does not claim a new full parity run.
