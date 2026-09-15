# q6/q8 replacement search (metadata-only)

This bounded search used a **0.5% relative mass-ratio window**, declared before filtering: q6 ∈ [5.97, 6.03] and q8 ∈ [7.96, 8.04]. It scanned the cached `sxscatalogdata` snapshot dated 2025-05-14, requiring no deprecated marker, at least two resolutions, catalog eccentricity `<=1e-3`, and maximum reference spin norm `<=1e-4`. It excluded the original seven runs and failed frozen candidates SXS:BBH:3630 and SXS:BBH:3622. Ranking is descending highest two available levels, then ascending numeric SXS ID. No H5 was downloaded and no waveform or fit residual was used.

## Frozen candidates

The exhaustive cached-catalog result is:

- q6 holdout: historical rank 1 **SXS:BBH:3630** is preserved as failed; new ranks are **SXS:BBH:2489** (rank 2; levels Lev4/Lev5) and **SXS:BBH:2164** (rank 3; Lev2/Lev3).
- q8 training: historical rank 1 **SXS:BBH:3622** is preserved as failed; new rank 2 is **SXS:BBH:2707** (rank 2; Lev3/Lev4).

All three current CaltechDATA APIs were validated on 2026-09-15 and are public/open. Exact record IDs, raw API JSON cache paths and SHA-256 hashes, metadata, ExtraWaveforms sidecars, and every top-two file URL/MD5/size are in `candidate-inventory.json`. The catalog scan found one near-q6 deprecated record, SXS:BBH:0181, which was rejected by the declared nondeprecated filter. Under this snapshot and filter, counts are q6=2 and q8=1; this is not a claim that other catalog versions or sources contain no candidates.

| SXS ID | CaltechDATA record | DOI | q (catalog) | e (catalog) | top-two exact max spin norms | ExtraWaveforms H5 bytes (top two) |
|---|---|---:|---:|---:|---:|---:|
| SXS:BBH:2489 | [q7rzh-bzw94](https://data.caltech.edu/records/q7rzh-bzw94) | [10.26138/SXS:BBH:2489v3.0](https://doi.org/10.26138/SXS:BBH:2489v3.0) | 5.9995319744 | 0.0005865 | Lev4 1.5536e-4; Lev5 2.7811e-5 | 57,118,119 |
| SXS:BBH:2164 | [v4wkb-9m642](https://data.caltech.edu/records/v4wkb-9m642) | [10.26138/SXS:BBH:2164v3.0](https://doi.org/10.26138/SXS:BBH:2164v3.0) | 5.9985323837 | 0.0004762 | Lev2 1.3912e-4; Lev3 1.9373e-5 | 51,623,983 |
| SXS:BBH:2707 | [gwmd0-0rd53](https://data.caltech.edu/records/gwmd0-0rd53) | [10.26138/SXS:BBH:2707v3.0](https://doi.org/10.26138/SXS:BBH:2707v3.0) | 8.0001492004 | 0.0003110 | Lev3 6.7803e-5; Lev4 8.0569e-5 | 69,947,995 |

The exact Lev metadata show that q6 candidates pass the frozen **highest-resolution** spin/eccentricity gate, while their lower member of the retained pair has spin norm above `1e-4`; this is recorded rather than hidden. q8 passes at both retained levels. Relative-L2, preprocessing, post-peak duration, and actual N2/N3 waveform validity remain pending H5 screening.

The current records expose top-level `Strain_N2` files. Their metadata-only `ExtraWaveforms.json` sidecars declare embedded `/Strain_N3.dir` and `/Strain_Outer.dir` paths; same-level H5 structural decoding is required before treating those products as usable N3/finite-radius inputs. The official [SXS waveform database](https://data.black-holes.org/waveforms/) documents outermost-extraction and N=2,3,4 products, and cautions that higher extrapolation orders can become noisy during merger/ringdown. The [SXS waveform tutorial](https://sxs.readthedocs.io/en/stable/tutorials/04-Waveforms/) documents the N2/N3/N4/Outer waveform objects.

The CaltechDATA API records public/open access, but the returned per-record metadata and custom fields contain no rights/license value. The inventory records that absence without assigning a license. The SXS database page states that its data are free to use and requests acknowledgement of each simulation’s first scientific publication; this is quoted as SXS attribution/use guidance, without an independent legal interpretation.
