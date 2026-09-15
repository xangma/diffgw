# Candidate discovery (metadata-only)

This is a bounded preflight inventory for the v2 calibration. The frozen policy is recorded by SHA-256 `1a863811b90c410cd5a9b0b52a0bd53696b869a0d503cb8fa61d9c054842138d`. The SXS catalog input is the cached `sxscatalogdata` snapshot dated 2025-05-14; live CaltechDATA API records were re-read on 2026-09-15 and their raw JSON is cached and SHA-256 recorded in `candidate-inventory.json`. No waveform H5 was downloaded.

## Selection scope and split

The old seven simulations are excluded. Their superseded-by links seed the candidate search. Up to three candidates per target were retained using the declared metadata ordering: highest two available levels first, then smallest numeric SXS ID; q4 retains 2485, 1220, 1906 and records 2499 as rank 4 omitted before waveform inspection. q3 and q6 are simulation-level holdouts frozen before any fit. q2 retained old-map rank 1 (2425), then added 1166 and 1167 as ranks 2 and 3 from a bounded search (`|reference_mass_ratio-2|<=0.1`, empty keywords, at least two levels, eccentricity `<=1e-3`, maximum reference spin norm `<=1e-4`), before any H5 download. No candidates were swapped after waveform results.

The inventory has 13 candidates: q=1.5 (2331), q=2 (2425, 1166, 1167), q=3 holdout (1221, 2265, 2498), q=4 (2485, 1220, 1906), q=5 (3619), q=6 holdout (3630), and q=8 (3622). Metadata failures are preserved: 2425, 1221, 1906 fail the highest-level spin bound; 2485 and 1220 pass at the highest level but fail on one of the two retained levels. Every candidate remains `strict_viable_under_frozen_policy=false` pending waveform checks.

## N3 and ExtraWaveforms availability

The current CaltechDATA file inventories expose top-level `Strain_N2.h5/json`; they do not expose top-level `Strain_N3.h5/json` for these candidates. The metadata-only `ExtraWaveforms.json` sidecar is different: all 26 cached top-two sidecars declare `/Strain_Outer.dir`, `/Strain_N3.dir`, `/Psi4_Outer.dir`, and `/Psi4_N3.dir` inside the corresponding `ExtraWaveforms.h5`. Therefore N3 and finite-radius strain are provider-declared embedded products, rather than absent data. Their H5 structure, mode arrays, time grids, radial samples, and ability to reproduce the required interface still require validation. The next concrete step is a same-level H5 structural read for the frozen cases; no numerical fit is implied.

The official SXS waveform documentation says the database supplies outermost-extraction and N=2,3,4 extrapolated waveforms, and warns that higher extrapolation orders can become noisy in merger/ringdown; it also documents `ExtraWaveforms` access and the N2/N3/N4/Outer objects. See [SXS waveform database](https://data.black-holes.org/waveforms/), [waveform documentation](https://data.black-holes.org/waveforms/documentation.html), and [SXS waveforms tutorial](https://sxs.readthedocs.io/en/stable/tutorials/04-Waveforms/).

## Provenance and licensing

Each candidate records its exact current CaltechDATA API URL, DOI, creators, every catalog file entry (URL, MD5, and byte size), raw API cache path, and raw-record SHA-256. The live API records are public/open, but their returned metadata and custom fields do not expose a per-record license. The inventory records license as unavailable and does not transfer the CC-BY-4.0 value seen in the superseded Zenodo audit. The public SXS site says the waveform data are free to use and requests acknowledgement of the first scientific publication; this is recorded as site-level use guidance, not an invented per-record SPDX license.

An older q3 Zenodo record ([10.5281/zenodo.3272829](https://zenodo.org/records/3272829)) has a different raw-file layout (including `rhOverM` data) and is not evidence that the current migrated records expose standalone N3 files. This bounded check covers only the retained replacement records and that documented older record; it is not a claim that no other SXS record contains N3.
