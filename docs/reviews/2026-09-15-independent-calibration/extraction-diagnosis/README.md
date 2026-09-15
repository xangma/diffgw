# Why the q≈3 NR extraction and LAL comparisons disagree

## Conclusion

The evidence associates the largest discrepancy with **resolution-dependent
early-ringdown behavior in the original SXS:BBH:0030 Lev3 (3,3) target**.
Higher resolution substantially reduces its N2/N3 disagreement and, for N3,
its distance from LAL. Separately, our pilot paired six waveforms with metadata from the
wrong resolution. That provenance error must be corrected before another fit.

This is a post-hoc diagnosis of the frozen pilot. No coefficients were refitted,
and these inspected cases cannot serve as fresh independent validation.

## 1. Resolution changes the apparent disagreement

`Lev3`, `Lev4`, and `Lev5` are increasing numerical resolutions of the same
simulation. `N2`, `N3`, and `N4` are polynomial orders used to extrapolate
finite-radius radiation measurements to infinity.

With the original FFT controls and start time held fixed, the geometric
Fourier-amplitude results are:

| q≈3 resolution | N2 versus N3 | Frozen fit versus N3 | N3 versus LAL |
| --- | ---: | ---: | ---: |
| Lev3, original | 30.14% | 27.70% | 42.21% |
| Lev4 | 15.53% | 18.41% | 31.69% |
| Lev5 | 3.40% | 7.39% | 4.49% |

Each entry is `||first − second||₂ / ||second||₂`, using geometric amplitude
in the original ringdown frequency band. The fit and LAL curves are fixed;
their mutual error remains **8.60%**. N2 at Lev5 differs from LAL by 1.46%,
but selecting an order because it matches LAL would bias an independent test.

The native-relaxation check in the [resolution report](resolution-diagnosis.json)
gives essentially the same trend. This is resolution sensitivity, not proof of
formal convergence: the record lengths differ, spectra are interpolated over
their common band, and these are correlated versions of one simulation.
The frozen-prediction comparison has 3,364 bins for Lev3/5 and 3,363 for Lev4,
which lacks the final bin. Neither extraction order is established as truth.

See the [comparison plot](resolution-versus-frozen.png),
[frozen-prediction comparison](resolution-versus-frozen.json), and
[resolution spectra](resolution-q3-amplitude-spectra.png).

## 2. The difference is already present in the time-domain ringdown

At Lev3, the median N2/N3 (3,3) amplitude ratio is approximately 1.0005
around merger (−20 to +20M relative to the h22 peak), then falls to **0.602**
at +20–50M and **0.453** at +50–80M. The strong ringdown differs before the
late numerical floor. The (2,2) control is much closer around merger.
The q≈6 control has only **0.0591%** N2/N3 Fourier-amplitude disagreement.

A smooth time partition of the complex N2−N3 residual identifies +0–40M as
the largest contribution to the ringdown-band FFT residual. Its norm is 0.80
times the total residual norm. These are overlapping coherent contributions,
not additive percentages; the independently computed full FFT is reconstructed
to relative error `1.96e-16`. See the
[numerical report](numerical-diagnosis.json) and
[time-domain plot](time-domain-ratios.png).

The paired files have identical time grids. Fresh decoding with the recorded
analysis package, SXS 2026.0.3, exactly reproduces the cached time, h33, and h22
arrays. The sidecars describe an older processing stack; within each pair,
they record the same frame correction and truncation
settings. This rules out a cache/indexing discrepancy introduced by the pilot;
it is not an independent validation of the SXS decoder itself.

## 3. Why extrapolation order can matter this much

Higher polynomial order can amplify imperfect finite-radius data. SXS
explicitly cautions that a high order useful during inspiral can become noisy
around merger; its FFT tutorial illustrates unstable N4 ringdown. In this
case, Lev3 N4 versus N3 is even worse, about **112%**. The order and resolution
trends are consistent with amplified numerical extraction error.
[SXS waveform documentation](https://data.black-holes.org/waveforms/documentation.html),
[SXS FFT tutorial](https://sxs.readthedocs.io/en/main/tutorials/05-PreprocessingForFFTs/).

The third SXS catalog describes ringdown anomalies in many early simulations,
consistent with insufficient ringdown resolution, and notes a correction in
2015. SXS:BBH:0030 was run in 2012 and is now deprecated. This supports the
interpretation but does **not** identify the exact defect in this simulation.
We have not reconstructed its finite-radius extrapolation or rerun SpEC.
[SXS catalog paper](https://arxiv.org/html/2505.13378),
[exact dataset record](https://zenodo.org/records/13167328).

An optional [polynomial sensitivity illustration](radial-extrapolation-sensitivity.json)
uses recorded coordinate-radius ranges, equal weights, and independent
equal-variance errors. It is a toy calculation, not the SXS extraction operator
or an estimate of the actual waveform uncertainty.

## 4. Our metadata and selection errors

The [exact-record audit](metadata-consistency.json) verified 18 metadata files
against their published MD5 checksums. Nine checked metadata fields match a
higher resolution exactly in six of the seven pilot cases:

| Simulation | Waveform resolution | Resolution matching the recorded metadata |
| --- | --- | --- |
| SXS:BBH:0008 | Lev4 | Lev5 |
| SXS:BBH:0169 | Lev3 | Lev5 |
| SXS:BBH:0030 | Lev3 | Lev5 |
| SXS:BBH:0182 | Lev2 | Lev4 |
| SXS:BBH:0056 | Lev3 | Lev5 |
| SXS:BBH:0166 | Lev5 | Lev5, consistent |
| SXS:BBH:0063 | Lev3 | Lev5 |

The cause is in the [preparation helper](../../../../scripts/prepare_nr33_calibration_data.py):
its `CASES` entries select a waveform resolution but independently embed scalar
values copied from the simulation-level catalog. `main()` copies those values
into the fit-facing metadata. The correctly matched waveform sidecar is saved
as an opaque object; the helper never loads and validates the matching
`Lev#:metadata.json`. A waveform checksum therefore passed while this pairing
error remained undetected.

For q≈3, Lev3's relaxation/reference time is **640M**, whereas the manifest
records **260M**, matching Lev5. Lev3's remnant mass/spin also differ from the
recorded values. Reprocessing N3 with the 640M cut changes its band amplitude
by only **0.000481%**; the N2/N3 gap stays about 30.16%. Thus the relaxation
mistake does not explain the main discrepancy.

Correcting the Lev3 remnant mass normalization and spin changes the frozen
prediction by **0.0402%** on its exact saved frequency bins. Correcting q as
well gives **0.0401%**. The original prediction is reproduced exactly before
these perturbations. Thus these particular metadata errors also cannot
explain the large gap; this does not assess their cumulative effect across
the five training cases. See the
[metadata prediction sensitivity](metadata-prediction-effect.json).

All seven simulations carry deprecated status, omitted from our manifest.
SXS lists 1221, 2265, and 2498 as replacements for 0030. Deprecation alone
does not establish an error magnitude; it should have triggered quality
screening before these records became calibration inputs.

SXS waveform units use total reference mass, while metadata quantities are
in code units. The original README's initial-mass statement was incorrect.
A corrected pipeline must use each waveform resolution's metadata and its
reference-mass normalization. The reference/initial mass-scale differences
alone are at the parts-per-million level; the remnant-metadata perturbations
are measured separately above.
[SXS unit conventions and metadata description](https://arxiv.org/html/2505.13378).

## Implications for the next calibration

The original 42% NR/LAL gap is not evidence of a comparably large failure of
the independently fitted formula. Our NR target was unstable. However, the
remaining **7–9%** q≈3 fit discrepancy has not been solved: the pilot uses only
five training simulations and a small quadratic parameter fit, with the input
problems above. Shared published settings alone do not reproduce another
calibration.

Before another fit, pair waveform and metadata by exact record/resolution,
record deprecation/replacement information, and apply a quality rule using
multiple resolutions and extraction orders without consulting LAL agreement.
Then freeze a new training split and reserve fresh simulations for validation.
Fixes must create a new experiment rather than silently altering this frozen
one. Original code remains MIT; the separate NR-data attribution still applies.

## Reproduction and integrity

Run from the repository root with the pilot's recorded Python environment:

```sh
PYTHONPATH=src python -m scripts.diagnose_nr33_extraction \
  --manifest docs/reviews/2026-09-15-independent-calibration/nr33-data-manifest.json \
  --output docs/reviews/2026-09-15-independent-calibration/extraction-diagnosis/numerical-diagnosis.json \
  --plot-dir docs/reviews/2026-09-15-independent-calibration/extraction-diagnosis
PYTHONPATH=src python -m scripts.diagnose_nr33_resolution --download
PYTHONPATH=src python -m scripts.compare_nr33_resolution_to_frozen
PYTHONPATH=src python -m scripts.diagnose_nr33_metadata_effect \
  --output docs/reviews/2026-09-15-independent-calibration/extraction-diagnosis/metadata-prediction-effect.json
```

The resolution helper verifies downloads against the pinned exact-record
inventory; the comparison calls public LAL only after checking frozen hashes.
Raw inputs remain in the `/tmp/torchwave-nr33-*` caches. Metadata download
URLs/checksums are in the audit; no upstream contact was made.

The calibration SHA256 remains
`f447acda86f64f1d13745c8a5d725acdf295b050a2638d1731d4b6a139e1358d`.
See [final verification](verification.json) for report identities, focused
checks, and runtime-source integrity.
