# Author-data Kerr QNM representation

The new `kerr_qnm_author.py` provides all 15 previously supported positive-m
frequencies: `(l,m) = (2,2), (2,1), (3,3), (3,2), (4,4)` and `n = 0,1,2`.
`dimensionless_qnm(l,m,n,final_spin)` returns `Re(M omega)` and positive
`-Im(M omega)`. Its explicit spin domain is **0 <= a <= 0.99999**; it rejects
unsupported modes, nonfinite spins, and extrapolation. Higher-level ringdown
integration is separate from this module.

This representation replaces approximate three-parameter frequencies with
interpolation of independently obtained author numerical data. It does not
establish parity of a complete waveform, angular convention, or another QNM
implementation. All numerical errors below are measured errors at withheld
author samples, not proven bounds between samples.

## Fresh primary sources and attribution

1. Gregory B. Cook, *Kerr Modes: Phase fixed gravitational QNMs and TTMs*, v4,
   published 2025-08-11, [Zenodo record 16801267](https://zenodo.org/records/16801267),
   [DOI 10.5281/zenodo.16801267](https://doi.org/10.5281/zenodo.16801267).
   The original [2019 record](https://zenodo.org/records/2650358) led to this
   newer release. Fresh v4 HDF5 numerical files, rather than the old local
   fundamental file, were used for generation.
2. Cook and Zalutskiy, *Gravitational perturbations of the Kerr geometry:
   High-accuracy study*, Phys. Rev. D **90**, 124021 (2014),
   [paper](https://arxiv.org/pdf/1410.7698v2),
   [DOI 10.1103/PhysRevD.90.124021](https://doi.org/10.1103/PhysRevD.90.124021).

The author database is licensed under the
[Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/).
This work contains information from Gregory B. Cook's database, made available
under that license. The committed spline coefficients are a modified, derived
representation, and the test fixture contains an attributed subset of original
author frequency rows. The attribution notice also accompanies the runtime
module and the generated metadata.

| Author file | Published MD5 | Verified SHA256 |
| --- | --- | --- |
| KerrQNM_00.h5 | `62a1086b7e59da567c624f331c39d119` | `9797479ed07ebe8f5172b5c591b5cf0bf3196cb3a843c0bcd987df68265971cc` |
| KerrQNM_01.h5 | `e767ac934083c364790d250689ee009c` | `e3a52a852dbc3687a415da4047c4e0aaccecf0a0578f2d55086af964d80b830f` |
| KerrQNM_02.h5 | `32a8b90d276f778e161a37479f5c9b40` | `8834f486ad99592f6f23765aa4a39f0a67727a3c7cb7c7377d18bde74706bb40` |

The generation script pins each complete input file's SHA256. Its validation
JSON records the direct author download URL, published MD5, SHA256, exact HDF5
dataset key, source row count, and every training row index. Only columns 0–2
(`a`, `Re(M omega)`, `Im(M omega)`) enter the frequency fit. No solver source,
QNM package, LAL implementation, binary oracle, oracle results, or previous
implementation provenance was used to derive or adjust these coefficients.

## Representation and held-out validation

A degree-five interpolating B-spline uses `x = sqrt(1-a)`. Training rows are
selected deterministically from the author's ascending-spin arrays:

- Exclude row indices congruent to 1 or 3 modulo 10.
- Below spin 0.9 retain indices divisible by 4; from 0.9 to 0.97 retain even
  indices; at and above 0.97 retain all remaining eligible indices.
- Retain the source sequence endpoints, which bracket the advertised domain.
- Use rows congruent to 3 modulo 10 for development and rows congruent to 1
  modulo 10 for the final test; restrict reported holdouts to the domain.

This gives 310–349 training nodes per mode. Cubic and several higher-degree
alternatives were explored using development data; the degree-five method,
training selection and domain were fixed before reading final holdout errors.
The 1,556 final test rows are disjoint from training. No adjustments were made
after their errors were measured. Generation retains only the sparse training
representation, not a final refit using the test rows.

Torch evaluates the saved knots and coefficients with the de Boor recurrence.
The pieces share four continuous derivatives. Computation uses float64; tensor
inputs retain their floating output dtype, device and batch shape, and spin
gradients pass through the evaluation. The runtime needs Torch only; NumPy,
SciPy and h5py are generation dependencies.

Across all modes with **a <= 0.99**, the final holdouts give maximum absolute
errors **4.647e-10** in the real frequency and **2.891e-10** in its imaginary
component. The wider domain has larger errors for the rapidly varying `(2,1)`
overtone sequences above 0.99. Users requiring smaller errors there must not
treat this representation as uniformly accurate to the low-spin figures.

| Mode `(l,m,n)` | Holdout max abs real error | Holdout max abs imaginary error |
| --- | ---: | ---: |
| 2,2,0 | 1.146e-10 | 8.565e-11 |
| 2,1,0 | 1.955e-10 | 2.223e-10 |
| 3,3,0 | 1.088e-12 | 4.479e-13 |
| 3,2,0 | 1.524e-8 | 2.510e-9 |
| 4,4,0 | 1.568e-13 | 4.158e-14 |
| 2,2,1 | 4.218e-11 | 1.745e-10 |
| 2,1,1 | 2.235e-5 | 1.918e-6 |
| 3,3,1 | 2.468e-12 | 3.699e-12 |
| 3,2,1 | 3.109e-7 | 2.754e-7 |
| 4,4,1 | 6.721e-13 | 1.446e-13 |
| 2,2,2 | 1.763e-10 | 8.436e-11 |
| 2,1,2 | 9.366e-6 | 1.317e-5 |
| 3,3,2 | 1.131e-11 | 2.782e-12 |
| 3,2,2 | 6.064e-9 | 3.103e-8 |
| 4,4,2 | 3.901e-13 | 2.375e-13 |

The maximum component relative errors over the full domain are 4.140e-5
for the real part and 1.181e-4 for the imaginary part. Per-mode absolute and
relative errors, including separate development results, are stored in
`tests/data/kerr_qnm_author_validation.json`.

## Angular data inventory; no angular implementation in this change

The v4 metadata explicitly distinguishes its column layout from v1/v2. Its
zero-based columns are 0: spin; 1–2: complex frequency; 3–4: complex angular
separation constant; 5: zero-based sorted angular eigenvalue index; 6–7:
complex SL-C-to-SL-Ind phase factor; and 8 onward: real/imaginary pairs of
spheroidal-to-spherical expansion coefficients. The coefficients begin at
`l_min = max(|m|, |s|)`, with `s=-2`. The existing supported sequences contain
12–15 coefficients per row.

Cook–Zalutskiy Eq. (49) defines the spherical expansion. Equations (52a–d)
give multiplication by cos(theta), (53a–e) its square, (55) the pentadiagonal
angular matrix, and (56) the eigenproblem. The text after Eq. (55) specifies
unit coefficient norm and a real coefficient at spherical index `l'=l`.

The newer v4 stored coefficients instead use the metadata's **SL-C** phase,
which connects smoothly to the spherical limit. Direct numerical inspection
of all supported author rows in our domain finds unit norms within 6.7e-16,
but the coefficient at `l'=l` is generally complex. All corresponding
SL-C-to-SL-Ind phase factors in these sequences happen to be exactly 1+0i;
they do not convert to the 2014 real-coefficient convention. A comparison
against a real-positive convention must rotate each full coefficient vector
by `conj(C_l)/abs(C_l)` first. These inventory observations are independent
of the frequency fit, which does not consume any angular columns.

## Reproduction and checks

From the repository root:

```sh
python scripts/generate_kerr_qnm_author.py --data-dir /tmp/cook-qnm --download
python scripts/generate_kerr_qnm_author.py --data-dir /tmp/cook-qnm --check
python -m pytest -q tests/test_kerr_qnm_author.py
```

The generation/check environment used Python 3.13.9, NumPy 2.3.5, SciPy 1.16.3,
h5py 3.15.1, and Torch 2.9.1. Exact regeneration passes. All 38 focused tests
pass, covering untouched author values for every mode, independent finite
difference gradient checks for every mode, dtype/batch behavior, Schwarzschild
normalization, physical decay sign, finite endpoint derivatives and explicit
domain/mode rejection.
