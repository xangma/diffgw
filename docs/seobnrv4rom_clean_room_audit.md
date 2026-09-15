# SEOBNRv4 ROM public-data audit

The official `SEOBNRv4ROM_v3.0.hdf5` asset is identified by SHA-256
`4ff5317967d757fcad9323a74b75f4f63894ed87f4b1852d46cbe47cde488b9c`.
The file is the public Zenodo release associated with
[record 14999310](https://zenodo.org/records/14999310).

Its three public submodels have the following logical table dimensions:

| group | frequency basis | eta grid | chi1 grid | chi2 grid |
| --- | ---: | ---: | ---: | ---: |
| `sub1` | 259 | 70 | 12 | 12 |
| `sub2` | 325 | 57 | 33 | 21 |
| `sub3` | 325 | 13 | 11 | 21 |

`Amp_ciall` and `Phase_ciall` are flattened arrays whose sizes equal
`N_f * (N_eta + 2) * (N_chi1 + 2) * (N_chi2 + 2)`.  The public HDF5 attributes
`ncx`, `ncy`, and `ncz` are exactly the corresponding physical parameter grid
plus 2.  The basis matrices are square `N_f × N_f` tables, and both amplitude
and phase use the same physical frequency grid in each group.

Pürrer (2014), [arXiv:1402.4146](https://arxiv.org/abs/1402.4146), and Pürrer
(2016), [arXiv:1512.02248](https://arxiv.org/abs/1512.02248), specify a
tensor-product cubic-spline interpolation of reduced-basis projection
coefficients.  They do not specify the convention that maps the two extra
coefficient planes in the flattened `*_ciall` tables to spline boundary
conditions.  TorchWave therefore validates the official file and its table
hashes in `tests/test_official_rom_data.py`, but does not enable an evaluator
based on an unverified boundary guess.  The existing native TorchWave ROM
path remains unchanged until that public-data ambiguity is resolved.
