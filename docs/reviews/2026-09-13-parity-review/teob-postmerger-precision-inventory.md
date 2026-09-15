# TEOB postmerger coefficient precision inventory

This inventory covers the coefficients consumed by the clean
`teob_postmerger.py` and `teob_uib2016_postmerger.py` components. The search was
restricted to primary publications and author supplied ancillary material.
No implementation source or waveform output was used to infer digits.

## Public primary material

The primary TEOBResumS postmerger specification is Nagar et al.,
[arXiv:1806.01772](https://arxiv.org/abs/1806.01772), Appendix “NR-informed
description of merger and postmerger”. Its [v2 TeX source](https://arxiv.org/src/1806.01772v2)
contains the complete `merger_fit`, `QNM_fits`, `rng`, and `NQC` tables. The
tables provide decimal coefficients at their printed precision. The v2 source
archive contains no coefficient data file, supplementary table, or executable
asset with additional digits.

The primary UIB remnant-fit release is Jiménez-Forteza et al.,
[arXiv:1611.00332](https://arxiv.org/abs/1611.00332). Its [author ancillary
material](https://arxiv.org/src/1611.00332v2) includes Python/Mathematica
fit helpers, covariance information, and NR data for final spin and radiated
energy. It contains no TEOB `merger_fit`, QNM, `rng`, or NQC coefficient table.
Those remnant coefficients are a separate input family and do not provide
extra precision for the postmerger shape fits.

| Coefficient family | Allowed primary location | Additional precision found | Consequence |
| --- | --- | --- | --- |
| `merger_fit`, `QNM_fits`, `rng`, `NQC` | 1806.01772v2 Appendix/TeX | No | Retain the published decimal values. |
| UIB final spin and radiated energy | 1611.00332v2 ancillary material | Yes, for remnant fits only | Already handled by the separate remnant helper; it does not resolve postmerger/QNM precision. |

Consequently, no admissible higher precision TEOB postmerger or QNM
coefficients were found. The observed coefficient precision limitation cannot
be repaired from the allowed public releases without an additional author
data artifact. No runtime coefficient changes are made by this inventory.
