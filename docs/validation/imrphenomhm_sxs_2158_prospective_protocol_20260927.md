# IMRPhenomHM SXS:BBH:2158: prospective aligned-spin holdout

**Freeze this protocol and its runner before retrieving or decoding 2158 strain.**
The source selection used only the official SXS v3.0.0 catalog metadata and
file manifest. Record a failed or infeasible comparison without changing the
case, conditioning, optimizer, or gates. This is one aligned-spin holdout; it
does not replace or change any earlier SXS receipt.

## Research target and case selection

[London et al. (2018)](https://arxiv.org/pdf/1708.00404v4) defines the
aligned-spin IMRPhenomHM model and compares it with full numerical-relativity
signals. Its reported high matches are population results, not a guarantee for
this particular simulation or each viewing angle. The model's omitted
spheroidal-mode mixing and its finite six-mode content remain possible physical
sources of mismatch. The
[SXS Catalog 3 paper](https://arxiv.org/html/2505.13378v2) documents the v3
strain product. The physical comparison, numerical conventions, and operational
thresholds below follow the previously frozen
[3144 protocol](imrphenomhm_sxs_3144_prospective_protocol_20260927.md),
except for the explicit case, resolutions, and spin inputs here.

Use the official
[`sxscatalogdata` v3.0.0 `simulations.json`](https://raw.githubusercontent.com/sxs-collaboration/sxscatalogdata/v3.0.0/simulations.json),
SHA-256 `132d38520b3562c8f254b3c9aaf8efd4678428f94dbbabfef4436bc50c1c87c6`.
Among nondeprecated `SXS:BBH:` entries, require run date at least
`2018-01-18`, `3 <= reference_mass_ratio <= 8`, both reference spin z
components in `[0.2,0.7]`, both transverse spin magnitudes below `0.01`,
reference eccentricity below `0.002`, at least 20 post-reference orbits, and
at least two `Lev*:Strain_N2.h5` files. Sort by
`|q-4| + |chi1z-0.5| + |chi2z-0.5|`, then numeric SXS ID. The five eligible
IDs in that order are 2158, 2163, 2162, 2155, 2502. This metadata-only
rule selects **SXS:BBH:2158v3.0**. Selection does not establish complete
independence from model development.

The [official case DOI](https://doi.org/10.26138/SXS:BBH:2158v3.0) resolves
to [CaltechDATA record 3gb3n-e5t37](https://data.caltech.edu/records/3gb3n-e5t37).
The catalog gives reference `q=3.0001323450539252`, spin vectors
`(-4.0913123854969844e-8, 4.2441616226361754e-8, 0.49997718539303)`
and `(-1.4945164577544106e-8, -1.639491623205979e-8,
0.5000516952582452)`, eccentricity `8.87e-5`, reference time `1384 M`,
22.603589434918916 post-reference orbits, initial orbital frequency
`0.015690006867/M`, and four available N2 strain resolutions. Use **Lev4
and Lev3**, the two highest. At `100 M_sun`, the initial approximate (2,2)
frequency is below the 30 Hz metric floor.

Retrieve only these six files from the catalog's
`/api/records/3gb3n-e5t37/files/<URL-encoded-key>/content` links **after**
the protocol and runner are committed. Verify byte size and published MD5
before decoding; record local SHA-256. Keep raw and derived strain outside Git.

| Catalog key | Bytes | Published MD5 |
| --- | ---: | --- |
| `Lev4:Strain_N2.h5` | 2990318 | `2c2ccd0a8505ae5d223283dbe39734df` |
| `Lev4:Strain_N2.json` | 2232 | `3b6e8c5656052f726771a95c0a9b4408` |
| `Lev4:metadata.json` | 3347 | `6f75989b5a2bb444a4ea612636994da2` |
| `Lev3:Strain_N2.h5` | 3280093 | `42bb7d97a1531a76d13b3cef0aa82d5c` |
| `Lev3:Strain_N2.json` | 2231 | `6f1c761e840c02ba4723e37048e35174` |
| `Lev3:metadata.json` | 3345 | `955f6b41c590f543656f88f34bdc291d` |

## Frozen signals, comparison, and gates

Apply every conditioning and scoring choice in the 3144 protocol **with Lev4
as the reference and Lev3 as the lower resolution**: full available inertial
NR `2 <= ell <= 5` including `m=0`; the public `sxs` loader; `100 M_sun`,
`500 Mpc`, `f_ref=30 Hz`; common post-reference interval, cubic complex-mode
interpolation to 8192 Hz, half-cosine `500 M` rise and `100 M` fall; two
fixed `(inclination,source psi)` views `(0.7,0)` and `(1.2,0.4)`; direct
30–1024 Hz Fourier sum; PyCBC `aLIGOZeroDetHighPower` PSD; one common
Lev3-to-Lev4 orbital-phase/time alignment using the 2048-point phase grid
and eight-peak refinement; independent Torch/JAX template time, orbital
phase, and polarization optimization with four seeds `(17,23,41,67)`.
Use exact catalog reference `q` for masses and model aligned spins
`chi1=0.49997718539303`, `chi2=0.5000516952582452`. The tiny transverse
catalog spins are not fitted. Native `phic=tc=phi0=0`; do not fit intrinsic
parameters, inclination, distance, any mode-specific factor, or overall
amplitude.

Both Lev3/Lev4 views must have match `>=0.995` and weighted norm ratio in
`[0.95,1.05]` under the **same** global alignment. Each Torch and JAX
model/Lev4 view must have optimized match `>=0.980`, weighted norm ratio
in `[0.85,1.15]`, and four-seed match spread `<=1e-5`. Report combined
resolution metrics, all optimizers, omitted-mode diagnostics, source and
package hashes, and direct/FFT consistency checks. A resolution failure
prevents physical qualification regardless of model scores. A model failure
may reflect a physical model limit and does not by itself prove an
implementation defect. One pass establishes only this case, band, mass,
PSD, views, and finite-resolution comparison; retain earlier red gates.
