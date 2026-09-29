# IMRPhenomHM SXS:BBH:3144: prospective physical-validation protocol

**Frozen before strain retrieval or model scoring.** The only case-selection
input inspected was the official SXS v3.0.0 *catalog metadata* described
below. Do not revise the case, waveform conditioning, optimization, or gates
after decoding its strain. Record failures and infeasible comparisons as
such. This is a new test; the existing [2265](imrphenomhm_sxs_2265_receipt_20260927.md),
[3631](imrphenomhm_sxs_3631_receipt_20260927.md),
[3619](imrphenomhm_sxs_3619_projection_receipt_20260927.md), and
[3630](imrphenomhm_sxs_3630_projection_receipt_20260927.md) gates and
receipts retain their original status. The selection, conditioning, and
gates below were fixed before examining any 3144 strain or score.

## Literature target and limits

[London et al. (2018)](https://arxiv.org/html/1708.00404v4) compared
IMRPhenomHM against full numerical-relativity signals through `ell=5` using
the Advanced LIGO design noise curve. Their physical comparison used a
30 Hz floor and, for the figure discussed here, `100 M_sun`; intrinsic
parameters and source inclination were fixed while template arrival time,
polarization, and initial orbital phase were optimized. They reported
typically greater than 0.99 match for nonspinning mass ratios at most eight
in their 19-waveform set, with signal-strength-weighted angular summaries.
The present two-view, one-case gate is deliberately narrower and does not
reproduce that population or angular average. It tests an independently
selected later SXS simulation with a comparable PSD, frequency floor,
mass, and extrinsic optimization.

[SXS Catalog 2019 Appendix B](https://arxiv.org/html/1904.04831#S.app2)
motivates comparing `h=h_+ - i h_×` with one **common** arrival-time and
orbital-azimuth adjustment across NR modes, using plus and cross as a
two-detector network. The present fixed taper, fixed band, and two views
are specified below; they do not claim to reproduce that paper's Planck
taper, adaptive frequency band, or sky sampling. The
[SXS Catalog 3 paper](https://arxiv.org/html/2505.13378v2) describes v3
waveform conventions and its separate sphere-averaged resolution analyses.
The resolution screen below is an operational two-view network test, not a
claim of catalog-style convergence or a NR error bound.

## Prospective metadata-only case selection and source identity

Use the official `sxscatalogdata` release
[`v3.0.0/simulations.json`](https://raw.githubusercontent.com/sxs-collaboration/sxscatalogdata/v3.0.0/simulations.json),
associated with the [SXS v3.0.0 release](https://zenodo.org/records/15415231).
The locally cached JSON is 32,618,880 bytes and has SHA-256
`132d38520b3562c8f254b3c9aaf8efd4678428f94dbbabfef4436bc50c1c87c6`.
Its 4,170 catalog entries contain metadata and file manifests, not strain
arrays. Preserve this hash and the exact filtering rule in the receipt.

For each key beginning `SXS:BBH:`, retain entries with no `deprecated`
keyword, `reference_mass_ratio` in `3 <= q <= 8`, Euclidean magnitudes of
*both* `reference_dimensionless_spin1` and
`reference_dimensionless_spin2` vectors below `0.001`,
`reference_eccentricity < 0.001`, `date_run_earliest >=
2018-01-18T00:00:00`, `number_of_orbits >= 20`, a finite
`number_of_orbits_from_reference_time`, and at least two available
`Lev*:Strain_N2.h5` files in the manifest. Explicitly remove the four
previously examined IDs 2265, 3631, 3619, and 3630. Sort the remaining
entries by **descending** `number_of_orbits_from_reference_time`, then by
ascending numeric SXS ID, and take the first. Longer post-relaxation data
improve the chance of a complete 30 Hz comparison at `100 M_sun`; the date
cut avoids simulations run before the cited London et al. revision, while
`q >= 3` emphasizes systems where higher modes matter. This selects
**SXS:BBH:3144v3.0**, with 29.592976667962976 post-reference orbits.
The remaining eligible IDs in sort order are 2374, 2499, 2164, 2498.
The chronological choice does not prove independence from every later
model-development activity.

The [official SXS:BBH:3144v3.0 DOI](https://doi.org/10.26138/SXS:BBH:3144v3.0)
resolves to [CaltechDATA record 1g62w-w0b49](https://data.caltech.edu/records/1g62w-w0b49).
The catalog gives reference `q=4.499621846563142`, eccentricity
`0.0002219`, reference spin vectors
`(4.021819846540122e-7, 1.7207127249235393e-7, 2.9715550036843704e-5)`
and `(8.852292173961585e-8, 2.673455377441355e-8,
1.6474344378490387e-5)`, reference time `1580.5 M`, 33.26561261546802
total orbits, and run start `2019-11-30T19:56:54`. The initial orbital
frequency `0.014212721029/M` implies an approximate initial `(2,2)`
frequency of 9.2 Hz at `100 M_sun`, below the metric floor. The manifest
lists only Lev2 and Lev3 strain; therefore **both available resolutions**
are mandatory for the resolution screen.

Retrieve only the following six files from the record's
`/api/records/1g62w-w0b49/files/<URL-encoded-key>/content` endpoints in
the execution phase. Verify published byte size and MD5 before decoding,
record local SHA-256, and keep raw and derived strain outside Git. No such
files had been retrieved when this protocol was written.

| Catalog key | Bytes | Published MD5 |
| --- | ---: | --- |
| `Lev3:Strain_N2.h5` | 3948147 | `78a1bed032fa8ae31a167d4fa30203fb` |
| `Lev3:Strain_N2.json` | 2229 | `73edba3df5fbcef49acc0f6a8c5c5bc9` |
| `Lev3:metadata.json` | 3373 | `fbdddb96126b46b4a1881ba84ced2e8d` |
| `Lev2:Strain_N2.h5` | 4149375 | `0324ae1236564a44f4a998364b829d4f` |
| `Lev2:Strain_N2.json` | 2231 | `e9551b14ff83ecd735dce7524091cc1b` |
| `Lev2:metadata.json` | 3374 | `21415531730d1b93b5fd7da69f87b584` |

## Fixed signals and conditioning

Decode the official inertial-frame, extrapolated N2, spin-weight `-2`
strain with the public `sxs` loader (request `2026.0.6`; record the exact
installed package and dependency versions). Require every NR mode
`2 <= ell <= 5`, `-ell <= m <= ell`, **including `m=0`**. Reject missing
modes, nonfinite arrays, an inconsistent frame, or sidecar/metadata
contradictions. Record v3 center-of-mass and memory modifications from the
sidecars; do not undo them. The native IMRPhenomHM template contains its
six published positive-`m` modes `(2,2), (3,3), (4,4), (2,1), (3,2),
(4,3)` and symmetry partners. Report that missing NR modes, especially
`m=0` and the remaining `ell<=5` modes, are a possible contribution to
model mismatch; do not remove them from the primary comparison.

Fix detector-frame total mass to `100 M_sun`, component masses to
`100q/(1+q)` and `100/(1+q) M_sun` with the **catalog reference q above**,
model spins to zero, luminosity distance to `500 Mpc`, `f_ref=30 Hz`, and
native `phic=tc=phi0=0`. The tiny nonzero NR reference spins are treated
as numerical residuals, not fitted model spins. Use nominal
`GM_sun=1.3271244e20 m^3 s^-2`, exact `c=299792458 m/s`, and
`1 pc=3.0856775814913673e16 m`; scale NR time by `GM/c^3` and strain by
`GM/(c^2 D)`. Record the conversion and any overall strain normalization
provided by the loader. Reject a convention ambiguity rather than fitting
an amplitude scale.

Use the common time interval starting at the maximum of the Lev2/Lev3
reference times and file starts and ending at the minimum file end. Cubic
spline each complex mode's real and imaginary parts onto **one** uniform
8192 Hz time grid. Set time zero at the Lev3 maximum of `|h_22|`. Apply
the same half-cosine rise across the first `500 M` and fall across the last
`100 M` to every mode of both resolutions; reject an interval unable to
support both ramps. Form the `h=h_+-ih_×` projection with spin-weight
`-2` harmonics and Fourier convention `exp(-2πift)`, using a direct
Riemann sum at integer `f=30,...,1024 Hz` for the reported spectra.
Fix source azimuth to zero and the two source views to
`(iota,psi_s)=(0.7,0.0)` and `(1.2,0.4)` radians. For the physical
model/NR comparison, form each scalar detector spectrum as
`cos(2ψ)h_+ + sin(2ψ)h_×`, with `ψ_s` fixed for NR and template `ψ`
optimized. Apply template orbital phase `φ` to each positive-frequency
carrier as `exp(-imφ)` before projection. Do not optimize masses, spins,
inclinations, distance, or any mode-specific phase or amplitude.

## PSD, common NR alignment, and model optimization

Generate PyCBC's documented
[`aLIGOZeroDetHighPower`](https://pycbc.org/pycbc/latest/html/psd.html)
PSD with length `1026`, `delta_f=1 Hz`, and low-frequency cutoff `30 Hz`.
Require finite positive samples at all comparison bins and record PSD
SHA-256 and library version. For any one real detector channel define
`(a|b)=4 Re Σ[f=30..1024] a*(f)b(f)/S_n(f)` and match
`(a|b)/sqrt((a|a)(b|b))`. A plus/cross network uses the sum of channel
inner products before normalization; a two-view network additionally
sums both views. Norm ratio always means `sqrt((template|template) /
(reference|reference))` with the same channel set and PSD.

**One global Lev2-to-Lev3 alignment is required for every NR resolution
check.** Apply `exp(-imφ_NR)` to *all* Lev2 `(ell,m)` modes and one common
arrival shift `τ_NR`. Choose this pair by maximizing the normalized
plus/cross **two-view network** match of full `ell=2..5` Lev2 against
Lev3, with both view inclinations fixed and their common, unrotated plus
and cross channels. The polarization basis cancels from a plus/cross
network, so neither source `ψ_s` nor any new polarization rotation is
optimized here. Search `φ_j=2πj/2048`, `j=0,...,2047`; at each trial find
the best arrival shift `τ=k/8192 s`, integer `k=-819,...,819`, via a
zero-padded 8192-point real inverse FFT of the weighted cross spectrum.
Take the eight highest distinct periodic local phase maxima and refine
each within `±2π/2048` by bounded scalar minimization with
`xatol=1e-10`, `maxiter=500`. Retain the best grid/refined result. Verify
FFT sign and normalization on a synthetic exact shift and with direct
inner products at the optimum; an FFT-derived spectrum must reconstruct
the direct spectrum at zero phase within relative maximum error `1e-10`
per plus/cross channel and view. Record candidate scores and optimum.
Freeze **that same** `(φ_NR,τ_NR)` for the combined and each individual
view's Lev2/Lev3 match and norm ratio and for any per-mode diagnostic.
There is no per-view, per-polarization, or per-mode retuning.

At each fixed source view, optimize the native Torch and JAX model
frontends independently against Lev3 in the one-detector channel over
template arrival shift `k/8192 s` in the same range, orbital
`φ∈[0,2π)`, and polarization `ψ∈[0,π)`. Use SciPy
`differential_evolution` for the two angles with seeds `(17,23,41,67)`,
`popsize=12`, `maxiter=120`, `tol=1e-9`, `polish=True`, one worker. Take
the best of four and require their final matches to agree within `1e-5`;
otherwise label optimization infeasible. At each trial angle maximize
over the discrete arrival shifts using the same FFT rule. Check reported
optima by direct inner products. Apply no global amplitude fit. Record
all optimized parameters, signal norms, full mode list, model/backend
versions, and numeric conditioning diagnostics in the result receipt.

## Precommitted gates and interpretation

Both resolution views must have match `>=0.995` and Lev2/Lev3 network
norm ratio in `[0.95,1.05]` **under the one common alignment above**;
report the combined two-view values as diagnostics. A 0.5% overlap deficit
and 5% weighted-amplitude interval are predeclared limits for using these
two finite-resolution waveforms as a reference, not a published estimate
of their continuum error. If this screen fails or is infeasible, no
physical model qualification is awarded, even if model scores are high.

For **each** Torch and JAX native frontend, at **both** views require
optimized model/Lev3 one-detector match `>=0.980` and weighted norm ratio
in `[0.85,1.15]`. London et al.'s reported typical `>0.99` nonspinning
matches motivate a high but deliberately weaker one-case `0.98` floor;
the 15% amplitude interval is a conservative, explicit operational
allowance for a higher-mode model, not a theorem derived from their
mode-error plots. These thresholds are fixed before any selected strain
is decoded. The older gates remain unchanged and cannot be rescued by
this result. No failed gate may be relaxed after inspection.

A pass supports only this q≈4.5, nearly nonspinning SXS case at `100 M_sun`,
these two views, band, PSD, conditioning, and six-mode template against
the full available NR signal. It does not establish all-sky performance,
the 19-case result of London et al., spin or mass extrapolation, continuum
NR accuracy, or agreement with another model implementation. Report the
omitted-mode contribution, weak-mode diagnostics, and any convention or
optimization limitations as diagnostics without redefining the gates.
