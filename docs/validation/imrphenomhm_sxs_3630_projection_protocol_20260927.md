# IMRPhenomHM SXS:BBH:3630 projected-match holdout: frozen protocol

Commit this document **before retrieving or decoding either 3630 strain**.
This is one new, independent physical holdout. Do not change its sources,
conditioning, optimizers, or gates after seeing the waveform. The earlier
[3619 projected holdout](imrphenomhm_sxs_3619_projection_receipt_20260927.md)
remains red under its frozen, no-orbital-retuning resolution screen, even
though a clearly post-result orbital-alignment diagnostic gave high NR
agreement. The [2265](imrphenomhm_sxs_2265_receipt_20260927.md) and
[3631](imrphenomhm_sxs_3631_receipt_20260927.md) six-mode gates also stay red.

## Pre-data case selection and source identity

The official `sxs.load("dataframe", tag="3.0.0")` metadata were filtered for
nondeprecated runs beginning after 2018-05-01, `5<=q<=8`, both reference spin
magnitudes `<0.001`, and reference eccentricity `<0.005`. Only 3619 and 3630
met these criteria; 3619 has already been inspected. The remaining
**SXS:BBH:3630v3.0** has q `6.000520906220441`, spin magnitudes
`5.9110497766760964e-05` and `2.4576284716767174e-05`, reference
eccentricity `0.000434`, `21.24783108430796` orbits, initial orbital
frequency `0.019315320434/M`, and Lev3 reference time `778.0 M`. The run
began 2022-05-11. Selection used metadata only, with no 3630 strain outcome.
The initial `(2,2)` frequency at the frozen `100 M_sun` mass is about
`12.5 Hz`, below the 30 Hz metric floor. The 2022 run date is later than the
2018 model paper, but is not proof that no later model development used it.

The [official SXS:BBH:3630v3.0 DOI](https://doi.org/10.26138/SXS:BBH:3630v3.0)
resolves to [CaltechDATA record 28pv5-4wj69](https://data.caltech.edu/records/28pv5-4wj69).
The untouched Lev3/Lev2 N2 strain, sidecar, and metadata files below are the
only source files. Retrieve them from the record's `/api/records/28pv5-4wj69/files/<key>/content`
endpoints. Check byte size and the published MD5 *before decoding* and record
SHA-256. Keep raw and derived arrays outside Git.

| key | bytes | official MD5 |
| --- | ---: | --- |
| `Lev3:Strain_N2.h5` | 3040680 | `f415773285072330601c0c39d4d3f0b5` |
| `Lev3:Strain_N2.json` | 2227 | `67aeb7928db5fdab25bb008465bdbeee` |
| `Lev3:metadata.json` | 3364 | `6629f45f5a1fd26c0e2b284c18ae57ea` |
| `Lev2:Strain_N2.h5` | 3276474 | `e29f4cafff41290630179327824ac7ba` |
| `Lev2:Strain_N2.json` | 2227 | `c136be72e617904f2f1895321406cc5d` |
| `Lev2:metadata.json` | 3367 | `a4846f6ab9c7a265476eed5245e40b4b` |

## Frozen NR and model signals

Use the official `sxs` decoder, requested version `2026.0.6`, and record
exact dependency versions. Require decoded inertial-frame spin-`-2` strain
`h=h_+-ih_×`, mass and radius scaled out, and every `(ell,m)` for
`2<=ell<=5`, `-ell<=m<=ell`. Use *all* these NR modes, including `m=0`; the
native IMRPhenomHM template uses its six published positive-`m` modes with
symmetry partners. Check sidecars and decoded conventions, record CoM/memory
modifications, and never undo them. Reject missing modes, nonfinite data,
non-inertial frame, or metadata inconsistent with q≈6 and negligible spins.

Set detector-frame total mass `100 M_sun`, native component masses `600/7`
and `100/7 M_sun`, zero model spins, distance `500 Mpc`, `f_ref=30 Hz`, and
native `phic=tc=phi0=0`. Use nominal `GM_sun=1.3271244e20 m^3 s^-2`, exact
`c=299792458 m/s`, and `1 pc=3.0856775814913673e16 m`. Scale NR time by
`GM/c^3` and NR strain by `GM/(c^2 D)`. For both resolutions use the common
interval from the largest of Lev3 reference time `778.0 M` and file starts
to the smallest file end. Cubic-spline interpolate each complex mode's real
and imaginary parts to one `8192 Hz` uniform grid. Set time zero at Lev3
maximum `|h_22|`; apply a half-cosine rise over the first `500 M` and fall
over the last `100 M` to every mode. Reject an interval too short for both
ramps. Fourier transform with `exp(-2πift)` at precisely integer
`f=30,...,1024 Hz`; use the direct Riemann sum for full NR projections. For
the resolution-screen orbital basis, a zero-padded FFT on the same samples is
permitted only if its sum at zero orbital phase reconstructs the direct full
NR spectra to relative maximum error `<=1e-10` for plus and cross at each
inclination. No fitted mode-dependent factor or cut is allowed.

Project NR with spin-weight `-2` spherical harmonics at source inclinations
and polarizations `(iota,psi_s)=(0.7,0.0),(1.2,0.4)` radians and zero source
azimuth. For the model, use the independent six-mode scalar projection in
`diffgw_reference.imrphenomhm_projection`. At template orbital phase `φ`,
multiply each native positive-frequency carrier `(ell,m)` by `exp(-imφ)`
before projection. The one-detector model/NR spectra are
`cos(2ψ) h_+ + sin(2ψ) h_×`, with the frozen source `ψ_s` for NR and optimized
template `ψ`. The NR resolution comparison instead uses both unrotated
plus/cross channels at each inclination, as specified below.

## Frozen metrics, optimization, and pass conditions

Generate [PyCBC's documented `aLIGOZeroDetHighPower`](https://pycbc.org/pycbc/latest/html/psd.html)
PSD with length `1026`, spacing `1 Hz`, and low cutoff `30 Hz`; require
finite positive samples at every comparison bin and record the PSD SHA-256.
This is a validation-time dependency, not copied LAL source or array. For
one detector define `(a|b)=4 Re Σ_(f=30)^1024 a*(f)b(f)/S_n(f)` and the
normalized overlap. For the Lev2–Lev3 screen use the two-detector network
sum of this inner product over **plus and cross**, with both channel norms
summed before normalization. This follows the [SXS Catalog 2019 Appendix B
alignment](https://arxiv.org/html/1904.04831#S.app2): optimize only a
*single common orbital azimuth* applied to all Lev2 modes and one arrival
shift. Do **not** optimize a relative polarization rotation for this NR
screen. The paper's separate Planck taper, sphere sampling, and adaptive band
are not claimed here; the conditioning and band above remain fixed for this
one-case test.

For both model and NR screen, arrival shifts are `k/8192 s` for integer
`k=-819,...,819`, evaluated at each trial angle by a zero-padded 8192-point
real inverse FFT of the weighted cross spectrum. Verify the FFT sign and
scale on a synthetic exact shift and by direct inner-product evaluation at
reported optima. On the model side, optimize template `φ∈[0,2π)` and
`ψ∈[0,π)` with SciPy `differential_evolution`, seeds `(17,23,41,67)`,
`popsize=12`, `maxiter=120`, `tol=1e-9`, `polish=True`, one worker. Take the
best of four; all final matches must be within `1e-5`, else optimization is
infeasible. On the NR resolution side, evaluate `φ_j=2πj/2048` for
`j=0,...,2047`, select the eight highest **distinct periodic local maxima**,
then refine each within `±2π/2048` using bounded scalar minimization with
`xatol=1e-10`, `maxiter=500`. Take the best of the 2048 grid points and eight
refinements; record all candidate scores, phase, arrival shift, and norm
ratio. The same phase rotates all `2<=ell<=5` Lev2 modes by `exp(-imφ)` in
the time domain before projecting plus/cross, with no polarization change.

At both angles the Lev2–Lev3 network match must be `>=0.995`. For **each**
native Torch and JAX frontend, at both angles require optimized one-detector
model/Lev3 match `>=0.980` and PSD-weighted template/Lev3 norm ratio in
`[0.85,1.15]`. Both backends must pass independently; their parity alone is
not physical qualification. A pass is limited to this case, mass, band, PSD,
angles, modes, and method and cannot override prior red holdouts. Report red
or infeasible gates without relaxation or production coefficient edits.
