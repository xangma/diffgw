# IMRPhenomHM independent SXS holdout: frozen protocol

This protocol is committed **before decoding either holdout waveform**. Its
acceptance gates are not to be changed in response to the result. A red or
infeasible run remains a red or infeasible result and needs a separately named
future protocol. The test concerns the public, default-D-carrier, aligned-spin
`IMRPhenomHM` six-mode Torch and JAX APIs. It uses no LAL code or coefficients.

## Sources, case, and data identity

* Model source: [London et al., *First higher-multipole model of gravitational
  waves from spinning and coalescing black-hole binaries*](https://arxiv.org/html/1708.00404),
  especially the Results section. The paper reports typical full-polarization
  NR matches above 0.99 for nonspinning mass ratios through 8 using an aLIGO
  design PSD, 30 Hz lower bound, and optimized time, polarization and initial
  orbital phase. It reports individual `(3,2)`/`(4,3)` averages near 0.92 and
  typical subdominant-mode amplitude errors well below 15%. Our flat-band,
  one-22-alignment gates below are deliberately more permissive because those
  reported match values are not the identical statistic.
* Numerical source: SXS Collaboration,
  [SXS:BBH:2265v3.0](https://doi.org/10.26138/SXS:BBH:2265v3.0),
  [CaltechDATA record 258wv-a5t45](https://data.caltech.edu/records/258wv-a5t45),
  [SXS catalog paper](https://doi.org/10.1103/PhysRevD.99.064045), and
  [SXS Catalog 3](https://doi.org/10.26138/SXSCatalog3). The metadata report
  `reference_mass1=0.7500097310012441`,
  `reference_mass2=0.24999389181355253`,
  `reference_dimensionless_spin1_z=8.310310779581836e-06`,
  `reference_dimensionless_spin2_z=7.14361976879799e-06`,
  `reference_eccentricity=7.17e-05`, `reference_time=1077.5 M`, and
  65.56197 orbits from the start. This is a near-nonspinning q=3 case inside
  the paper's stated domain. It was published after the 2018 model paper and
  is used only for validation, never fitting.
* Source files are the official record's `Lev3` and `Lev2` **N2** strain and
  sidecar/metadata pairs. Verify both size and MD5 against this frozen list
  before decoding. The exact content URLs have the form
  `https://data.caltech.edu/api/records/258wv-a5t45/files/<key>/content`.

| key | bytes | MD5 |
| --- | ---: | --- |
| `Lev3:Strain_N2.h5` | 5894558 | `2a7336507c63dbea3b3980cd6d3a783a` |
| `Lev3:Strain_N2.json` | 2229 | `76283d83e102361309030d4a4f6bd9fb` |
| `Lev3:metadata.json` | 3304 | `ea2133e45d1075aa4f53068a3cc0e437` |
| `Lev2:Strain_N2.h5` | 6246326 | `6c98c77914cd7cd30f558b35f6fd7d7c` |
| `Lev2:Strain_N2.json` | 2227 | `268bf2bd33cb16c7b4b10134da0694ad` |
| `Lev2:metadata.json` | 3309 | `97a683f1e7f738d81895e09ee6d3236f` |

The raw SXS arrays stay outside Git. SXS asks users to acknowledge the first
scientific publication describing a simulation. The record API does not
declare a per-record SPDX license, so no such license is inferred.

## Frozen conditioning and conventions

Use the official `sxs` decoder, requested version `2026.0.6` if available,
with NumPy/SciPy, and record exact installed versions. Require strain type
`h`, spin weight `-2`, the six positive-m modes
`(2,2),(2,1),(3,3),(3,2),(4,4),(4,3)` and their negative-m partners in both
resolutions. Reject absent/nonfinite modes. SXS uses
`h=h_+ - i h_× = Σ h_lm {}_{-2}Y_lm`; the sidecar states mass and radius
are scaled out. Require the two sidecars to agree on these conventions.

Set detector-frame total mass `80 M_sun`, component masses `60,20 M_sun`,
aligned spins `0,0` (the metadata's residual spins are numerical), distance
`500 Mpc`, `f_ref=30 Hz`, and model `phic=tc=phi0=0`. Convert SXS code time
to seconds with `GM_sun/c^3` and strain to physical dimensionless amplitude
with `80 GM_sun/(c^2 D)`. Use nominal IAU 2015
`GM_sun=1.3271244e20 m^3 s^-2`, exact `c=299792458 m s^-1`, and
`1 pc=3.0856775814913673e16 m`. These public unit definitions are not
copied from LAL.

For each resolution, retain the common SXS time interval beginning at the
larger of `1077.5 M` and the two file start times and ending at the smaller
file end time. Resample real and imaginary parts of each selected complex mode
onto a uniform `1/8192 s` grid by SciPy cubic spline; use the same physical
grid for both resolutions. Set the time origin at the maximum of the Lev3
`|h_22|` on this grid. Apply a half-cosine amplitude ramp over the first
`500 M` and last `100 M` of the common interval. Both windows must fit without
overlap; otherwise the protocol is infeasible. This removes initial junk and
endpoint leakage. No per-mode cuts, rescalings or phase fits are allowed.

Evaluate the direct Riemann-sum Fourier transform at exactly
`f=30,31,...,350 Hz` with kernel `exp(-2πift)` and `dt=1/8192 s`.
The native positive-frequency mode target is
`H_lm(f)=dt Σ conj(h_lm(t)) exp(-2πift)` times the physical amplitude scale.
This is the conjugate of the negative-frequency SXS positive-m strain mode;
it follows from the stated Fourier convention, not an empirical adapter.
Project *only the six selected ±m pairs* to complex positive-frequency plus
and cross with the independently specified spin-weighted harmonics in
`src/diffgw_reference/imrphenomhm_projection.py`. Do not compare to the full
SXS multipolar polarization, which contains modes absent from IMRPhenomHM.
Use observer angles `(inclination, azimuth)=(0.7,0.3)` and `(1.2,-0.2)` radians.

Align **once**, using only the Lev3 `(2,2)` data and the native `(2,2)` mode
over `30..80 Hz`: search `Δt` on `[-0.1,0.1] s` at steps of `1/8192 s`,
maximizing `|Σ H_NR,22 conj(H_model,22) exp(2πifΔt)|`. Select the earliest
grid point on an exact tie. Let `φ=-arg(C)/2`, for the complex sum `C` at the
selected shift. Apply `exp(-2πifΔt) exp(-imφ)` to **every** native mode and its
resulting plus/cross; the polarization projection must use these aligned
modes. There is no per-mode time, phase, amplitude, polarization or orbital
phase optimization. Use the identical rule on Lev2 versus Lev3 `(2,2)` solely
to estimate numerical resolution error, applying that one Lev2 alignment to
all Lev2 modes/projections. Record both shifts and phases.

## Frozen statistics and gates

All statistics use the **full 30..350 Hz grid**, flat frequency weighting,
with no pointwise masking or PSD. For complex vectors `a` (Lev3 NR) and `b`
(aligned native), define phase-sensitive `match=Re Σ conj(a)b/(||a||||b||)`
and `amplitude_ratio=||b||/||a||`. Norms must be finite and nonzero. Compute
the same statistics for Lev3 versus aligned Lev2. Torch and JAX must each
pass independently; their mutual parity is a separate gate elsewhere.

| observable | minimum model–Lev3 match | model/Lev3 amplitude ratio | minimum Lev2–Lev3 match |
| --- | ---: | ---: | ---: |
| `(2,2)`, `(3,3)` | 0.97 | 0.90–1.10 | 0.995 |
| `(2,1)`, `(4,4)` | 0.94 | 0.80–1.20 | 0.99 |
| `(3,2)`, `(4,3)` | 0.85 | 0.75–1.25 | 0.97 |
| six-mode `plus` and `cross` at each angle | 0.95 | 0.85–1.15 | 0.99 |

The resolution gate must pass before interpreting any model comparison.
These bounds are conservative case-specific tests informed by London et al.'s
reported NR behavior and by the explicit two-resolution screen. They do not
prove the paper's PSD-weighted/population claims. Any passing claim is limited
to this q=3 nearly nonspinning source, `80 M_sun`, these two orientations,
six-mode truncation, this conditioning and 30–350 Hz flat band. A gate failure
is reported without weakening a bound. A decoder, data, or convention failure
is reported as infeasible, not as a pass or an optional skip.
