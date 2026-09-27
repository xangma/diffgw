# IMRPhenomHM SXS:BBH:3619 projected-match holdout: frozen protocol

This protocol was committed **before downloading or decoding the 3619 strain**.
The two earlier SXS:BBH:2265 and 3631 phase-locked six-mode holdouts are red;
their data and outcomes were not used to select or tune this case. Do not change
the present gates after seeing 3619. A failure remains a failure. This is a
separate *projected-strain* claim; a pass cannot override either six-mode gate.

## Sources and immutable source identity

The model and metric source is [London et al., *First higher-multipole model of
gravitational waves from spinning and coalescing black-hole binaries*, Results
§](https://arxiv.org/html/1708.00404v4#S5): the paper compares full NR with
IMRPhenomHM using an aLIGO design-noise PSD from 30 Hz, holding intrinsic
parameters and inclination fixed while optimizing template arrival time,
polarization, and initial orbital phase. It reports typical matches above 0.99
for nonspinning mass ratios through 8. Our one-case protocol is paper-like,
not a reproduction of its 19-case ensemble or signal-strength-weighted angle
average. [PyCBC's documented `aLIGOZeroDetHighPower` PSD](https://pycbc.org/pycbc/latest/html/psd.html)
is a *validation-time* generator; no PSD array or LAL source is copied here.

The untouched numerical source is the SXS Collaboration's
[SXS:BBH:3619v3.0](https://doi.org/10.26138/SXS:BBH:3619v3.0),
[CaltechDATA record 6fw56-9q785](https://data.caltech.edu/records/6fw56-9q785),
interpreted with [SXS Catalog 3](https://arxiv.org/html/2505.13378v2).
The official version-3 catalog metadata, inspected before strain retrieval,
give mass ratio `5.000307`, component reference masses `0.833370` and
`0.166664`, spin magnitudes `0.000003` and `0.000066`, reference eccentricity
`0.000074`, reference time `342.5 M`, `28.811753` orbits, and initial orbital
frequency `0.015795/M`. The run began `2021-06-23`, after the 2018 model paper.
The metadata and record listing alone determined this choice. This is an
off-calibration *simulation-date* holdout, not proof that no later model
development ever used it.

Download only these official `Lev3`/`Lev2` N2 strain, sidecar, and metadata
files from `https://data.caltech.edu/api/records/6fw56-9q785/files/<key>/content`.
Verify byte count and published MD5 *before decoding*, and also record SHA-256
of each downloaded file. Raw files and derived arrays stay outside Git.

| key | bytes | MD5 |
| --- | ---: | --- |
| `Lev3:Strain_N2.h5` | 3699208 | `bad0925ff865cc144c2091f2f4aefdc7` |
| `Lev3:Strain_N2.json` | 2226 | `9ebeb958fee86210c3eda7efb526b1da` |
| `Lev3:metadata.json` | 3339 | `36bf4cc58ec6ef560746be61321b9547` |
| `Lev2:Strain_N2.h5` | 3814528 | `176cf650dac7672796bdd1a709f51af9` |
| `Lev2:Strain_N2.json` | 2227 | `a685135bc12a6fc7abe722d8dd221726` |
| `Lev2:metadata.json` | 3340 | `ff63fa10b797e22feeb216793661203e` |

## Frozen signal and conditioning

Use the official `sxs` decoder, requested version `2026.0.6`, and report exact
NumPy, SciPy, PyCBC, Torch, JAX, and Python versions. Require both files to
decode as inertial-frame, spin-`-2` strain `h=h_+-ih_×`, with mass and radius
scaled out and every `(ell,m)` for `2<=ell<=5`, `-ell<=m<=ell`. Use *all* of
these NR modes, including `m=0`; the native IMRPhenomHM template has its six
published positive-`m` modes and symmetry partners. Require sidecar and
decoded conventions to agree. SXS v3 sidecars may record CoM/memory
corrections; record them, and do not reverse or silently recondition them.

Set detector-frame mass `100 M_sun`, native masses `250/3` and `50/3 M_sun`
(q=5), spins zero, luminosity distance `500 Mpc`, `f_ref=30 Hz`, and native
`phic=tc=phi0=0`. Use nominal IAU 2015 `GM_sun=1.3271244e20 m^3 s^-2`, exact
`c=299792458 m/s`, and `1 pc=3.0856775814913673e16 m`. Convert NR time
by `GM/c^3` and NR strain by `GM/(c^2 D)`. At this mass, the catalog's initial
orbital frequency implies initial `(2,2)` frequency about `10.2 Hz`, below
the 30 Hz comparison floor.

For both resolutions use one common time interval from the largest of the
Lev3 reference time `342.5 M` and their file start times to the smallest file
end. Cubic-spline interpolate each mode's real and imaginary parts to the same
uniform `8192 Hz` grid. Set time zero at the Lev3 maximum of `|h_22|`. Multiply
all modes by a half-cosine amplitude ramp over the first `500 M` and last
`100 M`. Reject nonfinite or missing modes, a non-inertial frame, an interval
too short for both ramps, or an empty grid as *infeasible*. Apply a direct
Riemann-sum Fourier transform with `exp(-2πift)` at exactly
`f=30,31,...,1024 Hz`; no mode-specific cuts or fitted phase factors.

Project the full NR `ell<=5` strain with spin-weight `-2` spherical harmonics
into positive-frequency plus/cross. Use source inclination and polarization
angles `(0.7,0.0)` and `(1.2,0.4)` radians; source orbital azimuth is zero.
The template uses the independently specified six-mode projection in
`diffgw_reference.imrphenomhm_projection`. For template orbital phase `φ`,
multiply each native positive-frequency carrier `(ell,m)` by `exp(-imφ)` before
projection at zero azimuth; template detector strain is
`cos(2ψ) h_+(φ) + sin(2ψ) h_×(φ)`. The NR detector strain uses its frozen
source polarization. Both are real detector strains represented at positive
Fourier frequencies. No per-mode alignment or amplitude adjustment is allowed.

## Frozen paper-like match and gates

Generate a design PSD with PyCBC
`aLIGOZeroDetHighPower(1026, delta_f=1 Hz, low_frequency_cutoff=30 Hz)`;
require every sample for 30–1024 Hz to be finite and strictly positive, and
record its SHA-256 and dependency versions. The extra final PSD bin avoids a
known endpoint zero. For detector spectra `a,b`, use the one-sided real inner
product `4 Re Σ_(f=30)^1024 a*(f)b(f)/S_n(f)` and normalize by the two
self-products. Optimize the template over arrival-time shifts
`k/8192 s`, `k=-819,...,819`, orbital `φ∈[0,2π)`, and polarization
`ψ∈[0,π)`. Evaluate all discrete times at each angle pair by a zero-padded
8192-point real FFT of the weighted cross spectrum. Use SciPy
`differential_evolution` on `(φ,ψ)` with seeds `(17,23,41,67)`, `popsize=12`,
`maxiter=120`, `tol=1e-9`, `polish=True`, and one worker. The best of four is
the result; their final matches must agree within `1e-5`, or optimization is
infeasible. The maximizing time/angles and PSD-weighted template/NR norm ratio
are diagnostic outputs, not fitted correction parameters.

For the Lev2-versus-Lev3 numerical-resolution screen, retain the same fixed
orbital azimuth because these are resolutions of the *same* simulation.
Maximize the same PSD-weighted detector match over the stated discrete arrival
times and candidate polarization grid `ψ=jπ/1024`, `j=0,...,1023`; the Lev3
source polarization remains fixed. A valid case needs resolution match
`>=0.995` at both angles. For **each** Torch and JAX frontend, at both
source angles, require model/Lev3 optimized detector match `>=0.980` and
PSD-weighted template/Lev3 norm ratio in `[0.85,1.15]`. Both backends must
pass independently; mutual Torch/JAX parity alone cannot pass this gate.

The 0.980 floor leaves a one-percentage-point margin below London's stated
typical `>0.99` nonspinning performance because this is a single new SXS v3
case, with a specified finite band and memory-corrected NR. The amplitude
bound guards against a high normalized match masking a large SNR error. A pass
would qualify only these two detector projections at q=5, zero spins, 100
solar masses, this band/PSD/conditioning and optimization. Report any red
gate without changing it, and keep the prior six-mode red receipts visible.
