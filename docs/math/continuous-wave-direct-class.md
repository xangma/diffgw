# ContinuousWave monochromatic H1 direct class

Specification ID: `cw-monochromatic-h1-direct-class/v1`.

<!-- BEGIN GENERATED WAVEFORM DEFINITION -->

| Definition field | Catalog value |
| --- | --- |
| Model ID | `ContinuousWaveH1Direct` |
| Reference composition | `cw-monochromatic-h1-direct-class/v1` |
| Scope | Direct Torch/JAX ContinuousWave class: selected float64 CPU monochromatic principal-axis 2f strain at H1, GPS epoch in 2015–2016. |
| Domain and modes | `time`; `(2, 2)` |
| Native backends | `torch`, `jax` |
| Scoped verification claims | 1 |

| Option | Default | Native choices | Reference-only choices |
| --- | --- | --- | --- |
| `call_route` | `direct_class` | `direct_class` | None |
| `detector` | `H1` | `H1` | None |
| `spindown` | `f1=f2=0` | `f1=f2=0` | None |
| `amplitude_input` | `h0` | `h0`, `ellipticity_distance` | None |

<!-- END GENERATED WAVEFORM DEFINITION -->

## Scope and status

This version covers selected float64 CPU outputs from the direct Torch/JAX
`ContinuousWave` class with `detector="H1"`, `f1=f2=0`, and the principal-axis
$2f_{\rm rot}$ branch. Both `h0` and `(ellipticity,distance)` input routes
are checked. It is a **sampled implementation-verification claim**. The class
also permits other detectors and spin-down parameters; those calls are outside
this version.

## Inputs, conventions, and options

`t` and `t0` are GPS seconds, with samples in the 17-second GPS–UTC interval
from 2015-07 to 2016-12. `t0=1126259462` seconds sets both intrinsic frequency
and zero circular-orbit phase. `f0` is GW frequency in Hz, twice the stellar
rotation frequency. `ra`, `dec`, `psi`, and `phi0` are radians; `cos_iota` is
dimensionless. Either `h0` is given directly or the class computes it from
ellipticity, SI distance, and `i_zz` in kg m². The source is principal-axis,
non-precessing, and monochromatic. The project model uses a circular 1-AU
orbit, obliquity 0.4090928042223289 rad, and sidereal year 365.256363004
days. Earth rotation uses IAU 1982 GMST with UTC substituted for UT1.

## Mathematics and components

[Jaranowski, Królak and Schutz (1998), PDF Eqs. (23)–(24)](https://arxiv.org/pdf/gr-qc/9804014v1)
give $h_0=4\pi^2GI_{zz}\epsilon f_0^2/(c^4d)$ after mapping their
rotation frequency to the native GW frequency. With $\Delta t=t-t_0$,
Eq. (18)'s geometric Roemer delay is $\tau=\mathbf n\cdot
(\mathbf r_{\rm orbit}+\mathbf r_{\rm H1})/c$. The circular orbit and its
phase origin are project conventions. The monochromatic limit of Eq. (14)
is $\Phi=\phi_0+2\pi f_0(\Delta t+\tau)$.

JKS PDF Eqs. (1)–(6) form $F_+$ and $F_\times$ from the H1 arm tensor at the
Greenwich sidereal angle. Eqs. (19)–(22), at the principal-axis angle, give

$$h(t)=h_0\left[\frac{1+\cos^2\iota}{2}F_+(t)\cos\Phi(t)
+\cos\iota F_\times(t)\sin\Phi(t)\right].$$

The scalar reference computes these equations directly. The native class
contracts the JKS four-amplitude basis, so the test also checks the full
composition and convention signs.

| Component ID | Source and scope |
| --- | --- |
| `cw.direct_h1.amplitude.v1` | JKS PDF Eqs. 23–24; both native amplitude routes |
| `cw.direct_h1.clock.v1` | Aoki 1982 GMST, UTC approximation and fixed leap offset |
| `cw.direct_h1.delay_phase.v1` | JKS PDF Eqs. 14, 18; monochromatic circular-orbit adapter |
| `cw.direct_h1.antenna.v1` | JKS PDF Eqs. 1–6; conditioned on supplied H1 site vectors |
| `cw.direct_h1.strain.v1` | JKS PDF Eqs. 19–22 and native four-amplitude composition |

## Implementation mapping

`diffgw_reference.continuous_wave_direct_class.monochromatic_h1_sample`
uses mpmath and the existing paper-literal quadrupole and phase-response
references. It does not import native waveform functions or constants. The
H1 vertex and normalized arm vectors are explicit conditioning inputs from
the detector object. The two native `ContinuousWave` classes run their own
amplitude, timing, antenna, basis, and strain code paths.

## Evidence and limitations

The maintained test compares 13 checks: both backends at two sky/frequency
configurations, five GPS sample times per amplitude route, plus selected
$f_0$ and $\phi_0$ automatic derivatives against scalar differentiation. The
existing component tests separately check the H1 arm directions to the four
decimals printed in [Anderson et al., Appendix B Table I](https://arxiv.org/pdf/gr-qc/0008066v1).
The inherited full-precision H1 survey numbers are **not** independently
verified by the scalar comparison. [IERS Bulletin C49](https://datacenter.iers.org/data/html/bulletinc-049.html)
supports the 2015 leap offset; UTC remains an approximation to UT1 and no
Earth-orientation series is supplied. The circular orbit has no planetary
ephemeris, Einstein/Shapiro delay, binary orbit, or proper motion. The JKS
historical H1 arm table was corrected by Anderson; it is not used as an exact
H1 oracle. Native nonzero spin-down terms contain higher powers of the delay
than JKS Eq. (14), recorded by a strict expected-failure diagnostic and
excluded here. This claim establishes neither external binary compatibility
nor astrophysical or domain-wide physical accuracy.

## Reproduce the checks

Run `python -m pytest -q tests/test_continuous_wave_direct_class_math_reference.py`
and `python -m pytest -q tests/test_continuous_wave_phase_response_reference.py`
with the optional reference, Torch, and JAX dependencies installed. Run
`python scripts/generate_waveform_library_table.py --check` to verify generated
catalog tables.
