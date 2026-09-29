# EMRI research specifications

This file separates the checked Newtonian eccentric source strain from the older provisional `EMRIKludge`. The checked model `NewtonianEccentricEMRI` is registered in the public waveform catalog (`emri.newtonian_eccentric_quadrupole.v1`) within its declared parameter domain; the older prototype `EMRIKludge` remains outside the catalog. The checked model is a weak-field, leading-quadrupole, fixed-observer limit; it is not a generic Kerr EMRI, the complete Barack–Cutler analytic kludge (AK), a numerical kludge (NK), or a LISA response.

<!-- BEGIN GENERATED WAVEFORM DEFINITION -->

| Definition field | Catalog value |
| --- | --- |
| Model ID | `emri.newtonian_eccentric_quadrupole.v1` |
| Reference composition | `emri-newtonian-eccentric-quadrupole/v1` |
| Scope | Weak-field, leading-quadrupole Newtonian eccentric EMRI polarizations at a fixed observer with adiabatic Peters p,e evolution. |
| Domain and modes | `time`; `(2, 2)` |
| Native backends | `torch`, `jax` |
| Scoped verification claims | 0 |

| Option | Default | Native choices | Reference-only choices |
| --- | --- | --- | --- |
| `n_harmonics` | `5` | `2`, `3`, `4`, `5`, `6`, `7`, `8` | None |

<!-- END GENERATED WAVEFORM DEFINITION -->

## Scope and status

`NewtonianEccentricEMRI` implements adiabatic Peters evolution of Keplerian `p,e`, a mean anomaly, the finite Peters–Mathews eccentric Bessel sum written in [Barack and Cutler (2004), Sec. III B, Eqs. (2)–(10), (27)](https://arxiv.org/pdf/gr-qc/0310125), and source-frame polarizations at a fixed observer. A separate SymPy/mpmath specification and numerical comparisons check this bounded model. Its definition is `emri.newtonian_eccentric_quadrupole.v1`.

The pre-existing `EMRIKludge` uses a frozen initial flux, hand-set sidebands, approximate frequencies, and static LISA antenna multiplication. Its only independently checked component is the leading Peters `p,e` flux. It has no whole-waveform qualification and remains provisional. The new class does not call that class or its detector projection.

## Inputs, conventions, and options

`M` (central mass) and `mu` (secondary mass) are in solar masses; `p=a(1-e²)/M` is dimensionless; `e` is eccentricity; `t` is an increasing one-dimensional array of seconds; `phase0`, observer inclination `iota`, and periapsis angle `gamma` are radians; `D` is in Mpc. Inclination is the angle between fixed orbital angular momentum and line of sight. The return is source-frame `(h_plus,h_cross)` on the supplied time grid, with the polarization convention of Barack–Cutler Eq. (10). `phase0` is the mean anomaly at the first sample. No sky position or detector frame is accepted.

The native option has integer `2 <= n_harmonics <= 8`, default 5, and sums positive harmonics `1,...,N`. It requires positive `M,mu,D`, `mu/M <= 10^-3`, `p >= 12`, and `0 <= e <= 0.6` for the whole trajectory. The maximum time step is at most a quarter of the initial Kepler period `2π M_sec [p/(1-e²)]^(3/2)`. Executable comparisons use CPU float64; JAX requires `jax_enable_x64=True`. Eager calls reject out-of-domain parameters and grids. Under JAX `jit`, traced numeric inputs bypass Python validation, so callers must check the same domain before execution. No accuracy promise is made for longer grids or parameter regions beyond the tested cases. No spin, precession, generic Kerr frequencies, higher-PN corrections, LISA response, interpolation table, or external asset enters this definition.

The older `EMRIKludge` has different parameters and behavior. Its native flux clamps `p>=1`, `0<=e<=0.9999`, and `1-e²>=1e-8`; the flux reference checks its unclamped interior only. The old class uses first-order initial-flux extrapolation and a rectangular phase sum, not an integrated inspiral.

## Mathematics and components

[Peters (1964), Eqs. (5.6)–(5.7)](https://doi.org/10.1103/PhysRev.136.B1224) gives the leading orbit-averaged rates. With `A=a/M`, `p=A(1-e²)`, `q=mu_sec/M_sec²`, and masses in geometric seconds,

\[
\dot A=-\frac{64}{5}q A^{-3}(1-e^2)^{-7/2}
 \left(1+\frac{73}{24}e^2+\frac{37}{96}e^4\right),\quad
\dot e=-\frac{304}{15}q e A^{-4}(1-e^2)^{-5/2}
 \left(1+\frac{121}{304}e^2\right).
\]

The exact chain rule `dot p=(1-e²)dot A-2Ae dot e` gives

\[
\dot p=-\frac{64}{5}q p^{-3}(1-e^2)^{3/2}
 \left(1+\frac78e^2\right),\quad
\dot e=-\frac{304}{15}q e p^{-4}(1-e^2)^{3/2}
 \left(1+\frac{121}{304}e^2\right),\quad
\dot\Phi=\frac{(1-e^2)^{3/2}}{M_{\rm sec}p^{3/2}}.
\]

For each `n`, let `J_k=J_k(ne)`, `A_h=mu_sec(1-e²)/p`, and `u=cos(iota)`. The [Barack–Cutler quadrupole convention](https://arxiv.org/pdf/gr-qc/0310125) is

\[
a_n=-n A_h\left[J_{n-2}-2eJ_{n-1}+\frac{2}{n}J_n+2eJ_{n+1}-J_{n+2}\right]\cos(n\Phi),
\]
\[
b_n=-n A_h\sqrt{1-e^2}\left[J_{n-2}-2J_n+J_{n+2}\right]\sin(n\Phi),\qquad
c_n=2A_h J_n\cos(n\Phi),
\]
\[
h_+=\frac{\sum_n[-(1+u^2)(a_n\cos2\gamma-b_n\sin2\gamma)+(1-u^2)c_n]}{D_{\rm sec}},\quad
h_\times=\frac{\sum_n 2u(b_n\cos2\gamma+a_n\sin2\gamma)}{D_{\rm sec}}.
\]

Here `D_sec=D_Mpc × Mpc_SI/c`. The circular limit has only `n=2`: `h_+=2A_h(1+u²)cos[2(Φ+γ)]/D_sec`, `h_cross=-4A_h u sin[2(Φ+γ)]/D_sec`. This is the [Peters–Mathews (1963)](https://doi.org/10.1103/PhysRev.131.435) leading quadrupole sector. The finite `N` sum truncates eccentric power; it is not an all-harmonic model. The native Bessel implementation evaluates the convergent integer-order series through `k=20` (21 terms) for `ne<=4.8`; the reference uses mpmath Bessel functions.

| Component ID/version | Paper source and option | Independent specification | Native Torch/JAX | Evidence status |
| --- | --- | --- | --- | --- |
| `emri.flux.peters_leading.v1` | Peters 1964, Eqs. (5.6)–(5.7), exact `a` to `p` change | `diffgw_reference.emri_peters_flux` | `peters_mathews_radiation_reaction` | Exact symbolic conversion; sampled float64 value/derivative checks |
| `emri.orbit.peters_kepler_rk4.v1` | Peters rates and Barack–Cutler Eq. (27) Newtonian mean anomaly; fixed-grid RK4 | `diffgw_reference.emri_newtonian_quadrupole.trajectory` | `NewtonianEccentricEMRI.trajectory` | Sampled waveform and grid stability checks; no adaptive/global error bound |
| `emri.harmonics.peters_mathews.v1` | Barack–Cutler Eqs. (7)–(9), finite `N=2,...,8` | SymPy expressions and mpmath Bessel evaluation | Native integer-Bessel series | Symbolic circular limit; sampled scalar/native checks, including `N=8,e=0.6` |
| `emri.observer.fixed_polarizations.v1` | Barack–Cutler Eq. (10), fixed `u,γ,D` | SymPy expressions, mpmath evaluation | Native source-frame `polarizations` | Sampled value and parameter-gradient checks |
| `emri.ak_kerr.v0`, `emri.nk_kerr.v0`, `emri.lisa.v0` | Complete AK/NK/Kerr and TDI models | None | None in new class | Open; no such capability claim |

## Implementation mapping

The reference file `src/diffgw_reference/emri_newtonian_quadrupole.py` records `SPEC_VERSION=emri-newtonian-eccentric-quadrupole/v1` and the three new component IDs. It builds harmonic and observer expressions in SymPy, evaluates them with mpmath Bessel functions, and integrates paper-derived rates at high precision. It imports neither native frontend nor native constants; the existing symbolic flux reference supplies only Peters' rates. Both native frontends independently run RK4 on the requested grid and differentiate through native tensor operations. Torch uses autograd; JAX supports `grad`, batch broadcasting, `jit`, and `lax.scan`. This comparison checks implementation agreement with the bounded specification; a common fixed-grid RK4 method can share numerical truncation error.

For capability context, [FastEMRIWaveforms (FEW) v2.1.1 waveform documentation](https://fastemriwaveforms.readthedocs.io/en/v2.1.1/user/main.html) lists distinct `FastSchwarzschildEccentricFlux`, `FastKerrEccentricEquatorialFlux`, and `Pn5AAKWaveform` options. Its [trajectory documentation](https://fastemriwaveforms.readthedocs.io/en/v2.1.1/tutorial/Trajectory_tutorial.html) identifies Schwarzschild eccentric flux, Kerr eccentric equatorial flux, and PN5 spinning inclined trajectories. These include relativistic or higher-PN dynamics absent from this Newtonian fixed-observer option. None is an equation-identical oracle for this finite Peters–Mathews quadrupole model; no FEW output is used as a numerical comparison here. The fuller [Barack–Cutler AK](https://arxiv.org/pdf/gr-qc/0310125) evolves precession and higher-order terms, while the [Babak et al. NK](https://arxiv.org/pdf/gr-qc/0607007) follows Kerr geodesics and multipole radiation. The provisional legacy `EMRIKludge` does not implement either model despite its name.

## Evidence and limitations

`tests/test_emri_newtonian_quadrupole_reference.py` checks the exact circular limit and absence of non-2 harmonics at `e=0`; both native strains against the scalar reference at `e=0,0.2,0.55`, several observer angles and times; independent five-point high-precision derivatives for `p0,e0,iota,gamma`; batch values and JAX `jit`; grid stability; and `N=8,e=0.6` near the bounded edge. Value tolerances are relative `3e-12` and absolute `2e-34`; derivative tolerance is relative `4e-10` and absolute `1e-34`. The tests do not establish physical accuracy against a relativistic EMRI, long-time phase accuracy, a rigorous RK4 error bound, uniform accuracy across the full parameter box, JIT runtime domain rejection, or a detector response. The full harmonic sum beyond `N=8` and generic inclination/precession remain open. See the dated EMRI evidence receipt in `docs/validation/` for the executed environment and results.

The existing `tests/test_emri_peters_flux_math_reference.py` also proves the exact `A` to `p` conversion and checks native `dot p,dot e` and derivatives at `(p,e)=(8,0),(8,0.3),(14,0.7)` and another `e` derivative at `(11,0.2)`, for `M=10^6`, `mu=10` solar masses on CPU float64. It checks only the unclamped flux interior. The wrong historical `604/48` eccentricity coefficient was corrected to `304/15` in that component; this new model inherits the checked expression.

## Reproduce the checks

With SymPy, mpmath, PyTorch, JAX (x64 enabled), NumPy, and pytest available, run `PYTHONPATH=src python -m pytest -q tests/test_emri_newtonian_quadrupole_reference.py tests/test_emri_peters_flux_math_reference.py`. The reference file is not called by either native implementation. Any change to the source equations, finite harmonic policy, variables, or numerical method requires a version review and renewed exact, scalar, and derivative checks. Promoting a generic EMRI, AK/NK, or LISA waveform requires separate paper-derived reference components and external or independently implemented waveform evidence.
