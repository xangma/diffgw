# EMRIKludge: provisional component qualification

This research specification is outside the public waveform catalog. The class is a short-time harmonic prototype. Only its leading secular `p,e` flux has an independent source-derived reference; no whole-waveform EMRI claim follows.

## Scope and status

The Torch and JAX classes produce time-domain `h_plus,h_cross` from a finite sum of hand-set eccentric sidebands, optionally multiplied by static LISA antenna patterns labeled `A,E,T`. There is no generic Kerr-geodesic orbit, self-force inspiral, source-faithful analytic-kludge (AK) evolution, numerical-kludge (NK) geodesic trajectory, or demonstrated time-delay-interferometry (TDI) transfer. The leading orbit-averaged Peters flux component is **executable and checked** on CPU float64. Orbit, harmonics, and detector response are **red qualification gates**. No coefficient table, trained model, or external asset is used.

## Inputs, conventions, and options

Central mass `M` and secondary mass `mu` are solar masses; `p` is dimensionless semilatus rectum in units of `M`, `e` is in `[0,1)`, dimensionless spin and angles use radians, input times use seconds, and distance uses Mpc. `n_harmonics=5` and the sum uses positive `n=1,...,N`. The scalar reference requires `M,mu,p>0` and `0<=e<1`; native flux clamps `p>=1`, `0<=e<=0.9999` and `1-e²>=1e-8`, so the reference checks only the unclamped interior. The reference has no interpolation or external data. The classes use initial fluxes in `p(t)=p0+dot(p0,e0)t`, `e(t)=e0+dot(e0,p0)t`, then clamp them; phase sums use a rectangular step based on the first time spacing. This is not an integrated inspiral ODE or an accuracy-controlled evolution. Phases, inclinations and sky frames are not yet source-qualified. Clamp boundaries are not differentiability claims.

## Mathematics and components

The primary source for the checked component is [Peters 1964, Eqs. (5.6)-(5.7)](https://doi.org/10.1103/PhysRev.136.B1224). Let `A=a_orbit/M` be the dimensionless semimajor axis, `p=A(1-e²)`, and `q=mu/M²`, with both masses expressed as geometric seconds. Peters gives

\[
\dot A=-\frac{64}{5}q A^{-3}(1-e^2)^{-7/2}
 \left(1+\frac{73}{24}e^2+\frac{37}{96}e^4\right),\qquad
\dot e=-\frac{304}{15}q e A^{-4}(1-e^2)^{-5/2}
 \left(1+\frac{121}{304}e^2\right).
\]

Apply the chain rule **before** substituting `A=p/(1-e²)`: `dot p=(1-e²)dot A-2Ae dot e`. Exact simplification gives

\[
\dot p=-\frac{64}{5}q p^{-3}(1-e^2)^{3/2}
 \left(1+\frac78e^2\right),\qquad
\dot e=-\frac{304}{15}q e p^{-4}(1-e^2)^{3/2}
 \left(1+\frac{121}{304}e^2\right).
\]

The `304/15` coefficient survives the change to `p`; the previous carrier value `604/48` was incorrect. These Newtonian, orbit-averaged two-body expressions become leading EMRI fluxes in the extreme mass-ratio limit. They do not supply Kerr-geodesic frequencies or higher-order self-force evolution.

| Component ID/version | Source / variant | Status | Scalar reference | Torch / JAX | Gate |
| --- | --- | --- | --- | --- | --- |
| `emri.flux.peters_leading.v1` | Peters 1964 Eqs. (5.6)-(5.7), exact `A` to `p` conversion | Executable, scoped check | `diffgw_reference.emri_peters_flux` | `peters_mathews_radiation_reaction` | CPU float64 values and derivatives; unclamped `p,e` only |
| `emri.orbit.v0` | [Barack–Cutler 2004 Eqs. (27)-(31)](https://arxiv.org/pdf/gr-qc/0310125) AK versus [Babak et al. 2007](https://arxiv.org/pdf/gr-qc/0607007) NK | Pending | None | `kerr_geodesic_frequencies`, `orbital_trajectory` | Choose a model, its variables, generic frequencies, evolution and convergence |
| `emri.harmonics.v0` | Barack–Cutler 2004 Eqs. (7)-(10), Bessel coefficients | Diagnostic | None | `polarizations` | Replace or bound hand-set powers of `e`, projection and phase |
| `emri.lisa.v0` | TDI transfer convention not chosen | Pending | None | `project_to_lisa` | Independent time-delay response and frame check |

## Implementation mapping

The source-derived scalar reference constructs Peters' `dot A`, `dot e` and the chain-rule expression symbolically with exact rational coefficients, then evaluates values and derivatives with mpmath. It imports neither native backend nor native constants. The Torch and JAX carrier functions now use `304/15` for `dot e`. The reference is **not** called by either frontend. Their frozen initial-flux trajectory does not solve Barack–Cutler's five coupled AK evolution equations, and their `kerr_geodesic_frequencies` ignores `e`; the circular-equatorial-looking `1/(p^{3/2}+a)` expression cannot represent generic eccentric, inclined Kerr frequencies. The emitted `n phi+2 gamma` phase omits the computed nodal phase. The sideband amplitudes are powers of `e`, not the paper's Bessel coefficients. Static antenna multiplication is not a measured TDI delay operator. Babak et al.'s NK instead follows Kerr geodesics and constructs radiation from multipole moments; that algorithm is absent.

## Evidence and limitations

`tests/test_emri_peters_flux_math_reference.py` proves the exact `A` to `p` conversion and independently checks each frontend's `dot p`, `dot e` and `p` derivatives at `(p,e)=(8,0),(8,0.3),(14,0.7)`, plus an `e` derivative at `(11,0.2)`. The domain is `M=10^6` solar masses, `mu=10` solar masses, CPU float64, with relative thresholds `2e-13` for values and `3e-13` for derivatives (absolute `1e-28`). This checks a flux component and one mass scale, not an inspiral or waveform. It does not test behavior at native clamps. No same-model external comparison or physical EMRI validation has been made. The unresolved orbit, harmonic, and detector gates above explicitly block promotion into the public catalog.

## Reproduce the checks

With `sympy`, `mpmath`, `torch`, `jax`, and pytest installed, run `PYTHONPATH=src python -m pytest -q tests/test_emri_peters_flux_math_reference.py`; the ordinary EMRI and cross-backend tests can be run separately. The maintained `--require-math-reference` CI gate discovers this file in a full reference-suite invocation and rejects skips/xfails. A change to the source equation, variable convention, coefficient, or numerical policy requires a component version increment and renewed exact symbolic, scalar-to-native and derivative evidence. Whole-waveform promotion additionally requires independent AK or NK equations, harmonic and TDI references and their own tests.
