# IMRPhenomXAS intermediate amplitude: quartic option

Specification `imrphenomxas-intermediate-amplitude-quartic/v1` covers one
component of the nonprecessing dominant-mode model. Its source is
[Pratten et al., arXiv:2001.11412v1, Sec. VI.B](https://arxiv.org/html/2001.11412v1#S6.SS2),
Eqs. (67), (71), (79)–(83), and Table 2. The paper generally uses the
**fifth-order** alternative, Eq. (72), for its reported default results.

Let `f` denote dimensionless `Mf` and `f1 < f3` the intermediate-region
boundaries. Define `t=(f-f1)/(f3-f1)` and

`P(t)=c0+c1*t+c2*t²+c3*t³+c4*t⁴`.

Eq. (71), with Eq. (67), then reads `A(f)=A0(f)/P(t)` where
`A0(f)=K*f^(-7/6)` and `K` is independent of frequency. `t` is only a change
of polynomial basis from the paper's powers of `f`; it does not change the
quartic family. Define `B(f)=A(f)/A0(f)=1/P(t)`. The Table-2 conditions are

`B(f1)=v1`, `B((f1+f3)/2)=v2`, `B(f3)=v3`,
`B'(f1)=d1`, and `B'(f3)=d3`.

For `y_i=1/v_i`, `h=f3-f1`, and `s_i=-h*d_i/v_i²`, the five linear
constraints on `P` are `P(0)=y1`, `P(1/2)=y2`, `P(1)=y3`,
`P_t(0)=s1`, and `P_t(1)=s3`. The [SymPy kernel](../../src/diffgw_reference/imrphenomxas_intermediate.py)
solves this matrix system with exact scalar inputs. For physical amplitude
conditions from Eqs. (79)–(83), it first applies
`B=A/A0` and `B'=(A'+7*A/(6*f))/A0`. The domain is `0<f1<f3`,
positive collocation amplitudes, and positive `K`; no clamp or extrapolation
is specified by the paper. Derivatives use `Mf`, so slopes in hertz require a
total-mass conversion.

The paper's Eq. (81) labels the right endpoint `f4`, while Eq. (83) and
Table 2 use `f3`. This component follows Table 2's three-node quartic
definition. It does not use the APS parameter-fit supplement or any LAL
source or constants.

| Component ID | Evidence | Torch/JAX relationship | Maintained check |
| --- | --- | --- | --- |
| `phenomxas.amp.intermediate.newtonian_factor.v1` | Eq. (67) | Waveform amplitude prefactor; full assembly is not yet qualified | `test_paper_quartic_and_physical_slope_constraints` |
| `phenomxas.amp.intermediate.quartic.v1` | Eq. (71) | `solve_intermediate_amplitude_quartic` in [Torch](../../src/torchwave/approximants/imrphenom_xas.py) and [JAX](../../src/jaxwave/approximants/imrphenom_xas.py), algebraic helper only | `test_frontend_quartic_against_scalar_matrix_solve` |
| `phenomxas.amp.intermediate.collocation.v1` | Eqs. (79)–(83), Table 2 | The same two helpers; full boundary-value adapter is not yet qualified | `test_midpoint_coefficient_derivative`, `test_component_evidence_and_implementation_links` |
| `phenomxas.amp.intermediate.constant_normalization.v1` | Algebraic variant; pointwise LALSimulation binary check below, **not** the printed Eq. (71) | Full Torch/JAX carrier assembly | `test_observed_quartic_and_normalization_distinction`, `test_full_frontend_intermediate_amplitude_shape`, `test_external_binary_intermediate_amplitude_fixture` |

The [maintained tests](../../tests/test_imrphenomxas_math_reference.py) check
the paper equation and five physical constraints with a known quartic, then
compare each native *helper* against the independent scalar matrix solve and
its midpoint derivative. Separate tests evaluate each full Torch/JAX carrier
at interior frequencies against the observed scalar convention and three
pinned binary amplitude samples. These check assembly and sampled calibrated
behavior, not whether the fits follow the printed paper equation.

**Normalization investigation.** The waveform assembly supplies the quartic
helper with endpoint values of `scaled_amp/f^(7/6)` and reconstructs
`scaled_amp_int=f^(7/6)/P`. Multiplication by the physical Newtonian
prefactor gives `A_int=K/P`, where `K` is independent of frequency. Its
physical boundary slope is `A'/K`; its midpoint fit supplies `A/K`.
The printed Eqs. (67) and (71) instead give `A_int=K*f^(-7/6)/P`,
whose boundary conversion is `B'=(A'+7*A/(6*f))/A0`. These two quartic
families are not interchangeable by changing constant coefficients: equal
`P` would yield amplitudes in the ratio `f^(-7/6)`.

The implementation convention has direct external evidence. With the pinned
LALSuite 7.21 binary (LALSimulation 5.3.1), CPU FP64, default XAS settings,
`m1=m2=30 M_sun`, zero spins, `f_ref=f_low=20 Hz`, `f_high=512 Hz`,
`distance=500 Mpc`, face-on, and a `0.125 Hz` grid, Torch and LAL agree
across the 1,373 bins between
`Mf=0.02731805` and `0.07802608`: magnitude relative L2 is `2.13e-14`
and maximum pointwise relative error is `3.24e-14`. At the midpoint
`Mf≈0.05267`, the fitted collocation value is `A/K≈29.14`; multiplying it
by `f^(7/6)` gives the waveform's Newtonian-normalized shape `A/A0≈0.94`.
Applying the printed prefactor to the **same** polynomial would multiply
the midpoint amplitude by about 31. This comparison identifies the
implementation's normalization; it does not establish that the paper has a
typographical error or validate the waveform against hybrid/NR data.
Three raw binary amplitudes at grid indices 812, 1426, and 1760 are retained
as an external-output fixture in the maintained test, separate from the
paper-derived scalar reference.

The executable reference therefore retains both functions with separate
names and evidence. The printed equation remains a literature reference;
the `K/P` function is a binary-checked implementation reference. Whether
the authors intended a different normalization for Eq. (71), or used a
distinct convention during calibration, remains unresolved without further
author clarification or independent calibration evidence. Neither the
published parameter-space fits nor any LAL source or constants have been
copied into this component.
