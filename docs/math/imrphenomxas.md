# IMRPhenomXAS intermediate amplitude

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

## Fifth-order paper option

Specification `imrphenomxas-intermediate-amplitude-fifth-order/v1` records
the paper's Eq. (72), Eqs. (73)–(78), and Table 1 as a **reference-only**
component. The paper uses this option for its reported results, but the
current Torch and JAX carriers implement the quartic option above. No
frontend or LAL parity is claimed for the fifth-order component.

For four supplied nodes `0<f1<f2<f3<f4`, define `t=(f-f1)/(f4-f1)` and
`P5(t)=c0+c1*t+c2*t²+c3*t³+c4*t⁴+c5*t⁵`. The printed equation is
`A(f)=K*f^(-7/6)/P5(t)`. Let `B=A/(K*f^(-7/6))=1/P5`.
Given the four physical amplitudes `A(fi)=ai` and endpoint slopes
`A'(f1)=d1`, `A'(f4)=d4`, the [scalar reference](../../src/diffgw_reference/imrphenomxas_fifth_order.py)
converts them to `B(fi)` and
`B'(fj)=(A'(fj)+7*A(fj)/(6*fj))/(K*fj^(-7/6))` for `j=1,4`.
It solves the resulting six linear constraints on `P5` exactly in SymPy.
The [maintained test](../../tests/test_imrphenomxas_math_reference.py)
recovers a known fifth-degree polynomial from all six physical constraints,
including both derivatives.

| Component ID | Evidence | Reference and frontend status | Maintained check |
| --- | --- | --- | --- |
| `phenomxas.amp.intermediate.fifth_order.v1` | Eqs. (67), (72) | `paper_amplitude` in the scalar reference; reference-only | `test_paper_fifth_order_and_physical_slope_constraints` |
| `phenomxas.amp.intermediate.fifth_order_collocation.v1` | Eqs. (73)–(78), Table 1 | `collocation_coefficients` and `physical_collocation_coefficients`; reference-only | `test_paper_fifth_order_and_physical_slope_constraints`, `test_fifth_order_reference_only_evidence` |

The caller supplies the four nodes and collocation amplitudes. The printed
Table 1 labels `f1` as `f1^W` and prints its two interior-frequency formulas
in terms of `f_T^A` and `f_peak`; this component does not reinterpret those
notations or import parameter-space fits. As with the quartic paper equation,
this is a literal literature specification, not evidence that the production
carrier uses the same frequency prefactor. Physical validation remains pending.

## Inspiral and merger-ringdown amplitude

Specification `imrphenomxas-inspiral-ringdown-amplitude/v1` independently
transcribes [Pratten et al., arXiv:2001.11412v1](https://arxiv.org/html/2001.11412v1)
into the [SymPy reference](../../src/diffgw_reference/imrphenomxas_amplitude.py).
Frequency is dimensionless `f=Mf`. Eq. (67) gives
`A0=sqrt(2 eta/(3 pi^(1/3))) f^(-7/6)`. Eqs. (68)–(69) give
`A_Ins/A0=sum_{i=0}^6 A_i(pi f)^(i/3) + sum_{i=1}^3 rho_i(pi f)^((6+i)/3)`.
The seven `A_i` are exact rational expressions in `eta`, ordered spins, and
`delta=(m1-m2)/(m1+m2)` from Appendix A.1, Eqs. (120)–(126). The `rho_i`
are calibrated inputs; this component does not infer their physical values
from the paper or native fit constants.
The physical domain used for the scalar comparisons has `f>0`,
`0<eta<=1/4`, and `|chi1|,|chi2|<=1`; the carriers order the masses so
`delta>=0`. Ringdown requires `f_damp>0`, `sigma>0`, and a positive join
frequency. The simple stationary-peak expression assumes `0<lambda<1`.

The printed Eq. (70) collocation nodes are `{1/2,3/4,1} f_MECO`.
The [Torch](../../src/torchwave/approximants/imrphenom_xas.py) and
[JAX](../../src/jaxwave/approximants/imrphenom_xas.py) carriers instead
pass `f_T^A=f_MECO+(f_ISCO-f_MECO)/4` (Eq. (52)) to the same three-node
helper. These endpoints are generally different. Tests check exact recovery
at the paper's nodes and helper algebra for a supplied endpoint. They do
not claim the carrier endpoint agrees with printed Eq. (70).

Eq. (84) prints `A_MR=a_R Gamma/((f-f_RD)^2+Gamma^2)
exp[-lambda(f-f_RD)/Gamma]`, where `Gamma=sigma f_damp`. The reference
also solves `a_R` from a supplied amplitude at one node. Differentiating
Eq. (84) gives the stationary point
`f_RD+Gamma(sqrt(1-lambda^2)-1)/lambda` when `0<lambda<1` and the result
is positive. Printed Eq. (59) instead contains `sqrt(1-lambda)`. The
reference keeps both formulas separate; the carriers use the stationary
formula with numerical guards. This does not establish which expression
was intended in the paper's calibration.

The carriers use Eq. (84) as a *scaled* ringdown shape, multiplying it by
`f^(-7/6)` in full waveform assembly. That differs from printed Eq. (84)
as a physical amplitude. The [maintained tests](../../tests/test_imrphenomxas_amplitude_math_reference.py)
check the PN coefficients and `d/d(Mf)` against both native helpers, the
collocation solve, the ringdown stationary point and one-node scale, and
full carrier *shape* for inspiral and ringdown at four binary/spin points.
The full-carrier checks use native fitted parameters as supplied inputs;
they establish equation implementation, not independent calibration or
physical validation against hybrid/NR waveforms. Phase, tapering, cutoff,
and full waveform composition remain outside this component.

| Component ID | Published evidence | Native relationship | Maintained check |
| --- | --- | --- | --- |
| `phenomxas.amp.inspiral.pn.v1` | Eqs. (67), (69), (119)–(126) | `IMRPhenomXAS.pn_amp_func` in Torch/JAX | `test_frontend_pn_shape_and_derivative` |
| `phenomxas.amp.inspiral.pseudo_pn.v1` | Eq. (68) | Full inspiral carrier assembly | `test_full_frontend_inspiral_and_ringdown_shapes` |
| `phenomxas.amp.inspiral.collocation.v1` | Eqs. (52), (68), (70) | `solve_inspiral_residual_collocation`; carrier endpoint differs from Eq. (70) | `test_collocation_endpoint_distinction_and_exact_recovery`, `test_frontend_collocation_solve` |
| `phenomxas.amp.ringdown.lorentzian.v1` | Eq. (84) | Ringdown expression in both carriers, then extra `f^(-7/6)` | `test_ringdown_stationary_peak_and_scale`, `test_full_frontend_inspiral_and_ringdown_shapes` |
| `phenomxas.amp.ringdown.peak_printed.v1` | Eq. (59) | Paper-literal reference only | `test_ringdown_stationary_peak_and_scale` |
| `phenomxas.amp.ringdown.peak_stationary.v1` | Derived from Eq. (84) | Carrier join frequencies within the guard-free domain | `test_ringdown_stationary_peak_and_scale` |

## Intermediate and merger-ringdown phase

Specification `imrphenomxas-intermediate-ringdown-phase/v1` transcribes
[Pratten et al., arXiv:2001.11412v1](https://arxiv.org/html/2001.11412v1),
Eqs. (90) and (92)–(97), into an independent
[SymPy reference](../../src/diffgw_reference/imrphenomxas_phase_branches.py).
Its frequency `f` means dimensionless `Mf`, with `f>0`, `eta>0`, and
`f_damp>0`. All collocation values, ringdown frequencies, and fitted
coefficients are supplied inputs. The reference gives both phase
derivatives and their explicit antiderivatives; additive and affine phase
gauges are kept separate.

Eq. (96) has a constant, powers `f^(-1/3)`, `f^(-2)`, and `f^(-4)`, and a
Lorentzian proportional to `c0 a_phi`. The five nodes of Eq. (97) begin at
`f_T^phi`, put the fourth node at `f_RD`, and end at
`f_RD+1.25 f_damp`. Eq. (90) has five inverse-power coefficients and a
broadened Lorentzian with width `2 f_damp`; its five nodes follow Eq. (92).
Independent linear solves recover coefficients from supplied values of
`eta phi'(f)` at those nodes.
Eqs. (93)–(95) define the intermediate fitted-value offsets and the
weighted second value used by the native carriers. The prose after Eq. (90)
lists `b5` among “five” coefficients; the displayed equation contains only
`b0` through `b4`, which is the basis specified here.

There is a convention difference: **printed Eq. (90) has a minus sign**
before the intermediate Lorentzian, whereas the
[Torch](../../src/torchwave/approximants/imrphenom_xas.py) and
[JAX](../../src/jaxwave/approximants/imrphenom_xas.py) carriers use a plus
sign. The reference retains the printed expression and a separately named
carrier-sign branch. In the native code the factor `a_phi` is absorbed into
the fitted `c0` supplied to both regions. The paper excerpt does not resolve
the sign discrepancy; matching the carrier branch does not establish that
it matches the printed equation.

The [maintained tests](../../tests/test_imrphenomxas_phase_branches_math_reference.py)
check both primitives by symbolic differentiation, the sign difference,
five-node collocation recovery, all five carrier ringdown nodes and the four
intermediate nodes beyond the inspiral boundary, phase values for four
binary/spin points against both frontends, and the value and derivative at
their join. They remove the carrier's overall time and phase gauge before
comparison. Inspiral TaylorF2 terms, the inspiral pseudo-PN fit, the
calibrated parameter-space fits, and independent physical validation are
outside this component.

For the native intermediate solve, the fifth fitted derivative is evaluated
at `f_T^phi`, although the fifth collocation node is
`f_T^phi + delta_R/2`. The maintained check preserves and exposes that
choice; it does not derive it from Eq. (92). The fitted collocation values
remain supplied inputs from the existing author-code-derived fit functions.

| Component ID | Published evidence | Native relationship | Maintained check |
| --- | --- | --- | --- |
| `phenomxas.phase.ringdown.ansatz.v1` | Eq. (96) | Both carrier ringdown phase branches | `test_printed_ansatze_primitives_and_sign_distinction`, `test_full_frontend_phase_branches_and_join` |
| `phenomxas.phase.ringdown.collocation.v1` | Eqs. (96)–(97) | Both carriers' five-node solve | `test_collocation_nodes_and_exact_coefficient_recovery`, `test_native_collocation_values` |
| `phenomxas.phase.intermediate.printed.v1` | Eq. (90) | Paper-literal minus-sign reference | `test_printed_ansatze_primitives_and_sign_distinction` |
| `phenomxas.phase.intermediate.collocation.v1` | Eqs. (90), (92)–(95) | Both carriers' five-node solve with native sign | `test_collocation_nodes_and_exact_coefficient_recovery`, `test_native_collocation_values` |
| `phenomxas.phase.intermediate.carrier_sign.v1` | Eq. (90), sign distinction recorded above | Both carrier intermediate phase branches | `test_printed_ansatze_primitives_and_sign_distinction`, `test_full_frontend_phase_branches_and_join` |
| `phenomxas.phase.intermediate_ringdown.join.v1` | Antiderivatives of Eqs. (90), (96) | Both carriers' affine ringdown adjustment | `test_full_frontend_phase_branches_and_join` |
