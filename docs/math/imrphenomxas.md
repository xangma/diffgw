# IMRPhenomXAS mathematical specification

<!-- BEGIN GENERATED WAVEFORM DEFINITION -->

| Definition field | Catalog value |
| --- | --- |
| Model ID | `IMRPhenomXAS` |
| Reference composition | `imrphenomxas-composed-polarizations-carrier/v1` |
| Scope | Aligned-spin (2,2) carrier composition with supplied fit and QNM outputs. |
| Domain and modes | `frequency`; `(2, 2)` |
| Native backends | `torch`, `jax` |
| Scoped verification claims | 0 |

| Option | Default | Native choices | Reference-only choices |
| --- | --- | --- | --- |
| `intermediate_amplitude` | `carrier_quartic` | `carrier_quartic` | `paper_quartic`, `paper_fifth_order` |
| `intermediate_phase` | `carrier_sign` | `carrier_sign` | `printed_sign` |
| `branch_joins` | `lower_branch_at_join` | `lower_branch_at_join` | None |

<!-- END GENERATED WAVEFORM DEFINITION -->

## Scope and status

This page covers the aligned-spin dominant-mode amplitude, phase, and complex
polarizations. Generate `IMRPhenomXAS` with either backend through the
[frequency-domain API](../guides/waveform-generation.md). Higher-mode and
precessing X-family models have different scope. The reference distinguishes
paper-literal formulas, current Torch/JAX carrier conventions, and selected
checks using a separately supplied author fit asset.

## Inputs, conventions, and options

The public carrier uses component masses in solar masses, aligned spins,
distance in Mpc, inclination and orbital phase in radians, time shift in
seconds, and an Hz frequency grid. Intrinsic formulas use the dimensionless
frequency `u=M_seconds*f_Hz`. Calibrated fit outputs, QNM frequencies, and
join locations are supplied to the independent scalar composition. The
generated table above identifies the native quartic/sign choices and the
paper alternatives that remain reference only. SI scaling, phase origin,
polarization signs, and the `f<=0` mask are specified in
[composed complex polarizations](#composed-complex-polarizations).
For direct Torch/JAX model calls, `f_ref` defaults to absent. Both frontends
accept `window_type`, `phasing_convention`, and `reference_frame` for API
compatibility but do not use them to choose a window, phase prescription, or
frame. Once an explicit frequency grid is supplied, direct calls also ignore
`f_lower`, `f_final`, and `delta_f`; the public waveform wrapper can use
frequency limits when constructing that grid.

## Mathematics and components

### Quartic intermediate amplitude

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
`P_t(0)=s1`, and `P_t(1)=s3`. The [SymPy kernel](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/imrphenomxas_intermediate.py)
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
| `phenomxas.amp.intermediate.newtonian_factor.v1` | Eq. (67) | Waveform amplitude prefactor; full complex assembly is checked below | `test_paper_quartic_and_physical_slope_constraints` |
| `phenomxas.amp.intermediate.quartic.v1` | Eq. (71) | `solve_intermediate_amplitude_quartic` in [Torch](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/imrphenom_xas.py) and [JAX](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/imrphenom_xas.py), algebraic helper only | `test_frontend_quartic_against_scalar_matrix_solve` |
| `phenomxas.amp.intermediate.collocation.v1` | Eqs. (68)–(69), (79)–(84), Table 2 | Full carrier boundary values and slopes from the independent inspiral/ringdown shapes, then the quartic solve; fitted parameters are supplied inputs | `test_full_frontend_intermediate_boundary_construction`, `test_midpoint_coefficient_derivative` |
| `phenomxas.amp.intermediate.constant_normalization.v1` | Algebraic variant; pointwise LALSimulation binary check below, **not** the printed Eq. (71) | Full Torch/JAX carrier assembly | `test_observed_quartic_and_normalization_distinction`, `test_full_frontend_intermediate_amplitude_shape`, `test_external_binary_intermediate_amplitude_fixture` |

The [maintained tests](https://github.com/xangma/diffgw/blob/main/tests/test_imrphenomxas_math_reference.py) check
the paper equation and five physical constraints with a known quartic, then
compare each native *helper* against the independent scalar matrix solve and
its midpoint derivative. Separate tests evaluate each full Torch/JAX carrier
at interior frequencies against the observed scalar convention and three
pinned binary amplitude samples. These check assembly and sampled calibrated
behavior, not whether the fits follow the printed paper equation.

The complete carrier boundary adapter starts from the Eq. (68)–(69)
inspiral shape `S_ins` and Eq. (84) ringdown shape `S_RD`, with native fitted
parameters supplied. At `f1=f_T^A` and `f3=f_peak`, its constant-normalized
physical endpoint values are `A1/K=f1^(-7/6) S_ins(f1)` and
`A3/K=f3^(-7/6) S_RD(f3)`; their slopes are the derivatives of those full
expressions. The independently supplied midpoint fit is `A2/K=v_A`.
These five conditions determine `P` through the exact scalar solve. The
maintained test checks the resulting coefficients against each carrier at
four mass/spin points and checks all five reconstructed constraints. This
closes the boundary-value algebra, conditional on the supplied calibration;
it does not turn the carrier's `K/P` convention into the printed Eq. (71).

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

### Fifth-order paper option

Specification `imrphenomxas-intermediate-amplitude-fifth-order/v1` records
the paper's Eq. (72), Eqs. (73)–(78), and Table 1 as a **reference-only**
component. The paper uses this option for its reported results, but the
current Torch and JAX carriers implement the quartic option above. No
frontend or LAL parity is claimed for the fifth-order component.

For four supplied nodes `0<f1<f2<f3<f4`, define `t=(f-f1)/(f4-f1)` and
`P5(t)=c0+c1*t+c2*t²+c3*t³+c4*t⁴+c5*t⁵`. The printed equation is
`A(f)=K*f^(-7/6)/P5(t)`. Let `B=A/(K*f^(-7/6))=1/P5`.
Given the four physical amplitudes `A(fi)=ai` and endpoint slopes
`A'(f1)=d1`, `A'(f4)=d4`, the [scalar reference](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/imrphenomxas_fifth_order.py)
converts them to `B(fi)` and
`B'(fj)=(A'(fj)+7*A(fj)/(6*fj))/(K*fj^(-7/6))` for `j=1,4`.
It solves the resulting six linear constraints on `P5` exactly in SymPy.
The [maintained test](https://github.com/xangma/diffgw/blob/main/tests/test_imrphenomxas_math_reference.py)
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

### Inspiral and merger-ringdown amplitude

Specification `imrphenomxas-inspiral-ringdown-amplitude/v1` independently
transcribes [Pratten et al., arXiv:2001.11412v1](https://arxiv.org/html/2001.11412v1)
into the [SymPy reference](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/imrphenomxas_amplitude.py).
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
The [Torch](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/imrphenom_xas.py) and
[JAX](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/imrphenom_xas.py) carriers instead
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
as a physical amplitude. The [maintained tests](https://github.com/xangma/diffgw/blob/main/tests/test_imrphenomxas_amplitude_math_reference.py)
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

### Inspiral phase

Specification `imrphenomxas-inspiral-phase/v1` records
[Pratten et al., arXiv:2001.11412v1](https://arxiv.org/html/2001.11412v1),
Eqs. (85)–(88) and Appendix A.2 Eqs. (129)–(138), in an independent
[SymPy reference](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/imrphenomxas_inspiral_phase.py).
Its `f` is dimensionless `Mf`, with `f>0`, `eta>0`, aligned component spins,
`chi_s=(chi1+chi2)/2`, and `chi_a=(chi1-chi2)/2`.
The intrinsic phase uses `v=(pi f)^(1/3)` and
`3/(128 eta v^5) sum_k phi_k v^k-pi/4`, followed by the Eq. (85)
four-coefficient default
`[sigma0+sigma1 f+(3/4)sigma2 f^(4/3)+(3/5)sigma3 f^(5/3)
+(1/2)sigma4 f^2]/eta`. Its derivative is checked symbolically.
Coalescence time and phase are separate affine gauges. All fitted values are
supplied inputs; this reference does not independently reproduce the
parameter-space calibration.

Eqs. (86) and (129) display a sum through `k=7`, while Appendix A.2 also
prints the 4PN spin-orbit tail `phi8` in Eq. (138). Both native canonical
carriers include `phi8`; the reference exposes both the printed truncated sum
and the carrier's Appendix-extended sum. The existing aligned-spin TaylorF2
reference includes an additional 3.5PN spin-spin tail from Appendix A.3 that
is **absent** from this XAS canonical Eq. (137), so it is not reused as an
unchanged base here.

The four collocation nodes printed in Eq. (88) omit the lower-frequency offset
at the two middle nodes:
`{fL, delta/4, 3 delta/4, fH}`, where `fL=0.0026`,
`fH=1.02 f_MECO`, and `delta=fH-fL`. Both carriers instead use
`{fL, fL+delta/4, fL+3 delta/4, fH}`. The reference names both
sets and solves the four-point basis `1, f^(1/3), f^(2/3), f` exactly.
The native values are `{v3+d13, v3+d23, v3, v3+d43}` and its coefficient
scale is `-5/(128 pi^(5/3))`. This scale and the supplied fitted values are
recorded as **carrier conventions**; they are not derived solely from the
published Eq. (87) residual definition. `sigma1` is restated by the derivative
join to the intermediate phase and `sigma0` enforces its value join.

The [maintained tests](https://github.com/xangma/diffgw/blob/main/tests/test_imrphenomxas_inspiral_phase_math_reference.py)
compare the Appendix phase and analytic derivative with both Torch and JAX at
zero, equal, and unequal spins, recover collocation coefficients, check fitted
carrier coefficients, and test the inspiral phase and C1 join at four binary
points. They remove the final carrier time and constant phase gauge before
comparison. This verifies implementation of the published intrinsic equations
and explicitly labeled carrier choices, not calibration or physical accuracy
against hybrid/NR waveforms. Full waveform composition remains a separate
component.

| Component ID | Published evidence | Native relationship | Maintained check |
| --- | --- | --- | --- |
| `phenomxas.phase.inspiral.pn.v1` | Eqs. (86), (129)–(137) | `pn_phase_canonical_*`, `d_pn_phase_canonical_*` in Torch/JAX | `test_native_pn_phase_and_analytic_slope` |
| `phenomxas.phase.inspiral.spin_tail_4pn.v1` | Eq. (138) | Both canonical PN helpers include this term | `test_printed_pn_tail_and_pseudo_derivative`, `test_native_pn_phase_and_analytic_slope` |
| `phenomxas.phase.inspiral.pseudo_pn.v1` | Eq. (85) | Both full inspiral carrier branches | `test_printed_pn_tail_and_pseudo_derivative`, `test_native_collocation_and_inspiral_join` |
| `phenomxas.phase.inspiral.collocation.v1` | Eqs. (87)–(88) | Both carrier four-point solves with the shifted nodes and supplied fits | `test_printed_and_carrier_node_distinction_and_exact_solve`, `test_native_collocation_and_inspiral_join` |
| `phenomxas.phase.inspiral.join.v1` | Eq. (85), Sec. VII.1 continuity statement | Both carriers' `sigma0`/`sigma1` and full inspiral branch | `test_native_collocation_and_inspiral_join` |

### Intermediate and merger-ringdown phase

Specification `imrphenomxas-intermediate-ringdown-phase/v1` transcribes
[Pratten et al., arXiv:2001.11412v1](https://arxiv.org/html/2001.11412v1),
Eqs. (90) and (92)–(97), into an independent
[SymPy reference](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/imrphenomxas_phase_branches.py).
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
[Torch](https://github.com/xangma/diffgw/blob/main/src/torchwave/approximants/imrphenom_xas.py) and
[JAX](https://github.com/xangma/diffgw/blob/main/src/jaxwave/approximants/imrphenom_xas.py) carriers use a plus
sign. The reference retains the printed expression and a separately named
carrier-sign branch. In the native code the factor `a_phi` is absorbed into
the fitted `c0` supplied to both regions. The paper excerpt does not resolve
the sign discrepancy; matching the carrier branch does not establish that
it matches the printed equation.

The [maintained tests](https://github.com/xangma/diffgw/blob/main/tests/test_imrphenomxas_phase_branches_math_reference.py)
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

### Composed complex polarizations

Specification `imrphenomxas-composed-polarizations-carrier/v1` combines the
independently recorded amplitude and phase branches in the
[scalar reference](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/imrphenomxas_waveform.py).
It accepts the carrier's calibrated fit outputs, QNM frequencies, and join
locations as **supplied inputs**. No fitted parameter-space rows or native
waveform implementation are imported into the reference. The preceding
sections distinguish paper-literal from carrier conventions; this composition
selects the **carrier** intermediate-amplitude normalization `K/P`, the
carrier's plus sign for the intermediate phase Lorentzian, and the carrier's
phase-fit normalization. Therefore it is a verification specification for the
present Torch/JAX waveform, not a claim that those disputed choices follow
from the printed paper.

[Pratten et al., Sec. II.2, Eqs. (6), (14)–(16)](https://arxiv.org/html/2001.11412v1)
define the positive-frequency mode `h_2,-2=A exp(-i phi)` and the angular
factors `(1+cos² iota)/2` and `-i cos iota` for plus and cross. Eqs. (66)–(67)
give the dimensionless Newtonian mode amplitude. Restoring Fourier strain
units with `M_sec²/D_sec` and the harmonic factor gives the common physical
scale
`sqrt(5 eta/24) M_sec^(5/6) f_Hz^(-7/6)/(pi^(2/3) D_sec)`.
This restoration is a dimensional derivation, **not** an equation printed in
the paper. `M_sec` and `D_sec` use the independently audited exact SI/IAU
definitions in [the units specification](units.md); `u=M_sec f_Hz` selects
the dimensionless intrinsic branches.

For `f_Hz>0`, the carrier assembles
`Psi=phi_raw(u)-phi_raw(u_ref)+tshift_sec*(f_Hz-f_ref)
+2 pi f_Hz t_c-2 phi_c+3 pi/4` when `f_ref` is supplied. Without `f_ref`,
the two reference terms are absent. The assembled polarizations are
`h_+=((1+cos² iota)/2) A_phys exp(-i Psi)` and
`h_x=-i cos(iota) A_phys exp(-i Psi)`. Both are zero for `f_Hz<=0`.
The exact `f_ref` subtraction, `tshift_sec`, and fixed `3 pi/4` are carrier
gauges. The [XHM companion paper, Appendix C](https://arxiv.org/html/2001.10914v1)
documents related reference-frequency, orbital-phase, and waveform-time
conventions; its mode-phase relation is distinct from the assembled
polarization phase. The arbitrary upper reconstruction cutoff discussed in
XAS Sec. V (Eqs. (64)–(65), $Mf=0.3$) is not an intrinsic cutoff in this carrier; public wrapper
`f_lower`/`f_final` limits are outside this component.

The [maintained test](https://github.com/xangma/diffgw/blob/main/tests/test_imrphenomxas_waveform_math_reference.py)
compares absolute complex plus/cross strain and unwrapped phase with both
FP64 frontends at equal-mass zero-spin and unequal aligned-spin points. It
samples both sides of every amplitude and phase transition, the ringdown,
nonpositive frequencies, and `f_ref` absent or placed in each phase branch,
with nonzero
distance, inclination, time, and orbital phase. It also checks the exact
polarization algebra, and Torch/JAX automatic derivatives against the scalar
reference for four extrinsic parameters and smooth-branch frequency at
inspiral, intermediate, and ringdown points. It does not validate
parameter-space calibration, physical agreement with NR/hybrid waveforms,
the fifth-order paper amplitude option, the public wrapper's user masks, or
full-domain intrinsic derivatives. Selected mass-ratio and spin tangents are
checked below.

The [maintained pinned-binary whole-polarization gate](../validation/imrphenomxas_whole_waveform_20260927.md)
compares Torch and JAX directly with the public LALSimulation 7.26.15
`IMRPhenomXAS` waveform-sequence API over every positive-frequency sample
below the paper's $Mf=0.3$ upper limit. It checks absolute complex $h_+$ and
$h_\times$, without phase/time alignment, over a catalog case and two frozen
unequal-mass spin holdouts. The public binary zeroes bins above that limit;
the native carrier continues to evaluate there, so the gate explicitly tests
the public cutoff and does not count its zero tail as intrinsic carrier error.

| Component ID | Evidence | Native relationship | Maintained check |
| --- | --- | --- | --- |
| `phenomxas.waveform.branch_selection.v1` | XAS Eqs. (52), (71), (85), (90), (96); carrier join choice | Both `XASCarrierState.eval_waveform` implementations | `test_full_complex_polarizations` |
| `phenomxas.waveform.physical_scale.v1` | XAS Eqs. (14)–(16), (66)–(67), dimensional restoration; [units](units.md) | SI frequency/mass/distance adapter | `test_full_complex_polarizations`, `test_polarization_and_extrinsic_algebra` |
| `phenomxas.waveform.phase_adapter.v1` | XAS mode convention Eq. (14); carrier gauge | Reference subtraction, time/phase offset in both carriers | `test_full_complex_polarizations` |
| `phenomxas.waveform.polarizations.v1` | XAS Eqs. (15)–(16), azimuth fixed in carrier | Positive-frequency plus/cross signs | `test_full_complex_polarizations`, `test_polarization_and_extrinsic_algebra` |
| `phenomxas.waveform.nonpositive_mask.v1` | XAS positive-frequency mode Eq. (14); carrier mask | Both carriers output zero strain for `f<=0` | `test_full_complex_polarizations` |
| `phenomxas.waveform.extrinsic_derivatives.v1` | Product rule on the preceding equations | Closed partials at fixed intrinsic inputs | `test_polarization_and_extrinsic_algebra`, `test_native_extrinsic_gradients` |
| `phenomxas.waveform.frequency_derivatives.v1` | Product rule and chain rule on the preceding equations | Smooth-branch complex `d(h+,hx)/df_Hz` at fixed fits and `f_ref` | `test_native_complex_frequency_gradient` |

### Total-mass similarity and derivative

Specification `imrphenomxas-total-mass-scaling/v1` extends the unchanged
composed-polarization specification. [Pratten et al., Secs. II.1 and V–VII,
Eqs. (1), (14)–(16), (52)–(67)](https://arxiv.org/html/2001.11412v1)
parameterize intrinsic fits and transitions by mass ratio, dimensionless spins,
and `u=M_sec*f_Hz`. The SI amplitude factor above has mass dependence
`M_sec^(5/6)` at fixed physical frequency and distance. The carrier's
`tshift_sec=M_sec*tshift` is a separately identified phase-gauge choice.
At fixed mass ratio and spins, calibrated dimensionless fits, QNM coordinates,
and joins are unchanged when both component masses scale by `lambda>0`.

Consequently, away from user frequency limits and with distance, inclination,
and orbital phase fixed, the Fourier strain obeys the exact similarity
`h(lambda*m1,lambda*m2;f/lambda,f_ref/lambda,lambda*t_c)
=lambda²*h(m1,m2;f,f_ref,t_c)` for each polarization. Omit both `f_ref`
arguments when no reference frequency is supplied. This identity follows
from the dimensional adapter and is not a fitted relation.

For a derivative instead holding physical `f`, `f_ref`, `t_c`, and distance
fixed, set `a=a(u)` for the selected amplitude shape and `phi=phi(u)` for
the selected raw phase. With `S` the SI amplitude scale above and `C` either
polarization's inclination factor, the scalar reference computes

`d h/d ln M = C*S*exp(-i Psi) * [(5/6)*a + u*a' - i*a*Psi_M]`,

where `Psi_M=u*phi'(u)-u_ref*phi'(u_ref)
+tshift_sec*(f-f_ref)`. The reference terms are absent when `f_ref` is
absent. At `f=f_ref`, `Psi_M=0` exactly. A prime here means differentiation
with respect to dimensionless `u`. These formulas apply only to smooth
positive-frequency branch interiors; neither `f` nor `f_ref` may lie on a
moving join. The [maintained test](https://github.com/xangma/diffgw/blob/main/tests/test_imrphenomxas_waveform_math_reference.py)
checks the similarity for both frontends at equal and unequal masses, and
compares the closed full-carrier mass tangent against Torch/JAX automatic
derivatives through carrier construction at unequal mass, across amplitude
branches and all reference-frequency phase branches. This does not test
mass-ratio or spin derivatives of the calibrated fit surfaces, or establish
independent calibration or physical accuracy.

| Component ID | Evidence | Native relationship | Maintained check |
| --- | --- | --- | --- |
| `phenomxas.waveform.mass_scaling.v1` | XAS Eqs. (1), (14)–(16), (52)–(67); dimensional similarity and carrier time gauge | Both `XASCarrierState` constructors and `eval_waveform` | `test_total_mass_similarity` |
| `phenomxas.waveform.mass_derivatives.v1` | Chain rule on the composed model and mass similarity | Full complex `d(h+,hx)/d ln M` through both native constructors | `test_native_log_total_mass_gradient`, `test_mass_phase_tangent_cancels_at_reference` |

The same scalar composition also checks selected intrinsic tangents through
the entire carrier. At `q=5`, `chi1=0.5`, `chi2=-0.3`, fixed total mass and
physical frequency, the [intrinsic derivative test](https://github.com/xangma/diffgw/blob/main/tests/test_imrphenomxas_waveform_math_reference.py)
uses centered shifts in each of `q`, `chi1`, and `chi2` to rebuild the
scalar waveform from the carrier's named fit/collocation outputs. It compares
complex plus/cross and unwrapped-phase tangents with Torch and JAX automatic
derivatives in the inspiral, intermediate, and ringdown regions, with a fixed
reference frequency in the intermediate phase. The separate author-asset check
below independently verifies the underlying fit-surface derivatives. This
composition check supplies carrier state outputs at each shifted point; it is
therefore a check of waveform assembly and automatic differentiation, not an
independent derivation of every state-construction derivative or a full-domain
intrinsic accuracy bound.

| Component ID | Evidence | Native relationship | Maintained check |
| --- | --- | --- | --- |
| `phenomxas.waveform.intrinsic_composition_derivatives.v1` | Chain rule on the composed amplitude and phase; APS fit-surface derivatives checked separately below | Full complex and unwrapped-phase tangents through both native constructors | `test_intrinsic_composition_derivatives` |

## Implementation mapping

The component tables above map stable IDs and source locations to the
independent scalar functions, both native carriers, and maintained checks.
The full carrier selects the quartic `K/P` amplitude and carrier-sign phase;
the printed quartic normalization, fifth-order amplitude, and printed phase
sign have separate reference expressions. Torch and JAX use their own tensor
implementations. The [composition](#composed-complex-polarizations) names the
supplied calibration and QNM inputs; neither frontend evaluates through the
scalar reference.

## Evidence and limitations

The component tables and their linked tests distinguish source transcription,
sampled scalar-to-native comparison, and external binary agreement. They do
not establish independent calibration or physical accuracy. The following
asset check applies only when the pinned author file is supplied.

### Author fit-surface cross-check

The opt-in [external-fit verifier](https://github.com/xangma/diffgw/blob/main/scripts/verify_xas_author_fits.py)
reads the [APS Mathematica companion file](https://journals.aps.org/prd/supplemental/10.1103/PhysRevD.102.064001/IMRPhenomXAS_Parameter_Space_Fits_Public_Release_Candidate.m)
at run time, verifies its SHA-256
`94b9334c19f3a7909864891ea278ab9a411e4f33b7b7e17cc01d927056d9ac52`,
and evaluates the selected source versions independently of either frontend.
The file and its formula rows are excluded from this repository and package;
its redistribution terms remain unresolved. Run with
`python scripts/verify_xas_author_fits.py --asset /path/to/IMRPhenomXAS_Parameter_Space_Fits_Public_Release_Candidate.m`
in an environment with both FP64 frontends and the reference dependencies.

The verifier maps all 25 non-precessing parameter-fit methods exposed by
`XASParameterFits` to named source functions and explicit version selectors.
The current quartic carrier calls 23 of them; `fit_int_amp_v2` and
`fit_int_amp_v3` are available for the paper's fifth-order alternative but
are not used by the current carrier. The deposited `InsPhased53Fit` surface
is not used or implemented by the current inspiral phase. The verifier also
checks the two `(2,2)` QNM rational fits. The separate `fit_final_spin_nnlo`
method is a precession helper and is not called by this aligned-spin carrier.
A structural check fails if either frontend adds a fit method without updating
the map.
Values are compared at nine mass-ratio/spin points, parameter derivatives
`(d/d eta, d/d chi1, d/d chi2)` at two unequal-mass points, and QNM spin
derivatives at five points. The scaled error divides by `max(1, |source|)`;
the gates are `2e-10` for values and `2e-8` for derivatives. On the pinned
asset, 24 parameter surfaces and both QNM functions agree with Torch and JAX;
the largest derivative scaled error is `3.05e-12` for the QNM real fit.

The remaining midpoint amplitude fit, `IntAmpvAFit["104"]`, uses
`chiPNhat` in the deposited file, whereas both binary-compatible carriers
use `sTotR`. The source/carrier value difference reaches `0.042` on the
declared cases. Re-evaluating the *same deposited expression* with the
carrier's declared `sTotR` argument agrees in value and all three derivatives.
This is an identified model choice, not agreement with the published source
fit. The check verifies the deposited surfaces and their local derivatives;
it does not reproduce the original NR calibration or prove physical accuracy.

| Diagnostic evidence | Source | Native relationship | Maintained check |
| --- | --- | --- | --- |
| Parameter-fit surfaces | Pinned APS companion file, explicit function/version map | 24 matching surfaces and one declared midpoint spin-argument variant in both frontends | `verify_xas_author_fits.py` (opt-in external asset) |
| QNM fit surfaces | Pinned APS companion file, `(2,2)` real/imaginary functions | Both carriers' QNM fit functions | `verify_xas_author_fits.py` (opt-in external asset) |

These opt-in comparisons are diagnostic evidence for the supplied calibration
outputs. They are outside the executable scalar composition and do not add
calibration components or verification claims to the public catalog.

## Reproduce the checks

From the repository root with the optional reference, Torch, and JAX
dependencies installed, run
`JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest tests/test_imrphenomxas*math_reference.py --reference-device=cpu -ra`.
For qualification, run the [full strict CPU reference gate](../waveforms/qualification-status.md#reproduction).
The opt-in author-asset verifier command and required hash are given in the
[cross-check](#author-fit-surface-cross-check); the maintained CI gate runs
without that external file.
