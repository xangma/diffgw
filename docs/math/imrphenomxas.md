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

The [maintained tests](../../tests/test_imrphenomxas_math_reference.py) check
the paper equation and five physical constraints with a known quartic, then
compare each native *helper* against the independent scalar matrix solve and
its midpoint derivative. These tests establish the helper's conditional
algebra, not agreement of the complete XAS amplitude with Eq. (71).

**Open normalization discrepancy.** The current waveform assembly supplies
the helpers with `scaled_amp/f^(7/6)` at its endpoints and reconstructs
`scaled_amp_int=f^(7/6)/P`. Multiplication by the physical Newtonian
prefactor, which contains `f^(-7/6)`, therefore gives an intermediate
amplitude proportional to `1/P`. The printed Eq. (71) gives an amplitude
proportional to `f^(-7/6)/P`. This is a different functional family. The
project's parameter fits may absorb some of this difference, but helper
agreement alone cannot certify literature fidelity. The next XAS step must
resolve the normalization against the paper's definitions and independent
waveform evidence before qualifying complete amplitude or waveform claims.
