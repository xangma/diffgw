# IMRPhenomT TaylorT3 aligned-spin frequency

## Scope and status

`T-PN-T3-v1` specifies the 3.5PN TaylorT3 contribution to the positive
`(2,2)` angular frequency and its phase primitive in the native IMRPhenomT,
THM, TP, and TPHM carriers. It is a component specification, not a complete
IMRPhenomT waveform claim. The fitted pseudo-PN extension and its six-point
collocation solve are specified separately by `T-PN-FREQ-COLLOC-v0`. The
scalar component is executable, and its backend agreement is checked for the
domain in the [len receipt](../validation/phenomt_taylor_t3_20260927.md).

## Inputs, conventions, and options

The scalar inputs are `0 < eta <= 1/4`, `delta=(m1-m2)/(m1+m2) >= 0`, aligned
dimensionless spins `chi1,chi2`, and positive
`theta=[eta(t0-t)/(5M)]^(-1/8)` with `t<t0`. Here `m1,m2` inside the PN
coefficients are *fractions of total mass*, `(1+delta)/2` and
`(1-delta)/2`; the carrier accepts physical solar masses and constructs these
fractions. Angular frequency is in inverse total-mass time, so
`f22=omega22/(2*pi*M*MSUN_SEC)` Hz. Both frontends use float64 for this
qualification. The phase primitive has an arbitrary additive constant;
the carrier uses the convention specified below.

The `paper_literal` scalar choice transcribes original T/TP Appendix A Eq.
(A5g) exactly. The `author_carrier` choice adds the `235925 eta^3/1769472`
term omitted from that printed equation. This correction is explicitly noted
in the pinned MIT [phenomxpy `internals.py`, `set_pn_coefficients`](https://gitlab.com/imrphenom-dev/phenomxpy/-/blob/4121fafa0484f3db2a7352f4eea697faaaf29f65/phenomxpy/phenomt/internals.py);
it is also in the native Torch/JAX carriers. `author_carrier` is the native
choice; `paper_literal` is reference-only. No LAL source is needed for either.

## Mathematics and components

Original [Estellés et al., arXiv:2004.08302v2, Eqs. (6b), (A4),
(A5a)–(A5h)](https://arxiv.org/pdf/2004.08302v2) gives the orbital
TaylorT3 frequency. Updated [THM, arXiv:2012.11923v1, Eqs. (8)–(11)](https://arxiv.org/pdf/2012.11923v1)
uses those same coefficients in its `(2,2)` inspiral extension. With
`c_k=hat omega_k` and `c_0=1,c_1=0`, the mode frequency is twice the orbital
frequency:

`omega22^T3(theta) = theta^3/4 * [1+c2 theta^2+c3 theta^3+c4 theta^4+c5 theta^5+(c6_const+(107/280)log(theta))theta^6+c7 theta^7]`.

`c6_const` contains `(107/280)log(2)` per the printed `log(2 theta)`.
For the carrier option it also contains `235925 eta^3/1769472`. The scalar
reference retains the paper's compact mass-weighted spin terms, while the
native code uses expanded polynomials. The independent reference implements
all six coefficients as exact-rational mpmath expressions.

Since `dt/dtheta=40 M/(eta theta^9)`, the phase primitive satisfies
`d phi22/d(t/M)=omega22^T3`. One convenient primitive is

`phi22 = (10/eta)[-theta^-5/5-c2 theta^-3/3-c3 theta^-2/2-c4 theta^-1+c5 log(theta)+c6_const theta+(107/280)theta(log(theta)-1)+c7 theta^2/2]`.

The carrier subtracts `(10/eta)c5 log(5)/8` from this primitive. This is
only a phase-origin convention and does not alter its frequency derivative.
The printed Appendix A gives the frequency coefficients; the primitive above
is obtained by integrating those terms, not by copying a phase formula from
another implementation.

| Component | Source and variant | Status | Scalar | Torch | JAX | Check |
| --- | --- | --- | --- | --- | --- | --- |
| `T-PN-T3-v1` | T/TP A5a–h, THM 8–9; paper/carrier 3PN distinction | executable | `phenomt_taylor_t3.coefficients`, `frequency`, `phase` | `imrphenomt._pn_coefficients`, `_poly_pn_frequency`, `_pn_phase` | same names in JAX module | `test_phenomt_taylor_t3_math_reference.py` |

## Implementation mapping

The reference evaluates the paper's terms using `m1=(1+delta)/2` and
`m2=(1-delta)/2`, and isolates the `log(2 theta)` at 3PN. Torch/JAX expand
the same terms into tensor polynomials, put `log(2)` inside their constant 3PN
coefficient, and add `(107/280)log(theta)` at evaluation. Native `_pn_phase`
uses its carrier phase-origin convention. Neither frontend imports the scalar
reference. The six pseudo-PN coefficients are supplied by the separate
collocation component, and calibrated inputs remain a separate qualification
gate.

## Evidence and limitations

The maintained gate compares every PN coefficient, frequencies, and phase
differences against both native frontends for equal and unequal masses,
unequal/zero spins, and several inspiral `theta` values. It also checks the
scalar phase derivative and the precise paper-to-carrier 3PN difference.
Numerical results and package versions belong in the validation receipt.
This checks the 3.5PN kernel only; it does not certify calibrated `t0`, fitted
collocation frequencies, joins, full modes, LAL agreement, or physical
accuracy. The printed Eq. (A5g) omission is preserved visibly as a variant,
not silently corrected.

## Reproduce the checks

On `len`, use CPU float64 and run
`JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=src:tests python -m pytest -q tests/test_phenomt_taylor_t3_math_reference.py tests/test_phenomt_inspiral_collocation_math_reference.py --reference-device=cpu -ra`.
The repository-wide `--require-math-reference` flag additionally requires
collection of the entire math-reference suite, so use it for the full suite.
Keep the two source variants distinct if a later source revision changes the
coefficient. Do not change native defaults as part of reference coverage.
