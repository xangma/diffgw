# IMRPhenomXHM (3,2) transition and mixing reference

## Scope and status

This is a **paper-equation** reference for the `(3,2)` intermediate phase,
ringdown join, and spherical–spheroidal projection. It does not provide the
calibration fits, QNM frequencies, or complex mixing-coefficient tables needed
to reproduce a particular LALSuite release. Its component IDs are
`xhm.phase.32.intermediate-ringdown-join.v1` and
`xhm.mixing.32.spherical-spheroidal.v1`.

## Inputs, conventions, and options

Frequency is dimensionless `Mf`; a complex mode is written
`h = A exp(-i φ)`. The four intermediate nodes and phase-derivative targets,
join frequency, QNM ring/damping frequencies, and complex mixing coefficients
are caller inputs. The mixed mode must be nonzero where its phase derivative
is evaluated. No release-specific fit or mixing option is selected.

## Mathematics and components

The source is García-Quirós et al.,
[arXiv:2001.10914v2](https://arxiv.org/pdf/2001.10914v2), Eqs. (5.4),
(5.6), (6.6)–(6.8), and (A1)–(A3).

Equation (5.4) gives

\[
\phi'_{32,\mathrm{int}}(f)=
 a_\lambda\frac{f_\mathrm{damp}}{f_\mathrm{damp}^{2}+(f-f_\mathrm{ring})^{2}}
 +\sum_{k=0}^{4}a_k f^{-k}.
\]

The six coefficients solve a linear system: four prescribed derivative values
at intermediate collocation frequencies, then the **spherical mixed**
ringdown phase's first and second derivatives at the join frequency. A final
additive phase offset matches its value there. These conditions make the
phase, first derivative, and second derivative continuous at the join;
the intermediate coefficients are not themselves calibrated parameters.

The spheroidal `(3,2)` ringdown phase primitive and derivative are transcribed
from Eqs. (6.6)–(6.7), including the arctangent primitive of the Lorentzian.
The scalar reference checks the derivative analytically. The four ringdown
collocation values in Eq. (6.8) remain caller data.

Appendix A gives the two-mode basis map. Eliminating the spheroidal `(2,2)`
mode yields

\[
h_{32}=\frac{\alpha_{322}}{\alpha_{222}}h_{22}
 +\left(\alpha_{332}-\frac{\alpha_{322}\alpha_{232}}{\alpha_{222}}\right)h^S_{32}.
\]

| Component ID | Source | Scalar reference | Torch/JAX |
| --- | --- | --- | --- |
| `xhm.phase.32.intermediate-ringdown-join.v1` | Eqs. (5.4), (5.6), (6.6)–(6.8) | `intermediate_phase_coefficients`, `intermediate_phase` | `_intermediate_32_phase_coefficients` |
| `xhm.mixing.32.spherical-spheroidal.v1` | App. A, Eqs. (A1)–(A3) | `spherical_32_from_spheroidal`, `mixed_phase_derivatives` | `_spherical_32_from_spheroidal` |

## Implementation mapping

The independent scalar implementation is
[`imrphenomxhm_transition_mixing.py`](https://github.com/xangma/diffgw/blob/main/src/diffgw_reference/imrphenomxhm_transition_mixing.py).
It takes all four complex `α` values as inputs and evaluates the mixed phase
derivatives by logarithmic differentiation. Both production `imrphenom_xhm.py`
frontends route their `(3,2)` collocation solve through the named helper.
The real/imaginary Cartesian Torch mixing path takes a real `h32_s` amplitude.

## Evidence and limitations

[`test_xhm_transition_mixing_math_reference.py`](https://github.com/xangma/diffgw/blob/main/tests/test_xhm_transition_mixing_math_reference.py)
checks the projection identity, phase derivatives, four intermediate
collocation values, value/first/second-derivative join, and the same solve
and projection in Torch and JAX. The production `(3,2)` branch uses named
helpers for the tested collocation solve. Test inputs are synthetic and do
not assert release-calibration accuracy.

The frozen [XHM/XPHM release qualification](../validation/xhm_xphm_release_qualification_20260925.json)
still fails its catalog and holdout waveform gates. The release-`122022`
calibration surfaces and mixing tables have not been proven identical to the
paper-era companion assets; this component must not be described as full
waveform qualification.

## Reproduce the checks

On `len`, with the project on `PYTHONPATH`, run
`python -m pytest -q tests/test_xhm_transition_mixing_math_reference.py`.
