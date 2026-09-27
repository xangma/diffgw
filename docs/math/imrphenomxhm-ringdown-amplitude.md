# IMRPhenomXHM ringdown amplitude shape

## Scope and status

`xhm.amplitude.ringdown.ansatz.v1` specifies only the dimensionless
ringdown amplitude ratio in García-Quirós et al., Eq. (42). Its fitted
coefficients, QNM frequencies, normalization to physical strain, interval
selection, intermediate attachment, and spherical/spheroidal (3,2) mixing
are outside this component. This is not a full-mode qualification.

Before native comparison, the fixed CPU float64 gate uses the two cases and
frequencies below. Relative value tolerance is `5e-13` with absolute floor
`2e-12`; frequency and parameter derivative tolerance is `2e-11` relative
with absolute floor `2e-10`. Scalar analytical derivatives must agree with
70-digit numerical differentiation to `1e-55` absolute.

## Inputs and conventions

Use dimensionless positive Fourier frequency `f=M f_GW`, positive ring
frequency `r`, damping frequency `d`, and `sigma>0`. The supplied real
parameters are `a_lambda != 0` and `lambda`. The `(3,2)` inputs refer to the
spheroidal ringdown basis. The reference and native gate use:

| Case | Mode role | `r` | `d` | `sigma` | `a_lambda` | `lambda` | `f` grid |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | 33, fixed Table 8 sigma | .09 | .016 | 1.3 | .015 | .4 | .06, .08, .09, .11, .15 |
| B | 21, supplied sigma | .08 | .012 | .91 | -.023 | .65 | .05, .07, .08, .10, .14 |

The chosen interior grid has no active native numerical clamp. Each case is
also evaluated as a batch item. The scalar reference uses 70-digit mpmath;
native checks use Torch/JAX CPU float64 and JAX JIT.

## Mathematics and components

[García-Quirós et al., arXiv:2001.10914v2, Sec. VI.1, Eq. (42) and
Table 8](https://arxiv.org/html/2001.10914v2) specify, with `w=d sigma`,

\[
 H(f)=\frac{A_{RD}^{\ell m}}{A_0^{22}}
 = f^{-1/12}\frac{|a_\lambda|w}{(f-r)^2+w^2}
   \exp\!\left[-\frac{(f-r)\lambda}{w}\right].
\]

Table 8 fixes `sigma=1.3` for (3,3) and `1.33` for (3,2) and (4,4);
the (2,1) `sigma` remains fitted. All are caller-supplied here. An
independent derivative identity used by the gate is

\[
 \partial_f H=H\left[-\frac{1}{12f}
 -\frac{2(f-r)}{(f-r)^2+w^2}-\frac{\lambda}{w}\right],\quad
 \partial_\lambda H=-\frac{f-r}{w}H,\quad
 \partial_{a_\lambda}H=H/a_\lambda.
\]

## Implementation mapping

The scalar implementation lives in
`diffgw_reference.imrphenomxhm_ringdown_amplitude`. The production
`eval_h_rd` closures in the Torch and JAX `imrphenom_xhm.py` modules call
their own `_ringdown_amplitude_ratio` helpers. Native positive-frequency,
exponential, and denominator guards remain for numerical robustness; this
component checks the published expression only where those guards are idle.

## Evidence and limitations

The gate compares 70-digit scalar values with native batched values and
analytical frequency/coefficient gradients, including negative `a_lambda`,
and verifies JAX compilation. This establishes the functional shape for
supplied parameters. It does not establish the provenance or quality of
`a_lambda`, `lambda`, the (2,1) `sigma`, Kerr QNM values, physical amplitude
normalization, or the full XHM ringdown and attachment.

## Reproduce the checks

```bash
JAX_ENABLE_X64=1 PYTHONPATH=src:tests python -m pytest -q \
  tests/test_xhm_ringdown_amplitude_math_reference.py
```
