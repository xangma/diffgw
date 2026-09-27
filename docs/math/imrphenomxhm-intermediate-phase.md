# IMRPhenomXHM non-mixing intermediate phase

## Scope and status

`xhm.phase.intermediate.nonmix-collocation.v1` is a research-only component for
the `(2,1)`, `(3,3)`, and `(4,4)` intrinsic mode phases. It specifies the
printed default collocation geometry and the five-coefficient phase-derivative
solve for **supplied** fitted targets. It does not certify the target fit
surfaces, inspiral/ringdown attachment, `(3,2)` mode, or a complete waveform.

The acceptance thresholds below were fixed before comparing the native
frontends: default-node absolute error at most `2e-14`, coefficient relative
error at most `5e-8` (with absolute floor `1e-10`), phase/derivative absolute
error at most `2e-8` on the declared dimensionless-frequency fixtures, and
autodiff against independent central differences to `2e-5` relative.

## Inputs, conventions, and options

Inputs are dimensionless Fourier frequencies `Mf > 0`, symmetric mass ratio
`0 < eta <= 1/4`, a positive inspiral phase cut `f_ins`, ring frequency
`f_ring > f_ins`, damping frequency `f_damp > 0`, and five supplied fitted
phase-derivative targets. The default nodes are the paper's `[C1,C2,E2,E3,E5]`
subset. This component leaves the additive phase constant at zero; the full
model fixes it at an attachment. The scalar evaluator uses mpmath at 70 decimal
digits, while native comparisons use CPU float64.

## Mathematics and components

García-Quirós et al.,
[arXiv:2001.10914v2](https://arxiv.org/html/2001.10914v2), Sec. V B,
Eqs. (38)--(39), Table 7, and the paragraph following Table 7 give

\[
  \phi'_{\ell m}(f)=a_\lambda\frac{d}{d^2+(f-r)^2}
  +a_0+\frac{a_1}{f}+\frac{a_2}{f^2}+\frac{a_4}{f^4},
  \quad (\ell,m)\in\{(2,1),(3,3),(4,4)\},
\]

where `r=f_ring`, `d=f_damp`, and `a_3=0`. The five coefficients solve
`phi'(f_i)=F_i`. With `s=beta(eta) f_ins`, `e=f_ring`, and
`beta=1+0.001(0.25/eta-1)`, the default nodes are

\[
  (f_1,f_2,f_3,f_4,f_6)=
  \left(s,\frac{(\sqrt3+2)s-(\sqrt3-2)e}{4},
  \frac{e+3s}{4},\frac{e+s}{2},\frac{7e+s}{8}\right).
\]

Termwise integration, with an unfixed constant, gives
`a_lambda atan((f-r)/d)+a_0 f+a_1 log(f)-a_2/f-a_4/(3f^3)`.
The independent scalar reference constructs the five linear equations from
these expressions and solves them at high precision.

## Implementation mapping

The independent reference is
`diffgw_reference.imrphenomxhm_intermediate_phase`. Torch and JAX production
call `_nonmix_intermediate_phase_nodes`,
`_nonmix_intermediate_phase_coefficients`, and the phase and derivative
evaluators in their respective `imrphenom_xhm.py` modules. Neither frontend
imports the reference or fitted targets from it.

## Evidence and limitations

The gate checks default nodes, coefficient solve, all five interpolation
conditions, integrated phase and analytic derivative, native batch values,
Torch parameter gradients, and JAX JIT/grad. Targets are deliberately supplied
fixtures: agreement proves the printed algebra and native solve for those
cases, not the provenance or accuracy of the actual fitted surfaces. The
paper's alternate sparse-calibration subset is outside this v1 component.
The full XHM and XPHM release gates remain unqualified.

## Reproduce the checks

On a float64 Torch/JAX CPU environment:

```bash
JAX_ENABLE_X64=1 PYTHONPATH=src:tests python -m pytest -q \
  tests/test_xhm_intermediate_phase_math_reference.py
```
