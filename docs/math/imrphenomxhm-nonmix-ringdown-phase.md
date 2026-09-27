# IMRPhenomXHM non-mixing ringdown phase and C1 join

## Scope and status

`xhm.phase.ringdown.nonmix-c1.v1` specifies the published ringdown phase
shape and value/slope attachment for the `(2,1)`, `(3,3)`, and `(4,4)`
intrinsic modes. It takes the phase coefficients, QNM frequencies, join
frequency, and intermediate phase and slope at the join as inputs. The fitted
coefficient surfaces, Eq. (45) rescaling, intermediate fit targets, physical
mode phase gauge, and complete waveform remain separate qualifications.

Before native comparison, the fixed CPU float64 gate uses the three cases and
grids below. Phase and shift tolerances are `5e-13` relative with `2e-11`
absolute floor. Derivative and parameter-gradient tolerances are `2e-11`
relative with `2e-9` absolute floor. Join value/slope residuals are at most
`2e-11` and `2e-9`, respectively. The scalar derivative identity must agree
with 70-digit numerical differentiation to `1e-55` absolute.

## Inputs, conventions, and options

All frequencies are dimensionless `Mf`. Require `0 < j <= f <= 0.3`,
`r>0`, and `d>0`, where `j` is the ringdown join, `r` the mode ring
frequency, and `d` its damping frequency. Supplied real coefficients are
`a2`, `a_lambda`; supplied boundary data are `p=phi_int(j)` and
`g=phi_int'(j)`. The additive phase gauge is fixed only relative to `p`.

| Mode role | `j` | `r` | `d` | `a2` | `a_lambda` | `p` | `g` | `f` grid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 33 | .08 | .095 | .015 | .28 | .13 | 2.7 | 26 | .08, .09, .095, .13, .20 |
| 44 | .10 | .12 | .018 | -.12 | .08 | -1.4 | 18.5 | .10, .11, .12, .16, .25 |
| 21 | .063 | .075 | .012 | .4 | -.07 | .8 | 16.2 | .063, .07, .075, .105, .15 |

The scalar evaluator uses 70-digit mpmath. The native gate uses a three-row
CPU float64 batch, Torch autograd, and JAX JIT/grad.

## Mathematics and components

[García-Quirós et al., arXiv:2001.10914v2, Sec. VI.2.1, Eqs. (43)--(44)
and the paragraph following Eq. (45)](https://arxiv.org/html/2001.10914v2)
give, for non-mixing modes,

\[
 \phi'_{RD}(f)=a_2\frac{r^2}{f^2}
   +a_\lambda\frac{d}{d^2+(f-r)^2}+t,
 \qquad
 \phi_{RD}(f)=-a_2\frac{r^2}{f}
   +a_\lambda\arctan\frac{f-r}{d}+tf+c.
\]

The paper requires a smooth connection to the intermediate reconstruction;
for its value `p` and slope `g` at `j`, the unique shifts are

\[
 t=g-a_2\frac{r^2}{j^2}
     -a_\lambda\frac{d}{d^2+(j-r)^2},\qquad
 c=p+a_2\frac{r^2}{j}
     -a_\lambda\arctan\frac{j-r}{d}-tj.
\]

Hence `phi_RD(j)=p` and `phi'_RD(j)=g`. An independent second derivative is
`-2a2 r²/f³ - 2a_lambda d(f-r)/[d²+(f-r)²]²`. These derivative identities
provide stronger checks than phase-value agreement alone.

## Implementation mapping

The scalar reference is
`diffgw_reference.imrphenomxhm_nonmix_ringdown_phase`. Production Torch
and JAX `imrphenom_xhm.py` modules call their own
`_nonmix_ringdown_phase_shifts` and `_nonmix_ringdown_phase` helpers in
the `(2,1)`, `(3,3)`, and `(4,4)` paths. Neither frontend imports the
reference.

## Evidence and limitations

The gate checks scalar antiderivatives, both C1 conditions, native shifts,
batched phases and frequency/parameter gradients for all three mode roles.
It proves the printed phase shape and join algebra for supplied inputs, not
the fitted inputs or agreement with numerical relativity. The `(3,2)`
spheroidal phase and mixed spherical reconstruction are outside this slice.

## Reproduce the checks

```bash
JAX_ENABLE_X64=1 PYTHONPATH=src:tests python -m pytest -q \
  tests/test_xhm_nonmix_ringdown_phase_math_reference.py
```
