# IMRPhenomXHM default ringdown cutting frequencies

## Scope and status

`xhm.frequency.ringdown-cut.default.v1` specifies the default amplitude and
phase ringdown cuts for the `(2,1)`, `(3,3)`, `(3,2)`, and `(4,4)` modes. This
is a supplied-QNM frequency algebra gate, not a QNM-fit, attachment, or
whole-waveform qualification. The paper notes exceptional `(3,2)` cut
adjustments for high-mass, high-spin cases; those are outside this default
slice.

Before comparison with native code, fix CPU float64 value tolerance to
`rtol=0, atol=2e-15` and all input Jacobian tolerance to `rtol=0,
atol=2e-14`. The fixed three-row input grid is below. Both frontends must
agree with the independent 70-digit scalar evaluator and the analytic sparse
Jacobian, including zero cross-row and unused `(3,2)` QNM derivatives.

## Inputs, conventions, and options

Frequencies are positive dimensionless `Mf`. `r_lm` and `d_lm` are supplied
fundamental QNM ring and damping frequencies, respectively. Rows are ordered
`(22,21,33,32,44)`; no QNM provider or intrinsic binary parameterization is
specified by this component.

| Row | `r22,r21,r33,r32,r44` | `d22,d21,d33,d32,d44` |
| --- | --- | --- |
| 1 | `.080,.075,.095,.091,.120` | `.012,.012,.015,.014,.018` |
| 2 | `.087,.079,.100,.095,.130` | `.010,.010,.013,.012,.016` |
| 3 | `.075,.072,.088,.086,.110` | `.015,.014,.018,.017,.020` |

## Mathematics and components

[García-Quirós et al., arXiv:2001.10914v2, Sec. V, Eq. (36) and Table 2](https://arxiv.org/html/2001.10914v2)
give `f_RD = delta_RD * r_lm + epsilon_RD`, with distinct amplitude and phase
choices. The eight default cuts are

| Mode | Amplitude cut | Phase cut |
| --- | --- | --- |
| 21 | `0.75 r21` | `r21 - d21` |
| 33 | `0.95 r33` | `r33 - d33` |
| 32 | `r22` | `r22 - 0.5 d22` |
| 44 | `0.90 r44` | `r44 - d44` |

The Jacobian follows directly: each displayed coefficient is the partial
derivative with respect to its input, with `-1` for each non-mixing phase
damping frequency and `-0.5` for `d22` in the `(3,2)` phase cut. All other
partials are zero.

## Implementation mapping

The independent mpmath evaluator is
`diffgw_reference.imrphenomxhm_ringdown_cuts`. Torch and JAX each call their
own `_ringdown_cut_frequencies` helper from the production XHM waveform path;
neither native frontend imports the scalar reference. The native helper
returns separate amplitude and phase cut dictionaries keyed by mode string.

## Evidence and limitations

The gate compares all eight outputs and their full cross-mode and cross-row
Jacobian for three supplied-QNM rows. This checks the published default
frequency selection that feeds the intermediate and ringdown attachment
paths. It does not validate a QNM provider, exceptional `(3,2)` cut rules,
the fitted collocation targets, or the attached phase/amplitude itself.

The adjacent non-mixing ringdown phase coefficient rule in Eq. (45) is not
qualified here. That equation leaves the mode constants `w_lm` unspecified,
whereas the existing Torch/JAX paths use independently calibrated per-mode
coefficient surfaces rather than an explicit Eq. (45) rescaling. Equating
those without an independently sourced calibration would change the claim.

## Reproduce the checks

```bash
JAX_ENABLE_X64=1 PYTHONPATH=src:tests python -m pytest -q \
  tests/test_xhm_ringdown_cuts_math_reference.py
```
