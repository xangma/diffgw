# IMRPhenomXHM (3,2) spheroidal ringdown phase shape

## Scope and status

`xhm.phase.32.spheroidal-ringdown-shape.v1` is the published phase primitive
and derivative of Eqs. (46)–(47), conditional on **supplied** QNM frequencies,
four phase-derivative coefficients, and an integration offset. It isolates the
shape used in the production `(3,2)` spheroidal branch. It does not qualify the
four-target Eq. (48) fit, the alignment data of Eqs. (49)–(50), the basis
projection, or a complete `(3,2)` mode.

Before native comparison, freeze CPU FP64 tolerances as `rtol=2e-13,
atol=5e-12` for both phase and first-derivative values, and `rtol=2e-12,
atol=5e-10` for their complete input Jacobians. The scalar calculation uses
70 decimal digits. Cross-row derivatives and the phase-derivative sensitivity
to the integration offset must vanish. Both Torch and JAX must pass, including
JAX JIT and a batched frequency axis.

## Inputs, conventions, and options

All frequencies are positive dimensionless `Mf`. Each row supplies
`(f_ring32, f_damp32, alpha2, alpha4, alpha_lambda, dphi_RD, phi_RD)`.
`phi_RD` is the phase integration offset; the coefficients are independent
caller inputs, not values inferred from the old three-target production fit.
The test probes the frequency grid shown below. In each row, `f` is the first
Jacobian input, followed by the seven parameters in the displayed order.

| Row | `f_ring32, f_damp32, alpha2, alpha4, alpha_lambda, dphi_RD, phi_RD` | Probe frequencies |
| --- | --- | --- |
| 1 | `.091,.014,.20,.0001,.30,8,.17` | `.070,.084,.091,.112` |
| 2 | `.115,.018,.13,-.00008,-.24,4,-.31` | `.086,.106,.115,.142` |
| 3 | `.078,.012,-.11,.00004,.41,-3,.09` | `.061,.073,.078,.101` |

## Mathematics and components

[García-Quirós et al., arXiv:2001.10914v2, Eqs. (46)–(47)](https://arxiv.org/html/2001.10914v2)
give, with `r=f_ring32`, `d=f_damp32`, and `L=d/[d²+(f-r)²]`,

```text
phi' = alpha2/f² + alpha4/f⁴ + alpha_lambda L + dphi_RD,
phi  = -alpha2/f - alpha4/(3f³)
       + alpha_lambda atan((f-r)/d) + dphi_RD f + phi_RD.
```

The scalar reference differentiates the phase independently and checks it
against the printed derivative. The Jacobian gate differentiates each scalar
expression with mpmath rather than sharing the native autodiff graph.
The natural real arctangent branch is used; `d>0` ensures a smooth primitive.

## Implementation mapping

The scalar source is `diffgw_reference.imrphenomxhm_transition_mixing`.
Production Torch and JAX `imrphenom_xhm.py` expose native phase and derivative
helpers and call them from the `(3,2)` ringdown branch. The helpers accept
per-row parameters and broadcast them across a frequency axis. No native
frontend imports the scalar reference.

## Evidence and limitations

The gate evaluates both equations and all derivatives with respect to `f`,
the supplied QNM pair, four coefficients, and offset, on the fixed three-row
grid. This evidence validates only the shape, primitive relationship, and
native differentiation for those inputs. It cannot establish the provenance
or physical accuracy of any fitted target or QNM provider.

The paper's Table 9 lists four `(3,2)` spheroidal phase nodes for Eq. (48),
but the existing production branch uses three inherited fitted targets and
fixes `alpha2_32=alpha2_22`. That release-specific calibration remains
unqualified. Likewise, the paper identifies exceptional high-mass/high-spin
`(3,2)` ringdown cuts after Eq. (36)/Table 2 without publishing a complete
parameter trigger and replacement cut formula; the default-cut gate must not
be extended by inference.

## Reproduce the checks

```bash
JAX_ENABLE_X64=1 PYTHONPATH=src:tests python -m pytest -q \
  tests/test_xhm_32_spheroidal_ringdown_phase_math_reference.py
```
