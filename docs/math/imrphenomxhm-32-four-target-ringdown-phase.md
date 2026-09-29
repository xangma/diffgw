# IMRPhenomXHM (3,2) four-target spheroidal ringdown phase solve

## Scope and status

`xhm.phase.32.spheroidal-four-target-solve.v1` is the conditional algebra of
García-Quirós et al.'s Eq. (48): solve the four coefficients in the published
Eq. (46) phase derivative from **caller-supplied** derivative targets. This
does not supply or qualify the target fits, the production calibration, or a
complete `(3,2)` mode.

Freeze CPU FP64 coefficient and collocation-residual tolerances
`rtol=2e-11, atol=5e-10`, and full input-Jacobian tolerances
`rtol=2e-9, atol=5e-7` before comparing any native helper. The independent
scalar reference uses 70 decimal digits. Torch eager and JAX JIT must pass
on the same rows. A full-rank, reasonably conditioned scaled matrix is a
precondition, not a claim about all physical parameters.

## Inputs, conventions, and options

Inputs are `(r22,r32,d32,G1,G2,G3,G4)`, with positive dimensionless `Mf`
QNM frequencies and four phase-derivative targets in Table 9 order. The
returned coefficients are `(alpha2,alpha4,alpha_lambda,dphi_RD)`. No phase
integration constant is solved by Eq. (48). The frozen rows are:

| Row | `r22,r32,d32` | `G1,G2,G3,G4` |
| --- | --- | --- |
| 1 | `.080,.095,.012` | `42,39,35,31` |
| 2 | `.082,.110,.014` | `-10,-9,-7,-6` |
| 3 | `.070,.087,.010` | `15,16,18,20` |

These are synthetic supplied targets, not author-fit evaluations. Row-wise
Jacobians cover all seven inputs; derivatives across rows must vanish. A
coincident-node case `r22=r32-1.5*d32` must fail explicitly in the scalar
reference and Torch eager helper; JAX JIT accepts only prevalidated domain
inputs. A nearby distinct-node case is diagnostic for conditioning, not a
production claim.

## Mathematics and components

[García-Quirós et al., arXiv:2001.10914v2, Eqs. (46) and (48), Table 9](https://arxiv.org/html/2001.10914v2)
specify

```text
f = (r22, r32-1.5*d32, r32-0.5*d32, r32+0.5*d32)
G_i = alpha2/f_i² + alpha4/f_i⁴
      + alpha_lambda*d32/[d32²+(f_i-r32)²] + dphi_RD.
```

To expose numerical rank without changing this equation, let `x_i=f_i/r32`
and `delta=d32/r32`. The dimensionless matrix has row
`B_i=(x_i^-2, x_i^-4, delta/[delta²+(x_i-1)²], 1)` and solves
`B b=G`, where
`b=(alpha2/r32²,alpha4/r32⁴,alpha_lambda/r32,dphi_RD)`.
Recover the printed coefficients by multiplying `b` by
`(r32²,r32⁴,r32,1)`. The gate reports `rank(B)` and spectral condition
number `kappa_2(B)=sigma_max/sigma_min` on every frozen row. Positive,
distinct nodes and nonzero determinant are required; closeness to a
coincident node can still make the solve poorly conditioned.

## Implementation mapping

The independent 70-digit solve belongs in
`diffgw_reference.imrphenomxhm_transition_mixing`. The existing
`torchwave.approximants.xhm_phase_clean` offers a standalone four-target
helper; a JAX counterpart may be checked against the scalar reference.
Neither standalone helper is called by the production waveform. The
production Torch and JAX branches currently use targets at
`(r22,r32,r32+0.5*d32)` and fix `alpha2_32=alpha2_22`, so agreement with
the conditional four-target solve cannot qualify that branch.

## Evidence and limitations

The source archive for arXiv:2001.10914v2 has SHA-256
`cc05e7dbff7bf0c42a10b7da1f4840eefc55b252168d6bfb085155a7bdc5318e`.
Its [author phase-fit notebook](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/IMRPhenomXHM_PhaseFits.nb)
(SHA-256 `48cc263ab9f2837220cd11fbfd99205562b8863af077074f46d730e35382108b`)
contains expressions for all four paper-era targets. The ancillary `.m`
directory instead exports `dphiS_p1.m`, `dphiS_p4.m`, and an alternate
`dphiS_fring32.m`; its `3Dfit_p1_lm_32.m` through `p4` are intermediate-phase
targets. The [arXiv non-exclusive distribution license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
does not establish MIT redistribution permission for the ancillary fits.
This component embeds no fitted expressions or values.

The tests can establish Eq. (48) linear-algebra and differentiation fidelity
for supplied targets only. They cannot establish that production has the
four required fit surfaces, a release-specific mapping from paper-era fits,
or a physically validated `(3,2)` ringdown phase.

## Reproduce the checks

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_xhm_32_four_target_ringdown_phase_math_reference.py
```

The test needs `mpmath`, `numpy`, `torch`, `jax`, and `pytest`. It imports no
ancillary fit asset. Changing node order, coefficient basis, or calibration
requires a new component version.
