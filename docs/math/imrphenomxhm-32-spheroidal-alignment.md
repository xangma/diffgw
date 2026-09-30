# IMRPhenomXHM (3,2) spheroidal ringdown alignment

## Scope and status

`xhm.phase.32.spheroidal-alignment.v1` specifies the algebra that fixes the
linear time and constant phase offsets of a supplied spheroidal `(3,2)`
ringdown phase relative to a supplied `(2,2)` carrier. It is conditional on
QNM frequencies, shape coefficients, carrier values, and two fitted alignment
targets. It does not qualify those fitted surfaces, the Eq. (48) coefficient
calibration, exceptional cuts, or the full mode.

Freeze CPU FP64 value tolerances `rtol=2e-13, atol=5e-12` and complete-input
Jacobian tolerances `rtol=2e-12, atol=5e-10` before native comparison. Use
70-digit scalar arithmetic and require both Torch and JAX JIT to pass.

## Inputs, conventions, and options

All frequencies are positive dimensionless `Mf`. The ordered input row is
`(r22,d22,r32,d32,a2,a4,a_lambda,base_time,carrier_slope,carrier_phase,delta_t,delta_phi)`.
The carrier slope is evaluated at `f0=r22+d22`; carrier phase at `f1=r22`.
`delta_t` and `delta_phi` are *supplied* fitted offsets, with the signs printed
in Eqs. (49)–(50). `base_time` is the unaligned Eq. (48) linear coefficient;
the result must be independent of it.

| Row | `r22,d22,r32,d32,a2,a4,a_lambda,base_time,carrier_slope,carrier_phase,delta_t,delta_phi` |
| --- | --- |
| 1 | `.083,.013,.091,.014,.20,.0001,.30,8,11,-.4,2,.15` |
| 2 | `.104,.016,.115,.018,.13,-.00008,-.24,4,-7,1.2,-1,-.3` |
| 3 | `.072,.010,.078,.012,-.11,.00004,.41,-3,5,.2,.6,-.1` |

The arXiv:2001.10914v2 source archive is identified by SHA-256
`cc05e7dbff7bf0c42a10b7da1f4840eefc55b252168d6bfb085155a7bdc5318e`.
Its `anc/ParspaceFits/IMRPhenomXHM_PhaseFits.nb` (SHA-256
`48cc263ab9f2837220cd11fbfd99205562b8863af077074f46d730e35382108b`)
identifies the reference frequencies, and its `PhaseFits/32/deltadPhiS.m`
and `deltaPhiS.m` hold the respective fit records (SHA-256
`07107d0436b316792e0b2cd343b8cce3efa2a96276a26304f6ecbdfb7c975455`
and `e92a6657458934bea6bb8466b97c7cd90a45d57d98eff9b546e7d64804fac772`).
The [arXiv license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
grants arXiv a non-exclusive distribution license; it does not state that the
ancillary fit records can be redistributed under this repository's MIT license.
No fit coefficients are embedded in this specification or reference.

## Mathematics and components

[García-Quirós et al., arXiv:2001.10914v2, Eqs. (46)–(50)](https://arxiv.org/html/2001.10914v2)
define the ringdown phase shape and the two relative offsets. The
[author-published phase-fit notebook](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/IMRPhenomXHM_PhaseFits.nb)
fixes `f0=r22+d22` for Eq. (49) and `f1=r22` for Eq. (50). For an unaligned
phase `P(f;t,c)` from Eq. (47), set

```text
t_final = t_base + [carrier_slope + delta_t - P'(f0;t_base,0)]
c_final = carrier_phase + delta_phi - P(f1;t_final,0).
```

Then `P'(f0;t_final,c_final)-carrier_slope=delta_t` and
`P(f1;t_final,c_final)-carrier_phase=delta_phi`. These are algebraic
identities for supplied inputs, not evidence for the fitted input values.

The same notebook explicitly defines four Eq. (48) derivative targets at
Table 9 nodes `r22`, `r32-1.5*d32`, `r32-0.5*d32`, and `r32+0.5*d32`.
The `.m` archive contains `dphiS_p1.m`, `dphiS_p4.m`, and an alternate
`dphiS_fring32.m`, but no separate p2/p3 `.m` records. The files
`3Dfit_p1_lm_32.m` through `p4` are **intermediate-phase**, not Eq. (48),
targets. Their names must not be used as ringdown fit provenance.

## Implementation mapping

`diffgw_reference.imrphenomxhm_transition_mixing` holds the independent
scalar alignment. Both `torchwave` and `jaxwave`
`approximants/imrphenom_xhm.py` call their own alignment helper from the
production `(3,2)` ringdown branch. Neither frontend imports the scalar
reference. The current production coefficient calibration uses targets at
`(r22,r32,r32+0.5*d32)` and fixes `a2_32=a2_22`: it is a **three-target
alternate calibration**, not the paper's four-target Eq. (48) system.

## Evidence and limitations

The maintained test compares both final offsets and their complete input
Jacobians on the frozen three-row grid, and checks both defining residuals.
This qualifies only the alignment algebra for supplied inputs. The
author-published notebook supplies paper-era fit expressions, but no
redistribution terms establishing MIT compatibility or version map to a
specific later XHM release. Do not silently replace the production fit or
claim it implements Eq. (48).

The paper's Eq. (36)/Table 2 specify default `(3,2)` cuts and mention
high-mass/high-spin adjustments. Neither the paper text nor this fit notebook
gives a complete trigger and replacement frequency for that exception.
The amplitude-intermediate veto table is a different rule and cannot fill
this gap. The exceptional cut remains unqualified.

## Reproduce the checks

```bash
JAX_ENABLE_X64=1 PYTHONPATH=src:tests python -m pytest -q \
  tests/test_xhm_32_spheroidal_alignment_math_reference.py
```

The test needs `mpmath`, `numpy`, `torch`, `jax`, and `pytest`; no ancillary
fit assets are imported. Changing the paper-era frequencies, alignment
convention, or selected calibration requires a new component version.
