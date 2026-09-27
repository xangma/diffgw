# IMRPhenomHM phase-offset component

## Scope and status

`hm.phase.offsets.eq9.v1` specifies London's three-branch phase composition
for the six positive-frequency higher modes, conditional on a supplied smooth
$(2,2)$ carrier phase and QNM frequencies. It is a research component of the
provisional [IMRPhenomHM family](../waveforms/research/phenom-higher-modes.md),
not a complete waveform qualification.

## Inputs, conventions, and options

$f$, $f_0^\phi$, $f_{\ell m}^{\rm RD}$, and $f_{22}^{\rm RD}$ use positive,
dimensionless $Mf$ units, with $0<f_0^\phi<f_{\ell m}^{\rm RD}$ and
$f_{22}^{\rm RD}>2f_0^\phi/m$. The component accepts a supplied carrier phase
$\phi_{22}(g)$ in radians; the scalar reference contains no PhenomD fits or QNM
table. Each join value belongs to the lower-frequency branch. The paper chooses
$f_0^\phi=0.014 f_{\ell m}^{\rm RD}/f_{22}^{\rm RD}$; the tests use this value.
The phase origin is fixed by the printed low-frequency PN/symmetry offset.

## Mathematics and components

[London et al., arXiv:1708.00404v4, Methods Eqs. (3), (8)–(9), (13)](https://arxiv.org/html/1708.00404v4)
give $\phi_{\ell m}=\kappa\phi_{22}[g(f)]+\Delta_{\ell m}$.
The [frequency-map component](imrphenomhm-frequency-map.md) specifies $g$:
for the PhenomD phase, its slopes are $2/m$, $s=(f_{22}^{\rm RD}-2f_0^\phi/m)/
(f_{\ell m}^{\rm RD}-f_0^\phi)$, and $f_{22}^{\rm RD}/f_{\ell m}^{\rm RD}$.
Their reciprocals are $\kappa_0,\kappa_1,\kappa_2$. With
$g_0=2f_0^\phi/m$, $Q=f_{22}^{\rm RD}$, and
$\Delta_0=\pi[3\ell+((\ell+m)\bmod 2)-2]/2$, continuity fixes

$$
\Delta_1=\Delta_0+(\kappa_0-\kappa_1)\phi_{22}(g_0),\qquad
\Delta_2=\Delta_1+(\kappa_1-\kappa_2)\phi_{22}(Q).
$$

The one-sided phase derivatives agree at both joins because $\kappa_i g_i'=1$
for all three branches, assuming a differentiable carrier. The high-frequency
map here is the paper's PhenomD-specific multiplicative Eq. (13), rather than
its earlier additive map. The high offset is the continuity value implied by
Eqs. (3) and (8)–(9) after that substitution.

| Component ID/version | Source | Variant | Status | Scalar reference | Torch | JAX | Maintained check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `hm.phase.offsets.eq9.v1` | London Methods Eqs. (3), (8)–(9), (13) | supplied carrier/QNM | executable, conditional | `mapped_phase` | `_production_mapped_phase_c1` | `_production_mapped_phase_c1` | `test_imrphenomhm_phase_offsets_math_reference.py` |

## Implementation mapping

The independent mpmath evaluator is
`src/diffgw_reference/imrphenomhm_phase_offsets.py`; the tests call the live
`_production_mapped_phase_c1` helpers in Torch and JAX. Neither frontend
imports the scalar reference. The frontend helpers are called by the public HM
assembly, but this check supplies a polynomial carrier to isolate the offset
algebra from PhenomD calibration and QNM fits. The separate Eq. (14) carrier
phase correction is not covered here.

## Evidence and limitations

The hand-reduced low-branch value, independent 70-digit scalar values around
both joins, and symbolic derivatives with respect to all four frequency inputs
and symbolic one-sided frequency derivatives at both joins are checked for
the five non-22 modes in both CPU FP64 frontends. This tests the source
transcription and sampled implementation agreement. The paper specifies
$f_0^A=0.018 f_{\ell m}^{\rm RD}/f_{22}^{\rm RD}$ and
$f_0^\phi=0.014 f_{\ell m}^{\rm RD}/f_{22}^{\rm RD}$, whereas the public
Torch/JAX assembly currently passes the opposite assignments. These are
distinct source conventions: [Khan et al.'s PhenomD carrier](https://arxiv.org/html/1508.07253)
joins amplitude at $Mf=0.014$ and phase at $Mf=0.018$, while London explicitly
optimized separate higher-mode mapping transitions. The public assembly uses
the PhenomD pair. This component gate supplies London's printed phase
transition and therefore does not qualify the public assembly's transition
choice. It also does not
establish amplitude scaling, fitted QNM values, Eq. (14) carrier correction,
whole-mode binary agreement, or physical accuracy across the model domain.

## Reproduce the checks

With the reference, Torch, JAX, and test extras installed, run
`JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest -q
tests/test_imrphenomhm_phase_offsets_math_reference.py --reference-device=cpu
-ra`. The strict mathematical-reference CI job discovers
this file. A changed equation or convention needs a new component version and
reviewed source/fixture update.
