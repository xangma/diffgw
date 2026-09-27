# IMRPhenomHM public intrinsic phase composition

## Scope and status

`hm.phase.public-composition.eq3-14.v1` checks how the public D-carrier HM
generator composes previously checked frequency-map, phase-offset, and
ringdown-correction components. This is a bounded executable research
component for the five non-(2,2) positive-frequency modes. It is conditional
on supplied PhenomD phase, fit coefficients, and HM/D QNM frequencies; it is
not a full intrinsic-mode or waveform qualification.

## Inputs, conventions, and options

The input is dimensionless positive $x=Mf$. The tested modes are $(2,1)$,
$(3,3)$, $(3,2)$, $(4,4)$, and $(4,3)$. The carrier callback supplies the
public PhenomD phase $P_D(y)$ with $f_{\rm ref}=\varnothing$, $\phi_c=t_c=0$.
The tested systems have $m_1>m_2$, so the native signed odd-$m$ amplitude
factor is positive and the unprojected public mode has unit phasor
$\exp[-i\Phi_{\ell m}(x)]$. Other phase origins and amplitude sign branches
are separate gates.

Both public outer-map choices are frozen: `phenomd_runtime` places the
amplitude/phase cutoffs at $0.014/0.018$ times
$f_{\ell m}^{\rm RD}/f_{22}^{\rm RD}$; `london_2018` uses the paper's
$0.018/0.014$ pair. These are named options, not interchangeable gauges.
The D carrier, $\eta$, $\alpha_4$, $\alpha_5$, HM QNM pairs and D QNM pair
are supplied from each frontend. The scalar composer neither imports nor
derives those functions or fits.

## Mathematics and components

[London et al., arXiv:1708.00404v4, Methods Eqs. (3), (8)–(9), (13)–(14)](https://arxiv.org/html/1708.00404v4)
map the quadrupole phase through a three-branch frequency map, its inverse
slopes $\kappa_b$, and continuity offsets $\Delta_b$. Their Eq. (14)
changes the merger-ringdown arctangent of
[Khan et al., arXiv:1508.07253v2, Eq. (14)](https://arxiv.org/html/1508.07253v2).
The existing [frequency-map](imrphenomhm-frequency-map.md),
[phase-offset](imrphenomhm-phase-offsets.md), and
[ringdown-phase](imrphenomhm-ringdown-phase.md) specifications derive the
three ingredients separately. This composition fixes their order:

$$
C_{\ell m}(y)=P_D(y)+\delta_{\ell m}^{C^1}(y),\qquad
\Phi_{\ell m}(x)=\kappa_b C_{\ell m}(g_b(x))
                         +\Delta_b[C_{\ell m}].
$$

Here $g_b$ is London's PhenomD-specific Eq. (13) high branch with the
Eq. (5) lower branches, and the offsets are evaluated from the **corrected**
carrier at both joins. The native $\delta^{C^1}$ uses the D-anchored QNM
adaptation and tangent subtraction documented in the ringdown specification;
the literal printed London–Khan term is recovered only when their 22 QNM
pairs coincide. London's Eq. (9) gives phase offsets; its modified
merger-ringdown arctangent is Eq. (14).

The reciprocal map slopes imply $C^1$ phase across $x=f_0^\phi$ and
$x=f_{\ell m}^{\rm RD}$ when $C_{\ell m}$ is differentiable. The separate
D attachment $J=f_{22}^{\rm RD,D}/2$ is sampled after mapping back to the
mode frequency, to check the full public composition at that third join.

| Component ID/version | Sources | Variant | Status | Scalar reference | Torch/JAX public target | Maintained check |
| --- | --- | --- | --- | --- | --- | --- |
| `hm.phase.public-composition.eq3-14.v1` | London Methods Eqs. (3), (8)–(9), (13)–(14); Khan Eq. (14) | D carrier; both named outer-map settings | executable, conditional | `imrphenomhm_phase_composition.composed_phase` | `IMRPhenomHM(..., return_unprojected_modes=True)` | `test_imrphenomhm_phase_composition_math_reference.py` |

## Implementation mapping

`src/diffgw_reference/imrphenomhm_phase_composition.py` composes the two
independent mpmath references; it imports no native frontend. The maintained
test obtains supplied D phase values at all needed mapped frequencies and
join points, then compares the resulting unit phasor with public
`src/torchwave/approximants/imrphenom_hm.py` and
`src/jaxwave/approximants/imrphenom_hm.py` unprojected modes. This reaches
the actual public assembly instead of comparing another helper against
itself. A nonzero mode amplitude is required to define its phase.

## Evidence and limitations

The [len CPU-FP64 receipt](../validation/imrphenomhm_phase_composition_20260927.md)
records values and one-sided slopes across all three joins for two unequal-mass
aligned-spin systems, all five non-22 modes, both map settings, and both
frontends. It fixes the public phase sign and the zero reference/time gauge.
The check does not independently qualify D phase fits, $\alpha$ surfaces,
remnant/QNM fits, amplitude sign/zero behavior, $f_{\rm ref}$ or time/orbital
phase gauges, negative frequencies, projection, whole-mode binary agreement,
or NR physical accuracy. In particular, a passing composition check cannot
promote the whole IMRPhenomHM family.

## Reproduce the checks

With reference, Torch, JAX and test extras installed, run
`JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest -q
tests/test_imrphenomhm_phase_composition_math_reference.py
--reference-device=cpu -ra`. Strict math-reference CI discovers this file.
Changes to the source equation, sign/gauge adapter, or named map option
require a versioned review.
