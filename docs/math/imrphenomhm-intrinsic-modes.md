# IMRPhenomHM complex intrinsic-mode assembly

## Scope and status

`hm.intrinsic.complex-assembly.eq3.v1` composes the source-derived HM frequency,
amplitude, and phase components into complex positive-frequency modes. The
reference is conditional on supplied PhenomD carrier amplitude and phase,
ringdown frequencies, and phase-fit inputs. It checks the assembly equation;
it does not claim independently calibrated inputs or whole-model accuracy.

## Inputs, conventions, and options

The frequency $x=Mf$ and all QNM frequencies are positive and dimensionless.
The six modes are $(2,2)$, $(2,1)$, $(3,3)$, $(3,2)$, $(4,4)$, and $(4,3)$.
The paper variant uses London's printed transition frequencies
$f_0^A=0.018 f_{\ell m}^{\rm RD}/f_{22}^{\rm RD}$ and
$f_0^\varphi=0.014 f_{\ell m}^{\rm RD}/f_{22}^{\rm RD}$. The $(2,1)$,
$(3,2)$, and $(4,3)$ ratios use the named Mishra-source 1.5PN options;
the $(3,3)$ and $(4,4)$ ratios use their leading-frequency terms. The native
`phenomd_runtime` transition setting and binary-compatible PN defaults are
separate variants and are not represented as the paper choices.

The carrier callbacks and calibrated inputs are arguments to the scalar
reference. A frontend test may obtain these from its public D and QNM
interfaces, but that makes the check **conditional**: shared errors in those
inputs are outside its reach. The 22 component reduces directly to the
supplied D carrier. The maintained frontend check covers
`f_ref=None` and `f_ref=40 Hz`; at finite reference frequency it supplies
the frontend's D peak slope as the explicit HM-relative gauge input.

## Mathematics and components

[London et al., arXiv:1708.00404v4, Methods Eqs. (2)–(5), (8)–(9),
(13)–(14)](https://arxiv.org/html/1708.00404v4) write the intrinsic Fourier
mode as $\tilde h_{\ell m}=A_{\ell m}\exp(i\varphi_{\ell m})$ and prescribe

$$
A_{\ell m}(x)=|\beta_{\ell m}(x)|\,A_{22}(g_A(x)),\qquad
\varphi_{\ell m}(x)=\kappa_b C_{\ell m}(g_\varphi(x))
                          +\Delta_b[C_{\ell m}].
$$

Here $g_A$ is Eq. (5), $g_\varphi$ uses the PhenomD-specific high branch,
and $\beta$ is Eq. (4) with the retained PN envelopes. In the generic paper
composer, the supplied carrier phase is the input to Eqs. (8)–(9); the public
adapter first constructs the Eq. (14)-corrected D phase. The offsets and
inverse slopes enforce the piecewise phase relation. Their separate
derivations and gates are recorded
in the [frequency map](imrphenomhm-frequency-map.md),
[amplitude ratios](imrphenomhm-amplitude-ratio.md), and
[phase composition](imrphenomhm-phase-composition.md).

The paper formula uses $|\beta|$. The native positive-frequency carrier
stores the same assembly with an $\exp(-i\varphi)$ Fourier sign, a named
odd-$m$ body-exchange sign, and optional orbital/reference/time gauges.
Those operations are an explicit public adapter, not part of London's
Eq. (3). [García-Quirós et al., arXiv:2001.10914v2, Sec. II Eqs. (2),
(4)–(9)](https://arxiv.org/html/2001.10914v2) fixes the related
positive-frequency mode-pair convention. The
[SXS convention bridge](imrphenomhm-sxs-conventions.md) derives that pair,
the frequency-reversing conjugation, and the finite-reference invariant.
Angular projection is separately
specified in the [six-mode projection](imrphenomhm-projection.md).

| Component ID/version | Source | Scalar reference | Public targets | Maintained check |
| --- | --- | --- | --- | --- |
| `hm.intrinsic.complex-assembly.eq3.v1` | London Methods Eqs. (2)–(5), (8)–(9), (13)–(14); Mishra PN inputs | `imrphenomhm_intrinsic_modes` | Torch/JAX `IMRPhenomHM` unprojected modes | `test_imrphenomhm_intrinsic_modes_math_reference.py` |

## Implementation mapping

`src/diffgw_reference/imrphenomhm_intrinsic_modes.py` composes independent
scalar reference components without importing Torch or JAX. The maintained
test compares complex unprojected mode values from both public frontends
under explicitly selected options. The separate projection gate compares
mode-to-polarization coefficients and summed plus/cross values. Neither
reference is a second implementation of D amplitude or the fitted HM QNMs.

## Evidence and limitations

This is a source-to-composition and conditional frontend-agreement gate, not
an NR accuracy result. The [run receipt](../validation/imrphenomhm_intrinsic_modes_20260927.md)
records the tested domain, tolerances, dependencies, and untested gauges. London's
literal Eq. (4) ratio has a $0/0$ limit for equal-mass equal-spin odd modes;
the native zero convention is an adapter and needs a separate check. The
inherited PhenomD fits, remnant and QNM calibration, and the project's N61
QNM refit retain their own provenance and domain limits. The subsequent
[convention-bridge receipt](../validation/imrphenomhm_sxs_conventions_20260927.md)
extends the public complex check to finite `f_ref`; it does not change the
original run's scope.

## Reproduce the checks

With both frontends and reference dependencies installed, run
`JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python -m pytest -q
tests/test_imrphenomhm_intrinsic_modes_math_reference.py
--reference-device=cpu -ra`. Strict math-reference CI discovers this file.
Review the exact variant and supplied-input contract before treating a pass
as evidence for another option tuple.
