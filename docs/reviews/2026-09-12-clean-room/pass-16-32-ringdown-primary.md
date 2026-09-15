# Pass 16: primary audit of the IMRPhenomXHM `(3,2)` spheroidal ringdown phase

## Result

The primary paper specifies a four point fit of the **spheroidal phase derivative**.  The four fitted targets are

\[
G^i_{32}=\left.\frac{d\phi^{\rm RD}_{32,S}}{df}\right|_{f=f^i_{32}},\qquad i=1,2,3,4,
\]

not the four coefficients themselves.  The four coefficients solved from them are
\(\alpha^{32}_2,\alpha^{32}_4,\alpha^{32}_\lambda,d\phi^{32}_{\rm RD}\).  Table IX's points are

| target | evaluation frequency in the paper |
|---|---|
| \(G^1_{32}\) | \(f^{22}_{\rm ring}\) |
| \(G^2_{32}\) | \(f^{32}_{\rm ring}-\tfrac32 f^{32}_{\rm damp}\) |
| \(G^3_{32}\) | \(f^{32}_{\rm ring}-\tfrac12 f^{32}_{\rm damp}\) |
| \(G^4_{32}\) | \(f^{32}_{\rm ring}+\tfrac12 f^{32}_{\rm damp}\) |

The current native branch has only three derivative targets: point 1, an extra target at \(f^{32}_{\rm ring}\), and point 4.  It fixes \(\alpha^{32}_2=\alpha^{22}_2\) and solves a 3 by 3 system for the other three coefficients.  That is a deliberate inherited-gap implementation, but it is not the four-target prescription in Table IX.

The official arXiv ancillary directory contains `dphiS_p1.m`, `dphiS_fring32.m`, and `dphiS_p4.m`, plus the two alignment fits `deltaPhiS.m` and `deltadPhiS.m`.  It contains no `dphiS_p2.m` or `dphiS_p3.m`.  Therefore the permitted primary assets do not provide every target required by Table IX.  The missing point-2 and point-3 fits are a hard blocker to an exact clean-room implementation of the published four-point calibration; the extra `fring32` fit cannot be substituted for either published target.

## Primary derivation

The source is Sec. VI (the ringdown model), subsection “Modes with mode-mixing,” in `hmc2.tex`.  Equation `eq:dphiS_ansatz` gives

\[
\frac{d\phi_{32,S}}{df}=\frac{\alpha^{32}_2}{f^2}
 +\frac{\alpha^{32}_4}{f^4}
 +\alpha^{32}_\lambda
 \frac{f^{32}_{\rm damp}}
 {(f^{32}_{\rm damp})^2+(f-f^{32}_{\rm ring})^2}
 +d\phi^{32}_{\rm RD}.
\]

The following paragraph says explicitly that its four free coefficients are determined by the four equations \(\phi_{32,S}'(f^i_{32})=G^i_{32}\), with the targets obtained by parameter-space fits.  Thus the source does not authorize importing \(\alpha^{22}_2\) as a constraint.  The integrated phase is

\[
\phi_{32,S}=-\frac{\alpha^{32}_2}{f}-\frac{\alpha^{32}_4}{3f^3}
 +\alpha^{32}_\lambda\tan^{-1}\!\frac{f-f^{32}_{\rm ring}}{f^{32}_{\rm damp}}
 +d\phi^{32}_{\rm RD}f+\phi^{32}_{\rm RD}.
\]

The paper defines the ring and damping frequencies from the real and absolute imaginary parts of the QNM frequency, \(f_{\rm ring}^{\ell m}=\operatorname{Re}\omega_{\ell m}\) and \(f_{\rm damp}^{\ell m}=|\operatorname{Im}\omega_{\ell m}|\).  All four node formulas use that same frequency coordinate and the same mass-rescaled convention used throughout the frequency-domain model (`Mf` in the paper); the table introduces no 2\(\pi\), mode-number, or physical-Hz conversion.  The derivative targets are values of \(d\phi_{32,S}/df\), not values of \(\phi_{32,S}\), and are evaluated in the spheroidal basis before the spheroidal-to-spherical mixing transformation.

The paper then defines two **additional** parameter-space fits, not replacements for the four \(G^i_{32}\):

\[
\Delta T_{32,S}=\phi'_{32,S}(f_0)-\phi'_{22}(f_0),\qquad
\Delta\phi_{32,S}=\phi_{32,S}(f_1)-\phi_{22}(f_1).
\]

It only says that \(f_0\) and \(f_1\) are suitable frequencies in the ringdown region.  Their exact values are not stated in the paper.  Consequently, the alignment fit files establish that two extra calibrated quantities exist, but the primary text alone does not fully specify their reference-frequency convention.

## Ancillary asset audit

The official `PhaseFits/32` listing is:

* `3Dfit_p1_lm_32.m` through `3Dfit_p6_lm_32.m`, `Lambda.m`;
* `dphiS_p1.m`, `dphiS_fring32.m`, `dphiS_p4.m`;
* `deltaPhiS.m`, `deltadPhiS.m`.

The `3Dfit_p*` files are the ordinary `(3,2)` phase collocation fits and are not the spheroidal ringdown targets: for example, their point-1 and point-4 expressions do not match the separate `dphiS_p1.m` and `dphiS_p4.m` expressions.  The only files whose names and expressions identify the spheroidal ringdown derivative targets cover Table IX points 1 and 4, together with an unlisted `f_ring32` target.  There is no permitted source asset for Table IX points 2 or 3.  `deltaPhiS.m` and `deltadPhiS.m` encode the two alignment quantities and cannot supply the missing derivative targets.

The ancillary files are Wolfram Language mathematical exports, beginning with `(* Created with the Wolfram Language : www.wolfram.com *)`.  No notebook or generated implementation was used in this audit.

## Native comparison after the derivation

In [`imrphenom_xhm.py`](/Users/xangma/repos/torchwave/src/torchwave/approximants/imrphenom_xhm.py:848), the `(3,2)` branch sets `alpha2_32 = alpha2_22`, evaluates `phase_32_s_dphi_p1`, `phase_32_s_dphi_fring32`, and `phase_32_s_dphi_p4`, and solves a 3 by 3 system at

\[
\{f^{22}_{\rm ring},\ f^{32}_{\rm ring},\ f^{32}_{\rm ring}+\tfrac12f^{32}_{\rm damp}\}.
\]

The native alignment uses `deltaPhiS` and `deltadPhiS`; its selected references are `f1 = f_ring22` and `f0 = f_ring22 + f_damp22`.  Those choices are implementation facts, not values supplied by the paper's “suitable reference frequencies” wording.

A source-faithful implementation plan is therefore conditional: obtain or independently reconstruct the missing parameter-space fits for \(G^2_{32}\) and \(G^3_{32}\), evaluate all four targets at the Table IX frequencies, and solve the full 4 by 4 linear system for \((\alpha_2,\alpha_4,\alpha_\lambda,d\phi_{\rm RD})\).  Preserve the two extra alignment fits separately, and do not use `dphiS_fring32` as a proxy for either missing Table IX target.  Until the missing fits and the reference-frequency convention are resolved from an allowed primary asset, changing the runtime would require unsupported assumptions.

## Source record and hashes

Primary records:

* Paper: [arXiv:2001.10914v2](https://arxiv.org/abs/2001.10914v2), revised 2020-11-03; published as Phys. Rev. D 102, 064002, [doi:10.1103/PhysRevD.102.064002](https://doi.org/10.1103/PhysRevD.102.064002).
* TeX source: [arXiv source archive](https://arxiv.org/src/2001.10914v2), local archive SHA-256 `cc05e7dbff7bf0c42a10b7da1f4840eefc55b252168d6bfb085155a7bdc5318e`; `hmc2.tex` SHA-256 `5c7132720db24d9caaa804f53d6961b075996cacee00b3ce0a51865a6dbf107e` (the derivation is at local lines 1153–1198).
* Ancillary README: [ParspaceFits/README](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/README), local SHA-256 `368184fb645eb93dc04963902c61bd37fd6be8be69be1842abed7994c20684f4`.
* Direct asset base: [arXiv ancillary `PhaseFits/32`](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/32/).

Relevant local asset hashes (SHA-256):

* `dphiS_p1.m`: `3c6ad2109a6d85202f794579c88b9b52ebe9b39340e1aa9f02ef4fa1f01b89d7`.
* `dphiS_fring32.m`: `1a1b3519df07ffef8a0a065184a3ce66417895569ba540852877b8f7c4ca3ddf`.
* `dphiS_p4.m`: `c417a56b22950dae88c782f35d4470b22af92073cffe92dc273e8e71c155d082`.
* `deltaPhiS.m`: `e92a6657458934bea6bb8466b97c7cd90a45d57d98eff9b546e7d64804fac772`.
* `deltadPhiS.m`: `07107d0436b316792e0b2cd343b8cce3efa2a96276a26304f6ecbdfb7c975455`.
* Near-miss ordinary phase assets `3Dfit_p2_lm_32.m` and `3Dfit_p3_lm_32.m`: `eb28bab3fdcf6e9c806746a454e82e22421eb2b26594dc071f18c42daea2b9ab` and `e908d8812f7f189c4e21d1612fdf502eb824243cf5ead9d629041f4bf6cb4950`, respectively.

No runtime or test files were changed, and no parity or other tests were run.
