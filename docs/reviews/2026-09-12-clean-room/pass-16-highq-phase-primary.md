# Pass 16: high-\q inspiral phase branch (primary-source audit)

## Scope and source identity

This is a clean-room reading of the v2 paper and its TeX/ancillary material.  I did not use LAL/PyCBC or translated/generated implementations, notebooks, git history, verification outputs, residuals, or public binaries.  The primary paper is [arXiv:2001.10914v2](https://arxiv.org/abs/2001.10914v2); the source archive is [arXiv source v2](https://arxiv.org/src/2001.10914v2), SHA-256 `cc05e7dbff7bf0c42a10b7da1f4840eefc55b252168d6bfb085155a7bdc5318e`.  The local main TeX file `hmc2.tex` has SHA-256 `5c7132720db24d9caaa804f53d6961b075996cacee00b3ce0a51865a6dbf107e`.  The ancillary parameter-fit README has SHA-256 `368184fb645eb93dc04963902c61bd37fd6be8be69be1842abed7994c20684f4`.

## What the paper specifies

The low-frequency scaling is Eq. (4.7), labelled `inspiral_scaling` in the TeX:

\[
\phi_{\ell m}(f)\simeq {m\over2}\phi_{22}\!\left({2f\over m}\right).
\]

The PN relative correction is defined by Eq. (4.8), `lambdaPN_eq`, as

\[
\Lambda_{\ell m}^{\rm PN}(f)=
\arctan\!\left({\Im A_{\ell m}^{\rm PN}\over\Re A_{\ell m}^{\rm PN}}\right),
\]

re-expanded through linear order in frequency.  The measured relative phase is Eq. (4.9), `deltaphiIns`,
`Delta phi_lm^Ins := phi_lm(f) - (m/2) phi_22(2 f/m) ~= Lambda_lm^PN`.  The paper says this approximation degrades at high mass ratio/high spin, so it fits the leading extreme-mass-ratio behavior.

The high-q inspiral ansatz is Eq. (4.10), `phase_ins_ansatz` (TeX lines 752--756):

\[
\phi_{\ell m}(f)={m\over2}\phi_{22}^{X}\!\left({2f\over m}\right)
 +\Lambda_{\ell m}(f)+d\phi^{\rm Ins}_{\ell m}f+\phi^{\rm Ins}_{\ell m}.
\]

Its exact branch selection is Eq. (4.10)'s following `Lambda_lm` definition (TeX lines 758--766):

\[
\Lambda_{\ell m}(f)=
\begin{cases}
\Lambda_{\ell m}^{\rm PN}(f),&q<100,\\
\Lambda_{\ell m}^{\rm fit},&q\ge100.
\end{cases}
\]

The boundary is inclusive on the fit side.  For code using only symmetric mass ratio, the equivalent boundary is
\(\eta\le100/101^2=100/10201\simeq0.009802960494069209\), because
\(\eta=q/(1+q)^2\) is monotone for (q\ge1).  This branch is mode-independent and applies to the calibrated higher modes (21, 32, 33, and 44); it is not a frequency-dependent node choice.

The paper explicitly says the phase-inspiral calibration is needed only at large mass ratio and that the high-q phase fits use \(\hat S\) (TeX lines 430--439), where

\[
\hat S={m_1^2\chi_1+m_2^2\chi_2\over m_1^2+m_2^2}.
\]

Thus the ancillary `S` variable is the rescaled total spin \(\hat S\), with \(\eta\) as the other independent argument.  The generic discussion of a possible \(\Delta\chi\) correction does not appear in the `Lambda.m` expressions: the exported Lambda functions contain only `\[Eta]` and `S` (plus Mathematica fit diagnostics), not `chi1`, `chi2`, or a spin-difference symbol.

The linear coefficient (d\phi^{\rm Ins}_{\ell m}) is not an independently fitted Lambda coefficient: the paper says it is fixed by continuity with the intermediate-region ansatz (TeX line 767).  The additive (\phi^{\rm Ins}_{\ell m}) is fixed by Eq. (4.11), `phase_alignment` (TeX lines 769--774):

\[
\lim_{f\to0}\left(\Psi_{\ell m}-{m\over2}\Psi_{22}\right)
 ={3\pi\over4}\left(1-{m\over2}\right).
\]

This follows from `Psi_lm = 2 pi f t_c - phi_lm - pi/4 + psi_0` and the adopted `psi_0=pi` (TeX lines 556--567).  In particular, once the same scaled 22 carrier is used, the phase constant must cancel the finite (f\to0) value of the selected Lambda term; it is not a free calibrated time/phase offset.  A fitted Lambda that is constant in frequency changes this additive gauge constant, while contributing zero to the phase derivative.

The appendix `app:inspiral_hm` gives the exact carrier rescaling needed when the 22 pseudo-PN expansion is evaluated without a new frequency array (TeX lines 1670--1685):

\[
c_m^i=\left({m\over2}\right)^{(8-i)/3}
 \left(c_{22}^i-d_{22}^i\log{m\over2}\right),\qquad
d_m^i=\left({m\over2}\right)^{(8-i)/3}d_{22}^i.
\]

Evaluating the existing 22 phase at (2f/m) is mathematically equivalent if the same phase convention and coefficients are retained.

## Ancillary fit coverage

The official author directory is [ParspaceFits/PhaseFits](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits).  It contains one `Lambda.m` for each required mode:

| mode | local size | SHA-256 |
|---|---:|---|
| 21 | 59,561 bytes | `d8868c0d038bf69b3c9d3852930b1a8d635a85ff10cce25e7ad9999a280c4906` |
| 32 | 72,583 bytes | `da5e09900972cb53b8ea3e419533a3e9058a8ab80b619b6f3a001b32bfcf6716` |
| 33 | 54,224 bytes | `5e7da0c943cab189cb6a3d49a6db81e8b561ac98654c30e67e84baf334fd5461` |
| 44 | 64,967 bytes | `c701d46ab1c7825582ef870747aec8b55c8113ee05f8416f5b0935891767be2d` |

Direct file URLs are [21/Lambda.m](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/21/Lambda.m), [32/Lambda.m](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/32/Lambda.m), [33/Lambda.m](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/33/Lambda.m), and [44/Lambda.m](https://arxiv.org/src/2001.10914v2/anc/ParspaceFits/PhaseFits/44/Lambda.m).

Each file starts with a Wolfram Language export whose first expanded fit member is a complete ((\eta,S)\) expression.  The leading expressions are, respectively, the full 21 polynomial through (S^3), the full 32 polynomial through (S^3), the 33 rational-in-(S) expression, and the 44 rational-in-\eta plus (S,S^2) expression.  The files also contain replacement rules, standard errors, covariance/diagnostic data, and later hierarchy records.  There is no missing Lambda coefficient for any of 21/32/33/44.  The fact that these are complete fits is independently visible from the complete first expressions and does not require opening the excluded author notebooks.

The ancillary README says that the phase notebook exports the functional forms, but the notebook itself is outside this audit.  Consequently, two metadata details remain unsupported by the permitted primary text/assets: the intended machine-readable selector among the repeated Mathematica fit records, and an explicit validity-domain/error statement for the Lambda surfaces.  The coefficient formulas themselves are present.  The first expanded expression is the defensible import target: after replacing Mathematica `^` by the host power operator, it is algebraically the expression already present in the native `phase_lambda_21`, `phase_lambda_32`, `phase_lambda_33`, and `phase_lambda_44` definitions.  This comparison was made only after the source derivation above.

## Native implementation comparison and plan

The native phase loop currently constructs the intermediate derivative from the five Table-VII values and uses a scaled 22 carrier plus a hard-coded `delta_target` for the inspiral.  The existing `XHMParameterFits.phase_lambda_*` methods are definitions only; no call site selects them.  The optional `experimental_phase_connections` path calls `_pn_phase_lambda` for every mass ratio and therefore does not implement the (q<100)/(q\ge100) branch.  It also supplies only the frequency-linear PN term and does not use the fitted high-q Lambda value or the Eq. (4.11) low-frequency constant gauge.

A source-faithful implementation plan is therefore:

1. Form the physical (q=m_1/m_2) and select PN for (q<100), fit for (q\ge100) (or use the exact \eta threshold above, with the fit inclusive at equality), for each of 21/32/33/44.
2. Use the existing Lambda fit functions with ((\eta,\hat S)) in the fit branch.  Treat them as frequency-independent, so their derivative is zero; retain the linear-in-(f) PN expansion and its derivative in the PN branch.
3. Set (d\phi^{\rm Ins}) from derivative continuity at the inspiral/intermediate join, subtracting the scaled-carrier and selected-\Lambda derivatives.
4. Set (\phi^{\rm Ins}) from Eq. (4.11), which cancels the selected Lambda's finite low-frequency value in the relative (\Psi) gauge.  Remove the hard-coded offset only after this derivation is encoded, since odd-mode phase branches may have a separate \pi ambiguity that must still be reconciled with the paper's tetrad convention.

The high-q coefficient source is complete; the remaining implementation uncertainty is metadata/convention handling (which exported Mathematica record is canonical and the exact PN constant extraction), not absent 21/32/33/44 fit coefficients.  No runtime or test files were modified in this pass.
