# HM PN truncation: public equation record (2026-09-13)

This note records the primary-paper equations and the remaining interpretation issue for the phrase “1.5PN in f”. It is an equation review only; no implementation or runtime file was changed.

## Primary sources and versions

* London et al., *First higher-multipole model...*, arXiv:1708.00404. The arXiv record says v1 was submitted 2017-08-01, v2 2017-08-03, v3 2017-12-22, and v4 (the last revision) 2018-01-18; the journal reference is PRL 120, 161102 (2018). Source: https://arxiv.org/abs/1708.00404.
* Mishra et al., *Ready-to-use post-Newtonian gravitational waveforms...*, arXiv:1601.05588v1, submitted 2016-01-21; its arXiv record has only v1 and identifies the journal publication as PRD 93, 084054 (published 2016-04-29). Sources: https://arxiv.org/abs/1601.05588 and https://journals.aps.org/prd/abstract/10.1103/PhysRevD.93.084054.

The Mishra arXiv v1 PDF and the published-paper metadata identify the same equation numbering used here. No later arXiv version is listed, so there is no arXiv v1-to-later equation revision to reconcile.

## Equations actually stated by Mishra

Mishra Eq. (10) defines the time-domain mode structure. Their SPA mode equation (11) is

\[
\tilde h_{\ell m}(f)=\frac{M^2}{D_L}\sqrt{\frac{2\eta}{3}}\,\pi\,V_m^{-7/2}
 e^{-i[m\Psi_{\rm SPA}(V_m)+\pi/4]}\,\hat H_{\ell m}(V_m),
\qquad V_m=(2\pi Mf/m)^{1/3}.
\]

Thus the common SPA factor is mode dependent (`V_m^{-7/2}`); it is not part of the dimensionless `\hat H` polynomials.

The relevant dimensionless polynomials in Mishra Eq. (12) are (spin vectors projected along \(\hat L_N\); writing the projections as \(\chi_s,\chi_a\)):

* Eq. (12a), `H22`, starts with `-1` (higher terms are printed through `V_2^4`).
* Eq. (12b):
  \[
  \hat H_{21}=-\frac{\sqrt2}{3}\left[\delta V_1-rac32(\chi_a+\delta\chi_s)V_1^2
  +\delta\left(\frac{335}{672}+\frac{117}{56}\eta\right)V_1^3+O(V_1^4)\right].
  \]
  The published equation also prints the `V_1^4` tail/spin terms, but they are beyond the truncation discussed by London below.
* Eq. (12c):
  \[
  \hat H_{33}=-\frac34\sqrt{\frac57}\,\delta V_3+O(V_3^2).
  \]
* Eq. (12d):
  \[
  \hat H_{32}=-\frac13\sqrt{\frac57}\left[(1-3\eta)V_2^2+4\eta\chi_sV_2^3+O(V_2^4)\right].
  \]
* Eq. (12f):
  \[
  \hat H_{44}=-\frac49\sqrt{\frac{10}{7}}(1-3\eta)V_4^2+O(V_4^4).
  \]
* Eq. (12g):
  \[
  \hat H_{43}=-\frac34\sqrt{\frac3{35}}\,\delta(1-2\eta)V_3^3+O(V_3^4).
  \]

The displayed remainder orders above retain only the terms relevant to London’s restricted HM prescription; Mishra’s full Eq. (12) prints additional nonspin and spin terms for several modes.

## What London explicitly prescribes

London Eq. (3) defines the mapped higher-mode amplitude using a dimensionless PN factor `\beta_{\ell m}` and a mapped 22 amplitude. London Eq. (4) gives the PN factor as

\[
\beta_{\ell m}(f)=
\frac{\hat H_{\ell m}(f_{22}^A)}{\hat H_{22}(f_{22}^A)}
\frac{\hat H_{\ell m}(f)}{\hat H_{\ell m}(2f/m)}.
\]

Immediately after Eq. (4), London states that the restricted PN results use only leading order in `f` for `\ell=|m|`, while for `|m|=\ell-1` they use “1.5PN in f” to approximate the spin dependence. The six HM modes listed in that paper are `(2,2),(3,3),(4,4),(2,1),(3,2),(4,3)`. Therefore the explicitly paper-supported restricted factors are:

| mode class | leading power in Mishra | terms supported by an absolute-through-`V^3` reading of London’s phrase |
|---|---:|---|
| `l=m`: 22, 33, 44 | `V_m^0`, `V_m^1`, `V_m^2` respectively | leading term only: 22 `-1`; 33 `-(3/4)sqrt(5/7) delta V_3`; 44 `-(4/9)sqrt(10/7)(1-3 eta)V_4^2` |
| `m=l-1`: 21 | `V_1` | `V_1`, spin `V_1^2`, and nonspin `V_1^3` terms from Eq. (12b) |
| `m=l-1`: 32 | `V_2^2` | leading `V_2^2` and spin `V_2^3` term from Eq. (12d) |
| `m=l-1`: 43 | `V_3^3` | leading `V_3^3` term from Eq. (12g); no `V_3^4` term |

This table is a transparent truncation of the equations, not an assertion about any undocumented implementation.

## The unresolved wording

“1.5PN in f” can be read in two mathematically different ways:

1. **Absolute frequency-power reading (the literal restricted HM use):** retain powers through `V^3` in the mode factors. This gives 21 through its published `V_1^3` term, 32 through its published `V_2^3` spin term, and 43 only at `V_3^3`. This is the only reading that yields a useful common cutoff and matches London’s explicit contrast with leading-order `l=m` factors.
2. **Relative-to-each-mode-leading reading:** retain three powers of `V_m` beyond each mode’s own leading power. That would require 21 through `V_1^4`, 32 through `V_2^5`, and 43 through `V_3^6`; those terms are not supplied by London’s Eq. (4) prescription and are not the stated restricted factors. Mishra Eq. (12) does print some higher terms (e.g. 21 `V_1^4`), but does not thereby say that London’s HM model uses them.

The primary texts do not define “1.5PN in f” with a separate formal order convention. Accordingly, diagnostic variants should be labeled by the retained powers (for example, `21: V1+V2spin` versus `21: through V1^3`) rather than calling one variant “the” 1.5PN implementation. No coefficient, sign, or truncation should be selected from waveform residuals.

## Analytic limits useful for diagnostics

The equations imply: (i) 21 vanishes for equal masses when `delta=0` unless the `chi_a V_1^2` term is present; (ii) 33 and 43 are odd under exchange through `delta`; (iii) 32 and 44 have nonspin leading factors proportional to `1-3 eta`; and (iv) the 32 spin correction is exactly `4 eta chi_s V_2^3` inside the Eq. (12d) braces. These are equation-level checks, independent of any binary waveform oracle.
