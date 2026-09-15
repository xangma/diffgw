# TEOBResumS postmerger public-data check

**Date:** 2026-09-14  
**Scope:** clean-room provenance search only; no runtime or coefficient edits.

## Result

No standalone, higher-precision postmerger coefficient artifact with a
documented data license was found in the focused public sources below.  The
only reproducible coefficient contract located is the finite-decimal data
printed in Nagar et al., arXiv:1806.01772v2, Appendix F.  Consequently this
search supplies no replacement data for the current implementation and does
not resolve the reported approximately `2e-5` postmerger waveform residual.

The current clean implementation is
[`teob_postmerger.py`](../../../../../../src/torchwave/approximants/eob/teob_postmerger.py)
(SHA-256 `7324ff72dc1fd36149f137b9f8eafa68f354fd897ff5d56478ce5a6cf2f8c5f6`).
Its independent coefficients match the Appendix F literals.

## Sources and artifact evidence

* [arXiv record for v2](https://arxiv.org/abs/1806.01772) and [rendered
  Appendix F](https://arxiv.org/html/1806.01772v2) reproduce the merger,
  QNM, and ringdown-shape tables.  Appendix F says the fits use 135 SXS, 5
  BAM (`q=18`), and test-particle data, and that the QNM fits were informed by
  public Berti tables.  It does not link a coefficient table, fit-data
  archive, or supplemental numeric file.
* The [v2 TeX source bundle](https://arxiv.org/src/1806.01772v2), downloaded
  from `https://export.arxiv.org/e-print/1806.01772v2`, has SHA-256
  `e2271ae2dac01b123e2d7146b784751517fcfdef9b9238df4888f308d1da49da`.
  Its extracted `paper20181016.tex` has SHA-256
  `90975ac411f92357595333c7ad2edfc5a1b8440f255c9b1d3735071af92d1196`.
  The archive contains the TeX, bibliography, and figures only; no
  ancillary coefficient/data file or data-license file.  Appendix F in this
  source contains the same finite decimal literals as the HTML and the clean
  module, with no additional digits.
* The paper's implementation pointer is
  [`bitbucket.org/account/user/eob_ihes/projects/EOB`](https://bitbucket.org/account/user/eob_ihes/projects/EOB).
  It is a code pointer, not a standalone postmerger-data release, so it is
  not used as coefficient or licensing evidence.
* The related [Zenodo GW190521 release](https://zenodo.org/records/7081337)
  was checked as an author-associated data result.  Its description and
  `data_gw190521_dyn.zip` concern posterior samples for arXiv:2106.05575 and
  refer to TEOBResumS code; they do not publish the 1806.01772 fit data or a
  license for those coefficients.  No author or Zenodo search result located
  a separate postmerger-fit data record.
* The article's [arXiv non-exclusive distribution license](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
  grants arXiv permission to distribute the article.  arXiv's [license
  guidance](https://info.arxiv.org/help/license/index.html) describes this as
  limited article-distribution rights and distinguishes it from a reusable
  data license.  No standalone data license was found.

## Which published values are precision-limited

All finite decimal entries in Appendix F Tables 7--9 are precision-limited
published fit values; the source does not identify hidden digits.  This
includes:

* **Table 7:** every merger-amplitude and merger-frequency coefficient: the
  three orbital-in-`nu` terms and the `n_(nu=1/4)`, `d_(nu=1/4)`, `n_1`,
  `n_2`, `d_1`, and `d_2` spin-fit terms for each quantity.
* **Table 8:** all seven rational-fit coefficients
  `Y0,b1,b2,b3,c1,c2,c3` for each of
  `Y = omega_1^22`, `alpha_1^22`, and `alpha_21^22` (21 literals total).
* **Table 9:** every `b_j^Y` literal for
  `Y = c3^A,c3^phi,c4^phi`, including the explicitly printed zero terms.
  These are the global shape fits in `nu`, `X_AB`, and the spin variable;
  the paper states that `c3^A` uses `a_eff` instead of `S`.

The constrained quantities `c2^A`, `c4^A`, `c1^A`, `c1^phi`, and `c2^phi`
are equations (76)--(80), not additional tabulated data.  The clean module
uses the nonspinning branches of Tables 7 and 9 and all three Table 8 fits.
Recovering more digits from the cited Berti/NR inputs would require an
independent refit or an author-supplied artifact; neither is documented by
the sources checked and neither was attempted.
