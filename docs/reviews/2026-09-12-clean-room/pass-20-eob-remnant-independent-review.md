# Pass 20 — EOB remnant independent review

**Decision: RETAIN.** The native final-mass equation and Hofmann final-spin
equation/coefficients are transcribed correctly. Keep `fit_order` explicit;
the permitted primary sources do not establish a unique SEOBNRv4 default
block. No runtime or test file was changed.

I independently checked `remnant.py` against the TeX sources for SEOBNRv4
(Bohé et al., arXiv:1611.03703v2), SEOBNRv2/Taracchini et al.
(arXiv:1311.2544), Barausse–Morozova–Rezzolla (arXiv:1206.3803), Hemberger
et al. (arXiv:1305.5991), and Hofmann–Barausse–Rezzolla (arXiv:1605.01938v2)
in `/tmp/pass20`. The direct source-derived check was
`pytest -q tests/test_clean_room_eob_remnant.py`: **59 passed, 4 skipped**.

## Equation audit

* Taracchini's merger-ringdown paragraph selects the Barausse radiated-energy
  interpolation and replaces its equal-mass term with Hemberger Eq. (9),
  \(E_{\rm eq}=0.00258-0.07730/(1.6939+\chi_i)\). With
  \(\nu=m_1m_2/M^2\), Barausse's interpolation is
  \(E_{\rm rad}=\nu(1-4\nu)(1-E_{\rm ISCO}(a))+16\nu^2E_{\rm eq}(\chi_i)\).
  The native
  `1 - nu*(1-4*nu)*(1-energy) - 16*nu**2*equal_mass_radiation` is exactly
  \(M_f/M\). Its `total_spin` and `weighted_spin` reduce respectively to
  \(a=(m_1^2\chi_1+m_2^2\chi_2)/M^2\) and
  \(\chi_i=(m_1^2\chi_1+m_2^2\chi_2)/(m_1^2+m_2^2)\), matching the later
  explicit author formula (Ossokine et al., arXiv:2004.09442, Eq. 3.5).
  The equal-mass limit is reached continuously by the \(16\nu^2\)
  interpolation; no equality branch is present in the source.
* Hofmann Eq. (7), the ISCO expressions, Eq. (11) constraint
  \(\sqrt3/2+\sum_i k_{i0}/4^{2+i}=0.68646\), and all Table 1 entries match
  the native implementation. This includes
  \(\xi=0.41616,0.463926,0.474046\) and the `(1,2)`, `(3,3)`, `(3,4)`
  coefficient blocks; the native Horner evaluation is algebraically the
  printed double sum.
* SEOBNRv4 states only that it uses Hofmann et al.'s final-spin formula. It
  does not name `(n_M,n_J)` or a coefficient block. Hofmann reports all three
  fits and says `(1,2)` appears optimal to avoid overfitting, but this is a
  fit-quality recommendation rather than an explicit SEOBNRv4 selection.

Thus the exact blocker is missing target metadata naming the Hofmann block.
Making `fit_order` optional and choosing `(1,2)` would add an unsupported
calibration convention; retain the required argument and make any future
selection explicit at the caller.
