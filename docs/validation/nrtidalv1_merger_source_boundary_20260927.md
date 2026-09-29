# NRTidal v1 merger-fit source boundary — 27 September 2026

**Decision:** No author-published source checked here supplies the full-precision
merger-fit parameters or an alternative NRTidal v1 variant that identifies
the pinned public binary's nonzero-tide taper frequency. The published Eq. (11)
remains the mathematical reference. The [frozen nonzero-tide binary gate](nrtidalv1_taper_20260927.md)
still fails; this audit does not qualify the fully tapered carrier.

## Sources and scope

- [Dietrich et al. (2018), Eq. (11) and its preceding and following paragraphs](https://arxiv.org/abs/1804.02235)
  print $\omega_0=0.3586$, $n_1=3.354\times10^{-2}$,
  $n_2=4.315\times10^{-5}$, $d_1=7.542\times10^{-2}$, and
  $d_2=2.236\times10^{-4}$. The authors explicitly say that LAL
  substitutes $\kappa_{\rm eff}^T$ for $\kappa_2^T$, and set the Planck
  taper from $f_{\rm mrg}$ to $1.2 f_{\rm mrg}$. The article's [arXiv v1
  source archive](https://arxiv.org/src/1804.02235) has the same printed
  digits in `paper20180406.tex`; its other entries are figures and the
  bibliography, with no coefficient table or companion code. The
  [accepted manuscript](https://link.aps.org/accepted/10.1103/PhysRevD.99.024029)
  also prints the same precision. No separate author or publisher supplement
  with additional digits was found in the checked records; this is a bounded
  search result, not a claim that unpublished fit data do not exist.
- [Bernuzzi et al. (2014), merger-frequency fit](https://arxiv.org/abs/1402.6244)
  gives an earlier, different rational fit for moderate mass ratios. In
  particular, it has no $d_2$ term. It is not a higher-precision version of
  Dietrich et al. (2018), Eq. (11).
- [Dietrich et al. (2019), Eq. (25)](https://arxiv.org/abs/1905.06011)
  refers back to the 2018 merger frequency and $1.2 f_{\rm mrg}$ taper.
  The [later author review, Eq. (89)](https://doi.org/10.1007/s10714-020-02751-6)
  repeats the merger-fit coefficients at the same precision; neither
  supplies the missing digits or defines a separate v1 public-binary fit.

## What the frozen diagnostic establishes

The [public-interface convention diagnostic](nrtidalv1_taper_conventions_20260927.md)
shows that one merger-frequency shift explains the pinned LALSuite 7.26.15
binary's Planck-window modulus at seven taper samples. On five nonzero-tide
systems, the binary-implied frequency is $2.51\times10^{-5}$ to
$4.87\times10^{-5}$ below the printed Eq. (11) fit in relative terms;
the zero-tide system agrees. The $\kappa_2^T$ alternative fails the
unequal-mass cases and cannot explain an equal-mass difference. Rounding
intervals of the printed parameters can contain a discrepancy of this size,
but the binary diagnostic does **not** identify which parameter differs or
establish its unpublished value. No coefficient has been extracted from
LAL source or fitted to its binary output.

## Missing contract and next gate

The missing source contract is either (a) the author-published, sufficiently
precise *joint* tuple $(\omega_0,n_1,n_2,d_1,d_2)$ for the Eq. (11) fit used
by the specified NRTidal v1 variant, together with its coupling convention,
or (b) a published, complete variant definition that fixes the binary's
merger frequency independently. A single inferred $f_{\rm mrg}$, or one
coefficient chosen within a rounding interval, cannot supply that contract.
If such a source becomes available, freeze its equation, domain, cases, and
tolerance before adding a separate scalar/Torch/JAX and public-binary gate.
Until then, keep the source-derived scalar-to-native checks and the failed
nonzero-tide boundary diagnostic distinct; do not change native constants
or relax the frozen tolerance.
