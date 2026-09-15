# Pass 19 independent review: TaylorF2_HM

Date: 2026-09-13

**Disposition: RETAIN the primary report and its no-patch blocker.**  The
report's central conclusion is supported: the permitted papers and the
official Henry asset do not select one amplitude truncation, one stored-mode
phase convention, or one complete coefficient translation for this API.
Adding a coefficient or changing the phase constant would therefore be an
unjustified model choice.  I inspected only
`docs/reviews/2026-09-12-clean-room/pass-19-taylorf2-hm-primary.md`, the
native module `src/torchwave/approximants/taylorf2_hm.py`, its direct base
amplitude/phase implementation, and `tests/test_taylorf2_hm.py`.

## Independent source checks

The report's source ledger is reproducible and its quoted equations have the
right implications:

* Arun et al., [arXiv:0810.5336](https://arxiv.org/abs/0810.5336), Eqs.
  (6.7)--(6.14), sums over harmonics and defines
  `V_k=[2 pi M (f_k/k)]^(1/3)` with `f_k=kF`.  The SPA term is evaluated at
  `f/k` and includes the common stationary-phase `-pi/4`.
* Kidder, [arXiv:0710.0614](https://arxiv.org/abs/0710.0614), and Arun give
  mode-dependent PN amplitude corrections and spin terms; the official Henry
  v2 asset ([arXiv:2210.15602](https://arxiv.org/abs/2210.15602)) contains
  broader aligned-spin mode expressions, including `(3,2)`.  Thus the four
  native leading ratios are a declared approximation, not the uniquely
  implied published model.
* The non-precessing relation
  `h^(l,-m)=(-1)^l conjugate(h^(l,m))` and the
  `h=h_+-i h_x=sum h^(lm) {}_-2Y_lm` convention support the report's
  observer/symmetry discussion.  They do not by themselves identify the
  native `h_(l,-m)` storage sign together with `exp(-i psi_lm)` and its
  `3 pi/4 (1-m/2)` offset.

The report also correctly identifies a physical support boundary: the
permitted higher-mode amplitude sources contain aligned-spin terms, while no
audited source supplies higher-mode tidal amplitude corrections.  Passing
lambda values only to the base TaylorF2 phase cannot justify adding such
terms.  Likewise, `f_cut,m=(m/2) f_ISCO_22` is algebraically the harmonic
conversion if `f_ISCO_22=2F_ISCO`, but the ISCO and roll-off policy is an API
choice rather than a TaylorF2_HM equation.

## Native checks and residual uncertainty

The native module supports only `(2,2),(2,1),(3,3),(4,4)`, uses
`amp0=TaylorF2.evaluate_amplitude(f,...)`, then one shared
`v22=(pi M f)^(1/3)` for every mode.  This deserves an explicit caution in
the primary report: Arun's Eq. (6.8) requires the harmonic orbital variable
`V_m=(2 pi M f/m)^(1/3)`.  The report does not derive that the native
mode-ratio constants absorb all of the resulting `m` dependence.  Therefore
the native reuse of `v22` and of the quadrupole-frequency `amp0` is a
source-relevant scaling question for `m != 2`; it is not safe to correct by
simply replacing `v22` with `V_m` without re-deriving the complete SPA
normalization and coefficient set.

The native phase mapping (`f_scaled=(2/m)f`, then `(m/2)*psi_base`) is
consistent with the frequency argument in the published harmonic sum at a
formal level.  The source equations do not establish that its additional
offset and negative-m storage are equivalent to the native projection, so
the report is right to retain this as convention metadata rather than patch
it.  The same applies to the omitted spin-difference terms in odd modes:
the existing equal-mass test uses zero, equal spins, but it does not test the
source-supported equal-mass/unequal-spin case.

The focused tests are structural smoke tests: 22 equality, zero odd modes
for the default equal-spin equal-mass case, nonzero unequal-mass modes,
cutoff ordering, batching, gradients, and API reachability.  None fixes the
PN amplitude order, harmonic `V_m` normalization, absolute phase, observer
sign, spin odd-mode terms, or tidal amplitude semantics.  No runtime or test
file should be changed from this audit.

## Decision

**RETAIN.**  The source-complete action remains to declare an amplitude
truncation (Arun 2.5PN or Henry 3.5PN), import the complete coefficients,
derive the harmonic-frequency normalization, and specify the stored-mode and
observer convention.  Until those choices are source-backed, the missing
truncation/convention metadata is a real blocker.  No tests were run and no
runtime/test files were edited.
