# SEOBNRv4 ell=8 source disposition

This bounded follow-up starts from branch revision `093f44d` and the
[frozen source-closure criterion](../2026-09-27/flux_selector_source_closure_plan.json).
It changes no equations, source-eligibility rule, numerical threshold,
machine-readable result or release status.

## Two distinct questions

1. **Mathematical coefficient:** the prior
   [source audit](../2026-09-27/flux_selector_source_closure_audit.json)
   records Pan's generic even-parity leading-spin relation. Specifically,
   [Appendix F.2, Eqs. (F38)–(F39), printed p. 36](https://arxiv.org/pdf/1006.0431v2#page=36)
   gives `f_spin = -2*ell*q*v^3/3` and `rho_spin = f_spin/ell` in the
   nonspinning-test-particle limit about a spinning primary. Substitution of
   `ell=8` gives `-16*q*v^3/3` and `-2*q*v^3/3` for the four even-m modes.
   This derivation uses the paper alone; the notebook is unnecessary for the
   generic coefficient. The omission from the four tabulated ell=8 residuals
   remains a separate truncation question. The separately
   hosted Fujita notebook supports inclusion; its reuse terms remain
   unspecified. Article-level licensing is not assumed to cover that notebook.
2. **Historical model selection:** establishing a coefficient in the analytic
   test-particle waveform does not establish that SEOBNRv4 retained it in its
   truncated radiation-reaction flux. A notebook license alone would therefore
   not close historical fidelity. This needs a versioned model prescription.

The [saved exact result](../2026-09-27/flux_selector_source_closure.json)
remains 31/35 eligible source rows. No newly established source selector is
claimed by this follow-up. Binary agreement, a preferred NR score, and
independently reproducing the universal leading term cannot substitute for
the frozen model-selection evidence.

The bounded primary-source pass found no additional published selector or
corrective erratum. Fujita Eq. (23) specifies the factorization, not the
historical SEOBNRv4 truncation. This negative search result is not a proof
that no clarification exists. The research worker encountered unsolicited
LAL snippets in search results, did not open or use them, and is excluded
from subsequent clean-room implementation. No implementation source or
archive was opened in this follow-up; the parent verified Pan's equations
directly. No numerical computation was needed.

## Unsent author clarification

Subject: Published specification of the SEOBNRv4 even-m ell=8 flux residuals

We are independently implementing the published SEOBNRv4 mathematics and
would like to resolve one narrow source ambiguity without consulting its
implementation source.

For `(ell,m)=(8,2),(8,4),(8,6),(8,8)`, Pan et al.
([arXiv:1006.0431v2](https://arxiv.org/abs/1006.0431v2)) give a generic
even-parity leading-spin contribution in Eqs. (F38)–(F39), while their tabulated ell=8 residuals
omit it. Fujita's later paper
([arXiv:1412.5689v2](https://arxiv.org/abs/1412.5689v2), Eq. 23)
uses the residual-amplitude factorization and links analytic expressions.
The SEOBNRv4 source chain
([Taracchini et al.](https://arxiv.org/abs/1311.2544),
[Bohe et al.](https://arxiv.org/abs/1611.03703v3)) leaves us uncertain about
the retained truncation for these four modes.

Could you provide or point to an independently documented public prescription
that answers the following?

- Are the missing leading-spin terms in Pan's ell=8 tables an intentional
  truncation or an omission? Please state the coefficient and its placement
  in `rho_lm` or `f_lm`, with the spin and velocity definitions.
- Which choice is retained in **SEOBNRv4's radiation-reaction flux**, for each
  of these four modes? Please distinguish this from the output waveform and
  from later model versions, and identify any spin substitution or switch.
- Can the clarification be publicly archived and cited as mathematical
  specification for an independent implementation? If a notebook is the
  intended specification, please identify its version and explicit reuse terms.

We are seeking equations and model definitions, not implementation excerpts.

This is a draft only; no author was contacted.

## Resume and stopping rules

Resume this particular source gate only when a new eligible publication,
erratum, author clarification or explicitly licensed author artifact supplies
the missing selection. Review the mathematical and historical questions
separately; neither automatically answers the other. Give any subsequent
implementation to a fresh worker with primary equations and acceptance tests,
without source-code excerpts or implementation-derived recipes.

Without that evidence, retain the unresolved gate rather than repeat archive
searches, inspect runtime behavior to choose a formula, or tune the flux.
Other independently justified work may proceed, but the native generator
remains outside the qualified release surface. The frozen whole-path JAX
gradient gate stays passed in its declared scope; numerical convergence,
whole-waveform fidelity and ROM holdout remain separate obligations.
