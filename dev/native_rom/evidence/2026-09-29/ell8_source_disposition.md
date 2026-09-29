# SEOBNRv4 ell=8 source disposition

This bounded follow-up starts from branch revision `093f44d` and the
[frozen source-closure criterion](../2026-09-27/flux_selector_source_closure_plan.json).
It changes no equations, source-eligibility rule, numerical threshold,
machine-readable result or release status.

## Selected compatibility scope (2026-09-29)

Use the existing explicit `spin_truncation="tabulated"` option for subsequent
compatibility assessments. For `(8,2)`, `(8,4)`, `(8,6)`, and `(8,8)`, this
retains the paper-tabulated residual `rho_8m = 1 + c2*v**2`. The omitted cubic
spin term is a truncation choice, not a claim that its physical coefficient
vanishes. The scalar reference and both frontends already implement this
choice from public equations; no physics implementation change is needed.

The checked [LAL 7.26 residual implementation](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_spin_e_o_b_factorized_waveform_8c_source.html)
and [pySEOBNR commit 4fef6d1](https://github.com/AEI-ACR/pyseobnr/blob/4fef6d1c08f4dd09b660dce39d5d74950b5b2853/pyseobnr/eob/waveform/waveform.pyx#L951-L1000)
omit that term in these residuals. This supports compatibility of the selected
terms with those versions, not independent provenance or full-waveform parity.
Their implementation lineage is recorded below; no implementation code is
copied or supplied as a mathematical specification.

Keep the alternative `generic-leading-ell8` available and all frozen campaign
choices intact. Future compatibility receipts must explicitly record the
selector and source revision. The strict historical source gate remains
**31/35**; no manifest row, numerical tolerance, or qualification claim is
promoted by this decision. Further source searching is paused until new
eligible evidence appears. Do not contact authors or other parties.

### Existing whole-waveform evidence

The [latest release ledger](../2026-09-27/historical_fidelity_release.json)
remains unqualified. Its frozen numerical campaign has final ODE and grid
complex drifts of `8.7485e-6` and `6.0815e-6`, exceeding its `1e-6` budget.
That nonspinning row cannot distinguish the two spin truncations.

The later [native/public FD diagnostic](../2026-09-25/native_public_fd.json)
reports fixed-distance, time/phase-fitted relative L2 differences from
`4.4348e-5` to `0.0040307` across four selected cases. It has no parity
acceptance threshold and does not explicitly pin the ell=8 selector; it is
not a certificate for the selected compatibility scope. Identical-input
conditioning/FFT agreement and fixed-chart gradient passes remain separate
checks. Domain convergence, whole-waveform fidelity, and ROM holdout are
still required before qualification.

### Focused validation on len

At source revision `173d58a0d5aaeae18809f550459a54844e14c129`, the existing
reference/Torch/JAX flux checks completed with **46 passed in 33.70 s** on
2026-09-29. Runtime: Torch `2.14.0+cpu`, JAX `0.11.2`, SymPy `1.14.0`,
pytest `9.1.1`; CPU execution with JAX float64 enabled. These checks cover
explicit selector handling, scalar/frontend agreement, Newtonian and exchange
limits, and JAX batch/JIT/derivative behavior. They test both truncations;
they do not compare against external implementation source or certify a full
waveform.

Host `len`, cwd `/home/xangma/tmp/diffgw-ell8-compatibility-20260929`, created
from `git archive` of that revision. Log: `compatibility-tests.log` in that
directory. Reproduce with the existing tests:

```sh
env PYTHONPATH=src:. JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/xangma/.cache/diffgw-taylorf2-hm-venv/bin/python -m pytest -q \
  tests/test_clean_room_seobnrv4_flux.py \
  tests/test_clean_room_jax_seobnrv4_published_flux.py \
  dev/native_rom/reference/tests/test_published.py::test_factorized_8_even_newtonian_limits_exchange_and_truncation \
  dev/native_rom/reference/tests/test_torch.py::test_factorized_8_even_amplitudes_against_reference
```

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

The initial bounded primary-source pass found no additional published selector or
corrective erratum. Fujita Eq. (23) specifies the factorization, not the
historical SEOBNRv4 truncation. This negative search result is not a proof
that no clarification exists. The research worker encountered unsolicited
LAL snippets in search results, did not open or use them, and is excluded
from subsequent clean-room implementation. No implementation source or
archive was opened in that initial pass; the parent verified Pan's equations
directly. No numerical computation was needed.

## User-authorized GitHub audit (2026-09-29)

The subsequent audit deliberately inspected implementation sources. Both the
parent and its source-audit worker are now source-exposed and must not implement
a resulting physics change. This section is an audit receipt, not an input to
a clean-room implementation worker. No physics code or gate changed.

- [pySEOBNR, pinned commit 4fef6d1](https://github.com/AEI-ACR/pyseobnr/blob/4fef6d1c08f4dd09b660dce39d5d74950b5b2853/pyseobnr/eob/waveform/waveform.pyx#L400)
  explicitly identifies LAL SEOBNRv4HM as the source of older coefficients.
  Its agreement is therefore not independent mathematical evidence.
- [NRPy's older factorized-mode tutorial, pinned commit a32e120](https://github.com/zachetienne/nrpytutorial/blob/a32e120f5642bee00e32e9e04dd8cb4c58ae661c/in_progress-SEOBNR/Tutorial-SEOBNR_Factorized_Modes.ipynb)
  describes a LAL SEOBNRv3 implementation. The audit also found derivative
  notebooks in SEOBNRv5_BOB; these do not supply an independent prescription.
- The historical implementation points to Taracchini 2012's Appendix A.
  Independently reading the [paper, Eqs. (A14a)–(A14h), printed p. 18](https://arxiv.org/pdf/1202.0790v2#page=18)
  confirms explicit ell=8 expressions ending at the quadratic velocity term.
  However, [Taracchini 2014, printed p. 2](https://arxiv.org/pdf/1311.2544#page=2)
  subsequently adopts known test-particle spin effects from Pan, and
  [Bohe 2017, Sec. II B](https://pure.port.ac.uk/ws/portalfiles/portal/13281743/PhysRevD.95.044028.pdf#page=3)
  builds on that later model. Our inference: the earlier table alone does not
  uniquely resolve which ell=8 terms the later adoption retains.
- A tempting explanation based on zeroing test-spin terms in higher-mode
  output does not settle the radiation-reaction question. The
  [versioned LAL 7.26 call path](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_i_m_r_spin_aligned_e_o_b_8c_source.html#l01454)
  distinguishes the RR preparation from later output-mode preparation.
  This is behavioral evidence only, not an eligible formula source.
- The implementation's [T1600383 technical-note pointer](https://dcc.ligo.org/T1600383)
  redirected to authentication during this audit; its public-form URL did
  not return a document. Its contents remain unverified. The accessible
  [SEOBNRv5 technical note T2300060-v2, Sec. III](https://dcc.ligo.org/public/0186/T2300060/002/SEOBNRv5_theory.pdf)
  discusses corrections relative to v4 but supplies no selector for these
  four historical ell=8 RR rows.

Outcome: implementation lineage explains why multiple repositories agree,
but the bounded audit found no new eligible historical selector. The source
gate remains 31/35. No fresh implementation chat was started because there is
no newly justified change to hand off. The actionable remaining source lead
is an accessible, versioned author prescription (potentially the v4 technical
note or a public clarification). Existing numerical and ROM obligations are
independent of this source question.

## Archived, unsent author clarification

Retained as research history only. The user has explicitly prohibited contact;
this draft is not a next action.

Subject: Published specification of the SEOBNRv4 even-m ell=8 flux residuals

We are independently implementing the published SEOBNRv4 mathematics and
would like to resolve one narrow source ambiguity from a public mathematical
prescription independent of implementation source.

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
