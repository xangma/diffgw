# Review of the Pass 36 stopping conclusion

Date: 2026-09-13. Scope: review and reproducible diagnostics; no runtime or
acceptance-test changes.

The reported 11/21 catalog and 4/13 holdout counts are credible. The conclusion
that all further numerical progress requires guessing or oracle fitting is
not supported. The repository mixes demonstrated defects, unfinished model
implementations, unresolved compatibility conventions, and potentially missing
calibration data under one “evidence blocked” label. These require different
actions. Full parity has not been demonstrated, but stopping is premature.

1. **TEOBResumS has actionable construction errors.**

   In `src/torchwave/approximants/teobresums.py:639–670`, the trajectory uses
   Hamiltonian angular velocity but the amplitude substitutes `pphi/r²`.
   These are different quantities in this relativistic Hamiltonian. At
   equal mass, zero spin, `r=10, pphi=sqrt(10), pr*=0`, an independent review
   probe found 0.0274077186 versus 0.0316227766, a 15.38% difference.

   Lines 680–684 FFT `A exp(-2i phi)` and keep positive-frequency bins,
   although this carrier rotates at negative frequency under the chosen
   transform. Lines 722–731 produce plus and cross as real multiples of the
   same spectrum. The saved probe reproduces nonzero **hp == hc exactly at
   face-on inclination**. The existing EOB `_quadrupole_fd` at
   `eob/waveform.py:369` already projects real polarizations before their
   Fourier transforms, with the spherical-harmonic normalization.

   The default 1,500 steps of 0.5M span only **0.0103435 seconds** for the
   frozen 2.8-solar-mass case starting at 30 Hz. A short fragment is being
   compared with a complete waveform. That catalog case has zero spins and
   zero tides, so missing tidal terms cannot explain its failure.

   **Required:** consistent Hamiltonian frequency, correct mode projection
   and Fourier construction, physical integration termination, numerical
   convergence, then complete the published radiation/NQC/merger model.
   Fixing the construction alone will not make this prototype TEOBResumS.

2. **The ROM's calibration data exist; its evaluator is absent.**

   `rom/surrogate.py:148` evaluates the bundled polynomial/Chebyshev surrogate
   of an earlier native prototype. It is not an evaluator for the public
   SEOBNRv4 ROM. `rom/official.py` validates the official file's schema/hash
   but stops over its two extra coefficient planes.

   The 333.3 MB `SEOBNRv4ROM_v3.0.hdf5` is available in the official
   [waveform data release](https://zenodo.org/records/14999310). The
   [GSL spline specification](https://www.gnu.org/software/gsl/doc/html/bspline.html)
   gives `ncontrol = nbreak + order - 2`; cubic splines therefore have
   **nbreak + 2 coefficients**. The observed dimensions are consistent with
   ordinary cubic B-splines, not necessarily unexplained padding. This is a
   mathematical lead, not proof of tensor ordering, endpoint knots, submodel
   stitching, or phase gauge.

   **Required:** independent tensor-product spline evaluator, basis
   reconstruction, amplitude scaling, frequency interpolation, submodel
   transitions and phase/time conventions. Validate interiors, boundaries,
   and transitions. Resolve narrowly missing asset metadata and data-use
   provenance separately. This path need not wait for a new native EOB
   training generator or new NR simulations.

3. **Precession compares different defaults and lacks a required remnant path.**

   On the installed LAL 7.7.0 / LALSimulation 6.2.0, public dictionary
   lookups return precession 300, final-spin 4, convention 1. Binary calls
   on the tested catalog inputs give exactly equal outputs for default
   versus explicit precession 223, and default versus explicit final-spin 3.
   This is finite-case equivalence, not an assertion about internal dispatch
   or every parameter value. Pass 12 had already documented corresponding
   joint-option controls on all four fixed XP/XPHM cases.

   TorchWave defaults to NNLO102 and rejects final-spin 3; its MSA path uses
   the aligned carrier remnant. The main harness forwards an explicit
   `prec_version`, but does not forward `final_spin_mod` or `convention` to
   LAL. They therefore need dedicated paired option controls.

   Fresh XP worst plus/cross relative L2 is 0.0725486 with omitted options,
   0.0611688 with precession 102 on both sides, and 0.0331144 with 223 on
   both sides. All still fail. These are diagnostic controls, not replacement
   acceptance cases; setting the angle option alone does not fix the remnant.

   **Required:** finish the documented MSA/remnant/frame combination and its
   version selection, then validate Euler angles, carrier remnant, reference
   rotations, and projection separately. The
   [precession paper](https://arxiv.org/html/2004.06503v2) specifies version
   families and conventions. Resolve its initial/final-mass ambiguity using
   dimensional consistency, aligned limits and, where necessary, author
   clarification. Develop the generic unequal-mass branch while separately
   specifying singular limits; edge cases do not justify abandoning all
   ordinary cases. Never replace missing metadata with fitted residuals.

4. **Several residual classifications claim more than their evidence proves.**

   - PhenomHM: Pass 35 calls discrepancies a consequence of the model's
     approximate mapping and missing mixing, while also acknowledging that
     the target PhenomHM model omits mixing. A model's approximation error
     against NR does not explain disagreement between implementations of
     that same model. The [PhenomHM paper](https://arxiv.org/abs/1708.00404)
     describes an untuned mapping construction. Recheck frequency mappings,
     amplitude rescaling, QNM choices, joins, and per-mode phase conventions;
     adding new mixing physics would change the target model.
   - XPHM: a 22-only *precessed polarization* still contains Euler rotations
     and projection. Pass 33's output does not isolate a coprecessing carrier
     and cannot rule out a rotation error. First pair all reference options,
     then validate nonprecessing XHM modes and rotation separately.
   - PhenomD: smooth residuals do not establish coefficient precision as the
     cause. Runtime already imports unrounded coefficients from
     `phenomd_unrounded_fits.py`, whose provenance remains under review.
     Isolate fit evaluation, remnant mass/spin, QNM values/interpolation,
     amplitude/phase joins and reference anchoring. An independently more
     accurate Kerr solver need not reproduce an older interpolant to 1e-6.
   - NSBH: Pass 35 records magnitude L2 of 7.77e-8 and 4.60e-9, versus
     complex L2 of approximately 2.965e-5 and 1.758e-3. The immediate
     discrepancy is phase dominated. Isolate the PhenomD phase baseline,
     `_legacy_phase_offsets`, reference-frequency handling and tidal phase
     before claiming that higher-precision amplitude calibration is required.

5. **SEOBNR_ODE needs a completed driver, not another prototype comparison.**

   `eob/waveform.py:87` defaults all three model selections to `prototype`;
   it uses 1,500 steps and can splice at the final sample without reaching an
   orbital-frequency maximum. The substantially more complete assembly in
   `eob/published_waveform.py` is not the frozen catalog path. Its recorded
   nonspinning refinement difference, 1.0304643e-4, is also above the target.

   **Required:** complete and certify the nonspinning published driver first
   (the catalog is nonspinning), then the calibrated spinning model. Establish
   initial conditions, full flux, physical termination, NQC, ringdown
   attachment, time origin, tapering and the same public TD-to-FD conditioning
   contract. Budget numerical error comfortably below 1e-6, rather than
   spending the entire gate on integration/interpolation error. Route the
   compatible implementation into the normal API once validated.

6. **Kerr needs a precise reference adapter.**

   Pass 30 already obtained a damped single-mode waveform from public
   `SimBlackHoleRingdown`; frequency and damping agree. The unresolved
   quantities are energy/amplitude normalization, angular normalization and
   phase, plus the transform adapter. The frozen single-mode problem does
   not automatically require new overtone or merger-excitation data.

   **Required:** freeze the public target API and all physical/transform
   inputs; compare QNM, spheroidal angular function and energy normalization
   independently; derive the finite sampled transform and complete the
   adapter. Document any genuinely missing normalization convention precisely.
   Do not turn an inclination-dependent ratio into a fitted correction.

Recommended execution order:

1. Preserve the existing gates and freeze binary versions, data hashes, model
   options, mode selection and conditioning. Add explicit paired diagnostic
   configurations without redefining the default acceptance contract.
2. Fix TEOB's demonstrable construction errors; start the independent official
   ROM evaluator. These have concrete implementation deliverables now.
3. Close shared PhenomD/NSBH phase and QNM discrepancies; then HM/Pv2 and
   XHM/XP components before their XPHM assembly. Finish one component with
   external evidence instead of repeatedly expanding audit reports.
4. Complete converged EOB drivers and the Kerr reference adapter. For each
   remaining blocker name the exact missing equation, number, or convention,
   demonstrate sensitivity, and identify a specific source or author question.
5. Re-run the unchanged 21+13 cases, require every specified gate to pass,
   then add fresh, independently selected validation points and boundary
   tests. Repeatedly investigated “holdouts” are now regression cases, not a
   substitute for unseen validation. TaylorF2_HM's current dominant-mode-only
   verification must remain explicitly scoped when claiming catalog coverage.

Clean-room validation permits checking independently specified alternatives
against a public binary; it does not require refusing every such experiment.
Continuous coefficient fitting, undocumented correction curves, or changing
acceptance cases to match the implementation would not establish the requested
parity. If a required coefficient is truly available only in prohibited source,
and oracle fitting remains excluded, obtain an admissible author release or
specification. Exact parity under those constraints cannot be promised merely
because the model's equations are public.

Verification and provenance:

- Fresh focused run: `PYTHONPATH=src:tests PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest -q
  tests/test_parity_harness_controls.py tests/test_lal_parity.py`:
  **64 passed in 42.14 seconds**. These are controls, not 64 external parity
  successes. This review did not rerun the entire 21+13 suite.
- [probe.py](probe.py) and [probe-results.json](probe-results.json) retain the
  fresh binary comparisons and TEOB polarization/duration reproduction.
- No runtime changes, oracle-fitted constants, or NR runs were made.
- The inherited unrounded PhenomD coefficients explicitly lack resolved
  independent provenance. The legacy local specification
  `docs/reviews/2026-09-11-clean-room-spec/parity-calibration-spec.md` also
  contains implementation-derived descriptions. A pass-specific assertion
  about sources read does not certify the entire repository's provenance.
- Review exposure disclosure: I encountered that legacy specification, and
  one web search unexpectedly returned LAL implementation snippets. I did
  not open the corresponding source pages or use those snippets to prescribe
  changes. This review is not a certificate of source-isolated authorship;
  strict clean-room implementation should proceed from independently
  admissible specifications with unexposed implementers where required.

Implementation-period exposure addendum (2026-09-13): another general web
search unexpectedly returned LAL and a third-party waveform implementation
snippet despite negative search terms. The coordinator did not open those
source pages or pass their contents to the clean implementers. Subsequent
publication searches use an explicit primary-publication domain allowlist.
New mathematical implementations require independently recorded paper
equations or public black-box component contracts. This disclosure does not
certify unresolved inherited code or calibration provenance.

Exposure addendum (2026-09-14): a later broad search again returned an
unsolicited third-party waveform implementation snippet and LAL API
documentation results. The coordinator did not open the implementation page,
copy its content, or pass it to the implementers. The version-102 beta
hypothesis (omission of the published pi-squared term in the 3PN angular
momentum) and its independent black-box verification preceded that search.
The previous statement about consistently using a publication allowlist was
therefore too broad. Further searches must use primary-publication domain
allowlists. Implementer isolation does not resolve the coordinator's exposure
or the separate provenance questions for inherited code and data.

Further exposure disclosure (2026-09-14): a coefficient-source search used
`site:` query terms without the required tool-level domain filter. Its results
included a LAL D API reference, an unsolicited LAL THM implementation snippet,
and third-party waveform references. The coordinator did not open those
implementation pages or pass their contents to the Luna implementers. No THM
implementation work followed. Subsequent searches use the tool's explicit
`domains` filter for primary papers and author data. This records an additional
search-control failure; it does not erase or certify the coordinator's exposure.

User-authorized reference audit (2026-09-14): the user subsequently requested
cross-checking LAL and handing the findings to clean-room Luna implementers.
The coordinator now deliberately inspects the reference implementation as
well as its public binary. The audit is pinned to the installed LALSuite
commit `c90e1175f5e238741185744a4e516e58d42a9aec` (v7.26.1; lal 7.7.0,
lalsimulation 6.2.0). Source downloads and the exposure manifest stay outside
the repository in `/tmp/torchwave-reference-audit-root-20260914/`.

Implementers receive fresh contexts or continue already isolated contexts.
They receive mathematical/interface contracts, primary-paper references and
public-binary validation cases; they do not receive source excerpts, source
tables, the coordinator's transcript or access instructions for the audit
cache. This is a deliberate separation of auditor and implementer roles,
not a claim that the coordinator remains source-unexposed. The
[sanitized contracts](reference-audit-contracts.md) record the new handoffs.
Unresolved inherited calibration provenance and external-data terms remain
separate from numerical compatibility and MIT package metadata.

The subsequent TD-to-FD audit added the pinned generator conditioning,
legacy generator, inspiral interface, and Butterworth filtering sources to
the external audit cache. The EOB support audit also inspected two pinned
headers and the adaptive-integration wrapper. The TEOB driver audit added its
top-level generator and option header; the download manifest now records 29 files;
that count excludes incidental web source snippets already disclosed above.
The implementer-facing result is the [functional conditioning contract](td-fd-conditioning-contract.md),
with independent biquad derivation and public filter/FFT checks. Reference
source and its download manifest remain outside torchwave.
The driver handoff is the [independent adaptive EOB specification](eob-adaptive-driver-contract.md),
which separates accepted adaptive states, uniform natural-cubic resampling,
termination events and the near-merger restart. It supplies no integrator code.
The [TEOB PA handoff](teob-postadiabatic-contract.md) specializes published
post-adiabatic equations and states polynomial interpolation requirements;
it supplies no reference code or calibration tables.
The [Hamiltonian derivative handoff](eob-numerical-derivative-contract.md)
uses the public numerical-differentiation manual, independent polynomial
derivation and the canonical Cartesian chain rule. The coordinator inspected
the reference derivative wrapper, but no GSL implementation source; Luna
receives neither that wrapper nor reference source text.

The coordinator subsequently inspected the cached TEOB accepted-state event
and post-event stepping rules, including its constant 0.1M cap. The
[continuation handoff](teob-adaptive-continuation-contract.md) expresses those
observable numerical requirements without reference implementation text.
The cached SEOB attachment phase-lifting loop was also inspected; independent
FP64 representations were compared only in a coordinator diagnostic, with
no resulting runtime edit. The source cache still contains 29 files.

Further TEOB audit (2026-09-14): the coordinator inspected the cached
auxiliary-merger sampling, discrete NQC solve, series join, numerical peak
estimate and ringdown attachment. This included exposure to the reference's
NQC calibration tables. Those source-only tables were not passed to Luna or
installed in torchwave. The [new merger contract](teob-merger-assembly-contract.md)
specifies mathematical and sampling behavior, requiring explicit target
provenance. The coordinator separately checked the primary paper's postmerger
equations and handed off its distinct amplitude/phase exponents. The printed
global NQC fit and the reference's default mass-ratio-dependent target family
must not be conflated. A subsequent primary-source check resolved the latter
for `1<=q<4`: [Nagar et al. (2017)](https://arxiv.org/pdf/1703.06814), Table II
for equal masses and Table IV with Eq. (A3) otherwise. The 2018 paper's
Sec. II.B.4 replaces these targets only for `q>=4`. Luna was directed to read
and implement the published tables independently, with a separate provider
and explicit provenance. This supersedes the earlier unavailable-data claim
for that sector.

The coordinator also inspected cached TEOB residual-phase and tail-factor
functions, including the reference's speedy-tail fit tables and digamma list.
Those implementation tables were not passed to Luna or added to runtime.
The separate primary source [Damour et al. (2012), Eq. (23)](https://arxiv.org/pdf/1212.4357v2)
specifies the TEOB tail scale `r0=2GM/sqrt(e)`. Relative to the clean helper's
`2GM` convention this changes the phase by `-m*Hreal*Omega`. That mathematical
factor and the public-binary fixture were handed to Luna. The diagnostic
uses the existing independent gamma-function implementation and introduces
no fitted reference coefficients. The external source cache still contains
29 files.

The subsequent PA attribution audit re-inspected the cached PA quadrature,
flux branches, horizon absorption, and postmerger fits. The coordinator saw
source-only postmerger coefficient precision and nonspinning NQC tables;
neither was transmitted to the implementers. A separate published distinction
is actionable: [Nagar et al. (2018), Sec. II.A](https://arxiv.org/html/1806.01772v2)
specifies leading-order horizon absorption. The zero-spin limit of
[Damour and Nagar (2014), Eqs. (98)–(99)](https://arxiv.org/pdf/1406.6913)
with unit PN amplification gives the normalized horizon ratio
`(X1^4+X2^4)*vphi^8`. A source-isolated worker is comparing that published
limit with the separate resummed prescription, using public waveform outputs
only. No source-only coefficient or reference implementation enters that
diagnostic. The cache file count remains 29.

The coordinator subsequently inspected the cached conservative radial
acceleration approximation and SEOB NQC target precision. Luna receives
independent Hamiltonian derivations and a clearly labeled frozen-mobility
approximation to compare with the exact conservative chain rule, not the
reference expression or its comments. No source-only calibration precision
is passed to the implementers. A root-only sensitivity calculation of the
already disclosed fast-tail polynomial uses its tables solely in the
external audit area; only error magnitudes are reported in
[implementation status](implementation-status.md). It does not supply a
runtime approximation or an admissible coefficient release. The reference
source cache remains at 29 files.
