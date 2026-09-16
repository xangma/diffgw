# Clean-room implementation status

Date: 2026-09-15. This continues the earlier read-only [review](review.md).
Luna agents implement new mathematical components from primary publications;
the coordinating agent reviews their changes and reproduces material results.
The coordinator's prior exposure and subsequent user-authorized reference
source audit are disclosed in that review. Source-isolated implementers now
receive the [sanitized functional contracts](reference-audit-contracts.md),
primary-paper references and public-binary validation cases. Reference source
and source-only tables remain outside their context and the repository.
Existing unresolved runtime implementations are available to those
implementers only as black boxes.

## Acceptance status

The latest completed strict suite records **17/21 catalog and 11/13 holdout**
at the September 15 no-contact baseline. Inventory and manifest-drift
checks passed. Neither the frozen inputs nor the original input files changed
during the 73.85-second execution. See
[catalog](../2026-09-15-no-contact/verification/baseline/catalog.json),
[holdout](../2026-09-15-no-contact/verification/baseline/holdout.json), and the
[source record](../2026-09-15-no-contact/verification/baseline/source-record.json). These are
whole-waveform and detector gates; passing component tests below does not
increase these counts. Previously investigated holdouts are regression inputs,
not unseen validation data.

The remaining frozen failures are XHM and XPHM in both scopes, and SEOBNR_ODE
and TEOBResumS in the catalog. TaylorF2_HM still has unresolved independent
higher-mode oracle coverage despite its diagnostic case passing. Passing a
frozen case does not certify an entire parameter domain or resolve licensing.

The [September 15 no-contact investigation](../2026-09-15-no-contact/README.md)
used three Luna agents and coordinator verification. SEOB public-state
substitution reduces a pre-attachment diagnostic residual by 59.66%; TEOB
tolerance replays show amplification during NQC/attachment processing but no
isolated cause. The checked XHM paper archive does not establish current
release calibration or its admissible redistribution basis. Thirteen focused
tests pass; 139 frozen inputs remain identical to the strict baseline.
No runtime default was changed and the gate count remains **28/34**.

Latest checkpoint cross-links: [xhm calibration inventory](xhm-calibration-critical-path.md),
[hybrid remnant diagnostic](verification/teob-remnant-hybrid-clean/diagnostic.json), and
[SEOB attachment localization](verification/eob-attachment-localization-clean/report.json).
Hybrid remnant is opt-in and improves the BNS catalog from
`6.12424e-5` to `5.68100e-5`, while the full test grows from
`8.27542e-5` to `9.89360e-5`. SEOB first-divergence is unresolved after the
direct-public-grid check; implied NQC target differences are diagnostic only.
The [complete PA waveform comparison](verification/teob-pa-whole-waveform-clean/public-fd-frozen-comparison.json)
shows mixed results. The quadratic initializer lowers 30+20 combined catalog-band
relative L2 from `2.66364e-4` to `1.59913e-4` and full-band error from
`3.30863e-4` to `1.93640e-4`. For 1.4+1.4 it raises catalog-band error from
`6.12424e-5` to `1.52606e-4` and full-band error from `8.27542e-5` to
`2.80396e-4`. These measurements used the former unsimplified quadratic
discriminant and do not describe the subsequently stabilized helper. Root verified
all cache hashes, frequency axes, and polarization/combined relative L2 metrics;
the focused adapter and PA regression run passed 26 tests with one expected
failure at that checkpoint. No canonical promotion is justified; strict gate count remains **28/34**.

## Latest reference-contract results

### Independent TEOB QNM recalibration (experimental)

The frozen Eq. 91 refit uses 17 training and 7 disjoint direct-Leaver holdout
spins over `0 ≤ χ ≤ 0.95`, with unweighted per-quantity least squares. Relative
to the rounded paper coefficients, held-out calibration RMSE improves by factors
90.91 (`ω220`), 54.18 (`α220`), and 12.25 (`α21`). On exact 2,009 bins for
30+20 nonspinning, 20–1024 Hz, `df=0.5`, both polarizations, combined FD L2 is
`3.2708438016984477e-4` baseline, `3.298344651847134e-4` refit (+0.8407876321%),
and `3.3039250070446134e-4` direct (+1.011396672%); refit-vs-direct residual
norm/public norm is `3.4656130901691793e-6`. Fresh public arrays equal the frozen reference bitwise,
axes exact, and symbols restored. Root verified 42 refit/assembly/conditioning/
MIT-packaging tests plus 18 replay tests (60 disjoint). The isolated substitution
does not close parity; no exact-release-fit claim or default promotion is made.
Scope is experimental positive spin only; acceptance counts remain **28/34**,
with no new strict suite. See the
[recalibration conclusion](verification/teob-qnm-recalibration-clean/conclusion.md)
and frozen [calibration](verification/teob-qnm-recalibration-clean/refit-calibration.json)
/[verification](verification/teob-qnm-recalibration-clean/freeze-verification.json).

The latest Luna pass independently audited the clean TEOB continuation and
merger assembly against their reviewed contracts; no convention mismatch was
found. It fixed initial-state validation so complex inputs and nonfinite phases
are rejected before the circular solve, while preserving the Hamiltonian dtype.
Luna also derived a closed `P^1_5` orbital potential with an analytic radial
derivative from the published PN series. The coordinator reviewed its coefficient
equations, dtype/broadcast handling and gradients, including the zero-mass-ratio
limit. The final focused run passed **60 tests**, including MIT packaging.

Two complete 30+20 replays do **not** support promoting the closed potential:
with analytic PA, combined FD relative L2 worsens from `3.27084380e-4` to
`3.36530022e-4`; with direct-quadratic PA, it worsens from `4.15317712e-4` to
`4.43366433e-4`. Both compare 2,009 exact bins against fresh public arrays that
are bitwise equal to the frozen reference. Root independently recomputed the
errors and verified source hashes. A subsequent complex-input guard changed
only validation; its hash and the absence of another waveform replay are
recorded separately. See the [closed-potential result](verification/teob-closed-potential-replay-clean/conclusion.md).
No default or acceptance threshold changed, and no fresh full strict suite was
run: the latest frozen gate count remains **28/34**.

The [primary coefficient inventory](teob-postmerger-precision-inventory.md)
found no extra postmerger/QNM digits in the two inspected primary TeX/ancillary
releases. Root verified their archive hashes and file lists. This bounded search
does not establish that no other author data artifact exists.

Luna implemented the SEOBNRv4 numerical peak-time convention by independently
rounding each published time-offset coefficient to six significant digits
before evaluation. The published-precision helper is retained; only the
explicit numerical compatibility attachment/driver uses the new helper.
The coordinator reviewed the implementation and reproduced **57 passing
tests** across NQC, numerical attachment, the published driver and MIT
packaging, then reran the strengthened two-test precision check. These
overlapping counts must not be summed. See the
[precision audit](seob-nqc-public-precision-audit.md).

The coordinator also reproduced the new
[public NQC diagnostic](verification/eob-first-difference-clean/report.json).
For the saved 30+20 nonspinning fine trajectory, the timing change is
`+6.75008e-6 M`. On 291 strictly pre-attachment samples, the public-mode
relative L2 changes from `5.41750e-5` to `5.51774e-5`; the convention correction
alone slightly worsens this case. The public peak-frequency API separately
differs from the printed target by `-2.52126e-7` in dimensionless GW frequency.
Neither observation uniquely attributes the full residual to calibration,
trajectory, interpolation or timing. Public NQC coefficients are recorded
only as validation observations, never used as runtime inputs.

The subsequent [complete attachment replay](verification/eob-fine-stage-replay-clean/nqc-offset-precision-replay.json)
reconstructs all 18,904 saved attachment samples to relative L2 `4.23e-12`.
On 591 exact shared public-grid samples, the old/new errors are
`6.93279e-5` and `7.07545e-5`. The coordinator reran the diagnostic and checked
its recorded hashes. No time/phase alignment, new trajectory, or fitted
correction was used; the measured worsening is retained explicitly.

The new [TEOB direct-TD diagnostic](verification/teob-first-difference-clean/report.json)
uses the cached 30+20 system's lowered generation frequency and original
20 Hz conditioning request. After applying the declared observer-frame sign,
raw TD relative L2 is `4.08273e-4` at equal elapsed indices. Both series have
4,087 samples, but their physical epochs differ by `-1.50704e-6 s`.
Conditioning the public TD yields combined-polarization relative L2
`1.33913e-4` against cached `SimInspiralFD` on 2,009 exact bins from 20 to
1024 Hz. The coordinator reran the script and independently recomputed this
metric and its input hashes. This comparison alone does not isolate a
conditioner bug: the direct and generic generation inputs must also agree.

The subsequent [public conditioning-stage control](verification/teob-conditioner-public-clean/report.json)
isolates that confound for this case. With identical public TD inputs, the
clean and public conditioning stages agree to relative L2 `2.47e-13` or
better. The public scalar support APIs return a generation start of
`16.900094009839023 Hz`, one binary64 step above the clean plan's
`16.90009400983902 Hz`. Regenerating the public TD at that exact public start
and applying the clean conditioner reduces the combined FD error from
`1.33913e-4` to `2.60412e-13` on the same 2,009 bins. The coordinator reran
the diagnostic, checked its recorded hashes and recomputed the combined
error. This establishes a generation-input discrepancy in the earlier
conditioner comparison; it does not establish native TEOB waveform parity.

Luna then implemented the declared binary64 mass conversions and direct
TaylorT3 frequency inversion in the clean support planner. The 30+20 start
now exactly matches the public scalar APIs. Across six mass/spin controls,
remaining start-frequency differences are zero to three ULP; no numerical
nudge was added. Float32 retains the scaled mass calculation to avoid SI
product overflow. The coordinator reviewed the code, reproduced the
[scalar report](verification/td-fd-support-precision.json), recorded its
[source hashes](verification/td-fd-support-precision-source-record.json), and
passed **44 focused tests**, including gradient, batch/dtype, conditioning,
adapter and MIT packaging checks. See the
[support precision note](td-fd-support-precision-note.md).

The coordinator ran the resulting complete 30+20
[native TEOB replay](verification/teob-native-replay-public-clean/replay-report.json)
in 31.13 seconds. Combined FD relative L2 is `3.2708438016984477e-4` on
2,009 exact bins from 20 to 1024 Hz. Fresh public `SimInspiralFD` arrays
match the saved public reference exactly; all recorded runtime source hashes
remained unchanged, and the coordinator independently recomputed the metric.
The historical native cache gives `3.308629843533685e-4` against that same
reference. Intervening driver changes mean this difference cannot be
attributed solely to the support fix. The native residual remains above the
strict threshold; the experimental adapter is not promoted and the frozen
acceptance count remains **28/34**.

The follow-up [cached TD localization](verification/teob-native-residual-localization-clean/residual-localization-report.json)
compares all 4,087 native/public samples at equal elapsed indices, with
identical generation frequency and sample spacing. Combined TD relative L2
is `2.90745e-4`; the public epoch is `1.09712e-6 s` earlier. Window residuals
are `4.04081e-6` in PA, `1.12695e-4` in the subsequent inspiral,
`1.23586e-3` from the auxiliary NQC grid's start through attachment, and
`1.53975e-3` in the tail. The first sample already differs by relative
envelope `2.75055e-6`. All samples include the final NQC processing, so these
windows do not isolate dynamics from NQC, projection or sampling. The next
comparison must separate those inputs; another conditioner change is not
supported by this evidence. The coordinator reproduced the report exactly,
independently recomputed its window metrics, and verified stable input and
script hashes in the [root verification](verification/teob-native-residual-localization-clean/root-verification.json).
This raw TD comparison is diagnostic, not strict FD acceptance.

The current [exact-start PA-only report](verification/teob-pa-exact-start-clean/report.json), independently checked in its
[root verification](verification/teob-pa-exact-start-clean/root-verification.json), contains 2,637 exact elapsed samples. Before NQC processing, its raw relative L2 is `3.425709e-7` and its amplitude relative L2 is `1.076802e-8`; the root SciPy cross-check is `9.32e-15`. The roughly `1.06472e-3` difference from processed native output is expected NQC processing, rather than an error measurement against the same pipeline.

The [NQC 80-digit roundoff report](verification/teob-nqc-roundoff-clean/diagnostic.json), with [root verification](verification/teob-nqc-roundoff-clean/root-verification.json), rules out arithmetic roundoff at the saved sampled inputs: the coefficient delta is `4.526e-10`, corrected phase delta `4.652e-11`, and corrected amplitude delta `4.270e-15`. It does not rule out differences in NQC targets, input trajectory, or timing.

The [zero-spin PN-spin-order control](verification/teob-zero-spin-policy-clean/report.json) made two fresh public calls with explicit `PNSpinOrder=0`: PA_ONLY=2 returned 2,637 exact-index samples and differs from the current pre-NQC and saved default PA caches by complex relative L2 `6.96765e-4` and `6.96992e-4`; full TD with `TEOB_use_postadiabatic=1` returned 4,089 samples versus the cached default/native 4,087, so TD comparisons were refused without truncation or alignment. This is an attribution control for an alternative public policy, not frozen parity.

The opt-in [direct PA quadratic contract](teob-direct-quadratic-contract.md) was replayed on the exact 30+20 case. Its combined FD relative L2 is `4.153177115176164e-4`, worsening the analytic-initializer full-replay result `3.2708438016984477e-4`; fresh public FD is bitwise equal to the frozen reference, and source hashes remain stable ([replay report](verification/teob-direct-quadratic-replay-clean/replay-report.json), [root verification](verification/teob-direct-quadratic-replay-clean/root-verification.json)). The earlier stabilized-versus-analytic comparison was a near-identity PA diagnostic only, distinct from this complete direct-policy replay. Root verified **43 focused tests**; no fresh full strict suite was run, and no canonical promotion is justified.

The bounded [XHM release search](xhm-data-release-followup.md) did not locate
a complete standalone `122022` calibration artifact in the checked primary
sources. It includes an unsent upstream request for versioned data and schema.
The remaining `(3,2)` condition could require a fourth target or a documented
reduced-system rule; the current inventory does not establish which supplies
the complete release contract. No calibration data or license metadata were
changed in this follow-up, and no new full strict suite was run.

The [external-data workflow](../../external-data.md) supports explicit
download and SHA-256 verification of the official SEOBNRv4ROM v3 asset in a
user cache. ROM loading resolves that cache without downloading during normal
evaluation. No downloaded dataset is bundled, and the upstream data terms
remain separate from the MIT code. A complete public XHM `122022` asset has
not been identified; download support does not supply those missing targets.

Luna implemented the [XHM inverse-quintic intermediate-amplitude join](../../xhm-intermediate-amplitude-derivation.md)
from the primary paper, with caller-supplied normalized amplitudes and
frequency derivatives. It includes interval-wide conservative positivity
checks, batched evaluation and differentiable constraint solving. It supplies
no calibration values or release-specific veto policy and is not routed into
the canonical waveform. The coordinator reviewed the algebra and reproduced
**50 passing tests** covering the new helper, existing XHM inspiral assembly,
external-data resolution and MIT packaging. This set is separate from the
50 EOB/TEOB tests below; neither changes the strict gate count.

The [PA initializer replay](verification/teob-conditioned-waveform-clean/pa-initializer-replay-report.json)
now uses the original cache's recorded `leading-order` horizon policy and
exact PA grid settings. The coordinator reproduced both complete analytic
state/time arrays exactly. The stabilized quadratic initializer differs by
at most `1.07e-12` in state for 30+20 and `8.88e-16` for 1.4+1.4. Earlier
replay differences arose from a mismatched horizon policy and are superseded.
These results do not support changing the initializer to resolve parity.

The [SEOB NQC stencil diagnostic](verification/eob-fine-stage-replay-clean/nqc-stencil-sensitivity.json)
reconstructs the saved NQC products and phase basis within `6.7e-16` and
checks all five corrected matching quantities against the published targets.
The coordinator reran the script and verified its input/dependency hashes.
The largest matching residual is `1.36e-11` on the original fine grid,
`1.59e-7` at stride two, and `2.60e-7` at stride four. Pre-NQC residuals are
expected solve inputs, not unexplained waveform errors. This confirms the
captured solve satisfies its own targets; it does not establish that those
targets or the surrounding waveform match LAL. The coordinator also reran
**50 passing tests** across clean EOB NQC/conservative dynamics and TEOB PA/
circular initialization.

Spark stabilized the opt-in circular initializer using the exact factored
discriminant and analytic zero-spin limit. Root reviewed the algebra and
independently reran **34 passing tests**, including 80-digit references at
positive and negative spin gradients down to `1e-12` and the finite radius
gradient at zero. The lower-root expression still has an absolute-value cusp
with respect to the spin gradient; the full PA diagnostic remains value-only.
This changes roundoff in that diagnostic, not the default analytic PA route.

Luna added the opt-in `frozen-mobility-stage-aware` acceleration convention:
the PA prefix, including its endpoint, uses the squared real-energy factor;
the ODE suffix retains its existing acceleration. Root reviewed the change
and independently reran the PA, initializer, conditioned-adapter and packaging
checks: **38 passed**. The default route and strict acceptance thresholds are
unchanged. These checks overlap the earlier 34-test run.

The [final stage-aware replay](verification/teob-conditioned-waveform-clean/stage-aware-acceleration-report.json)
reproduces the cached baseline FD series exactly in both polarizations and
both systems. Source and NPZ hashes remained unchanged. Root verified those
hashes and independently recomputed baseline errors against the same public
NPZ and frequency masks. On 30+20 the full-band combined relative L2 changes
from `3.039621274922444e-4` to `3.03961792595108e-4`; on 1.4+1.4 it changes
from `8.287892831982565e-5` to `8.287929712031192e-5`. The explicit catalog-band
errors change from `2.663640763123017e-4` to `2.663637438966757e-4` and from
`6.124239392190717e-5` to `6.124369827440565e-5`, respectively. These tiny changes
do not close a gate. Earlier full-band numbers from other comparison reports
must not be used as the baseline for this replay. No canonical promotion is
justified, and no new full strict suite was run.

The [XHM artifact audit](xhm-target-artifact-audit.md) corrects a rejected
provider: the four proposed files were intermediate-phase targets, not
spheroidal ringdown targets. The provider and its test were removed. Root
verified the five reported spheroidal-file hashes against the public archive;
these comprise three derivative targets and two alignment quantities.

The [SEOB native-grid diagnostic](verification/eob-native-stagewise-clean/report.json)
reproduces the full cached waveform error `2.766799454794218e-5`. A direct
fine-grid capture from `result.attachment.fine` now records the returned NQC,
corrected-inspiral, and attached-mode intermediates. Exact native-grid
indexing reproduces the cached native attached mode at relative L2 `0.0`, while
the same samples retain a `6.932790459280097e-5` relative L2 difference from
the public binary. This validates cache/pipeline identity only; it does not
establish attachment accuracy or localize the remaining public residual. See
[fine-stage replay](verification/eob-fine-stage-replay-clean/report.json).

The latest source-directed follow-up corrects the XHM release inventory:
`122022` updates amplitude while retaining the earlier phase model. A fresh
source-isolated Luna implementer derived the four-condition `(3,2)` ringdown
phase solver from the public paper. Root reviewed the implementation and
independently ran its tests with the existing SPA checks: **16 passed**.
The helper is not production-wired: the inherited path has only three
spheroidal targets and fixes one coefficient to the quadrupole value. The
remaining condition or a documented reduced-system rule still needs a
sourced release contract. See the
[precise integration gap](xhm-calibration-critical-path.md).

The separate [SEOB precision audit](seob-nqc-public-precision-audit.md)
includes a reproducible printed-rounding diagnostic. It does not establish
the cause of the waveform residual or supply runtime coefficients. The
coordinator also checked the TEOB default evolution/resampling branch against
the existing auxiliary-grid implementation and found no new discrepancy in
that branch. No production route or license metadata changed in this
follow-up; the latest strict result remains **28/34**, not a new suite run.

The XP production adapter now selects the reviewed clean MSA v223 helper for
its supported default tuple. Its frozen catalog and holdout polarization
relative L2 errors are at most **7.36e-14 and 3.52e-14**. Root independently
reran **40 passing tests** for the adapter, v223 contract and existing
precession interface. A subsequent exact-alignment convention fix applies
paired alpha/gamma offsets while retaining finite cos(beta). Eight additional
exact aligned/zero-spin rows in conventions 0 and 1 now pass with raw errors
at most **1.84e-8**, including independently selected parameter probes. Root
reran **43 passing endpoint/helper/production tests**. See the
[endpoint report](verification/xp-aligned-boundary-clean/report.json).
Tiny in-plane spins still fail at roughly **7.35e-3**, as does the unsupported
extreme anti-aligned-J guard; see the
[production diagnostics](verification/xp-msa-production-clean/README.md).
The exact-endpoint operation does not apply to nonzero transverse spins, and
does not introduce an XAS shortcut or relax a bound.

The separate [release-122022 XHM inspiral adapter](../../../src/torchwave/approximants/xhm_inspiral_assembly_release122022_clean.py)
now implements the mode-scaled MECO boundary, re-expanded PN baseline for
every mode, geometric-strain collocation units and explicitly retained nodes.
It requires caller-supplied calibration values with a provenance record and
does not call the inherited fit provider. Root reviewed the adapter and
independently ran **65 passing tests** across its new contract, the existing
clean XHM equation and assembly helpers, and MIT packaging. Current-release
intermediate/ringdown assembly and admissible calibration inputs remain
separate work; the canonical XHM/XPHM routes have not changed.

The native adaptive SEOB mode now uses the independently implemented public
state-scaled RKF45 controller. On the complete 30+20 solar-mass, zero-spin,
20 Hz, 0.5M-output comparison, this reduces raw complex mode relative L2 from
**1.22960e-4 to 2.766799e-5**. Amplitude relative L2 is **8.58e-7**; the native
and public last nonzero samples both occur at 5114.5M. All recorded dependency
hashes were unchanged. Stable RKF45 cancellation arithmetic is confirmed in
the [diagnostic](verification/teob-rkf45-cancellation-clean/diagnostic.json).
See also the [full-duration result](verification/eob-adaptive-root-review/full20-controller.json).
This is still a strict complex-waveform failure: TEOB RKF45 does not close parity.
Root's controller, batching,
conditioning and numerical-contract rerun passed **48 tests**.

Two frozen-source attribution runs did not close that residual: substituting
the public initial state for diagnosis gives **2.79463e-5**, and reducing
absolute/relative tolerances from 1e-10/1e-9 to 1e-12/1e-11 gives
**1.52390e-4**. See the [initial-state and tolerance comparison](verification/eob-adaptive-root-review/initial-state-and-tolerance.json).
Neither substitution is installed as a runtime policy. A separate
[public numerical-derivative experiment](verification/eob-numerical-root-review/public-derivative-attachment-local.json)
changes orbital frequencies by at most 1.4e-11 near attachment, yet changes
the solved phase coefficients enough to move complex waveform residuals by
about 1e-4. It establishes numerical sensitivity, not a passing derivative
choice. The [FP64 phase-lifting comparison](verification/eob-numerical-root-review/phase-lift-attribution.json)
rules out the two tested equivalent unwrapping representations as a material
cause on these fixtures. PA fixed-node and native-node stencil comparisons are
also non-explanatory for this endpoint mismatch: the roundoff report shows tiny
fixed-vs-native differences and materially larger native-vs-reference endpoint
residuals.


The conditioned native high-mass EOB smoke test now completes after fixing
NamedTuple mode extraction. For 80+60 solar masses, 80 Hz and df=8 Hz, raw
plus/cross relative L2 errors are **2.38949e-4 / 2.39948e-4**; see the
[native FD report](verification/eob-conditioned-root-review/native-highmass-controller.json).
The generic inspiral interface and the direct EOB polarization interface use
different observer frames. The corrected contract preserves the generic
frame. Earlier `public-td-through-native-conditioning.json` was an exploratory
wrong-frame comparison and is superseded by
[public-support-start-conditioning.json](verification/eob-conditioned-root-review/public-support-start-conditioning.json).
With exactly paired public generation-start inputs, native conditioning of
public TD samples agrees with public FD outputs at approximately 1e-14–3e-13.
One-ULP differences in the generation start can perturb the reference's
adaptive waveform materially; the [arithmetic attribution](verification/eob-conditioned-root-review/support-arithmetic-attribution.json)
remains diagnostic, without an oracle-dependent runtime offset.

The independent TEOB PA component now supplies the missing long inspiral
on a descending radial grid, with eight alternating momentum orders and
independently derived polynomial differentiation and quadrature. Its bounded
r=20 to 14 probe takes about 0.067 seconds and has a radial balance residual
below 2e-18; angular balance converges with PA order. See the
[PA verification](verification/teob-postadiabatic-clean/README.md).
Root's PA, vacuum-driver and radiation rerun passed **18 tests**. The
[continuation contract](teob-adaptive-continuation-contract.md) now specifies
the PA seam, accepted-state frequency event, proposed-next-step convention,
and fixed-step support through two total-mass time units after the event.
The implementation now completes this bounded physical trajectory. Root's
independent rerun takes **29.01 seconds**, accepts **579 adaptive and 21 fixed
steps**, reaches the frequency-decrease event at **12316.726873406744M** and
ends at **12318.826873406742M**, with zero phase-seam error and unchanged
dependency hashes. It exactly reproduces the worker's recorded states and
counters; see the [root probe](verification/teob-adaptive-continuation-clean/root-diagnostic.json).
Failure paths retain accepted states and absolute times without extra field
probes. This is a trajectory result, not a complete TEOB waveform.

An independent correction to the published TEOB postmerger ansatz uses
`alpha21` in the phase exponential, retaining `alpha21/2` in the amplitude.
Its merger-frequency error falls from **0.1068159 to 5.55e-17** on the
audited component point. See the [phase report](../../../verification/teob-postmerger-phase-clean/report.json).
Root's combined adaptive, continuation, numerical-contract, postmerger and
derivative regression run passed **66 tests in 2.31 seconds**.
The next [merger assembly contract](teob-merger-assembly-contract.md)
specifies auxiliary sampling, discrete NQC matching, peak interpolation,
ringdown attachment and output sampling. Calibration targets remain explicit
caller inputs. A primary-source citation chain now resolves the default NQC
target family for `1<=q<4`: [Nagar et al. (2017)](https://arxiv.org/pdf/1703.06814),
Table II for equal masses and Table IV with Eq. (A3) otherwise. The 2018
paper's Sec. II.B.4 replaces these fits only for `q>=4`. The published tables
include the equal-mass zero-spin targets at the required printed precision.
Luna implemented that target provider and the independent auxiliary-grid
NQC and ringdown assembly. This supersedes the earlier claim that these
particular target coefficients were unavailable. Root's latest combined
tail, radiation, NQC, ringdown, remnant and packaging rerun passed **62 tests
with one opt-in external-asset skip in 1.67 seconds**.

The earlier [physical NQC cache](verification/teob-merger-nqc-clean/diagnostic.json)
and [ringdown diagnostic](../../../verification/teob-ringdown-assembly-clean/physical-diagnostic.json)
used the generic DIN carrier exposed by `teob_published_driver.teob22_mode`.
That carrier does not use the TEOB non-Keplerian radius or updated residual
phase. Its tail-adjusted full relative L2 of **0.00866844** is therefore an
assembly diagnostic, not a complete TEOB carrier result. Its exact `r=20`
PA grid also differs by one knot from the public inverse-frequency radius.
A new paired run uses the actual TEOB radiation component and the identical
floating-point start radius; the old cache remains preserved and labeled.

A PA-only public-binary comparison also isolated a TEOB tail-phase convention.
[Damour et al. (2012), Eq. (23)](https://arxiv.org/pdf/1212.4357v2) uses
`r0=2GM/sqrt(e)`, while the shared clean tail helper uses `2GM`. Applying the
paper's additional phase `-m*Hreal*Omega` in a root diagnostic reduces raw
complex relative L2 from **0.0286139 to 9.96166e-5** on the rounded-radius
30+20 solar-mass PA probe. Amplitude relative L2 stays **9.79e-7**; the initial
phase error falls to **1.25e-9**, but the final phase residual remains
**2.69e-4 radians**. See the [gauge comparison](verification/teob-pa-public-root/gauge-report.json).
Luna implemented this factor in the TEOB-specific mode path, with checks
for one application and the negative-mode conjugation convention. The
[corrected dense-grid diagnostic](verification/teob-pa-residual-clean/report.json)
regenerates all **19,048** public PA samples exactly and independently
interpolates amplitude and lifted phase. The published leading-order horizon
alternative gives raw relative L2 **8.42212e-5** and amplitude relative L2
**8.18093e-7** on the same grid. It improves the residual modestly and remains
a diagnostic alternative. An earlier double-tail comparison of this branch
was invalid and is superseded by that report. The remaining PA error and
complete waveform still require validation.

The reference's fast-tail approximation is described in the 2018 paper's
Appendix E.2, but its implementation coefficient tables have not been
transferred. A source-exposed root sensitivity calculation finds a maximum
phase difference from the independent exact Gamma function of **4.94e-9**
radians on this PA interval and **1.44e-7** on the older full trajectory.
It cannot explain the observed PA phase drift. No approximation coefficients
were installed or fitted.

The next radiation audit found concrete published terms omitted from the
TEOB table: the finite-mass 3PN octupole corrections in
[Faye, Blanchet and Iyer (2015), Eqs. (64a–b)](https://arxiv.org/html/1409.3546),
the 3PN terms of modes (4,1), (4,3) and (5,5), and the eight 2PN `ell=8`
terms in [Damour et al. (2012), Appendix D](https://arxiv.org/pdf/1212.4357v2).
The earlier table documentation incorrectly described the last group as
ending at 1PN. The 2018 TEOB paper's footnote 17 explicitly confirms the
finite-mass octupole terms belong to its normal model. These are available
mathematical inputs, not missing calibration assets. Luna implemented these
terms only in the TEOB-specific table, preserving the shared DIN helper.
Root verified the displayed paper coefficients and independently ran **30
passing state-adapter, radiation, tail, reaction and packaging tests**.
The frozen radiation hash is
`dca2c9791403467d85572f25a062cbc995d8228743d33b6b2e5dc28c13336bd6`.

The corrected physical assembly now covers all **24,938** public samples
without alignment or support truncation. With completed radiation and the
correct TEOB carrier, current-horizon raw relative L2 is **0.00176358**;
the published leading-order horizon candidate gives **0.00110160**. The
latter's pre-attachment error is dominated by amplitude, whose relative L2
is **0.00110607**. These are full TD diagnostic results, not strict catalog
passes. See the [paired current-horizon report](../../../verification/teob-paired-driver-clean/diagnostic.json)
and [leading-order report](../../../verification/teob-paired-driver-clean/diagnostic-lo-horizon.json).
The subsequent [paired acceleration comparison](verification/teob-radial-acceleration-paired-clean/diagnostic.json)
uses the independently derived frozen-mobility approximation `rddot=C*pdot`
in the NQC amplitude basis. On the leading-order horizon trajectory it lowers
full raw relative L2 to **8.87005e-5**, with amplitude relative L2
**9.97870e-6**. Before attachment, amplitude relative L2 is **7.82991e-7**;
the remaining full error is chiefly phase near merger and ringdown. The
trajectory and calibration targets are unchanged. This remains a diagnostic
policy, not a canonical-route change.
Root's separate acceleration, PA, continuation, NQC, ringdown and conditioning
run passed **62 tests in 3.22 seconds**.

With the completed radiation table, the separate
[PA-only comparison](verification/teob-pa-radiation-completion-clean/report.json)
now agrees with all **19,048** public inspiral samples to raw relative L2
**3.06723e-8** using the published leading-order horizon term. This supersedes
the earlier PA residual quoted above; it does not establish full-waveform
parity.

A [public GSL API experiment](verification/eob-adaptive-root-review/public-embedded-error.json)
isolates cancellation in the clean Fehlberg embedded error. Computing two
complete state estimates and subtracting them makes a time-only component's
error depend spuriously on a constant state offset: the current error becomes
zero at offset `1e6` and about `-3.8e-6` at offset `1e10`. Summing the
independently derived tableau difference weights directly preserves the
`1.39604e-12` error and matches the public API at all four tested offsets.
Luna completed the resulting stable arithmetic check against the complete physical
trajectory; the TEOB report records a post-event `raw_complex.relative_l2` of
**9.26523e-5** for this RKF45 cancellation arithmetic. See
[teob-rkf45-cancellation-clean/diagnostic.json](verification/teob-rkf45-cancellation-clean/diagnostic.json).

Root also isolated the TEOB interface and Fourier conditioning through public
binary calls. Direct and generic TD interfaces return opposite polarization
signs on both tested observer configurations, with cancellation error below
**1.31e-16**. With exactly paired public generation-start inputs, the existing
clean conditioner reproduces public FD polarizations with above-cutoff
relative L2 of **1.03e-14 to 2.60e-13**, including the 1.4+1.4 solar-mass,
zero-tide case. See the [projection contract](verification/teob-conditioned-root-review/public-projection-contract.json)
and [conditioning isolation](verification/teob-conditioned-root-review/public-support-start-conditioning.json).
Public support-bound substitutions remain diagnostics; runtime has no LAL
dependency. A native TD-to-FD conditioned adapter is completed as opt-in,
with public FD comparisons completed in both fullband and catalog-subband bands.
It does not change canonical routes: catalog/holdout canonical status remains
17/21 and 11/13 respectively. The opt-in adapter comparison reports
30+20 fullband combined `relative_l2_unaligned = 1.93640e-4` and catalog-subband
combined `1.59913e-4`; 1.4+1.4 fullband `2.80396e-4` and catalog-subband
`1.52606e-4`. This is an improvement against the frozen comparison only; there is
no production promotion change. Strict catalog/holdout remains 28/34 (unchanged), and the PA initializer is known to have
known-nonfinite gradients on exact zero-G with a strict xfail. See the
[PAwholewaveformcomplete](verification/teob-pa-whole-waveform-clean/public-fd-frozen-comparison.json)
report.

Root also isolated a SEOB attachment sensitivity using public-binary NQC
outputs as diagnostic substitutions: replacing only the phase coefficients
reduces the 0.5M public-state attachment residual from **4.42294e-5** to
**3.29797e-6**. At 1/4096-second spacing the same substitution does not
improve the baseline. This identifies sensitivity rather than a production
fix; no public coefficients enter runtime. See the
[substitution report](verification/eob-numerical-root-review/public-nqc-substitution.json).

The separate [numerical derivative experiment](eob-numerical-derivative-contract.md)
does not close the SEOB residual. On public auxiliary trajectories at 0.5M
and 1/4096-second output spacing, replacing analytic orbital frequencies
with independently differentiated Hamiltonian frequencies gives raw errors
**8.16e-5 and 1.63e-4**, versus **4.42e-5 and 3.21e-6** for the analytic
baseline. Peak/support indices are unchanged. See the
[pointwise report](verification/eob-numerical-derivative-clean/pointwise-report.json).
No runtime derivative policy is selected from this experiment. Root separately
reproduced its **8 passing tests** before the combined rerun above.

The evidence below records earlier implementation stages. Where numerical
results or routing descriptions differ, the completed checkpoint and latest
results above take precedence.

## Implemented and reviewed

| Component | Verified change | Current limit |
| --- | --- | --- |
| PhenomD QNM sampling | Independently solved Kerr roots, cyclic-frequency rounding to six significant digits, and natural cubic interpolation on the audited spin grid | All four original D cases pass: three polarization residuals are below 6.3e-14 and the remaining holdout is 2.23e-7. The certified independent-root sampling domain is bounded; this does not establish provenance for inherited calibration fits. |
| Pv2 composition | Independent source-frame projection, D QNM representation, ten-sample natural-cubic carrier time derivative, and explicit sequence/uniform support | Both original Pv2 gates pass below 6.4e-14. Five additional waveform probes pass below 8e-14; these do not certify every precessing configuration. |
| TEOB waveform construction | Hamiltonian angular velocity; real polarization projection before Fourier transformation; exact Fourier integral of the sampled piecewise-linear signal; fractional event support and explicit budget exhaustion | The default still produces a short prototype trajectory. Complete TEOB dynamics, radiation, NQC and merger assembly are required. |
| Official SEOBNRv4 ROM | Independent cubic tensor B-splines, basis reconstruction, amplitude scaling, published 31-knot phase stitching and phase gauge; canonical ROM routing uses the fresh wrapper | Frozen catalog polarization relative L2 is 1.20e-11 and detector gates pass. Five independent parameter probes are below 3e-11. Official HDF5 data stay external; their distribution terms remain unresolved. |
| Published SEOBNRv4 assembly | Complete 20 Hz nonspinning evolution in the isolated published driver; orbital-peak epoch comparison without optimization | Full-duration 0.5M/0.25M refinement gives 4.07e-5 relative complex difference with quintic interpolation. The finer mode differs from the binary by 8.09e-4; the production prototype is unchanged. |
| NSBH phase and amplitude | Husa BBH mass and Zappa NSBH spin supplied consistently to the D phase and peak slope; positive tidal frequency, continuous small-tide amplitude, plus ringdown window, and outer PN window factor in the mildly disruptive no-torus branch | All 22 expanded strict regression cases pass. These include the original 16 and three additional systems at zero and restored tides; they do not prove parity over the entire parameter domain. |
| Kerr waveform and angular function | Independently derived Leaver recurrence and adaptive quadrature, public-series truncation convention, causal energy-normalized TD/FD waveform, and separate public TD-to-FD Simpson reference | The frozen catalog passes with polarization relative L2 below 9.8e-8. All four expanded public TD-to-FD probes pass the original 1e-6 bound, including spin 0.9. Waveform verification covers the fundamental (2,2) mode only. |
| TEOB factorized (2,2) stage | Published nonspinning mode, consistent 22-only emitted flux and directional mode-frequency derivative | The convergence experiment uses a caller-supplied SEOB Hamiltonian and trajectory. It is not a complete TEOB dynamics or waveform comparison. |
| NRTidal calibration | Replaced the inherited fit module with values and constraints printed in the primary papers, including amplitude denominator 13477.8 | All four affected strict cases still pass. This does not resolve the inherited PhenomD/X calibration provenance. |
| Precession | Fresh NNLO angle, source-frame and Eq. (45) assembly for explicit XP options 102 / convention 0 or 1 / final-spin 0; correct carrier gauge and aligned-spin ISCO | Nine systems in both conventions pass raw public waveform checks below 2e-10 over 20–512 Hz. This does not replace default 223/MSA precession. Its fresh components remain under review. |
| TEOB vacuum radiation | Correct odd-ell-5 mass factor, finite-mass phase truncations, published horizon flux, and full-state radiation wiring for zero-spin, zero-tide systems | Root independently reran 54 passing radiation, driver and related TEOB tests. Full tidal BNS dynamics and merger assembly remain separate work. An older test still assumes the superseded vacuum torque and needs to test the retained tidal fallback instead. |
| TD-to-FD conditioning | Independent support planner, paired taper/filter/trim, Fourier sizing and scaled rFFT | Root reran all seven focused tests. Three complete synthetic conditioning comparisons against the public binary agree below 2e-14 relative L2. Production EOB integration is still pending. |
| SEOB numerical attachment | Full waveform phase spline derivatives, scalar-series broadcasting, exact peak/index conventions, and high-resolution ringdown support | Root reran 15 tests and two paired public-state comparisons. Both last nonzero samples match. Amplitude residuals are below 2.80e-7; complex residuals remain 4.42e-5 and 3.21e-6, so attachment parity is still incomplete. |
| HM production assembly | Independent 61-knot QNM representation, London carrier mapping, and scalar-frequency parameter batching | Both frozen HM gates pass: catalog polarization residuals below 1.07e-7 and holdout residuals below 5.58e-7. Four additional waveform probes also pass the strict raw bound. |

The coordinator independently reran **70 passing tests** across the Pv2
waveform, source-frame, existing PhenomP interface, and parity-harness
controls. The controls include an oracle support edge above the comparison
band and a padded uniform grid. Neither comparison frequencies nor strict
thresholds were relaxed. The new frozen checkpoint confirms both HM gates
pass. A subsequent root rerun of the numerical EOB attachment,
HM API/reconstruction/QNM and published EOB/QNM tests passed **64 tests**
with the known XHM overlap test excluded. This is a distinct verification
set, not an increase to the strict counts. XHM, default XP/XPHM, and both
EOB production routes retain their unmet gates. The separate
[TD-to-FD conditioning contract](td-fd-conditioning-contract.md) identifies
another required EOB interface change beyond the physical dynamics.
The conditioning component now implements that contract; the
[coordinator's public-binary checks](verification/td-fd-conditioning-root-review.json)
cover zero and nonzero spins, a low-frequency safety cap, and paired suffix
padding. Those synthetic checks isolate the conditioning operation from the
unfinished physical waveform drivers.

The subsequent [attachment correction review](verification/eob-numerical-root-review/attachment-corrections.json)
records unchanged helper hashes around the coordinator's independent run.
It compares clean attachment on the public auxiliary trajectories, which
isolates attachment from evolution and does not establish a native-driver
waveform result. A separate coordinator run passed **41 MIT packaging and
parity-harness control tests**. Packaging checks do not resolve inherited
calibration provenance or establish legal clearance.

The coordinator subsequently reproduced **62 passing MSA boundary tests**
and **24 passing adaptive-driver/attachment tests**. Generic v223 precession
now agrees closely with public waveforms, but a small in-plane-spin boundary
still fails the raw bound. The initial XP production diagnostic supplied
explicit options to copies of frozen cases; those results do not increase the
strict counts. Default routing and untouched-case verification are being
corrected separately.

The new complete 20 Hz adaptive EOB run took 24.5 seconds, accepting 286 coarse
and 73 fine steps. With unchanged helper hashes it gives raw complex relative
L2 **1.2296e-4** and amplitude relative L2 **3.2626e-6**, comparing the complete
padded mode without time, phase or amplitude optimization. The last nonzero
sample agrees with the public mode at 5114.5M. Its fine restart is 4853.5M,
versus 4853.0M in the paired public trajectory. See
[full20-dt05.json](verification/eob-adaptive-root-review/full20-dt05.json).
This is a native time-domain diagnostic, not the canonical conditioned-FD
gate. Review also found a distinct-parameter batch defect in coarse evolution
and a controller difference documented in the
[adaptive contract](eob-adaptive-driver-contract.md#documented-controller-compatibility).

### EOB evidence

The saved late-inspiral runs use 30+20 solar masses, initial orbital frequency
0.04/M (about 51.7 Hz), and aligned spins (0,0) and (0.2,-0.1). The fixed
degree-six reconstruction gives complex relative refinement differences
**6.91e-8 and 1.96e-8** for step sizes 0.125M and 0.0625M. See the
[recovered report](verification/eob/recovered-result.json). The original
verifier crashed while reporting after integration; startup hashes were not
saved and cannot be reconstructed retrospectively. The verifier now persists
startup/progress records and accepts arbitrary valid refinement lists.

Review corrected a coarser-artifact selection bug, an overwritten complex-mode
variable, and comparison of different peak event definitions. Using the native
orbital-peak time origin, without an optimized shift, gives amplitude relative
L2 **0.001698 and 0.001575**, and instantaneous-frequency relative L2
**0.000468 and 0.000406** against public SEOBNRv4 TD outputs on the reported
common support. These replace neither the raw epoch diagnostics nor the
canonical FD gate. See the [TD comparison](verification/eob/published-waveforms.json).
The [phase note](verification/eob/phase-contract-note.json) is historical.
The full-duration comparisons below narrow the epoch discrepancy; the public
binary epoch also depends on the requested sampling interval. No
residual-fitted correction was installed.

The saved 20 Hz catalog-parameter trajectory uses 30+20 solar masses, zero
spins, step 0.5M and explicit initial frequency 20 Hz. It covers the complete
inspiral and merger. Comparing at the native orbital peak and the public binary
epoch gives raw complex relative L2 **8.459982e-4**, phase RMS **7.889794e-4 rad**,
and frequency RMS **0.0052327 Hz**. Using the NQC match as the time origin instead
causes a spurious 0.447 relative error. The binary's remaining output tail is
zero. See the [orbital-peak report](verification/eob-full20-orbital-peak/result.json).
The finer 0.25M run completed with unchanged dependency hashes. Comparing the
two saved runs with quintic interpolation gives complex relative L2
**4.1040e-5** in initial coordinates and **4.0668e-5** after subtracting each
orbital-peak time. Peak and match times change by **3.5472e-6M**. See the
[refinement report](verification/eob-full20-dt025/refinement.json).
Quintic interpolation against the binary gives **8.3324e-4** for the coarse
mode and **8.0927e-4** for the fine mode. The coordinating agent independently
reproduced the [fine result](verification/eob-full20-dt025-root-quintic/result.json).
Comparison uses common support; the omitted binary leading energy fraction is
**1.8031e-6**, and its omitted trailing energy is zero. These time-domain
results do not establish the strict frequency-domain gate.

The [public dynamics comparison](verification/eob/eob-clean-hamiltonian-public.json)
evaluates the clean Hamiltonian on the auxiliary vectors returned by the public
mode API. Interpreting their five rows as time, radius, orbital phase, tortoise
radial momentum and angular momentum is an observed schema, not a documented
API guarantee. Orbital-phase relative L2 against the fine native trajectory is
about **1.7e-8**, with an absolute difference below **1e-6 rad** near 4900M.
The two binary calls, at 0.5M and 1/4096 s, return epochs -1.232042294 s and
-1.232038090 s despite clean-Hamiltonian peak times differing by only
**1.28e-5M**. Each epoch must stay paired with its own returned vectors and
waveform. These diagnostics do not
justify changing the production epoch or fitting a residual time shift.

The clean analytic NQC derivative implementation evaluates directional
derivatives along the published Hamiltonian flow, including the flow's
Jacobian in the second derivative. Its three focused tests pass. In the
[full merger comparison](verification/eob/nqc-merger-full-grid.json), replacing
sampled derivatives with analytic derivatives reduces the post-match complex
residual from **0.007747 to 0.006954** at output step 0.5M, and from
**0.0004867 to 0.0004033** at output step 1/4096 s. The coordinating agent
independently reproduced the complete JSON exactly (SHA-256
`02e51187cffea6f52b009f6e480f18047ab1ad9cb51f8cbd806cf3b089c293db`).
The different output intervals still give materially different residuals;
the analytic NQC update does not resolve the public sampling and attachment
contract. This is a component experiment, without production routing changes.

The [four-grid epoch check](verification/eob/eob-epoch-contract-clean.json)
keeps each public mode, epoch and auxiliary vector set paired. It supplies
initial-time coordinates to the clean attachment and compares the resulting
complex mode at the same sample indices, with no fitted shift. Output steps
0.25M, 0.5M and 1M give post-match relative L2 errors **0.007770, 0.007747 and
0.007826**; 1/4096 s gives **0.0004867**. The clean degree-six orbital-peak
events vary by less than 2.4e-5M across these runs, whereas their epoch-relative
positions differ by about 0.017M. The coordinator reproduced all three
diagnostic tests. The subsequent [orbital-phase check](verification/eob/eob-public-phase-peak-clean.json)
does not reproduce the 0.017M split in the continuous event.

The [eight-spacing sweep](verification/eob/eob-public-epoch-sampling-clean.json)
keeps the physical system fixed and varies the output interval over
0.90–1.10M, with an exact 4096 Hz anchor. The clean continuous peak stays
within 5.0e-5M. When it lies before the sampled maximum, the public epoch
instead places zero approximately one high-resolution sample after that
maximum. When it lies after the sampled maximum, the epoch agrees with the
continuous event within 5.2e-5M. This supports a discrete bracketing effect.
The subsequent [bisection diagnostic](verification/eob/eob-public-epoch-bisection-clean.json)
finds that a fixed ten-step contraction of the nonbracketing interval explains
five epochs within their half-nanosecond rounding intervals. The same rule
misses the other three epochs by 7.5–11.1 ns, so it does not establish a common
algorithm. Fixed numerical Hamiltonian derivatives did not resolve those
three residuals. No fitted epoch correction was installed. Pure diagnostic
helpers now import without optional LAL bindings, with a subprocess regression.

The isolated published driver now evaluates the rectangular Fourier sum
directly at the requested frequencies. The previous padded FFT followed by
linear frequency interpolation introduced about **5.14%** relative error
against the direct sum on the saved full-duration waveforms. The new runtime
agrees with an independent direct sum to **6.3e-13**, and its fine-grid
plus/cross residuals against the public direct transform are **7.23e-4 /
7.14e-4**. See the [Fourier comparison](verification/eob/eob-published-fd-clean.json).
Frequency chunks limit each forward kernel allocation; autograd can retain
intermediate kernels for backward evaluation. Closed-form off-grid Fourier
and gradient tests pass. This fixes the transform in the experimental driver;
it does not replace the canonical SEOB route or certify its full waveform.

### Kerr and NRTidal evidence

The latest Kerr catalog reference calls only public `SimBlackHoleRingdown`
and Fourier-transforms its returned samples with explicit composite Simpson
weights. Adjacent sampling rates agree below 1e-7 on the complete comparison
band; the frozen case converges to 7.97e-9. No torchwave QNM or angular function
enters that reference. The case parameters and acceptance thresholds are
unchanged; its reference kind was corrected to an independent public TD API.
The coordinator reproduced **67 passed** across the NNLO driver/geometry,
Kerr reference, and harness controls, including optional public NNLO tests.

The original converged angular solution and the approximate public API differ
because the public API appears to stop its Frobenius series after a term below
0.01. This is an inferred numerical convention, not a fitted physical
calibration. Independently integrating that piecewise series explains the
angular shape. The fresh implementation derives its 21-point Kronrod and
10-point Gauss rules from polynomial orthogonality and moment equations.
Global adaptive integration begins on [-1,1], splits the interval with the
largest error estimate, and uses 1e-6 absolute/relative tolerance with a
1000-interval limit. This resolves the tested high-spin normalization
discrepancy. The series and quadrature choices are inferred numerical
compatibility conventions; no physical calibration was fitted.

The coordinator reproduced **29 passed, 1 skipped** across the current Kerr
angular, waveform and clean-room checks. The skip requires an opt-in external
Cook fixture; the public angular and waveform comparisons ran. Absolute
angular comparisons cover ten spins and four angles without cancelling the
normalization error. All four expanded waveform probes retain the original
1e-6 bound. Other Kerr modes are not certified by these results.

The two independent angular methods agree within 4.46e-14 on the saved real
and complex spheroidicity grid. The Frobenius solution's directly evaluated
angular-ODE residual is at most 2.67e-15. Quadrature, truncation, broadcasting
and gradient checks are in the dedicated tests and
[angular report](verification/kerr-angular/leaver_angular_convergence.json).
The report's runtime and generator hashes match the reviewed files.

The separate [gauge experiment](verification/kerr-angular/gauge.json) checks
the real-positive-a0 convention against the public binary angular API. A
constant phase change does not explain the angle-dependent difference.
Finite-difference angular-ODE residuals for the public output plateau under
step refinement on several spinning inputs; the independently constructed
solution satisfies the ODE to numerical precision. This is evidence to
resolve the API's approximation and compatibility contract, not grounds for
an oracle-fitted angular correction. The coordinating agent reproduced
**61 passed, 1 skipped** across the two angular test modules.

The fresh [NRTidal before/after comparison](verification/nrtidal/results.json)
records versions and unchanged source hashes through the run. All eight
before/after evaluations pass the original gates. The narrow provenance and
packaging work is documented in the [calibration audit](mit-calibration-audit.md).

### TEOB stage evidence

The [vacuum driver handoff](verification/teob-vacuum-rr-driver-clean/README.md)
wires the complete instantaneous state into the independent 35-mode infinity
and horizon flux. The circular-energy input sets radial momentum to zero but
retains the current angular momentum; total real energy and effective energy
remain distinct. The coordinator's 54-test rerun passed in 3.78 seconds.
The worker's bounded 4M evolution at 0.5M and 0.25M steps agrees at its endpoint
within 3.75e-15. This is a short vacuum check, not the complete tidal catalog
waveform. The old DIN torque regression still selects vacuum parameters and
expects the prior branch; its fallback coverage must be updated explicitly.

The isolated factorized-22 stage passes five focused tests. On the saved
caller-supplied SEOB trajectory, step sizes 0.5M and 0.25M agree to relative
complex L2 **2.02e-10** at 83 shared samples spanning 0–41M. See the
[convergence artifact](verification/teob22-convergence/). This short component
experiment does not validate the complete 30 Hz BNS waveform.

### NSBH and higher-mode evidence

The NSBH mass and spin corrections are integrated through a narrow optional
remnant provider in the D callable and its peak-slope calculation. The default
D provider remains identical under a direct numerical control. The current
catalog/holdout raw complex polarization residuals are **1.53535e-9 / 8.82875e-8**,
reproduced by the complete frozen suite above. Detector projections also pass
the original thresholds.

Four additional systems were added, again at zero and restored tides. The two
negative-spin systems originally failed on amplitude at both zero and restored
tides. The positive tidal-frequency convention fixes their zero-tide cases.
Their restored-tide amplitude errors were approximately 9.83e-5 and 1.13e-4.
The [public-property diagnostic](../../nsbh_amplitude_limit_diagnostic.json)
isolates a signed tidal frequency at the compactness-one-half limit. A positive
frequency convention matches the public binary exactly there; the paper's
printed equation does not explicitly include an absolute value. That
compatibility convention is integrated and tested.

The remaining amplitude failures are now resolved. The ringdown `delta2_prime`
uses the plus window required by the advertised BBH limit and corroborated by
the public component callable; the printed 2020 equation has the opposite
sign. The small-tide quadratic is `1 + Lambda/2 - Lambda**2/4`, fixed by its
endpoint values and derivative, and checked against direct binary waveforms.
For the mildly disruptive no-torus branch, the PN window uses an extra
`epsilon_ins` multiplying the complete mixed center. This is a black-box
compatibility composition inspired by the general 2015 Eq. (9): Eq. (28) and
the 2020 table literally print an already mixed center without that extra
factor. The implementation records that distinction and changes no fitted
coefficient. The original failing case and two neighbors have six-frequency
magnitude residuals between **1.2e-10 and 4.1e-10** in the
[component diagnostic](../../nsbh_amplitude_components_diagnostic.json).
All **22/22** expanded full-waveform and detector regressions pass, including
a third new system. Batch/scalar agreement and phase-sensitive gradients also
cover the corrected branch. No residual-fitted correction was installed.

The HM [PN truncation experiment](../../../scripts/diagnose_hm_pn_truncations.json)
shows that leading terms for modes (3,2) and (4,3) reproduce the binary's
low-frequency amplitude on the two frozen inputs. Their complete single-mode
waveforms still differ. The larger clean Eq. (4) substitution worsens the
summed waveform and was not integrated. Native agreement and binary agreement
are reported separately; neither is a substitute for whole-waveform acceptance.

The older `diagnose_hm_phase_clean.json` is not acceptance evidence: review
found inconsistent total/remnant mass units, an incorrect phasor sign, and an
identity check that asserted finiteness only. The replacement diagnostic
exposes additional mistakes in the phase transition and signed carrier
coordinates; it remains diagnostic work, with no runtime substitution.
The separate D callable contract now numerically verifies that its
returned phase already includes the reference and peak-slope corrections, and
that both polarizations carry `exp(-i * returned_phase)` including the requested
coalescence time. Its four-case identity error is below 2.3e-13 rad.

The corrected HM diagnostic conjugates the complete mapped positive D phase,
including the published multipole offset, and applies the analytically derived
reference-gauge term `(m/2 - 1) * slope_peak * Msec * f_ref`. Below the first
phase join this agrees with public binary modes to roughly **2e-6 rad** on four
systems. Substantial residuals begin above that join. This establishes the
low-frequency convention only; the transition and merger mapping remain under
review. Two-frequency binary derivative calls initially sampled padded output;
the replacement uses separate single-frequency calls and a D-mode derivative
control.

The subsequent [HM assembly comparison](../../../scripts/diagnose_hm_qnm_leaver_clean.json)
adds the corrected phase connection and published merger term to the fresh
amplitude mapping. Across four systems its raw plus-polarization relative L2
errors are about **6.4e-5–2.4e-4** with the printed London QNM fits, and
**4.0e-5–2.5e-4** with independently computed Leaver frequencies. The remaining
mode errors depend on the QNM and carrier inputs. These are component
diagnostics; the production HM implementation has not been replaced.
The fresh adapter now exposes separate amplitude and phase joins, defaulting
to the printed values 0.018 and 0.014. Existing compatibility diagnostics
explicitly retain 0.014 and 0.018. Its default harmonic projection now uses
the clean Torch Wigner function and preserves gradients, device placement,
float32 inputs, and vectorized inclination. Independent analytic quadrupole
and optional public harmonic checks pass.

The [exact N=61 reconstruction](verification/london-qnm-fit-exact61/README.md)
independently solves six fundamental QNM branches on a declared spin grid.
The resulting coefficients satisfy every printed rounding interval for
(2,2), (2,1), and (4,3), but some intervals fail for (3,3), (3,2), and (4,4).
N=61 was inferred from printed coefficients and is not a documented historical
training count. Its [HM waveform diagnostic](verification/hm-london-exact61/README.md)
reduces the four systems' plus/cross relative L2 errors to **4.71e-6 / 4.43e-6**
through **5.36e-5 / 5.80e-5**. This does not establish strict parity or justify
selecting fit coefficients from public waveform residuals. No production
coefficient was changed. The coordinator reproduced **48 tests** across HM
assembly, twisting, fit diagnostics, EOB Fourier conversion, and epoch helpers.

The fresh XHM PN helper now includes all five published higher-mode expressions
through v^6. Independent coefficient, exchange, finite-difference gradient and
collocation checks pass **25 tests** reproduced by the coordinator. The separate
[nonreexpanded (2,1) SPA branch](verification/xhm-spa/xhm-spa-public.json)
implements the published q<40 prescription, with absolute M^2/D scaling. Its
nine focused tests also pass independently. On q=1.5, 3 and 10 probes, the raw
amplitude residuals are **0.001476, 0.002152 and 0.021716**; the reexpanded E10
controls give **0.001058, 0.001047 and 0.013222**. Both omit the model's calibrated
inspiral correction, so this comparison does not select or fit a runtime
branch. The q=50 nonreexpanded call is explicitly rejected, and E10 remains
the diagnostic control there.

The [XHM inspiral assembly](verification/xhm-inspiral/xhm-inspiral-public.json)
now uses the final remnant spin in the Kerr ISCO, the printed inspiral
chiPN input, and provider-derived geometric amplitudes for the vetoes.
Public amplitudes are used only for comparison. Four systems and three modes
give physical amplitude residuals **0.0083–0.0357** on the reported inspiral
frequencies. The correspondence between the opaque provider's three outputs
and the frequency nodes remains a hypothesis. These results neither establish
the complete provider contract nor isolate calibration precision. The
coordinator reproduced **70 passed** across the EOB driver and XHM equation,
SPA and assembly tests; the sets overlap earlier counts.

The fresh Pv2 assembly isolates a remaining time-origin convention. Its
[aligned carrier check](pv2-clean/time-origin-diagnostic.json) finds identical
public Pv2/D amplitude up to approximately 3e-16, but a linear phase difference.
The published ringdown derivative accounts for most of its slope, leaving
**2.5e-6–6.3e-6 rad/Hz** on three systems. Predetermined interpolation checks
did not resolve that remainder. No residual-fitted time correction was added.
The fresh MSA composition remains diagnostic-only. The first comparison report
used zero transverse spins in the public waveform target and is superseded.
The corrected report records the shared full spin vectors, equation-level
source-frame transport, and a bounded v220/v223/v224 geometry/J/L audit. It
does not claim a production `prec_version` or full-band native-MSA parity.

All 209 printed PhenomD Table V coefficients were verified against the primary
paper's TeX. Using them literally worsens the tested waveforms; see the
[published-fit experiment](phenomd_published_report.md). This leaves
the inherited full-precision coefficients' provenance unresolved.

## Work in progress

- HM: Check mapped amplitude, remnant/QNM inputs and phase separately. The
  tested PN truncations alone do not resolve full-band compatibility.
- NSBH: The reported failures and expanded regressions are resolved. Preserve
  the branch and gradient coverage while the remaining models are completed.
- EOB/TEOB: Establish complete low-frequency support and the binary phase
  reference contract, then assemble the remaining published dynamics and
  merger stages.
- ROM: Numerical compatibility is resolved on the frozen case and expanded
  probes. Preserve external-only data handling while resolving redistribution
  terms; no HDF5 calibration data were added to the package.
- Remaining full-model work includes the PhenomD/X calibration provenance,
  precession carrier/frame assembly, full-duration EOB/TEOB evolution and
  merger. Preserve strict thresholds and
  require new independent validation points after these regressions pass.

## MIT boundary and verification

The project retains MIT metadata. Cook-derived fixtures and author Mathematica
assets with separate or unresolved terms were moved to external validation
storage; the runtime Kerr frequency data are independently generated. Wheel
and sdist inspections found neither the old NRTidal fit module nor `.m` files.
Existing obligations are recorded in `THIRD_PARTY_NOTICES.md`; changing a
package label does not resolve inherited calibration provenance. Full-source
MIT clearance and full numerical parity are not claimed.

An earlier coordinating-agent focused run passed **74 tests** across Kerr
angular, HM mapping, published EOB driver, NRTidal reconstruction and packaging
checks (Python 3.13.9, Torch 2.9.1). Earlier reviewed EOB event/attachment/NQC,
ROM and packaging checks passed 47 tests with one external-data skip. These
test sets overlap and should not be summed into an independent total.

An earlier coordinating-agent focused run passed **46 tests** across parity
harness controls, clean HM mapping and MIT packaging. Another earlier run passed
**71 tests** across NSBH amplitude/phase/reconstruction, analytic NQC and MIT
packaging, including all 22 strict NSBH regressions. These overlapping sets
must not be summed. An earlier full frozen run completed in 57.04 seconds
with unchanged input hashes and no manifest drift.

No new NR run, copied target implementation or oracle-fitted runtime constant
was introduced in this implementation work. The broad pre-existing working
tree remains uncommitted.
