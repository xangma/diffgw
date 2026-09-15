# Independent parity reconstruction: current evidence

Starting revision: `84ef9b3`. These results describe uncommitted reconstruction
work begun on 12 September 2026. **Full LAL parity is not achieved.** The
current [Pass 34 status](pass-34-summary.md) supersedes the older measurements
later in this cumulative record: the fresh strict catalog passes **11/21** and
the frozen holdout passes **4/13**, while the complete ordinary suite passes
**1,248 tests with 9 skipped**. Passes 31--34 add the NSBH reference sentinel,
the source-correct Pv2 opening angle, an eight-mode TaylorF2_HM inventory, and
per-mode XHM/XPHM residual localization. They do not relax a gate or add a
strict passing case. [Pass 35](pass-35-summary.md) closes the remaining bounded
PhenomHM, Taylor `(5,5)`, and NSBH audits without a speculative runtime edit.
[Pass 36](pass-36-summary.md) independently confirms that the remaining gaps
need calibration, convention, complete-driver, or independent-reference data
that the allowed public evidence does not provide, and inventories every
nonpassing strict case.
[Pass 15](pass-15-summary.md) adds ordered partial-node amplitude reconstruction
and numerical interval denominator checks. [Pass 17](pass-17-summary.md)
audits the remaining bounded XHM source assets without retaining another
runtime change. [Pass 19](pass-19-summary.md) retains the source-defined NSBH
remnant-frequency classifier and closes the current PhenomD, detector,
TaylorF2_HM, KerrRingdown and low-Lambda NSBH audits. **No additional strict
case passes; full LAL parity remains unfinished.** The figures below retain the
same strict case counts, with the Pass 19 NSBH regression recorded separately.
[Pass 20](pass-20-summary.md) independently audits the SEOBNRv4 `(2,2)` phase,
remnant fits, NQC, and merger attachment. [Pass 21](pass-21-eob-integration.md)
connects the prototype `SEOBNR_ODE` path to refined peak plus NQC/C1 attachment
when the peak resolves, retaining the legacy final-sample PhenomD fallback for
under-resolved runs. The explicit `assembly_model='seobnrv4_published'` path
requires all five choices. Numeric amplitude threshold units, sparse phase
dispatch, and the canonical high-q phase-fit record remain unresolved. The
post-change strict diagnostic remains failed with the same headline errors.
Pass-7 Kerr and pass-6 EOB experiments remain historical evidence and were not
repeated in passes 9--15. See the [remaining-work assessment](pass-12-remaining-work.md)
for the distinction between Phenom corrections and larger unfinished models.
The [initial report](initial-pass-summary.md) preserves the original matched
before/after measurements and their historical 383-test count.

The subsequent [calibration-data licensing review](calibration-data-licensing.md)
records the MIT code/ODC-By data distinction, packaged notices and the CC-BY ROM
data route. Pass 7 removes external numerical inputs from the Kerr-frequency runtime;
those earlier author-data fixtures remain verification-only. See the
[independent QNM report](pass-7-independent-qnm.md).

## Acceptance and coverage

The complete ordinary suite passed **1,248 tests, with 9 skipped**, across eight
parallel shards; the slowest shard finished in 71.42 s. Seven require CUDA and
two require generation-only gmpy2; MPS tests passed.
The strict catalog passes **11/21 cases**: nine fail numerically and one lacks
an independent complete waveform reference. The separate frozen holdouts pass
**4/13 cases**. TaylorF2_HM tests only its carrier, so two model-level reference
gaps remain: its higher modes and the complete Kerr-ringdown waveform.

All cases retain uniform CPU FP64 gates: unaligned complex relative L2 and
relative norm errors at most `1e-6`, flat and named-PSD mismatches at most
`1e-8`, for both polarizations and detector projections. Optimized matches are
additional diagnostics. No threshold was relaxed and no native self-comparison
counts as LAL parity. Passing frozen cases does not certify a parameter domain.

The three EOB cases now use public LAL binary references. Their failures replace
the original unresolved self-reference cases; they do not represent a new loss
of demonstrated parity. The external SEOBNRv4 ROM asset was downloaded with
user approval and is pinned by URL and hash. See [reference coverage](independent-coverage.md).

## Implemented and checked

Fresh-context implementers reconstructed components from primary papers and
author mathematical data. The primary agent reviewed changes and performed
public binary verification separately.

| Area | Current changes and limits | Source and verification records |
| --- | --- | --- |
| PhenomD and NSBH | Analytic joining derivatives, published phase conventions, removal of an unpublished default taper, and dimensionally consistent disruption windows. Zappa tidal remnant polynomials use initial BH spin, and the NSBH remnant frequency now classifies every merger branch as specified by Thompson Appendix A. The printed low-Lambda amplitude factor is discontinuous at Lambda=1 and its omitted `gamma1` notation remains unresolved. | [Baseline](baseline.md), [pass 2](pass-2-baseline.md), [NSBH](nsbh.md), [pass-9 precision fix](pass-9-baseline.md), [pass-11 spin correction](pass-11-baseline.md), [pass-18 classifier](pass-18-nsbh-classification.md), [pass-19 review](pass-19-summary.md) |
| HM and XHM | Published frequency maps, phase terms and derivatives, collocation connections, PN amplitudes and 32 mixing work. The carrier's common normalization factor cancels; production regression checks preserve it. Intermediate nodes now use constant-A0 targets. Parameter vetoes retain boundary derivatives; pole fallbacks retain surviving values. Partial-node quartics, conditional high-q vetoes and numerical interval denominator checks are active; four-point monotonicity is a documented interpretation. Phase defaults use nodes [1,2,3,4,6]. Odd-m collocation targets now use absolute fitted amplitudes. Official intermediate phase assets match every required native fit. Numeric amplitude-veto units, sparse phase dispatch, the canonical high-q `Lambda` record, and the apparent printed 32 amplitude-cut denominator error remain unresolved. | [Higher modes](higher-modes.md), [pass 2](pass-2-higher-modes.md), [pass-11 normalization review](pass-11-higher-modes.md), [pass-12 vetoes](pass-12-higher-modes.md), [pass-13 corrections](pass-13-summary.md), [pass-14 sign convention](pass-14-summary.md), [pass-15 selection](pass-15-summary.md), [pass-17 audit](pass-17-summary.md) |
| Precession | Published NNLO and experimental elliptic MSA terms; all five Table IV conventions, observer azimuth and reference-frequency batching. Unsupported NNLO opposed-J rotations reject explicitly. NNLO versions 101--104 use the derived remnant; explicit `final_spin_mod=0` now selects it independently of angle version. Standalone MSA averaged couplings and squared angular momentum are exposed; conflicting mass denominators in the paper prevent automatic MSA Kerr-remnant selection. Raw gauges and MSA/ODE version semantics remain open. | [Precession](precession.md), [pass 2](pass-2-precession.md), [conventions](pass-4-precession-conventions.md), [independent review](pass-4-precession-conventions-independent-review.md), [NNLO remnant](pass-10-nnlo-remnant.md), [MSA components](pass-11-precession.md), [pass-12 selector](pass-12-precession.md) |
| Kerr components | Our radial/angular eigenproblem solver and 16 regenerated QNM branches, spectral angular Teukolsky eigenfunctions, causal two-pole Fourier transforms, batched projection and finite-Q energy-to-amplitude conversion. The waveform is a restricted equal-twin, amplitude-supplied primitive; physical distance scaling and independent twin/overtone excitation metadata remain unresolved. | [Independent QNM data](pass-7-independent-qnm.md), [angular solver](pass-3-kerr-angular.md), [projection](pass-3-ringdown-projection.md), [energy](pass-3-ringdown-energy.md), [pass-19 waveform audit](pass-19-kerr-ringdown-primary.md) |
| Direct hybrid MECO | Standalone PN/Kerr energy minimization with branch/domain checks and differentiable roots. It does not replace XAS matching defaults or recalibrate waveform shapes. | [Method and 37 tests](pass-7-independent-meco.md) |
| EOB conservative dynamics | Repaired tortoise/real-Hamiltonian normalization and independent batched derivatives; explicit experimental published SEOBNRv4 Hamiltonian. | [EOB](pass-2-eob-conservative.md), [SEOBNRv4 Hamiltonian](pass-3-seobnrv4.md) |
| EOB radiation | All 35 DIN amplitudes through ell=8, published comparable-mass and Kerr spin terms, Gamma tail and flux/torque adapters; the ell=8 spin truncation is mandatory and explicit. Zero-mode flux Hessians are finite. | [Factorized modes](pass-3-factorized-modes.md), [spinning flux](pass-6-spinning-flux.md), [review](pass-6-spinning-flux-review.md) |
| EOB phase and remnants | Complex 22 mode and printed residual-phase components; published final mass and all three Hofmann spin-fit blocks. The Bohé energy normalization, inheritance of older spin terms, and Hofmann block are not uniquely selected by the primary sources and therefore remain explicit choices. | [Amplitude](pass-3-seobnrv4-mode22.md), [phase](pass-20-eob-phase22-primary.md), [phase review](pass-20-eob-phase22-independent-review.md), [remnant](pass-20-eob-remnant-primary.md), [remnant review](pass-20-eob-remnant-independent-review.md) |
| EOB waveform assembly | The prototype `SEOBNR_ODE` path now uses refined peak plus NQC/C1 attachment when the peak resolves, retaining the legacy final-sample PhenomD fallback for under-resolved runs. `assembly_model='seobnrv4_published'` exposes the existing end-to-end published path and requires all five choices explicitly. Two-input convergence checks pass their stated criterion; the post-change strict diagnostic remains failed at polarization L2 1.0206987416256423 and projection L2 1.0205836988469421. No LAL parity is claimed. | [Driver and evidence](pass-6-eob-driver.md), [pass-20 attachment audit](pass-20-eob-attachment-primary.md), [pass-21 integration](pass-21-eob-integration.md), [independent review](pass-20-eob-attachment-independent-review.md) |
| Tidal EOB prototype | Repaired common 3PN Padé metric, radial kinetic terms, inference derivatives and intrinsic batching. Inherited tidal resummation, self-spin and merger heuristics are not the calibrated TEOBResumS model. | [Tidal potential](pass-3-teob-potential.md) |
| Detectors | Split-day GMST, corrected GPS leap boundaries, normalized survey arm directions and batched FD sky projection. UTC approximates UT1; inherited geometry precision and the integer-shift TD approximation remain explicit limits. | [Detectors](pass-3-detectors.md), [pass-9 one-sample fix](pass-9-detectors.md) |

The complete ordinary suite tests mathematical consistency and API behavior;
it is not an external waveform parity certificate. The new EOB assembly is
experimental and requires explicit choices. Its presence does not establish
SEOBNRv4 parity or turn the existing surrogate into SEOBNRv4_ROM.

## Current strict measurements

Errors below are dimensionless maximum unaligned complex relative L2 errors.
The result also includes norm and mismatch gates, so L2 alone is insufficient.

### Catalog

| Case | Polarizations | Detector projection | Result |
| --- | ---: | ---: | --- |
| TF2_point_particle_bns | 1.6385727e-12 | 2.9621786e-07 | PASSED |
| TF2_unequal_spinning | 1.6259249e-12 | 2.9618316e-07 | PASSED |
| TF2_tidal_6PN | 1.6394512e-12 | 2.9551914e-07 | PASSED |
| TF2_tidal_6.5PN | 1.6011044e-12 | 2.9551916e-07 | PASSED |
| TF2Ecc_circular_limit | 1.6385727e-12 | 2.9621786e-07 | PASSED |
| TF2Ecc_eccentric_spinning | 1.5733368e-12 | 2.9626894e-07 | PASSED |
| IMRPhenomD_diagnostic | 4.9476516e-07 | 6.8021863e-07 | PASSED |
| IMRPhenomXAS_diagnostic | 4.2127909e-14 | 2.993189e-07 | PASSED |
| IMRPhenomHM_diagnostic | 0.0045409491 | 0.0045121740 | FAILED |
| IMRPhenomXHM_diagnostic | 0.38200117 | 0.37948173 | FAILED |
| IMRPhenomPv2_diagnostic | 0.25723438 | 0.25699423 | FAILED |
| IMRPhenomXP_diagnostic | 0.072548555 | 0.072424492 | FAILED |
| IMRPhenomXPHM_diagnostic | 0.16092642 | 0.16112354 | FAILED |
| IMRPhenomD_NRTidal_diagnostic | 2.7273566e-07 | 4.8384786e-07 | PASSED |
| IMRPhenomD_NRTidalv2_diagnostic | 2.7017343e-07 | 4.8208794e-07 | PASSED |
| IMRPhenomNSBH_diagnostic | 2.9516980e-05 | 2.9649914e-05 | FAILED |
| TaylorF2_HM_diagnostic | 1.6385725e-12 | 2.9621786e-07 | PASSED |
| KerrRingdown_diagnostic | — | — | UNRESOLVED_ORACLE |
| SEOBNR_ODE_diagnostic | 1.0206987 | 1.0205837 | FAILED |
| SEOBNR_ROM_diagnostic | 1.0019032 | 1.0022284 | FAILED |
| TEOBResumS_diagnostic | 1.0008916 | 1.0007454 | FAILED |

### Frozen holdouts

| Case | Polarizations | Detector projection | Result |
| --- | ---: | ---: | --- |
| IMRPhenomD_holdout_0 | 1.2321902e-05 | 1.244372e-05 | FAILED |
| IMRPhenomD_holdout_1 | 8.9629199e-07 | 1.0135173e-06 | FAILED |
| IMRPhenomD_holdout_2 | 9.6293422e-07 | 1.1433393e-06 | FAILED |
| IMRPhenomXAS_holdout_3 | 3.8380711e-14 | 3.1036481e-07 | PASSED |
| IMRPhenomXAS_holdout_4 | 9.4909419e-09 | 4.1285273e-07 | PASSED |
| IMRPhenomHM_holdout_5 | 0.019464597 | 0.019157343 | FAILED |
| IMRPhenomXHM_holdout_6 | 0.65865751 | 0.6533453 | FAILED |
| IMRPhenomD_NRTidal_holdout | 1.6021479e-07 | 3.895616e-07 | PASSED |
| IMRPhenomD_NRTidalv2_holdout | 1.575885e-07 | 3.880917e-07 | PASSED |
| IMRPhenomNSBH_holdout | 0.0017579727 | 0.0017580928 | FAILED |
| IMRPhenomPv2_holdout | 1.418307 | 1.4506322 | FAILED |
| IMRPhenomXP_holdout | 0.40327128 | 0.51091129 | FAILED |
| IMRPhenomXPHM_holdout | 1.1851135 | 1.2770461 | FAILED |

PhenomD's largest holdout polarization error decreased from about `4.32e-4`
in the initial report to `1.23e-5`. Two other PhenomD holdouts pass that
polarization bound but still fail detector projection bounds. This improvement
therefore adds no strict passing case. Higher-mode and precession residuals
are not uniformly monotonic as published equations replace inherited choices.

Pass 10 added the paper-derived NNLO remnant correction. The printed source
has an apparent extra-square inconsistency; the [source audit](pass-10-nnlo-remnant.md)
records the vector derivation and independent earlier paper. Default-setting
XP improves in the catalog; XPHM and both precession holdouts worsen. All other
catalog and holdout metrics are unchanged from pass 9. A separate, fixed
comparison explicitly sets NNLO version 102 and remnant option 0 in the public
reference: all four polarization errors fall by about 5--12%, but **0/4 pass**.
This diagnostic inventory does not replace the frozen acceptance cases;
[before/after measurements](verification/pass-10/comparison.json) retain both.
The new physics is derived from primary sources, not chosen from residuals.

Pass 11 adds no strict passing case. The initial-BH-spin correction raises
the NSBH holdout polarization error from `0.0013294683` to `0.0016230897`;
its catalog metrics and all other frozen case metrics are unchanged. The
source-defined spin variable is retained without fitting the remaining
coefficients. A separate 34-case probe explicitly selects NNLO/MSA versions
and individual HM/XHM mode pairs: **3/34 pass**, with every metric unchanged
from the pass-10 snapshot. These three overlapping carrier checks add no
frozen passing cases. All four native/reference mode-sum controls pass.
XHM reference probes activate both signed modes and disable multibanding;
earlier incompatible probe settings are archived. See the
[pass-11 comparison](verification/pass-11/comparison.json) and
[review decisions](pass-11-independent-review.md). Automatic MSA remnant
selection and an extra XHM carrier factor were excluded before final binary
verification because their source derivations were not sound.

Pass 12 adds the supported XHM parameter vetoes and opt-in NNLO remnant
selection. Every frozen metric and component result is unchanged. A separate
public-binary control confirms default outputs equal explicit 223/3 on all
four fixed XP/XPHM cases, consistent with the independently documented source
default. Native NNLO102 and incomplete MSA remain distinct prescriptions.
The [pass-12 comparison](verification/pass-12/comparison.json) and
[review decisions](pass-12-summary.md) record the retained scope and exclusions.

Pass 13 corrects XHM intermediate amplitude normalization, separates the
published parameter and pole fallbacks, and applies the default phase node f6.
Strict passing counts remain unchanged; all four full XHM/XPHM complex errors
worsen, despite some component amplitude improvements. Other frozen metrics
are unchanged. The separate component probe
still passes 3/34 and all four mode-sum controls pass. See the
[pass-13 report](pass-13-summary.md) for before/after complex and magnitude
errors, source evidence, and remaining fit-convention limits.

Pass 14 applies the source-defined absolute-value convention to odd-m amplitude
collocation targets and adds independent author-expression fixtures. No new
strict case passes. See the [pass-14 report](pass-14-summary.md) for the current
verification and remaining reconstruction gaps.

Pass 17 rejects an amplitude-threshold candidate whose physical normalization
could not be established, archives the complete official intermediate phase
fit set, and audits the high-q inspiral branch and the printed `(3,2)`
amplitude cut. No runtime or frozen verifier input changed. The restored
ordinary suite passed 1,140 tests with 9 skipped; strict measurements remain
the pass-15 results by exact runtime and test hashes. See the
[pass-17 report](pass-17-summary.md).

Pass 20 audits the SEOBNRv4 `(2,2)` phase, remnant selection, NQC, and merger
attachment against primary sources and independent reviews. The isolated
equations are retained, but the papers do not uniquely resolve the residual
spin-phase convention or Hofmann block, and the default `SEOBNR_ODE` path does
not wire the published peak/NQC/C1 sequence. No runtime or frozen-verifier
input changed. The combined focused EOB suites passed 149 tests with 4 skipped.
See the [pass-20 report](pass-20-summary.md).

Pass 21 wires the native refined peak/NQC/C1 attachment into the prototype
`SEOBNR_ODE` path, with a legacy final-sample PhenomD fallback when the run is
under-resolved. It exposes the existing end-to-end published assembly through
`assembly_model='seobnrv4_published'`, requiring explicit values for all five
published choices, and adds the phase/radiation adapter plus positive-domain
validity guards. The focused pass-21 tests passed 28 tests in 14.01 seconds.
The complete ordinary suite passed 1,161 tests with 9 skipped across four
parallel shards.
This is component and API evidence only: no LAL parity is claimed. The rerun
strict diagnostic remains failed with maximum polarization complex relative L2
1.0206987416256423 and projection complex relative L2 1.0205836988469421. See the
[pass-21 report](pass-21-eob-integration.md).

The independent [Kerr component measurements](verification/pass-7/kerr-components.json)
show a maximum relative complex frequency difference of `3.40401e-6` across
60 supported LAL QNM cases. The angular binary comparison, after removing a
constant phase for diagnosis only, differs by about `1e-4` to `1e-3` at nonzero
spin. Neither result certifies complete Kerr waveform parity. Our stricter equation-solve holdouts, frozen external reference
fixtures and angular differential-equation tests validate the components without fitting any native coefficient to binary residuals.

An additional [time-domain Kerr probe](verification/pass-7/kerr-waveforms.json)
uses a primary-derived finite-Q energy normalization and the public
`SimBlackHoleRingdown` API. Supplying phase zero to both APIs gives a maximum
relative L2 difference of `2.00374`; a separately recorded fixed native phase
of pi gives `0.0298245`. Neither probe optimizes alignment. The native phase,
angular gauge and reference energy semantics are not established as identical,
so these diagnostics do not replace the unresolved strict reference case.
The [energy conversion and its independent review](pass-3-ringdown-energy.md)
are complete, with sphere/time-integrated derivative tests.

The retained pass-6 [EOB convergence experiment](verification/pass-6/eob-convergence/result.json)
uses two fixed inputs and time steps 0.5, 0.25 and 0.125M. Successive complex
mode differences decrease to `1.03046e-4` and `8.10494e-6`, without alignment.
This does not certify `1e-6` numerical accuracy. A separate
[LAL TD probe](verification/pass-6/eob-waveforms.json) reuses the frozen
finest-step tensors and reports maximum polarization relative L2 errors
`1.87698` and `1.41289`. It preserves reference epochs and performs no fitted
alignment. Phase/time equivalence remains unresolved. These failures are
recorded separately from the unchanged strict catalog; see the
[driver report](pass-6-eob-driver.md) for conventions and support limits.

## Remaining work

1. Resolve the remaining precession version semantics, raw MSA phase gauges,
   singular limits and MSA/ODE carrier/remnant treatment. NNLO versions 101--104
   now include the derived remnant prescription; explicit NNLO comparisons still
   fail strict gates. MSA averaged components are now available, but the
   conflicting initial/final mass denominators require independent primary
   clarification before selecting a Kerr remnant. Published Table IV conventions
   and tidal initial-condition gradients are tested.
2. Resolve remaining author-defined higher-mode fit conventions, collocation
   conditions, phase constants and mode mixing. The common normalization in the
   32 carrier ratio has been checked and cancels. Pass 14 establishes author
   expression correspondence for eight intermediate amplitude fits. Pass 17
   establishes every required intermediate phase fit; repeating those lookups
   is unnecessary. Numeric amplitude-veto units, sparse phase-node dispatch,
   the canonical high-q `Lambda` record, and the printed 32 amplitude-cut
   denominator remain unresolved. Unsupported alternatives remain
   explicitly experimental rather than selected from oracle residuals.
3. Define the EOB target choices for the inherited 22 spin phase, ell=8 spin
   truncation and Hofmann block, then wire the default generator through the
   continuous factorized mode, orbital-frequency peak, NQC, explicit
   remnant/QNM selection and C1 merger ansatz. Pass 20 confirms that the
   equation-level pieces are present while the default endpoint/PhenomD
   attachment is not SEOBNRv4. Resolve physical phase/time compatibility and
   numerical accuracy against the defined model before regenerating the ROM.
   A complete calibrated tidal TEOBResumS reconstruction remains a separate
   implementation track.
4. Define a complete Kerr waveform API with physical distance and independent
   twin-mode/overtone excitation inputs, and select a complete TaylorF2_HM PN
   amplitude truncation with harmonic SPA normalization and stored-mode phase
   conventions. Pass 19 confirms that neither target can be filled in safely
   from the current API metadata alone.
5. **Deferred by the user on 2026-09-13:** generate remaining
   waveform/remnant/matter calibrations from our own simulations where new fits
   are required. Use the existing calibration tables for now; see the
   [calibration inventory](pass-7-independent-calibration-inventory.md).
   The [BBH NR pilot on len](pass-8-len-nr-pilot.md) passed its finite
   initial-data and checkpoint/restart checks. The subsequent 48-worker 100 M
   performance run was stopped at the user's request before its first checkpoint;
   inputs and partial output remain available, with no restart scheduled.
   Independently source or replace remaining inherited high-precision assets.
   Near agreement with a binary does not establish their provenance or determine
   unpublished coefficients. Full clean-room or repository licensing
   certification is outside what this reconstruction has demonstrated.

## Reproduction and provenance

Host `RONI2`, cwd `/Users/xangma/repos/torchwave`; CPU float64/complex128.
Interpreter `/Users/xangma/miniconda3/envs/pycbcorig313/bin/python`, Python 3.13.9,
PyTorch 2.9.1, LAL 7.7.0, LALSimulation 6.2.0 and PyCBC 2.10.0. With
`PYTHONPATH=src:tests`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and
`LAL_DATA_PATH=/tmp/torchwave-oracle-data`:

```sh
python -m pytest -q
python tests/parity_harness.py --strict --json-out docs/reviews/2026-09-12-clean-room/verification/pass-15/catalog.json
python tests/parity_harness.py --scope=holdout --strict --json-out docs/reviews/2026-09-12-clean-room/verification/pass-15/holdout.json
python scripts/verify_kerr_components.py --json-out docs/reviews/2026-09-12-clean-room/verification/pass-7/kerr-components.json
python scripts/verify_kerr_waveforms.py --json-out docs/reviews/2026-09-12-clean-room/verification/pass-7/kerr-waveforms.json
python scripts/verify_eob_published_convergence.py --output-dir docs/reviews/2026-09-12-clean-room/verification/pass-6/eob-convergence
python scripts/verify_eob_published_waveforms.py --convergence-dir docs/reviews/2026-09-12-clean-room/verification/pass-6/eob-convergence --json-out docs/reviews/2026-09-12-clean-room/verification/pass-6/eob-waveforms.json
```

The ordinary, catalog, holdout and explicit-component commands were rerun in
pass 19, with exit codes 0, 1, 1 and 1 respectively. Their counts remain
11/21, 4/13, 3/34 and 4/4 for the mode-sum controls; only the NSBH catalog
metrics changed materially.
The Kerr commands are retained pass-7 checks; the final two are retained pass-6
EOB experiments. The [pass-15 driver](verification/pass-15/verify.py) also runs
the separate 34-case explicit-component probe (exit 1, 31 strict failures).
Diagnostic scripts'
successful execution does not imply parity acceptance. Exact current process
metadata, input hashes, versions, case definitions, failures and logs are under
[pass-15 verification](verification/pass-15/). The earlier
[pass-11 setup probes](verification/pass-11/) preserve the signed-mode and
multibanding checks used to establish the unchanged component inventory.
Earlier runs, the
[pass-10 summary](pass-10-summary.md) and the
[pass-6 summary](pass-6-summary.md) remain in place. The
[worktree hash inventory](verification/worktree-sha256.json) records the current
changed files; its [initial snapshot](verification/initial-pass-worktree-sha256.json)
[pass-10 snapshot](verification/pass-10-worktree-sha256.json) and
[pass-11 snapshot](verification/pass-11-worktree-sha256.json) are preserved
separately. No commit or pull request has been created. All
verification processes have finished.

Implementers were instructed to use primary papers and independently published
author data, exclude LAL/PyCBC implementation source and translated ports, and
exclude the earlier source-derived calibration specification. This was an
instruction boundary, not OS-enforced isolation. Source URLs, versions,
hashes and equation references are recorded in the linked component reports.
The primary verifier uses public binary APIs separately and does not fit
implementation coefficients to their residuals.

An initial implementer search exposed unsolicited source snippets that were
not opened or used; later searches were restricted to primary papers. A
verifier API-documentation search also returned header declarations, which
were not forwarded as implementation evidence. Inherited TorchWave source
remains available and its provenance has not been certified. This work adds
no runtime waveform-oracle dependency and cannot certify the repository's
historical clean-room or licensing claims.
