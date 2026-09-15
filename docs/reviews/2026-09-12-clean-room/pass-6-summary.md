# Independent parity reconstruction: current evidence

Starting revision: `84ef9b3`. These results describe uncommitted reconstruction
work begun on 12 September 2026. **Full LAL parity is not achieved.** The
figures below describe the completed pass-6 verification, including the new
experimental published-equation EOB waveform assembly.
The [initial report](initial-pass-summary.md) preserves the original matched
before/after measurements and their historical 383-test count.

The subsequent [calibration-data licensing review](calibration-data-licensing.md)
records the MIT code/ODC-By data distinction, packaged notices and the CC-BY ROM
data route. It changes no numerical results reported here.

## Acceptance and coverage

The complete ordinary suite passed **884 tests, with five CUDA-only tests skipped,
in 50.09 s**. Source and test hashes were unchanged throughout each final run.
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
| PhenomD and NSBH | Analytic joining derivatives, published phase conventions, removal of an unpublished default taper, and dimensionally consistent disruption windows. Default compatibility and remnant/calibration gaps remain. | [Baseline](baseline.md), [pass 2](pass-2-baseline.md), [NSBH](nsbh.md) |
| HM and XHM | Published frequency maps, phase terms and derivatives, collocation connections, PN amplitudes and 32 mixing work. Some author-fit normalization and closure conventions remain unresolved. | [Higher modes](higher-modes.md), [pass 2](pass-2-higher-modes.md) |
| Precession | Published NNLO and experimental elliptic MSA terms; all five Table IV conventions, observer azimuth and reference-frequency batching. Unsupported NNLO opposed-J rotations reject explicitly. Raw MSA gauges, target version semantics and precession-aware remnants remain open. | [Precession](precession.md), [pass 2](pass-2-precession.md), [conventions](pass-4-precession-conventions.md), [independent review](pass-4-precession-conventions-independent-review.md) |
| Kerr components | Attributed Cook v4 QNMs for 15 mode/overtone combinations, spectral angular Teukolsky eigenfunctions, causal two-pole Fourier transforms, batched projection and finite-Q energy-to-amplitude conversion. Complete LAL energy, angular and phase conventions remain unresolved. | [QNM data](pass-3-qnm.md), [angular solver](pass-3-kerr-angular.md), [projection](pass-3-ringdown-projection.md), [energy](pass-3-ringdown-energy.md) |
| EOB conservative dynamics | Repaired tortoise/real-Hamiltonian normalization and independent batched derivatives; explicit experimental published SEOBNRv4 Hamiltonian. | [EOB](pass-2-eob-conservative.md), [SEOBNRv4 Hamiltonian](pass-3-seobnrv4.md) |
| EOB radiation | All 35 DIN amplitudes through ell=8, published comparable-mass and Kerr spin terms, Gamma tail and flux/torque adapters; the ell=8 spin truncation is mandatory and explicit. Zero-mode flux Hessians are finite. | [Factorized modes](pass-3-factorized-modes.md), [spinning flux](pass-6-spinning-flux.md), [review](pass-6-spinning-flux-review.md) |
| EOB phase and remnants | Complex 22 mode and printed residual-phase components; published final mass and all three Hofmann spin-fit blocks. Unresolved phase inheritance and fit selection require explicit choices. | [Amplitude](pass-3-seobnrv4-mode22.md), [phase](pass-6-eob-phase22.md), [remnant](pass-6-eob-remnant.md) |
| EOB waveform assembly | New `SEOBNRv4Published` path integrates to a refined frequency peak, solves all five NQC conditions, and joins C1 merger/ringdown with physical Fourier projection. Two-input convergence checks pass their stated criterion; independent LAL waveform diagnostics fail. Public frequency-mask broadcasting is repaired. | [Driver and evidence](pass-6-eob-driver.md), [attachment review](pass-6-eob-attachment-review.md), [driver review](pass-6-eob-driver-review.md) |
| Tidal EOB prototype | Repaired common 3PN Padé metric, radial kinetic terms, inference derivatives and intrinsic batching. Inherited tidal resummation, self-spin and merger heuristics are not the calibrated TEOBResumS model. | [Tidal potential](pass-3-teob-potential.md) |
| Detectors | Split-day GMST, corrected GPS leap boundaries, normalized survey arm directions and batched FD sky projection. UTC approximates UT1; inherited geometry precision and the integer-shift TD approximation remain explicit limits. | [Detectors](pass-3-detectors.md) |

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
| IMRPhenomD_diagnostic | 4.9476541e-07 | 6.8021885e-07 | PASSED |
| IMRPhenomXAS_diagnostic | 4.2127909e-14 | 2.993189e-07 | PASSED |
| IMRPhenomHM_diagnostic | 0.037661723 | 0.037816562 | FAILED |
| IMRPhenomXHM_diagnostic | 0.10095288 | 0.1003744 | FAILED |
| IMRPhenomPv2_diagnostic | 0.25723438 | 0.25699423 | FAILED |
| IMRPhenomXP_diagnostic | 0.075444865 | 0.075070458 | FAILED |
| IMRPhenomXPHM_diagnostic | 0.088421928 | 0.087895144 | FAILED |
| IMRPhenomD_NRTidal_diagnostic | 2.7273601e-07 | 4.8384815e-07 | PASSED |
| IMRPhenomD_NRTidalv2_diagnostic | 2.7017378e-07 | 4.8208823e-07 | PASSED |
| IMRPhenomNSBH_diagnostic | 2.95419e-06 | 3.0959597e-06 | FAILED |
| TaylorF2_HM_diagnostic | 1.6385725e-12 | 2.9621786e-07 | PASSED |
| KerrRingdown_diagnostic | — | — | UNRESOLVED_ORACLE |
| SEOBNR_ODE_diagnostic | 1.0206987 | 1.0205837 | FAILED |
| SEOBNR_ROM_diagnostic | 1.0019032 | 1.0022284 | FAILED |
| TEOBResumS_diagnostic | 1.0008916 | 1.0007454 | FAILED |

### Frozen holdouts

| Case | Polarizations | Detector projection | Result |
| --- | ---: | ---: | --- |
| IMRPhenomD_holdout_0 | 1.2321889e-05 | 1.2443707e-05 | FAILED |
| IMRPhenomD_holdout_1 | 8.9628849e-07 | 1.013514e-06 | FAILED |
| IMRPhenomD_holdout_2 | 9.6292775e-07 | 1.1433329e-06 | FAILED |
| IMRPhenomXAS_holdout_3 | 3.8380711e-14 | 3.1036481e-07 | PASSED |
| IMRPhenomXAS_holdout_4 | 9.4909419e-09 | 4.1285273e-07 | PASSED |
| IMRPhenomHM_holdout_5 | 0.051825363 | 0.050805122 | FAILED |
| IMRPhenomXHM_holdout_6 | 0.23418847 | 0.23294282 | FAILED |
| IMRPhenomD_NRTidal_holdout | 1.6021423e-07 | 3.8956121e-07 | PASSED |
| IMRPhenomD_NRTidalv2_holdout | 1.5758795e-07 | 3.8809132e-07 | PASSED |
| IMRPhenomNSBH_holdout | 0.0013294683 | 0.0013295885 | FAILED |
| IMRPhenomPv2_holdout | 1.418307 | 1.4506322 | FAILED |
| IMRPhenomXP_holdout | 0.38457136 | 0.51054504 | FAILED |
| IMRPhenomXPHM_holdout | 0.40563132 | 0.51307356 | FAILED |

PhenomD's largest holdout polarization error decreased from about `4.32e-4`
in the initial report to `1.23e-5`. Two other PhenomD holdouts pass that
polarization bound but still fail detector projection bounds. This improvement
therefore adds no strict passing case. Higher-mode and precession residuals
are not uniformly monotonic as published equations replace inherited choices.

The independent [Kerr component measurements](verification/pass-3/kerr-components.json)
show a maximum relative complex frequency difference of `3.40401e-6` across
60 supported LAL QNM cases. The angular binary comparison, after removing a
constant phase for diagnosis only, differs by about `1e-4` to `1e-3` at nonzero
spin. Neither result certifies complete Kerr waveform parity. Separate author
data holdouts and angular differential-equation tests validate the new
components without fitting any native coefficient to binary residuals.

An additional [time-domain Kerr probe](verification/pass-4/kerr-waveforms.json)
uses a primary-derived finite-Q energy normalization and the public
`SimBlackHoleRingdown` API. Supplying phase zero to both APIs gives a maximum
relative L2 difference of `2.00374`; a separately recorded fixed native phase
of pi gives `0.0298245`. Neither probe optimizes alignment. The native phase,
angular gauge and reference energy semantics are not established as identical,
so these diagnostics do not replace the unresolved strict reference case.
The [energy conversion and its independent review](pass-3-ringdown-energy.md)
are complete, with sphere/time-integrated derivative tests.

The new [EOB convergence experiment](verification/pass-6/eob-convergence/result.json)
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
   singular limits and precession-aware carrier/remnant treatment. Published
   Table IV conventions and tidal initial-condition gradients are now tested.
2. Resolve author-defined higher-mode fit normalization, collocation conditions,
   phase constants and mode-mixing conventions. Unsupported alternatives remain
   explicitly experimental rather than selected from oracle residuals.
3. Establish EOB target choices for the inherited 22 phase, ell=8 spin
   truncation and selected Hofmann fit block; then resolve physical phase/time
   compatibility and numerical accuracy against that defined model. Spinning
   radiation, peak refinement, published final mass and NQC/merger integration
   from the earlier [assessment](pass-4-eob-integration-assessment.md) are now
   implemented in the explicit experimental driver. A complete calibrated
   tidal TEOBResumS reconstruction remains unfinished. The existing ROM must
   be regenerated only from a defined, validated target.
4. Establish the complete Kerr waveform energy, angular and phase conventions
   against a public reference; define a separate independent higher-mode target
   for TaylorF2_HM.
5. Independently source remaining inherited high-precision calibration assets.
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
python tests/parity_harness.py --strict --json-out docs/reviews/2026-09-12-clean-room/verification/pass-6/catalog.json
python tests/parity_harness.py --scope=holdout --strict --json-out docs/reviews/2026-09-12-clean-room/verification/pass-6/holdouts.json
python scripts/verify_kerr_components.py --json-out docs/reviews/2026-09-12-clean-room/verification/pass-3/kerr-components.json
python scripts/verify_kerr_waveforms.py --json-out docs/reviews/2026-09-12-clean-room/verification/pass-4/kerr-waveforms.json
python scripts/verify_eob_published_convergence.py --output-dir docs/reviews/2026-09-12-clean-room/verification/pass-6/eob-convergence
python scripts/verify_eob_published_waveforms.py --convergence-dir docs/reviews/2026-09-12-clean-room/verification/pass-6/eob-convergence --json-out docs/reviews/2026-09-12-clean-room/verification/pass-6/eob-waveforms.json
```

The recorded exit codes are 0, 1, 1, 0, 0, 0 and 0. Diagnostic scripts'
successful execution does not imply parity acceptance. Exact pass-6 process metadata,
input hashes, versions, case definitions, failures and logs are under
[pass-6 verification](verification/pass-6/); the Kerr probes retain
their earlier evidence paths. Earlier runs remain in place. The
[worktree hash inventory](verification/worktree-sha256.json) records the current
changed files; its [initial snapshot](verification/initial-pass-worktree-sha256.json)
is preserved separately. No commit or pull request has been created. All
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
