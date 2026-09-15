# Parity repair work plan

Baseline: `84ef9b3`, 12 September 2026. The original catalog reported 21/21
passing cases, but relaxed thresholds, internal references and limited mode
coverage do not establish full LAL parity.

## Implementation boundary

Three implementers were started with fresh conversation contexts and disjoint
file ownership. They may inspect inherited TorchWave code and obtain primary
papers or independently published author mathematical data. They must record
source URLs, versions, hashes for acquired assets, and equation/table references.
They must not inspect LAL/LALSimulation/PyCBC implementation source, translated
ports, historical source audits, or the existing `parity-calibration-spec.md`.
That specification explicitly cites LAL internals and is excluded as a source
for this repair. Inherited source provenance is not certified by this process.

The primary verifier uses public LAL binary APIs. Implementers receive behavioral
diagnoses, not implementation excerpts or fitted residual corrections. No oracle
regression, transplanted private coefficients, relaxed acceptance gates or runtime
oracle dependency is allowed. This is an instruction boundary, not an
OS-enforced separation of filesystem access.

The subsequent calibration-data allowance permits independently obtained,
separately licensed datasets while retaining MIT for original evaluator code.
The [asset licensing record](calibration-data-licensing.md) identifies the
bundled Cook ODC-By data and the published CC-BY ROM data route. Unverified
inherited coefficients remain unresolved; the allowance does not change the
implementation-source or oracle-regression boundary above.

## Work packages

1. **PhenomD and QNM accuracy.** Audit interpolation, published remnant fits and
   phase/time anchoring. Verify across mass ratios, spin signs and reference
   frequencies; descendants depend on this carrier.
2. **Precession.** Audit Pv2 and XP/XPHM frame geometry, angular momentum order,
   MSA/NNLO selection, initialization, reference offsets and polarization rotation.
   Preserve each prescription's identity and expose unresolved compatibility.
3. **Higher modes.** Audit the distinct HM mapping and XHM calibrated mode
   constructions, all advertised modes, mixing, phase connections, projection
   normalization and equal-mass/unequal-spin symmetry.
4. **NSBH and provenance follow-up.** Audit remaining amplitude/remnant and phase
   defects from the Thompson/Pannarale primary model after shared-carrier changes.
   Independently trace inherited high-precision calibration assets; unavailable
   author data remains a documented blocker.
5. **Verification.** Restore the original CPU FP64 bounds: complex relative L2
   and relative norm error at most `1e-6`; flat and named-PSD mismatch at most
   `1e-8`. Prevent internal calculations from certifying LAL parity. Preserve
   the canonical 21-case inventory and add separately identified holdouts.
6. **Independent coverage.** TaylorF2_HM's `(2,2)` comparison certifies only its
   carrier. Kerr ringdown needs independent eigenvalue and waveform conventions.
   Native EOB integrators and their surrogate need independently defined target
   models, calibrated dynamics/radiation/merger attachment and external reference
   comparisons; self-consistency does not close these gaps.

Each package must deliver source-supported changes, meaningful component tests,
reviewed diffs and independent validation. A failing strict catalog remains a
reported failure. Completion of this repair pass is distinct from full parity.

## Initial pass

All three implementers finished and the primary verifier reviewed their changes.
The bounded baseline, precession, higher-mode and NSBH corrections are applied;
uniform verification gates, independent-reference exclusions and 13 frozen
holdouts are implemented. The full unit suite passes 383 tests. Strict catalog
and holdout acceptance remain failed at 11/21 and 4/13 cases respectively.
Independent provenance, complete model reconstruction and external coverage
remain open. The [report](summary.md) records matched before/after errors,
regressions, exact validation commands and the remaining implementation queue.

## Continued reconstruction

Following the user's instruction to continue, later passes added published
MSA terms, additional HM/XHM phase and amplitude equations, Fourier and EOB
normalization fixes, and public external EOB references. The SEOBNRv4 ROM
reference data was downloaded after explicit user approval and pinned by hash.
EOB cases now measure numerical failures against external binaries instead
of native self-references.

Further independent work provides Cook author-data QNMs, a spectral angular
Teukolsky solution, an explicit published SEOBNRv4 conservative Hamiltonian,
and DIN/Pan factorized radiation options. These are equation-supported
components; the model names do not establish complete calibrated waveforms.
Standalone pre-NQC 22 amplitude, trajectory-based NQC calibration and C1
merger/ringdown components are complete, as are detector time/geometry repairs.
The tidal prototype now shares consistent point-mass dynamics, intrinsic
batching and tidal/quadrupole initial-condition gradients.

All five published precession convention rows, reference-frequency batching,
observer azimuth and explicit unsupported NNLO-domain handling are now
implemented and independently reviewed. Finite-Q ringdown energy normalization
is also implemented and checked by independent sphere/time integration.

Pass 6 completes the published spinning flux, complex 22 phase components,
final mass and three signed final-spin fits, orbital-frequency peak detection,
trajectory NQC calibration and C1 merger/ringdown assembly. The new
`SEOBNRv4Published` API requires explicit unresolved model choices. Its
two-input convergence experiment has decreasing successive differences;
the independent LAL waveform probe fails. The [driver report](pass-6-eob-driver.md)
records conventions, numerical limits and all evidence. It supersedes the
unimplemented-component status in the earlier
[EOB integration assessment](pass-4-eob-integration-assessment.md).

The pass-6 ordinary suite passed 884 tests with five CUDA-only tests skipped;
strict runs in `verification/pass-6/` remain at 11/21 catalog and 4/13 holdout
cases. Each run records unchanged source/test input hashes. Full parity is
unfinished. Author phase/fit-selection semantics, inherited calibration
provenance, higher-mode/precession compatibility, complete tidal reconstruction
and independent Kerr waveform coverage remain open.
The [current report](summary.md) supersedes the historical initial-pass count;
the [initial report](initial-pass-summary.md) preserves its matched comparison.

## Independently generated numerical data

Pass 7 implements our own Kerr radial/angular eigenproblem solver, continuation
and spline generator. All 16 frequency branches passed frozen development,
reverse and stricter validation gates; the resulting data now supplies the
QNM runtime under MIT. External tables and binaries remain verification-only.
The direct hybrid-MECO solver is independently implemented and tested as a
standalone component. No new waveform/remnant/matter calibration is claimed.

The final ordinary suite passes 1,050 tests with eight skips; strict acceptance
remains 11/21 catalog and 4/13 holdouts. The [QNM report](pass-7-independent-qnm.md)
records the numerical certificate and device checks. The
[calibration inventory](pass-7-independent-calibration-inventory.md) and
[own-NR pilot](pass-7-own-nr-pilot.md) define the next simulation work. Host, queue,
allocation and compute budget are required before launching the pilot.
