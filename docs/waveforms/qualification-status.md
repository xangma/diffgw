# Qualification status and next steps

## Closure standard

Each family is reviewed against the specific public model, version, modes,
parameters and waveform options it claims. A completed *bounded scope* needs
the cited published equations and conventions, an independently evaluable
reference where the public information permits one, tests connecting that
reference to the native Torch and JAX paths, and recorded external comparisons
with their actual thresholds and results. The reference must not be inferred
solely from the implementation being checked. A passing internal parity test
does not establish agreement with a paper, an independent code, or numerical
relativity.
Any claim that a family matches another code's offering also needs a
versioned, option-specific comparison of supported modes, parameters and
assets; matching a model name alone is insufficient.

If a complete public construction, calibration asset, or convention cannot be
identified, record the exact missing input and close only the components that
can be independently evaluated. Keep binary compatibility, historical-model
fidelity, numerical convergence, differentiability and physical accuracy as
distinct claims. Failed gates remain visible; neither a newly chosen tolerance
nor a match to one sampled binary case silently widens a qualified domain.

The current family reviews are being developed in separate worktrees. Their
family-specific math, reference, backend, test and evidence changes will be
integrated here only after review. Shared dispatch, coverage and release claims
are owned by this integration branch so parallel work cannot overwrite them.

This 29 September 2026 review separates implemented mathematical scopes from
remaining implementation, provenance and physical-accuracy questions. It
summarizes the evidence available at integration revision `9c4a3a8`; the linked
specifications and receipts define the actual options, samples and thresholds.
It does not add catalog claims or expand any validated domain.

## Supported scopes and remaining work

| Family or scope | Supported implementation evidence | Remaining work outside that scope |
| --- | --- | --- |
| [HM published composition](../math/imrphenomhm-waveform.md#implementation-closure-2026-09-28) | Review closed for 22 CPU FP64 sampled claims, including separately versioned orbital and SPA conventions | Known legacy covariance defect, inherited-default provenance and untested domains remain explicit. Failed NR gates remain failed; mixing/refitting is separate research. |
| [TaylorF2 and extensions](../math/taylorf2.md), [eccentric](../math/taylorf2-eccentric.md), [HM inspiral](../math/taylorf2-hm.md) | Versioned base and extension references with their own sampled claims | Additional physics, general parameter coverage, and independent external eight-mode HM comparison are separate claims. |
| [D](../math/imrphenomd-intrinsic.md) and [XAS](../math/imrphenomxas.md) | Versioned carrier compositions and scoped reference/binary comparisons | Supplied fit/QNM inputs do not establish every upstream calibration or whole-domain error bound. |
| [XP](research/xp-v223-reference-closure.md) | Scoped v223 geometry, selected generic chart and nonzero-azimuth composition; internal near-aligned continuity now passes | Generic equal-mass remnant and coalescing-root continuation, other versions and binary boundary cases remain open. |
| [XHM/XPHM](research/phenom-higher-modes.md) | Independent amplitude, supplied-target phase and projection components; [precession reference scopes](research/xphm-precession-reference-closure.md) | Full mixed-mode carrier, calibrated surfaces and release-specific asset mapping remain incomplete. Recorded binary failures are not NR results. |
| [P/Pv2 and SpinTaylor](research/phenom-precessing.md) | Native routes and bounded component/comparison evidence | Complete independent frame/angle/carrier composition; SpinTaylor RHS, integration and gradient contract. |
| [XPNR](research/imrphenomxpnr.md) | Source-linked scalar and native component checks | Public terminal-state/attachment/carrier composition and inherited calibration provenance remain incomplete. A test threshold is not measured physical accuracy. |
| [T/THM/TP/TPHM](research/imrphenomt-family.md) | Selected inspiral, supplied-input ringdown, rotations, angle/RHS and trajectory components | Full amplitude/merger/mode composition and fit provenance; recorded precession-convention and binary discrepancies remain unresolved. |
| [NRTidal v1](../math/nrtidalv1-merger-taper.md) | Printed nonspinning tidal factor/taper, bounded premerger carrier checks and [versioned source-revision audit](../validation/nrtidal_family_qualification_20260929.md) | The historical taper convention retains its documented bounds; broader spinning/full-carrier scopes remain separate. |
| [NRTidal v2 paper-D](../math/nrtidalv2-paper-d-composition.md) | Printed spinning tidal quotient through taper against both frontends; [derivative/JIT and conditional Pv2 assembly evidence](../validation/nrtidal_family_qualification_20260929.md) | Nonzero-tide D taper binary gates still fail; default amplitude provenance and full precessing Pv2 assembly remain unresolved. |
| [NSBH](research/tidal-eccentric-higher-modes.md) | Bounded source-derived disruption, amplitude-factor, QNM, phase and rounding components; [conditional polarization composition](../validation/nsbh_polarization_composition_20260929.md) from supplied carriers passes on both backends | Native/paper amplitude and join variants, extrapolation guards and full waveform composition require source resolution. |
| [Kerr](../math/kerr-ringdown-direct-class.md), [CW](../math/continuous-wave-direct-class.md), [memory](../math/memory-dominant-22-direct-class.md) | Cataloged direct-class slices retain their sampled claims; the review corrected memory's per-waveform distance broadcasting in both backends | Wider mode, detector, timing and memory-coupling physics is outside those slices. The legacy memory coupling diagnostic remains an expected failure. |
| [EMRI](../math/emri-kludge.md) | Leading flux components | An explicit complete target and orbit/harmonic/response composition remain open. |
| [ppE](../math/ppe-phase-dispersion.md) | [Versioned public-equation reference and scalar oracle](../validation/ppe-supported-equations-v2.md) for inspiral amplitude, phase, dispersion and supplied-carrier composition; Torch/JAX per-waveform batching passes | The published inspiral equation does not uniquely specify merger/ringdown branches. No external full-waveform parity or generic full-IMR claim is established. |

The [current public dispatch inventory](coverage.md) is distinct from the
[retired prototypes](research/retired-release-surface.md). Retired native EOB,
TEOB and ROM pilots are not unfinished release routes that should be silently
restored. The retained [official SEOBNR ROM evaluator](../external-data.md)
has its own external-asset contract.

## Recovered native SEOBNR development state

The separate branch was recovered at
[`093f44d`](https://github.com/xangma/diffgw/tree/093f44d3b2706e2f1c3224d523eedc69ea24ed2b/dev/native_rom).
Its current
[historical-fidelity decision](https://github.com/xangma/diffgw/blob/093f44d3b2706e2f1c3224d523eedc69ea24ed2b/dev/native_rom/evidence/2026-09-27/historical_fidelity_release.json)
supersedes the interrupted chat's earlier JAX failure as the status source.
The frozen whole-path gradient check passes: eager/JIT complex relative L2 is
`3.4370e-11` against `1e-10`. This is one initial-radius derivative with fixed
mesh and attachment decisions, not general physical-parameter autodiff.
The old NQC coefficient sensitivity is not thereby declared repaired.

Four release gates remain unmet: source completeness, numerical convergence,
whole-waveform fidelity and independent ROM holdout. The saved numerical
receipt misses its `1e-6` complex-L2 budget: ODE medium-to-fine is `8.7485e-6`
and output-grid medium-to-fine is `6.0815e-6`, with nonmonotonic grid refinement.
The latest flux audit has 31 of 35 eligible source rows; four even-m ell=8
rows still need eligible source/reuse evidence, and historical runtime
selection is a separate unresolved contract. Hamiltonian, QNM and initial-data
source questions also remain. These are concrete blockers, not requests for
another NR campaign.

The recorded SXS:BBH:2569 overlap near `0.98914` remains below `0.99`, including
for the pinned historical public model. That physical result must be disclosed
separately from historical fidelity. It does not authorize adjusting physics
or training a new ROM before the generator's required gates pass.

## Work order

The integration review corrected one concrete API defect: memory distances
shaped `(B,)` now scale waveforms shaped `(B, N)` along the batch axis,
including the previously silent `B == N` case. Independent linear-mode,
batch-versus-scalar, distance-gradient and JAX JIT regressions cover the fix.
On len, the maintained memory/reference/catalog test selection passed
27 tests with one existing strict expected failure for the legacy memory
coupling formula. This does not change catalog claims or physical gates.

1. Correct demonstrated implementation/API defects and stale summaries; review
   the integration PR against its actual scoped claims.
2. Preserve SEOBNR's passed frozen gradient gate. Resolve one explicit public
   source/provenance blocker before commissioning more historical-port work;
   do not restart old diagnostics from stale chat messages.
3. For the next independent composition, prefer a bounded NRTidal printed-D
   scope: explicitly choose a tidal quotient or a full independently evaluated
   carrier. The existing quotient result does not already certify the latter.
4. Keep missing calibration assets and source definitions visible. New physics,
   refits and broader NR accuracy need separately agreed models and holdouts.
