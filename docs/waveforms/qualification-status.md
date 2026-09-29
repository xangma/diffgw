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

The bounded family reviews were developed in separate worktrees and their
reviewed math, reference, backend, test and evidence changes are integrated
here. The SpinTaylor reference-angle change was independently rederived under
the clean-room provenance rule. Shared dispatch, coverage and release claims
remain owned by this integration branch.

This 29 September 2026 review separates implemented mathematical scopes from
remaining implementation, provenance and physical-accuracy questions. It
summarizes the evidence on this integration branch; the linked
specifications and receipts define the actual options, samples and thresholds.
It does not add catalog claims or expand any validated domain.

## Supported scopes and remaining work

| Family or scope | Supported implementation evidence | Remaining work outside that scope |
| --- | --- | --- |
| [HM published composition](../math/imrphenomhm-waveform.md#implementation-closure-2026-09-28) | Review closed for 22 CPU FP64 sampled claims, including separately versioned orbital and SPA conventions | Known legacy covariance defect, inherited-default provenance and untested domains remain explicit. Failed NR gates remain failed; mixing/refitting is separate research. |
| [TaylorF2 and extensions](../math/taylorf2.md), [eccentric](../math/taylorf2-eccentric.md), [HM inspiral](../math/taylorf2-hm.md) | Versioned base and extension references with their own sampled claims | Additional physics, general parameter coverage, and independent external eight-mode HM comparison are separate claims. |
| [D](../math/imrphenomd-intrinsic.md) and [XAS](../math/imrphenomxas.md) | Versioned carrier compositions and scoped reference/binary comparisons | Supplied fit/QNM inputs do not establish every upstream calibration or whole-domain error bound. |
| [XP](research/xp-v223-reference-closure.md) | Scoped v223 geometry, selected generic chart and nonzero-azimuth composition; internal near-aligned continuity now passes | Generic equal-mass remnant and coalescing-root continuation, other versions and binary boundary cases remain open. |
| [XHM/XPHM](research/phenom-higher-modes.md) | Independent amplitude, supplied-target phase and projection components; [precession reference scopes](research/xphm-precession-reference-closure.md); [ordinary intermediate-amplitude algebra](research/xhm-xphm-qualification-20260929.md) passes on both backends | Full mixed-mode carrier, calibrated release-122022 surfaces and asset mapping remain incomplete. The four strict frozen binary cases fail; these are compatibility checks, not NR results. |
| [P/Pv2 and SpinTaylor](research/phenom-precessing.md) | Native routes and bounded component/comparison evidence; the [clean-room SpinTaylor reference-angle qualification](../validation/spintaylor_reference_angle_20260929.md) checks the paper-derived frame gauge, grid invariance and Torch/JAX gradients | Independent qualification of the full SpinTaylor RHS and carrier composition remains open. Direct complex LAL errors of about 25% for the documented case fail the whole-waveform gate. |
| [XPNR](research/imrphenomxpnr.md) | [37 versioned published-form expressions](../math/imrphenomxpnr-components.md) with SymPy/mpmath and isolated Torch/JAX checks; four option-pinned, phase-optimized binary diagnostics pass their declared threshold | Terminal-state, full attachment/carrier composition and calibrated asset provenance remain incomplete. The diagnostics do not establish strict whole-waveform or physical accuracy. |
| [T/THM/TP/TPHM](research/imrphenomt-family.md) | Selected inspiral, [supplied-input dominant merger](../validation/phenomt_merger_20260929.md), ringdown, rotations, angle/RHS and trajectory components | Full calibrated amplitude/mode composition and fit provenance; four q=10 precession binary gates still fail. |
| [NRTidal v1](../math/nrtidalv1-merger-taper.md) | Printed nonspinning tidal factor/taper, bounded premerger carrier checks and [versioned source-revision audit](../validation/nrtidal_family_qualification_20260929.md) | The historical taper convention retains its documented bounds; broader spinning/full-carrier scopes remain separate. |
| [NRTidal v2 paper-D](../math/nrtidalv2-paper-d-composition.md) | Printed spinning tidal quotient through taper against both frontends; [derivative/JIT and conditional Pv2 assembly evidence](../validation/nrtidal_family_qualification_20260929.md) | Nonzero-tide D taper binary gates still fail; default amplitude provenance and full precessing Pv2 assembly remain unresolved. |
| [NSBH](research/tidal-eccentric-higher-modes.md) | Bounded source-derived disruption, amplitude-factor, QNM, phase and rounding components; [conditional polarization composition](../validation/nsbh_polarization_composition_20260929.md) from supplied carriers passes on both backends | Native/paper amplitude and join variants, extrapolation guards and full waveform composition require source resolution. |
| [Kerr](../math/kerr-ringdown-direct-class.md), [CW](../math/continuous-wave-direct-class.md), [memory](../math/memory-dominant-22-direct-class.md) | Cataloged direct-class slices retain their sampled claims; the review corrected memory's per-waveform distance broadcasting in both backends | Wider mode, detector, timing and memory-coupling physics is outside those slices. The legacy memory coupling diagnostic remains an expected failure. |
| [EMRI](../math/emri-kludge.md) | Leading flux and [finite Newtonian eccentric quadrupole source strain](../validation/emri_newtonian_quadrupole_20260929.md), with a SymPy/mpmath reference and both native frontends | Kerr/AK/NK orbit and self-force physics, unrestricted harmonics, detector response, long-duration accuracy and external model equivalence remain open. |
| [ppE](../math/ppe-phase-dispersion.md) | [Versioned public-equation reference and scalar oracle](../validation/ppe-supported-equations-v2.md) for inspiral amplitude, phase, dispersion and supplied-carrier composition; Torch/JAX per-waveform batching passes | The published inspiral equation does not uniquely specify merger/ringdown branches. No external full-waveform parity or generic full-IMR claim is established. |

The [current public dispatch inventory](coverage.md) is distinct from the
[retired prototypes](research/retired-release-surface.md). Retired native EOB,
TEOB and ROM pilots are not unfinished release routes that should be silently
restored. The retained [official SEOBNR ROM evaluator](../external-data.md)
has its own external-asset contract.

## Recovered native SEOBNR development state

The separate native branch is at
[`aa1f365f`](https://github.com/xangma/diffgw/tree/aa1f365f/dev/native_rom).
Its [tabulated-option follow-up](https://github.com/xangma/diffgw/blob/aa1f365f/dev/native_rom/evidence/2026-09-29/tabulated_qualification_followup.md)
and [historical-fidelity decision](https://github.com/xangma/diffgw/blob/aa1f365f/dev/native_rom/evidence/2026-09-27/historical_fidelity_release.json)
give the current bounded status.
The frozen whole-path gradient check passes: eager/JIT complex relative L2 is
`3.4370e-11` against `1e-10`. This is one initial-radius derivative with fixed
mesh and attachment decisions, not general physical-parameter autodiff.
The old NQC coefficient sensitivity is not thereby declared repaired.

Four release gates remain unmet: source completeness, numerical convergence,
whole-waveform fidelity and independent ROM holdout. The selected
`spin_truncation="tabulated"` option now flows into the native CLI and its
receipts. Both completed, option-recorded numerical campaigns fail the frozen
`1e-6` ODE complex-L2 budget: `9.079206e-6` nonspinning and `4.938449e-5`
spinning. The nonspinning grid also fails its complex budget and decreasing
error check (`6.149855e-6` final); the spinning grid passes but its final ODE
amplitude L2 (`1.470160e-6`) fails. Both cases completed five solves without
runtime errors. The flux source audit remains 31 of 35 eligible rows, and
Hamiltonian, QNM and initial-data source questions remain. No whole-waveform
comparison was rerun at this exact tabulated revision; older comparisons
cannot establish parity for it.

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
3. For NRTidal, separate the independently checked printed-D tidal quotient
   from the still-failing nonzero-tide taper binary gates and the unqualified
   full precessing Pv2 assembly. Do not infer a carrier claim from the quotient.
4. Keep missing calibration assets and source definitions visible. New physics,
   refits and broader NR accuracy need separately agreed models and holdouts.
