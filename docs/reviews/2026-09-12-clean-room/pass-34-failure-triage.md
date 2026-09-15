# Pass 34: strict-failure triage and clean-room evidence blockers

Date: 2026-09-13

## Scope

This is a bounded triage of the latest available strict artifacts after the
Pass 33 audits. It uses TorchWave, the published primary papers, and the
public LAL waveform API only as a black-box measurement interface. No
LALSimulation, PyCBC, TEOBResumS, or other waveform implementation source was
inspected or copied, and no residual was used to fit a runtime parameter.

The latest strict results remain **11/21 catalog** and **4/13 holdout**. The
largest numerical failures are:

| model | catalog complex L2 | holdout complex L2 | clean-room disposition |
| --- | ---: | ---: | --- |
| IMRPhenomPv2 | 0.252 | 1.418 | Direct opening-angle factor was corrected in Pass 32; the post-patch residual is unchanged and the remaining frame/remnant prescription is not uniquely recoverable from the public projected output. |
| IMRPhenomXHM | 0.382 | 0.659 | Pass 33 mode decomposition localizes the error to higher-mode reconstruction; missing calibration preprocessing, phase gauge, and complete author fit metadata prevent a source-complete patch. |
| IMRPhenomXPHM | 0.155 | 1.185 | Pass 33 finds a carrier-level residual before higher-mode projection; a projection or Euler-angle change is therefore not uniquely supported. |
| IMRPhenomXP | 0.059 | 0.403 | Pass 28 already audited carrier/twist-up conventions; the remaining MSA/version and invalid-row continuation choices are source-incomplete. |
| IMRPhenomHM | 0.0045 | 0.0195 | Published model limitations and the unresolved mixing/transition construction are already audited; this is below the dominant precession gaps. |

The remaining failures are either the already-audited NSBH affine phase gauge,
PhenomD precision assets, and EOB/ROM/TEOB calibration inputs, or the
unresolved Kerr ringdown oracle. None supplies a new uniquely source-backed
runtime correction within this pass's scope.

## Evidence audit

### Precession family

The Pass 32 Pv2 correction implements the published identity
`S_perp = m1^2 chi_p` and the geometric `atan2(S_perp, L+S_parallel)` form.
The Pass 33 measurements show no material change in either catalog or holdout
strict residual. The remaining discrepancy is therefore not evidence for
reinstating the old extra-`A1` factor. For XP/XPHM, the Pass 28 and Pass 33
controls are decisive at the level permitted by the clean-room boundary:

* TorchWave XPHM 22-only versus public LAL XPHM 22-only is 0.048649 (plus) and
  0.061171 (cross).
* Public LAL XPHM 22-only versus public LAL XP is already 0.042301 (plus) and
  0.053757 (cross).

Thus a substantial residual exists in the underlying precessing carrier before
higher-mode assembly. A frame-angle or Wigner-factor edit would mix a carrier
error with a projection error and would not be source-backed. The missing
evidence is an independently specified coprecessing XAS/XP carrier together
with the selected MSA version, remnant convention, and complete continuation
rules for degenerate rows.

### Higher-mode XHM family

Pass 33 compares individual unprojected modes against the public one-mode API.
The `(2,2)` carrier agrees exactly, while the diagnostic high-frequency
amplitude ratios are 0.540 for `(2,1)`, 1.088 for `(3,2)`, and 0.160 for
`(4,4)`. The corresponding full-band optimized complex L2 values remain
0.997, 0.883, and 0.891. The primary XHM paper specifies the fit structure,
but the allowed author materials do not resolve all intermediate-target
preprocessing, absolute phase/time gauge, and `(3,2)` mixing-entry metadata.
Changing any of those from the residual would be calibration fitting.

### Kerr ringdown

The Kerr case is `UNRESOLVED_ORACLE`, rather than a numerical strict failure.
The prior source audit establishes that the native primitive is a restricted
equal-twin, amplitude-supplied waveform. The primary ringdown papers leave
generic twin-mode excitation coefficients, overtone weights, physical
distance normalization, and merger initial data as inputs. An independent
waveform/excitation reference is required before this case can become a strict
parity measurement. Adding a default excitation or changing the native
normalization would invent model semantics.

## Ranked next evidence

1. An independently published, coprecessing XAS/XP reference with the exact
   MSA version, remnant branch, and degenerate-row continuation. This has the
   largest likely strict impact because the Pv2/XP/XPHM failures dominate the
   catalog and holdout.
2. Complete author calibration metadata for XHM intermediate and ringdown
   targets, including the fit-target conversion and `(3,2)` mixing mapping.
   This is required before a higher-mode runtime change can be distinguished
   from a fitted residual correction.
3. Independent Kerr ringdown excitation and twin-mode data, including the
   distance and overtone convention, to replace the unresolved oracle.
4. Full SEOBNR/TEOB calibration trajectories and model-selection metadata for
   the EOB cases. These are explicitly outside this bounded pass and the
   deferred NR run remains untouched.

## Decision

No runtime change is justified. Every candidate with material strict impact
is either already corrected/audited or requires missing primary calibration
metadata. A speculative patch would violate the clean-room requirement and
could reduce one public-oracle residual while making the independent model
less defensible.

## Validation

The existing clean-room Kerr native, physical-normalization, and public API
tests pass; `git diff --check` passes. This report adds no source or test
changes and does not alter strict acceptance gates.
