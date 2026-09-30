# NRTidalv2 Pv2 public-assembly boundary

## Scope and status

Specification **`nrtidalv2-pv2-public-assembly/v0`** is a source audit and
**pending** whole-waveform gate, not an executable waveform reference. It
concerns positive-frequency, precessing `IMRPhenomPv2_NRTidalv2` after the
separately checked aligned BNS carrier and positive-22 projection. Neither
Torch nor JAX currently dispatches that BNS approximant. This page freezes
the unresolved choices rather than choosing them from a binary residual.

## Inputs, conventions, and options

The proposed gate would use positive physical frequencies in Hz, component
masses in solar masses, dimensionless six-component spins, nonnegative tidal
deformabilities, distance in Mpc, inclination and coalescence phase in radians,
and an explicit positive reference frequency in Hz. It would compare complex
`h_plus` and `h_cross` in CPU FP64 **without** phase, time, polarization, or
amplitude alignment. The inherited Pv2 note defines a J-aligned inertial
frame at `f_ref`; its Eq. (6) sets the minimal-rotation angle derivative and
its Sec. 5 fixes the observer/J chart. The native BBH drivers expose
`coalescence` and `reference_frequency` carrier-phasing options. No published
BNS rule examined here selects one, fixes the additive finite-size phase at
`f_ref`, or specifies a merger-time convention for the composed BNS carrier.

## Mathematics and components

[Dietrich et al. (2019), Sec. III.4](https://arxiv.org/html/1905.06011)
instructs adding tidal corrections to the aligned point-particle model
*before* applying the inherited Pv2 rotation. The aligned finite-size
composition and supplied-angle projection are independently gated in the
[carrier](nrtidalv2-pv2-aligned-carrier.md) and
[projection](nrtidalv2-pv2-positive-22-projection.md) specifications.
The [Pv2 technical note, Eqs. (6), (13), (26), (31)–(36)](https://dcc.ligo.org/public/0122/T1500602/004/PhenomPv2_technicalnotes.pdf)
defines the BBH frame/rotation, but does not prescribe a natural-cubic
remnant-frequency phase slope for a BNS carrier. Dietrich's time and phase
maximization in Eq. (31) is a validation overlap, not a generator origin.

| Component ID/version | Source equation or algorithm | Variant | Status | Scalar reference | Torch | JAX | Maintained check or blocker |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `nrtidalv2-pv2-aligned-carrier/v1` | Dietrich Eqs. (17)–(29), Sec. III.4 | Paper Pv2, supplied BBH carrier | Conditional executable | `nrtidalv2_pv2_terms.assembled_carrier_from_bbh` | `assemble_nrtidalv2_pv2_aligned_carrier` | Same named helper | `test_nrtidalv2_pv2_aligned_carrier_math_reference.py` |
| `nrtidalv2-pv2-public-angles/v0` | Pv2 note Eqs. (6), (26), (31)–(36); Dietrich Sec. III.4 | Paper-exact BNS angles | Pending | None | BBH driver uses compatibility choices | Same | Longitudinal-spin and orbital-angular-momentum conflicts below |
| `nrtidalv2-pv2-public-time-origin/v0` | No BNS slope/interpolation or phase-origin algorithm in the cited papers | BNS carrier at `f_ref` and merger | Pending | None | Ten-node **D-only** slope in BBH driver | Same | Must select D-only or BNS-carrier slope, interpolation and sign |
| `nrtidalv2-pv2-public-assembly/v0` | Dietrich Sec. III.4; Pv2 note Eq. (13) | Complete BNS polarizations | Pending | None | No BNS dispatch | No BNS dispatch | Requires both pending contracts |

The time-origin ambiguity changes the *complex* result. Let `S'` denote the
derivative at `f_RD` of the native ten-node natural cubic. The existing BBH
driver uses `k_D = S'[psi_D]` and multiplies its carrier by `exp(i f k_D)`.
If that same operation is instead applied to the BNS phase
`psi_D + psi_finite`, linearity gives
`k_BNS - k_D = S'[psi_finite]`; the two outputs differ by
`exp(i f S'[psi_finite])`. Neither source fixes which operation, spline
sample interval, or interpolation rule defines the *BNS* model. An overlap
maximized over time cannot resolve this unaligned waveform contract.

## Implementation mapping

The BBH Torch driver
`src/torchwave/approximants/phenompv2_waveform_clean.py` and JAX driver
`src/jaxwave/approximants/phenompv2_waveform.py` both sample the D-only
phase at ten nodes on `[0.8 f_RD, min(1.2 f_RD, 0.2/M)]`, differentiate a
natural cubic at `f_RD`, and use a coalescence-phase D carrier by default.
Their NNLO angle input is `chi_eff / X_heavy`; the Pv2 note Eq. (32) instead
specifies the heavier body's aligned spin for those angles. The note Eq. (26)
prints the 2PN orbital-angular-momentum coefficient
`3.373 - 19 eta/8 - eta^2/24`. The native coefficient is
`27/8 - 19 eta/8 - eta^2/24`, a mixed convention: the distinct
[Pratten et al. (2020), Eq. (27)](https://arxiv.org/html/2004.06503)
prints `27/8 - 19 eta/8 + eta^2/24` for IMRPhenomXPHM. None of these three
is silently designated the paper-exact Pv2_NRTidalv2 choice. The BBH
compatibility driver and historical D default remain unchanged.

## Evidence and limitations

This is a direct source/implementation audit, **not** a Torch/JAX numerical
pass, binary comparison, or physical validation. The sources establish
finite-size-before-rotation and the BBH source-frame geometry. They do not
establish the D-only versus BNS slope, BNS phase/time origin, or a single
angle coefficient matching the native drivers. A binary oracle may diagnose
an explicitly supported `IMRPhenomPv2_NRTidalv2` option, but must not supply
missing mathematics by fitting a residual. The source-versus-Fig. 7 Love–Q
and merger-frequency conflicts in the aligned-carrier spec remain separate.

The next qualifying step is to obtain a citable author equation/technical
algorithm fixing the BNS slope and phase gauge, then name a paper-literal
angle variant (including Eq. (32) and Eq. (26), or a justified correction).
Freeze that variant, input chart, frequency support and strict unaligned
complex tolerance before constructing a scalar reference and Torch/JAX
gate. If only a binary-specific convention is available, label its gate as
binary compatibility rather than published-math qualification.

## Reproduce the checks

Read Dietrich Sec. III.4 and Eq. (31), Pv2 note Eqs. (6), (13), (26),
(31)–(36), and Pratten Eq. (27) at the linked source revisions above. For
native mapping, inspect `_phase_time_shift`, `_L2`, `chil`, and the carrier
calls in the two driver paths named above. The executable *conditional*
carrier/projection gates remain in their linked specs; no public BNS test is
added by this source audit. A change to a source convention or origin
requires versioning this pending spec before any numerical comparison.
