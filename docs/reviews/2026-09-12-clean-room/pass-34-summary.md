# Pass 34: clean-room parity status

Date: 2026-09-13

Full LAL parity is not achieved. The fresh strict catalog passes **11/21**
cases and the frozen holdout passes **4/13** cases. All strict gates remain
unchanged: complex relative L2 and relative norm error at most `1e-6`, and
flat and named-PSD mismatch at most `1e-8` for polarizations and detector
projections.

## Retained source-backed changes

- `IMRPhenomPv2` now uses the published physical in-plane spin
  `S_perp = m1^2 chi_p` in its opening angle. The former path multiplied by
  `A1` twice. Exact aligned-spin cones and their gradients are finite. Fresh
  black-box measurements show no material reduction in the remaining Pv2
  residual, so the correction is retained for its source derivation rather
  than an oracle improvement.
- `TaylorF2_HM` now supports eight leading-PN modes:
  `(2,2)`, `(2,1)`, `(3,1)`, `(3,2)`, `(3,3)`, `(4,2)`, `(4,3)`, and `(4,4)`.
  The new `(4,2)` term, SPA phase, and spin-weighted spherical-harmonic
  projection follow the published Fourier-domain formula. `(4,1)` remains
  excluded because the available time-domain formula does not uniquely fix
  this implementation's shared SPA normalization and negative-`m` phase
  convention.
- `IMRPhenomNSBH` treats `f_ref=0` as the documented lower-band sentinel and
  rejects negative or nonfinite values. The SEOBNRv4 published and native
  drivers likewise reject nonpositive or nonfinite explicit start
  frequencies before trajectory construction.

The direct reports and independent reviews are
[Pv2](pass-32-pv2-strict.md),
[Pv2 review](pass-32-pv2-independent-review.md),
[Taylor `(4,2)`](pass-32-taylorf2-hm-42.md),
[Taylor review](pass-32-taylorf2-hm-42-review.md),
[Taylor `(4,1)` blocker](pass-34-taylorf2-hm-41.md),
[Taylor `(5,5)` blocker](pass-35-taylorf2-hm-55.md),
[NSBH sentinel](pass-31-nsbh-reference-sentinel-review.md), and
[SEOBNRv4 driver](pass-32-seobnrv4-driver.md).

## Fresh strict measurements

The complete JSON results and logs are under
[`verification/pass-34-final`](verification/pass-34-final).

### Catalog failures

| Case | Max polarization L2 | Max detector L2 | Result |
| --- | ---: | ---: | --- |
| IMRPhenomHM | 0.0045409491 | 0.0045121740 | FAILED |
| IMRPhenomXHM | 0.38200117 | 0.37948173 | FAILED |
| IMRPhenomPv2 | 0.25723438 | 0.25699423 | FAILED |
| IMRPhenomXP | 0.072548555 | 0.072424492 | FAILED |
| IMRPhenomXPHM | 0.16092642 | 0.16112354 | FAILED |
| IMRPhenomNSBH | 2.9516980e-5 | 2.9649914e-5 | FAILED |
| KerrRingdown | - | - | UNRESOLVED_ORACLE |
| SEOBNR_ODE | 1.0206987 | 1.0205837 | FAILED |
| SEOBNR_ROM | 1.0019032 | 1.0022284 | FAILED |
| TEOBResumS | 1.0009162 | 1.0007424 | FAILED |

The other 11 catalog cases pass.

### Holdout failures

| Case | Max polarization L2 | Max detector L2 | Result |
| --- | ---: | ---: | --- |
| IMRPhenomD 0 | 1.2321902e-5 | 1.2443720e-5 | FAILED |
| IMRPhenomD 1 | 8.9629199e-7 | 1.0135173e-6 | FAILED |
| IMRPhenomD 2 | 9.6293422e-7 | 1.1433393e-6 | FAILED |
| IMRPhenomHM 5 | 0.019464597 | 0.019157343 | FAILED |
| IMRPhenomXHM 6 | 0.65865751 | 0.65334530 | FAILED |
| IMRPhenomNSBH | 0.0017579727 | 0.0017580928 | FAILED |
| IMRPhenomPv2 | 1.4183070 | 1.4506322 | FAILED |
| IMRPhenomXP | 0.40327128 | 0.51091129 | FAILED |
| IMRPhenomXPHM | 1.1851135 | 1.2770461 | FAILED |

The other four holdouts pass.

## Residual localization and blockers

Public one-mode API measurements show exact agreement for the XHM `(2,2)`
carrier after the documented convention conversion. Every tested higher mode
retains a large frequency-dependent residual: optimized full-band complex L2
is `0.857` to `0.998`, and the worst high-frequency median amplitude ratio is
`0.160` for diagnostic `(4,4)`. The primary paper and archived author assets
do not supply all intermediate-target preprocessing, absolute phase/time
gauge, and `(3,2)` mixing metadata. The
[localization](pass-33-xhm-localize.md) and
[independent review](pass-34-xhm-review.md) therefore retain no speculative
runtime edit. A separate [PhenomHM mode localization](pass-35-hm-mode-localization.md)
finds the largest aligned residual in `(2,1)` but likewise identifies no
missing source-defined equation.

For XPHM, the 22-only residual is already `0.048649`/`0.061171` in plus/cross,
while the complete waveform is `0.152528`/`0.157206` for the diagnostic probe.
This prevents a unique attribution to Euler angles or projection. Pv2, XP,
and XPHM need an independent coprecessing carrier specification with the exact
precession, remnant, and degenerate-row continuation choices. See the
[XPHM localization](pass-33-xphm-localization.md) and
[ranked evidence audit](pass-34-failure-triage.md).

The remaining PhenomD errors require higher-precision published fit assets;
the EOB, ROM, and TEOB paths require complete calibrated dynamics and merger
metadata; and KerrRingdown needs independent excitation, normalization, and
twin/overtone waveform data. These inputs cannot be reconstructed uniquely
from the allowed papers. Fitting them to public LAL residuals would violate
the clean-room boundary. The [Pass 35 summary](pass-35-summary.md) records the
subsequent report-only PhenomHM, Taylor `(5,5)`, and NSBH audits.

## Verification and provenance

The full ordinary suite passed **1,248 tests with 9 skipped** across eight
workers. Shard lists and logs are under
[`verification/pass-33/full-suite-run3`](verification/pass-33/full-suite-run3).
Focused independent reviews also passed, and `git diff --check` is clean.

No LAL, PyCBC, TEOBResumS, or other waveform implementation source was read or
copied. Public LAL APIs were used only as black-box measurements. No runtime
constant was fitted to an oracle residual, no acceptance gate was relaxed,
and the deferred NR run was not started.
