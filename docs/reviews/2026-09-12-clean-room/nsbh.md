# Bounded PhenomNSBH equation review

Reviewed on 2026-09-12 against freshly downloaded primary papers. This review
used the inherited implementation and primary literature, with no LAL/PyCBC
implementation, previous calibration notes, waveform-oracle calls, or residual
fitting. It establishes the two corrections below, not complete numerical or
provenance certification of PhenomNSBH.

The later [pass-2 phase review](pass-2-baseline.md) replaces the duplicated
phase described below. Numerical checks in this document describe pass 1.

## Primary inputs

| Paper | Fresh source | SHA256 of downloaded PDF |
|---|---|---|
| Thompson et al., PRD 101, 124059 | [arXiv:2002.08383v2](https://arxiv.org/pdf/2002.08383v2) | `7e8bf4ba34a7fb98d54fa1a405267fdb4d06a78e1b164f57361b0b2abebd34e2` |
| Pannarale et al., PRD 92, 084050 | [arXiv:1509.00512v1](https://arxiv.org/pdf/1509.00512v1) | `12b8266f53b55f301c6a4fa4a9a46ffcef28a41e23624f6607e9f3355282fc0c` |
| Santamaria et al., PRD 82, 064016 | [arXiv:1005.3306v3](https://arxiv.org/pdf/1005.3306v3) | `39b717079e726dedcf8b184acd2f953dbe8e6bf2bc8adab1213ecc4257363fc1` |

No fit coefficient was inferred or adjusted from waveform differences.

## Corrections

1. **Amplitude-window units.** Pannarale Sec. III defines frequencies in
   total-mass units; Eq. (13) and its discussion set the window width to
   `d + sigma_tide`, with `d = 0.015`. Thompson Eqs. (4), (30) retain that
   window. The inherited implementation multiplied this dimensionless width
   by the ringdown frequency in Hz. Consequently its width in `M*f` was too
   small by a factor `M_sec*f_RD`. The new `_amplitude_window` evaluates
   `M_sec*(f_Hz-f0_Hz)` directly against the dimensionless width. At offsets
   `+/-width/4` in `M*f`, the window must equal `(1 +/- tanh(1))/2`, independently
   of the total mass and ringdown frequency. This changes merger amplitudes;
   the magnitude of that change depends on the merger class.

2. **Spin in disruption fits.** Thompson Sec. II defines `chi` as the initial
   BH spin, for a nonspinning NS. Eqs. (21)-(24), defining `x_ND`, `x'_ND`,
   `x_D`, and `x'_D`, now use `chi1`. They previously used the mass-weighted
   effective spin. The PhenomC amplitude coefficients continue to use their
   own effective spin; Santamaria Appendix A defines that distinct quantity.

Also corrected the compactness/root equation references to Thompson Eqs. (32)
and (10), and removed the unconditional claim that a fixed-count Newton
iteration guarantees convergence to the largest positive root.

## Checks

`tests/test_clean_room_nsbh.py` checks the analytic tanh-window values at
specified total-mass frequency offsets for two masses and widths, complementary
windows, compactness's black-hole limit and C1 join, scalar/batched waveform
agreement through merger, and finite amplitude derivatives with respect to BH
spin and tidal deformability. The two waveform examples use masses 5/1.4 and
7/1.4 solar masses, BH spins 0 and 0.4, and Lambda 800 and 2600.

Combined with `tests/test_clean_room_phenomd_baseline.py`: **9 passed**, using
Python 3.13 and float64 tensors. These tests have no waveform-oracle dependency.
They do not certify strict raw complex-strain parity.

The separate parent verifier measured canonical raw relative L2 error changing
from `9.77e-6` to `2.95e-6`. Its final matched comparison of the spinning 7/1.4,
chi_BH=0.4, Lambda=2600, f_ref=40 Hz, inclination=0.9 holdout measured
`1.2466e-2` before and `1.3295e-3` after. Both remain above the `1e-6` gate.
The verifier's earlier regression summary was incorrect: its `8.12e-6` baseline
used Lambda=600, a different probe. This paragraph was corrected by the parent
after running the same frozen holdout against starting revision `84ef9b3`.
The width units and BH-spin definition were rechecked against
the above primary sources after receiving the earlier summaries. The identity
`(1+tanh(4*x))/2 = sigmoid(8*x)` also confirms the implemented window factor.
These equation corrections therefore **do not establish target parity**;
no residual-driven adjustment was made to hide the disagreement. This review
did not directly invoke or inspect the verifier implementation.

## Unresolved inputs and reconstruction gaps

- **Low-Lambda pre-merger factor:** Thompson Eq. (26) prints `1.25` above
  Lambda=1 and `1 - 0.5*Lambda - 0.25*Lambda^2` below it, but its prose describes
  a smooth BBH continuation. These give one-sided values 1.25 and 0.25 at
  Lambda=1. The equation also does not explicitly include the PhenomC
  `gamma1` factor; Pannarale Eq. (15) contains `1.25*gamma1`. The inherited
  multiplication and printed minus sign remain unchanged. An author correction
  or independently published unambiguous definition is needed to resolve this.
- **Remnant selection/classification:** Thompson's amplitude workflow calls
  for the NSBH remnant fit, the London ringdown fit, then classification using
  that ringdown frequency (Table I, Eqs. (13)-(19)). The inherited code selects
  a BBH remnant in some cases using a separate QNM-based BBH frequency, then
  retains that condition during amplitude classification. No attempt was made
  to change this coupled amplitude/phase construction in the bounded review.
- **Remnant-fit provenance:** the embedded UIB coefficients and Zappa
  correction coefficients have not been independently certified here. The
  implementation's effective-spin argument in the latter also requires checking
  against the original fit's precise variable definition and author data.
- **PhenomC PN spin reduction:** the inherited PN terms use an equal-spin
  effective representation. The correct mapping of the original two-spin PN
  expansion into this NSBH implementation remains to be established.
- **Phase and conventions:** the inherited phase combines PhenomD and NRTidalv2
  coefficients, anchors an affine time shift at the requested upper frequency,
  and evaluates its reference phase using the inspiral expression. This review
  does not establish author-specified time/phase gauges, all reference-frequency
  branches, or the origin of every inherited high-precision fit digit.

Accordingly the module description now calls this a literature reconstruction
and points to this limited review instead of claiming exclusive, fully
established clean-room provenance.
