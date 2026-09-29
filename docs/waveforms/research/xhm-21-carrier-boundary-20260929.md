# XHM (2,1) carrier and fit boundary, 2026-09-29

This audit starts from integration `c70bceae`. It compares the published
[XHM v2 paper](https://arxiv.org/html/2001.10914v2), the independent scalar
PN reference, native Torch/JAX, and public binary calls in pinned LALSuite
`7.26.15` / LALSimulation `6.2.1` on `len`. It uses no LAL source or historical
source diff. The binary's default XHM release is `122022`; `122019` is the
author-era control. The comparison uses three aligned-spin cases:

| Case | Masses in solar masses | Aligned spins | Reference frequency (Hz) |
| --- | --- | --- | ---: |
| Probe | 36, 29 | -0.1, 0.2 | 37.3 |
| Catalog | 40, 20 | 0.3, -0.1 | 20 |
| Holdout | 55, 11 | -0.4, 0.2 | 27 |

## Carrier and mode selection

The paper's positive-frequency carrier is Fourier \(h_{\ell,-m}\), while the
native `return_unprojected_modes` dictionary labels that positive-frequency
carrier with `(l,+m)` and its conjugate partner with `(l,-m)`. The isolated
carrier comparison therefore uses native `(2,+1)` and LAL's public
`SimIMRPhenomXHMFrequencySequenceOneMode` with `(2,-1)`. The native
`modes=((2,1),)` polarization path includes both signs. A public LAL mode
array with only `(2,+1)` differs from one with both signs by 8.7% in plus and
10.6% in cross for the probe, so mode-array selection is material but cannot
account for the original roughly eleven-fold native (2,1) norm. The native
positive-key convention is documented here because changing its public keys
would also change XPHM consumers; it is not the amplitude error fixed below.

The scalar reference and both native backends agree on the paper's Appendix E
normalized (2,1) PN amplitude at 25, 50, and 100 Hz across these cases. The
normalization to the paper's leading \(A_{22}\) also reproduces the public
`122019` node values; no missing factor of two or \(\pi\) was found.

## Decisive inspiral fit check

For each case, the three old-paper nodes are \(0.5 f_{\rm ins}\),
\(0.75 f_{\rm ins}\), and \(f_{\rm ins}\). Public one-mode `122019` amplitudes,
divided by the leading \(A_{22}\), reproduce the inherited **fit 3, fit 2,
fit 1** surfaces at those ascending frequencies, respectively, when evaluated
with the paper's normalized \(\chi_{\rm PN}\) (`chiPNHat`). The old production
assembly instead used **fit 1, fit 2, fit 3** and the merger-spin `S`. For
example:

| Case | Node frequencies (Hz) | Public `122019` normalized amplitudes |
| --- | --- | --- |
| Probe | 30.213421, 45.320131, 60.426841 | 0.0127204854, 0.0153658458, 0.0183130198 |
| Catalog | 33.415412, 50.123117, 66.830823 | 0.0169910888, 0.0188865787, 0.0216642783 |
| Holdout | 17.689250, 26.533875, 35.378501 | 0.0463215425, 0.0457680534, 0.0462397678 |

The inherited surfaces with reversed indices and `chiPNHat` reproduce all
nine public `122019` values to floating-point precision. The Torch and JAX
(2,1) assembly now uses that order and spin variable. Other modes keep their
previous fit arguments. A fixed-node regression requires both backends to
reproduce the public `122019` amplitudes without loading LAL at test time.

## Binary outcome and remaining release boundary

The following values are relative L2 amplitude errors of the unprojected
native (2,1) carrier against the public `(2,-1)` one-mode binary on a
20–300 Hz grid at 0.25 Hz spacing. The narrow band measures the inherited
inspiral wiring; the full band also includes intermediate and ringdown code.

| Case | Release | 20–50 Hz before → after | 20–300 Hz before → after |
| --- | ---: | ---: | ---: |
| Probe | 122019 | 0.396639 → 0.001435 | 9.807357 → 1.218208 |
| Probe | 122022 | 0.391522 → 0.015577 | 9.739417 → 1.211758 |
| Catalog | 122019 | 0.235572 → 0.000341 | 2.002847 → 3.087929 |
| Catalog | 122022 | 0.224478 → 0.010646 | 1.977240 → 3.053650 |
| Holdout | 122019 | 0.068367 → 0.068500 | 1.988522 → 1.967847 |
| Holdout | 122022 | 0.060907 → 0.060246 | 1.996590 → 1.975875 |

The catalog full-band error grows despite the exact old-release inspiral
nodes. That is a measured boundary of this local correction, not evidence for
reversing the verified fit mapping. In the probe's isolated **projected**
(2,1) polarization comparison against default LAL, plus amplitude L2 falls
from 10.695721 to 1.359491 and plus norm ratio from 10.996537 to 1.958647;
cross amplitude L2 falls from 8.784065 to 1.070997. The isolated (2,2)
plus amplitude L2 remains 0.009737. Neither result qualifies the assembled
XHM or XPHM waveform.

The unchanged four-case default-release binary gate, measured through
`SimInspiralChooseFDWaveform`, remains **0/4**. Its complex plus-polarization
L2 values on the same pinned host and environment are:

| Frozen case | Before → after | Strict limit |
| --- | ---: | ---: |
| XHM catalog | 0.374023 → 0.380883 | 0.000001 |
| XPHM catalog | 0.228818 → 0.295708 | 0.000001 |
| XHM holdout | 0.546374 → 0.544842 | 0.000001 |
| XPHM holdout | 0.504113 → 0.437500 | 0.000001 |

The companion mismatch limit remains \(10^{-8}\); all four cases fail both
limits. The after receipt is `/tmp/xhm21_frozen_after_c70bceae.json` on
`len`.

The binary `122019` and `122022` (2,1) amplitudes differ. Public option
lookups reporting sub-option labels `122018` for both releases do not prove
that their binary amplitude branches are identical. The default `122022`
geometric Fourier-strain node provider and its region cuts are still absent
from the native production assembly; the [component boundary
receipt](../../validation/xhm_carrier_components_20260925.json) describes
what is missing. The native old-paper inspiral cutoff is 60.426841 Hz for
the probe, versus 34.174989 Hz for the release-122022 clean assembly
contract. This patch does not transplant the new cutoff without its matching
calibration. The existing [strict qualification
record](xhm-xphm-qualification-20260929.md) therefore remains unpromoted.

Reproduction artifacts on `len`: the before and after probe logs are
`/tmp/xhm21_probe_c70bceae.log` and
`/tmp/xhm21_after_probe_c70bceae.log`; isolated polarization logs are
`/tmp/diffgw-xhm21-modes-c70bceae.log` and
`/tmp/xhm21_modes_after_c70bceae.log`. The patched checkout is
`/tmp/diffgw-xhm21-after-c70bceae`, using the pinned environment at
`/tmp/diffgw-waveform-library-1315f6e/venv-latest`.
