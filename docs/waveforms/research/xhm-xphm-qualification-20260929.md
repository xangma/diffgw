# XHM and XPHM qualification boundary, 2026-09-29

## Supported scope

The published [XHM v2 paper](https://arxiv.org/html/2001.10914v2)
constructs aligned-spin `(2,2)`, `(2,1)`, `(3,3)`, `(3,2)`, and `(4,4)`
carriers. It inherits the XAS `(2,2)` carrier; the spherical `(3,2)`
ringdown couples to spheroidal `(2,2)` and `(3,2)` carriers. The published
[XPHM v2 paper](https://arxiv.org/html/2004.06503v2) twists those same five
coprecessing carriers using per-mode stationary-phase angles at
`f22 = 2 f / m'`, Wigner rotations, and an observer polarization-basis map.

The paper-era XPHM precession options are NNLO 101–104 and MSA 220–224,
with MSA 223 the published default. The native SpinTaylor 310/311/320/321
and PNR 300/330 routes are separately sourced extensions. The supported
precession option, remnant, and convention tuples and exact native reductions
are in the [XPHM precession closure](xphm-precession-reference-closure.md).
The XP core is an inherited dependency and is not qualified here.

## Public-code comparison, pinned by version and option

| Public implementation | Documented capability | Native DiffGW gap measured here |
| --- | --- | --- |
| [LALSuite 7.26 / LALSimulation 6.2.1](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/group___l_a_l_sim_i_m_r_phenom_x__c.html), binary distribution `7.26.15` on `len` | XHM aligned-spin five-mode polarizations and individual modes; `PhenomXHMReleaseVersion=122022` default, `122019` selectable. XPHM precession flags include NNLO `101–104`, MSA `220–224` (default `223`), SpinTaylor `310/311/320/321`, and PNR-related `330`; remnant and convention flags are separate options. | The native ordinary intermediate-amplitude algebra is reference-checked for supplied targets, but a source-pinned `122022` calibration provider is absent. The four default-option frozen full-waveform cases below fail the installed LAL binary gate. Those failures do not establish outcomes for every non-default flag tuple. |
| [PyCBC waveform documentation](https://pycbc.org/pycbc/latest/html/waveform.html), installed `2.11.0` on `len` | Exposes XHM and XPHM approximant names in frequency and time domain; its XPHM example selects `(2,2)`, `(2,1)`, `(3,3)`, `(3,2)`, and `(4,4)` through `mode_array`. Availability depends on the processing context. | The native PyCBC-style interface names alone do not demonstrate numerical equivalence to the documented XHM/XPHM provider or its selected modes. The strict comparison targets installed LAL directly, so PyCBC presentation is not an independent equation oracle. |
| [Bilby 2.8 example](https://bilby-dev.github.io/bilby/compact-binary-coalescence-parameter-estimation.html) | Selects `waveform_approximant="IMRPhenomXPHM"` through `bilby.gw.source.lal_binary_black_hole`. | This documents an analysis interface to the LAL model, not a separate calibration source. No native Bilby workflow equivalence is claimed from the DiffGW model name. |
| [ripple README at `58c5302`](https://github.com/tedwards2412/ripple/blob/58c53024718697e0430b666ef5dd40ff3c2e029d/README.md) | Its listed JAX waveforms include XAS, D, Pv2, TaylorF2 tides, and D_NRTidalv2; the list does not include XHM or XPHM. | The native JAX helper here covers one XHM algebra branch; neither its JIT/gradient checks nor ripple's XAS support establish a qualified differentiable higher-mode model. The README list is a documented-scope observation, not a claim about every experimental ripple branch. |

These interfaces and options must be compared as specific configurations.
An approximant name shared by two backends is not evidence of numerical
parity. The [XHM paper](https://arxiv.org/html/2001.10914v2) and
[XPHM paper](https://arxiv.org/html/2004.06503v2) supply the paper-era
equations; the LALSuite 7.26 documentation supplies the current public
release and option semantics. The paper-era equations alone cannot certify
the later `122022` amplitude calibration.

## Source and release boundary

The author's arXiv:2001.10914v2 archive supplies paper-era amplitude,
phase, QNM, and mixing files, but no authoritative manifest maps those
assets to the installed default XHM release `122022`. [LALSuite 7.26 API
documentation](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/group___l_a_l_sim_i_m_r_phenom_x__c.html)
identifies `122022` as the default and `122019` as the older version;
the newer version recalibrates higher-mode amplitudes without changing phase
prescriptions. A changed spheroidal `(3,2)` amplitude can still change the
mixed spherical `(3,2)` phase. The paper-era archive has no verified
release-122022 fit-table redistribution grant or formula-to-file mapping.
Its availability cannot justify translating it into the current provider.

The [regional carrier audit](../../math/imrphenomxhm-inspiral-amplitude.md)
locates the earliest measured binary mismatch in higher-mode inspiral
amplitude. Existing source-derived references verify supplied-target
carrier, phase, cuts, ringdown, mixing, projection, and XPHM rotation
components in their stated domains. The new [intermediate-amplitude
gate](../../math/imrphenomxhm-intermediate-amplitude.md) closes the ordinary
degree-five supplied-target algebra for Torch and JAX. These component
claims do not qualify the fitted surfaces or the assembled waveform.

## Current-branch verification and decision

The current [raw strict binary receipt](xhm-xphm-frozen-binary-20260929.json)
was produced from base commit `64e666165b043267caa5a513a4542c40dc6718e4`
plus this family-scoped addition on host `len` with Python 3.12.3,
LALSuite `7.26.15`, LALSimulation `6.2.1`, Torch `2.14.0+cu130`, JAX
`0.11.2`, and NumPy `2.3.5`. It invokes the frozen catalog and holdout
cases through the public `SimInspiralChooseFDWaveform` binary API. Each
polarization must satisfy complex relative L2 `<=1e-6` and flat mismatch
`<=1e-8`. The table gives complex L2 and mismatch for plus/cross; all are
strict failures.

| Case | Plus: L2 / mismatch | Cross: L2 / mismatch |
| --- | ---: | ---: |
| XHM catalog | `0.374023 / 0.063006` | `0.366201 / 0.060753` |
| XPHM catalog | `0.228818 / 0.025734` | `0.226417 / 0.025321` |
| XHM holdout | `0.546374 / 0.127476` | `0.462455 / 0.094623` |
| XPHM holdout | `0.504113 / 0.112109` | `0.556021 / 0.134134` |

The strict verdict is **0/4**. The harness labels these rows
`SOVEREIGN_NR_CALIBRATED` even when `legacy_oracle_passed=false`; that label
is not a passed binary gate. The older
[2026-09-25 receipt](../../validation/xhm_xphm_binary_oracle_20260925.json)
also failed 0/4. Its waveform base changed before this branch, so the older
numeric errors cannot be attributed to the new helper or used as a paired
regression measurement. The helper remains unconnected to the production
waveform. These binary checks measure compatibility with the pinned public
LAL binary. They are not an NR physical-accuracy test.

The receipt audit disproved two shortcuts: the harness status is not the
strict verdict, and the older numeric errors are not a same-base regression
reference. The next decisive check is a verified `122022` amplitude provider
against the same frozen cases, with the first divergent mode and frequency
region isolated before any full-waveform release claim.

The focused supplied-target and native family tests passed **94/94** on
`len` (six XHM/XPHM test modules, including the new Torch/JAX/mpmath
intermediate-amplitude reference tests). This verifies those components in
their tested domains, not the assembled XHM/XPHM waveform or every public
option tuple. No deployment or release qualification was performed.

Further release work needs an author-published, checksum-pinned 122022
calibration package, or independently reproducible calibration data and
procedure, with versioned surfaces, transition cuts, veto order, spheroidal
`(3,2)` inputs, mixing prescription, and reuse terms. The complete carrier
must then pass the unchanged catalog and holdout binary gates, followed by
physical NR holdouts. XPHM full-waveform release depends on that carrier and
separate precession-domain validation. Until then both full models remain
unqualified for release.
