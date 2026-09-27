# Waveform specification coverage

This page maps the public Torch and JAX dispatch names to a model or variant.
It also lists standalone waveform classes omitted from those dispatch registries.
`Versioned slice` means that a canonical specification and independent scalar
reference cover only the stated slice in the [waveform library](index.md).
`Provisional` means the waveform is implemented but does not yet have that
complete, source-linked specification/reference contract. "Separate SEOBNR
track" identifies the retained external-data ROM wrapper without claiming
that this library work validates its external calibration asset.

The [retired prototype record](research/retired-release-surface.md) preserves
the former SEOBNR, TEOBResumS, and native-ROM routes outside this current-public
surface. The model names below link to the most relevant existing guide or evidence record;
those links are not substitutes for a six-section canonical specification.
The [support catalog](../models.md), [model evidence manifest](../model_manifest.json),
and [validation overview](../validation.md) remain the places to assess specific
options and numerical evidence. A dispatch alias is not a separate mathematical
model. Aliases without a backend label occur in both dispatchers; exceptions
are labeled in the table. The same alias can still select different backend
class routes, as the [Pv2 ledger](research/phenom-precessing.md) explains.
`Direct class only` means use the class API, not the FD/TD dispatcher.

<!-- BEGIN GENERATED COVERAGE TABLE -->

| Model or variant | Form | Spec status | Public dispatch |
| --- | --- | --- | --- |
| [TaylorF2](../math/taylorf2.md) | native FD | [versioned slice](index.md) | `taylorf2` |
| [TaylorF2_HM](../math/taylorf2-hm.md) | higher-mode extension | [versioned slice](index.md) | `taylorf2hm` |
| [TaylorF2_Eccentric](../math/taylorf2-eccentric.md) | eccentric extension | [versioned slice](index.md) | `taylorf2eccentric`, `eccentrictaylorf2` |
| [IMRPhenomD](../math/imrphenomd-intrinsic.md) | native FD | [versioned slice](index.md) | `imrphenomd` |
| [IMRPhenomHM](research/phenom-higher-modes.md) | higher-mode extension | provisional | `imrphenomhm` |
| [IMRPhenomXAS](../math/imrphenomxas.md) | native FD | [versioned slice](index.md) | `imrphenomxas`, `phenomxas`, `xas` |
| [IMRPhenomXHM](research/phenom-higher-modes.md) | higher-mode extension | provisional | `imrphenomxhm` |
| [IMRPhenomP](research/phenom-precessing.md) | precessing composition | provisional | `imrphenomp` |
| [IMRPhenomPv2](research/phenom-precessing.md) | precessing variant | provisional | `imrphenompv2` |
| [IMRPhenomXP](research/phenom-precessing.md) | precessing composition | provisional | `imrphenomxp`, `phenomxp`, `xp` |
| [IMRPhenomXPHM](research/phenom-precessing.md) | precessing higher-mode composition | provisional | `imrphenomxphm`, `phenomxphm`, `xphm` |
| [IMRPhenomXPHM_SpinTaylor](research/phenom-precessing.md) | XPHM precession variant | provisional | `imrphenomxphmspintaylor`, `phenomxphmspintaylor`, `xphmspintaylor` |
| [IMRPhenomXPNR](research/imrphenomxpnr.md) | XPHM PNR variant | provisional component evidence; full waveform unqualified | `imrphenomxpnr`, `phenomxpnr`, `xpnr`, `pnr`; Torch only: `imrphenomxnpr`, `phenomxnpr`, `xnpr` |
| [IMRPhenomT](research/imrphenomt-family.md) | native TD | provisional | `imrphenomt` |
| [IMRPhenomTHM](research/imrphenomt-family.md) | T higher-mode extension | provisional | `imrphenomthm` |
| [IMRPhenomTP](research/imrphenomt-family.md) | T precessing extension | provisional | `imrphenomtp` |
| [IMRPhenomTPHM](research/imrphenomt-family.md) | T precessing higher-mode extension | provisional | `imrphenomtphm` |
| [IMRPhenomD_NRTidal](research/tidal-eccentric-higher-modes.md) | tidal composition | provisional | `imrphenomdnrtidal`, `nrtidal` |
| [IMRPhenomD_NRTidalv2](research/tidal-eccentric-higher-modes.md) | tidal composition | provisional | `imrphenomdnrtidalv2`, `nrtidalv2` |
| [IMRPhenomNSBH](research/tidal-eccentric-higher-modes.md) | matter composition | provisional | `imrphenomnsbh`, `nsbh` |
| [KerrRingdown](../math/standalone-family-research-ledger.md) | ringdown generator | provisional | `kerrringdown`, `ringdown` |
| [OfficialSEOBNRv4ROM](../external-data.md) | external-data wrapper | separate SEOBNR track | `seobnrrom`, `seobnrv4rom`, `officialseobnrv4rom` |
| [ContinuousWave](../math/standalone-family-research-ledger.md) | direct waveform class | provisional | direct class only |
| [MemoryWaveform](../math/standalone-family-research-ledger.md) | direct waveform class | provisional | direct class only |
| [EMRIKludge](../math/standalone-family-research-ledger.md) | direct waveform class | provisional | direct class only |
| [ParameterizedPostEinstein](../math/standalone-family-research-ledger.md) | direct waveform class | provisional | direct class only |

<!-- END GENERATED COVERAGE TABLE -->

The XPHM SpinTaylor and PNR rows are option-selecting dispatch variants of
IMRPhenomXPHM. `IMRPhenomPv2` is a distinct JAX class; Torch selects a Pv2
path through its IMRPhenomP dispatch class. The T family is native time domain.
Other FD models may be converted to time series by `get_td_waveform`; that
conversion does not make them native TD approximants. The official v4 ROM uses
external data; direct `SEOBNR_ROM` class construction selects this same
evaluator. The former `seobnrv5rom` alias selected a different native
surrogate and has been retired; it must not silently select the official ROM.
Sine-Gaussian glitch wavelets and HybridMECO are supporting tools rather than
standalone dispatched waveforms. SpinPrecessionODE is retired.
