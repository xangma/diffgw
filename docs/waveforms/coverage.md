# Waveform specification coverage

This page maps the public Torch and JAX dispatch names to a model or variant.
It also lists standalone waveform classes omitted from those dispatch registries.
`Versioned slice` means that a canonical specification and independent scalar
reference cover only the stated slice in the [waveform library](index.md).
`Provisional` means the waveform is implemented but does not yet have that
complete, source-linked specification/reference contract. "Separate SEOBNR
track" records the SEOBNR models without claiming that the current library work
validates their external assets, ODE dynamics, or reconstruction.

The model names link to the most relevant existing guide or evidence record;
those links are not substitutes for a six-section canonical specification.
The [support catalog](../models.md), [model evidence manifest](../model_manifest.json),
and [validation overview](../validation.md) remain the places to assess specific
options and numerical evidence. A dispatch alias is not a separate mathematical
model. `Direct class only` means use the class API, not the FD/TD dispatcher.

<!-- BEGIN GENERATED COVERAGE TABLE -->

| Model or variant | Form | Spec status | Torch dispatch | JAX dispatch |
| --- | --- | --- | --- | --- |
| [TaylorF2](../math/taylorf2.md) | native FD | [versioned slice](index.md) | `taylorf2` | `taylorf2` |
| [TaylorF2_HM](../models.md) | higher-mode extension | provisional | `taylorf2hm` | `taylorf2hm` |
| [TaylorF2_Eccentric](../models.md) | eccentric extension | provisional | `taylorf2eccentric`, `eccentrictaylorf2` | `taylorf2eccentric`, `eccentrictaylorf2` |
| [IMRPhenomD](../math/imrphenomd-intrinsic.md) | native FD | [versioned slice](index.md) | `imrphenomd` | `imrphenomd` |
| [IMRPhenomHM](../models.md) | higher-mode extension | provisional | `imrphenomhm` | `imrphenomhm` |
| [IMRPhenomXAS](../math/imrphenomxas.md) | native FD | [versioned slice](index.md) | `imrphenomxas`, `phenomxas`, `xas` | `imrphenomxas`, `phenomxas`, `xas` |
| [IMRPhenomXHM](../models.md) | higher-mode extension | provisional | `imrphenomxhm` | `imrphenomxhm` |
| [IMRPhenomP](../models.md) | precessing composition | provisional | `imrphenomp` | `imrphenomp` |
| [IMRPhenomPv2](../models.md) | precessing variant | provisional | `imrphenompv2` | `imrphenompv2` |
| [IMRPhenomXP](../models.md) | precessing composition | provisional | `imrphenomxp`, `phenomxp`, `xp` | `imrphenomxp`, `phenomxp`, `xp` |
| [IMRPhenomXPHM](../models.md) | precessing higher-mode composition | provisional | `imrphenomxphm`, `phenomxphm`, `xphm` | `imrphenomxphm`, `phenomxphm`, `xphm` |
| [IMRPhenomXPHM_SpinTaylor](../model_manifest.json) | XPHM precession variant | provisional | `imrphenomxphmspintaylor`, `phenomxphmspintaylor`, `xphmspintaylor` | `imrphenomxphmspintaylor`, `phenomxphmspintaylor`, `xphmspintaylor` |
| [IMRPhenomXPNR](../model_manifest.json) | XPHM PNR variant | provisional | `imrphenomxpnr`, `phenomxpnr`, `xpnr`, `imrphenomxnpr`, `phenomxnpr`, `xnpr`, `pnr` | `imrphenomxpnr`, `phenomxpnr`, `xpnr`, `pnr` |
| [IMRPhenomT](../imrphenomt.md) | native TD | provisional | `imrphenomt` | `imrphenomt` |
| [IMRPhenomTHM](../imrphenomt.md) | T higher-mode extension | provisional | `imrphenomthm` | `imrphenomthm` |
| [IMRPhenomTP](../imrphenomt.md) | T precessing extension | provisional | `imrphenomtp` | `imrphenomtp` |
| [IMRPhenomTPHM](../imrphenomt.md) | T precessing higher-mode extension | provisional | `imrphenomtphm` | `imrphenomtphm` |
| [IMRPhenomD_NRTidal](../models.md) | tidal composition | provisional | `imrphenomdnrtidal`, `nrtidal` | `imrphenomdnrtidal`, `nrtidal` |
| [IMRPhenomD_NRTidalv2](../models.md) | tidal composition | provisional | `imrphenomdnrtidalv2`, `nrtidalv2` | `imrphenomdnrtidalv2`, `nrtidalv2` |
| [IMRPhenomNSBH](../models.md) | matter composition | provisional | `imrphenomnsbh`, `nsbh` | `imrphenomnsbh`, `nsbh` |
| [KerrRingdown](../models.md) | ringdown generator | provisional | `kerrringdown`, `ringdown` | `kerrringdown`, `ringdown` |
| [OfficialSEOBNRv4ROM](../external-data.md) | external-data wrapper | separate SEOBNR track | `seobnrrom`, `seobnrv4rom`, `officialseobnrv4rom` | `seobnrrom`, `seobnrv4rom`, `officialseobnrv4rom` |
| [SEOBNR_ROM](../models.md) | native surrogate | separate SEOBNR track | `seobnrv5rom` | `seobnrv5rom` |
| [SEOBNR_ODE](../models.md) | numerical ODE generator | separate SEOBNR track | `seobnr`, `seobnrode`, `seobnrv5` | `seobnr`, `seobnrode`, `seobnrv5` |
| [SEOBNRv4Published](../model_manifest.json) | published-model reconstruction | separate SEOBNR track | `seobnrv4published` | `seobnrv4published` |
| [SEOBNRv4PublishedNonspinning](../model_manifest.json) | published-model restricted variant | separate SEOBNR track | `seobnrv4publishednonspinning` | `seobnrv4publishednonspinning` |
| [TEOBResumS](../models.md) | numerical ODE generator | provisional | `teobresums`, `teobresum`, `teob` | `teobresums`, `teobresum`, `teob` |
| [ContinuousWave](../models.md) | direct waveform class | provisional | direct class only | direct class only |
| [MemoryWaveform](../models.md) | direct waveform class | provisional | direct class only | direct class only |
| [EMRIKludge](../models.md) | direct waveform class | provisional | direct class only | direct class only |
| [ParameterizedPostEinstein](../models.md) | direct waveform class | provisional | direct class only | direct class only |

<!-- END GENERATED COVERAGE TABLE -->

The XPHM SpinTaylor and PNR rows are option-selecting dispatch variants of
IMRPhenomXPHM. `IMRPhenomPv2` is a distinct JAX class; Torch selects a Pv2
path through its IMRPhenomP dispatch class. The T family is native time domain.
Other FD models may be converted to time series by `get_td_waveform`; that
conversion does not make them native TD approximants. The official v4 ROM uses
external data, while `SEOBNR_ROM` under the `seobnrv5rom` alias is a separate
native surrogate. Sine-Gaussian glitch wavelets, SpinPrecessionODE, and
HybridMECO are supporting tools rather than standalone dispatched waveforms.
