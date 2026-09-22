# DiffGW and related waveform packages

This page is a dated coverage map for DiffGW, LALSimulation, Ripple, PhenomXPy
and GWFAST. It records named models and execution boundaries documented by each
project; a shared name does not establish identical physics, conventions,
parameter domains or validation. See [differences from LAL](differences_from_lal.md)
for model-specific numerical and convention differences, [validation](validation.md)
for evidence, and [performance](performance.md) for measured benchmark
workloads.

## Waveform coverage

<a id="reading-the-coverage-table"></a>

Checked **22 September 2026**; see [sources and versions](#sources-and-versions).
The table covers DiffGW's catalog and selected alternatives, not every model in
other packages. Entries identify named models or explicitly grouped variants.
LAL denotes LALSimulation.

**Legend**

- **N** (native): the package documents an implementation of the named model.
- **N+A** (native with an external asset): evaluation requires separately acquired model or calibration data.
- **E** (external wrapper): the package calls another library's implementation.
- **-** (not listed): the checked documentation does not name the variant; this does not establish that it is unsupported.
- **?** (unverified): the documentation is insufficient to assign a status.
- **W?** (wrapper unverified): GWFAST documents a generic LAL interface, but support for this model and its options is unconfirmed.

These statuses describe documented availability, not accuracy or calibration.
Notes below each group explain model-specific qualifications. For DiffGW's
frontend scope and validation evidence, see the [model catalogue](models.md).

<div class="waveform-coverage" markdown="1">

### Post-Newtonian

| Model or variant | DiffGW | LAL | Ripple | Phenom<wbr>XPy | GWFAST |
| --- | --- | --- | --- | --- | --- |
| `TaylorF2` <sup>1</sup> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native (tidal-capable)" aria-label="Native (tidal-capable)">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Native (TaylorF2_RestrictedPN)" aria-label="Native (TaylorF2_RestrictedPN)">N</abbr> |
| `TaylorF2_HM` <sup>2</sup> | <abbr title="Native (eight modes)" aria-label="Native (eight modes)">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> |
| `TaylorF2_Eccentric` <sup>3</sup> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native (TaylorF2Ecc)" aria-label="Native (TaylorF2Ecc)">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Native (eccentric option)" aria-label="Native (eccentric option)">N</abbr> |

<div class="coverage-notes" markdown="1">

- **1.** Ripple: Native (tidal-capable). GWFAST: Native (`TaylorF2_RestrictedPN`).
- **2.** DiffGW: Native (eight modes).
- **3.** LALSimulation: Native (`TaylorF2Ecc`). GWFAST: Native (eccentric option).

</div>

### Aligned-spin IMR

| Model or variant | DiffGW | LAL | Ripple | Phenom<wbr>XPy | GWFAST |
| --- | --- | --- | --- | --- | --- |
| `IMRPhenomD` | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Native" aria-label="Native">N</abbr> |
| `IMRPhenomHM` | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Native" aria-label="Native">N</abbr> |
| `IMRPhenomXAS` | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> |
| `IMRPhenomXHM` | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Wrapper unverified" aria-label="Wrapper unverified">W?</abbr> |

### Tidal models

| Model or variant | DiffGW | LAL | Ripple | Phenom<wbr>XPy | GWFAST |
| --- | --- | --- | --- | --- | --- |
| `IMRPhenomD_NRTidal` | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> |
| `IMRPhenomD_NRTidalv2` | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Native" aria-label="Native">N</abbr> |
| `IMRPhenomNSBH` | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Native" aria-label="Native">N</abbr> |
| `IMRPhenomXAS_NRTidalv2` | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Wrapper unverified" aria-label="Wrapper unverified">W?</abbr> |
| `IMRPhenomXAS_NRTidalv3`, `IMRPhenomXP_NRTidalv3` | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Wrapper unverified" aria-label="Wrapper unverified">W?</abbr> |

### Precessing IMR

| Model or variant | DiffGW | LAL | Ripple | Phenom<wbr>XPy | GWFAST |
| --- | --- | --- | --- | --- | --- |
| `IMRPhenomP` | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Wrapper unverified" aria-label="Wrapper unverified">W?</abbr> |
| `IMRPhenomPv2` | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Wrapper unverified" aria-label="Wrapper unverified">W?</abbr> |
| `IMRPhenomXP`, `IMRPhenomXPHM` | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Wrapper unverified" aria-label="Wrapper unverified">W?</abbr> |
| XPHM MSA, SpinTaylor and PNR options <sup>4</sup> | <abbr title="Native (documented options)" aria-label="Native (documented options)">N</abbr> | <abbr title="Unverified (options not enumerated here)" aria-label="Unverified (options not enumerated here)">?</abbr> | <abbr title="Unverified (options not enumerated here)" aria-label="Unverified (options not enumerated here)">?</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Wrapper unverified" aria-label="Wrapper unverified">W?</abbr> |

<div class="coverage-notes" markdown="1">

- **4.** DiffGW: Native (documented options). LALSimulation: Unverified (options not enumerated here). Ripple: Unverified (options not enumerated here).

</div>

### Time-domain IMR

| Model or variant | DiffGW | LAL | Ripple | Phenom<wbr>XPy | GWFAST |
| --- | --- | --- | --- | --- | --- |
| `IMRPhenomT`, `IMRPhenomTHM`, `IMRPhenomTP`, `IMRPhenomTPHM` | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Wrapper unverified" aria-label="Wrapper unverified">W?</abbr> |

### Effective-one-body

| Model or variant | DiffGW | LAL | Ripple | Phenom<wbr>XPy | GWFAST |
| --- | --- | --- | --- | --- | --- |
| `SEOBNR_ROM`, `SEOBNRv4_ROM` <sup>5</sup> | <abbr title="Native + external asset" aria-label="Native + external asset">N+A</abbr> | <abbr title="Native (SEOBNRv4_ROM)" aria-label="Native (SEOBNRv4_ROM)">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Wrapper unverified" aria-label="Wrapper unverified">W?</abbr> |
| `SEOBNRv4HM`, `SEOBNRv4P`, `SEOBNRv4PHM` | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Wrapper unverified" aria-label="Wrapper unverified">W?</abbr> |
| `SEOBNRv4_ROM_NRTidalv2`, `SEOBNRv4_ROM_NRTidalv2_NSBH`, `SEOBNRv4T_surrogate` | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Wrapper unverified" aria-label="Wrapper unverified">W?</abbr> |
| `SEOBNRv5_ROM`, `SEOBNRv5HM_ROM`, `SEOBNRv5_ROM_NRTidalv3` | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Wrapper unverified" aria-label="Wrapper unverified">W?</abbr> |
| `SEOBNR_ODE` | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> |
| `TEOBResumS` | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Wrapper unverified" aria-label="Wrapper unverified">W?</abbr> |
| `TEOBResumSPA` | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="External" aria-label="External">E</abbr> |

<div class="coverage-notes" markdown="1">

- **5.** LALSimulation: Native (`SEOBNRv4_ROM`).

</div>

### NR surrogate

| Model or variant | DiffGW | LAL | Ripple | Phenom<wbr>XPy | GWFAST |
| --- | --- | --- | --- | --- | --- |
| `NRSur4d2s`, `NRSur7dq4` | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Wrapper unverified" aria-label="Wrapper unverified">W?</abbr> |

### Ringdown and bursts

| Model or variant | DiffGW | LAL | Ripple | Phenom<wbr>XPy | GWFAST |
| --- | --- | --- | --- | --- | --- |
| `KerrRingdown` | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Unverified" aria-label="Unverified">?</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> |
| `SineGaussian` <sup>6</sup> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Native (burst routines)" aria-label="Native (burst routines)">N</abbr> | <abbr title="Native (time domain)" aria-label="Native (time domain)">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> |
| `MemoryWaveform` | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> |

<div class="coverage-notes" markdown="1">

- **6.** LALSimulation: Native (burst routines). Ripple: Native (time domain).

</div>

### Continuous waves

| Model or variant | DiffGW | LAL | Ripple | Phenom<wbr>XPy | GWFAST |
| --- | --- | --- | --- | --- | --- |
| `ContinuousWave` <sup>7</sup> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Unverified" aria-label="Unverified">?</abbr> | <abbr title="Not listed under this name" aria-label="Not listed under this name">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> |
| `ExactPulsarSignal`, `PulsarSignal`, `BinaryPulsarSignal` <sup>8</sup> | <abbr title="Not listed under these names" aria-label="Not listed under these names">-</abbr> | <abbr title="Unverified" aria-label="Unverified">?</abbr> | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> |

<div class="coverage-notes" markdown="1">

- **7.** Ripple: Not listed under this name.
- **8.** DiffGW: Not listed under these names.

</div>

### Specialized

| Model or variant | DiffGW | LAL | Ripple | Phenom<wbr>XPy | GWFAST |
| --- | --- | --- | --- | --- | --- |
| `EMRIKludge`, `ParameterizedPostEinstein`, `SpinPrecessionODE`, `HybridMECO` | <abbr title="Native" aria-label="Native">N</abbr> | <abbr title="Unverified" aria-label="Unverified">?</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> | <abbr title="Not listed" aria-label="Not listed">-</abbr> |

</div>

The LAL column is based on the [LALSimInspiral approximant API][lal-inspiral]
and [LALSimulation module index][lal-modules]. The Ripple column uses the
project's [current waveform catalogue][ripple-catalogue]. PhenomXPy's official
repository documents the four T-family entries in its [README][phenomxpy].
GWFAST's [waveform documentation][gwfast-waveforms] identifies its native Python
models and its separate [LAL and TEOBResumS wrappers][gwfast-waveforms].

`Native` does not mean the implementation is independent of published
calibration data. `External` identifies a runtime wrapper; it
also does not imply numerical equivalence. For all columns, omitted options,
modes, domains and derivative orders remain unverified.

`TEOBResumSPA` is a separately named frequency-domain
variant, not a claim of identical behavior to `TEOBResumS`.

<a id="what-diffgw-provides"></a>
<a id="how-diffgw-differs"></a>

## Execution capabilities

Model coverage and execution capabilities are separate questions. The following
summary records the interfaces documented on the same date; it does not infer
support for every model in a package's catalogue.

| Package | Native execution interfaces | Batching and compilation | Differentiation and external boundaries |
| --- | --- | --- | --- |
| **DiffGW** | Native PyTorch (`torchwave`) and JAX (`jaxwave`) frontends under `diffgw`; frequency- and time-domain models are listed in the [catalogue](models.md). | Waveforms expose batch-shaped tensor evaluation. Supported paths document `torch.compile` and `jax.jit`; measured CPU/CUDA scope is in [performance](performance.md). | Automatic differentiation is model- and path-specific. Native evaluation does not call LAL; ROMs and explicitly documented data assets remain external. See [validation](validation.md). |
| **LALSimulation** | LAL's documented high-level routines generate time- or frequency-domain waveforms for named approximants. | The checked API documents scalar waveform-generation entry points and sequence/grid variants; this page makes no blanket claim about tensor batching or autodiff. | No general native PyTorch/JAX autodiff interface is documented in the checked LAL pages. |
| **Ripple** | Differentiable JAX waveform functions; the current catalogue lists frequency-domain CBC models, time-domain bursts and continuous waves. | JAX execution supports compilation and accelerator use; see the [project documentation][ripple-readme]. | Support depends on the model and transformed path. The current catalogue does not list the T family. |
| **PhenomXPy** | Python T, THM, TP and TPHM implementations, with time-domain and Fourier-transformed polarizations. | The README documents Numba CPU acceleration and CuPy GPU execution, parallelized over time samples. [T/THM CPU timings](package_benchmarks.md#phenomxpy-time-domain-comparison-22-september-2026) compare it with DiffGW. | The README does not document a native PyTorch/JAX autodiff interface. |
| **GWFAST** | Vectorised Python waveform models plus detector/Fisher analysis; native model list is in the [waveform documentation](https://gwfast.readthedocs.io/en/latest/waveforms.html). | Native models are documented as vectorised; the package uses JAX for automatic differentiation. | LAL and TEOBResumS are explicit wrappers; the documentation states that derivatives through wrapped models use numerical differentiation. |

For DiffGW's actual model options, derivative order, device limits and
unresolved cases, use the [model catalogue](models.md), [usage guidance](usage.md)
and [validation evidence](validation.md). For LAL-reference differences, use
the canonical [differences-from-LAL page](differences_from_lal.md).

## Related analysis and waveform projects

[Jim](https://github.com/GW-JAX-Team/Jim) provides JAX Bayesian inference using
Ripple. [Bilby](https://bilby-dev.github.io/bilby/api/bilby.gw.source.html) and
[PyCBC](https://pycbc.org/pycbc/latest/html/waveform.html) provide analysis
frameworks that integrate waveform providers. [WF4Py](https://github.com/CosmoStatGW/WF4Py)
supplies standalone Python waveform models associated with GWFAST.
[pySEOBNR](https://waveforms.docs.ligo.org/software/pyseobnr),
[TEOBResumS](https://teobresums.bitbucket.io/) and
[GWSurrogate](https://github.com/sxs-collaboration/gwsurrogate) are related
model and data projects with their own calibration domains and assets.

## Sources and versions

Coverage was checked on **22 September 2026**. The DiffGW column describes this
checkout; timing receipts separately pin their own source revision. The LAL
approximant list is pinned to **LALSuite 7.26 / LALSimulation 6.2.1**; its module
index is rolling documentation. Ripple's stable catalogue, PhenomXPy's main-branch
README and GWFAST's latest documentation are moving targets. The GWFAST page
identifies itself as **1.1.2**. These are documentation observations, not fresh
execution tests of every entry. Recheck sources when updating this table.

<a id="measured-batch-performance"></a>

Published timing results are in [Performance](performance.md).

<a id="physics-coverage-and-scientific-agreement"></a>
<a id="model-specific-differences-from-lal"></a>
<a id="numerical-precession-spintaylor-and-pnr"></a>
<a id="native-imrphenomt-family"></a>
<a id="reproducing-model-checks"></a>

For detailed reference differences and reproduction commands, see
[Differences from LAL](differences_from_lal.md). The [validation overview](validation.md)
explains what each kind of evidence establishes.

[lal-inspiral]: https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/group___l_a_l_sim_inspiral__h.html
[lal-modules]: https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/modules.html
[ripple-readme]: https://github.com/GW-JAX-Team/ripple
[ripple-catalogue]: https://gw-jax-team.github.io/ripple/stable/guides/catalogue/
[phenomxpy]: https://gitlab.com/imrphenom-dev/phenomxpy/-/blob/main/README.md
[gwfast-waveforms]: https://gwfast.readthedocs.io/en/latest/waveforms.html
