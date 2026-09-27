# Waveform library

The library connects a model's **usable waveform**, its **published definition**,
and the **checks actually performed**. Start here when you want to know which
equations a DiffGW model uses or what a mathematical-reference result covers.
For every implemented model, including those without a versioned mathematical
reference, use the [full dispatch and specification coverage](coverage.md) and
the broader [catalog and support guide](../models.md).
Provisional entries in the coverage table link to source-backed research
ledgers. They identify the equations, native components, and missing evidence
needed before an executable reference can be trusted.
Former complete-waveform prototypes have a separate
[historical record](research/retired-release-surface.md); they do not appear
in the current public dispatch inventory.

<!-- BEGIN GENERATED WAVEFORM TABLE -->

| Model and specification | Definition version | Scope | Backends |
| --- | --- | --- | --- |
| [TaylorF2](../math/taylorf2.md) | `taylorf2-nonspinning-3.5pn/v1` | Nonspinning 3.5PN inspiral with restricted quadrupole amplitude and positive frequencies. | `torch`, `jax` |
| [TaylorF2_Eccentric](../math/taylorf2-eccentric.md) | `taylorf2-eccentric-nonspinning-e2-3pn/v1` | Nonspinning comparable-mass (q <= 2), 0 < e0 <= 0.1, positive-frequency inspiral: restricted quadrupole amplitude and 3PN order-e0² eccentric phase at explicit f0_ecc and f_ref, with amplitude correction off. | `torch`, `jax` |
| [TaylorF2_HM](../math/taylorf2-hm.md) | `taylorf2-hm-leading-carrier/v1` | Eight leading nonspinning SPA mode amplitudes with inherited aligned-spin TaylorF2 phase and negative-m polarization projection. | `torch`, `jax` |
| [IMRPhenomD](../math/imrphenomd-intrinsic.md) | `imrphenomd-intrinsic-composition/v1` | Aligned-spin (2,2) IMR composition with declared fit rows and QNM data. | `torch`, `jax` |
| [IMRPhenomXAS](../math/imrphenomxas.md) | `imrphenomxas-composed-polarizations-carrier/v1` | Aligned-spin (2,2) carrier composition with supplied fit and QNM outputs. | `torch`, `jax` |
| [KerrRingdown220Direct](../math/kerr-ringdown-direct-class.md) | `kerr-ringdown-220-energy-direct-class/v1` | Direct Torch/JAX KerrRingdown class only: one physical-energy spheroidal (2,2,0) mode with independent QNM; selected FP64 inputs. | `torch`, `jax` |
| [ContinuousWaveH1Direct](../math/continuous-wave-direct-class.md) | `cw-monochromatic-h1-direct-class/v1` | Direct Torch/JAX ContinuousWave class: selected float64 CPU monochromatic principal-axis 2f strain at H1, GPS epoch in 2015–2016. | `torch`, `jax` |
| [MemoryWaveform22Direct](../math/memory-dominant-22-direct-class.md) | `memory-dominant-22-direct-class/v1` | Only Torch/JAX MemoryWaveform.evaluate_22 with supplied physical-strain h22, symmetric mirror, and relative memory offset on selected CPU float64 grids. | `torch`, `jax` |

<!-- END GENERATED WAVEFORM TABLE -->

Each model link opens its canonical specification with equations, component IDs,
source versions, variants, and evidence. The TaylorF2 entry covers the
nonspinning base; its separately versioned spin, tidal, and quadrupole
extensions are linked from that specification. The
[component specification index](../math/index.md) lists individual
components and extensions. The [generation guide](../guides/waveform-generation.md)
explains how to evaluate a waveform; the [validation overview](../validation.md)
records numerical comparisons and their limits.

These are different claims: a cited equation needs source review; agreement
with an independent mathematical reference checks an implementation on declared
cases; agreement with LALSimulation and accuracy against physical data require
their own evidence. An entry in this library is not a blanket accuracy rating.

Python users can inspect the same model and component relationships through
[`diffgw.waveforms`](../api/diffgw.md#waveform-library-metadata), without
installing the optional SymPy reference dependencies.
