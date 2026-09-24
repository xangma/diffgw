# Waveform library

The library connects a model's **usable waveform**, its **published definition**,
and the **checks actually performed**. Start here when you want to know which
equations a DiffGW model uses or what a mathematical-reference result covers.
For every implemented model, including those without a versioned mathematical
reference, use the broader [catalog and support guide](../models.md).

<!-- BEGIN GENERATED WAVEFORM TABLE -->

| Model and specification | Definition version | Scope | Backends |
| --- | --- | --- | --- |
| [TaylorF2](../math/taylorf2.md) | `taylorf2-nonspinning-3.5pn/v1` | Nonspinning 3.5PN inspiral with restricted quadrupole amplitude and positive frequencies. | `torch`, `jax` |
| [IMRPhenomD](../math/imrphenomd-intrinsic.md) | `imrphenomd-intrinsic-composition/v1` | Aligned-spin (2,2) IMR composition with declared fit rows and QNM data. | `torch`, `jax` |
| [IMRPhenomXAS](../math/imrphenomxas.md) | `imrphenomxas-composed-polarizations-carrier/v1` | Aligned-spin (2,2) carrier composition with supplied fit and QNM outputs. | `torch`, `jax` |

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
