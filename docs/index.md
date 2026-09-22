# DiffGW

<p align="center">
  <a href="https://xangma.github.io/diffgw/"><img src="https://img.shields.io/badge/docs-diffgw-indigo.svg" alt="Documentation"></a>
  <a href="https://github.com/xangma/diffgw"><img src="https://img.shields.io/badge/github-diffgw-blue.svg" alt="GitHub"></a>
  <img src="https://img.shields.io/badge/python-3.10%20--%203.14-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/framework-PyTorch%20%7C%20JAX-orange.svg" alt="Frameworks">
  <img src="https://img.shields.io/badge/precision-FP64%20Support-success.svg" alt="Precision">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</p>

DiffGW provides gravitational-wave models and analysis tools in native
**PyTorch** and **JAX**. Generate waveform tensors, batch binary parameters,
and differentiate supported model paths within your chosen framework.
Use the unified `diffgw` interface or either frontend directly.

<a id="installation"></a>
<a id="documentation-navigation"></a>

## Start with your question

| I want to… | Start here |
| --- | --- |
| Try DiffGW on a binary system | [Installation](installation.md), then [quickstart](tutorials/01_quickstart_and_parity.md) |
| Compare waveform coverage with other packages | [Dated package comparison](comparison.md) |
| Choose and configure a DiffGW waveform | [Waveform catalog](models.md) and [parameter conventions](conventions.md) |
| Understand a difference from LAL | [Differences from LAL](differences_from_lal.md) |
| Assess accuracy or derivative evidence | [Validation](validation.md) |
| Evaluate speed for my workload | [Performance](performance.md) |
| Use batching, gradients or inference tools | [User guide](usage.md) |
| Look up an exact function | [Unified API](api/diffgw.md), [PyTorch API](api/torchwave.md), [JAX API](api/jaxwave.md) |

<a id="quickstart"></a>

## Generate a waveform

After installing either backend, this example generates frequency-domain
IMRPhenomD polarizations through the unified API:

```python
import diffgw

hp, hc = diffgw.get_fd_waveform(
    "IMRPhenomD",
    backend="torch",  # use "jax" for JAX arrays
    mass1=30.0,
    mass2=25.0,
    chi1=0.2,
    chi2=-0.1,
    distance=410.0,
    f_lower=20.0,
    f_final=512.0,
    delta_f=0.25,
)
```

Masses are in solar masses, distance in Mpc and frequencies in Hz. The
[quickstart](tutorials/01_quickstart_and_parity.md) demonstrates both frontends
and a scoped numerical consistency check. For modes, precession, tides and
time-domain sampling, continue to [waveform generation](guides/waveform-generation.md).

<a id="waveform-models-framework-parity"></a>

## Scope and evidence

The [catalog](models.md) covers inspiral, merger-ringdown, tidal, precessing,
higher-mode and specialized models. Availability, agreement with LAL and
independent physical validation are separate properties. Each model's
restrictions and evidence are linked from its catalog entry.

Detector projections, likelihoods and Fisher tools connect waveforms to
[analysis workflows](guides/inference.md). Derivative support and measured
performance depend on the selected model, parameters and execution path.

DiffGW's original software is MIT-licensed. Literature, author calibrations and
optional external assets are documented in the [provenance ledger](provenance_ledger.md),
[external-data guide](external-data.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).
