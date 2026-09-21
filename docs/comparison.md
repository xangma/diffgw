# DiffGW compared with related packages

DiffGW brings gravitational-wave models into **native PyTorch and JAX** workflows.
Its waveforms are tensors that can be batched over binaries, compiled for repeated
evaluation, and differentiated through supported waveform and analysis paths.
The same package includes detector projections, likelihoods and Fisher tools,
so waveform generation can stay inside the framework used for inference.

This page explains how that design compares with LALSimulation, Ripple,
PhenomXPy and GWFAST, and where DiffGW's model implementations differ.

## What DiffGW provides

- **Two native backends.** `torchwave` and `jaxwave` share an analytical physics
  core and expose corresponding waveform APIs. The unified `diffgw` interface
  selects either backend, so PyTorch and JAX users can work with familiar tensor
  and differentiation tools.
- **Batched, compiled evaluation.** Parameters for multiple binaries become
  waveform arrays of shape `(B, N)`. Supported paths use `torch.compile` or
  `jax.jit`, with broadcasting and JAX `vmap` for vectorization. CPU and CUDA
  performance is measured [below](#measured-batch-performance).
- **Waveforms connected to analysis.** Detector responses, noise PSDs, Whittle
  and relative-binning likelihoods, and forward-mode Fisher calculations are
  available alongside waveform generation. The tutorials demonstrate
  [Fisher analysis](tutorials/02_fisher_analysis.md) and
  [gradient-based inference](tutorials/03_parameter_estimation_nuts.md).
- **Frequency- and time-domain models.** The catalogue spans post-Newtonian,
  phenomenological, tidal, precessing and higher-mode models, with native
  time-domain IMRPhenomT, EOB implementations and Kerr ringdown tools. The
  [model notes](#model-specific-differences-from-lal) identify each family's
  calibration and validation scope.

DiffGW's code is MIT-licensed. The native waveform implementations run without
LAL; optional black-box LAL comparisons provide reference checks. Installation
can select [PyTorch, JAX or both](usage.md), with
[external ROM data](external-data.md) acquired separately where required.

## How DiffGW differs

| Related package | Common ground | DiffGW's approach |
| --- | --- | --- |
| **[LALSimulation](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/index.html)** | Established waveform families and reference conventions. | Native tensor implementations make batching and automatic differentiation part of waveform evaluation. LAL supplies compiled C routines with Python bindings; its waveform API does not document a general native PyTorch/JAX autodiff interface. DiffGW records numerical agreement and deliberate modeling differences per approximant. |
| **[Ripple](https://github.com/GW-JAX-Team/ripple)** | Differentiable JAX waveforms, JIT compilation and vectorization. | DiffGW provides a PyTorch implementation alongside JAX, plus native time-domain IMRPhenomT, EOB and Kerr QNM tools. Ripple's [catalogue](https://gw-jax-team.github.io/ripple/stable/guides/catalogue/) includes frequency-domain CBC models with precessing higher modes and NRTidalv3 variants, as well as burst and pulsar models; the catalogues overlap without being identical. |
| **[PhenomXPy](https://gitlab.com/imrphenom-dev/phenomxpy)** | Native time-domain PhenomT physics and the authors' calibration and reconstruction formulas. | DiffGW integrates the model into both tensor backends, with shared time grids, batching and first-order derivatives. PhenomXPy documents the T/THM/TP/TPHM family using Numba on CPU and CuPy on GPU, with parallel evaluation over time samples; native PyTorch/JAX autodiff is not documented in its [package overview](https://pypi.org/project/phenomxpy/). |
| **[GWFAST](https://gwfast.readthedocs.io/en/latest/)** | Differentiable waveforms, detector responses, SNR and Fisher calculations. | DiffGW connects these calculations to its PyTorch and JAX waveform APIs and Bayesian likelihood tools. GWFAST focuses on detector-network and population forecasting, with JAX/analytical derivatives for native models and numerical waveform derivatives by default for its [external LAL/TEOB wrappers](https://gwfast.readthedocs.io/en/latest/waveforms.html). |

Derivative and compilation support is documented per DiffGW model. For example,
[native IMRPhenomT](imrphenomt.md) supports first-order derivatives on fixed
grids; higher-order derivatives through its frequency inversion are unsupported,
and automatic duration selection is an eager operation because it changes array
shape. The CPU/CUDA measurements below cover FP64 execution; MPS has precision
and compiler limitations, and no TPU measurements are reported here.

## Measured batch performance

DiffGW's batching advantage is visible in the
[FP64 package benchmarks](package_benchmarks.md): at **1,000 binaries**,
compiled PyTorch has the lowest measured CPU batch latency for both
IMRPhenomD and IMRPhenomXAS. The table shows complete-batch median latencies
for that batch size, with **1,969 frequency bins and both polarizations**.
CPU results use one Ryzen 9 7950X3D core; GPU results use the same RTX 5090.

| Workload | DiffGW | Compared implementation | Speedup in this workload |
| --- | ---: | ---: | ---: |
| CPU, IMRPhenomD | Compiled PyTorch: **73.1 ms** | LAL: 199 ms | **2.7×** |
| CPU, IMRPhenomXAS | Compiled PyTorch: **114 ms** | LAL: 160 ms | **1.4×** |
| CPU, IMRPhenomXAS | Compiled PyTorch: **114 ms** | Ripple JAX JIT: 400 ms | **3.5×** |
| GPU, IMRPhenomXAS | JAX JIT: **4.70 ms** | Ripple JAX JIT: 5.34 ms | **1.1×** |

Every comparison in this table passes the campaign's strict LAL reference gate:
direct relative L2 at most `1e-11` and flat-weight, constant-phase-maximized
mismatch at most `1e-14`, for every binary and both polarizations. Across all
measured batches and devices, DiffGW's largest relative L2 error is `9.21e-13`
for PhenomD and `1.67e-13` for PhenomXAS.

These are warmed waveform-generation timings with inputs already on the device;
they exclude transfers, setup and derivative evaluation. LAL has the lowest
single-waveform CPU latency for the three measured models. GPU results vary by
model, batch and backend. The full report retains all timings, uncertainty
intervals, adapter details and accuracy failures, including TaylorF2 results
that miss the strict gate. PhenomXPy and GWFAST are compared by capability here;
they were not timed in this campaign.

## Physics coverage and scientific agreement

DiffGW combines implementations of published waveform models with independently
calibrated components. For example, PhenomD and PhenomXAS closely reproduce the
tested LAL outputs, while the XHM/XPHM higher-mode ringdown uses independent
SXS-based fits. These distinctions are recorded in the
[model manifest](model_manifest.json) and
[provenance ledger](provenance_ledger.md), alongside reference versions, tested
configurations and unresolved cases.

DiffGW separates three kinds of validation:

| Check | Evidence it provides |
| --- | --- |
| **Torch/JAX agreement** | Consistency between DiffGW's two implementations for the tested inputs. |
| **Same-model reference agreement** | Agreement with a specified LAL or author implementation, using declared modes, conventions, grids and tolerances. |
| **Independent physical validation** | Agreement with NR or another independent physical reference over a stated test set. |

The checks serve different purposes: frontend agreement does not establish
physical accuracy, and a shared author calibration makes the PhenomXPy
comparison an implementation check. Reference results retain direct complex
errors and the definition of any phase/time alignment or mismatch weighting.
Gradient checks are separate from waveform-value checks.

## Model-specific differences from LAL

DiffGW implements waveform physics in its own tensor backends. The following
details explain how its coefficients, conventions, calibration and numerical
methods relate to LAL, and summarize the available reference evidence.

| Model or family | DiffGW implementation | Reference evidence |
| --- | --- | --- |
| **IMRPhenomD / IMRPhenomXAS** | Published models with explicit coefficient precision, reference-phase and time-origin conventions. XAS uses the author-supplied symbolic calibration deposit identified in the provenance ledger. | Both DiffGW frontends pass the strict LAL gate for all measured D/XAS configurations in the [FP64 campaign](package_benchmarks.md#accuracy-across-the-measured-configurations). |
| **IMRPhenomXP** | The documented clean path uses MSA version 223, `convention=1`, `final_spin_mod=3` and `phi0=0`. Nonzero `phi0` retains legacy geometry; unsupported cases use explicit fallback. | Frozen catalog and holdout cases agree closely. The manifest records residuals at exact alignment, zero spin and tiny in-plane spins, with detector-projection errors distinguished from polarization errors. |
| **IMRPhenomXHM / IMRPhenomXPHM** | Five modes: `(2,2)`, `(2,1)`, `(3,3)`, `(3,2)` and `(4,4)`. The four higher modes use independent SXS-based ringdown fits and native Kerr QNM calculations; XPHM adds precession to this carrier. | The calibration differs from LAL's. Recorded NR checks examine phase in a restricted ringdown band with reference amplitudes and affine phase alignment; they do not validate full-waveform amplitude and phase accuracy. |
| **SEOBNR_ODE / TEOBResumS** | Native EOB dynamics with continuous spline peak location and ringdown attachment, using their own modeling and numerical choices. | The manifest's LAL diagnostic cases have polarization relative L2 errors about 1.02 and 1.00 respectively, failing the reference checks. These implementations are not currently validated as LAL-equivalent. |
| **TaylorF2_HM** | Literature-based frequency-domain PN extension with eight leading-order modes through `l=4`: `(2,2)`, `(2,1)`, `(3,1)`, `(3,2)`, `(3,3)`, `(4,2)`, `(4,3)`, `(4,4)`. | The external TaylorF2 check covers `(2,2)`. An independent full-waveform reference for the higher modes, higher-PN completeness and spin-amplitude completeness remain unresolved in the manifest. |
| **SEOBNR_ROM** | Data-backed reduced-order evaluation; the chosen surrogate and calibration asset define the model. Official SEOBNRv4ROM data are external. | Validation applies to the selected asset. See [external data](external-data.md) for acquisition and [third-party notices](../THIRD_PARTY_NOTICES.md) for unresolved redistribution terms. |

The EOB [peak locator](../src/torchwave/approximants/eob/continuous_matching.py)
solves a natural cubic spline's derivative on accepted samples, selecting an
interior maximum and rejecting unresolved or tied maxima. The selected root is
locally differentiable, while interval and maximum selection remain piecewise;
continuous attachment does not remove integration-step dependence.
JAX's non-symmetric Kerr angular eigensolver uses a host CPU callback.

### Numerical precession: SpinTaylor and PNR

`IMRPhenomXPHM-SpinTaylor` (`prec_version=320`) integrates spin-precession
equations with fixed-grid RK4 in velocity space. `IMRPhenomXPNR`
(`prec_version=330`, `final_spin_mod=2`) combines NR-tuned PNR angles with
DiffGW's native XHM coprecessing carrier. The carrier, precession integration,
angle attachment and minimal-rotation quadrature contribute to differences from
LAL; comparisons specify the precession and tuned-coprecessing options.

The [SpinTaylor tests](https://github.com/xangma/diffgw/blob/5226141/tests/test_imrphenom_xphm_spintaylor.py)
and [PNR tests](https://github.com/xangma/diffgw/blob/5226141/tests/test_imrphenom_xpnr.py)
use optimized plus-polarization overlaps with model-specific tolerances.
Their scope differs from the direct, both-polarization FP64 gate above; these
variants were not timed in that campaign.

### Native IMRPhenomT

The residuals below are historical time-domain observations, separate from
the new frequency-domain timing campaign. The current regression tests retain
one pinned author-reference fixture and five optional LAL cases, with LAL
acceptance bounds of relative L2 below `1e-3` and mismatch below `1e-6`.

`IMRPhenomT` evaluates the aligned-spin dominant `(2,±2)` time-domain model
from Estellés et al. (2021), using the authors' MIT-licensed phenomxpy
calibration and reconstruction release pinned in the provenance ledger.
The time-domain sampling wrapper follows diffgw's existing returned-time
convention: the amplitude peak is at `tc`, with a shared fixed grid for
batches and explicit `duration`. It does not promise LAL's automatically
selected epoch or array length. `phic` fixes orbital phase at `f_ref`.

At baseline `5226141`, DiffGW provides no `THM`, `TP` or `TPHM` variant
and no direct frequency-domain implementation. Optional black-box LAL checks and native
Torch/JAX parity tests accompany the implementation; LAL is not a runtime
dependency.

Validation against the pinned author implementation reproduces dimensionless
frequency and amplitude within `2e-14` absolute error across five checked
mass/spin configurations, including mass ratio 10 and primary spin 0.9.
Phase agrees up to an arbitrary integration constant at approximately
`1e-11` radians. A pinned high-spin reference test requires no external oracle.

Against the installed LALSimulation binary (version 6.2.1), five cases at
2048 Hz sampling, 20 Hz reference frequency and inclination 0.5 give direct
complex-strain relative L2 errors from `4.6e-6` to `5.7e-4`. The largest
residual occurs for masses `(60, 6)` and spins `(0.9, 0.1)`; its normalized
mismatch is `7.6e-8`. Here mismatch uses an unweighted complex inner product,
maximizes constant phase, and applies no time shift. These results establish
close agreement for the tested cases, not bitwise equivalence to LAL.
The optional regression cases are in the
[IMRPhenomT reference tests](https://github.com/xangma/diffgw/blob/5226141/tests/test_phenomt_reference.py).

### Reproducing model checks

DiffGW's catalog harness retains model-specific references and diagnostic
failures:

```bash
PYTHONPATH=src python tests/parity_harness.py --scope=analytic
PYTHONPATH=src python tests/parity_harness.py --scope=catalog
```

Each result records its reference identity, metrics and verification scope.
The [benchmark protocol](benchmarking_protocol.md#12-the-gatekeeper-rule-parity-first)
sets out how those checks qualify performance results, and the
[package benchmark report](package_benchmarks.md#reproduction-and-receipts)
provides timing receipts and reproduction commands.

## Related analysis and waveform projects

DiffGW's waveform and analysis components also overlap with projects that work
at different levels. [Jim](https://github.com/GW-JAX-Team/Jim) provides JAX
Bayesian inference using Ripple;
[Bilby](https://bilby-dev.github.io/bilby/api/bilby.gw.source.html) and
[PyCBC](https://pycbc.org/pycbc/latest/html/waveform.html) provide analysis
frameworks that integrate waveform providers.
[WF4Py](https://github.com/CosmoStatGW/WF4Py) supplies standalone Python waveform
models associated with GWFAST.
[pySEOBNR](https://waveforms.docs.ligo.org/software/pyseobnr) and
[TEOBResumS](https://teobresums.bitbucket.io/) are reference points for DiffGW's
EOB work, while [GWSurrogate](https://github.com/sxs-collaboration/gwsurrogate)
provides NR surrogate models with their own calibration domains and data assets.

## Sources and versions

The capability comparison was checked on **21 September 2026** against upstream
documentation and DiffGW baseline **`5226141`**, which also underlies the measured
benchmarks. The exact benchmark environment is recorded in the receipts.

| Distribution | Published version checked | Release date | Code licence reported upstream |
| --- | --- | --- | --- |
| [LALSuite](https://pypi.org/project/lalsuite/) (includes LALSimulation) | 7.26.15 | 13 June 2026 | GPL-2.0-or-later |
| [rippleGW](https://pypi.org/project/rippleGW/) | 0.4.0 | 2 September 2026 | MIT |
| [PhenomXPy](https://pypi.org/project/phenomxpy/) | 2.0.3 | 22 June 2026 | MIT |
| [GWFAST](https://pypi.org/project/gwfast/) | 1.1.2 | 4 November 2024 | [GNU GPLv3](https://github.com/CosmoStatGW/gwfast/blob/master/LICENSE) |

Upstream development documentation and `stable`/`latest` URLs can move beyond
these releases. DiffGW's [third-party notices](../THIRD_PARTY_NOTICES.md) record
calibration provenance and external-asset terms separately from its MIT code
licence.
