# Package comparison

Choose a gravitational-wave package around the waveform model, numerical accuracy
and analysis workflow you need. DiffGW provides native PyTorch and JAX frontends;
the packages below offer different combinations of waveform generation,
acceleration and inference tools.

This overview was checked on **21 September 2026** against the linked project
documentation and DiffGW baseline commit **`5226141`**. Upstream development documentation
can describe features ahead of a release. Pin package versions or commit hashes
before reproducing a comparison. Capability tables summarize the documentation;
the separate [measured FP64 comparison](package_benchmarks.md) covers a specific
subset of models, devices and parameter ranges.

## Which package fits your workflow?

| Package | Consider it when you need… | Scope to check |
| --- | --- | --- |
| **DiffGW** | Waveforms and derivatives in an existing PyTorch or JAX workflow, with batching, detector projections, likelihoods and Fisher calculations. | Select the frontend, model and tested parameter range. Consult the [model-specific validation](#model-specific-differences-from-lal) and [manifest](model_manifest.json). |
| **[LALSimulation](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/index.html)**, part of LALSuite | Reference waveform implementations and a broad catalogue of established approximants. | Availability of modes, domains and physical effects depends on the approximant. Use the precise model version and options when checking agreement. |
| **[Ripple](https://github.com/GW-JAX-Team/ripple)** | Differentiable JAX waveforms for accelerated waveform generation or use in a JAX inference pipeline. | Use the active GW-JAX-Team project's [waveform catalogue](https://gw-jax-team.github.io/ripple/stable/guides/catalogue/). Its CBC models are frequency-domain, including precessing higher modes and NRTidalv3 variants; burst and pulsar models extend its scope. |
| **[PhenomXPy](https://gitlab.com/imrphenom-dev/phenomxpy)** | CPU/GPU evaluation of the native time-domain IMRPhenomT family, including higher modes and precession. | The documented family is `IMRPhenomT`, `THM`, `TP` and `TPHM`. Compare native time-domain calculations and account for Fourier conversion when requesting frequency-domain output. |
| **[GWFAST](https://gwfast.readthedocs.io/en/latest/)** | Detector-network SNR and Fisher forecasts, including large populations and future ground-based detectors. | Its [native frequency-domain catalogue](https://gwfast.readthedocs.io/en/latest/waveforms.html) includes TaylorF2, PhenomD/HM and matter variants. External LAL and TEOBResumSPA wrappers have different derivative and execution paths. |

## Execution and derivatives

Automatic differentiation, parallel evaluation over samples and batching over
different binaries are separate capabilities. “Not documented” means the cited
interface does not establish the capability; it is not a claim of impossibility.

| Implementation | Derivatives | Batching and compilation | Hardware scope |
| --- | --- | --- | --- |
| **DiffGW / torchwave** | PyTorch autograd; forward-mode Fisher examples. Parameter and derivative-order support is model-specific. | Tensor broadcasting over binaries; eager execution and `torch.compile` for supported paths. | CPU and CUDA examples and reported timings; MPS support has precision and compiler limitations. |
| **DiffGW / jaxwave** | JAX forward/reverse autodiff on supported paths. | `jax.vmap`, broadcasting and `jax.jit`; fixed array shapes or shape buckets for repeated calls. | CPU and CUDA examples and reported timings. TPU is a framework target; this page supplies no TPU measurements. |
| **LALSimulation** | A general native JAX/PyTorch autodiff interface is not documented for the waveform API. Numerical derivatives can be computed by callers. | Compiled C routines with Python bindings; parameter-batch execution depends on the caller/model. | CPU reference baseline in DiffGW's report; no package-wide GPU claim is made here. |
| **Ripple** | Native JAX autodiff. | [`jax.jit`, `jax.grad` and `jax.vmap`](https://gw-jax-team.github.io/ripple/stable/quickstart/) are documented. | CPU and NVIDIA GPU; the [installation guide](https://pypi.org/project/rippleGW/) documents the CUDA extra. |
| **PhenomXPy** | Native JAX/PyTorch autodiff is not documented in its [package overview](https://pypi.org/project/phenomxpy/). | Numba on CPU, CuPy on GPU; the overview describes parallelism over time samples. Binary-batch support must be checked separately. | CPU and CUDA GPU. |
| **GWFAST** | JAX/analytical derivatives for native models; numerical waveform derivatives by default for external wrappers. | Vectorized population calculations and [optional derivative JIT](https://gwfast.readthedocs.io/en/latest/single_detector.html). | CPU by default; its [installation instructions](https://pypi.org/project/gwfast/) also document JAX GPU/TPU backends. Check the selected waveform provider separately. |

See DiffGW's [usage guide](usage.md), [Fisher tutorial](tutorials/02_fisher_analysis.md)
and [performance report](performance.md), and GWFAST's
[waveform documentation](https://gwfast.readthedocs.io/en/latest/waveforms.html).
Validate gradients through the complete calculation you will use, including
projection, interpolation, cutoffs and likelihood evaluation. A differentiable
waveform does not establish that every surrounding operation has valid gradients.

For example, DiffGW's [native IMRPhenomT](imrphenomt.md) supports
first-order derivatives on fixed grids, but not higher-order derivatives through
its frequency inversion. Automatic duration selection changes array shape and is
an eager operation. At the documented baseline `5226141`, its `THM`, `TP` and
`TPHM` variants are not implemented; subsequent development is outside this
comparison's scope.

## Physics coverage and scientific agreement

Compare exact approximants rather than a count of model names. Record the
following for each model/version pair:

- **Physical scope:** aligned or precessing spins, included multipoles, tides,
  eccentricity, inspiral/merger/ringdown coverage and calibration range.
- **Output contract:** native time/frequency domain, polarizations or individual
  modes, uniform/arbitrary grids, units, frames, reference phase and time origin.
- **Validation scope:** tested masses and spins, boundary cases, numerical
  precision, reference version and treatment of phase/time alignment.

Three different checks answer different questions:

| Check | What it establishes | What it does not establish |
| --- | --- | --- |
| **Torch/JAX agreement** | The two DiffGW implementations agree for the tested inputs. | Independent agreement with LAL or physical accuracy against NR. |
| **Same-model reference agreement** | An implementation reproduces a specified reference configuration within stated tolerances. | Accuracy throughout the model's parameter domain or equivalence with a differently calibrated model. |
| **Independent physical validation** | Agreement with NR or another suitable independent physical reference over a declared test set. | Identical calibration, conventions or numerical outputs across packages. |

Report direct complex relative error alongside a clearly defined overlap or
mismatch. State the PSD, frequency interval, modes and whether time/phase were
optimized. Test derivatives separately, including near piecewise joins and
parameter boundaries. An acceptable mismatch for a search is not automatically
an accuracy guarantee for parameter estimation at every signal-to-noise ratio.

DiffGW's [manifest](model_manifest.json) distinguishes declared-case validation,
independent calibrations and unresolved references. Its XHM/XPHM calibrations
differ from LAL's, and its XP entry records convention and boundary limitations.
The [model notes below](#model-specific-differences-from-lal) and
[provenance ledger](provenance_ledger.md) provide the model-specific context.
Shared IMRPhenomT author formulas also mean a PhenomXPy comparison checks
implementation agreement; it is not independent physical validation.

## Model-specific differences from LAL

These notes consolidate the former “Differences from LAL” page. They describe
the comparison baseline, not guarantees across the full parameter domain.
The [manifest](model_manifest.json) records per-model reference configurations,
verification scope and unresolved cases; the
[provenance ledger](provenance_ledger.md) records equations and calibration sources.

| Model or family | Difference to account for | What the available checks establish |
| --- | --- | --- |
| **IMRPhenomD / IMRPhenomXAS** | Coefficient precision, reference phase and time origin must match. XAS uses the author-supplied symbolic calibration deposit identified in the provenance ledger. | Both DiffGW frontends pass the strict LAL gate for all measured D/XAS configurations in the [FP64 campaign](package_benchmarks.md#accuracy-across-the-measured-configurations). This is evidence for those inputs and versions. |
| **IMRPhenomXP** | The documented clean path uses MSA version 223, `convention=1`, `final_spin_mod=3` and `phi0=0`. Nonzero `phi0` retains legacy geometry; unsupported cases use explicit fallback. | Frozen catalog and holdout cases agree closely, but the manifest records residuals at exact alignment, zero spin and tiny in-plane spins. Detector-projection errors also differ from polarization errors. |
| **IMRPhenomXHM / IMRPhenomXPHM** | The family supports `(2,2)`, `(2,1)`, `(3,3)`, `(3,2)` and `(4,4)`. The four higher modes use independent SXS-based ringdown fits and native Kerr QNM calculations; XPHM adds precession to the carrier. | Independent calibration does not establish equivalence to LAL's calibration. The recorded NR ringdown/phase checks do not certify full-waveform amplitude and phase accuracy throughout the domain. |
| **SEOBNR_ODE / TEOBResumS** | Native EOB dynamics use continuous spline peak location and ringdown attachment. These are separate implementations with their own modeling and numerical choices. | The manifest's LAL diagnostic cases fail, with polarization relative L2 errors about 1.02 and 1.00 respectively. Smooth attachment alone establishes neither LAL equivalence nor independence from integration step size. |
| **TaylorF2_HM** | Literature-based frequency-domain PN extension with eight leading-order modes through `l=4`: `(2,2)`, `(2,1)`, `(3,1)`, `(3,2)`, `(3,3)`, `(4,2)`, `(4,3)`, `(4,4)`. | The external TaylorF2 check covers only `(2,2)`. The manifest leaves an independent full-waveform reference for the higher modes, higher-PN completeness and spin-amplitude completeness unresolved. |
| **SEOBNR_ROM** | The selected surrogate and its calibration asset are part of the model identity. The official SEOBNRv4ROM data are external. | Validate the actual data-backed path; a result for one asset does not qualify another. See [external data](external-data.md) for acquisition and [third-party notices](../THIRD_PARTY_NOTICES.md) for unresolved redistribution terms. |

For the EOB peak locator, the current
[implementation](../src/torchwave/approximants/eob/continuous_matching.py)
solves a natural cubic spline's derivative on accepted samples. It selects an
interior maximum and rejects unresolved or tied maxima. The selected root is
differentiable locally; interval and maximum selection remain piecewise.
Likewise, JAX's non-symmetric Kerr angular eigensolver uses a host CPU callback,
so a native tensor frontend does not imply that every solver runs on the GPU.

### Numerical precession: SpinTaylor and PNR

`IMRPhenomXPHM-SpinTaylor` (`prec_version=320`) integrates the spin-precession
equations with fixed-grid RK4 in velocity space. `IMRPhenomXPNR`
(`prec_version=330`, `final_spin_mod=2`) combines NR-tuned PNR angles with the
native XHM coprecessing carrier. The carrier, precession integration, angle
attachment and minimal-rotation quadrature all affect agreement; explicitly
match LAL's precession and tuned-coprecessing options before interpreting a
residual as numerical error.

The existing regression checks use optimized plus-polarization overlaps with
model-specific tolerances. They do not establish the direct, both-polarization
FP64 gate used in the new benchmark campaign, validate gradients, or certify
accuracy for every parameter-estimation analysis. These variants were not timed
in that campaign. See the
[SpinTaylor tests](https://github.com/xangma/diffgw/blob/5226141/tests/test_imrphenom_xphm_spintaylor.py)
and [PNR tests](https://github.com/xangma/diffgw/blob/5226141/tests/test_imrphenom_xpnr.py)
for the actual configurations and overlap definitions.

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

This addition provides no `THM`, `TP` or `TPHM` variant and no direct
frequency-domain implementation. Optional black-box LAL checks and native
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

The repository's catalog harness retains model-specific references and
diagnostic failures:

```bash
PYTHONPATH=src python tests/parity_harness.py --scope=analytic
PYTHONPATH=src python tests/parity_harness.py --scope=catalog
```

Inspect each case's reference identity, metrics and verification scope; a
dominant-mode check or independent calibration is not full LAL parity. The
[benchmark protocol](benchmarking_protocol.md#12-the-gatekeeper-rule-parity-first)
explains how these distinctions affect performance qualification.

## Performance: compare the workload you will run

The [measured package benchmarks](package_benchmarks.md) compare DiffGW's PyTorch
and JAX frontends with **LALSimulation and Ripple**, on one CPU core and an
RTX 5090, for FP64 `TaylorF2`, `IMRPhenomD` and `IMRPhenomXAS`. Each timed batch
has explicit LAL errors; timings that fail the strict gate remain labelled
**unqualified**. PhenomXPy and GWFAST are capability comparisons here and have
not been timed in this campaign.

For this workload, LAL has the lowest single-waveform CPU latency for all three
models. At batch 1,000, compiled DiffGW Torch has the lowest CPU latency for
PhenomD and PhenomXAS, with both passing the strict gate. GPU results depend on
the model and execution path; consult the full tables and uncertainty intervals.

The older [performance report](performance.md) contains a different campaign.
Its **14.4 million waveforms/s** result is compiled **FP32 TaylorF2**, batch
**5,000**, on an **RTX 5090**; it does not describe FP64, single-waveform latency
or every model.

A broader comparison should cover these workloads:

| Workload | Candidate comparison | Required qualification |
| --- | --- | --- |
| Frequency-domain aligned binaries | `TaylorF2`, `IMRPhenomD`, `IMRPhenomXAS` across DiffGW, LAL and Ripple; GWFAST where the pinned model is available. | Same approximant, physical settings, frequency grid and polarizations. |
| Tidal or precessing binaries | A shared tidal model and a shared precession model from the pinned catalogues. | Match tidal/precession versions and mode selections. Label different calibrations explicitly. |
| Native time-domain binaries | Dominant-mode `IMRPhenomT` across DiffGW, PhenomXPy and LAL. | Match time samples, duration, phase and epoch conventions; account for initialization and resampling. |
| Fisher forecasts | DiffGW and GWFAST using a shared waveform. | Same detectors, PSDs, parameters, derivative method and integration settings. |

For each supported configuration:

1. Pin software versions, model options, hardware and thread counts. Run the
   declared scientific checks before interpreting timings; retain failures.
2. Start with FP64, then evaluate FP32 separately with its own accuracy results.
   Use identical grids and parameter batches, including long BNS signals where
   relevant. A waveform length is part of the workload.
3. Measure batch sizes **1, 10, 100 and 1,000**. Report compilation/initialization,
   warmed execution, waveform-plus-gradient cost and peak memory separately.
4. Synchronize device work, specify whether transfers are included, and use
   repeated measurements with uncertainty. Keep CPU-to-CPU and GPU-to-GPU
   comparisons explicit; identify GPU-to-single-core-CPU ratios as such.
5. Distinguish total batch latency from amortized time per waveform. For
   throughput $R = B/t$, batch latency is $B/R$ and amortized latency is $1/R$.

The [benchmark protocol](benchmarking_protocol.md) defines timing and strict
reference requirements, including limitations of the legacy campaign harness.
Agreement at FP64 or between two frontends does not
qualify a different FP32 or external-reference result.

The available `scripts/compare_packages.py` runner measures FP64 forward waveform
generation for the three aligned frequency-domain models, with explicit LAL
checks and separately labelled unqualified timings. See the
[results, receipts and reproduction commands](package_benchmarks.md). Native
time-domain, derivative, Fisher and memory workloads are proposed comparisons;
this runner does not measure them.

## Installation, integration and related packages

DiffGW offers separate [PyTorch and JAX installation options](usage.md) and
[optional external ROM data](external-data.md). Record data downloads, compiler
requirements and accelerator dependencies as part of reproducibility. Ripple
distributes as `rippleGW` and documents an API that may change before v1.0;
PhenomXPy requires a CuPy installation matched to CUDA for its GPU path.

The release metadata checked for this overview is listed below. These are
documentation snapshots; the measured subset's exact environment is recorded in
the benchmark receipts. Ripple's `stable`
and GWFAST's `latest` documentation URLs can change after the listed release.

| Distribution | Published version checked | Release date | Code licence reported upstream |
| --- | --- | --- | --- |
| [LALSuite](https://pypi.org/project/lalsuite/) (includes LALSimulation) | 7.26.15 | 13 June 2026 | GPL-2.0-or-later |
| [rippleGW](https://pypi.org/project/rippleGW/) | 0.4.0 | 2 September 2026 | MIT |
| [PhenomXPy](https://pypi.org/project/phenomxpy/) | 2.0.3 | 22 June 2026 | MIT |
| [GWFAST](https://pypi.org/project/gwfast/) | 1.1.2 | 4 November 2024 | [GNU GPLv3](https://github.com/CosmoStatGW/gwfast/blob/master/LICENSE) |

Check code and data terms separately: DiffGW's code is MIT-licensed, while its
[third-party notices](../THIRD_PARTY_NOTICES.md) describe external assets and
provenance limitations. Use the upstream licence for the exact version being
compared; the code licence alone does not settle the terms of model datasets.

Several packages complement waveform generators or compare at a different level:

| Package | Relevant comparison |
| --- | --- |
| **[Jim](https://github.com/GW-JAX-Team/Jim)** | JAX Bayesian inference using Ripple waveforms. Compare complete likelihood/sampling workflows and effective samples per unit time with matched convergence checks. |
| **[Bilby](https://bilby-dev.github.io/bilby/api/bilby.gw.source.html)** and **[PyCBC](https://pycbc.org/pycbc/latest/html/waveform.html)** | Analysis workflows and waveform integration. Distinguish framework overhead from the selected waveform provider; wrappers around LAL are not independent waveform implementations. |
| **[WF4Py](https://github.com/CosmoStatGW/WF4Py)** | Standalone Python waveform models associated with GWFAST; useful when detector-network forecasting is outside the comparison scope. |
| **[pySEOBNR](https://waveforms.docs.ligo.org/software/pyseobnr)** and **[TEOBResumS](https://teobresums.bitbucket.io/)** | EOB dynamics, calibration and physical coverage. A DiffGW model with a related name does not by itself establish implementation equivalence. |
| **[GWSurrogate](https://github.com/sxs-collaboration/gwsurrogate)** | NR surrogate coverage, accuracy, calibration domains and external data requirements. |

For an end-to-end inference comparison, hold the data, detector response, priors
and waveform choices fixed. Report posterior agreement and convergence together
with runtime; waveform throughput alone does not determine inference cost.
