# Validation evidence and scope

DiffGW provides native tensor waveform models and inference tools. Each result
below supports a particular implementation, configuration, and numerical check.
A model name or available API does not establish accuracy throughout its
parameter space.

## What each check establishes

| Check | Evidence | Scope |
| --- | --- | --- |
| Waveform performance and LAL agreement | [FP64 package benchmarks](package_benchmarks.md), with committed CPU/CUDA receipts | Forward evaluation of TaylorF2, IMRPhenomD and IMRPhenomXAS on declared grids and binary samples. Both polarizations are checked; failed configurations remain visible. |
| PyTorch/JAX consistency | [Parity tests](https://github.com/xangma/diffgw/tree/main/tests/parity) | Agreement between two implementations for specified inputs and per-test tolerances. Shared coefficients or shared mistakes can produce agreement. |
| Broader model reference checks | [Model comparison](comparison.md#model-specific-differences-from-lal), [manifest](model_manifest.json), and linked archived runs | Frozen cases and model-specific metrics. The archived catalog has 17/21 passing cases and the holdout set 11/13; this is partial validation, not a passing whole-catalog result. New model additions have separate evidence. |
| Derivatives and Fisher matrices | [Inference parity tests](https://github.com/xangma/diffgw/blob/main/tests/parity/test_inference_parity.py), [Fisher example](../examples/02_fisher_analysis.py) | Selected parameters, models and derivative orders. Waveform-value agreement alone does not validate derivatives or posterior uncertainties. |
| Relative-binning likelihood | [Numerical tests](https://github.com/xangma/diffgw/blob/main/tests/test_relative_binning.py) and inference parity tests | Local approximation and gradient checks described below. There is no published DiffGW timing campaign establishing a general likelihood speedup. |
| Physical calibration and provenance | [Provenance ledger](provenance_ledger.md) and [third-party notices](../THIRD_PARTY_NOTICES.md) | Literature, calibration sources, component checks, and unresolved asset provenance. A reference implementation check is distinct from independent physical validation. |

## Relative binning

DiffGW's dense Whittle likelihood evaluates a noise-weighted residual on the
frequency grid. Relative binning approximates it using summary coefficients
computed once from the data, PSD, and a fiducial waveform, followed by candidate
waveform evaluations at bin frequencies. The method relies on a sufficiently
smooth ratio between candidate and fiducial waveforms within each bin.
See [Zackay, Dai and Venumadhav (2018)](https://arxiv.org/abs/1806.08792).

The maintained CPU FP64 tests use IMRPhenomD and an analytical aLIGO PSD:

- **Likelihood values:** 32,768 frequencies from 20–400 Hz, a fiducial binary
  with masses `(36, 29)` solar masses at 410 Mpc, and identical noiseless plus
  polarizations in two detector-labelled streams. At 40 bins, one nearby mass
  perturbation must have absolute log-likelihood error below `0.05`. A batch of
  16 nearby mass pairs checks that refinement from 40 to 80 bins reduces the
  maximum error and that the 80-bin result meets the `0.05` bound.
- **Gradients:** 4,096 frequencies from 25–400 Hz and 50 bins. A single
  distance/phase perturbation checks relative gradient-vector error below
  `1e-2` for those two parameters. This does not cover mass or spin gradients
  throughout a posterior.

These are test acceptance criteria. They do not establish machine-precision
likelihood agreement, a universal bin count, or posterior-wide accuracy. The
current value test compares a trapezoidal dense integral with rectangular
summary sums, so its discrepancy includes that quadrature difference.
Automatic differentiation of the binned likelihood gives derivatives of the
approximation.

A meaningful speed comparison would time **candidate waveform generation plus
likelihood evaluation** against the dense Whittle pipeline with matching model,
parameters, PSD, precision, hardware and batch size. Report setup cost separately,
repeat and synchronize timings, and measure likelihood and gradient errors over
the same candidate set. The published waveform benchmark does not measure this
workload, and speedups in method papers are not DiffGW measurements.

## Gradients and statistical interpretation

Automatic differentiation applies the chain rule to the implemented numerical
operations. Its accuracy and availability depend on precision, interpolation,
piecewise branches, solver tolerances, and derivative order. For example,
[IMRPhenomT](imrphenomt.md) documents fixed-grid first-order derivatives and
limitations of its frequency inversion. Check the relevant parameter path
against independent derivatives or converged finite differences.

The Fisher tools form noise-weighted products of waveform derivatives. An
inverse Fisher matrix is a local uncertainty forecast under additional
assumptions; it is not generally the posterior covariance. Eigenvalue
regularization also changes that inverse. See the [Fisher tutorial](tutorials/02_fisher_analysis.md)
and [Vallisneri (2008)](https://arxiv.org/abs/gr-qc/0703086). MAP convergence and
MCMC diagnostics require separate checks, as does the accuracy of an approximate
likelihood used during sampling.

## Reproducing the inference checks

With the development dependencies and both frontends installed:

```bash
python -m pytest tests/test_relative_binning.py tests/parity/test_inference_parity.py -q
python examples/01_quickstart_and_parity.py
python examples/02_fisher_analysis.py
```

Record the source revision, dependencies, platform, passed/failed/skipped cases,
and output when reporting a run. Optional external-reference tests may skip
when a reference package or data asset is unavailable; a skip supplies no
agreement evidence. Performance comparisons additionally follow the
[benchmark protocol](benchmarking_protocol.md).
