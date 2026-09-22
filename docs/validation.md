# Accuracy and validation

DiffGW provides native tensor waveform models and inference tools. Each result
below supports a particular implementation, configuration, and numerical check.
A model name or available API does not establish accuracy throughout its
parameter space.

If you are replacing a LAL waveform, start with [Differences from LAL](differences_from_lal.md).
It identifies intentional modeling choices, unresolved discrepancies and their
practical implications. The reports below provide the underlying evidence.

## What each check establishes

| Check | Evidence | Scope |
| --- | --- | --- |
| Waveform performance and LAL agreement | [FP64 package benchmarks](package_benchmarks.md), with committed CPU/CUDA receipts | Forward evaluation of TaylorF2, IMRPhenomD and IMRPhenomXAS on declared grids and binary samples. Both polarizations are checked; failed configurations remain visible. |
| PyTorch/JAX consistency | [Parity tests](https://github.com/xangma/diffgw/tree/main/tests/parity) | Agreement between two implementations for specified inputs and per-test tolerances. Shared coefficients or shared mistakes can produce agreement. |
| Time-domain performance and PhenomXPy agreement | [T/THM CPU comparison](package_benchmarks.md#phenomxpy-time-domain-comparison-22-september-2026), with a pinned author reference and receipt | Both polarizations on a fixed time segment; every binary checked, with strict-gate failures retained as diagnostic timings. Shared author fits mean this is not independent physical validation. |
| Broader model reference checks | [Differences from LAL](differences_from_lal.md#model-specific-differences), [manifest](model_manifest.json), and linked archived runs | Frozen cases and model-specific metrics. The archived catalog has 17/21 passing cases and the holdout set 11/13; this is partial validation, not a passing whole-catalog result. New model additions have separate evidence, including the [T-family report](validation/phenomt.md). |
| Derivatives and Fisher matrices | [Inference parity tests](https://github.com/xangma/diffgw/blob/main/tests/parity/test_inference_parity.py), [Fisher example](../examples/02_fisher_analysis.py) | Selected parameters, models and derivative orders. Waveform-value agreement alone does not validate derivatives or posterior uncertainties. |
| Relative-binning likelihood | [Numerical tests](https://github.com/xangma/diffgw/blob/main/tests/test_relative_binning.py) and inference parity tests | [Local approximation and gradient checks](validation/inference.md#relative-binning). There is no published DiffGW timing campaign establishing a general likelihood speedup. |
| Physical calibration and provenance | [Provenance ledger](provenance_ledger.md) and [third-party notices](../THIRD_PARTY_NOTICES.md) | Literature, calibration sources, component checks, and unresolved asset provenance. A reference implementation check is distinct from independent physical validation. |

## Waveform agreement and physical validation

Start with [execution restrictions](models.md#execution-restrictions) for the
configuration you intend to use. The [differences from LAL](differences_from_lal.md#model-specific-differences)
summarize deliberate model differences and unresolved reference failures.
The [PhenomT report](validation/phenomt.md) contains its measured errors,
regression thresholds and component diagnostics.

The [dated FP64 campaign](package_benchmarks.md#numerical-qualification) checks
both polarizations of every timed binary, with direct errors and an explicitly
defined mismatch. Its strict thresholds differ from the model-specific catalog
and precession checks. Each threshold belongs to its stated reference and
configuration; a passing label from one check cannot qualify another.

Independent physical validation requires an appropriate physical reference and
coverage statement. The XHM/XPHM NR checks described in the LAL-differences page use a
restricted ringdown phase band and reference amplitudes; they do not establish
full-waveform amplitude and phase accuracy. Calibration provenance is tracked
in the [ledger](provenance_ledger.md), separately from implementation agreement.

## Derivative support and evidence

| Path | Documented support or check | Scope and missing evidence |
| --- | --- | --- |
| Torch/JAX waveform derivatives and Fisher matrices | [Fisher tutorial](tutorials/02_fisher_analysis.md) and [inference parity tests](https://github.com/xangma/diffgw/blob/main/tests/parity/test_inference_parity.py) | Selected models, parameters and orders. Cross-framework agreement is not an independent derivative reference or validation throughout parameter space. |
| IMRPhenomT family on fixed grids | First-order AD; implicit first derivative through frequency inversion | Higher orders through inversion are unsupported; automatic duration is eager and the start mask can be discontinuous. See [model restrictions](imrphenomt.md#arbitrary-grids-and-derivatives). |
| IMRPhenomD relative-binning likelihood | CPU FP64 gradient-vector check for distance and phase, with relative error below `1e-2` required by the test | One perturbation on the stated grid; no posterior-wide mass/spin gradient claim. See [exact test scope](validation/inference.md#relative-binning). |
| Other model/option/parameter combinations | Use model-specific documentation and tests | No complete model-by-parameter derivative qualification matrix is published here. Waveform-value checks and framework autodiff availability do not supply it. |

Automatic differentiation follows the implemented numerical operations. Precision,
interpolation, piecewise branches, solver tolerances and derivative order affect
the result. Independently check the required parameter path, for example against
converged finite differences. Published forward-only timings do not measure
[derivative performance](performance.md#workloads-without-published-measurements).

## Relative binning

The [inference validation report](validation/inference.md#relative-binning)
records the dense-versus-binned likelihood and gradient checks, their grids,
parameters and acceptance criteria. Its approximation must be checked over the
region used for inference; no general likelihood speedup is established by the
waveform campaign.

## Gradients and statistical interpretation

Fisher forecasts and MCMC results require statistical checks beyond derivative
agreement. Read the [inference report](validation/inference.md#gradients-and-statistical-interpretation)
for covariance regularization and interpretation. The [NUTS API walkthrough](tutorials/03_parameter_estimation_nuts.md)
is illustrative and does not supply a converged posterior or a proper box prior.

## Reproducing the inference checks

Use the [inference check commands](validation/inference.md#reproducing-the-inference-checks).
The [catalog commands](differences_from_lal.md#reproducing-model-checks) and
[PhenomT report](validation/phenomt.md) cover waveform checks. Record revisions,
configurations, results and skips; tests linked on `main` describe the current
checks, whereas dated reports describe their recorded runs. A skipped optional
reference test supplies no agreement evidence.
