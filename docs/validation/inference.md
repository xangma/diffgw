# Inference validation

This report covers likelihood checks and interpretation of derivative-based
forecasts. See the [validation overview](../validation.md) for waveform evidence.

## Parameter-estimation campaign

Parameter estimation infers a posterior distribution for source parameters by
combining a waveform likelihood with a prior. Our validation generates signals
with known parameters and checks numerical integration and sampling against that
known experiment. Start with the [current results and their meaning](inference/parameter-estimation-results.md).

These are bounded **two-parameter** experiments with other parameters fixed.
The [30 September study](inference/parameter-estimation-campaign.md) tests
zero-noise grid integration across the waveform inventory. The later completion
study separately checks common-data reference posteriors, native samplers and
noisy ensemble calibration. A plan, successful process or hardware benchmark is
not a passed scientific stage.

| Question | Required evidence |
| --- | --- |
| Does numerical integration resolve this posterior? | Stable joint density, normalization, marginal probabilities and quantiles under the frozen refinement checks |
| Does the sampler recover that posterior? | Mixing, effective samples, Monte Carlo uncertainty, divergences and agreement with an accepted independent numerical reference |
| Are credible probabilities calibrated? | Executed prior-drawn noisy ensembles, failed/missing-case denominators and finite-ensemble simultaneous uncertainty |
| Is the waveform physically accurate? | Independent physical injections with characterized uncertainty; same-model recovery cannot establish this |

The [common-data pilot](inference/nrtidal-convergence-pilot.md),
[Torch sampler pilot](inference/native-sampler-validation.md),
[initial noisy study](inference/noisy-calibration.md) and
[compiled noisy reference study](inference/compiled-noisy-reference-pilot.md)
retain their own frozen settings and historical outcomes. The current result
summary keeps these experiments separate rather than promoting earlier pilots
to later campaign acceptance.

Self-injection tests the implemented waveform and inference pipeline together;
shared waveform errors can remain invisible. A pinned external implementation
checks implementation compatibility. Independent physical signals are needed
to measure waveform-model bias. Existing failures retain their separate status
in the [qualification dashboard](../waveforms/qualification-status.md#how-to-read-the-claims).

### Freeze the experiment

Each case must name the public dispatcher or direct class, model/version,
frontend, options, modes, precision, device, two free native parameters, fixed
parameters, proper prior density/support, injection truth, noise seed, detector
projection, PSD, time/frequency grid and conditioning. Use meaningful mass
coordinates for compact binaries and appropriate physical parameters for other
families. Fixed spins, distance, phase and geometry remain outside the recovered
scope. An external-data dependency or unsupported route remains explicit in the
inventory rather than disappearing from the denominator.
Record any scalar mapping, native batching or optional JIT strategy separately
from physical model options; a passing adapted experiment does not qualify an
untested public batch path. The campaign [case inventory](inference/parameter-estimation-campaign.md#coverage-and-claim)
identifies the selected waveform and fixed FFT adapters.

Specify the prior in the coordinates actually integrated. A uniform rectangle
in component masses differs from a uniform rectangle in chirp mass and mass
ratio; transforming the former requires its Jacobian and transformed support.
Evaluate zero density outside the support. A finite out-of-range penalty is not
a proper bounded prior. Preserve body ordering and model validity constraints.

Freeze case IDs, grids, priors, refinement criteria, numerical tolerances and
exclusions before running the campaign. Record amendments separately with their
reason and new version; retain failed results. Never choose prior widths or a
stopping rule after inspecting whether the truth lies in an interval.

### Match noise to the likelihood

Use the maintained dense Whittle likelihood and matching detector projection
for injection and candidate templates. For positive-frequency bins with PSD
\(S_i\) and quadrature weights \(w_i\), its residual term is

\[
\log\mathcal L=-2\sum_i w_i\,|d_i-h_i|^2/S_i.
\]

Independent real and imaginary noise components therefore have variance
\(S_i/(4w_i)\) for this discretized Gaussian experiment. The frequency-array
API uses trapezoidal integration: on a uniform grid, endpoint weights are half
an interior weight. The scalar-`df` API uses rectangular weights. Generate noise
with the selected convention and include only bins with positive integration
weight. This quadrature-matched experiment does not establish a realistic
detector-noise simulation. State any real/DC/Nyquist-bin treatment separately.

For time-domain models, record the time grid, duration, window, Fourier
normalization, frequency mask and common candidate/injection transformation.
Changing duration or padding must not silently change the likelihood weighting.
Zero-noise runs retain PSD weighting and isolate recovery bias; noisy runs use
independent complex noise with recorded seeds. Requested SNR and achieved
noise-weighted SNR are distinct fields.

### Accept a bounded posterior result

The dense grid supplies a sampler-independent baseline only when numerical
integration converges. Integrate likelihood times prior with parameter-space
quadrature weights, using stable log arithmetic. Record the finite log
normalizer, moments, marginal CDFs/credible intervals, mode and boundary mass.
Renormalization to one alone is not a convergence check. The implemented
likelihood omits parameter-independent Gaussian normalization; its grid
normalizer is not a model-evidence comparison across different datasets.

| Check | Required evidence |
| --- | --- |
| Valid case | Finite, nonzero templates and PSD; declared parameter support and transformations; correct native shapes and options |
| Independent likelihood check | Matched fixed-point likelihoods and analytic/simple-model tests of weights, noise and prior integration |
| Grid convergence | Coarse/refined agreement of the joint density, log normalizer, marginal quantiles, interval widths and joint-mode coordinates under frozen tolerances; resolve intervals relative to cell width |
| Prior boundaries | Report posterior mass near every edge and whether an edge is physical or artificial; separate convergence of the declared bounded posterior from limitations caused by prior support |
| Identifiability | Show posterior change from the prior in both free dimensions; report degeneracy or prior dominance rather than a nominal two-parameter success |
| Frontend agreement | A separate parity check compares common data, PSD, prior and candidate points; independent frontend self-injections do not establish this agreement |

The reference experiments use uniform independent native-coordinate priors and
tensor-product trapezoidal posterior integration. The initial campaign declares
CPU FP64, target SNR 20, zero noise, and square grids with 31, 61, 121, 241 and
at most 481 nodes per axis. It stops after an adjacent-grid comparison passes
every gate:

| Numerical gate | Frozen threshold |
| --- | --- |
| Joint-density total-variation distance | At most 0.05, using the fine-grid quadrature measure and normalized bilinear interpolation of the coarse density |
| Absolute log integrated-likelihood drift | At most 0.05 |
| Largest change of marginal 5%, 50% or 95% quantile | At most 5% of that axis's fine-grid 90% interval width |
| Change of marginal 90% interval width | At most 5% of the fine-grid width |
| Change of each joint-mode coordinate | At most one coarse-grid interval |
| Fine-grid resolution of each marginal 90% interval | At least four intervals; width must be finite and positive |

Mean and marginal-CDF drift are recorded diagnostics without separate acceptance
thresholds. Each edge diagnostic integrates the outer 5% of the frozen prior
span; mass above 1% marks that axis `prior_support_limited`. Joint and marginal
KL divergence from the uniform prior below 0.01 nats marks `prior_dominated`.
These are declared diagnostic conventions, not universal PE requirements.
Neither flag changes the numerical convergence outcome. Passing these gates
establishes this bounded numerical baseline, rather than a proof of convergence
for every possible posterior shape.

Protocol v2 adds the joint-density gate before the full campaign. A v1 pilot
could pass the marginal and normalizer checks while a thin correlated ridge
remained visibly underresolved. Marginal resolution alone does not establish
transverse joint resolution. Preserve those v1 preflight receipts and their
original criteria; they do not qualify as a v2 completed baseline. The joint
comparison is
\(\frac12\sum_{ij}w_iw_j|p^{\rm fine}_{ij}-\widetilde p^{\rm coarse}_{ij}|\),
where both densities integrate to one under the same fine-grid weights.
The subsequent v2 IMRPhenomD pilot remained underresolved at 121 nodes per
axis. Before the full campaign, the declared refinement schedule was extended
to 241 and 481 with the same priors, waveform options and acceptance thresholds.
This is numerical refinement, not retuning of the waveform or prior.

A numerically converged bounded posterior may still be prior dominated or
limited by its support. Keep those classifications separate from convergence;
do not expand a frozen prior merely to obtain a passing result.

A converged noisy posterior may exclude the injected truth from a chosen
credible interval. That exclusion alone neither fails the waveform nor justifies
altering the experiment. Report truth CDFs, recovered-minus-truth bias, posterior
width and intervals as measured diagnostics. An individual inclusion is not a
qualification gate. Failed execution, failed numerical checks and inconclusive
recovery must remain distinct from a measured physical-model discrepancy.

An ensemble calibration/P-P claim requires prior-drawn truths and data from the
matching generative likelihood, with finite-ensemble uncertainty and a declared
selection rule. Fixed-truth noise repeats, a zero-noise grid or an NR catalog do
not by themselves establish coverage. Marginal calibration also does not prove
joint posterior correctness. See [Talts et al.](https://arxiv.org/abs/1804.06788).

Qualify approximate likelihoods and gradient samplers separately against this
baseline over posterior support. Full-dimensional PE, gradients, physical
accuracy, real-noise robustness and population coverage remain separate scopes.

### Retain the evidence

Keep maintained case/configuration generators, focused regression tests,
compact receipts and a few curated figures in the repository. Receipts must
record source revision/dirty state, dependencies, full case inventory,
passed/failed/blocked/unexecuted cases, settings, frozen thresholds, results,
and hashes identifying the retained data and executed source. Retain bulk posterior arrays, generated
plot collections and logs externally with durable links, SHA256 hashes and
sizes; a scratch `/tmp` path is not an archive.

Retain the original executed source archive and its hash when maintained code
changes during or after execution. A current checkout or a later classification
fix cannot stand in for that snapshot. Preserve historical receipt outcomes;
record any later interpretation separately, and identify the source of
aggregation and plotting utilities independently of waveform execution.

The campaign report links its actual figures and receipts once available. Useful
figures show two-dimensional posterior/prior/truth overlays, numerical
refinement and recovery/bias summaries across cases. Use P-P/rank plots only for
an appropriate executed calibration ensemble. Follow the
[inference guide](../guides/inference.md) for API usage; the
[NUTS tutorial](../tutorials/03_parameter_estimation_nuts.md) is an illustrative
workflow, not campaign evidence.

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
[IMRPhenomT](../imrphenomt.md) documents fixed-grid first-order derivatives and
limitations of its frequency inversion. Check the relevant parameter path
against independent derivatives or converged finite differences.

The Fisher tools form noise-weighted products of waveform derivatives. An
inverse Fisher matrix is a local uncertainty forecast under additional
assumptions; it is not generally the posterior covariance. Eigenvalue
regularization also changes that inverse. See the [Fisher tutorial](../tutorials/02_fisher_analysis.md)
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
[benchmark protocol](../benchmarking_protocol.md).
