# Parameter-estimation results

Reviewed 20 October 2026. These are **restricted two-parameter numerical
validation experiments**, with all other source and detector parameters fixed.
They test whether the implemented likelihood, prior, waveform adapter and sampler
recover the intended posterior. They do not establish full-dimensional inference
or physical waveform accuracy.

## What the experiments test

Parameter estimation (PE) infers a probability distribution for source parameters
from data. The posterior combines a waveform-based likelihood with a prior.
In these experiments, the injected parameters are known, so numerical recovery
can be checked against a separately integrated posterior.

The validation has three distinct stages:

1. **Reference posterior:** refine numerical integration over the complete
   declared prior. Check the joint density, normalization, marginal probabilities
   and quantiles. In the completion study, also check common-data Torch/JAX
   agreement and independent quadrature/CDF calculations.
2. **Sampler validation:** compare actual retained NUTS draws with an accepted
   reference. Check mixing, effective sample size, Monte Carlo uncertainty,
   divergences, marginal probabilities and joint posterior structure.
3. **Ensemble calibration:** recover independently generated truths and matched
   noise across the declared 50-case ensemble. Assess coverage with finite-ensemble
   simultaneous uncertainty, keeping failed and missing cases in the denominator.

A stage passes only its own frozen checks. A completed process, smooth-looking
posterior or injected truth inside one credible interval is insufficient.
Zero-noise recovery retains PSD weighting but does not test noisy coverage.
Fixed-truth noise repeats also do not constitute a prior-drawn calibration
ensemble. See the [protocol](../inference.md#parameter-estimation-campaign).

## Current results

The [original 30 September grid study](parameter-estimation-campaign.md) evaluated
29 waveform/API entries in each frontend: 28 supported waveform scopes and one
component-only entry. **Each frontend recorded 18 converged grid baselines,
10 unresolved results and one blocked component.** Those historical outcomes
remain unchanged by later studies.

The completion study has accepted common-data zero-noise references for four
models and **17 unique model/condition/frontend sampler records, comprising
272,000 retained draws**. These records span four models; they are not 17
qualified waveforms. Rechecking the same record adds no new case or ensemble
event. **No supported model has completed the entire expanded campaign or its
50-case calibration stage: 0/28.**

| Model | Accepted paired zero-noise reference | Accepted individual fixed-noise reference pairs | Accepted Torch sampler cases | Accepted JAX sampler cases |
| --- | --- | --- | --- | --- |
| IMRPhenomD | Yes | 4 | Zero + fixed 000–003 | Fixed 001, 003 |
| IMRPhenomPv2 | Yes | 4 | Zero | Zero + fixed 000 |
| MemoryWaveform | Yes | 4 | Zero + fixed 000–003 | None |
| NewtonianEccentricEMRI | Yes | None | Zero | Zero |

“Paired” means Torch and JAX analyze common data under the matched prior and
likelihood. A sampler case is one frontend's accepted run for one condition;
four chains do not count as four cases. Memory's four individual noisy references
are accepted, while the combined four-case stage remains incomplete. Historical
pilot samples are reported separately and are not added to these counts.

Newtonian's accepted Torch zero-noise case uses four chains, 1,000 warmup steps
and 4,000 retained draws per chain, target acceptance 0.95 and maximum depth 8.
The free parameters are `p0` and `e0`, with independent uniform priors
`p0 ∈ [15.5, 16.5]`, `e0 ∈ [0.1, 0.3]` and injected values `(16, 0.2)`.
All original sampler gates pass: maximum rank/folded R-hat 1.0029, minimum bulk
ESS 2,688.9, minimum tail ESS 2,884.5, zero retained divergences, and all six
marginal-CDF and nine joint Rosenblatt checks. One warmup divergence is reported
separately. This qualifies that bounded zero-noise sampler case only.

## Unresolved results and incomplete stages

| Waveform/API scope | Interpretation of the current evidence |
| --- | --- |
| TaylorF2, TaylorF2_HM, TaylorF2_Eccentric; IMRPhenomHM, IMRPhenomXAS, IMRPhenomXHM | Original Torch/JAX grid baselines converged. The stricter completion references and sampler/calibration stages are not accepted. |
| IMRPhenomP, IMRPhenomXPNR | Original grids converged; no accepted completion reference or sampler. |
| IMRPhenomXP, IMRPhenomXPHM | Original grids converged. Additional convention/option checks remain unresolved; these do not erase the historical grid outcome. |
| IMRPhenomD_NRTidal | The separate refined-reference and Torch zero-noise NUTS pilots passed. Compiled noisy references accepted 3/4 cases; fixed 002 and a compiled precision check remain unresolved. The original rectangular-grid outcome is still unresolved. |
| IMRPhenomD_NRTidalv2 | Original grid and later independent CDF convergence remain unresolved. No accepted completion reference or sampler. |
| IMRPhenomNSBH | The separate 3841-by-3841 supplement passes both required consecutive refinements, frontend parity and Gauss quadrature. Independent CDF convergence still fails, so the reference is not accepted. |
| IMRPhenomT | Construction and derivative prerequisite checks pass. The completion reference and samplers are not accepted; its original grid result remains unresolved. |
| IMRPhenomTHM | No accepted completion reference or sampler; original grid convergence remains unresolved. |
| IMRPhenomTP, IMRPhenomTPHM | Reference convergence and onset/branch behavior remain unresolved. |
| IMRPhenomXPHMSpinTaylor | Original credible-interval-width refinement failed; additional parity and optimizer checks remain unresolved. |
| EMRIKludge | Later refinement has only one consecutive pass where two are required. No accepted completion reference or sampler. |
| ContinuousWave | Refinement and independent quadrature remain unresolved. |
| KerrRingdown, KerrRingdownDirect | Original grids converged; no accepted completion reference or sampler. |
| SEOBNRv4_ROM | Original grids converged with the required external ROM data; no accepted completion reference or sampler. |
| ParameterizedPostEinstein | Original grids converged; later reference-density and matched-noise checks remain unresolved. |
| TEOBResumSHamiltonian | A Hamiltonian component without a public waveform API; excluded from the 28 supported waveform PE scopes. |

For NSBH, the final independent CDF refinement changes a cut probability by
0.00343784, above its frozen 0.0002 limit. The three second-mass quantile checks
also fail their required integration-uncertainty margins. Stable normalization
and a passing grid do not remove these failed checks. This is unresolved
numerical reference integration, not evidence by itself of a physical waveform
error.

## What a pass means

| Evidence | Supported conclusion | Outside its scope |
| --- | --- | --- |
| Converged bounded grid | Numerical posterior integration met the stated checks for that experiment | All parameter values, all priors or every waveform option |
| Accepted sampler case | Its retained draws agree with the accepted reference within the frozen mixing and uncertainty checks | Every chain length, other datasets or full-dimensional PE |
| Torch/JAX common-data agreement | The two implementations agree for the tested data, options and posterior | Independent physical accuracy; shared errors can remain |
| Prior-drawn noisy calibration | Coverage under the assumed generative model, once the declared ensemble passes | Real-noise robustness or physical-model accuracy |

These are primarily same-model injection-and-recovery experiments. Independent
physical injections, such as appropriately characterized numerical-relativity
signals, are needed to measure waveform-model bias. Existing mathematical,
implementation and physical-accuracy failures retain their separate status in
the [waveform qualification dashboard](../../waveforms/qualification-status.md).
Prior-limited or prior-dominated posteriors must also retain those labels even
when numerical integration converges.

## Evidence and use

The [compact current results](parameter-estimation-results-20261020.json) list all
29 inventory entries, the 17 accepted sampler records, review identifiers and
SHA256 bindings. The [original grid receipt](parameter-estimation-20260930.json)
retains the frozen case options, priors, measured values and original outcomes.
Bulk chains, arrays and full reproduction records are retained externally;
their hashes identify the reviewed data, and no public bulk download is implied.

The earlier [common-data pilot](nrtidal-convergence-pilot.md),
[Torch sampler pilot](native-sampler-validation.md),
[initial noisy study](noisy-calibration.md) and
[compiled noisy references](compiled-noisy-reference-pilot.md) retain their
own dated settings and outcomes. They are separate experiments.
For practical inference, start with the [PE guide](../../guides/parameter-estimation.md)
and [IMRPhenomD example](../../tutorials/04_public_parameter_estimation.md).
