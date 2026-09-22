# User guide

Install with the [installation guide](installation.md), select a model from the
[waveform catalog](models.md), or start with the [quickstart tutorial](tutorials/01_quickstart_and_parity.md).

<a id="waveform-and-analysis-user-guide"></a>
<a id="table-of-contents"></a>
<a id="1-quickstart"></a>
<a id="pytorch-quickstart-torchwave"></a>
<a id="jax-quickstart-jaxwave"></a>
<a id="2-model-availability-cross-framework-parity"></a>
<a id="4-frequency-domain-waveforms-get_fd_waveform"></a>
<a id="basic-evaluation-frequency-grids"></a>
<a id="batched-waveform-generation"></a>
<a id="generic-3d-precessing-spins-msa-spintaylor-pnr"></a>
<a id="higher-order-multipoles-ell-m"></a>
<a id="tidal-deformability-for-neutron-stars"></a>
<a id="orbital-eccentricity"></a>
<a id="6-time-domain-waveforms-get_td_waveform"></a>
<a id="native-imrphenomt"></a>
<a id="7-black-hole-ringdown-kerr-quasinormal-modes"></a>
<a id="ringdown-waveforms-get_ringdown_waveform"></a>
<a id="programmatic-kerr-qnm-eigensolver-torchwaveqnmsolve"></a>
<a id="8-specialized-astrophysical-models"></a>

## Waveform generation

- [Quickstart](guides/waveform-generation.md#quickstart)
- [Frequency-domain waveforms](guides/waveform-generation.md#frequency-domain-waveforms-get_fd_waveform)
- [Time-domain waveforms](guides/waveform-generation.md#time-domain-waveforms-get_td_waveform)
- [Ringdown and quasinormal modes](guides/waveform-generation.md#black-hole-ringdown-and-kerr-quasinormal-modes)
- [Specialized models](guides/waveform-generation.md#specialized-astrophysical-models)

<a id="3-hardware-acceleration-precision"></a>
<a id="pytorch-device-precision"></a>
<a id="5-jax-frontend-acceleration-jaxwave"></a>
<a id="waveform-generation-waveformgenerator"></a>
<a id="just-in-time-compilation-jaxjit"></a>
<a id="vectorized-batch-evaluation-jaxvmap"></a>
<a id="autodiff-parameter-gradients-jaxgrad"></a>
<a id="shape-bucketing-for-mcmc-sampling-bucketedwaveformevaluator"></a>
<a id="unified-dispatch-via-diffgw"></a>
<a id="9-automatic-differentiation-in-pytorch"></a>

## Execution and differentiation

- [Hardware and precision](guides/execution.md#hardware-acceleration-and-precision)
- [JAX compilation, batching and gradients](guides/execution.md#jax-compilation-batching-and-transforms)
- [PyTorch autodiff](guides/execution.md#automatic-differentiation-in-pytorch)

<a id="10-detector-networks-projections"></a>
<a id="ground-based-networks-h1-l1-v1-k1-et-ce"></a>
<a id="power-spectral-densities-psds"></a>
<a id="space-based-interferometers-lisanetwork"></a>
<a id="pulsar-timing-arrays-ptanetwork"></a>
<a id="11-bayesian-inference-suite"></a>
<a id="signal-to-noise-ratio-snr"></a>
<a id="whittle-gaussian-likelihood"></a>
<a id="accelerated-relative-binning-likelihood"></a>
<a id="fisher-information-matrix-sky-localization"></a>
<a id="gradient-based-map-optimization"></a>
<a id="no-u-turn-sampler-nuts-mcmc"></a>

## Detectors and inference

- [Detector networks and projections](guides/inference.md#detector-networks-and-projections)
- [Likelihoods, Fisher analysis and sampling](guides/inference.md#bayesian-inference-suite)

<a id="12-command-line-tools"></a>
<a id="torchwave-bench"></a>
<a id="torchwave-profile"></a>
<a id="torchwave-data"></a>
<a id="torchwave-generate-qnm"></a>
<a id="torchwave-generate-london-qnm"></a>
<a id="controlled-benchmarking-protocol"></a>
<a id="running-a-multi-backend-benchmark-campaign"></a>
<a id="generating-figures"></a>

## Command-line tools

- [Benchmark, profile, data and QNM commands](guides/cli.md)
