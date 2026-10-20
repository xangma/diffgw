# diffgw (torchwave & jaxwave)

<p align="center">
  <a href="https://xangma.github.io/diffgw/"><img src="https://img.shields.io/badge/docs-diffgw-indigo.svg" alt="Documentation"></a>
  <a href="https://github.com/xangma/diffgw"><img src="https://img.shields.io/badge/github-diffgw-blue.svg" alt="GitHub"></a>
  <img src="https://img.shields.io/badge/python-3.10%20--%203.14-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/framework-PyTorch%20%7C%20JAX-orange.svg" alt="Frameworks">
  <img src="https://img.shields.io/badge/precision-FP64%20Support-success.svg" alt="Precision">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
</p>

**diffgw** provides batched gravitational-wave models and inference tools in two native frontends, sharing constants and coefficient tables through the framework-independent `diffgw_core` package:

- **`torchwave`**: PyTorch tensor evaluation, batched parameters, and autograd on supported model paths.
- **`jaxwave`**: JAX evaluation with JIT compilation, vectorization, and automatic differentiation on supported model paths.

The original software is distributed under the **MIT License**, with PyTorch or JAX and scientific Python dependencies. Waveform models follow published peer-reviewed literature, with calibration provenance recorded explicitly. IMRPhenomT also uses the authors’ MIT-licensed calibration release and reconstruction formulas. Optional external datasets (such as `SEOBNRv4ROM_v3.0.hdf5`) are managed via [`diffgw-data`](docs/external-data.md); see [third-party notices](THIRD_PARTY_NOTICES.md).

---

## Key Highlights

- **Batched execution**: Tensor batches and optional compilation support repeated waveform evaluation. [Measured package benchmarks](docs/package_benchmarks.md) report workload-specific CPU/CUDA timings and numerical checks.
- **Numerical validation**: Reference comparisons and cross-framework tests cover declared cases with model-specific tolerances. See [validation evidence](docs/validation.md) for scope and outstanding differences.
- **Two native frontends**: PyTorch and JAX interfaces share physical constants and coefficient tables.
- **Automatic differentiation**: Supported model paths provide derivatives for Fisher forecasts, MAP optimization, and gradient-based sampling.
- **Detector models**: Ground-detector projections and analytical noise models, plus LISA and pulsar-timing response utilities.
- **Likelihoods**: Dense Whittle Gaussian likelihood and a relative-binning approximation that evaluates candidate waveforms at fewer frequencies. Accuracy and speed depend on the grid, binning, and reference waveform; see [likelihood validation](docs/validation/inference.md#relative-binning).
- **Parameter estimation**: Public `diffgw.inference` priors, posteriors and NUTS sampling with explicit CPU/GPU devices. Native Torch NUTS and optional NumPyro/JAX return named physical samples and diagnostics; see [the PE guide](docs/guides/parameter-estimation.md). See [PE results and their interpretation](docs/validation/inference/parameter-estimation-results.md) for the tested scientific scope.

---

## Installation

You can install `diffgw` in editable mode with your choice of backends:

```bash
# Clone the repository
git clone https://github.com/xangma/diffgw.git
cd diffgw

# Option 1: Install with PyTorch support only
pip install -e ".[torch]"

# Option 2: Install with JAX support only
pip install -e ".[jax]"

# Option 3: Install all backends, inference tools, and documentation suite
pip install -e ".[all,dev,docs]"
```

---

## Quickstart

### 1. PyTorch Frontend (`torchwave`)

```python
import torch
import torchwave

# Generate a binary black hole waveform
hp, hc = torchwave.get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=30.0,
    mass2=25.0,
    chi1=0.2,
    chi2=-0.1,
    f_lower=20.0,
    f_final=512.0,
    delta_f=0.25,
    distance=410.0,
    dtype=torch.float64,
)
print("PyTorch polarization shape:", hp.shape)  # torch.Size([1969])

# Batched generation across 10,000 systems in a single call (B, N)
m1_b = torch.linspace(20.0, 50.0, 10000, dtype=torch.float64)
m2_b = torch.linspace(15.0, 45.0, 10000, dtype=torch.float64)
hp_batch, hc_batch = torchwave.get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=m1_b,
    mass2=m2_b,
    f_lower=20.0,
    f_final=512.0,
    delta_f=1.0,
)
print("Batched tensor shape:", hp_batch.shape)  # torch.Size([10000, 493])

# Autograd differentiation with respect to component masses
m1_param = torch.tensor(30.0, dtype=torch.float64, requires_grad=True)
m2_param = torch.tensor(20.0, dtype=torch.float64, requires_grad=True)
hp_diff, _ = torchwave.get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=m1_param,
    mass2=m2_param,
    f_lower=20.0,
    f_final=512.0,
    delta_f=0.5,
)
(hp_diff.real.pow(2) + hp_diff.imag.pow(2)).sum().backward()
print(f"Gradients: d||hp||^2/dm1 = {m1_param.grad:.4e}, d||hp||^2/dm2 = {m2_param.grad:.4e}")
```

### 2. JAX Frontend (`jaxwave`)

This FP64 IMRPhenomD example uses `@jax.jit`, `jax.vmap`, and `jax.grad`. Compatibility with these transformations depends on the model and parameter path:

```python
import jax
import jax.numpy as jnp
import jaxwave

# Class-based generator with cached XLA JIT compilation
generator = jaxwave.WaveformGenerator(
    approximant="IMRPhenomD",
    f_lower=20.0,
    f_final=512.0,
    delta_f=0.25,
    use_jit=True,
)

hp, hc = generator.generate_fd_waveform(
    mass1=30.0, mass2=25.0, chi1=0.2, chi2=-0.1
)
print("JAX polarization shape:", hp.shape)  # (1969,)

# Vectorized generation across batches with jax.vmap
m1_vec = jnp.array([20.0, 30.0, 40.0])
m2_vec = jnp.array([15.0, 30.0, 35.0])
chi1_vec = jnp.array([0.1, 0.2, -0.3])
chi2_vec = jnp.array([-0.1, 0.0, 0.2])

vmapped_wf = jax.vmap(generator.generate_fd_waveform, in_axes=(0, 0, 0, 0))
hp_vmapped, hc_vmapped = vmapped_wf(m1_vec, m2_vec, chi1_vec, chi2_vec)
print("JAX vmap shape:", hp_vmapped.shape)  # (3, 1969)

# Automatic differentiation with respect to chirp mass
@jax.jit
@jax.grad
def mc_loss(mc):
    hp_val, _ = generator(chirp_mass=mc, mass_ratio=1.0)
    return jnp.sum(jnp.abs(hp_val) ** 2)

grad = mc_loss(26.1)
print(f"d(Power)/d(M_c) = {grad:.4e}")
```

### 3. Unified Dispatch (`diffgw`)

You can also use the top-level `diffgw` namespace to seamlessly switch between backends:

```python
import diffgw

# Dispatch to PyTorch backend
hp_torch, hc_torch = diffgw.get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=30.0,
    mass2=25.0,
    f_lower=20.0,
    f_final=512.0,
    delta_f=0.25,
    backend="torch",
)

# Dispatch to JAX backend
hp_jax, hc_jax = diffgw.get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=30.0,
    mass2=25.0,
    f_lower=20.0,
    f_final=512.0,
    delta_f=0.25,
    backend="jax",
)
```

---

## Inference and Detector Networks

The inference modules combine waveform evaluation with detector and statistical tools:

- **Detector Geometries & Networks**: Built-in ground-based (`H1`, `L1`, `V1`, `KAGRA`, `EinsteinTelescope`, `CosmicExplorer`), space-based (`LISANetwork` with TDI $A, E, T$ channels), and pulsar timing arrays (`PTANetwork` with Hellings-Downs correlations).
- **Noise Power Spectral Densities (PSDs)**: Analytical PSD models for advanced LIGO (Zero Detuned High Power), advanced Virgo, KAGRA, Cosmic Explorer, and Einstein Telescope.
- **Likelihood Functions**: Dense Whittle likelihood, approximate relative binning likelihood, and optimal / matched-filter SNR computation.
- **Fisher Information & Parameter Estimation**: Noise-weighted inner products of waveform first derivatives, local Fisher uncertainty forecasts, sky localization, gradient-based likelihood optimization, and No-U-Turn Sampler (NUTS) MCMC.

```python
# PyTorch Inference:
from torchwave.detectors.psd import aLIGOZeroDetHighPower
from torchwave.inference import WhittleLikelihood, RelativeBinningLikelihood
from torchwave.inference.fisher import optimal_snr, fisher_information_matrix

# JAX Inference:
from jaxwave.detectors.psd import aLIGOZeroDetHighPower as jax_aLIGO
from jaxwave.inference.likelihood import WhittleLikelihood as jax_WhittleLikelihood
from jaxwave.inference.fisher import fisher_information_matrix as jax_fisher
```

---

## Tutorials & Runnable Examples

Step-by-step walkthroughs paired with runnable standalone scripts:

1. **[Tutorial 1: Quickstart & Parity Check](docs/tutorials/01_quickstart_and_parity.md)** ([`examples/01_quickstart_and_parity.py`](examples/01_quickstart_and_parity.py)): Waveform generation, batching, and a numerical comparison of one FP64 IMRPhenomD case.
2. **[Tutorial 2: Fisher Analysis](docs/tutorials/02_fisher_analysis.md)** ([`examples/02_fisher_analysis.py`](examples/02_fisher_analysis.py)): Forward-mode autodiff (`torch.autograd.forward_ad` & `jax.jvp`), noise-weighted Fisher Information Matrix, and Cramér-Rao parameter bounds.
3. **[Tutorial 3: NUTS API Walkthrough](docs/tutorials/03_parameter_estimation_nuts.md)** ([`examples/03_parameter_estimation_nuts.py`](examples/03_parameter_estimation_nuts.py)): An illustrative synthetic signal, relative binning, Adam optimization, and NUTS sampling.

---

## Waveform Models & Framework Parity

See the [model support catalog](docs/models.md) for physical coverage, backend
and derivative restrictions, external data, and known validation limitations.
The [package comparison](docs/comparison.md) explains differences from related
projects; [accuracy and validation](docs/validation.md) summarizes the evidence.

---

## Command-Line Suite

The package includes standalone console scripts for benchmarking, profiling, data management, and Kerr quasinormal mode generation:

1. **`torchwave-bench`**: Benchmark models across batch sizes, hardware platforms, and autograd backward passes.
2. **`torchwave-profile`**: Profile kernel execution, CUDA memory allocations, and cProfile hotspots.
3. **`torchwave-data`**: Inspect, verify, and download optional external ROM assets (see [external datasets guide](docs/external-data.md)).
4. **`torchwave-generate-qnm`**: Kerr QNM tables computed directly from the Leaver continued-fraction eigensolver.
5. **`torchwave-generate-london-qnm`**: London $N=61$ polynomial fits for higher-harmonic ringdown models.

---

## Documentation Directory

Read the [online documentation](https://xangma.github.io/diffgw/) or browse the
same pages in this repository:

| Section | Contents |
| --- | --- |
| Start here | [Overview](docs/index.md), [installation](docs/installation.md), [quickstart](docs/tutorials/01_quickstart_and_parity.md) |
| Package comparison | [Dated waveform coverage and execution capabilities](docs/comparison.md) |
| Waveforms | [Catalog and support](docs/models.md), [parameters and conventions](docs/conventions.md), [T-family guide](docs/imrphenomt.md) |
| User guide | [Generation, execution and inference](docs/usage.md), [Fisher tutorial](docs/tutorials/02_fisher_analysis.md), [NUTS API walkthrough](docs/tutorials/03_parameter_estimation_nuts.md) |
| Validation | [Evidence overview](docs/validation.md), [differences from LAL](docs/differences_from_lal.md), [T-family report](docs/validation/phenomt.md), [inference checks](docs/validation/inference.md) |
| Performance | [Results](docs/performance.md), [dated FP64 benchmark](docs/package_benchmarks.md), [running benchmarks](docs/benchmarking_protocol.md) |
| Reference | [Unified API](docs/api/diffgw.md), [Torch](docs/api/torchwave.md), [JAX](docs/api/jaxwave.md), [CLI](docs/guides/cli.md), [external data](docs/external-data.md), [provenance](docs/provenance_ledger.md) |

For project contributions and attribution, see [contributing](CONTRIBUTING.md),
[citations](CITATIONS.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

---

## Testing & Verification

To run cross-framework checks for the declared cases and per-test tolerances:

```bash
pytest tests/parity/ -v
```

To run gradient and batch-shape regression tests:

```bash
pytest tests/test_recompiles_and_gradients.py -v
```
