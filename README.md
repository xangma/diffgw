# diffgw (torchwave & jaxwave)

<p align="center">
  <a href="https://xangma.github.io/diffgw/"><img src="https://img.shields.io/badge/docs-diffgw-indigo.svg" alt="Documentation"></a>
  <a href="https://github.com/xangma/diffgw"><img src="https://img.shields.io/badge/github-diffgw-blue.svg" alt="GitHub"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/framework-PyTorch%20%7C%20JAX-orange.svg" alt="Frameworks">
  <img src="https://img.shields.io/badge/precision-FP64%20Parity%20%3C%201e--11-success.svg" alt="Precision">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
</p>

**diffgw** is a high-performance, batched, autograd-differentiable gravitational-wave (GW) modeling and Bayesian parameter estimation engine. It provides two native, first-class, co-equal frontends powered by a shared zero-dependency analytical physics core (`diffgw_core`):

- **`torchwave`**: Built natively in PyTorch for CPU, NVIDIA CUDA, and Apple Silicon MPS, featuring batched tensor computation `(B,) -> (B, N)` and autograd differentiation.
- **`jaxwave`**: Built natively in JAX for CPU, NVIDIA CUDA GPU, and Google Cloud TPU, featuring XLA just-in-time compilation (`@jax.jit`), automatic vectorization (`jax.vmap`), and forward/reverse-mode automatic differentiation (`jax.grad`, `jax.jvp`).

Both frontends are fully self-contained, pure Python packages distributed under the permissive **MIT License**, requiring only PyTorch or JAX and standard scientific Python dependencies. All waveform models, calibration surfaces, and ringdown solvers are independently implemented from published peer-reviewed literature under strict clean-room principles. Optional external datasets (such as `SEOBNRv4ROM_v3.0.hdf5`) can be managed seamlessly via [`diffgw-data`](docs/external-data.md); see [third-party notices](THIRD_PARTY_NOTICES.md).

---

## Key Highlights

- **Fused JIT Scalability**: Generates up to **14.4 million waveforms per second** on an NVIDIA GeForce RTX 5090 using fused Inductor JIT kernels (`torch.compile`) and JAX XLA compilation.
- **Double-Precision Parity**: PyTorch (`torchwave`) and JAX (`jaxwave`) agree to double-precision numerical noise ($L_2 \le 10^{-11}$, mismatch $\le 10^{-14}$), continuously verified by 1,021 automated tests.
- **Co-Equal Frontends**: Unified physics contracts guarantee exact functional and numerical parity between PyTorch and JAX across all 12 waveform model families.
- **Native Autodiff**: Exact parameter gradients for forward-mode Fisher Information Matrix (FIM), Maximum A Posteriori (MAP) optimization, and No-U-Turn Sampler (NUTS) MCMC.
- **Detector Networks**: Built-in antenna projections and noise PSDs for Ground interferometers (`H1`, `L1`, `V1`, `KAGRA`, `ET`, `CE`), Space (`LISANetwork`), and Pulsar Timing Arrays (`PTANetwork`).
- **Accelerated Likelihoods**: Standard Whittle Gaussian likelihood and $50\times$ faster heterodyned Relative Binning for ultra-fast parameter estimation.

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

# Generate equal-mass binary black hole waveform
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

`jaxwave` runs natively in FP64 precision, fully supporting `@jax.jit` compilation, `jax.vmap` vectorization, and automatic differentiation (`jax.grad`):

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

Both `torchwave` and `jaxwave` provide end-to-end differentiable gravitational-wave inference:

- **Detector Geometries & Networks**: Built-in ground-based (`H1`, `L1`, `V1`, `KAGRA`, `EinsteinTelescope`, `CosmicExplorer`), space-based (`LISANetwork` with TDI $A, E, T$ channels), and pulsar timing arrays (`PTANetwork` with Hellings-Downs correlations).
- **Noise Power Spectral Densities (PSDs)**: Analytical PSD models for advanced LIGO (Zero Detuned High Power), advanced Virgo, KAGRA, Cosmic Explorer, and Einstein Telescope.
- **Likelihood Functions**: Fast Whittle frequency-domain Gaussian likelihood, accelerated relative binning likelihood, and optimal / matched-filter SNR computation.
- **Fisher Information & Parameter Estimation**: Exact Fisher Information Matrix computation via autodiff Hessian (`jax.jacfwd` / `torch.autograd.functional.hessian`), sky localization, gradient-based MAP optimization, and No-U-Turn Sampler (NUTS) MCMC.

```python
# PyTorch Inference:
from torchwave.detectors.psd import aLIGOZeroDetHighPower
from torchwave.inference.likelihood import WhittleLikelihood, RelativeBinningLikelihood
from torchwave.inference.fisher import optimal_snr, fisher_matrix

# JAX Inference:
from jaxwave.detectors.psd import aLIGOZeroDetHighPower as jax_aLIGO
from jaxwave.inference.likelihood import WhittleLikelihood as jax_WhittleLikelihood
from jaxwave.inference.fisher import fisher_information_matrix as jax_fisher
```

---

## Tutorials & Runnable Examples

Step-by-step walkthroughs paired with runnable standalone scripts:

1. **[Tutorial 1: Quickstart & Parity Check](docs/tutorials/01_quickstart_and_parity.md)** ([`examples/01_quickstart_and_parity.py`](examples/01_quickstart_and_parity.py)): Waveform generation, batched evaluations, and double-precision PyTorch vs JAX parity verification ($L_2 < 10^{-11}$, mismatch $< 10^{-14}$).
2. **[Tutorial 2: Fisher Analysis](docs/tutorials/02_fisher_analysis.md)** ([`examples/02_fisher_analysis.py`](examples/02_fisher_analysis.py)): Forward-mode autodiff (`torch.autograd.forward_ad` & `jax.jvp`), noise-weighted Fisher Information Matrix, and Cramér-Rao parameter bounds.
3. **[Tutorial 3: Parameter Estimation & NUTS](docs/tutorials/03_parameter_estimation_nuts.md)** ([`examples/03_parameter_estimation_nuts.py`](examples/03_parameter_estimation_nuts.py)): Synthetic signal injection in detector noise, accelerated relative binning likelihood, Adam MAP optimization, and No-U-Turn Sampler (NUTS) MCMC.

---

## Waveform Models & Framework Parity

All models below are natively implemented in **both PyTorch (`torchwave`) and JAX (`jaxwave`)** with complete cross-framework parity. Evaluated waveforms in PyTorch and JAX agree to double-precision numerical noise ($L_2 \le 10^{-11}$, mismatch $\le 10^{-14}$), continuously verified by 1,021 automated tests on CPU and remote NVIDIA RTX 4090 GPU (`len`).

| Family | Approximant(s) | Harmonics / Multipoles | Physical Features & Regimes |
| :--- | :--- | :---: | :--- |
| **Post-Newtonian** | `TaylorF2`<br>`TaylorF2_HM`<br>`TaylorF2_Eccentric` | $(2,2)$<br>$(2,1), (3,3), (3,2), (4,4)$<br>$(2,2)$ | 3.5PN point-particle & spin phasing, BNS tidal deformability<br>Analytical 3PN higher-order multipoles<br>Moore et al. (2016) 3PN stationary-phase eccentric corrections |
| **Aligned-Spin IMR** | `IMRPhenomD`<br>`IMRPhenomHM`<br>`IMRPhenomXAS`<br>`IMRPhenomXHM` | $(2,2)$<br>$(2,\pm 2), (2,\pm 1), (3,\pm 3), (3,\pm 2), (4,\pm 4)$<br>$(2,\pm 2)$<br>$(2,\pm 2), (2,\pm 1), (3,\pm 3), (3,\pm 2), (4,\pm 4)$ | Aligned-spin dominant mode, 5-collocation intermediate<br>Multi-harmonic PhenomD calibration with mode mixing<br>4th-generation aligned-spin dominant mode (Pratten et al. 2020)<br>4th-generation higher multipoles (García-Quirós et al. 2020) |
| **Precessing IMR** | `IMRPhenomP` / `Pv2`<br>`IMRPhenomXP`<br>`IMRPhenomXPHM` | $(2,\pm 2)$ twisted<br>$(2,\pm 2)$ twisted<br>Multipolar twisted | Generic 3D spins with NNLO PN precession angles & Wigner $d$-matrices<br>4th-gen precessing carrier with MSA analytical precession<br>Multipolar precession supporting MSA (`v223`), SpinTaylor (`v320`), and PNR (`v330`) |
| **Tidal & Matter** | `IMRPhenomD_NRTidal`<br>`IMRPhenomD_NRTidalv2`<br>`IMRPhenomNSBH` | $(2,2)$<br>$(2,2)$<br>$(2,2)$ | Dietrich et al. (2017) rational Padé BNS tidal phasing (NRTidal v1)<br>Dietrich et al. (2019) NRTidal v2 with self-spin & Planck windowing<br>Neutron star–black hole model with tidal disruption & remnant torus |
| **Effective-One-Body** | `SEOBNR_ROM` (v4/v5)<br>`SEOBNR_ODE`<br>`TEOBResumS` | $(2,2)$<br>$(2,2)$<br>$(2,2)$ | Fast B-spline reduced-order surrogate for aligned-spin SEOBNR<br>Time-domain Hamiltonian ODE integrator with continuous QNM matching<br>Damour-Nagar EOB dynamics with spin-orbit coupling and tidal potentials |
| **Ringdown & Bursts** | `KerrRingdown`<br>`SineGaussian`<br>`MemoryWaveform` | $(2,2,0), (l,m,n)$<br>Burst packet<br>$(2,0)$ | Damped sinusoids from Kerr quasinormal modes (time & frequency domain)<br>Standard unmodeled sine-Gaussian burst wavelet<br>Non-linear Christodoulou gravitational-wave memory effect $h_{20}^{\text{mem}}(t)$ |
| **Specialized Models** | `ContinuousWave`<br>`EMRIKludge`<br>`ParameterizedPostEinstein`<br>`SpinPrecessionODE` | Monochromatic<br>Multi-harmonic<br>Modified phasing<br>Orbit-averaged | Continuous waves from triaxial rotating neutron stars with Doppler modulation<br>Extreme mass ratio inspirals (LISA) with Kerr geodesic orbital frequencies<br>Model-independent testing of General Relativity ($\beta_{\rm ppE}, b_{\rm ppE}$)<br>Orbit-averaged spin precession ODE integrator |
| **Time-Domain Tools** | `get_td_waveform`<br>`HybridMECO` | All models<br>Diagnostic | FFT conditioning and Parseval-preserving time-domain waveform generation<br>Minimum Energy Circular Orbit hybrid velocity & frequency root solver |

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

The complete documentation is organized into five balanced sections matching the [**Online Documentation Portal**](https://xangma.github.io/diffgw/):

- **User Guide**:
  - [Getting Started & Usage Guide](docs/usage.md) — Comprehensive guide covering approximants, parameters, batching, JIT, and detector networks.
  - [External Datasets](docs/external-data.md) — Managing offline ROM datasets and cache verification.
- **Tutorials**:
  - [1. Quickstart & Parity Check](docs/tutorials/01_quickstart_and_parity.md) ([`examples/01_quickstart_and_parity.py`](examples/01_quickstart_and_parity.py))
  - [2. Fisher Information Analysis](docs/tutorials/02_fisher_analysis.md) ([`examples/02_fisher_analysis.py`](examples/02_fisher_analysis.py))
  - [3. Bayesian Parameter Estimation & NUTS](docs/tutorials/03_parameter_estimation_nuts.md) ([`examples/03_parameter_estimation_nuts.py`](examples/03_parameter_estimation_nuts.py))
- **Benchmarks & Scientific Validation**:
  - [Performance & Scaling Report](docs/performance.md) — NVIDIA RTX 5090 scaling analysis across PyTorch eager, Inductor (`torch.compile`), JAX CUDA, and LAL.
  - [Differences from LAL](docs/differences_from_lal.md) — Mathematical improvements including $C^1$ cubic Hermite spline interpolation.
  - [Controlled Benchmark Protocol](docs/benchmarking_protocol.md) — Benchmarking methodology, Gate 1 scientific parity enforcement, and cryptographic receipts.
  - [Mathematical Provenance Ledger](docs/provenance_ledger.md) — Phasing derivations, calibration tables, and literature provenance.
- **API Reference**:
  - [`torchwave` API](docs/api/torchwave.md) — PyTorch frontend reference.
  - [`jaxwave` API](docs/api/jaxwave.md) — JAX frontend reference.
  - [`diffgw` API](docs/api/diffgw.md) — Unified dispatch reference.
- **Project & Community**:
  - [Citations](CITATIONS.md) — Literature citations and BibTeX entries.
  - [Contributing Guidelines](CONTRIBUTING.md) — Development workflow, testing protocols, and parity requirements.
  - [Third-Party Notices](THIRD_PARTY_NOTICES.md) — Licensing and attribution notices.

---

## Testing & Verification

To run the full suite of cross-framework mathematical parity tests ($L_2 \le 10^{-11}$, mismatch $\le 10^{-14}$):

```bash
pytest tests/parity/ -v
```

To run gradient and recompilation regression tests:

```bash
pytest tests/test_recompiles_and_gradients.py -v
```
