# diffgw (torchwave & jaxwave)

[![Documentation](https://img.shields.io/badge/docs-diffgw-blue.svg)](https://xangma.github.io/diffgw/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**diffgw** is a high-performance, batched, autograd-differentiable gravitational-wave (GW) modeling and inference engine. It provides two native, first-class, co-equal frontends powered by a shared zero-dependency analytical physics core (`diffgw_core`):

- **`torchwave`**: Built natively in PyTorch for CPU, NVIDIA CUDA, and Apple Silicon MPS, featuring batched tensor computation `(B,) -> (B, N)` and autograd differentiation.
- **`jaxwave`**: Built natively in JAX for CPU, NVIDIA CUDA GPU, and TPU, featuring XLA just-in-time compilation (`@jax.jit`), automatic vectorization (`jax.vmap`), and forward/reverse-mode automatic differentiation (`jax.grad`, `jax.jacfwd`).

Both frontends are fully self-contained, pure Python packages distributed under the permissive **MIT License**, requiring only PyTorch or JAX and standard scientific Python dependencies. All waveform models, calibration surfaces, and ringdown solvers are independently implemented from published peer-reviewed literature. Optional external datasets (such as `SEOBNRv4ROM_v3.0.hdf5`) can be managed seamlessly via [`diffgw-data`](docs/external-data.md); see [third-party notices](THIRD_PARTY_NOTICES.md).

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

# Option 3: Install all backends, inference tools, and dev dependencies
pip install -e ".[all,dev]"
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
    mass2=30.0,
    chi1=0.2,
    chi2=-0.1,
    f_lower=20.0,
    f_final=1024.0,
    delta_f=0.25,
    distance=410.0,
    device="cpu",
)

# Batched generation across 3 systems
m1 = torch.tensor([20.0, 30.0, 40.0])
m2 = torch.tensor([15.0, 30.0, 35.0])
hp_batch, hc_batch = torchwave.get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=m1,
    mass2=m2,
    f_lower=20.0,
    f_final=512.0,
    delta_f=1.0,
)
print("PyTorch Batched Shape:", hp_batch.shape)  # torch.Size([3, 493])

# JIT-compiled waveform evaluation with torch.compile (0 graph breaks, fused GPU kernels)
f_grid = torch.linspace(20.0, 512.0, 1024, dtype=torch.float64)

@torch.compile
def fast_waveform(m1, m2, s1z, s2z):
    return torchwave.get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=m1,
        mass2=m2,
        chi1=s1z,
        chi2=s2z,
        sample_frequencies=f_grid,
    )

hp_fast, hc_fast = fast_waveform(
    torch.tensor(30.0), torch.tensor(20.0), torch.tensor(0.2), torch.tensor(-0.1)
)

# Autograd differentiation with respect to component masses
m1_param = torch.tensor(30.0, requires_grad=True)
m2_param = torch.tensor(20.0, requires_grad=True)
hp, _ = fast_waveform(m1_param, m2_param, 0.2, -0.1)
(hp.real.pow(2) + hp.imag.pow(2)).sum().backward()
print(f"Gradients d||hp||^2/dm1 = {m1_param.grad:.4e}, d||hp||^2/dm2 = {m2_param.grad:.4e}")
```

### 2. JAX Frontend (`jaxwave`)

`jaxwave` automatically runs in FP64 precision, fully supports `@jax.jit` compilation, `jax.vmap` vectorization, and automatic differentiation (`jax.grad`):

```python
import jax
import jax.numpy as jnp
import jaxwave

# Generate waveform with JAX
hp, hc = jaxwave.get_fd_waveform(
    approximant="IMRPhenomXAS",
    mass1=30.0,
    mass2=25.0,
    chi1=0.2,
    chi2=-0.1,
    f_lower=20.0,
    f_final=512.0,
    delta_f=0.25,
)

# JIT-compiled waveform evaluation
@jax.jit
def fast_waveform(m1, m2, chi1, chi2):
    return jaxwave.get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=m1,
        mass2=m2,
        chi1=chi1,
        chi2=chi2,
        f_lower=20.0,
        f_final=512.0,
        delta_f=0.5,
    )

# Vectorized generation across batches with jax.vmap
m1_vec = jnp.array([20.0, 30.0, 40.0])
m2_vec = jnp.array([15.0, 30.0, 35.0])
chi1_vec = jnp.array([0.1, 0.2, -0.3])
chi2_vec = jnp.array([-0.1, 0.0, 0.2])

vmapped_wf = jax.vmap(fast_waveform, in_axes=(0, 0, 0, 0))
hp_vmapped, hc_vmapped = vmapped_wf(m1_vec, m2_vec, chi1_vec, chi2_vec)
print("JAX vmap shape:", hp_vmapped.shape)  # (3, 985)

# Automatic differentiation with respect to component masses
def waveform_norm(m1, m2):
    hp, hc = fast_waveform(m1, m2, 0.2, -0.1)
    return jnp.sum(jnp.abs(hp)**2)

grad_fn = jax.grad(waveform_norm, argnums=(0, 1))
dm1, dm2 = grad_fn(30.0, 25.0)
print(f"Gradients d||hp||^2/dm1 = {dm1:.4e}, d||hp||^2/dm2 = {dm2:.4e}")
```

### 3. Unified Dispatch (`diffgw`)

You can also use the top-level `diffgw` namespace to seamlessly switch between backends:

```python
import diffgw

# Dispatch to PyTorch
hp_torch, hc_torch = diffgw.get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=30.0,
    mass2=25.0,
    f_lower=20.0,
    f_final=512.0,
    delta_f=0.25,
    backend="torch",
)

# Dispatch to JAX
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

- **Detector Geometries & Networks**: Built-in ground-based (`H1`, `L1`, `V1`, `KAGRA`, `EinsteinTelescope`, `CosmicExplorer`), space-based (`LISANetwork` with TDI $A, E, T$ channels), and pulsar timing array (`PTANetwork` with Hellings-Downs correlations).
- **Noise Power Spectral Densities (PSDs)**: Analytical PSD models for advanced LIGO (Zero Detuned High Power), advanced Virgo, KAGRA, Cosmic Explorer, and Einstein Telescope.
- **Likelihood Functions**: Fast Whittle frequency-domain Gaussian likelihood, accelerated relative binning likelihood, and optimal / matched-filter SNR computation.
- **Fisher Information & Parameter Estimation**: Exact Fisher Information Matrix computation via autodiff Hessian (`jax.jacfwd` / `torch.autograd.functional.hessian`), sky localization, gradient-based MAP optimization, and No-U-Turn Sampler (NUTS) MCMC.

```python
# PyTorch Example:
from torchwave.detectors.psd import aLIGOZeroDetHighPower
from torchwave.inference.likelihood import WhittleLikelihood
from torchwave.inference.fisher import optimal_snr

# JAX Example:
from jaxwave.detectors.psd import aLIGOZeroDetHighPower as jax_aLIGO
from jaxwave.inference.likelihood import WhittleLikelihood as jax_WhittleLikelihood
from jaxwave.inference.fisher import fisher_information_matrix as jax_fisher
```

---

## Command-Line Suite

The package includes standalone console scripts for benchmarking, profiling, data management, and Kerr quasinormal mode generation:

1. **`torchwave-bench`**: Benchmark models across batch sizes, hardware platforms, and autograd backward passes.
2. **`torchwave-profile`**: Profile kernel execution, CUDA memory allocations, and cProfile hotspots.
3. **`torchwave-data`**: Inspect, verify, and download optional external ROM assets (see [external datasets guide](docs/external-data.md)).
4. **`torchwave-generate-qnm`**: Kerr QNM tables computed directly from the Leaver continued-fraction eigensolver.
5. **`torchwave-generate-london-qnm`**: London $N=61$ polynomial fits for higher-harmonic ringdown models.

---

## Tutorials & Runnable Examples

Step-by-step walkthroughs paired with runnable standalone scripts:
1. **[Tutorial 1: Quickstart & Parity Check](docs/tutorials/01_quickstart_and_parity.md)** ([`examples/01_quickstart_and_parity.py`](examples/01_quickstart_and_parity.py)): Waveform generation, batched evaluations, and double-precision PyTorch vs JAX parity verification.
2. **[Tutorial 2: Fisher Analysis](docs/tutorials/02_fisher_analysis.md)** ([`examples/02_fisher_analysis.py`](examples/02_fisher_analysis.py)): Forward-mode autodiff (`torch.autograd.forward_ad` & `jax.jvp`), noise-weighted Fisher Information Matrix, and Cramér-Rao parameter bounds.
3. **[Tutorial 3: Parameter Estimation & NUTS](docs/tutorials/03_parameter_estimation_nuts.md)** ([`examples/03_parameter_estimation_nuts.py`](examples/03_parameter_estimation_nuts.py)): Synthetic signal injection in detector noise, accelerated relative binning likelihood, Adam MAP optimization, and No-U-Turn Sampler (NUTS) MCMC.

---

## Scientific Validation and Provenance

All waveform models, phasing routines, and calibration surfaces in `diffgw` are derived directly from published peer-reviewed literature and independently verified against standard reference implementations. See:
- [**Documentation Portal**](https://xangma.github.io/diffgw/) (or local [docs/index.md](docs/index.md)) for the complete web documentation.
- [**CITATIONS.md**](CITATIONS.md) for literature bibliography.
- [**docs/usage.md**](docs/usage.md) for comprehensive API documentation and examples.
- [**docs/performance.md**](docs/performance.md) for the detailed performance analysis, figure walkthrough, and RTX 5090 scaling benchmarks.
- [**docs/differences_from_lal.md**](docs/differences_from_lal.md) for technical descriptions and comparisons with reference implementations.
- [**docs/benchmarking_protocol.md**](docs/benchmarking_protocol.md) for the controlled benchmarking methodology, Gate 1 scientific parity enforcement, and cryptographic receipt sealing.
- [**docs/provenance_ledger.md**](docs/provenance_ledger.md) for full mathematical derivation and coefficient ledger.
- [**docs/external-data.md**](docs/external-data.md) for external ROM dataset management and offline caching.
- [**CONTRIBUTING.md**](CONTRIBUTING.md) for development guidelines, testing protocols, and parity standards.

### Cross-Framework Parity Tests
To verify complete mathematical parity between PyTorch and JAX on your system:

```bash
pytest tests/parity/ -v
```

This runs 949 automated test cases covering individual mode polarizations, angles, tidal corrections, detector projections, PSDs, and inference likelihoods, verifying double-precision agreement to machine precision ($10^{-11}$ to $10^{-16}$).
