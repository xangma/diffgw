# DiffGW: Differentiable Gravitational-Wave Modeling & Inference

**DiffGW** is a high-performance, batched, autograd-differentiable gravitational-wave (GW) modeling and Bayesian parameter estimation engine. It provides two native, first-class frontends backed by a shared analytical physics core (`diffgw_core`):

- **`torchwave`**: Built natively in PyTorch for CPU, NVIDIA CUDA, and Apple Silicon MPS, featuring batched tensor computation `(B,) -> (B, N)` and autograd automatic differentiation.
- **`jaxwave`**: Built natively in JAX for CPU, NVIDIA CUDA GPU, and TPU, featuring XLA just-in-time compilation (`@jax.jit`), automatic vectorization (`jax.vmap`), and forward/reverse-mode automatic differentiation (`jax.grad`, `jax.jacfwd`).

Both packages are distributed under the permissive **MIT License** and require only PyTorch or JAX and standard scientific Python dependencies. All waveform approximants, calibration surfaces, and ringdown solvers are independently implemented from published peer-reviewed literature.

---

## Key Features

- **Extreme Scalability**: Generates up to **14.4 million waveforms per second** on an NVIDIA GeForce RTX 5090 using fused Inductor JIT kernels (`torch.compile`).
- **Full Mathematical Parity**: PyTorch (`torchwave`) and JAX (`jaxwave`) waveforms agree to double-precision numerical noise ($L_2 \le 10^{-11}$, mismatch $\le 10^{-14}$), continuously verified by 1,021 automated tests.
- **Native Autodiff**: Exact parameter gradients for gradient-based optimization (MAP), Hamiltonian Monte Carlo (HMC/NUTS), Fisher Information Matrix (FIM) calculations, and neural posterior estimation.
- **Scientific Parity**: Calibrated against LIGO-Virgo-KAGRA `LALSimulation` C implementations with documented continuous spline matching enhancements.
- **Detector Networks & Likelihoods**: Built-in detector geometry projections (`H1`, `L1`, `V1`, `K1`, `ET`, `CE`, `LISA`, `PTA`), Whittle Gaussian likelihoods, and relative binning acceleration.

---

## Waveform Models & Framework Parity

All models below are natively implemented in **both PyTorch (`torchwave`) and JAX (`jaxwave`)** with cross-framework parity:

| Family | Approximant(s) | Harmonics / Multipoles | Physical Features & Regimes |
| :--- | :--- | :---: | :--- |
| **Post-Newtonian** | `TaylorF2`<br>`TaylorF2_HM`<br>`TaylorF2_Eccentric` | $(2,2)$<br>$(2,1), (3,3), (3,2), (4,4)$<br>$(2,2)$ | 3.5PN point-particle & spin phasing, BNS tidal deformability<br>Analytical 3PN higher-order multipoles<br>Moore et al. (2016) 3PN stationary-phase eccentric corrections |
| **Aligned-Spin IMR** | `IMRPhenomD`<br>`IMRPhenomHM`<br>`IMRPhenomXAS`<br>`IMRPhenomXHM` | $(2,2)$<br>$(2,\pm 2), (2,\pm 1), (3,\pm 3), (3,\pm 2), (4,\pm 4)$<br>$(2,\pm 2)$<br>$(2,\pm 2), (2,\pm 1), (3,\pm 3), (3,\pm 2), (4,\pm 4)$ | Aligned-spin dominant mode, 5-collocation intermediate<br>Multi-harmonic PhenomD calibration with mode mixing<br>4th-generation aligned-spin dominant mode (Pratten et al. 2020)<br>4th-generation higher multipoles (García-Quirós et al. 2020) |
| **Precessing IMR** | `IMRPhenomP` / `Pv2`<br>`IMRPhenomXP`<br>`IMRPhenomXPHM` | $(2,\pm 2)$ twisted<br>$(2,\pm 2)$ twisted<br>Multipolar twisted | Generic 3D spins with NNLO PN precession angles & Wigner $d$-matrices<br>4th-gen precessing carrier with MSA analytical precession<br>Multipolar precession supporting MSA (`v223`), SpinTaylor (`v320`), and PNR (`v330`) |
| **Tidal & Matter** | `IMRPhenomD_NRTidal`<br>`IMRPhenomD_NRTidalv2`<br>`IMRPhenomNSBH` | $(2,2)$<br>$(2,2)$<br>$(2,2)$ | Dietrich et al. (2017) rational Padé BNS tidal phasing (NRTidal v1)<br>Dietrich et al. (2019) NRTidal v2 with self-spin & Planck windowing<br>Neutron star–black hole model with tidal disruption & remnant torus |
| **Effective-One-Body** | `SEOBNR_ROM` (v4/v5)<br>`SEOBNR_ODE`<br>`TEOBResumS` | $(2,2)$<br>$(2,2)$<br>$(2,2)$ | Fast B-spline reduced-order surrogate for aligned-spin SEOBNR<br>Time-domain Hamiltonian ODE integrator with continuous QNM matching<br>Damour-Nagar EOB dynamics with spin-orbit coupling and tidal potentials |
| **Ringdown & Bursts** | `KerrRingdown`<br>`SineGaussian`<br>`MemoryWaveform` | $(2,2,0), (l,m,n)$<br>Burst packet<br>$(2,0)$ | Damped sinusoids from Kerr quasinormal modes (time & frequency domain)<br>Standard unmodeled sine-Gaussian burst wavelet<br>Non-linear Christodoulou gravitational-wave memory effect $h_{20}^{\text{mem}}(t)$ |
| **Specialized Models** | `ContinuousWave`<br>`EMRIKludge`<br>`ParameterizedPostEinstein`<br>`SpinPrecessionODE` | Monochromatic<br>Multi-harmonic<br>Modified phasing<br>Orbit-averaged | Continuous waves from triaxial rotating neutron stars with Doppler modulation<br>Extreme mass ratio inspirals (LISA) with Kerr geodesic orbital frequencies<br>Model-independent testing of General Relativity ($\beta_{\rm ppE}, b_{\rm ppE}$)<br>Orbit-averaged spin precession ODE integrator |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/xangma/diffgw.git
cd diffgw

# Install with PyTorch support only
pip install -e ".[torch]"

# Install with JAX support only
pip install -e ".[jax]"

# Install all backends, inference tools, and dev dependencies
pip install -e ".[all,dev]"
```

---

## Quick Example: PyTorch (`torchwave`)

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
    device="cuda",
)

# Batched generation across 10,000 systems in a single call
m1 = torch.linspace(10.0, 50.0, 10000, device="cuda")
m2 = torch.linspace(10.0, 50.0, 10000, device="cuda")
hp_batch, hc_batch = torchwave.get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=m1,
    mass2=m2,
    f_lower=20.0,
    f_final=512.0,
    delta_f=1.0,
    device="cuda",
)
print("Batched output tensor:", hp_batch.shape)  # torch.Size([10000, 493])
```

---

## Quick Example: JAX (`jaxwave`)

```python
import jax
import jax.numpy as jnp
import jaxwave

# JIT-compile waveform generator
generator = jaxwave.WaveformGenerator(approximant="IMRPhenomD")

# Compute exact gradient of plus-polarization power with respect to chirp mass
@jax.jit
@jax.grad
def mc_loss(mc):
    hp, hc = generator(chirp_mass=mc, mass_ratio=1.0, f_lower=20.0, delta_f=0.25)
    return jnp.sum(jnp.abs(hp) ** 2)

grad = mc_loss(26.1)
print("d(Power)/d(M_c):", grad)
```

---

## Documentation Navigation

- **[Usage Guide & API Reference](usage.md)**: In-depth usage guide covering all approximants, batching, JAX JIT, detectors, and Bayesian inference.
- **[Performance & Benchmarks](performance.md)**: Complete benchmark report on NVIDIA RTX 5090 across PyTorch eager, PyTorch Inductor (`torch.compile`), JAXWave CUDA, and LALSimulation.
- **[Differences from LAL](differences_from_lal.md)**: Detailed audit of architectural improvements and numerical enhancements over LALSimulation.
- **[Benchmarking Protocol](benchmarking_protocol.md)**: Strict multi-tier benchmarking protocol, hardware specs, and reproducibility requirements.
- **[Provenance Ledger](provenance_ledger.md)**: Complete mathematical and primary literature derivation provenance ledger.
- **[External Datasets](external-data.md)**: Guide to fetching and managing external ROM and surrogate datasets.
- **[Scientific Citations](../CITATIONS.md)**: Literature bibliography for all models, phasing expressions, and calibrations.
- **[Third-Party Notices](../THIRD_PARTY_NOTICES.md)**: Dataset attributions, license terms, and provenance notices.
