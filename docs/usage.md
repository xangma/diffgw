# diffgw User Guide & API Reference (torchwave & jaxwave)

**diffgw** is a high-performance, batched, autograd-differentiable gravitational-wave (GW) modeling and Bayesian inference engine. It offers two first-class, co-equal frontends backed by a shared zero-dependency analytical physics engine (`diffgw_core`):

- **`torchwave`**: Native PyTorch implementation optimized for CPU, NVIDIA CUDA, and Apple Silicon MPS, with autograd and batched tensor computation `(B,) -> (B, N)`.
- **`jaxwave`**: Native JAX implementation with automatic FP64 precision, `@jax.jit` XLA compilation, `jax.vmap` batch vectorization, and reverse/forward-mode automatic differentiation (`jax.grad`, `jax.jacfwd`).

Both packages operate under strict clean-room principles with **zero runtime dependency on LAL**.

---

## Table of Contents

1. [Quickstart](#1-quickstart)
2. [Model Availability & Cross-Framework Parity](#2-model-availability--cross-framework-parity)
3. [Hardware Acceleration & Precision](#3-hardware-acceleration--precision)
4. [Frequency-Domain Waveforms (`get_fd_waveform`)](#4-frequency-domain-waveforms-get_fd_waveform)
   - [Basic Evaluation & Frequency Grids](#basic-evaluation--frequency-grids)
   - [Batched Waveform Generation](#batched-waveform-generation)
   - [Generic 3D Precessing Spins (MSA, SpinTaylor, PNR)](#generic-3d-precessing-spins-msa-spintaylor-pnr)
   - [Higher-Order Multipoles ($\ell, m$)](#higher-order-multipoles-ell-m)
   - [Tidal Deformability for Neutron Stars](#tidal-deformability-for-neutron-stars)
   - [Orbital Eccentricity](#orbital-eccentricity)
5. [JAX Frontend & Acceleration (`jaxwave`)](#5-jax-frontend--acceleration-jaxwave)
   - [Waveform Generation & `WaveformGenerator`](#waveform-generation--waveformgenerator)
   - [Just-In-Time Compilation (`@jax.jit`)](#just-in-time-compilation-jaxjit)
   - [Vectorized Batch Evaluation (`jax.vmap`)](#vectorized-batch-evaluation-jaxvmap)
   - [Autodiff Parameter Gradients (`jax.grad`)](#autodiff-parameter-gradients-jaxgrad)
   - [Unified Dispatch via `diffgw`](#unified-dispatch-via-diffgw)
6. [Time-Domain Waveforms (`get_td_waveform`)](#6-time-domain-waveforms-get_td_waveform)
7. [Black Hole Ringdown & Kerr Quasinormal Modes](#7-black-hole-ringdown--kerr-quasinormal-modes)
   - [Ringdown Waveforms (`get_ringdown_waveform`)](#ringdown-waveforms-get_ringdown_waveform)
   - [Programmatic Kerr QNM Eigensolver (`torchwave.qnm.solve`)](#programmatic-kerr-qnm-eigensolver-torchwaveqnmsolve)
8. [Specialized Astrophysical Models](#8-specialized-astrophysical-models)
   - [Extreme-Mass-Ratio Inspirals (`EMRIKludge`)](#extreme-mass-ratio-inspirals-emrikludge)
   - [Parameterized Post-Einsteinian & Tests of GR (`ParameterizedPostEinstein`)](#parameterized-post-einsteinian--tests-of-gr-parameterizedposteinstein)
   - [Orbit-Averaged Spin Precession ODE](#orbit-averaged-spin-precession-ode)
9. [Automatic Differentiation in PyTorch](#9-automatic-differentiation-in-pytorch)
10. [Detector Networks & Projections](#10-detector-networks--projections)
    - [Ground-Based Networks (`H1`, `L1`, `V1`, `K1`, `ET`, `CE`)](#ground-based-networks-h1-l1-v1-k1-et-ce)
    - [Power Spectral Densities (PSDs)](#power-spectral-densities-psds)
    - [Space-Based Interferometers (`LISANetwork`)](#space-based-interferometers-lisanetwork)
    - [Pulsar Timing Arrays (`PTANetwork`)](#pulsar-timing-arrays-ptanetwork)
11. [Bayesian Inference Suite](#11-bayesian-inference-suite)
    - [Signal-to-Noise Ratio (SNR)](#signal-to-noise-ratio-snr)
    - [Whittle Gaussian Likelihood](#whittle-gaussian-likelihood)
    - [Accelerated Relative Binning Likelihood](#accelerated-relative-binning-likelihood)
    - [Fisher Information Matrix & Sky Localization](#fisher-information-matrix--sky-localization)
    - [Gradient-Based MAP Optimization](#gradient-based-map-optimization)
    - [No-U-Turn Sampler (NUTS) MCMC](#no-u-turn-sampler-nuts-mcmc)
12. [Command-Line Tools](#12-command-line-tools)
    - [`torchwave-bench`](#torchwave-bench)
    - [`torchwave-profile`](#torchwave-profile)
    - [`torchwave-data`](#torchwave-data)
    - [`torchwave-generate-qnm`](#torchwave-generate-qnm)
    - [`torchwave-generate-london-qnm`](#torchwave-generate-london-qnm)

---

## 1. Quickstart

### PyTorch Quickstart (`torchwave`)
```python
import torch
import torchwave

# Generate a binary black hole waveform (IMRPhenomD)
hp, hc = torchwave.get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=30.0,            # Solar masses (m1 >= m2)
    mass2=25.0,            # Solar masses
    chi1=0.2,              # Dimensionless aligned spin of body 1 (-1 <= chi1 <= 1)
    chi2=-0.1,             # Dimensionless aligned spin of body 2 (-1 <= chi2 <= 1)
    distance=410.0,        # Luminosity distance in Mpc
    inclination=0.5,       # Inclination angle in radians
    f_lower=20.0,          # Starting frequency in Hz
    f_final=512.0,         # Ending frequency in Hz
    delta_f=0.25,          # Frequency step in Hz
)

print("PyTorch Polarization shape:", hp.shape)  # torch.Size([1969])
print("Data type:", hp.dtype)                   # torch.complex128
```

### JAX Quickstart (`jaxwave`)
```python
import jax
import jax.numpy as jnp
import jaxwave

# Generate waveform with JAX (runs automatically in FP64 precision)
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

print("JAX Polarization shape:", hp.shape)  # (1969,)
print("Data type:", hp.dtype)               # complex128
```

---

## 2. Model Availability & Cross-Framework Parity

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

## 3. Hardware Acceleration & Precision

Both frontends support hardware acceleration across platforms:
- **PyTorch (`torchwave`)**: CPU, NVIDIA CUDA, and Apple Silicon MPS.
- **JAX (`jaxwave`)**: CPU, NVIDIA CUDA GPU, and Google Cloud TPU.

### PyTorch Device & Precision
```python
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

hp_f64, hc_f64 = torchwave.get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=30.0, mass2=25.0,
    f_lower=20.0, f_final=512.0, delta_f=0.25,
    device=device,
    dtype=torch.float64,
)
```

---

## 4. Frequency-Domain Waveforms (`get_fd_waveform`)

### Basic Evaluation & Frequency Grids
Specify frequency bounds uniformly via `(f_lower, f_final, delta_f)` or provide arbitrary non-uniform evaluation points with `sample_frequencies`:

```python
# Non-uniform frequency grid
frequencies = torch.logspace(1.3, 3.0, 1000, dtype=torch.float64)

hp, hc = torchwave.get_fd_waveform(
    approximant="IMRPhenomXAS",
    mass1=36.0,
    mass2=29.0,
    chi1=0.32,
    chi2=-0.15,
    sample_frequencies=frequencies,
)
```

### Batched Waveform Generation
`torchwave` accepts 1D tensors of shape `(B,)` for astrophysical parameters to compute $B$ waveforms simultaneously:

```python
m1 = torch.tensor([20.0, 30.0, 45.0, 60.0])
m2 = torch.tensor([15.0, 25.0, 30.0, 40.0])
chi1 = torch.tensor([0.0, 0.2, -0.4, 0.8])
chi2 = torch.tensor([0.0, -0.1, 0.1, -0.5])

hp_batch, hc_batch = torchwave.get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=m1, mass2=m2, chi1=chi1, chi2=chi2,
    f_lower=20.0, f_final=512.0, delta_f=1.0,
)
print("Batched tensor shape:", hp_batch.shape)  # torch.Size([4, 493])
```

### Generic 3D Precessing Spins (MSA, SpinTaylor, PNR)
To simulate binary systems with orbital precession, supply 3D spin vectors (`spin1x`, `spin1y`, `spin1z`, `spin2x`, `spin2y`, `spin2z`). `torchwave` provides three state-of-the-art precession implementations under `IMRPhenomXPHM`:

1. **Analytical Multi-Scale Analysis (`msa`, default)**:
   ```python
   hp_msa, hc_msa = torchwave.get_fd_waveform(
       approximant="IMRPhenomXPHM",
       mass1=40.0, mass2=25.0,
       spin1x=0.3, spin1y=0.1, spin1z=0.2,
       spin2x=0.0, spin2y=-0.2, spin2z=-0.1,
       f_lower=20.0, f_final=512.0, delta_f=0.5,
   )
   ```

2. **Numerical SpinTaylor Precession Integration (`spintaylor`)**:
   ```python
   hp_st, hc_st = torchwave.get_fd_waveform(
       approximant="IMRPhenomXPHM-SpinTaylor",  # or precession_method="spintaylor"
       mass1=40.0, mass2=25.0,
       spin1x=0.3, spin1y=0.1, spin1z=0.2,
       spin2x=0.0, spin2y=-0.2, spin2z=-0.1,
       f_lower=20.0, f_final=512.0, delta_f=0.5,
   )
   ```

3. **Numerical Relativity Informed Precession (`pnr`)**:
   ```python
   hp_pnr, hc_pnr = torchwave.get_fd_waveform(
       approximant="IMRPhenomXPNR",  # or precession_method="pnr"
       mass1=40.0, mass2=25.0,
       spin1x=0.3, spin1y=0.1, spin1z=0.2,
       spin2x=0.0, spin2y=-0.2, spin2z=-0.1,
       f_lower=20.0, f_final=512.0, delta_f=0.5,
   )
   ```

### Higher-Order Multipoles ($\ell, m$)
Use the `modes` argument to select specific harmonic modes:

```python
hp_hm, hc_hm = torchwave.get_fd_waveform(
    approximant="IMRPhenomXPHM",
    mass1=50.0, mass2=15.0,
    spin1z=0.4, spin2z=0.1,
    inclination=1.0,
    modes=[(2, 2), (2, 1), (3, 3), (4, 4)],
    f_lower=20.0, f_final=512.0, delta_f=0.5,
)
```

### Tidal Deformability for Neutron Stars
Pass dimensionless tidal deformabilities `lambda1` ($\Lambda_1$) and `lambda2` ($\Lambda_2$):

```python
hp_bns, hc_bns = torchwave.get_fd_waveform(
    approximant="IMRPhenomD_NRTidalv2",
    mass1=1.4, mass2=1.3,
    chi1=0.02, chi2=-0.01,
    lambda1=450.0, lambda2=520.0,
    f_lower=30.0, f_final=1024.0, delta_f=0.5,
)
```

### Orbital Eccentricity
For eccentric inspirals, use `TaylorF2_Eccentric` with `eccentricity`:

```python
hp_ecc, hc_ecc = torchwave.get_fd_waveform(
    approximant="TaylorF2_Eccentric",
    mass1=15.0, mass2=10.0,
    eccentricity=0.15,
    f_lower=20.0, f_final=400.0, delta_f=0.25,
)
```

---

## 5. JAX Frontend & Acceleration (`jaxwave`)

`jaxwave` provides a pure JAX implementation designed for high-performance scientific computing and machine learning workflows.

### Waveform Generation & `WaveformGenerator`
Generate waveforms using the direct functional interface or preconfigured generator:

```python
import jaxwave

# Direct functional interface
hp, hc = jaxwave.get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=30.0, mass2=25.0,
    chi1=0.2, chi2=-0.1,
    f_lower=20.0, f_final=512.0, delta_f=0.25,
)

# Class-based generator for repeated evaluations
gen = jaxwave.WaveformGenerator(
    approximant="IMRPhenomXAS",
    f_lower=20.0, f_final=512.0, delta_f=0.25,
)
hp, hc = gen.generate_fd_waveform(mass1=30.0, mass2=25.0, chi1=0.1, chi2=0.0)
```

### Just-In-Time Compilation (`@jax.jit`)
Every `jaxwave` approximant is designed to be fully traceable and compilable via XLA:

```python
import jax
import jaxwave

@jax.jit
def fast_waveform(m1, m2, s1z, s2z):
    return jaxwave.get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=m1, mass2=m2, chi1=s1z, chi2=s2z,
        f_lower=20.0, f_final=512.0, delta_f=0.5,
    )

# First call triggers XLA compilation; subsequent calls execute in microseconds on GPU
hp, hc = fast_waveform(35.0, 25.0, 0.2, -0.1)
```

### Vectorized Batch Evaluation (`jax.vmap`)
Map waveform generation across parameter grids without Python loops:

```python
import jax
import jax.numpy as jnp
import jaxwave

m1_grid = jnp.array([20.0, 30.0, 40.0])
m2_grid = jnp.array([15.0, 25.0, 35.0])
s1_grid = jnp.array([0.1, 0.2, -0.3])
s2_grid = jnp.array([-0.1, 0.0, 0.2])

vmapped_fn = jax.vmap(fast_waveform, in_axes=(0, 0, 0, 0))
hp_batch, hc_batch = vmapped_fn(m1_grid, m2_grid, s1_grid, s2_grid)
print("JAX batched shape:", hp_batch.shape)  # (3, 985)
```

### Autodiff Parameter Gradients (`jax.grad`)
Compute exact mathematical derivatives $\partial h / \partial \theta$:

```python
import jax
import jax.numpy as jnp

def waveform_power(m1, m2):
    hp, hc = fast_waveform(m1, m2, 0.2, -0.1)
    return jnp.sum(jnp.abs(hp) ** 2)

# Compute gradient vector [dPower/dm1, dPower/dm2]
grad_fn = jax.grad(waveform_power, argnums=(0, 1))
dm1, dm2 = grad_fn(30.0, 25.0)
print(f"Gradients: dm1={dm1:.4e}, dm2={dm2:.4e}")
```

### Unified Dispatch via `diffgw`
Use `diffgw.get_fd_waveform` to select backends dynamically:

```python
import diffgw

hp_t, hc_t = diffgw.get_fd_waveform("IMRPhenomD", mass1=30.0, mass2=25.0, backend="torch")
hp_j, hc_j = diffgw.get_fd_waveform("IMRPhenomD", mass1=30.0, mass2=25.0, backend="jax")
```

---

## 6. Time-Domain Waveforms (`get_td_waveform`)

`torchwave.get_td_waveform` generates time-domain polarizations obeying Parseval's energy conservation theorem:

```python
hp_td, hc_td, times = torchwave.get_td_waveform(
    approximant="IMRPhenomD",
    mass1=30.0, mass2=25.0,
    chi1=0.2, chi2=-0.1,
    delta_t=1.0 / 2048.0,
    f_lower=20.0,
    return_times=True,
)
print("Time series duration:", times[-1].item() - times[0].item(), "seconds")
```

---

## 7. Black Hole Ringdown & Kerr Quasinormal Modes

### Ringdown Waveforms (`get_ringdown_waveform`)
Generate multi-mode damped sinusoid ringdowns:

```python
hp_rd, hc_rd = torchwave.get_ringdown_waveform(
    final_mass=65.0,
    final_spin=0.68,
    amplitudes={(2, 2, 0): 1.0, (3, 3, 0): 0.2},
    phases={(2, 2, 0): 0.0, (3, 3, 0): 0.5},
    delta_t=1.0 / 4096.0,
    duration=0.1,
)
```

### Programmatic Kerr QNM Eigensolver (`torchwave.qnm.solve`)
```python
from torchwave.qnm import solve

root = solve(a=0.7, ell=2, m=2, n=0)
print(f"Complex frequency M*omega: {root.omega:.6f}")
print(f"Real frequency:             {root.omega.real:.6f}")
print(f"Damping rate:               {root.omega.imag:.6f}")
```

---

## 8. Specialized Astrophysical Models

- **`EMRIKludge`**: Relativistic extreme-mass-ratio inspirals using Kerr geodesic orbital frequencies.
- **`ParameterizedPostEinstein` (`PPE`)**: Testing general relativity with parameterized PN phase modifications.
- **`SpinPrecessionODE`**: Orbit-averaged spin-precession trajectory integrator.

---

## 9. Automatic Differentiation in PyTorch

```python
import torch
import torchwave

m1 = torch.tensor(35.0, requires_grad=True, dtype=torch.float64)
m2 = torch.tensor(25.0, requires_grad=True, dtype=torch.float64)

hp, _ = torchwave.get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=m1, mass2=m2,
    f_lower=20.0, f_final=256.0, delta_f=1.0,
)
loss = torch.sum(torch.abs(hp) ** 2)
loss.backward()

print("d(loss)/dm1:", m1.grad.item())
print("d(loss)/dm2:", m2.grad.item())
```

---

## 10. Detector Networks & Projections

### Ground-Based Networks (`H1`, `L1`, `V1`, `K1`, `ET`, `CE`)
```python
from torchwave.detectors import DetectorNetwork

network = DetectorNetwork(["H1", "L1", "V1"])
f_grid = torch.arange(20.0, 512.0 + 0.125, 0.25, dtype=torch.float64)
hp, hc = torchwave.get_fd_waveform(approximant="IMRPhenomD", mass1=30.0, mass2=25.0, sample_frequencies=f_grid)

strains = network.project_fd_strain(
    f=f_grid, h_plus=hp, h_cross=hc,
    ra=1.5, dec=0.3, psi=0.8, t_gps=1126259462.0,
)
```

### Power Spectral Densities (PSDs)
Available in both `torchwave.detectors.psd` and `jaxwave.detectors.psd`:
- `aLIGOZeroDetHighPower`
- `aLIGOEarlyHighPower`
- `AdvancedVirgo`
- `KAGRA`
- `EinsteinTelescope`
- `CosmicExplorer`

### Space-Based Interferometers (`LISANetwork`)
TDI channels ($A, E, T$) with galactic confusion noise (`torchwave.detectors.LISANetwork` and `jaxwave.detectors.LISANetwork`).

### Pulsar Timing Arrays (`PTANetwork`)
Hellings-Downs correlation covariance matrix (`torchwave.detectors.PTANetwork` and `jaxwave.detectors.PTANetwork`).

---

## 11. Bayesian Inference Suite

Both `torchwave.inference` and `jaxwave.inference` provide high-performance parameter estimation tools.

### Signal-to-Noise Ratio (SNR)

Compute optimal SNR $\rho_{\text{opt}} = \sqrt{\langle h \mid h \rangle}$ across single detectors or multi-detector networks:

```python
# PyTorch
from torchwave.inference import optimal_snr, matched_filter_snr
from torchwave.detectors.psd import aLIGOZeroDetHighPower

psd = aLIGOZeroDetHighPower(f_grid)
snr_val = optimal_snr(hp, psd, df=0.25)
print(f"PyTorch Optimal SNR: {snr_val.item():.2f}")

# JAX
from jaxwave.inference.likelihood import inner_product
from jaxwave.detectors.psd import aLIGOZeroDetHighPower as jax_psd

snr_jax = jnp.sqrt(inner_product(hp_jax, hp_jax, jax_psd(f_grid), df=0.25))
print(f"JAX Optimal SNR: {snr_jax:.2f}")
```

### Whittle Gaussian Likelihood

Evaluate the exact frequency-domain Gaussian noise log-likelihood:

$$\ln \mathcal{L}(\theta) = -\frac{1}{2} \sum_{k} \langle d_k - h_k(\theta) \mid d_k - h_k(\theta) \rangle$$

```python
# PyTorch Whittle Likelihood
from torchwave.inference import WhittleLikelihood

whittle_pt = WhittleLikelihood(data=strains, psds=psds, df=0.25)
log_lik_pt = whittle_pt(strains)
log_lik_marg = whittle_pt.phase_marginalized_log_likelihood(strains)

# JAX Whittle Likelihood
from jaxwave.inference.likelihood import WhittleLikelihood as JAXWhittleLikelihood

whittle_jx = JAXWhittleLikelihood(data=strains_jax, psd=psd_jax, df=0.25)
log_lik_jx = whittle_jx(template_jax)
```

### Accelerated Relative Binning Likelihood

For fast parameter estimation without evaluating tens of thousands of dense frequency points, both frontends implement **Relative Binning** (Zackay et al. 2018). The data is compressed into summary coefficients across $\sim 30\text{--}50$ bins, evaluating candidate templates only at bin centers:

```python
# PyTorch Relative Binning
from torchwave.inference import RelativeBinningLikelihood

rel_bin = RelativeBinningLikelihood(
    data=strains,
    psds=psds,
    h0=strains,             # Fiducial reference waveform
    frequencies=f_grid,
    n_bins=40,              # Number of adaptive frequency bins
    df=0.25,
)

# During MCMC sampling, evaluate candidate templates ONLY at bin centers (e.g. 40 points)
hp_bin, hc_bin = torchwave.get_fd_waveform(
    approximant="IMRPhenomD",
    mass1=30.05, mass2=24.95,
    sample_frequencies=rel_bin.bin_centers,
)
template_bins = network.project_fd_strain(
    f=rel_bin.bin_centers,
    h_plus=hp_bin, h_cross=hc_bin,
    ra=1.5, dec=0.3, psi=0.8, t_gps=1126259462.0,
)
fast_log_lik = rel_bin(template_bins)

# JAX Relative Binning
from jaxwave.inference.relative_binning import setup_relative_binning, relative_binning_log_likelihood

rb_setup = setup_relative_binning(f_grid, strains_jax["H1"], h0_jax, psd_jax, n_bins=40)
fast_ll_jax = relative_binning_log_likelihood(template_bin_jax, rb_setup)
```

### Fisher Information Matrix & Sky Localization

Compute parameter uncertainties and covariances using automatic differentiation:

$$\Gamma_{ij} = \sum_k 4 \, \text{Re} \int_{f_{\text{low}}}^{f_{\text{high}}} \frac{1}{S_n^k(f)} \frac{\partial h_k(f)}{\partial \theta_i} \left(\frac{\partial h_k(f)}{\partial \theta_j}\right)^* df$$

```python
# PyTorch Fisher Matrix
from torchwave.inference import fisher_information_matrix, covariance_matrix

params = {"mass1": 30.0, "mass2": 25.0}

def waveform_model(p):
    hp_m, _ = torchwave.get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=p["mass1"], mass2=p["mass2"],
        sample_frequencies=f_grid,
    )
    return hp_m

gamma, param_names = fisher_information_matrix(
    parameters_dict=params,
    waveform_fn=waveform_model,
    psd_dict=psds["H1"],
    frequencies=f_grid,
    df=0.25,
)
cov = covariance_matrix(gamma)
print(f"PyTorch sigma(mass1): {torch.sqrt(cov[0, 0]).item():.4f} Msun")

# JAX Fisher Matrix
from jaxwave.inference.fisher import fisher_information_matrix as jax_fisher

def jax_wf_fn(theta):
    # theta = [mass1, mass2]
    hp_m, _ = jaxwave.get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=theta[0], mass2=theta[1],
        sample_frequencies=f_grid,
    )
    return hp_m

gamma_jax = jax_fisher(jnp.array([30.0, 25.0]), jax_wf_fn, psd_jax, f_grid, df=0.25)
cov_jax = jnp.linalg.inv(gamma_jax)
print(f"JAX sigma(mass1): {jnp.sqrt(cov_jax[0, 0]):.4f} Msun")
```

### Gradient-Based MAP Optimization

Quickly find the Maximum A Posteriori (MAP) parameter estimate using Adam optimization directly on the log-likelihood gradient:

```python
from torchwave.inference import optimize_map

best_params, loss_history = optimize_map(
    data=strains["H1"],
    psd=psds["H1"],
    initial_params={"mass1": 31.0, "mass2": 24.0},
    waveform_fn=lambda p: torchwave.get_fd_waveform(
        "IMRPhenomD", mass1=p["mass1"], mass2=p["mass2"], sample_frequencies=f_grid
    )[0],
    frequencies=f_grid,
    steps=100,
    lr=0.05,
)
```

### No-U-Turn Sampler (NUTS) MCMC

Run Hamiltonian Monte Carlo with dual-averaging step-size adaptation and Gelman-Rubin convergence diagnostics:

```python
from torchwave.inference import NUTS, gelman_rubin

def log_posterior(theta):
    hp_cand, _ = torchwave.get_fd_waveform(
        "IMRPhenomD", mass1=theta[0], mass2=theta[1], sample_frequencies=f_grid
    )
    return whittle_pt({"H1": hp_cand, "L1": hp_cand, "V1": hp_cand})

sampler = NUTS(log_prob_fn=log_posterior, target_accept=0.65)
initial_positions = torch.tensor([[30.1, 24.9], [29.8, 25.2]])
chains = sampler.sample_parallel(initial_positions, n_samples=300, n_warmup=150)
r_hat = gelman_rubin(chains)
print("Gelman-Rubin R_hat:", r_hat)
```

---

## 12. Command-Line Tools

TorchWave registers five standalone console executables:

### `torchwave-bench`
Benchmark model throughput, batch sizes, and GPU speedups:
```bash
# Benchmark standard models across batch sizes
torchwave-bench --models popular --batch-sizes 1,10,100,1000

# Compare against LALSimulation baseline and export Markdown table
torchwave-bench --models TaylorF2,IMRPhenomD,IMRPhenomXAS --compare-lal --markdown-out benchmark.md

# Benchmark with autograd backward pass
torchwave-bench --models IMRPhenomD --batch-sizes 100 --backward
```

### `torchwave-profile`
Profile operator kernels, execution bottlenecks, and memory allocation:
```bash
# Run comprehensive profiling (cProfile, torch profiler, memory)
torchwave-profile --model IMRPhenomD --mode all --top-k 15

# Export Chrome trace JSON for visualization in chrome://tracing
torchwave-profile --model TaylorF2 --trace-out trace_f2.json
```

### `torchwave-data`
Manage optional external ROM datasets outside git repositories and wheels:
```bash
# List available external datasets
torchwave-data list

# Inspect dataset metadata
torchwave-data info seobnrv4-rom-v3

# Fetch dataset to user cache or custom directory
torchwave-data fetch seobnrv4-rom-v3

# Verify pinned SHA-256 integrity
torchwave-data verify seobnrv4-rom-v3
```

### `torchwave-generate-qnm`
Generate clean-room Kerr quasinormal mode tables directly from the Leaver continued-fraction eigenproblem:
```bash
torchwave-generate-qnm --output-dir ./qnm_data --modes 2,2,0
```

### `torchwave-generate-london-qnm`
Recompute the clean-room London $N=61$ quasinormal mode polynomial fits:
```bash
torchwave-generate-london-qnm --output-dir ./london_qnm
```

