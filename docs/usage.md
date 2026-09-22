# Waveform and analysis user guide

**diffgw** provides batched gravitational-wave models and inference tools through two native frontends, sharing constants and coefficient tables in the framework-independent `diffgw_core` package:

- **`torchwave`**: PyTorch tensor evaluation, batching `(B,) -> (B, N)`, and autograd on supported model paths.
- **`jaxwave`**: Native JAX implementation with automatic FP64 precision, `@jax.jit` XLA compilation, `jax.vmap` batch vectorization, and reverse/forward-mode automatic differentiation (`jax.grad`, `jax.jacfwd`).

The original software is **MIT licensed**. Framework dependencies, optional external assets, and attributed adaptations are described in the [installation instructions](index.md#installation), [data guide](external-data.md), and [third-party notices](../THIRD_PARTY_NOTICES.md).

---

## Table of Contents

1. [Quickstart](#1-quickstart)
2. [Model Availability & Cross-Framework Parity](#2-model-availability-cross-framework-parity)
3. [Hardware Acceleration & Precision](#3-hardware-acceleration-precision)
4. [Frequency-Domain Waveforms (`get_fd_waveform`)](#4-frequency-domain-waveforms-get_fd_waveform)
   - [Basic Evaluation & Frequency Grids](#basic-evaluation-frequency-grids)
   - [Batched Waveform Generation](#batched-waveform-generation)
   - [Generic 3D Precessing Spins (MSA, SpinTaylor, PNR)](#generic-3d-precessing-spins-msa-spintaylor-pnr)
   - [Higher-Order Multipoles ($\ell, m$)](#higher-order-multipoles-ell-m)
   - [Tidal Deformability for Neutron Stars](#tidal-deformability-for-neutron-stars)
   - [Orbital Eccentricity](#orbital-eccentricity)
5. [JAX Frontend & Acceleration (`jaxwave`)](#5-jax-frontend-acceleration-jaxwave)
   - [Waveform Generation & `WaveformGenerator`](#waveform-generation-waveformgenerator)
   - [Just-In-Time Compilation (`@jax.jit`)](#just-in-time-compilation-jaxjit)
   - [Vectorized Batch Evaluation (`jax.vmap`)](#vectorized-batch-evaluation-jaxvmap)
   - [Autodiff Parameter Gradients (`jax.grad`)](#autodiff-parameter-gradients-jaxgrad)
   - [Shape Bucketing for MCMC Sampling (`BucketedWaveformEvaluator`)](#shape-bucketing-for-mcmc-sampling-bucketedwaveformevaluator)
   - [Unified Dispatch via `diffgw`](#unified-dispatch-via-diffgw)
6. [Time-Domain Waveforms (`get_td_waveform`)](#6-time-domain-waveforms-get_td_waveform)
7. [Black Hole Ringdown & Kerr Quasinormal Modes](#7-black-hole-ringdown-kerr-quasinormal-modes)
   - [Ringdown Waveforms (`get_ringdown_waveform`)](#ringdown-waveforms-get_ringdown_waveform)
   - [Programmatic Kerr QNM Eigensolver (`torchwave.qnm.solve`)](#programmatic-kerr-qnm-eigensolver-torchwaveqnmsolve)
8. [Specialized Astrophysical Models](#8-specialized-astrophysical-models)
9. [Automatic Differentiation in PyTorch](#9-automatic-differentiation-in-pytorch)
10. [Detector Networks & Projections](#10-detector-networks-projections)
    - [Ground-Based Networks (`H1`, `L1`, `V1`, `K1`, `ET`, `CE`)](#ground-based-networks-h1-l1-v1-k1-et-ce)
    - [Power Spectral Densities (PSDs)](#power-spectral-densities-psds)
    - [Space-Based Interferometers (`LISANetwork`)](#space-based-interferometers-lisanetwork)
    - [Pulsar Timing Arrays (`PTANetwork`)](#pulsar-timing-arrays-ptanetwork)
11. [Bayesian Inference Suite](#11-bayesian-inference-suite)
    - [Signal-to-Noise Ratio (SNR)](#signal-to-noise-ratio-snr)
    - [Whittle Gaussian Likelihood](#whittle-gaussian-likelihood)
    - [Accelerated Relative Binning Likelihood](#accelerated-relative-binning-likelihood)
    - [Fisher Information Matrix & Sky Localization](#fisher-information-matrix-sky-localization)
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

Use the [model support catalog](models.md) to compare physical features,
backend and derivative restrictions, external data requirements, and validation
status. [Accuracy and validation](validation.md) distinguishes frontend
consistency, external-reference agreement and independent physical evidence.

---

## 3. Hardware Acceleration & Precision

The [published FP64 benchmarks](package_benchmarks.md) cover CPU and NVIDIA CUDA for both frontends. Other framework targets are not established by those results. PyTorch MPS does not support FP64; use CPU for the FP64 examples on Apple Silicon. Model operations and available precision determine device compatibility.

### PyTorch Device & Precision
```python
device = "cuda" if torch.cuda.is_available() else "cpu"

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
To simulate binary systems with orbital precession, supply 3D spin vectors (`spin1x`, `spin1y`, `spin1z`, `spin2x`, `spin2y`, `spin2z`). `torchwave` provides three precession prescriptions under `IMRPhenomXPHM`:

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

`jaxwave` provides tensor waveform models for JAX scientific computing and machine learning workflows. Transformation and device support depend on the selected path; for example, the Kerr angular eigensolver uses a host CPU callback.

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

# Class-based generator for repeated evaluations (supports automatic JIT caching)
gen = jaxwave.WaveformGenerator(
    approximant="IMRPhenomXAS",
    f_lower=20.0, f_final=512.0, delta_f=0.25,
    use_jit=True,
)
hp, hc = gen.generate_fd_waveform(mass1=30.0, mass2=25.0, chi1=0.1, chi2=0.0)
```

### Just-In-Time Compilation (`@jax.jit`)
The following IMRPhenomD path supports JAX tracing and XLA compilation. Transformation support is model- and option-dependent; external-data callbacks and numerical solvers require separate checks:

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

# First call compiles; compatible subsequent calls reuse the compiled function
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
Compute derivatives $\partial h / \partial \theta$ of the implemented numerical model using automatic differentiation. Piecewise branches, interpolation, solver settings, and supported derivative orders still matter:

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

### Shape Bucketing for MCMC Sampling (`BucketedWaveformEvaluator`)

In MCMC sampling, parallel tempering, or ensemble walkers where batch sizes vary dynamically, standard `@jax.jit` functions recompile whenever a new batch size is encountered.

`jaxwave.inference.BucketedWaveformEvaluator` reduces the number of distinct batch shapes by padding inputs to discrete buckets, caching each bucket evaluator, and unpadding its outputs. A new bucket compiles on first use; changes to dtype or other traced shapes may also trigger compilation:

```python
from jaxwave.inference import BucketedWaveformEvaluator
import jax.numpy as jnp

evaluator = BucketedWaveformEvaluator(
    approximant="IMRPhenomD",
    f_lower=20.0, f_final=512.0, delta_f=0.5,
    buckets=(1, 4, 8, 16, 32, 64, 128),
)

# Evaluating 13 walkers pads to bucket 16, using the cached JIT kernel
m1_walkers = jnp.array([30.0] * 13)
m2_walkers = jnp.array([25.0] * 13)
hp, hc = evaluator.evaluate(mass1=m1_walkers, mass2=m2_walkers)
print(f"Output walkers: {hp.shape[0]}")  # 13
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

`get_td_waveform` is available in both frontends. Frequency-domain models use
a conditioned inverse Fourier transform:

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

### Native IMRPhenomT

See the [IMRPhenomT guide](imrphenomt.md) for arbitrary grids, differentiation,
JAX compilation, and validation details.

```python
from torchwave import get_td_waveform  # or: from jaxwave import get_td_waveform

hp, hc, times = get_td_waveform(
    "IMRPhenomT", mass1=30.0, mass2=25.0,
    chi1=0.2, chi2=-0.1, distance=100.0, inclination=0.5,
    f_lower=20.0, f_ref=20.0, delta_t=1 / 2048,
    duration=2.0, return_times=True,
)
```

This evaluates the native aligned-spin dominant `(2,±2)` model, including
inspiral, merger and ringdown. `IMRPhenomTHM` adds higher harmonics;
`IMRPhenomTP` and `IMRPhenomTPHM` add NNLO, MSA, or numerical precession. See the guide for
mode selection and Cartesian spin examples. Masses are in solar masses, distance in
Mpc, times in seconds and frequencies in Hz. `phic` is the orbital reference
phase at `f_ref`; omitted or zero `f_ref` uses `f_lower`.

With `center_coalescence=True`, the returned time grid ends near `t_pad`
and the model's amplitude peak is at `tc`. Samples before the intrinsic
frequency reaches `f_lower` are zero. A supplied `duration` can truncate the
inspiral; otherwise it is estimated from the model's frequency evolution.
With `center_coalescence=False`, the grid starts at zero and `tc` sets the
peak location on that grid. There is no periodic wrapping or Fourier taper.

For an arbitrary fixed time grid, call
`IMRPhenomT().time_domain(times, mass1, mass2, chi1=..., chi2=..., f_ref=...)`
from either frontend. This lower-level interface does not apply an `f_lower`
mask. Parameter batches broadcast with the time dimension last. Fixed grids
support first-order automatic differentiation; the JAX model supports `jit`
and `vmap`. The frequency inversion uses an implicit first derivative;
higher-order derivatives through that inversion are not supported.
Automatic duration selection is an eager operation because it changes array
shape. Use a fixed duration inside JAX transformations.

`get_fd_waveform` raises an error for every T-family model. Nonzero tidal
deformabilities, `f_final`, and unrecognized options are rejected. Only TP
and TPHM accept transverse spins. They offer `precession_method="nnlo"`
(the default), `"msa"`, or `"numerical"`; see the [T-family guide](imrphenomt.md#higher-modes-and-precession)
for versions and domain restrictions.

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

Evaluate the discrete Whittle log-likelihood for stationary Gaussian noise with a fixed PSD, up to a parameter-independent normalization:

$$\ln \mathcal{L}(\theta) = -\frac{1}{2} \sum_{k} \langle d_k - h_k(\theta) \mid d_k - h_k(\theta) \rangle$$

```python
# PyTorch Whittle Likelihood
from torchwave.inference import WhittleLikelihood

whittle_pt = WhittleLikelihood(data=strains, psds=psds, df=0.25)
log_lik_pt = whittle_pt(strains)
log_lik_marg = whittle_pt.phase_marginalized_log_likelihood(strains)

# JAX Whittle Likelihood
from jaxwave.inference.likelihood import WhittleLikelihood as JAXWhittleLikelihood

whittle_jx = JAXWhittleLikelihood(data=strains_jax, psds=psds_jax, df=0.25)
log_lik_jx = whittle_jx(template_jax)
```

### Accelerated Relative Binning Likelihood

Both frontends implement a **relative-binning approximation** using precomputed summary coefficients and candidate waveforms at bin centers. The ratio to a suitable fiducial waveform should vary slowly within each bin. Choose the bin count and validate likelihood and gradient errors against dense evaluation over the relevant parameter region. Runtime depends on the workload; see [validation evidence](validation/inference.md#relative-binning) and the [method paper](https://arxiv.org/abs/1806.08792).

```python
# PyTorch Relative Binning
from torchwave.inference import RelativeBinningLikelihood

rel_bin = RelativeBinningLikelihood(
    data=strains,
    psds=psds,
    h0=strains,             # Fiducial reference waveform
    frequencies=f_grid,
    n_bins=40,              # Chosen bin count; validate for the intended parameter region
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
from jaxwave.inference.relative_binning import RelativeBinningLikelihood as JAXRelativeBinningLikelihood

# Supply matching detector dictionaries and a fiducial waveform on the dense grid.
rb_jax = JAXRelativeBinningLikelihood(
    data=strains_jax, psds=psds_jax, h0=h0_jax,
    frequencies=f_grid_jax, n_bins=40, df=0.25,
)
# Evaluate candidate detector templates at rb_jax.bin_centers.
fast_ll_jax = rb_jax(template_bins_jax)
```

### Fisher Information Matrix & Sky Localization

Compute local Fisher uncertainty forecasts using automatic differentiation. Interpreting the inverse matrix as a posterior covariance requires a suitable local Gaussian approximation; see the [Fisher tutorial](tutorials/02_fisher_analysis.md).

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
from jaxwave.inference.fisher import covariance_matrix as jax_covariance

def jax_wf_fn(p):
    hp_m, _ = jaxwave.get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=p["mass1"], mass2=p["mass2"],
        sample_frequencies=f_grid_jax,
    )
    return hp_m

gamma_jax, param_names_jax = jax_fisher(
    parameters_dict=params, waveform_fn=jax_wf_fn,
    psd_dict=psds_jax["H1"], frequencies=f_grid_jax, df=0.25,
)
cov_jax = jax_covariance(gamma_jax)
print(f"JAX sigma(mass1): {jnp.sqrt(cov_jax[0, 0]):.4f} Msun")
```

### Gradient-Based MAP Optimization

The `optimize_map` helper optimizes the log likelihood with Adam and optional box bounds. This is equivalent to MAP optimization for a uniform prior within those bounds; the helper does not accept a general prior. Check convergence separately from the requested iteration count:

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

The sampler provides Hamiltonian Monte Carlo with dual-averaging step-size adaptation. Supply a differentiable `log_posterior(theta)` that includes the intended priors and likelihood, using the same detector projection for data and templates. The fragment below assumes that function is defined; `sample_parallel` currently runs the chains sequentially. Gelman-Rubin statistics are one diagnostic and do not establish convergence by themselves.

```python
from torchwave.inference import NUTS, gelman_rubin

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
Benchmark model throughput, batch sizes, autograd backward passes, and cross-framework GPU speedups:
```bash
# Benchmark standard models across batch sizes
torchwave-bench --models popular --batch-sizes 1,10,100,1000

# Benchmark across backends (Torch eager, Torch compiled, JAX JIT, or all)
torchwave-bench --models IMRPhenomD --backend all --batch-sizes 1,10,100

# Benchmark compiled PyTorch with dynamic shapes (can reduce recompilation across batch sizes)
torchwave-bench --models TaylorF2,IMRPhenomD --compile --dynamic --batch-sizes 1,10,50,100

# Measure forward + backward autograd gradient time
torchwave-bench --models IMRPhenomD --backend torch --batch-sizes 100 --backward

# Compare against LALSimulation baseline and export Markdown table
torchwave-bench --models TaylorF2,IMRPhenomD,IMRPhenomXAS --compare-lal --markdown-out benchmark.md
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
Generate Kerr quasinormal mode tables directly from the Leaver continued-fraction eigenproblem:
```bash
torchwave-generate-qnm --output-dir ./qnm_data --modes 2,2,0
```

### `torchwave-generate-london-qnm`
Recompute the London $N=61$ quasinormal mode polynomial fits:
```bash
torchwave-generate-london-qnm --output-dir ./london_qnm
```

---

## Controlled Benchmarking Protocol

To reproduce published results, use `scripts/compare_packages.py` with the
[dated campaign's commands and environment](package_benchmarks.md#reproduction-and-receipts).
Regenerate its figures with the [maintained plot command](package_benchmarks.md#regenerating-the-plots).
For a new experiment, follow the [benchmark methodology](benchmarking_protocol.md).

### Running a Multi-Backend Benchmark Campaign

The [runner guide](benchmarking_protocol.md#choose-a-runner) distinguishes the
maintained package comparison, local CLI experiments and legacy campaigns.
The older `run_benchmark_campaign.py` can use JAXWave or self-consistency as its
reference; its `gate_passed` flag alone does not establish LAL agreement.

### Generating Figures

Use the [published campaign's plot instructions](package_benchmarks.md#regenerating-the-plots)
for maintained CPU/GPU figures. `plot_benchmark_results.py` reads the separate
legacy receipt format; see the [legacy qualification rules](benchmarking_protocol.md#legacy-campaigns).
