# DiffGW: Differentiable Gravitational-Wave Modeling & Inference

<p align="center">
  <a href="https://xangma.github.io/diffgw/"><img src="https://img.shields.io/badge/docs-diffgw-indigo.svg" alt="Documentation"></a>
  <a href="https://github.com/xangma/diffgw"><img src="https://img.shields.io/badge/github-diffgw-blue.svg" alt="GitHub"></a>
  <img src="https://img.shields.io/badge/python-3.10%20--%203.14-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/framework-PyTorch%20%7C%20JAX-orange.svg" alt="Frameworks">
  <img src="https://img.shields.io/badge/precision-FP64%20Support-success.svg" alt="Precision">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</p>

**DiffGW** provides batched gravitational-wave models and inference tools in two native frontends, sharing constants and coefficient tables through the framework-independent `diffgw_core` package:

- **`torchwave`**: PyTorch tensor evaluation, batched parameters, and autograd on supported model paths.
- **`jaxwave`**: JAX evaluation with JIT compilation, vectorization, and automatic differentiation on supported model paths.

The original software is distributed under the **MIT License**, with PyTorch or JAX and scientific Python dependencies. Waveform models follow published literature, with calibration provenance recorded explicitly. IMRPhenomT also uses the authors' MIT-licensed calibration release and reconstruction formulas; see the [provenance ledger](provenance_ledger.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).

---

<div class="grid cards" markdown>

- :zap: **Batched Execution**
    ---
    Tensor batches and optional compilation support repeated waveform evaluation. [Measured benchmarks](package_benchmarks.md) report CPU/CUDA timings with the model, grid, precision, batch size and numerical checks.

- :dart: **Numerical Validation**
    ---
    Reference comparisons and cross-framework tests cover declared cases with model-specific tolerances. [Validation evidence](validation.md) records their scope and outstanding differences.

- :twisted_rightwards_arrows: **Two Native Frontends**
    ---
    PyTorch and JAX interfaces share physical constants and coefficient tables. Select either backend through the unified `diffgw` interface.

- :triangular_ruler: **Native Autodiff**
    ---
    Supported model paths provide derivatives for Fisher forecasts, MAP optimization, and gradient-based sampling. Derivative checks and statistical assumptions are described in the [tutorials](tutorials/02_fisher_analysis.md).

- :telescope: **Detector Networks**
    ---
    Ground-detector projections and analytical noise models, plus LISA and pulsar-timing response utilities.

- :rocket: **Likelihood Tools**
    ---
    Dense Whittle Gaussian likelihood and a relative-binning approximation that evaluates candidate waveforms at fewer frequencies. Accuracy and cost depend on the grid, binning and fiducial waveform; see [likelihood validation](validation.md#relative-binning).

</div>

---

## Quickstart

=== "PyTorch (`torchwave`)"

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
        dtype=torch.float64,
    )

    print("PyTorch polarization shape:", hp.shape)  # torch.Size([1969])

    # Batched generation across 10,000 systems in a single call (B, N)
    m1_b = torch.linspace(20.0, 50.0, 10000, dtype=torch.float64)
    m2_b = torch.linspace(15.0, 45.0, 10000, dtype=torch.float64)
    hp_batch, hc_batch = torchwave.get_fd_waveform(
        approximant="IMRPhenomD",
        mass1=m1_b, mass2=m2_b,
        f_lower=20.0, f_final=512.0, delta_f=1.0,
    )
    print("Batched tensor shape:", hp_batch.shape)  # torch.Size([10000, 493])
    ```

=== "JAX (`jaxwave`)"

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

    # Differentiate the implemented plus-polarization power with respect to chirp mass
    @jax.jit
    @jax.grad
    def mc_loss(mc):
        hp_val, _ = generator(chirp_mass=mc, mass_ratio=1.0)
        return jnp.sum(jnp.abs(hp_val) ** 2)

    grad = mc_loss(26.1)
    print(f"d(Power)/d(M_c) = {grad:.4e}")
    ```

=== "Unified Dispatch (`diffgw`)"

    ```python
    import diffgw

    # Dispatch to PyTorch backend
    hp_t, hc_t = diffgw.get_fd_waveform(
        "IMRPhenomD", mass1=30.0, mass2=25.0,
        f_lower=20.0, f_final=512.0, delta_f=0.25,
        backend="torch",
    )

    # Dispatch to JAX backend
    hp_j, hc_j = diffgw.get_fd_waveform(
        "IMRPhenomD", mass1=30.0, mass2=25.0,
        f_lower=20.0, f_final=512.0, delta_f=0.25,
        backend="jax",
    )
    ```

---

## Installation

=== "PyTorch Support"

    ```bash
    git clone https://github.com/xangma/diffgw.git
    cd diffgw
    pip install -e ".[torch]"
    ```

=== "JAX Support"

    ```bash
    git clone https://github.com/xangma/diffgw.git
    cd diffgw
    pip install -e ".[jax]"
    ```

=== "All Backends & Development"

    ```bash
    git clone https://github.com/xangma/diffgw.git
    cd diffgw
    pip install -e ".[all,dev,docs]"
    ```

---

## Waveform Models & Framework Parity

The catalog lists implemented model families. Interface availability, agreement with an external reference, and cross-framework consistency are separate properties; consult [validation evidence](validation.md) and the [model comparison](comparison.md#model-specific-differences-from-lal) for the tested configurations and known differences.

| Family | Approximant(s) | Harmonics / Multipoles | Physical Features & Regimes |
| :--- | :--- | :---: | :--- |
| **Post-Newtonian** | `TaylorF2`<br>`TaylorF2_HM`<br>`TaylorF2_Eccentric` | $(2,2)$<br>Eight modes through $\ell=4$<br>$(2,2)$ | 3.5PN point-particle & spin phasing, BNS tidal deformability<br>Leading-order higher-mode amplitudes<br>Moore et al. (2016) 3PN stationary-phase eccentric corrections |
| **Aligned-Spin IMR** | `IMRPhenomD`<br>`IMRPhenomHM`<br>`IMRPhenomXAS`<br>`IMRPhenomXHM` | $(2,2)$<br>$(2,\pm 2), (2,\pm 1), (3,\pm 3), (3,\pm 2), (4,\pm 4)$<br>$(2,\pm 2)$<br>$(2,\pm 2), (2,\pm 1), (3,\pm 3), (3,\pm 2), (4,\pm 4)$ | Aligned-spin dominant mode, 5-collocation intermediate<br>Multi-harmonic PhenomD calibration with mode mixing<br>4th-generation aligned-spin dominant mode (Pratten et al. 2020)<br>4th-generation higher multipoles (García-Quirós et al. 2020) |
| **Native Time-Domain IMR** | `IMRPhenomT`, `IMRPhenomTHM`, `IMRPhenomTP`, `IMRPhenomTPHM` | Up to $(5,\pm 5)$ | Aligned or precessing black hole binaries; see the [T-family guide](imrphenomt.md) |
| **Precessing IMR** | `IMRPhenomP` / `Pv2`<br>`IMRPhenomXP`<br>`IMRPhenomXPHM` | $(2,\pm 2)$ twisted<br>$(2,\pm 2)$ twisted<br>Multipolar twisted | Generic 3D spins with NNLO PN precession angles & Wigner $d$-matrices<br>4th-gen precessing carrier with MSA analytical precession<br>Multipolar precession supporting MSA (`v223`), SpinTaylor (`v320`), and PNR (`v330`) |
| **Tidal & Matter** | `IMRPhenomD_NRTidal`<br>`IMRPhenomD_NRTidalv2`<br>`IMRPhenomNSBH` | $(2,2)$<br>$(2,2)$<br>$(2,2)$ | Dietrich et al. (2017) rational Padé BNS tidal phasing (NRTidal v1)<br>Dietrich et al. (2019) NRTidal v2 with self-spin & Planck windowing<br>Neutron star–black hole model with tidal disruption & remnant torus |
| **Effective-One-Body** | `SEOBNR_ROM` / `SEOBNRv4_ROM`<br>`SEOBNR_ODE`<br>`TEOBResumS` | $(2,2)$<br>$(2,2)$<br>$(2,2)$ | External SEOBNRv4 ROM data with a native evaluation wrapper<br>Time-domain Hamiltonian ODE integrator with continuous QNM matching<br>Damour-Nagar EOB dynamics with spin-orbit coupling and tidal potentials |
| **Ringdown & Bursts** | `KerrRingdown`<br>`SineGaussian`<br>`MemoryWaveform` | $(2,2,0), (l,m,n)$<br>Burst packet<br>$(2,0)$ | Damped sinusoids from Kerr quasinormal modes (time & frequency domain)<br>Standard unmodeled sine-Gaussian burst wavelet<br>Non-linear Christodoulou gravitational-wave memory effect $h_{20}^{\text{mem}}(t)$ |
| **Specialized Models** | `ContinuousWave`<br>`EMRIKludge`<br>`ParameterizedPostEinstein`<br>`SpinPrecessionODE` | Monochromatic<br>Multi-harmonic<br>Modified phasing<br>Orbit-averaged | Continuous waves from triaxial rotating neutron stars with Doppler modulation<br>Extreme mass ratio inspirals (LISA) with Kerr geodesic orbital frequencies<br>Model-independent testing of General Relativity ($\beta_{\rm ppE}, b_{\rm ppE}$)<br>Orbit-averaged spin precession ODE integrator |
| **Time-Domain Tools** | `get_td_waveform`<br>`HybridMECO` | Supported FD models<br>Diagnostic | FFT conditioning and time-domain waveform generation<br>Minimum Energy Circular Orbit hybrid velocity & frequency root solver |

---

## Documentation Navigation

<div class="grid cards" markdown>

- :material-table: **[Package Comparison](comparison.md)**
    ---
    How DiffGW's tensor backends, model coverage, derivatives and measured performance compare with related packages.

- :material-book-open-page-variant: **[User Guide](usage.md)**
    ---
    In-depth guide covering all approximants, batching, JAX JIT, detectors, and Bayesian inference.

- :material-speedometer: **[Performance & Benchmarks](performance.md)**
    ---
    Recorded FP64 waveform benchmarks on CPU and RTX 5090, with timing boundaries, numerical checks and reproduction commands.

- :material-school: **[Step-by-Step Tutorials](tutorials/01_quickstart_and_parity.md)**
    ---
    Practical guides for quickstart & parity, forward-mode Fisher analysis, and NUTS Bayesian inference.

- :material-code-tags: **[API Reference](api/torchwave.md)**
    ---
    Auto-generated function signatures, parameter descriptions, and return types for `torchwave`, `jaxwave`, and `diffgw`.

- :material-certificate: **[Provenance & Citations](provenance_ledger.md)**
    ---
    Mathematical sources, calibration provenance and primary literature citations ([`CITATIONS.md`](../CITATIONS.md)).

</div>
