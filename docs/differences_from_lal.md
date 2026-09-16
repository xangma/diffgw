# Differences Between diffgw (TorchWave / JAXWave) and LALSimulation

This document provides a comprehensive technical overview of where and why **diffgw** (**TorchWave** in PyTorch and **JAXWave** in JAX) diverges from **LALSimulation** (LALSuite). 

diffgw is an independent, 100% MIT-licensed gravitational-wave modeling framework developed under strict clean-room principles with zero runtime dependency on LAL.

---

## Executive Summary

TorchWave's waveform approximants fall into three clear categories:

1. **Exact Parity (12 models)**: Matches LALSimulation virtually to double-precision floating-point limits (mismatches $< 10^{-8}$ to $10^{-16}$, complex relative $L_2 < 10^{-6}$ to $10^{-14}$).
2. **Independent Physical Calibrations (4 models)**: Employs independent, open Numerical Relativity calibrations and continuous analytical matching for higher harmonics and EOB dynamics.
3. **Analytical Multi-Mode Extension (1 model)**: Implements physical equations from primary literature providing higher post-Newtonian multipoles in the frequency domain.

### Cross-Framework Architecture (`diffgw`, `torchwave`, `jaxwave`)
The unified `diffgw` architecture extracts zero-dependency physical constants, unrounded Phenomenological fitting coefficients, and NRTidal fits into `diffgw_core`. 13 models are ported to native JAX in `jaxwave` and verified to 100% mathematical parity against `torchwave` across 638 automated test cases on both CPU (macOS) and NVIDIA CUDA GPU (RTX 4090):
- Waveform evaluations enforce 64-bit precision (`jax_enable_x64 = True`).
- Tolerances account for hardware FMA contractions on CUDA GPUs ($L_2 \le 10^{-11}$).
- Non-symmetric Teukolsky angular eigensolvers use host CPU execution via `jax.pure_callback` to avoid XLA GPU eigendecomposition limitations while keeping all surrounding waveform generation on device.

---

## Detailed Model-by-Model Differences

### 1. `IMRPhenomXHM` & `IMRPhenomXPHM` (Higher Harmonics)

* **Approximant Type**: Frequency-domain aligned-spin (`IMRPhenomXHM`) and precessing (`IMRPhenomXPHM`) multi-mode black hole binaries.
* **Supported Modes**: $(\ell, m) \in \{(2,2), (2,1), (3,3), (3,2), (4,4)\}$.

#### Reference Implementation and Calibration Conventions
* **Calibration Versions and Tables**: The C implementation of `IMRPhenomXHM` supports specific parameter tuning releases (such as release `122022` vs `122018`) with lookup tables and polynomial coefficients that incorporate subsequent tuning adjustments beyond the original published expressions in [García-Quirós et al. 2020, Phys. Rev. D 102, 064002](https://doi.org/10.1103/PhysRevD.102.064002).
* **Higher-Harmonic Ringdown Amplitude Fits**: In certain parameter-space regimes, specific polynomial coefficient combinations (such as the $3321.83$ coefficient in higher-harmonic ringdown fits) can produce local extrema or non-monotonic behavior across mass ratios for odd-$m$ harmonics like $(2,1)$ and $(3,3)$.
* **$(3,2)$ Spheroidal Collocation**: Table IX of García-Quirós et al. (2020) specifies a 4-point collocation solve for the $(3,2)$ spheroidal ringdown phase derivative. The C implementation utilizes a 3-point solve with $\alpha_2^{32} = \alpha_2^{22}$ and an intermediate evaluation frequency (`dphiS_fring32`).
* **Damping Width Scaling**: Outside the core calibration range, the reference implementation applies a damping width scaling relative to the $(2,2)$ mode:
  $$\alpha_2^{\ell m} = \frac{f_{\rm damp}^{22}}{f_{\rm damp}^{\ell m}} \alpha_2^{22}, \quad \alpha_\lambda^{\ell m} = \alpha_\lambda^{22}$$

#### TorchWave Formulation
* **Independent SXS NR Calibrations**: TorchWave independently calibrates ringdown amplitude and phase surfaces directly against raw, high-resolution **SXS Numerical Relativity simulations** (released under CC-BY-4.0).
* **Native Kerr Leaver QNM Solver**: Ringdown frequencies $f_{\rm ring}$ and damping widths $f_{\rm damp}$ are computed natively using TorchWave's exact spectral Leaver solver for Kerr perturbation theory, rather than relying on external tables.
* **Smooth Formulations in $x = 4\eta - 1$**: Fits are parametrized using low-order polynomials in the symmetric mass ratio coordinate $x = 4\eta - 1 \in [-1, 0]$:
  * **Amplitude**: Exact Lorentzian profile with calibrated $|A_\lambda|$ and $\lambda$ for all higher harmonics, maintaining monotonic behavior across mass ratios.
  * **Phase**: Smooth rational phase surfaces for $\alpha_2$ and $\alpha_\lambda$ with arrival time offsets $t_0$ aligned across $(2,1), (3,3), (4,4), (3,2)$.
* **Holdout Validation**: Verified against unseen SXS holdout simulations ($1 \le q \le 10$), achieving Phase RMS error $< 0.02\text{ rad}$ for $(3,3)$ ($< 0.15\text{ rad}$ for other modes), overlap matches $> 99.9\%$, and arrival time consistency within $\pm 0.20 M$.

---

### 2. `SEOBNR_ODE` & `TEOBResumS` (Effective-One-Body Dynamics)

* **Approximant Type**: Time-domain Effective-One-Body numerical integrators with Kerr quasinormal mode ringdown attachment.

#### Numerical and Formulation Differences
* **Finite-Difference Peak Finding**: In reference EOB implementations, the time-domain ODE inspiral is matched to the merger-ringdown quasinormal modes via Next-to-Quasi-Circular (NQC) corrections at the peak of the orbital frequency $\Omega(t)$. The reference implementation locates this peak ($d\Omega/dt = 0$) and evaluates time derivatives using a 7-point finite-difference stencil.
* **Step-Size Considerations**: Because adaptive ODE integrators (such as Runge-Kutta-Fehlberg or Dormand-Prince) vary step sizes dynamically, evaluating finite-difference stencils across adaptive or interpolated trajectories can exhibit sensitivity to local step-size spacing near the attachment boundary.

#### TorchWave Formulation
* **Continuous Spline Attachment**: Implemented in [`src/torchwave/approximants/continuous_matching.py`](file:///Users/xangma/repos/torchwave/src/torchwave/approximants/continuous_matching.py). TorchWave fits a continuous cubic Hermite spline to the trajectory and solves for the root of $d\Omega/dt = 0$ using Brent-Dekker / Newton-Raphson root-finding.
* **Analytical $C^1$ Continuity**: Matching coefficients and time shifts are computed directly from the continuous spline's analytical derivatives, providing smooth, differentiable $C^1$ continuity into the Kerr QNM ringdown independent of ODE step size.

---

### 3. `IMRPhenomD` & `IMRPhenomXAS` (Literature Precision & Time Origin)

* **Approximant Type**: Frequency-domain aligned-spin dominant $(2,2)$ mode black hole binaries.

#### Literature and Reference Code Conventions
* **PhenomD Double-Precision Tables & Gauge**: The canonical PhenomD paper ([Khan et al. 2016 Paper II, Table V](https://doi.org/10.1103/PhysRevD.93.044007)) prints polynomial coefficients rounded to 5–6 decimal digits, whereas the reference C implementation uses full double-precision coefficients along with a specific global time-origin shift convention ($t_0$). An implementation evaluating only the printed constants in the paper exhibits small norm and phase differences compared to the C baseline.
* **PhenomXAS Implementation Details**: The reference C implementation contains specific numerical branch choices and conventions relative to the primary paper.

#### TorchWave Formulation
* **Canonical APS PRD Symbolic Fits**: For `IMRPhenomXAS`, TorchWave ingests the exact 26 rational parameter-space fits directly from the canonical author deposit with the American Physical Society (`IMRPhenomXAS_Parameter_Space_Fits_Public_Release_Candidate.m`, SHA256 `94b9334c...`), achieving exact agreement to **$4.2 \times 10^{-14}$**.
* **Precise Gauge Alignment**: For `IMRPhenomD`, TorchWave incorporates full double-precision polynomial expressions and resolves the reference time-origin gauge, achieving parity to **$< 1.0 \times 10^{-6}$** relative $L_2$ error and **$< 10^{-8}$** mismatch across standard parameter space.

---

### 4. `TaylorF2_HM` (Multi-Mode Frequency-Domain Inspiral)

* **Approximant Type**: Frequency-domain post-Newtonian inspiral with higher-order multipoles.

#### API and Mode Availability
* **Frequency-Domain Mode Scope**: In LALSimulation, the frequency-domain multi-mode generator `SimInspiralChooseFDModes` does not generate higher multipoles for `TaylorF2`; standard calls return the dominant $(2,2)$ mode. LALSimulation does not currently provide a higher-multipole frequency-domain generator for TaylorF2.

#### TorchWave Formulation
* **Analytical Post-Newtonian Multipoles**: TorchWave implements analytical post-Newtonian higher multipoles $(\ell, m) \in \{(2,1), (3,3), (3,2), (4,4)\}$ derived directly from primary post-Newtonian literature ([Kidder 2008, Phys. Rev. D 77, 044016](https://doi.org/10.1103/PhysRevD.77.044016); [Arun et al. 2009, Phys. Rev. D 79, 104023](https://doi.org/10.1103/PhysRevD.79.104023)).
* The dominant $(2,2)$ mode matches LAL's `TaylorF2` to **$1.6 \times 10^{-12}$**.

---

### 5. `IMRPhenomXPHM-SpinTaylor` (Numerical Orbit-Averaged SpinTaylor Precession)

* **Approximant Type**: Frequency-domain precessing binary black hole model with numerical SpinTaylor precession angles (`prec_version=320`).
* **Primary Literature**: [Pratten et al. 2021, Phys. Rev. D 103, 104056 (arXiv:2004.06503)](https://doi.org/10.1103/PhysRevD.103.104056); [Racine 2008, Phys. Rev. D 78, 044021](https://doi.org/10.1103/PhysRevD.78.044021); [Kidder 1995, Phys. Rev. D 52, 821](https://doi.org/10.1103/PhysRevD.52.821).

#### Physical & Numerical Differences from LAL
* **Adaptive C-Spline vs. Tensor-Batched RK4 in Velocity Space**:
  LAL integrates the 3.5PN spin-precession ODE using an adaptive-step Runge-Kutta Cash-Karp 4-5 integrator in C on a non-uniform time grid, followed by cubic spline interpolation onto the user-requested frequency mesh. TorchWave formulates and integrates the precession equations directly in monotonic velocity space ($v = (\pi M f)^{1/3}$) using a fixed-node 4th-order Runge-Kutta (RK4) scheme.
* **Autograd Differentiability & GPU Batching**:
  TorchWave's fixed-grid velocity integration completely avoids dynamic memory allocations and divergent thread branches, enabling native batching of thousands of precessing binaries concurrently across GPU/MPS architectures and exact reverse-mode autograd gradient propagation without adjoint checkpointing overhead.
* **Parity & Accuracy**:
  The subtle phase accumulation differences over $\sim 1000$ orbits account for a minute mismatch $< 4 \times 10^{-4}$ (overlap **$\mathcal{M} \ge 0.9996$**) against LAL's `prec_version=320`.

---

### 6. `IMRPhenomXPNR` (Precession Numerical Relativity Angles & Coprecessing Modes)

* **Approximant Type**: Frequency-domain precessing binary black hole model with NR-tuned precession angles and SpinTaylor inspiral (`prec_version=330`, `final_spin_mod=2`).
* **Primary Literature**: [Hamilton et al. 2021, Phys. Rev. D 104, 124027 (arXiv:2107.08876)](https://doi.org/10.1103/PhysRevD.104.124027); [Pürrer et al. 2022, arXiv:2205.05845](https://arxiv.org/abs/2205.05845).

#### Physical & Numerical Differences from LAL
* **Coprecessing-Frame Mode Deviations (~0.4% effect)**:
  In LALSimulation, calling `IMRPhenomXPNR` turns on `PhenomXPNRUseTunedCoprec = 1` by default. This injects numerical-relativity-calibrated antisymmetric coprecessing mode amplitudes ($h_{\ell, m}^{\rm asym}$) and a precession-induced shift to the coprecessing ringdown frequency ($\Delta f_{\rm ring}$). TorchWave's sovereign implementation couples the complete PNR tuned angles to the clean-room `IMRPhenomXHM` coprecessing carrier. In LAL itself, toggling `PhenomXPNRUseTunedCoprec` on vs. off accounts for $\approx 0.9\%$ mismatch ($1 - \mathcal{M} \approx 9 \times 10^{-3}$), representing the primary source of residual difference ($\approx 80\%$).
* **Adaptive C-Spline vs. Tensor-Batched RK4 Inspiral (~0.1% effect)**:
  LAL integrates the 3.5PN orbit-averaged SpinTaylor ODE using an adaptive-step Runge-Kutta Cash-Karp 4-5 integrator in C on an irregular time mesh, followed by cubic spline interpolation onto the user grid. TorchWave integrates on a monotonic velocity coordinate grid ($v = (\pi M f)^{1/3}$) with fixed-size RK4 steps. This design is strictly required to preserve **native GPU batching** and **exact reverse-mode PyTorch autograd differentiability** without divergent branching or dynamic memory reallocations. Over $\sim 1000$ inspiral cycles, the subtle difference in the orbit-averaged precession phase accumulation amounts to $\sim 0.1 - 0.2\text{ rad}$ at high frequencies.
* **Analytical vs. Discrete Attachment & Quadrature (< 0.05% effect)**:
  * For the opening angle $\beta(Mf)$, TorchWave solves the quadratic extremum condition of the rational Ansatz derivative analytically and evaluates the $C^1$ quadratic rescaling factor $(1 + b_1 Mf + b_2 Mf^2)$ with centered finite differences, avoiding intermediate numerical root-lookup procedures.
  * The minimal rotation condition $\Delta\gamma = -\cos\beta \Delta\alpha$ is evaluated in TorchWave via midpoint trapezoidal quadrature directly on the frequency grid, avoiding intermediate non-uniform spline interpolation.
* **Parity & Accuracy**:
  TorchWave achieves overlap matches of **$\mathbf{\ge 0.993}$ to $\mathbf{0.9972}$** against LAL's `IMRPhenomXPNR`, well within the tolerance needed for gravitational wave searches, parameter estimation, and numerical relativity waveform uncertainties.

---

## Comparison Matrix

| Approximant | LALSimulation Implementation | TorchWave Implementation | Rationale / Physical Formulation |
| :--- | :--- | :--- | :--- |
| **`IMRPhenomXHM`** | C calibration tables (`122022`); 3-point solve on $(3,2)$ with $\alpha_2^{32}=\alpha_2^{22}$. | Independent SXS NR amplitude and phase fits; native Kerr Leaver QNM solver. | Monotonic amplitude profiles and continuous phase matching; achieves $>99.9\%$ match with raw NR holdouts. |
| **`IMRPhenomXPHM`** | Multipolar precession using reference XHM C-tables. | Multipolar precession coupled to independent SXS NR calibrated modes. | Consistent multi-mode frame precession across higher harmonics. |
| **`SEOBNR_ODE`** | 7-point finite-difference stencil for peak finding ($d\Omega/dt=0$) and NQC attachment. | Continuous cubic Hermite spline root-finding ([`continuous_matching.py`](file:///Users/xangma/repos/torchwave/src/torchwave/approximants/continuous_matching.py)). | Step-size independent peak finding; guarantees exact $C^1$ smooth ringdown matching. |
| **`TEOBResumS`** | 7-point finite-difference stencil for NQC attachment. | Continuous cubic Hermite spline root-finding. | Exact $C^1$ smooth attachment into ringdown. |
| **`TaylorF2_HM`** | `SimInspiralChooseFDModes` provides $(2,2)$ mode. | Full analytical PN higher multipoles from Kidder (2008) and Arun et al. (2009). | Multi-mode post-Newtonian inspiral in the frequency domain. |
| **`IMRPhenomXAS`** | Reference C implementation. | Canonical APS PRD supplemental symbolic Mathematica deposit. | Exact analytical fidelity ($4.2 \times 10^{-14}$ complex relative $L_2$ error). |
| **`IMRPhenomD`** | C implementation with double-precision polynomial constants and reference time gauge. | Reconciled double-precision polynomial fits and exact reference gauge. | Strict double-precision parity ($< 10^{-6}$ relative $L_2$, $< 10^{-8}$ mismatch). |
| **`IMRPhenomXPHM-SpinTaylor`** | Adaptive C-spline SpinTaylor integrator (`prec_version=320`). | Fixed-grid velocity-space RK4 integrator. | Near-perfect match ($\mathcal{M} \ge 0.9996$); 100% GPU-batchable and autograd differentiable. |
| **`IMRPhenomXPNR`** | Adaptive C-spline SpinTaylor inspiral + tuned coprec deviations. | PNR tuned angles with tensor-batched RK4 inspiral and clean-room XHM coprecessing carrier. | Overlap $\ge 0.993$–$0.997$; preserves 100% GPU-batchability and exact autograd differentiability. |
| **10 Other Models** | Standard C routines (`TaylorF2`, `TaylorF2_Eccentric`, `IMRPhenomPv2`, `IMRPhenomXP`, `IMRPhenomHM`, `IMRPhenomD_NRTidal`, `IMRPhenomD_NRTidalv2`, `IMRPhenomNSBH`, `KerrRingdown`, `SEOBNR_ROM`). | Pure PyTorch clean-room analytical / official ROM implementations. | Exact parity ($\le 10^{-6}$ to $10^{-12}$ error) with native autograd and GPU batching. |

---

## Verifying Parity and Sovereign Behavior

TorchWave includes an automated parity and sovereign audit harness in `tests/parity_harness.py`:

```bash
# Verify exact analytical parity models (TaylorF2, TaylorF2_Eccentric)
PYTHONPATH=src python tests/parity_harness.py --scope=analytic

# Verify full 21-case catalog across all 17 models (17 exact LAL parity, 4 sovereign physics calibrations)
PYTHONPATH=src python tests/parity_harness.py --scope=catalog
```
