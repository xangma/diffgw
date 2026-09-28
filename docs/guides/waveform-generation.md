# Waveform generation

Choose a model from the [model support catalog](../models.md). The sections below cover waveform generation and model conventions.

<a id="1-quickstart"></a>
## Quickstart

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


<a id="2-model-availability-cross-framework-parity"></a>
## Model availability and cross-framework parity

Use the [model support catalog](../models.md) to compare physical features,
backend and derivative restrictions, external data requirements, and validation
status. [Accuracy and validation](../validation.md) distinguishes frontend
consistency, external-reference agreement and independent physical evidence.

---


<a id="4-frequency-domain-waveforms-get_fd_waveform"></a>
## Frequency-domain waveforms (`get_fd_waveform`)

### Basic Evaluation & Frequency Grids

For native-D `IMRPhenomHM`, pass `hm_reference_phase="orbital"` with an explicit
positive `f_ref` to use an orbital phase origin consistently across modes.
The default `"legacy"` preserves earlier results. Both Torch and JAX support
the option; see the [HM mathematical specification](../math/imrphenomhm-waveform.md)
for conventions, calibration choices and validation limits.

The separate option `hm_phase_convention="spa_sxs_v1"` applies the derived
leading-SPA SXS-basis phase factor to each native-D mode before projection.
Both frontends support it; `hm_phase_convention="legacy"` remains the default.
It leaves the two $m=2$ modes and all intrinsic mode magnitudes unchanged.
Using this factor throughout the IMR signal is an explicit model convention,
not a physical polarization rotation or a claim of NR accuracy.

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

### IMRPhenomXP

`IMRPhenomXP` uses the same frequency-domain generation interface on both
frontends. Supply Cartesian spin components for a precessing binary:

```python
import diffgw

hp, hc = diffgw.get_fd_waveform(
    approximant="IMRPhenomXP",
    backend="torch",  # Use "jax" for the JAX frontend.
    mass1=40.0, mass2=25.0,
    spin1x=0.3, spin1y=0.1, spin1z=0.2,
    spin2x=0.0, spin2y=-0.2, spin2z=-0.1,
    distance=410.0, inclination=0.5,
    f_ref=20.0, f_lower=20.0, f_final=512.0, delta_f=0.5,
)
```

This demonstrates generation. For comparisons with LAL, also check the
[reference conventions and model options](../differences_from_lal.md#model-specific-differences).

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

   This resolves to `prec_version=223`, `convention=1`, and
   `final_spin_mod=3` when those options are omitted. An explicit
   `prec_version` takes precedence over the convenience method label.

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


<a id="6-time-domain-waveforms-get_td_waveform"></a>
## Time-domain waveforms (`get_td_waveform`)

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

See the [IMRPhenomT guide](../imrphenomt.md) for arbitrary grids, differentiation,
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
(the default), `"msa"`, or `"numerical"`; see the [T-family guide](../imrphenomt.md#higher-modes-and-precession)
for versions and domain restrictions.

---


<a id="7-black-hole-ringdown-kerr-quasinormal-modes"></a>
## Black hole ringdown and Kerr quasinormal modes

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


<a id="8-specialized-astrophysical-models"></a>
## Specialized astrophysical models

- **`EMRIKludge`**: Relativistic extreme-mass-ratio inspirals using Kerr geodesic orbital frequencies.
- **`ParameterizedPostEinstein` (`PPE`)**: Testing general relativity with parameterized PN phase modifications.

---
