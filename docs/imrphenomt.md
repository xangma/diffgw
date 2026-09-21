# IMRPhenomT family: time-domain waveforms

The IMRPhenomT family generates binary black hole inspiral, merger, and
ringdown directly in the time domain. Both PyTorch and JAX use the same
parameters, calibrated mode content, and sampling conventions.

| Approximant | Spins | Coprecessing or aligned modes |
| --- | --- | --- |
| `IMRPhenomT` | Aligned | `(2, ±2)` |
| `IMRPhenomTHM` | Aligned | `(2, ±2)`, `(2, ±1)`, `(3, ±3)`, `(4, ±4)`, `(5, ±5)` |
| `IMRPhenomTP` | Precessing, NNLO angles | `(2, ±2)` |
| `IMRPhenomTPHM` | Precessing, NNLO angles | Same set as THM |

The precessing variants expose the published **NNLO** option. Numerical
spin evolution and MSA precession are not implemented for this family.
Consequently, the default here differs from LAL's default numerical
precession prescription; see [validation](#scope-and-validation).

## Generate a sampled waveform

```python
import diffgw

hp, hc, times = diffgw.get_td_waveform(
    "IMRPhenomT", backend="torch",  # use "jax" for JAX arrays
    mass1=30.0, mass2=25.0,
    chi1=0.2, chi2=-0.1,
    distance=100.0, inclination=0.5,
    phic=0.2, tc=0.0,
    f_lower=20.0, f_ref=20.0,
    delta_t=1 / 2048, duration=2.0,
    return_times=True,
)
```

The returned real arrays are `h_plus`, `h_cross`, and a shared time grid.
Masses are in solar masses, distance in Mpc, time in seconds, frequencies in
Hz, and angles in radians. `chi1` and `chi2` are dimensionless aligned spins;
`spin1z` and `spin2z` are equivalent aliases. `phic` specifies the orbital
phase at `f_ref`. An omitted or zero `f_ref` uses `f_lower` in this wrapper.

By default the grid ends near `t_pad` and the amplitude peak occurs at `tc`.
Samples before the model reaches `f_lower` are zero. A fixed `duration` can
truncate the inspiral; omitting it selects a duration from the frequency
evolution. Set `center_coalescence=False` for a grid starting at zero and
choose `tc` to place the peak. There is no periodic wrapping or FFT taper.

## Higher modes and precession

Select `IMRPhenomTHM` to include its five aligned mode pairs. Use `modes`
to select a subset, for example `modes=[(2, 2), (3, 3)]`. Each entry selects
both signs of that harmonic when projecting the polarizations. The `(3, 2)`
mode and memory modes are not available in this implementation.

For precession, supply the Cartesian dimensionless spin components in the
reference source frame:

```python
hp, hc, times = diffgw.get_td_waveform(
    "IMRPhenomTPHM", backend="torch",
    mass1=40.0, mass2=15.0,
    spin1x=0.2, spin1y=0.1, spin1z=0.3,
    spin2x=0.0, spin2y=0.1, spin2z=-0.2,
    distance=120.0, inclination=0.8, phic=0.3,
    precession_method="nnlo",
    f_lower=20.0, f_ref=30.0,
    delta_t=1 / 2048, duration=2.0, return_times=True,
)
```

`IMRPhenomTP` provides the same precession prescription with only the
quadrupole carrier. The aligned variants reject nonzero transverse spins.
Spin vectors must have magnitude at most one.

## Arbitrary grids and derivatives

The model class accepts a shared one-dimensional grid, with parameter batch
dimensions before the final time axis. This interface applies no `f_lower`
mask and uses 20 Hz when `f_ref` is zero.

```python
import torch
from torchwave import IMRPhenomT

times = torch.linspace(-0.5, 0.02, 1024, dtype=torch.float64)
mass1 = torch.tensor(30.0, dtype=torch.float64, requires_grad=True)
model = IMRPhenomT()
hp, hc = model.time_domain(times, mass1, 25.0, chi1=0.2, f_ref=20.0)
d_power_dm1, = torch.autograd.grad((hp.square() + hc.square()).sum(), mass1)

# Intrinsic frequency evolution, relative to the amplitude peak:
frequency = model.frequency(times, mass1, 25.0, chi1=0.2)
start = model.time_at_frequency(20.0, mass1, 25.0, chi1=0.2)
```

For JAX, use the same methods from `jaxwave.IMRPhenomT`:

```python
import jax
import jax.numpy as jnp
from jaxwave import IMRPhenomT

times = jnp.linspace(-0.5, 0.02, 1024)
model = IMRPhenomT()
evaluate = jax.jit(lambda m1: model.time_domain(times, m1, 25.0, chi1=0.2))
hp, hc = evaluate(30.0)
hp_batch, hc_batch = jax.vmap(evaluate)(jnp.array([30.0, 35.0]))
```

Fixed grids support first-order automatic differentiation. Frequency
inversion supplies an implicit first derivative; higher-order derivatives
through that inversion are not supported. Automatic duration selection is
an eager operation, so provide `duration` inside JAX transformations. A hard
start mask is discontinuous when its boundary crosses a sample.

## Scope and validation

All four models use native time-domain evaluation; frequency-domain dispatch
raises an error. The public wrappers reject nonzero tidal deformabilities,
`f_final`, unsupported modes, and unknown options. TP/TPHM accept only
`precession_method="nnlo"`; unsupported precession methods raise an error.

The NNLO prescription follows the option described in Appendix A of
[Estellés et al. (2022)](https://arxiv.org/abs/2105.05872).
The carrier retains its aligned-spin inspiral and merger; the NNLO remnant
spin modifies its ringdown. Angles use the aligned carrier's frequency
through its amplitude peak.
After the peak, the opening angle is held constant and the other angles
follow the remnant's quasinormal frequencies and minimal-rotation condition.
The transition is defined at physical time `tc`, independently of the
requested sampling grid. This corresponds to the published `10210` angle
prescription with the NNLO remnant-spin choice (`FinalSpinMod=0`).
Precessing results must be compared with the same angle and remnant-spin
options in an external reference, rather than its default configuration.

The reconstruction and calibration fits follow the authors' MIT-licensed
`phenomxpy` release pinned in the [provenance ledger](provenance_ledger.md).
Tests cover independent author-reference values, branch continuity,
frequency inversion, gradients, batching, JIT, and Torch/JAX FP64 parity.
Optional LAL binary comparisons have measured residuals documented in
[Differences from LAL](comparison.md#model-specific-differences-from-lal). LALSuite is not a runtime
dependency. See [citations](../CITATIONS.md) and
[third-party notices](../THIRD_PARTY_NOTICES.md).
