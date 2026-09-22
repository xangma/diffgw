# IMRPhenomT family: time-domain waveforms

The IMRPhenomT family generates binary black hole inspiral, merger, and
ringdown directly in the time domain. Both PyTorch and JAX use the same
parameters, calibrated mode content, and sampling conventions.

| Approximant | Spins | Coprecessing or aligned modes |
| --- | --- | --- |
| `IMRPhenomT` | Aligned | `(2, ±2)` |
| `IMRPhenomTHM` | Aligned | `(2, ±2)`, `(2, ±1)`, `(3, ±3)`, `(4, ±4)`, `(5, ±5)` |
| `IMRPhenomTP` | Precessing, selectable angles | `(2, ±2)` |
| `IMRPhenomTPHM` | Precessing, selectable angles | Same set as THM |

The precessing variants accept **NNLO**, double-spin **MSA**, and **numerical**
spin evolution. NNLO remains the default for compatibility with existing
calls. Choose the method explicitly when comparing with LAL; see
[validation](#scope-and-validation).

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

Set `precession_method` and, optionally, `precession_version`:

| Method | Default version | Remnant spin |
| --- | --- | --- |
| `"nnlo"` | `10210` | Effective single-spin prescription (`FinalSpinMod=0`) |
| `"msa"` | `22310` | Precession-averaged double-spin prescription (`FinalSpinMod=3`) |
| `"numerical"` | `300` | Spin projections evolved to coalescence (`FinalSpinMod=4`) |

Precessing MSA requires unequal component masses and a supported MSA root
and remnant-spin domain. Its singular equal-mass limit is not implemented;
use numerical precession for those systems. Aligned equal-mass binaries
remain supported. Unsupported MSA points raise errors in eager evaluation
and produce NaN for affected rows when traced by JAX.
The equal-mass regularization must preserve version 223's radiation-reaction
and angle expansion orders. The equal-mass invariants described by
[Gerosa et al.](https://arxiv.org/abs/1612.05263) alone do not supply that
complete limit; substituting a leading-order equal-mass angle model would
change the selected prescription. See also the discussion of singular
couplings in Sec. IV.2 of [Pratten et al.](https://arxiv.org/abs/2004.06503).

NNLO and MSA support the following continuation choices:

| NNLO | MSA | Euler-angle treatment |
| --- | --- | --- |
| `10200` | `22300` | Analytical angles throughout the waveform |
| `10210` | `22310` | Analytical inspiral, quasinormal-mode ringdown continuation |
| `10211` | `22311` | Numerical minimal-rotation gamma, with ringdown continuation |
| `10221` | `22321` | Linear alpha/beta from MECO to the peak, numerical gamma, ringdown continuation |

The aliases `102` and `223` select `10210` and `22310`. Versions must match
the selected method. For example, use `precession_method="msa",
precession_version=22311` in the waveform call above. Numerical version
`300` evolves both spins and the orbital-plane direction using the aligned
T carrier frequency, then continues the angles through ringdown.
The spin equations retain the leading spin-orbit and spin-spin terms and
the NNLO and N4LO spin-orbit terms. They exclude the N3LO (3PN) quadratic-spin
terms, following the `phenomtp` selection documented in the
[public LAL API](https://lscsoft.docs.ligo.org/lalsuite/7.26/lalsimulation/_l_a_l_sim_inspiral_8h.html).
The leading self-spin coupling is retained with the black-hole quadrupole
coefficient equal to one. The internal `spin_derivatives` helper also offers
the full published equations, including N3LO quadratic-spin terms, for
independent equation checks.
Orbital-plane evolution uses the
nonspinning 3PN angular-momentum magnitude, with the spin-dependent angular
momentum corrections disabled as described in Sec. II.4.1 of the model paper.
The numerical solver requires `f_lower >= 1 Hz` and a reference frequency
no higher than the aligned carrier frequency at its amplitude peak. It uses
a fixed intrinsic grid independent of waveform samples. Direct queries earlier
than its integration interval (down to `min(1 Hz, f_ref/2)`) return NaN.
The published waveform benchmark does not measure this numerical solver's
timing or differentiation memory use.

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
`f_final`, unsupported modes, and unknown options. Unsupported precession
methods and incompatible version selections raise an error.

The precession prescriptions follow the options described in Appendix A of
[Estellés et al. (2022)](https://arxiv.org/abs/2105.05872).
The carrier retains its aligned-spin inspiral and merger; the selected
remnant spin modifies its ringdown. Angles use the aligned carrier's
frequency through its amplitude peak.
For the ringdown-continuation options, the opening angle is held constant
after the peak and the other angles
follow the remnant's quasinormal frequencies and minimal-rotation condition.
The transition is defined at physical time `tc`, independently of the
requested sampling grid. The default corresponds to the published `10210`
angle prescription with the NNLO remnant-spin choice (`FinalSpinMod=0`).
Precessing results must be compared with the same angle and remnant-spin
options in an external reference, rather than its default configuration.

The reconstruction and calibration fits follow the authors' MIT-licensed
`phenomxpy` release pinned in the [provenance ledger](provenance_ledger.md).
Tests cover independent author-reference values, branch continuity,
frequency inversion, gradients, batching, JIT, and Torch/JAX FP64 parity.
Optional LAL binary comparisons have measured residuals documented in
[T-family reference report](validation/phenomt.md). LALSuite is not a runtime
dependency. See [citations](../CITATIONS.md) and
[third-party notices](../THIRD_PARTY_NOTICES.md).
