# Pass 30: Kerr strict external oracle

Date: 2026-09-13

## Decision

The Kerr strict case remains unresolved.  The public LALSimulation API is
enough to establish that `SimBlackHoleRingdown` returns a single damped QNM
time series for this case, but it does not expose enough source-defined
conventions to freeze a 1e-6 complex waveform and 1e-8 mismatch reference.
No parity harness gate or physical TorchWave convention was changed.

## Clean-room evidence

The public Python signature is:

```text
SimBlackHoleRingdown(LIGOTimeGPS t0, double phi0, double deltaT,
                     double mass, double dimensionless_spin,
                     double fractional_mass_loss, double distance,
                     double inclination, int l, int m)
```

The public `lalsim-bh-ringdown --help` interface documents mass, spin,
distance, fractional energy loss, inclination, azimuth, and `(l,m)`.  The
public `SimBlackHoleRingdownMode` and
`SimBlackHoleRingdownSpheroidalWaveFunction` entry points expose mode and
angular values, but the API documentation does not state the exact mapping
between `phi0`/azimuth and a complex modal phase, the positive/negative-m
mirror coefficient, or the complex spheroidal normalization and phase used by
the returned plus/cross pair.  The ringdown routine also returns time-domain
series only; there is no public Fourier convention for this adapter.

The independent source equations used by TorchWave are Ruiz et al.,
arXiv:0707.4654, Flanagan & Hughes, gr-qc/9701039, Berti, Cardoso & Will,
gr-qc/0512160, and Cook & Zalutskiy, arXiv:1410.7698.  The public LAL
metadata describes an older `h0` energy-loss formula, but does not identify
its exact relation to TorchWave's finite-Q `native_pair` normalization.

## Black-box measurements

For `M=65 Msun`, `a=0.7`, fractional loss `0.01`, distance `100 Mpc`,
`l=m=2`, and `inclination=0`, the LAL time series is numerically one exact
damped complex exponential: fitting `h_plus+i h_cross` from adjacent samples
has residual at approximately machine precision.  Its frequency is
264.7638031321792 Hz and its damping rate is 252.3539699120653 s^-1.  These
frequencies agree with TorchWave's independent QNM backend, so the remaining
problem is convention and normalization rather than QNM shape.

Using the obvious `phi=0` TorchWave physical spheroidal path and comparing the
first sample gives:

```text
inclination       |LAL|/|TorchWave|   arg(LAL/TorchWave)
0.0 rad                 1.0009158260       -3.1131215064 rad
0.3 rad                 1.0008920781       -3.1131635560 rad
0.8 rad                 0.9999034498       -3.1151703913 rad
1.2 rad                 1.0006552725       -3.1222732148 rad
pi/2 rad               1.0009919287       -3.1415926536 rad
```

The residual is not a single global scale or phase: it varies with
inclination.  At inclination zero, varying spin also changes the ratio
(0.99999995 at `a=0`, 1.00012561 at `a=0.3`, 1.00091583 at `a=0.7`, and
0.99868596 at `a=0.9`).  Thus a fitted amplitude/phase bridge would be an
empirical correction, not a source-defined convention, and would fail the
strict gates outside the fitted point.

The public spheroidal-function call further shows a normalization/phase
difference.  For the same QNM parameter, direct values differ from the
TorchWave angular function by an approximately `sqrt(2*pi)` scale together
with a spin-dependent complex phase.  The public API does not document the
normalization or phase relation needed to turn that observation into an
independent reference convention.

## Harness status and controls

`KerrRingdown_diagnostic` remains marked `UNRESOLVED_ORACLE` in
`tests/parity_harness.py`; the former analytic oracle is intentionally not
used.  The existing physical-path tests remain the focused controls.  No
deferred NR cases were run and no LAL/PyCBC source was inspected or copied.

