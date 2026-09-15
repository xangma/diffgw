# Independent hybrid-energy MECO

Implemented an original, differentiable evaluation and minimum search in
`src/torchwave/approximants/phenom_meco.py`, with equation-derived tests in
`tests/test_clean_room_meco.py`. No calibrated waveform outputs, NR data,
ancillary fit coefficients, or external implementation source were used.

The standalone API is:

```python
from torchwave.approximants.phenom_meco import (
    hybrid_binding_energy, hybrid_meco_velocity, hybrid_meco_frequency,
)

v = hybrid_meco_velocity(30., 10., .6, -.2, pn_order=3.5)
mf = hybrid_meco_frequency(30., 10., .6, -.2, pn_order=3.5)
energy = hybrid_binding_energy(v, 30., 10., .6, -.2)
```

`mf` is the dimensionless dominant-mode frequency `M*f_22=v**3/pi`; divide
by total mass in seconds for Hz. Energy is per reduced mass. Masses may use
any common units. The energy API also returns first and second velocity
derivatives through `derivative=1` and `derivative=2`.

## Primary equations and convention

[Pratten et al., arXiv:2001.11412v1, Sec. V.1, Eqs. (48)–(49)](https://arxiv.org/html/2001.11412v1#S5.SS1)
define the PN/Kerr hybrid energy and describe fitting its minimum to avoid
root-finding. The explicit energy here follows
[Cabero et al., arXiv:1602.03134v3, Eqs. (2), (5)–(7), (23)–(26)](https://arxiv.org/html/1602.03134v3):

\[
E_h=E_{PN}/\eta-E_{K,PN}+E_K,\quad
E_K=\frac{1-2w+\chi w^{3/2}}{\sqrt{1-3w+2\chi w^{3/2}}}-1,
\quad w=\frac{v^2}{(1-\chi v^3)^{2/3}}.
\]

The Kerr spin is `chi_eff=(m1*chi1+m2*chi2)/(m1+m2)`, and both spin-induced
quadrupole constants equal one. Cabero's Sec. IV explicitly omits the
spin-cubic term; this implementation follows that convention. It includes
nonspinning and spin-spin terms through 3PN, plus spin-orbit terms through
3.5PN by default. `pn_order=3` omits the 3.5PN spin-orbit term. No 4PN option
is provided. The source paper recommends the 3.5PN hybrid prescription.

## Independent algebra and numerical method

Subtracting the truncated Kerr series symbolically gives
`E_h=E_K-(1/2)*sum(c_k*v**(k+2), k=2..7)`.
Every residual coefficient is factored by eta, so the zero-secondary-mass
limit is exactly Kerr without subtractive cancellation. For black holes,
the entire 2PN spin-spin term cancels its Kerr counterpart. For example,
`c_2=-eta/12`, `c_3=-(2/3)*eta*(chi1+chi2)`, and the 3PN spin-spin residual is

\[
\frac{5\eta}{18}\left[
x_1(7x_1+22)\chi_1^2+x_2(7x_2+22)\chi_2^2
+2(10-\eta)\chi_1\chi_2\right],\qquad x_i=m_i/M.
\]

The exact energy is rationalized to preserve its Newtonian limit for small
velocity. Velocity derivatives are analytic; writing `D=1-3w+2chi*w**1.5`
and `K=1-6w+8chi*w**1.5-3chi**2*w**2`, the Kerr contribution is
`dE_K/dv=-(1/2)*(dw/dv)*K/D**1.5`. Its derivative supplies the curvature.

The photon boundary follows the physical exterior solution of `D=0`.
The search scans 256 velocities from `1e-4*v_upper` to
`v_upper=(1-1e-5)*v_photon`, selects the first negative-to-positive energy
derivative crossing, and bisects 64 times. Three differentiable Newton
refinements recover the implicit first and second parameter derivatives
without differentiating the discrete bracket selection. Nonpositive
curvature, missing brackets, nonfinite values, or an unresolved stationary
point raise an error. There is no fallback fitted frequency.

For subextremal effective spin, the energy initially decreases and its Kerr
derivative diverges positively at the photon boundary; the finite PN
polynomial does not remove that sign change. This explains existence of a
minimum. It does not prove uniqueness or guarantee a finite scan resolves
all possible extrema. The selected minimum remains a PN prescription for
inspiral termination, not an independently established comparable-mass ISCO.

## Verification

The dedicated suite reports **37 passed in 2.09 seconds** on local CPU:

```sh
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/xangma/miniconda3/envs/pycbcorig313/bin/python -m pytest \
  tests/test_clean_room_meco.py -q
```

Checks include unsimplified paper energy minus the Kerr series; complex-step
energy derivatives and an independent Brent root search; a 70-digit Decimal
solution of the Kerr radial marginal-stability equation; convergence to that
test-particle limit; mass scaling and body exchange; broadcasting and dtype
promotion; `gradcheck`, `gradgradcheck`, and implicit stationarity gradients;
and explicit rejection of an unresolved float32 photon boundary.

The domain sweep uses ten mass ratios from 1 to 10^12 and eleven independently
varied component spins from -1 to 0.9999: 1,210 configurations per PN order.
Each minimum has positive curvature and slopes with opposite signs on either
side. A separate 2,048-point scan finds one minimum in every sampled case.

| Metric, float64 | 3PN | 3.5PN |
| --- | ---: | ---: |
| Velocity range | 0.337553–0.784297 | 0.337553–0.778437 |
| Largest velocity change, 64/24 vs 512/72 scan/bisection | 4.08e-14 | 3.79e-14 |
| Largest absolute energy derivative at returned root | 5.65e-11 | 1.31e-11 |
| Largest `abs(E')/(1+abs(E''))` | 4.19e-14 | 3.80e-14 |
| Smallest curvature | 4.69866 | 4.69866 |

The larger unscaled residuals occur in the steep near-prograde regime; the
curvature-scaled measure bounds the corresponding Newton velocity correction.
These checks verify equations and numerics, not waveform agreement with NR.

## Domain and integration status

Inputs require finite nonnegative masses with at least one positive mass,
component spins in `[-1,1]`, and effective spin no greater than 0.9999.
The upper effective-spin restriction excludes the degenerate prograde
endpoint. The energy requires a positive velocity below the photon orbit
and raises if that boundary cannot be resolved at the input precision.
Python-only inputs use float64; float32/float64 tensors broadcast on a common
device. The root internally uses float64, so its device must support float64.
CPU was tested; GPU execution was not tested. No tides, precession, or
neutron-star multipoles enter this black-hole prescription.

The existing XAS forward path was read: it uses its fitted `f_meco` to set
the amplitude and phase matching regions. Pratten's text does not identify
the PN order and every convention of that fit sufficiently to certify
equivalence. Therefore this change leaves XAS and its defaults untouched.
Adopting the direct solver in XAS requires an explicit modeling decision
and waveform validation; no calibrated-model equivalence is claimed.
